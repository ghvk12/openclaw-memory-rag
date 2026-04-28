"""
openclaw-memory-rag reranker sidecar.

A tiny FastAPI service that exposes a TEI-compatible /rerank endpoint backed by
sentence_transformers.CrossEncoder. Built specifically because HuggingFace's
text-embeddings-inference Docker image is amd64-only and crashes under Rosetta
on Apple Silicon (Intel MKL incompatibility), and Ollama's /api/embeddings
endpoint cannot serve true cross-encoder reranking.

Default model: BAAI/bge-reranker-v2-m3 (the same model the plugin assumes).
On Apple Silicon, sentence-transformers picks up MPS automatically for ~5x
speedup over CPU.

Endpoints:
  GET  /health         -> {"status": "ok", "model": "...", "device": "mps|cpu|cuda"}
  POST /rerank         -> TEI-compatible: returns sorted list of {index, score}.

Run locally:
  python -m uvicorn app:app --host 127.0.0.1 --port 8089

Run via launchd: see tools/reranker-sidecar/ai.openclaw.reranker.plist.template.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

LOG = logging.getLogger("openclaw-reranker-sidecar")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_ID = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_state: dict = {"model": None, "device": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    device = _pick_device()
    LOG.info("loading cross-encoder model_id=%s device=%s max_length=%s", MODEL_ID, device, MAX_LENGTH)
    _state["model"] = CrossEncoder(MODEL_ID, max_length=MAX_LENGTH, device=device)
    _state["device"] = device
    LOG.info("model ready; serving /rerank")
    yield
    LOG.info("shutting down")
    _state["model"] = None


app = FastAPI(title="openclaw-reranker-sidecar", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    texts: list[str] = Field(..., min_length=1, max_length=200)
    raw_scores: bool = Field(default=False, description="If true, return raw logits; else sigmoid-squashed.")
    truncate: bool = Field(default=True, description="Reserved for TEI compat; we always truncate via max_length.")
    return_text: bool = Field(default=False, description="Reserved for TEI compat; we never echo the texts.")


class RerankItem(BaseModel):
    index: int
    score: float


@app.get("/health")
def health() -> dict:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok", "model": MODEL_ID, "device": _state["device"]}


def _coerce_text(value) -> str:
    """
    Defensive: HuggingFace tokenizer raises `TypeError: TextInputSequence must
    be str` if any element of the batch isn't a str. Coerce silently and
    substitute a single space for empty strings (an empty pair confuses some
    cross-encoders).
    """
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return value if value.strip() else " "


@app.post("/rerank", response_model=list[RerankItem])
def rerank(req: RerankRequest) -> list[RerankItem]:
    """
    TEI-compatible reranking. Returns hits sorted by relevance descending.
    `index` is the position in the input `texts` list.
    """
    model: CrossEncoder | None = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    safe_texts = [_coerce_text(t) for t in req.texts]
    pairs = [(req.query, t) for t in safe_texts]
    activation_fn = None if req.raw_scores else torch.nn.Sigmoid()

    # Fast path: batched prediction.
    try:
        batch_scores = model.predict(pairs, activation_fn=activation_fn, convert_to_numpy=True)
        scores = [float(s) for s in batch_scores]
    except Exception as batch_exc:
        # The HuggingFace fast tokenizer raises `TypeError: TextInputSequence
        # must be str` for the entire batch when ANY single input has a shape
        # the Rust tokenizer rejects (e.g., specific control chars or
        # surrogate pairs in WhatsApp text). Fall back to per-pair prediction
        # so one bad row never kills the whole recall \u2014 the offender gets a 0.0
        # score and the rest score normally.
        LOG.warning("rerank batch failed (%s) \u2014 retrying per-pair", type(batch_exc).__name__)
        scores = []
        bad_indexes: list[int] = []
        for i, pair in enumerate(pairs):
            try:
                s = model.predict([pair], activation_fn=activation_fn, convert_to_numpy=True)
                scores.append(float(s[0]))
            except Exception:
                bad_indexes.append(i)
                scores.append(0.0)
        if bad_indexes:
            LOG.warning("per-pair fallback returned 0.0 for indexes=%s (offending texts)", bad_indexes)

    out = [RerankItem(index=i, score=scores[i]) for i in range(len(scores))]
    out.sort(key=lambda r: r.score, reverse=True)
    return out
