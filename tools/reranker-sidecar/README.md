# openclaw-memory-rag reranker sidecar

A tiny FastAPI service that exposes a [TEI](https://github.com/huggingface/text-embeddings-inference)-compatible `POST /rerank` endpoint backed by `sentence_transformers.CrossEncoder`.

## Why this exists

`openclaw-memory-rag` supports two reranker backends:

1. **`endpoint: "ollama"`** — calls Ollama's `/api/embeddings`. This works with embedding models but **cannot do true cross-encoder reranking** (cross-encoders need to take `(query, doc)` together and emit a single relevance logit; Ollama's embedding API just returns sentence embeddings).
2. **`endpoint: "tei"`** — calls a TEI-compatible `/rerank` endpoint. This is the correct protocol for cross-encoder reranking.

HuggingFace's official TEI Docker image is amd64-only and crashes under Rosetta 2 on Apple Silicon (Intel MKL incompatibility). This sidecar is a drop-in replacement that runs natively on Apple Silicon (auto-detects MPS) using `sentence-transformers` instead of TEI's ONNX runtime.

## Setup

Requires Python 3.13 (or 3.10+, but the launchd plist hard-codes 3.13).

```bash
cd tools/reranker-sidecar
./setup.sh
```

This creates a venv and installs ~1.5 GB of dependencies (torch + transformers + sentence-transformers).

## Running manually (for development)

```bash
source venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8089
```

First request triggers model download (~600 MB to `~/.cache/huggingface/`).

## Running as a launchd service

```bash
# Render the plist with your venv path baked in:
sed -e "s|__SIDECAR_DIR__|$(pwd)|g" \
    -e "s|__USER_HOME__|$HOME|g" \
    ai.openclaw.reranker.plist.template \
    > ~/Library/LaunchAgents/ai.openclaw.reranker.plist

launchctl load ~/Library/LaunchAgents/ai.openclaw.reranker.plist
launchctl kickstart -k gui/$(id -u)/ai.openclaw.reranker

# Tail logs:
tail -f ~/.openclaw/logs/reranker.log
```

## API

### `GET /health`

```json
{"status": "ok", "model": "BAAI/bge-reranker-v2-m3", "device": "mps"}
```

### `POST /rerank` (TEI-compatible)

```json
{
  "query": "What colour is octarine?",
  "texts": ["Octarine is the colour of magic.", "Tax filing varies by region."],
  "raw_scores": false
}
```

Response — sorted by score descending:

```json
[
  {"index": 0, "score": 0.94},
  {"index": 1, "score": 0.02}
]
```

## Environment variables

| Var | Default | Notes |
|---|---|---|
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Any HuggingFace cross-encoder works. |
| `RERANKER_MAX_LENGTH` | `512` | Tokens per (query, doc) pair. Truncates beyond. |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose. |
