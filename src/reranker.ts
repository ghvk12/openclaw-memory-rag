import type { Logger } from "./logger.js";
import type { RerankerConfig } from "./config.js";
import type { HybridSearchHit } from "./qdrant-client.js";

/**
 * Reranker contract: given a query and N hits, return at most topNOut hits
 * sorted by relevance descending, with a `rerankScore` populated on each.
 */
export type Reranker = {
  rerank(query: string, hits: HybridSearchHit[], topNOut: number): Promise<HybridSearchHit[]>;
  probe(): Promise<{ ok: boolean; reason?: string }>;
};

/**
 * Factory: pick the right reranker implementation based on `cfg.endpoint`.
 *   - "ollama": legacy path via Ollama's /api/embeddings (back-compat only).
 *   - "tei":    real cross-encoder via TEI /rerank (HuggingFace TEI, or the
 *               bundled tools/reranker-sidecar/ for Apple Silicon).
 *
 * In both cases, if the reranker is unreachable or returns malformed scores,
 * we degrade to the original hybrid scores instead of failing the whole
 * retrieval call (the engine's recall path is allowed to call .rerank() and
 * trust it to never throw).
 */
export function createReranker(cfg: RerankerConfig, logger: Logger): Reranker {
  if (cfg.endpoint === "tei") {
    return createTeiReranker(cfg, logger);
  }
  return createOllamaReranker(cfg, logger);
}

const RERANK_TIMEOUT_MS = 15_000;

/**
 * Cross-encoder reranker via Ollama's /api/embeddings. Many GGUF rerankers
 * (`bge-reranker-v2-m3`, `quentinz/bge-reranker-v2-m3`, etc.) accept the
 * input `"<query>\t<document>"` and return a 1-dim "score" embedding.
 *
 * In practice this rarely produces ranking-grade scores: most community
 * GGUF ports of cross-encoder models still return full sentence embeddings
 * (1024-dim), and the per-pair averaged-embedding fallback below does not
 * track relevance. Prefer `endpoint: "tei"` with the bundled sidecar.
 *
 * If the model is unavailable or returns a malformed score, we degrade to
 * the original hybrid scores instead of failing the whole retrieval call.
 */
export function createOllamaReranker(cfg: RerankerConfig, logger: Logger): Reranker {
  const baseUrl = cfg.url.replace(/\/$/, "");
  const model = cfg.model;

  async function scorePair(query: string, doc: string): Promise<number | null> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), RERANK_TIMEOUT_MS);
    try {
      const res = await fetch(`${baseUrl}/api/embeddings`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, prompt: `${query}\t${doc}` }),
        signal: controller.signal,
      });
      if (!res.ok) return null;
      const data = (await res.json()) as { embedding?: number[] };
      const vec = data.embedding;
      if (!Array.isArray(vec) || vec.length === 0) return null;
      if (vec.length === 1) return vec[0]!;
      let sum = 0;
      for (const v of vec) sum += v;
      return sum / vec.length;
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    async rerank(query: string, hits: HybridSearchHit[], topNOut: number): Promise<HybridSearchHit[]> {
      if (!cfg.enabled || hits.length === 0) {
        return hits.slice(0, topNOut);
      }
      const candidates = hits.slice(0, cfg.topNIn);
      const concurrency = 6;
      const out: Array<{ hit: HybridSearchHit; score: number | null }> = candidates.map((hit) => ({ hit, score: null }));

      let cursor = 0;
      let failures = 0;
      const workers = Array.from({ length: Math.min(concurrency, candidates.length) }, async () => {
        while (true) {
          const i = cursor++;
          if (i >= candidates.length) return;
          const text = candidates[i]!.payload?.text ?? "";
          const score = await scorePair(query, text.slice(0, 2000));
          if (score === null) failures++;
          out[i] = { hit: candidates[i]!, score };
        }
      });
      await Promise.all(workers);

      if (failures === candidates.length) {
        logger.warn(
          `memory-rag: reranker "${model}" produced no scores (model not pulled? bad endpoint?). Using hybrid scores.`,
        );
        return candidates.slice(0, topNOut);
      }

      const reranked = out
        .map((entry) => {
          if (entry.score === null) return entry.hit;
          return { ...entry.hit, score: entry.score };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, topNOut);
      return reranked;
    },

    async probe(): Promise<{ ok: boolean; reason?: string }> {
      if (!cfg.enabled) return { ok: true, reason: "Reranker disabled (skipped)." };
      const score = await scorePair("hello", "hi there");
      if (score === null) {
        return {
          ok: false,
          reason: `Reranker "${model}" not callable. Pull with: ollama pull ${model} (note: most Ollama ports of bge-reranker-v2-m3 are not real cross-encoders \u2014 prefer endpoint: "tei").`,
        };
      }
      return { ok: true };
    },
  };
}

/**
 * Real cross-encoder reranker via TEI's /rerank protocol. Compatible with both
 *   - HuggingFace text-embeddings-inference (amd64 / x86_64), and
 *   - tools/reranker-sidecar/ (FastAPI + sentence-transformers, Apple Silicon).
 *
 * Sends one batched POST per recall (vs N parallel requests for the Ollama
 * path), which is materially faster: a single forward pass through the cross
 * encoder over the full candidate list.
 *
 * Request:
 *   POST /rerank  { "query": str, "texts": str[], "raw_scores": false }
 * Response:
 *   [ { "index": int, "score": float }, ... ]   sorted desc by score.
 */
export function createTeiReranker(cfg: RerankerConfig, logger: Logger): Reranker {
  const baseUrl = cfg.url.replace(/\/$/, "");

  type TeiHit = { index: number; score: number };

  async function callRerank(
    query: string,
    texts: string[],
  ): Promise<TeiHit[] | null> {
    if (texts.length === 0) return [];
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), RERANK_TIMEOUT_MS);
    try {
      const res = await fetch(`${baseUrl}/rerank`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, texts, raw_scores: false }),
        signal: controller.signal,
      });
      if (!res.ok) {
        logger.warn(`memory-rag: TEI /rerank returned HTTP ${res.status}`);
        return null;
      }
      const data = (await res.json()) as TeiHit[] | { error?: string };
      if (!Array.isArray(data)) {
        logger.warn(`memory-rag: TEI /rerank returned non-array body: ${JSON.stringify(data).slice(0, 200)}`);
        return null;
      }
      return data;
    } catch (err) {
      logger.warn(`memory-rag: TEI /rerank fetch failed: ${String(err)}`);
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    async rerank(query: string, hits: HybridSearchHit[], topNOut: number): Promise<HybridSearchHit[]> {
      if (!cfg.enabled || hits.length === 0) {
        return hits.slice(0, topNOut);
      }
      const candidates = hits.slice(0, cfg.topNIn);
      // Defensive: legacy points in older Qdrant collections may have non-string
      // `payload.text` (null, numeric, object). Coerce to string and replace any
      // empties with a single space so the upstream tokenizer never sees a
      // value that breaks the whole batch (`TypeError: TextInputSequence must
      // be str`).
      const texts = candidates.map((c) => {
        const raw = c.payload?.text;
        const s = typeof raw === "string" ? raw : raw == null ? "" : String(raw);
        const trimmed = s.slice(0, 2000);
        return trimmed.length === 0 ? " " : trimmed;
      });
      const ranked = await callRerank(query, texts);
      if (!ranked) {
        logger.warn(`memory-rag: TEI reranker fell through; using hybrid scores.`);
        return candidates.slice(0, topNOut);
      }
      const out: HybridSearchHit[] = [];
      for (const entry of ranked) {
        const original = candidates[entry.index];
        if (!original) continue;
        out.push({ ...original, score: entry.score });
        if (out.length >= topNOut) break;
      }
      return out;
    },

    async probe(): Promise<{ ok: boolean; reason?: string }> {
      if (!cfg.enabled) return { ok: true, reason: "Reranker disabled (skipped)." };
      const ranked = await callRerank("ping", ["pong"]);
      if (ranked === null) {
        return {
          ok: false,
          reason: `TEI reranker at ${baseUrl} not callable. Is the sidecar/TEI container running?`,
        };
      }
      if (ranked.length === 0 || typeof ranked[0]?.score !== "number") {
        return {
          ok: false,
          reason: `TEI reranker at ${baseUrl} returned an unexpected payload shape.`,
        };
      }
      return { ok: true };
    },
  };
}
