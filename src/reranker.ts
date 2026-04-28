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
 * Cross-encoder reranker via Ollama's /api/embeddings. Many GGUF rerankers
 * (`bge-reranker-v2-m3`, `quentinz/bge-reranker-v2-m3`, etc.) accept the
 * input `"<query>\t<document>"` and return a 1-dim "score" embedding.
 *
 * If the model is unavailable or returns a malformed score, we degrade to
 * the original hybrid scores instead of failing the whole retrieval call.
 */
export function createOllamaReranker(cfg: RerankerConfig, logger: Logger): Reranker {
  const baseUrl = cfg.url.replace(/\/$/, "");
  const model = cfg.model;
  const RERANK_TIMEOUT_MS = 15_000;

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
          reason: `Reranker "${model}" not callable. Pull with: ollama pull ${model}`,
        };
      }
      return { ok: true };
    },
  };
}
