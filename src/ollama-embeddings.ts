import type { Logger } from "./logger.js";
import type { EmbeddingsConfig } from "./config.js";

/**
 * Custom error class for embedding failures, so the corpus supplement / hooks
 * can distinguish "Ollama is down" from "Ollama returned a malformed response".
 */
export class OllamaEmbeddingError extends Error {
  override readonly cause?: unknown;
  readonly status?: number;
  constructor(message: string, cause?: unknown, status?: number) {
    super(message);
    this.name = "OllamaEmbeddingError";
    if (cause !== undefined) this.cause = cause;
    if (status !== undefined) this.status = status;
  }
}

export type OllamaEmbedClient = {
  embedQuery(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  /** Probe Ollama health and confirm the model is pulled. */
  probe(): Promise<{ ok: boolean; reason?: string }>;
  readonly model: string;
  readonly dim: number;
  readonly url: string;
};

const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * mxbai-embed-large (and most BERT-derived embedders) cap at 512 tokens.
 * Char-to-token ratio varies wildly by script: ASCII English is ~4 chars/token,
 * but emoji, CJK, and Indic scripts can be as low as 0.5 chars/token (one
 * grapheme = 2-4 tokens). 1800 was too generous \u2014 messages with heavy emoji
 * or non-Latin content blew past 512 tokens. 1100 chars \u2248 275 tokens for ASCII
 * but degrades safely to ~400 tokens for emoji-dense input. Truncation is
 * applied ONLY to the text sent to Ollama; the full original text is preserved
 * in the Qdrant payload so retrieval still surfaces complete messages.
 *
 * If you bump this back up, also re-run `rebuild-from-wal` to pick up any
 * messages that previously failed embedding.
 */
const MAX_EMBED_CHARS = 1100;

function truncateForEmbedding(text: string): string {
  if (text.length <= MAX_EMBED_CHARS) return text;
  return text.slice(0, MAX_EMBED_CHARS);
}

/**
 * Thin client for Ollama's embeddings endpoint. Compatible with Ollama 0.1.27+.
 * Uses /api/embed (batch) for efficiency; falls back to /api/embeddings for
 * older daemons that don't expose the batch endpoint.
 */
export function createOllamaEmbedClient(cfg: EmbeddingsConfig, logger: Logger): OllamaEmbedClient {
  const baseUrl = cfg.url.replace(/\/$/, "");
  const model = cfg.model;
  const dim = cfg.dim;

  async function postJson<T>(path: string, body: unknown, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new OllamaEmbeddingError(
          `Ollama ${path} returned ${res.status}: ${text.slice(0, 200)}`,
          undefined,
          res.status,
        );
      }
      return (await res.json()) as T;
    } catch (err) {
      if (err instanceof OllamaEmbeddingError) throw err;
      throw new OllamaEmbeddingError(`Ollama ${path} request failed: ${String(err)}`, err);
    } finally {
      clearTimeout(timer);
    }
  }

  async function embedBatchViaApi(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const safeTexts = texts.map(truncateForEmbedding);
    try {
      const res = await postJson<{ embeddings: number[][] }>("/api/embed", {
        model,
        input: safeTexts,
      });
      if (!Array.isArray(res?.embeddings) || res.embeddings.length !== texts.length) {
        throw new OllamaEmbeddingError(
          `Ollama returned ${res?.embeddings?.length ?? 0} embeddings for ${texts.length} inputs`,
        );
      }
      return res.embeddings;
    } catch (err) {
      if (err instanceof OllamaEmbeddingError && err.status === 404) {
        logger.debug?.("/api/embed not available, falling back to /api/embeddings (one-by-one)");
        return embedBatchViaLegacy(texts);
      }
      throw err;
    }
  }

  async function embedBatchViaLegacy(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (const text of texts) {
      const res = await postJson<{ embedding: number[] }>("/api/embeddings", {
        model,
        prompt: truncateForEmbedding(text),
      });
      if (!Array.isArray(res?.embedding)) {
        throw new OllamaEmbeddingError("Ollama /api/embeddings returned no embedding array");
      }
      out.push(res.embedding);
    }
    return out;
  }

  function validateDim(vec: number[]): number[] {
    if (vec.length !== dim) {
      throw new OllamaEmbeddingError(
        `Embedding dim mismatch: got ${vec.length}, configured ${dim}. Update embeddings.dim or rebuild collection.`,
      );
    }
    return vec;
  }

  return {
    model,
    dim,
    url: baseUrl,

    async embedQuery(text: string): Promise<number[]> {
      const [vec] = await embedBatchViaApi([text]);
      if (!vec) throw new OllamaEmbeddingError("Ollama returned no embedding for query");
      return validateDim(vec);
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      if (texts.length === 0) return [];
      const vectors = await embedBatchViaApi(texts);
      return vectors.map(validateDim);
    },

    async probe(): Promise<{ ok: boolean; reason?: string }> {
      try {
        const tagsRes = await fetch(`${baseUrl}/api/tags`, { method: "GET" });
        if (!tagsRes.ok) {
          return { ok: false, reason: `Ollama /api/tags returned ${tagsRes.status}` };
        }
        const data = (await tagsRes.json()) as { models?: Array<{ name?: string; model?: string }> };
        const installed = (data.models ?? []).map((m) => m.name ?? m.model ?? "").filter(Boolean);
        const wanted = model.split(":")[0];
        const found = installed.some((name) => name === model || name.split(":")[0] === wanted);
        if (!found) {
          return {
            ok: false,
            reason: `Model "${model}" not pulled. Run: ollama pull ${model}`,
          };
        }
        const probe = await embedBatchViaApi(["ping"]);
        if (probe.length !== 1 || probe[0]!.length !== dim) {
          return {
            ok: false,
            reason: `Probe returned dim=${probe[0]?.length ?? 0}, expected ${dim}.`,
          };
        }
        return { ok: true };
      } catch (err) {
        return { ok: false, reason: `Ollama probe failed: ${String(err)}` };
      }
    },
  };
}

/**
 * Build an OpenClaw `MemoryEmbeddingProviderAdapter` shape so this provider
 * can be auto-selected via `agents.defaults.memorySearch.provider: "ollama"`.
 *
 * We register this adapter from the plugin entry. Type-shape is intentionally
 * untyped here to avoid a hard dep on a private SDK type during local builds;
 * the runtime `register()` call validates the structure at load time.
 */
export function buildMemoryEmbeddingProviderAdapter(cfg: EmbeddingsConfig, logger: Logger): {
  id: string;
  defaultModel: string;
  transport: "remote";
  autoSelectPriority: number;
  create: (options: { model?: string }) => Promise<{
    provider: {
      id: string;
      model: string;
      maxInputTokens: number;
      embedQuery: (text: string) => Promise<number[]>;
      embedBatch: (texts: string[]) => Promise<number[][]>;
    } | null;
  }>;
  formatSetupError: (err: unknown) => string;
} {
  return {
    id: "ollama",
    defaultModel: cfg.model,
    transport: "remote",
    autoSelectPriority: 50,
    async create(options) {
      const effectiveCfg: EmbeddingsConfig = {
        url: cfg.url,
        model: options.model ?? cfg.model,
        dim: cfg.dim,
      };
      const client = createOllamaEmbedClient(effectiveCfg, logger);
      const probe = await client.probe();
      if (!probe.ok) {
        logger.warn(`memory-rag: Ollama embedding adapter probe failed: ${probe.reason}`);
        return { provider: null };
      }
      return {
        provider: {
          id: "ollama",
          model: client.model,
          maxInputTokens: 512,
          embedQuery: client.embedQuery,
          embedBatch: client.embedBatch,
        },
      };
    },
    formatSetupError(err) {
      return `Ollama embedding provider error: ${err instanceof Error ? err.message : String(err)}`;
    },
  };
}
