import { Type, type Static } from "typebox";

export const QdrantConfigSchema = Type.Object({
  url: Type.String({ default: "http://localhost:6333" }),
  collection: Type.String({ default: "wa_memory_v1_mxbai_1024" }),
  apiKey: Type.Optional(Type.String()),
});

export const EmbeddingsConfigSchema = Type.Object({
  url: Type.String({ default: "http://localhost:11434" }),
  model: Type.String({ default: "mxbai-embed-large" }),
  dim: Type.Number({ default: 1024, minimum: 1, maximum: 8192 }),
});

export const RerankerConfigSchema = Type.Object({
  enabled: Type.Boolean({ default: true }),
  /**
   * Reranker backend.
   *   - "ollama": calls Ollama's /api/embeddings with `<query>\t<doc>`. Works
   *     with embedding models but cannot do true cross-encoder reranking
   *     (architectural mismatch). Kept as the default for back-compat with
   *     pre-existing configs; produces a warning and falls back to hybrid
   *     scores when the model can't return per-pair scalars.
   *   - "tei": calls a TEI-compatible POST /rerank endpoint. Supports the
   *     real bge-reranker-v2-m3 cross-encoder via either HuggingFace's TEI
   *     (amd64 only) or the bundled tools/reranker-sidecar/ Python service
   *     (recommended on Apple Silicon).
   */
  endpoint: Type.Union([Type.Literal("ollama"), Type.Literal("tei")], { default: "ollama" }),
  url: Type.String({ default: "http://localhost:11434" }),
  model: Type.String({ default: "bge-reranker-v2-m3" }),
  topNIn: Type.Number({ default: 30, minimum: 1, maximum: 200 }),
  topNOut: Type.Number({ default: 10, minimum: 1, maximum: 50 }),
});

export const RetrievalConfigSchema = Type.Object({
  topK: Type.Number({ default: 10, minimum: 1, maximum: 100 }),
  parentWindow: Type.Number({ default: 2, minimum: 0, maximum: 10 }),
  hybridFusion: Type.Union(
    [Type.Literal("rrf"), Type.Literal("dense_only"), Type.Literal("sparse_only")],
    { default: "rrf" },
  ),
  tokenBudget: Type.Number({ default: 4000, minimum: 200, maximum: 32000 }),
  minScore: Type.Number({ default: 0.0, minimum: 0.0, maximum: 1.0 }),
  recencyHalfLifeDays: Type.Number({ default: 30, minimum: 0 }),
});

export const StorageConfigSchema = Type.Object({
  wal: Type.String({ default: "~/.openclaw/memory-rag/wal" }),
  fallbackMd: Type.String({ default: "~/.openclaw/memory-rag/MEMORY.md" }),
});

export const FiltersConfigSchema = Type.Object({
  minTokens: Type.Number({ default: 3, minimum: 0 }),
  skipReactions: Type.Boolean({ default: true }),
  skipSystemMessages: Type.Boolean({ default: true }),
  autoCapture: Type.Boolean({ default: false }),
  autoRecall: Type.Boolean({ default: true }),
});

export const ChannelsConfigSchema = Type.Object({
  whitelist: Type.Union([Type.Array(Type.String()), Type.Null()], { default: null }),
});

export const IsolationModeSchema = Type.Union(
  [
    Type.Literal("per_chat"),
    Type.Literal("per_user"),
    Type.Literal("global_owner"),
    Type.Literal("global_all"),
  ],
  { default: "global_owner" },
);

export const PluginConfigSchema = Type.Object({
  qdrant: QdrantConfigSchema,
  embeddings: EmbeddingsConfigSchema,
  reranker: Type.Optional(RerankerConfigSchema),
  retrieval: Type.Optional(RetrievalConfigSchema),
  storage: Type.Optional(StorageConfigSchema),
  isolation: Type.Optional(IsolationModeSchema),
  ownerJids: Type.Optional(Type.Array(Type.String())),
  filters: Type.Optional(FiltersConfigSchema),
  channels: Type.Optional(ChannelsConfigSchema),
});

export type PluginConfig = Static<typeof PluginConfigSchema>;
export type QdrantConfig = Static<typeof QdrantConfigSchema>;
export type EmbeddingsConfig = Static<typeof EmbeddingsConfigSchema>;
export type RerankerConfig = Static<typeof RerankerConfigSchema>;
export type RetrievalConfig = Static<typeof RetrievalConfigSchema>;
export type StorageConfig = Static<typeof StorageConfigSchema>;
export type FiltersConfig = Static<typeof FiltersConfigSchema>;
export type ChannelsConfig = Static<typeof ChannelsConfigSchema>;
export type IsolationMode = Static<typeof IsolationModeSchema>;

export type ResolvedConfig = {
  qdrant: QdrantConfig;
  embeddings: EmbeddingsConfig;
  reranker: RerankerConfig;
  retrieval: RetrievalConfig;
  storage: StorageConfig;
  isolation: IsolationMode;
  ownerJids: string[];
  filters: FiltersConfig;
  channels: ChannelsConfig;
};

const DEFAULTS: ResolvedConfig = {
  qdrant: { url: "http://localhost:6333", collection: "wa_memory_v1_mxbai_1024" },
  embeddings: { url: "http://localhost:11434", model: "mxbai-embed-large", dim: 1024 },
  reranker: { enabled: true, endpoint: "ollama", url: "http://localhost:11434", model: "bge-reranker-v2-m3", topNIn: 30, topNOut: 10 },
  retrieval: { topK: 10, parentWindow: 2, hybridFusion: "rrf", tokenBudget: 4000, minScore: 0.0, recencyHalfLifeDays: 30 },
  storage: { wal: "~/.openclaw/memory-rag/wal", fallbackMd: "~/.openclaw/memory-rag/MEMORY.md" },
  isolation: "global_owner",
  ownerJids: [],
  filters: { minTokens: 3, skipReactions: true, skipSystemMessages: true, autoCapture: false, autoRecall: true },
  channels: { whitelist: null },
};

/**
 * Reject `http://` URLs that don't point at loopback. Non-loopback Qdrant or
 * Ollama endpoints over plaintext expose vector payloads (which include raw
 * conversation text) and `qdrant.apiKey` to network observers.
 *
 * Loopback hosts allowed: localhost, 127.0.0.1, ::1, and ipv6 loopback wrap.
 */
function assertSecureUrl(field: string, raw: string): void {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(`memory-rag: ${field} is not a valid URL: ${raw}`);
  }
  if (parsed.protocol === "https:") {
    return;
  }
  if (parsed.protocol !== "http:") {
    throw new Error(
      `memory-rag: ${field} must use http:// or https:// (got ${parsed.protocol})`,
    );
  }
  const host = parsed.hostname.toLowerCase();
  const loopback =
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "::1" ||
    host === "[::1]" ||
    host.endsWith(".localhost");
  if (!loopback) {
    throw new Error(
      `memory-rag: ${field} uses plaintext http:// to a non-loopback host (${host}). ` +
        `Use https:// to protect vector payloads and apiKey in transit.`,
    );
  }
}

/**
 * Merge user-supplied plugin config with defaults. Throws on schema-incompatible inputs,
 * but tolerates any missing optional sections by filling in defaults.
 */
export function resolveConfig(raw: unknown): ResolvedConfig {
  const partial = (raw && typeof raw === "object" ? (raw as Partial<PluginConfig>) : {}) as Partial<PluginConfig>;
  const resolved: ResolvedConfig = {
    qdrant: { ...DEFAULTS.qdrant, ...(partial.qdrant ?? {}) },
    embeddings: { ...DEFAULTS.embeddings, ...(partial.embeddings ?? {}) },
    reranker: { ...DEFAULTS.reranker, ...(partial.reranker ?? {}) },
    retrieval: { ...DEFAULTS.retrieval, ...(partial.retrieval ?? {}) },
    storage: { ...DEFAULTS.storage, ...(partial.storage ?? {}) },
    isolation: partial.isolation ?? DEFAULTS.isolation,
    ownerJids: partial.ownerJids ?? DEFAULTS.ownerJids,
    filters: { ...DEFAULTS.filters, ...(partial.filters ?? {}) },
    channels: { ...DEFAULTS.channels, ...(partial.channels ?? {}) },
  };

  assertSecureUrl("qdrant.url", resolved.qdrant.url);
  assertSecureUrl("embeddings.url", resolved.embeddings.url);
  assertSecureUrl("reranker.url", resolved.reranker.url);

  return resolved;
}
