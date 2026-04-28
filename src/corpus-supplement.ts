import type { MemoryEngine } from "./engine.js";

/**
 * Build a `MemoryCorpusSupplement`-shaped object that plugs into OpenClaw's
 * unified memory search. When the agent (or `openclaw memory search`) runs,
 * results from this supplement are merged with the built-in sqlite-vec
 * results so the user gets both.
 *
 * Shape matches `MemoryCorpusSupplement` from
 * `openclaw/plugin-sdk/memory-core` (v2026.4.20+):
 *   - search({ query, maxResults, agentSessionKey }) \u2192 MemoryCorpusSearchResult[]
 *   - get({ lookup, fromLine, lineCount, agentSessionKey }) \u2192 MemoryCorpusGetResult|null
 */
export function buildCorpusSupplement(engine: MemoryEngine): {
  search: (params: { query: string; maxResults?: number; agentSessionKey?: string }) => Promise<
    Array<{
      corpus: string;
      path: string;
      title?: string;
      score: number;
      snippet: string;
      id?: string;
      sourceType?: string;
      updatedAt?: string;
    }>
  >;
  get: (params: { lookup: string; fromLine?: number; lineCount?: number; agentSessionKey?: string }) => Promise<
    {
      corpus: string;
      path: string;
      title?: string;
      content: string;
      fromLine: number;
      lineCount: number;
      id?: string;
      sourceType?: string;
      updatedAt?: string;
    } | null
  >;
} {
  return {
    async search({ query, maxResults }) {
      const limit = Math.min(maxResults ?? engine.cfg.retrieval.topK, 50);
      const result = await engine.recall(query, undefined);
      return engine.toCorpusResults(result.blocks).slice(0, limit);
    },

    async get({ lookup }) {
      const m = lookup.match(/^memrag:\/\/(?<chatId>[^/]+)\/(?<index>\d+)$/);
      if (!m?.groups) return null;
      const chatId = m.groups.chatId!;
      const index = Number.parseInt(m.groups.index!, 10);
      if (!Number.isFinite(index)) return null;
      try {
        const window = await engine.qdrant.fetchWindow(
          chatId,
          Math.max(0, index - engine.cfg.retrieval.parentWindow),
          index + engine.cfg.retrieval.parentWindow,
          32,
        );
        if (window.length === 0) return null;
        window.sort((a, b) => (a.payload.messageIndex ?? 0) - (b.payload.messageIndex ?? 0));
        const lines = window.map((h) => {
          const ts = h.payload.timestamp ? new Date(h.payload.timestamp).toISOString() : "";
          const speaker = h.payload.senderName ?? h.payload.senderJid ?? h.payload.role;
          return `[${ts}] ${speaker}: ${h.payload.text}`;
        });
        const first = window[0]!;
        return {
          corpus: "memory-rag",
          path: lookup,
          title: `chat=${chatId} window`,
          content: lines.join("\n"),
          fromLine: 1,
          lineCount: lines.length,
          id: first.id,
          sourceType: "memory-rag",
          updatedAt: new Date(first.payload.timestamp).toISOString(),
        };
      } catch {
        return null;
      }
    },
  };
}
