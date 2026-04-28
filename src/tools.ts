import type { MemoryEngine } from "./engine.js";

/**
 * Plugin api shape for tool registration. Loose typing again so we survive
 * minor SDK shifts.
 */
export type ToolApi = {
  registerTool: (
    spec: {
      name: string;
      label?: string;
      description: string;
      parameters: unknown;
      execute: (
        toolCallId: string,
        params: Record<string, unknown>,
      ) => Promise<{
        content: Array<{ type: string; text: string }>;
        details?: Record<string, unknown>;
      }>;
    },
    opts?: { name?: string },
  ) => void;
};

/**
 * Register `memory_recall_rag` and `memory_store_rag` tools so the LLM can
 * explicitly invoke retrieval/store actions. Auto-recall (the prompt-build
 * hook) covers the common path; these tools are for surgical use cases.
 */
export function registerTools(api: ToolApi, engine: MemoryEngine): void {
  api.registerTool(
    {
      name: "memory_recall_rag",
      label: "Memory Recall (RAG)",
      description:
        "Search persistent hybrid (vector + BM25) memory across all conversations. Use when the user references something they've said before that isn't in the current context window.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Natural-language search query." },
          limit: { type: "number", description: "Max blocks to return (default: top_k).", minimum: 1, maximum: 50 },
        },
        required: ["query"],
        additionalProperties: false,
      },
      async execute(_toolCallId, params) {
        const query = String(params.query ?? "");
        const limit = typeof params.limit === "number" ? params.limit : engine.cfg.retrieval.topK;
        const result = await engine.recall(query, undefined);
        const trimmed = result.blocks.slice(0, limit);
        const text =
          trimmed.length === 0
            ? "No relevant memories found."
            : trimmed
                .map((block, i) => {
                  const lines = block.hits.map(
                    (h) => `  - ${h.payload?.senderName ?? h.payload?.role ?? "?"}: ${h.payload?.text ?? ""}`,
                  );
                  return `Memory ${i + 1} (chat=${block.chatId}, score=${block.bestScore.toFixed(3)}):\n${lines.join("\n")}`;
                })
                .join("\n\n");
        return {
          content: [{ type: "text", text }],
          details: { rawHitCount: result.rawHitCount, rerankedCount: result.rerankedCount, blocks: trimmed.length },
        };
      },
    },
    { name: "memory_recall_rag" },
  );

  api.registerTool(
    {
      name: "memory_store_rag",
      label: "Memory Store (RAG)",
      description:
        "Persist a fact, preference, or decision into long-term memory so it survives context resets. Prefer this for explicit user instructions (\"remember that...\").",
      parameters: {
        type: "object",
        properties: {
          text: { type: "string", description: "The fact or excerpt to remember." },
          chatId: { type: "string", description: "Chat scope id. Use 'global' for cross-chat facts." },
          channel: { type: "string", description: "Channel id (e.g. 'whatsapp', 'web')." },
        },
        required: ["text"],
        additionalProperties: false,
      },
      async execute(_toolCallId, params) {
        const text = String(params.text ?? "").trim();
        if (!text) {
          return { content: [{ type: "text", text: "Refusing to store empty text." }], details: { stored: false } };
        }
        const chatId = String(params.chatId ?? "global");
        const channel = String(params.channel ?? "tool");
        await engine.store({
          text,
          chatId,
          isGroup: false,
          channel,
          isOwner: true,
          role: "summary",
          source: "live",
        });
        return {
          content: [{ type: "text", text: `Stored: "${text.slice(0, 120)}..."` }],
          details: { stored: true, chatId, channel },
        };
      },
    },
    { name: "memory_store_rag" },
  );
}
