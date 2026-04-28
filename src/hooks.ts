import type { MemoryEngine } from "./engine.js";

/**
 * OpenClaw plugin api shape \u2014 we only use the bits we need. Loose typing here
 * because the public surface of `OpenClawPluginApi` is huge and we don't want
 * to hard-pin to a private SDK type that can shift between minor releases.
 */
export type PluginApiLike = {
  on: (hookName: string, handler: (event: unknown, ctx: unknown) => unknown) => void;
  logger?: { info: (m: string, d?: unknown) => void; warn: (m: string, d?: unknown) => void; error: (m: string, d?: unknown) => void };
};

type AgentEndEvent = {
  runId?: string;
  messages: unknown[];
  success: boolean;
  error?: string;
  durationMs?: number;
};

type BeforePromptBuildEvent = {
  prompt: string;
  messages: unknown[];
};

type AgentContext = {
  runId?: string;
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  channelId?: string;
};

/**
 * Register before_prompt_build (auto-recall) and agent_end (auto-capture)
 * hooks. Both are no-ops if their respective filters are disabled. Both are
 * try/catch wrapped so a memory failure cannot crash the agent loop.
 */
export function registerHooks(api: PluginApiLike, engine: MemoryEngine): void {
  if (engine.cfg.filters.autoRecall) {
    api.on("before_prompt_build", async (rawEvent, rawCtx) => {
      const event = rawEvent as BeforePromptBuildEvent;
      const ctx = rawCtx as AgentContext;
      try {
        if (!isCapturedChannel(engine, ctx.channelId)) return;
        if (!event?.prompt || event.prompt.length < 5) return;
        const recallCtx = deriveRetrieveContext(ctx);
        const { rendered, blocks } = await engine.recall(event.prompt, recallCtx);
        if (!rendered) return;
        engine.logger.info(`memory-rag: injected ${blocks.length} memory block(s) into prompt`);
        return { prependContext: rendered };
      } catch (err) {
        engine.logger.warn(`memory-rag: auto-recall failed: ${String(err)}`);
        return undefined;
      }
    });
  }

  if (engine.cfg.filters.autoCapture) {
    api.on("agent_end", async (rawEvent, rawCtx) => {
      const event = rawEvent as AgentEndEvent;
      const ctx = rawCtx as AgentContext;
      try {
        if (!event.success) return;
        if (!isCapturedChannel(engine, ctx.channelId)) return;
        const exchange = extractLatestExchange(event.messages ?? []);
        if (!exchange) return;
        const chatId = deriveChatId(ctx);
        if (!chatId) return;
        await engine.storeExchange({
          user: exchange.user,
          assistant: exchange.assistant,
          chatId,
          isGroup: deriveIsGroup(ctx),
          channel: ctx.channelId ?? "unknown",
          ...(exchange.senderJid ? { senderJid: exchange.senderJid } : {}),
          ...(ctx.agentId ? { agentId: ctx.agentId } : {}),
          ...(ctx.sessionId ? { sessionId: ctx.sessionId } : {}),
        });
      } catch (err) {
        engine.logger.warn(`memory-rag: auto-capture failed: ${String(err)}`);
      }
    });
  }

  api.on("gateway_start", async () => {
    try {
      await engine.init();
      const health = await engine.health();
      const flags: string[] = [];
      flags.push(health.qdrant.ok ? `qdrant=ok(${health.qdrant.count ?? "?"})` : `qdrant=DOWN(${health.qdrant.reason})`);
      flags.push(health.embeddings.ok ? `embed=ok` : `embed=DOWN(${health.embeddings.reason})`);
      flags.push(health.reranker.ok ? `rerank=ok` : `rerank=DOWN(${health.reranker.reason})`);
      flags.push(`wal=${health.wal.totalLines}lines/${health.wal.files}files`);
      engine.logger.info(`memory-rag: ready \u2014 ${flags.join(" ")}`);
    } catch (err) {
      engine.logger.error(`memory-rag: gateway_start init failed: ${String(err)}`);
    }
  });

  api.on("gateway_stop", async () => {
    try {
      await engine.close();
    } catch (err) {
      engine.logger.warn(`memory-rag: gateway_stop close failed: ${String(err)}`);
    }
  });
}

function isCapturedChannel(engine: MemoryEngine, channelId: string | undefined): boolean {
  const whitelist = engine.cfg.channels.whitelist;
  if (!whitelist || whitelist.length === 0) return true;
  if (!channelId) return false;
  return whitelist.includes(channelId);
}

function deriveRetrieveContext(ctx: AgentContext) {
  const chatId = deriveChatId(ctx);
  return {
    ...(chatId ? { chatId } : {}),
    isGroup: deriveIsGroup(ctx),
    ...(ctx.channelId ? { channel: ctx.channelId } : {}),
  };
}

/**
 * Derive a stable chat-id from the OpenClaw session key. Session keys look
 * roughly like `<channel>:<account>:<conversation>`; we hash the
 * conversation portion to avoid leaking phone numbers into the index name
 * while keeping per-chat isolation.
 */
function deriveChatId(ctx: AgentContext): string | undefined {
  const key = ctx.sessionKey ?? ctx.sessionId ?? ctx.agentId;
  if (!key) return undefined;
  return key;
}

function deriveIsGroup(ctx: AgentContext): boolean {
  const key = ctx.sessionKey ?? "";
  return /group|@g\.us|^g:/i.test(key);
}

/**
 * Walk the run's messages and return the most recent (user, assistant) pair.
 * Tolerant of pi-agent-core message shapes (string | array of content blocks).
 */
function extractLatestExchange(
  messages: unknown[],
): { user: string; assistant: string; senderJid?: string } | null {
  let lastUser: string | null = null;
  let assistantAccum: string[] = [];
  let lastUserSenderJid: string | undefined;

  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    const obj = m as Record<string, unknown>;
    const role = obj.role;
    const text = extractText(obj.content);
    if (role === "user" && text) {
      lastUser = text;
      assistantAccum = [];
      const meta = obj.metadata as { senderJid?: string } | undefined;
      lastUserSenderJid = meta?.senderJid;
    } else if (role === "assistant" && text) {
      assistantAccum.push(text);
    }
  }
  if (!lastUser || assistantAccum.length === 0) return null;
  return {
    user: lastUser,
    assistant: assistantAccum.join("\n\n"),
    ...(lastUserSenderJid ? { senderJid: lastUserSenderJid } : {}),
  };
}

function extractText(content: unknown): string | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object") {
      const obj = block as Record<string, unknown>;
      if (obj.type === "text" && typeof obj.text === "string") parts.push(obj.text);
    }
  }
  return parts.length > 0 ? parts.join("\n") : null;
}
