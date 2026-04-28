import { randomUUID } from "node:crypto";
import { writeFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Logger } from "./logger.js";
import type { ResolvedConfig } from "./config.js";
import { createOllamaEmbedClient, type OllamaEmbedClient } from "./ollama-embeddings.js";
import { createReranker, type Reranker } from "./reranker.js";
import {
  QdrantBackend,
  type HybridSearchHit,
  type MemoryPayload,
  type UpsertPoint,
} from "./qdrant-client.js";
import { Wal, type WalEvent } from "./wal.js";
import { hybridRetrieve, type RetrieveContext } from "./hybrid-retrieve.js";
import { expandAndBudget, renderContext, type ParentBlock } from "./parent-expansion.js";
import { buildSparseVector } from "./sparse-tokenizer.js";
import { ensureDir, expandHome } from "./paths.js";
import { looksLikePromptInjection } from "./prompt-injection.js";

export type StoreParams = {
  text: string;
  chatId: string;
  isGroup: boolean;
  channel: string;
  senderJid?: string;
  senderName?: string;
  isOwner?: boolean;
  channelMessageId?: string;
  agentId?: string;
  sessionId?: string;
  partnerJid?: string;
  role?: MemoryPayload["role"];
  timestamp?: number;
  source?: "live" | "backfill" | "wal-replay";
  metadata?: Record<string, unknown>;
};

export type RecallResult = {
  blocks: ParentBlock[];
  rendered: string;
  rawHitCount: number;
  rerankedCount: number;
};

const REACTION_LIKE_RE = /^[\s\p{Emoji_Presentation}\p{Extended_Pictographic}\p{Emoji}\u200D]{0,8}$/u;

const MAX_TEXT_LEN_FOR_INDEX = 8_000;

export class MemoryEngine {
  readonly cfg: ResolvedConfig;
  readonly logger: Logger;
  readonly qdrant: QdrantBackend;
  readonly embed: OllamaEmbedClient;
  readonly reranker: Reranker;
  readonly wal: Wal;
  private readonly chatIndexCache = new Map<string, number>();
  private indexInitDone = false;

  constructor(cfg: ResolvedConfig, logger: Logger) {
    this.cfg = cfg;
    this.logger = logger;
    this.embed = createOllamaEmbedClient(cfg.embeddings, logger);
    this.qdrant = new QdrantBackend(cfg.qdrant, cfg.embeddings.dim, logger);
    this.reranker = createReranker(cfg.reranker, logger);
    this.wal = new Wal(cfg.storage.wal, logger);
  }

  async init(): Promise<void> {
    if (this.indexInitDone) return;
    await this.wal.init();
    try {
      await this.qdrant.ensureCollection();
    } catch (err) {
      this.logger.warn(`memory-rag: Qdrant init failed; running in WAL-only degraded mode: ${String(err)}`);
    }
    this.indexInitDone = true;
  }

  async close(): Promise<void> {
    await this.wal.close();
  }

  isOwnerJid(jid: string | undefined): boolean {
    if (!jid) return false;
    const normalized = jid.replace(/@.*$/, "");
    return this.cfg.ownerJids.includes(normalized);
  }

  shouldCapture(text: string): boolean {
    if (!text) return false;
    const tokens = text.trim().split(/\s+/).length;
    if (tokens < this.cfg.filters.minTokens) return false;
    if (text.length > MAX_TEXT_LEN_FOR_INDEX) return false;
    if (this.cfg.filters.skipReactions && REACTION_LIKE_RE.test(text.trim())) return false;
    if (looksLikePromptInjection(text)) return false;
    return true;
  }

  async recall(query: string, context?: RetrieveContext): Promise<RecallResult> {
    if (!this.cfg.filters.autoRecall) {
      return { blocks: [], rendered: "", rawHitCount: 0, rerankedCount: 0 };
    }
    if (!query || query.trim().length < 3) {
      return { blocks: [], rendered: "", rawHitCount: 0, rerankedCount: 0 };
    }

    const raw = await hybridRetrieve({
      cfg: this.cfg,
      qdrant: this.qdrant,
      embed: this.embed,
      logger: this.logger,
      options: { query, ...(context ? { context } : {}) },
    });

    if (raw.length === 0) {
      return { blocks: [], rendered: "", rawHitCount: 0, rerankedCount: 0 };
    }

    const reranked = await this.reranker.rerank(query, raw, this.cfg.reranker.topNOut);
    const blocks = await expandAndBudget({
      hits: reranked,
      cfg: this.cfg.retrieval,
      qdrant: this.qdrant,
      logger: this.logger,
    });
    const rendered = renderContext(blocks);
    return { blocks, rendered, rawHitCount: raw.length, rerankedCount: reranked.length };
  }

  /**
   * Index a single piece of text. Always writes to WAL first (durable),
   * then attempts Qdrant upsert (best-effort, swallows errors).
   */
  async store(params: StoreParams): Promise<void> {
    if (!this.shouldCapture(params.text)) return;

    const isOwnerEffective =
      typeof params.isOwner === "boolean" ? params.isOwner : this.isOwnerJid(params.senderJid);

    const walEvent: WalEvent = await this.wal.append({
      kind: "exchange",
      text: params.text,
      chatId: params.chatId,
      isGroup: params.isGroup,
      isOwner: isOwnerEffective,
      channel: params.channel,
      ...(params.senderJid ? { senderJid: params.senderJid } : {}),
      ...(params.senderName ? { senderName: params.senderName } : {}),
      ...(params.channelMessageId ? { channelMessageId: params.channelMessageId } : {}),
      ...(params.agentId ? { agentId: params.agentId } : {}),
      ...(params.sessionId ? { sessionId: params.sessionId } : {}),
      ...(params.partnerJid ? { partnerJid: params.partnerJid } : {}),
      ...(params.metadata ? { metadata: params.metadata } : {}),
      ...(params.timestamp ? { ts: params.timestamp } : {}),
    });

    try {
      const messageIndex = await this.allocateMessageIndex(params.chatId);
      const dense = await this.embed.embedQuery(params.text);
      const sparse = buildSparseVector(params.text);
      const point: UpsertPoint = {
        id: walEvent.id,
        dense,
        sparse,
        payload: {
          text: params.text,
          chatId: params.chatId,
          messageIndex,
          isGroup: params.isGroup,
          isOwner: isOwnerEffective,
          channel: params.channel,
          role: params.role ?? "exchange",
          timestamp: walEvent.ts,
          embeddingModel: this.embed.model,
          source: params.source ?? "live",
          ...(params.senderJid ? { senderJid: params.senderJid } : {}),
          ...(params.senderName ? { senderName: params.senderName } : {}),
          ...(params.channelMessageId ? { channelMessageId: params.channelMessageId } : {}),
          ...(params.agentId ? { agentId: params.agentId } : {}),
          ...(params.sessionId ? { sessionId: params.sessionId } : {}),
          ...(params.partnerJid ? { partnerJid: params.partnerJid } : {}),
          ...(params.metadata ? { metadata: params.metadata } : {}),
        },
      };
      await this.qdrant.upsert([point]);
    } catch (err) {
      await this.appendFallbackMd(params.text, params.chatId, walEvent.ts);
      this.logger.warn(`memory-rag: Qdrant upsert failed (kept in WAL + MEMORY.md): ${String(err)}`);
    }
  }

  /**
   * Convenience: index a (user, assistant) exchange as two consecutive points.
   * Used by the agent_end hook so retrieval can later find either side.
   */
  async storeExchange(params: {
    user: string;
    assistant: string;
    chatId: string;
    isGroup: boolean;
    channel: string;
    senderJid?: string;
    senderName?: string;
    agentId?: string;
    sessionId?: string;
    partnerJid?: string;
    timestamp?: number;
  }): Promise<void> {
    if (params.user) {
      await this.store({
        text: params.user,
        chatId: params.chatId,
        isGroup: params.isGroup,
        channel: params.channel,
        ...(params.senderJid ? { senderJid: params.senderJid } : {}),
        ...(params.senderName ? { senderName: params.senderName } : {}),
        ...(params.agentId ? { agentId: params.agentId } : {}),
        ...(params.sessionId ? { sessionId: params.sessionId } : {}),
        ...(params.partnerJid ? { partnerJid: params.partnerJid } : {}),
        role: "user",
        ...(params.timestamp ? { timestamp: params.timestamp } : {}),
      });
    }
    if (params.assistant) {
      await this.store({
        text: params.assistant,
        chatId: params.chatId,
        isGroup: params.isGroup,
        channel: params.channel,
        senderJid: "assistant",
        senderName: "OpenClaw",
        isOwner: true,
        ...(params.agentId ? { agentId: params.agentId } : {}),
        ...(params.sessionId ? { sessionId: params.sessionId } : {}),
        ...(params.partnerJid ? { partnerJid: params.partnerJid } : {}),
        role: "assistant",
        ...(params.timestamp ? { timestamp: params.timestamp + 1 } : {}),
      });
    }
  }

  /** Allocate a unique, monotonically-increasing messageIndex per chat. */
  private async allocateMessageIndex(chatId: string): Promise<number> {
    const cached = this.chatIndexCache.get(chatId);
    if (typeof cached === "number") {
      const next = cached + 1;
      this.chatIndexCache.set(chatId, next);
      return next;
    }
    const max = await this.qdrant.getMaxMessageIndex(chatId).catch(() => -1);
    const next = max + 1;
    this.chatIndexCache.set(chatId, next);
    return next;
  }

  async health(): Promise<{
    qdrant: { ok: boolean; reason?: string; count?: number };
    embeddings: { ok: boolean; reason?: string };
    reranker: { ok: boolean; reason?: string };
    wal: { files: number; totalLines: number; lastTimestamp: number | null };
  }> {
    const [qdrantProbe, embedProbe, rerankProbe, walStats] = await Promise.all([
      this.qdrant.probe(),
      this.embed.probe(),
      this.reranker.probe(),
      this.wal.stats(),
    ]);
    let count: number | undefined;
    if (qdrantProbe.ok) {
      count = await this.qdrant.countAll();
    }
    return {
      qdrant: { ok: qdrantProbe.ok, ...(qdrantProbe.reason ? { reason: qdrantProbe.reason } : {}), ...(typeof count === "number" ? { count } : {}) },
      embeddings: embedProbe,
      reranker: rerankProbe,
      wal: walStats,
    };
  }

  /** Last-resort durable scratchpad when Qdrant is unreachable. */
  private async appendFallbackMd(text: string, chatId: string, ts: number): Promise<void> {
    try {
      const path = expandHome(this.cfg.storage.fallbackMd);
      await ensureDir(join(path, "..").replace(/\/$/, ""));
      const stamp = new Date(ts).toISOString();
      const line = `\n## ${stamp} (chat: ${chatId})\n${text}\n`;
      let existing = "";
      try {
        existing = await readFile(path, "utf8");
      } catch {
        existing = `# OpenClaw memory-rag fallback log\n\nQdrant was unreachable when the entries below were captured.\n`;
      }
      await writeFile(path, existing + line, "utf8");
    } catch (err) {
      this.logger.error(`memory-rag: fallback MD write failed: ${String(err)}`);
    }
  }

  /** Replay every WAL event into Qdrant. Idempotent (uses the WAL event id as point id). */
  async rebuildFromWal(progress?: (n: number) => void): Promise<{ replayed: number; failed: number }> {
    let replayed = 0;
    let failed = 0;
    const batch: UpsertPoint[] = [];
    const FLUSH_AT = 64;

    const flush = async (): Promise<void> => {
      if (batch.length === 0) return;
      try {
        await this.qdrant.upsert(batch);
      } catch (err) {
        failed += batch.length;
        this.logger.warn(`memory-rag: rebuild batch failed: ${String(err)}`);
      }
      batch.length = 0;
    };

    for await (const evt of this.wal.replay()) {
      try {
        const messageIndex = await this.allocateMessageIndex(evt.chatId);
        const [dense] = await this.embed.embedBatch([evt.text]);
        if (!dense) {
          failed++;
          continue;
        }
        const sparse = buildSparseVector(evt.text);
        batch.push({
          id: evt.id,
          dense,
          sparse,
          payload: {
            text: evt.text,
            chatId: evt.chatId,
            messageIndex,
            isGroup: evt.isGroup,
            isOwner: evt.isOwner,
            channel: evt.channel,
            role: evt.kind === "user_message" ? "user" : evt.kind === "assistant_message" ? "assistant" : "exchange",
            timestamp: evt.ts,
            embeddingModel: this.embed.model,
            source: "wal-replay",
            ...(evt.senderJid ? { senderJid: evt.senderJid } : {}),
            ...(evt.senderName ? { senderName: evt.senderName } : {}),
            ...(evt.channelMessageId ? { channelMessageId: evt.channelMessageId } : {}),
            ...(evt.agentId ? { agentId: evt.agentId } : {}),
            ...(evt.sessionId ? { sessionId: evt.sessionId } : {}),
            ...(evt.partnerJid ? { partnerJid: evt.partnerJid } : {}),
            ...(evt.metadata ? { metadata: evt.metadata } : {}),
          },
        });
        replayed++;
        if (batch.length >= FLUSH_AT) {
          await flush();
          progress?.(replayed);
        }
      } catch (err) {
        failed++;
        this.logger.warn(`memory-rag: rebuild item failed: ${String(err)}`);
      }
    }
    await flush();
    return { replayed, failed };
  }

  /** Convert a recall result into the snapshot the corpus-supplement returns. */
  toCorpusResults(blocks: ParentBlock[]): Array<{
    corpus: string;
    path: string;
    title: string;
    score: number;
    snippet: string;
    id: string;
    sourceType: string;
    updatedAt: string;
  }> {
    const rows: ReturnType<MemoryEngine["toCorpusResults"]> = [];
    let n = 0;
    for (const block of blocks) {
      n++;
      for (const hit of block.hits) {
        const payload = hit.payload as MemoryPayload | undefined;
        if (!payload?.text) continue;
        rows.push({
          corpus: "memory-rag",
          path: `memrag://${block.chatId}/${payload.messageIndex}`,
          title: `Memory ${n} \u00b7 ${payload.channel} \u00b7 chat=${block.chatId}`,
          score: hit.score,
          snippet: payload.text.slice(0, 800),
          id: hit.id,
          sourceType: "memory-rag",
          updatedAt: new Date(payload.timestamp).toISOString(),
        });
      }
    }
    return rows;
  }
}

export function generatePointId(): string {
  return randomUUID();
}
