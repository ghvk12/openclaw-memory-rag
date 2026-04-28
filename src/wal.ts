import { open, FileHandle } from "node:fs/promises";
import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { Logger } from "./logger.js";
import { ensureDir, ensureParentDir, expandHome, todayStamp } from "./paths.js";

/**
 * WAL event \u2014 append-only durable log of every memory write. Qdrant is a
 * derived index; the WAL is the source of truth and can rebuild the index.
 */
export type WalEvent = {
  id: string;
  ts: number;
  kind: "exchange" | "user_message" | "assistant_message" | "summary" | "backfill";
  text: string;
  chatId: string;
  isGroup: boolean;
  senderJid?: string;
  senderName?: string;
  isOwner: boolean;
  channel: string;
  channelMessageId?: string;
  agentId?: string;
  sessionId?: string;
  partnerJid?: string;
  metadata?: Record<string, unknown>;
};

/**
 * Append-only JSONL writer with daily file rotation. Buffers a single file
 * handle per day and reopens on rotation; safe across multiple plugin
 * instances because we always open in append mode.
 */
export class Wal {
  private readonly dir: string;
  private readonly logger: Logger;
  private currentDay: string | null = null;
  private currentHandle: FileHandle | null = null;
  private writeChain: Promise<void> = Promise.resolve();

  constructor(dirPath: string, logger: Logger) {
    this.dir = expandHome(dirPath);
    this.logger = logger;
  }

  async init(): Promise<void> {
    await ensureDir(this.dir);
  }

  /** Append an event. Resolves after the line is durably handed to the OS write queue. */
  async append(eventInput: Omit<WalEvent, "id" | "ts"> & Partial<Pick<WalEvent, "id" | "ts">>): Promise<WalEvent> {
    const event: WalEvent = {
      id: eventInput.id ?? randomUUID(),
      ts: eventInput.ts ?? Date.now(),
      kind: eventInput.kind,
      text: eventInput.text,
      chatId: eventInput.chatId,
      isGroup: eventInput.isGroup,
      isOwner: eventInput.isOwner,
      channel: eventInput.channel,
      ...(eventInput.senderJid ? { senderJid: eventInput.senderJid } : {}),
      ...(eventInput.senderName ? { senderName: eventInput.senderName } : {}),
      ...(eventInput.channelMessageId ? { channelMessageId: eventInput.channelMessageId } : {}),
      ...(eventInput.agentId ? { agentId: eventInput.agentId } : {}),
      ...(eventInput.sessionId ? { sessionId: eventInput.sessionId } : {}),
      ...(eventInput.partnerJid ? { partnerJid: eventInput.partnerJid } : {}),
      ...(eventInput.metadata ? { metadata: eventInput.metadata } : {}),
    };

    const line = `${JSON.stringify(event)}\n`;
    this.writeChain = this.writeChain.then(() => this.writeLine(line, event.ts));
    await this.writeChain;
    return event;
  }

  private async writeLine(line: string, ts: number): Promise<void> {
    const day = todayStamp(new Date(ts));
    if (day !== this.currentDay) {
      if (this.currentHandle) {
        try {
          await this.currentHandle.close();
        } catch {
          // best-effort
        }
      }
      const filePath = join(this.dir, `${day}.jsonl`);
      await ensureParentDir(filePath);
      this.currentHandle = await open(filePath, "a");
      this.currentDay = day;
    }
    await this.currentHandle!.write(line);
  }

  async close(): Promise<void> {
    if (this.currentHandle) {
      try {
        await this.currentHandle.close();
      } catch (err) {
        this.logger.warn(`memory-rag: WAL close failed: ${String(err)}`);
      }
      this.currentHandle = null;
      this.currentDay = null;
    }
  }

  /**
   * Stream all WAL events on disk, oldest-first, file by file. Used by
   * `memrag rebuild` to repopulate Qdrant from durable storage.
   */
  async *replay(): AsyncGenerator<WalEvent> {
    const files = await this.listJsonlFiles();
    for (const file of files) {
      const content = await readFile(file, "utf8");
      const lines = content.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          yield JSON.parse(trimmed) as WalEvent;
        } catch (err) {
          this.logger.warn(`memory-rag: bad WAL line in ${file}: ${String(err)}`);
        }
      }
    }
  }

  async listJsonlFiles(): Promise<string[]> {
    try {
      await stat(this.dir);
    } catch {
      return [];
    }
    const entries = await readdir(this.dir);
    return entries
      .filter((e) => e.endsWith(".jsonl"))
      .sort()
      .map((e) => join(this.dir, e));
  }

  async stats(): Promise<{ files: number; totalLines: number; lastTimestamp: number | null }> {
    const files = await this.listJsonlFiles();
    let totalLines = 0;
    let lastTs: number | null = null;
    for (const file of files) {
      const content = await readFile(file, "utf8");
      const lines = content.split("\n").filter((l) => l.trim());
      totalLines += lines.length;
      const last = lines[lines.length - 1];
      if (last) {
        try {
          const evt = JSON.parse(last) as WalEvent;
          if (typeof evt.ts === "number") lastTs = evt.ts;
        } catch {
          // skip
        }
      }
    }
    return { files: files.length, totalLines, lastTimestamp: lastTs };
  }
}
