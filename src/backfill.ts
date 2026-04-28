import { readFile, readdir, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { join, basename } from "node:path";
import type { MemoryEngine } from "./engine.js";
import { expandHome } from "./paths.js";

export type BackfillReport = {
  filesScanned: number;
  filesSkipped: number;
  messagesIngested: number;
  messagesFiltered: number;
  errors: number;
};

type BackfillOptions = {
  sessionsDir?: string;
  channel?: string;
  dryRun?: boolean;
  onProgress?: (msg: string) => void;
  /** Skip session files whose name contains these substrings. */
  skipSubstrings?: string[];
};

/**
 * Stream every session JSONL under `~/.openclaw/agents/main/sessions/` and
 * upsert all real conversation turns into Qdrant. Designed for large logs:
 * we read line-by-line (not full file in RAM) and embed in batches.
 */
export async function backfillSessions(
  engine: MemoryEngine,
  opts: BackfillOptions = {},
): Promise<BackfillReport> {
  const dir = expandHome(opts.sessionsDir ?? "~/.openclaw/agents/main/sessions");
  const channel = opts.channel ?? "session-backfill";
  const log = opts.onProgress ?? ((m: string) => engine.logger.info(m));
  // .trajectory.* and .checkpoint.* duplicate session content with tool-call
  // noise; skip them. .reset.* and .deleted.* are previous incarnations of a
  // session (rotated on the daily 14:00 UTC reset) and DO contain real
  // history, so we ingest them. Duplicates across rotation are fine because
  // every store() is keyed by a fresh WAL event id.
  const skipSubs = opts.skipSubstrings ?? [".trajectory.", ".checkpoint.", ".trajectory-path"];

  const report: BackfillReport = {
    filesScanned: 0,
    filesSkipped: 0,
    messagesIngested: 0,
    messagesFiltered: 0,
    errors: 0,
  };

  let entries: string[];
  try {
    entries = await readdir(dir);
  } catch (err) {
    log(`memory-rag backfill: cannot read sessions dir ${dir}: ${String(err)}`);
    return report;
  }

  // OpenClaw rotates session files on the daily 14:00 UTC reset:
  //   <uuid>.jsonl                         \u2190 active
  //   <uuid>.jsonl.reset.<ISO-stamp>.<id>Z \u2190 historical (still real history)
  //   <uuid>.jsonl.deleted.<ISO-stamp>.Z   \u2190 user-deleted (still on disk)
  //   <uuid>.jsonl.checkpoint.<id>.jsonl   \u2190 mid-session snapshot (skip, dup)
  //   <uuid>.trajectory.jsonl              \u2190 tool-call trace (skip, noisy dup)
  // We accept any name CONTAINING ".jsonl" (not just ending in it) so the
  // .reset/.deleted variants are picked up. The skipSubs filter then drops
  // .checkpoint, .trajectory, and .trajectory-path duplicates.
  const targets: string[] = [];
  for (const name of entries) {
    if (!name.includes(".jsonl")) continue;
    if (skipSubs.some((s) => name.includes(s))) continue;
    targets.push(name);
  }
  log(`memory-rag backfill: ${targets.length} session file(s) eligible (of ${entries.length} total entries)`);

  for (const name of targets) {
    const fullPath = join(dir, name);
    let st;
    try {
      st = await stat(fullPath);
    } catch {
      report.filesSkipped++;
      continue;
    }
    if (!st.isFile() || st.size === 0) {
      report.filesSkipped++;
      continue;
    }
    report.filesScanned++;
    const sessionId = basename(name, ".jsonl");
    log(`memory-rag backfill: \u2192 ${name} (${(st.size / 1024).toFixed(1)} KiB)`);
    try {
      const fileReport = await ingestSessionFile(engine, fullPath, sessionId, channel, opts.dryRun ?? false);
      report.messagesIngested += fileReport.ingested;
      report.messagesFiltered += fileReport.filtered;
    } catch (err) {
      report.errors++;
      engine.logger.warn(`memory-rag backfill: failed on ${name}: ${String(err)}`);
    }
  }

  log(
    `memory-rag backfill: done \u2014 files=${report.filesScanned}, skipped=${report.filesSkipped}, ingested=${report.messagesIngested}, filtered=${report.messagesFiltered}, errors=${report.errors}`,
  );
  return report;
}

type SessionLineMessage = {
  type?: string;
  message?: {
    role?: string;
    content?: unknown;
    timestamp?: number;
  };
  timestamp?: string;
};

async function ingestSessionFile(
  engine: MemoryEngine,
  path: string,
  sessionId: string,
  channel: string,
  dryRun: boolean,
): Promise<{ ingested: number; filtered: number }> {
  let ingested = 0;
  let filtered = 0;
  const stream = createReadStream(path, { encoding: "utf8" });
  const rl = createInterface({ input: stream, crlfDelay: Number.POSITIVE_INFINITY });
  let lastUserText: string | null = null;
  let lastUserTs: number | null = null;
  for await (const line of rl) {
    if (!line) continue;
    let parsed: SessionLineMessage;
    try {
      parsed = JSON.parse(line) as SessionLineMessage;
    } catch {
      continue;
    }
    if (parsed.type !== "message" || !parsed.message) continue;
    const role = parsed.message.role;
    const text = extractText(parsed.message.content);
    if (!text) continue;
    const ts = parsed.message.timestamp ?? (Date.parse(parsed.timestamp ?? "") || Date.now());
    if (role === "user") {
      if (isHeartbeatNoise(text)) {
        filtered++;
        continue;
      }
      lastUserText = text;
      lastUserTs = ts;
      if (engine.shouldCapture(text)) {
        if (!dryRun) {
          await engine.store({
            text,
            chatId: `session:${sessionId}`,
            isGroup: false,
            channel,
            sessionId,
            isOwner: true,
            role: "user",
            source: "backfill",
            timestamp: ts,
          });
        }
        ingested++;
      } else {
        filtered++;
      }
    } else if (role === "assistant") {
      if (text === "HEARTBEAT_OK" || text.length < 5) {
        filtered++;
        continue;
      }
      if (!engine.shouldCapture(text)) {
        filtered++;
        continue;
      }
      if (!dryRun) {
        await engine.store({
          text,
          chatId: `session:${sessionId}`,
          isGroup: false,
          channel,
          sessionId,
          isOwner: true,
          role: "assistant",
          source: "backfill",
          timestamp: ts,
          metadata: lastUserText ? { precedingUser: lastUserText.slice(0, 200) } : undefined,
        });
      }
      ingested++;
      lastUserText = null;
      lastUserTs = null;
    }
  }
  return { ingested, filtered };
}

function extractText(content: unknown): string | null {
  if (typeof content === "string") return content.trim() || null;
  if (!Array.isArray(content)) return null;
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const obj = block as Record<string, unknown>;
    if (obj.type === "text" && typeof obj.text === "string") {
      parts.push(obj.text);
    }
  }
  const joined = parts.join("\n").trim();
  return joined.length > 0 ? joined : null;
}

const HEARTBEAT_PATTERNS = [
  /^System:/m,
  /^System \(untrusted\):/m,
  /HEARTBEAT_OK/,
  /Read HEARTBEAT\.md if it exists/,
  /An async command completion event was triggered/,
  /Gateway restart restart ok/,
];

function isHeartbeatNoise(text: string): boolean {
  if (text.length > 4_000) return false;
  return HEARTBEAT_PATTERNS.some((re) => re.test(text));
}

/**
 * Best-effort import of OpenClaw's built-in sqlite-vec memory chunks. Useful
 * if the user previously ran `openclaw memory index`. Reads only `text` +
 * `path` so we don't depend on sqlite-vec column shape.
 */
export async function backfillFromSqlite(
  engine: MemoryEngine,
  sqlitePath: string,
  opts: { channel?: string; dryRun?: boolean; onProgress?: (msg: string) => void } = {},
): Promise<{ ingested: number; skipped: number }> {
  const log = opts.onProgress ?? ((m: string) => engine.logger.info(m));
  const channel = opts.channel ?? "memory-sqlite";
  let mod: typeof import("node:sqlite") | null = null;
  try {
    mod = await import("node:sqlite");
  } catch {
    log("memory-rag backfill: node:sqlite unavailable; skipping sqlite import.");
    return { ingested: 0, skipped: 0 };
  }
  const expanded = expandHome(sqlitePath);
  try {
    await readFile(expanded, { flag: "r" }).catch(() => {});
  } catch {
    log(`memory-rag backfill: sqlite path ${expanded} not readable`);
    return { ingested: 0, skipped: 0 };
  }
  const db = new (mod as unknown as { DatabaseSync: new (p: string) => { prepare: (s: string) => { all: () => unknown[] } } }).DatabaseSync(expanded);
  let ingested = 0;
  let skipped = 0;
  try {
    const rows = db.prepare("SELECT id, path, text, updated_at FROM chunks").all() as Array<{
      id: string;
      path: string;
      text: string;
      updated_at: number;
    }>;
    log(`memory-rag backfill: ${rows.length} sqlite chunk(s) found`);
    for (const row of rows) {
      if (!row.text || !engine.shouldCapture(row.text)) {
        skipped++;
        continue;
      }
      if (!opts.dryRun) {
        await engine.store({
          text: row.text,
          chatId: `mem-sqlite:${row.path}`,
          isGroup: false,
          channel,
          isOwner: true,
          role: "summary",
          source: "backfill",
          timestamp: row.updated_at,
          metadata: { originalPath: row.path },
        });
      }
      ingested++;
    }
  } catch (err) {
    log(`memory-rag backfill: sqlite read failed: ${String(err)}`);
  }
  return { ingested, skipped };
}
