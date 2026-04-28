import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdtemp, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import type { MemoryEngine } from "./engine.js";
import { expandHome } from "./paths.js";

/**
 * WhatsApp chat-export importer.
 *
 * WhatsApp's "Export chat" feature produces a `.zip` containing `_chat.txt`
 * (the conversation transcript) and any media attachments. This module parses
 * that transcript, drops media stubs and system notices, classifies each
 * message as "from the bot" or "from the user", and pushes them through
 * `engine.store()` so they land in Qdrant + WAL with `source: "backfill"`.
 *
 * Why a separate importer (vs. the existing session backfill):
 *   - Session JSONLs are noisy (heartbeats, tool-call traces, system msgs).
 *   - Sessions get rotated/wiped on the daily 14:00 UTC reset, so anything
 *     older than retention is gone.
 *   - WhatsApp keeps the canonical, noise-free conversation server-side; an
 *     export is the cleanest possible long-term record.
 *
 * Sender classification is by display name. WhatsApp exports use the *contact
 * name as saved in your phone* on the side of whoever exported the chat. In a
 * 1:1 chat there are exactly two distinct senders; the bot's name is supplied
 * via `botName`, and whoever is NOT the bot is treated as the user (and tagged
 * `isOwner: true`).
 *
 * Per the user's import preferences (recorded in the planning step):
 *   - Both bot and user turns are imported (`includeBot: true` default).
 *   - Bot turns are tagged `metadata.sourceTag = "whatsapp-export"` so they
 *     remain distinguishable from `agent_end`-captured live turns.
 *   - Media stubs (`<attached: ...>`, `<Media omitted>`, `* omitted`) are
 *     dropped entirely; multi-line messages with a media stub PLUS a caption
 *     keep the caption.
 *   - The whole import goes under one synthetic `chatId` so parent-window
 *     expansion works across the conversation.
 */
export interface WhatsAppImportOptions {
  /** Path to the .zip exported from WhatsApp ("Export chat → Without media" works too). */
  zipPath: string;
  /** Bot's display name as it appears in the export (e.g. "Virinchi Vedhas"). */
  botName: string;
  /** Synthetic chat id for the import. Defaults to `whatsapp-import:bot:<slug>`. */
  chatId?: string;
  /** Channel tag for the captured rows. Stays inside the configured whitelist. */
  channel?: string;
  /** Tag stored in metadata to distinguish import rows from live captures. */
  sourceTag?: string;
  /** Set false to skip bot-side messages. Default true (per user choice). */
  includeBot?: boolean;
  /** Override date format if auto-detection picks the wrong one. */
  dateFormat?: "auto" | "DMY" | "MDY" | "YMD";
  /** Parse only; don't write to engine/WAL/Qdrant. */
  dryRun?: boolean;
  /** Streaming progress callback. */
  onProgress?: (msg: string) => void;
}

export interface WhatsAppImportReport {
  zipPath: string;
  chatFile: string | null;
  chatId: string;
  detectedSenders: string[];
  detectedUserSender: string | null;
  totalLines: number;
  parsedMessages: number;
  importedUser: number;
  importedBot: number;
  droppedMedia: number;
  droppedSystem: number;
  droppedTombstone: number;
  filteredByEngine: number;
  errors: number;
  fromTimestamp: number | null;
  toTimestamp: number | null;
  dateFormatUsed: "DMY" | "MDY" | "YMD";
}

interface ParsedMessage {
  ts: number;
  sender: string;
  text: string;
}

/* Bidi/format marks that get fully removed from the line. */
const BIDI_REMOVE_RE = /[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]/g;
/* "Narrow" / non-breaking spaces that recent WhatsApp builds insert between
 * the time and the AM/PM marker. Replaced with a regular ASCII space so the
 * header regex doesn't have to enumerate them. */
const NARROW_SPACE_RE = /[\u00a0\u2007\u2009\u202f\u200a\u2008]/g;

/* iOS export header:    [YYYY-MM-DD, H:MM:SS PM] Sender: text  (current macOS export)
 *                       [DD/MM/YY, HH:MM:SS] Sender: text       (older / non-US)
 *                       [DD/MM/YYYY, H:MM:SS AM] Sender: text   (UK/IN locale)
 * Android header:       DD/MM/YY, HH:MM - Sender: text          (Android export)
 *
 * Sender match `([^:]+?)` is conservative: WhatsApp display names can include
 * dots, dashes, accents, and emoji — but never raw colons before the body
 * separator, so non-greedy match up to the first ": " is safe. */
const IOS_HEADER_RE =
  /^\[(\d{1,2}[\/.\-]\d{1,2}[\/.\-]\d{2,4}|\d{4}[\/.\-]\d{1,2}[\/.\-]\d{1,2}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][.\s]*[Mm][.\s]*)?)\]\s+([^:]+?):\s?(.*)$/;
const ANDROID_HEADER_RE =
  /^(\d{1,2}[\/.\-]\d{1,2}[\/.\-]\d{2,4}|\d{4}[\/.\-]\d{1,2}[\/.\-]\d{1,2}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][.\s]*[Mm][.\s]*)?)\s+-\s+([^:]+?):\s?(.*)$/;

/* Lines whose ENTIRE remaining content is a media stub get dropped. If the
 * media stub is followed by additional text on subsequent (continuation)
 * lines, the caption survives. */
const MEDIA_STUB_RE_LIST: RegExp[] = [
  /^<attached:[^>]+>$/i,
  /^<Media omitted>$/i,
  /^(audio|image|video|sticker|document|GIF|Contact card|VCARD|location|POLL)\s+omitted\.?$/i,
  /^null$/,
];

const TOMBSTONE_RE_LIST: RegExp[] = [
  /^This message was deleted\.?$/i,
  /^You deleted this message\.?$/i,
  /^<This message was deleted>$/i,
  /^Waiting for this message$/i,
];

const SYSTEM_RE_LIST: RegExp[] = [
  /^Messages and calls are end-to-end encrypted/i,
  /security code (changed|with .*has changed)/i,
  /changed (the )?subject from/i,
  /changed (this group's icon|the group description)/i,
  /^Tap (to|on) (learn more|change)/i,
  /^You're now an admin/i,
  /^You added /i,
  /^You removed /i,
  /^Disappearing messages /i,
  /joined using this group's invite link/i,
  /created group/i,
  /^You created group/i,
];

const EDIT_MARKER_RE = /\s*<This message was edited>$/;

function stripBidi(line: string): string {
  return line.replace(BIDI_REMOVE_RE, "").replace(NARROW_SPACE_RE, " ");
}

function isMediaStub(text: string): boolean {
  const t = stripBidi(text).trim();
  if (!t) return false;
  return MEDIA_STUB_RE_LIST.some((re) => re.test(t));
}

function isTombstone(text: string): boolean {
  const t = stripBidi(text).trim();
  if (!t) return false;
  return TOMBSTONE_RE_LIST.some((re) => re.test(t));
}

function isSystemNotice(sender: string, text: string): boolean {
  /* WhatsApp system notices on iOS often appear as a "header-less" line whose
   * "sender" field is actually the notice itself — we catch those by checking
   * the body too. */
  const t = stripBidi(text).trim();
  const s = stripBidi(sender).trim();
  return (
    SYSTEM_RE_LIST.some((re) => re.test(t)) ||
    SYSTEM_RE_LIST.some((re) => re.test(s))
  );
}

/** Drop the `<attached: foo.opus>` line but keep any caption text. */
function stripMediaPrefix(text: string): string {
  const lines = text.split("\n").map((l) => l);
  const cleaned: string[] = [];
  for (const line of lines) {
    if (isMediaStub(line)) continue;
    cleaned.push(line);
  }
  return cleaned.join("\n").trim();
}

function parseHeader(line: string): {
  dateStr: string;
  timeStr: string;
  sender: string;
  text: string;
} | null {
  const stripped = stripBidi(line);
  const m = IOS_HEADER_RE.exec(stripped) ?? ANDROID_HEADER_RE.exec(stripped);
  if (!m) return null;
  return {
    dateStr: m[1]!,
    timeStr: m[2]!.trim(),
    sender: m[3]!.trim(),
    text: (m[4] ?? "").trim(),
  };
}

function parseDateTime(
  dateStr: string,
  timeStr: string,
  fmt: "DMY" | "MDY" | "YMD",
): number | null {
  const parts = dateStr.split(/[\/.\-]/).map((p) => Number.parseInt(p, 10));
  if (parts.length !== 3 || parts.some((n) => Number.isNaN(n))) return null;

  let year: number;
  let month: number;
  let day: number;

  /* If the FIRST segment is 4 digits, treat as YMD regardless of `fmt` —
   * unambiguous and the export was clearly produced under an ISO locale. */
  if (String(parts[0]).length === 4) {
    [year, month, day] = parts as [number, number, number];
  } else if (fmt === "MDY") {
    [month, day, year] = parts as [number, number, number];
  } else if (fmt === "YMD") {
    [year, month, day] = parts as [number, number, number];
  } else {
    [day, month, year] = parts as [number, number, number];
  }
  if (year < 100) year += 2000;
  if (month < 1 || month > 12) return null;
  if (day < 1 || day > 31) return null;

  const t = parseTime(timeStr);
  if (!t) return null;
  const d = new Date(year, month - 1, day, t.hour, t.minute, t.second);
  const ms = d.getTime();
  return Number.isFinite(ms) ? ms : null;
}

function parseTime(timeStr: string): { hour: number; minute: number; second: number } | null {
  const cleaned = timeStr.replace(/\s+/g, " ").replace(/\./g, "").toUpperCase();
  const m = /^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|A M|P M)?$/.exec(cleaned);
  if (!m) return null;
  let hour = Number.parseInt(m[1]!, 10);
  const minute = Number.parseInt(m[2]!, 10);
  const second = m[3] ? Number.parseInt(m[3], 10) : 0;
  const ampm = m[4]?.replace(/\s/g, "");
  if (ampm === "PM" && hour < 12) hour += 12;
  if (ampm === "AM" && hour === 12) hour = 0;
  if (hour > 23 || minute > 59 || second > 59) return null;
  return { hour, minute, second };
}

/**
 * Auto-detect the date format (DMY vs MDY vs YMD) by scanning headers.
 * Decision rule: if any first-segment > 12, must be DMY (or YMD). If any
 * second-segment > 12, must be MDY. If everything is ambiguous, default DMY
 * (most common WhatsApp export format outside the US).
 */
function detectDateFormat(samples: string[]): "DMY" | "MDY" | "YMD" {
  let firstGt12 = false;
  let secondGt12 = false;
  let firstIs4digit = false;
  for (const dateStr of samples) {
    const parts = dateStr.split(/[\/.\-]/).map((p) => Number.parseInt(p, 10));
    if (parts.length !== 3 || parts.some((n) => Number.isNaN(n))) continue;
    if (String(parts[0]).length === 4) firstIs4digit = true;
    if ((parts[0] ?? 0) > 12) firstGt12 = true;
    if ((parts[1] ?? 0) > 12) secondGt12 = true;
  }
  if (firstIs4digit) return "YMD";
  if (firstGt12) return "DMY";
  if (secondGt12) return "MDY";
  return "DMY";
}

async function extractZip(zipPath: string, destDir: string): Promise<void> {
  return await new Promise<void>((resolvePromise, rejectPromise) => {
    const child = spawn("unzip", ["-o", "-q", zipPath, "-d", destDir]);
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", rejectPromise);
    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise();
      } else {
        rejectPromise(
          new Error(`unzip failed (exit=${code ?? "null"}): ${stderr.trim() || "unknown"}`),
        );
      }
    });
  });
}

async function findChatFile(dir: string): Promise<string | null> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return null;
  }
  for (const e of entries) {
    if (e.isFile() && e.name === "_chat.txt") {
      return join(dir, e.name);
    }
  }
  for (const e of entries) {
    if (e.isFile() && e.name.toLowerCase().endsWith(".txt")) {
      return join(dir, e.name);
    }
  }
  for (const e of entries) {
    if (e.isDirectory()) {
      const found = await findChatFile(join(dir, e.name));
      if (found) return found;
    }
  }
  return null;
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64) || "bot";
}

/**
 * Two-pass parse: pass 1 collects header date strings to detect the format,
 * pass 2 emits messages with stable timestamps. Multi-line bodies are
 * accumulated until the next header line is found.
 */
async function* streamMessages(
  chatFile: string,
  opts: { dateFormat: "auto" | "DMY" | "MDY" | "YMD" },
): AsyncGenerator<{ msg: ParsedMessage; raw: string; format: "DMY" | "MDY" | "YMD" }> {
  const sampleSize = 200;
  const dateSamples: string[] = [];
  {
    const stream = createReadStream(chatFile, { encoding: "utf8" });
    const rl = createInterface({ input: stream, crlfDelay: Number.POSITIVE_INFINITY });
    for await (const line of rl) {
      const h = parseHeader(line);
      if (h) {
        dateSamples.push(h.dateStr);
        if (dateSamples.length >= sampleSize) break;
      }
    }
    rl.close();
    stream.destroy();
  }

  const fmt: "DMY" | "MDY" | "YMD" =
    opts.dateFormat === "auto" ? detectDateFormat(dateSamples) : opts.dateFormat;

  const stream2 = createReadStream(chatFile, { encoding: "utf8" });
  const rl2 = createInterface({ input: stream2, crlfDelay: Number.POSITIVE_INFINITY });
  let pendingHeader: ReturnType<typeof parseHeader> | null = null;
  let pendingBodyLines: string[] = [];
  let pendingTs: number | null = null;
  let pendingRaw: string[] = [];

  const flush = function* (): Generator<{ msg: ParsedMessage; raw: string; format: "DMY" | "MDY" | "YMD" }> {
    if (!pendingHeader || pendingTs === null) return;
    const fullText = [pendingHeader.text, ...pendingBodyLines].join("\n").trim();
    yield {
      msg: {
        ts: pendingTs,
        sender: pendingHeader.sender,
        text: fullText,
      },
      raw: pendingRaw.join("\n"),
      format: fmt,
    };
  };

  for await (const line of rl2) {
    const h = parseHeader(line);
    if (h) {
      yield* flush();
      pendingHeader = h;
      pendingBodyLines = [];
      pendingRaw = [line];
      pendingTs = parseDateTime(h.dateStr, h.timeStr, fmt);
    } else {
      if (pendingHeader) {
        const stripped = stripBidi(line);
        pendingBodyLines.push(stripped);
        pendingRaw.push(line);
      }
    }
  }
  yield* flush();
  rl2.close();
  stream2.destroy();
}

/**
 * Run the importer. Always extracts the zip into a fresh temp dir and cleans
 * up afterward (even on error). Returns a structured report so the CLI can
 * print summary numbers.
 */
export async function importWhatsAppExport(
  engine: MemoryEngine,
  opts: WhatsAppImportOptions,
): Promise<WhatsAppImportReport> {
  const log = opts.onProgress ?? ((m: string) => engine.logger.info(m));
  const zipPath = expandHome(opts.zipPath);
  await stat(zipPath).catch(() => {
    throw new Error(`whatsapp-import: zip not found at ${zipPath}`);
  });

  const tempRoot = await mkdtemp(join(tmpdir(), "wa-import-"));
  const report: WhatsAppImportReport = {
    zipPath,
    chatFile: null,
    chatId: opts.chatId ?? `whatsapp-import:bot:${slugify(opts.botName)}`,
    detectedSenders: [],
    detectedUserSender: null,
    totalLines: 0,
    parsedMessages: 0,
    importedUser: 0,
    importedBot: 0,
    droppedMedia: 0,
    droppedSystem: 0,
    droppedTombstone: 0,
    filteredByEngine: 0,
    errors: 0,
    fromTimestamp: null,
    toTimestamp: null,
    dateFormatUsed: "DMY",
  };

  try {
    log(`whatsapp-import: extracting ${zipPath} \u2192 ${tempRoot}`);
    await extractZip(zipPath, tempRoot);

    const chatFile = await findChatFile(tempRoot);
    if (!chatFile) {
      throw new Error(`whatsapp-import: no _chat.txt (or any .txt) found inside zip`);
    }
    report.chatFile = chatFile;
    log(`whatsapp-import: found transcript at ${chatFile}`);

    const channel = opts.channel ?? "whatsapp";
    const sourceTag = opts.sourceTag ?? "whatsapp-export";
    const includeBot = opts.includeBot ?? true;
    const botNameNorm = opts.botName.trim();

    const senderCounts = new Map<string, number>();
    let userSender: string | null = null;
    const dryRun = !!opts.dryRun;

    for await (const { msg, format } of streamMessages(chatFile, {
      dateFormat: opts.dateFormat ?? "auto",
    })) {
      report.totalLines += msg.text.split("\n").length;
      report.parsedMessages++;
      report.dateFormatUsed = format;

      senderCounts.set(msg.sender, (senderCounts.get(msg.sender) ?? 0) + 1);

      if (isSystemNotice(msg.sender, msg.text)) {
        report.droppedSystem++;
        continue;
      }
      if (isTombstone(msg.text)) {
        report.droppedTombstone++;
        continue;
      }

      let body = msg.text.replace(EDIT_MARKER_RE, "").trim();

      const onlyMediaStub = body.split("\n").every((l) => !l.trim() || isMediaStub(l));
      if (onlyMediaStub) {
        report.droppedMedia++;
        continue;
      }
      const cleaned = stripMediaPrefix(body);
      if (cleaned !== body) {
        body = cleaned;
      }
      if (!body) {
        report.droppedMedia++;
        continue;
      }

      const isBot = msg.sender === botNameNorm;
      if (!isBot && !userSender) {
        userSender = msg.sender;
      }
      if (isBot && !includeBot) continue;

      if (!engine.shouldCapture(body)) {
        report.filteredByEngine++;
        continue;
      }

      if (dryRun) {
        if (isBot) report.importedBot++;
        else report.importedUser++;
        if (report.fromTimestamp === null || msg.ts < report.fromTimestamp) {
          report.fromTimestamp = msg.ts;
        }
        if (report.toTimestamp === null || msg.ts > report.toTimestamp) {
          report.toTimestamp = msg.ts;
        }
        continue;
      }

      try {
        await engine.store({
          text: body,
          chatId: report.chatId,
          isGroup: false,
          channel,
          isOwner: true,
          role: isBot ? "assistant" : "user",
          source: "backfill",
          timestamp: msg.ts,
          senderName: msg.sender,
          metadata: {
            sourceTag,
            botName: botNameNorm,
            ...(isBot ? { side: "assistant-export" } : { side: "user-export" }),
          },
        });
        if (isBot) report.importedBot++;
        else report.importedUser++;
        if (report.fromTimestamp === null || msg.ts < report.fromTimestamp) {
          report.fromTimestamp = msg.ts;
        }
        if (report.toTimestamp === null || msg.ts > report.toTimestamp) {
          report.toTimestamp = msg.ts;
        }
        if ((report.importedUser + report.importedBot) % 100 === 0) {
          log(
            `whatsapp-import: progress \u2014 user=${report.importedUser} bot=${report.importedBot} dropped=${report.droppedMedia + report.droppedSystem + report.droppedTombstone}`,
          );
        }
      } catch (err) {
        report.errors++;
        engine.logger.warn(`whatsapp-import: store failed: ${String(err)}`);
      }
    }

    report.detectedSenders = Array.from(senderCounts.keys());
    report.detectedUserSender = userSender;

    if (!report.detectedSenders.includes(botNameNorm)) {
      log(
        `whatsapp-import: WARNING bot name "${botNameNorm}" was NOT seen in the transcript. Detected senders: ${JSON.stringify(report.detectedSenders)}. Re-run with --bot-name="<exact name>".`,
      );
    }
  } finally {
    await rm(tempRoot, { recursive: true, force: true }).catch(() => {});
  }

  return report;
}
