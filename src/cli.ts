#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { resolveConfig, type ResolvedConfig } from "./config.js";
import { consoleLogger } from "./logger.js";
import { MemoryEngine } from "./engine.js";
import { backfillSessions, backfillFromSqlite } from "./backfill.js";
import { importWhatsAppExport } from "./whatsapp-import.js";
import { expandHome } from "./paths.js";

type Cmd =
  | "doctor"
  | "backfill"
  | "rebuild-from-wal"
  | "search"
  | "store"
  | "import-whatsapp"
  | "help";

async function main(argv: string[]): Promise<number> {
  const [, , rawCmd = "help", ...rest] = argv;
  if (rawCmd === "help" || rawCmd === "--help" || rawCmd === "-h") {
    printHelp();
    return 0;
  }
  const cmd = rawCmd as Cmd;

  const cfg = await loadConfig(extractFlag(rest, "--config"));
  const logger = consoleLogger();
  const engine = new MemoryEngine(cfg, logger);
  await engine.init();

  try {
    switch (cmd) {
      case "doctor":
        return await runDoctor(engine);
      case "backfill":
        return await runBackfill(engine, rest);
      case "rebuild-from-wal":
        return await runRebuild(engine);
      case "search":
        return await runSearch(engine, rest);
      case "store":
        return await runStore(engine, rest);
      case "import-whatsapp":
        return await runImportWhatsApp(engine, rest);
      default:
        printHelp();
        return 2;
    }
  } finally {
    await engine.close();
  }
}

async function loadConfig(path: string | null): Promise<ResolvedConfig> {
  if (!path) {
    return resolveConfig({});
  }
  const raw = await readFile(resolve(expandHome(path)), "utf8");
  const parsed = JSON.parse(raw);
  return resolveConfig(parsed);
}

function extractFlag(args: string[], flag: string): string | null {
  const eq = args.find((a) => a.startsWith(`${flag}=`));
  if (eq) return eq.slice(flag.length + 1);
  const i = args.indexOf(flag);
  if (i >= 0 && i + 1 < args.length) return args[i + 1] ?? null;
  return null;
}

function hasFlag(args: string[], flag: string): boolean {
  return args.includes(flag);
}

async function runDoctor(engine: MemoryEngine): Promise<number> {
  const h = await engine.health();
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(h, null, 2));
  const allOk = h.qdrant.ok && h.embeddings.ok;
  return allOk ? 0 : 1;
}

async function runBackfill(engine: MemoryEngine, rest: string[]): Promise<number> {
  const dryRun = hasFlag(rest, "--dry-run");
  const channel = extractFlag(rest, "--channel") ?? "session-backfill";
  const sessionsDir = extractFlag(rest, "--sessions-dir") ?? undefined;
  const sqlitePath = extractFlag(rest, "--sqlite") ?? null;

  const sessionReport = await backfillSessions(engine, {
    ...(sessionsDir ? { sessionsDir } : {}),
    channel,
    dryRun,
    onProgress: (m) => process.stdout.write(`${m}\n`),
  });
  process.stdout.write(`\nSessions: ${JSON.stringify(sessionReport)}\n`);

  if (sqlitePath) {
    const sqlReport = await backfillFromSqlite(engine, sqlitePath, {
      channel: "memory-sqlite",
      dryRun,
      onProgress: (m) => process.stdout.write(`${m}\n`),
    });
    process.stdout.write(`SQLite: ${JSON.stringify(sqlReport)}\n`);
  }
  return sessionReport.errors > 0 ? 1 : 0;
}

async function runRebuild(engine: MemoryEngine): Promise<number> {
  const r = await engine.rebuildFromWal((n) => process.stdout.write(`replayed=${n}\n`));
  process.stdout.write(`\nDone: ${JSON.stringify(r)}\n`);
  return r.failed > 0 ? 1 : 0;
}

async function runSearch(engine: MemoryEngine, rest: string[]): Promise<number> {
  const query = rest.filter((a) => !a.startsWith("--")).join(" ").trim();
  if (!query) {
    process.stderr.write("usage: memrag search <query>\n");
    return 2;
  }
  const result = await engine.recall(query, undefined);
  process.stdout.write(`\n${result.rendered || "(no hits)"}\n`);
  process.stdout.write(`\n[blocks=${result.blocks.length}, raw=${result.rawHitCount}, reranked=${result.rerankedCount}]\n`);
  return 0;
}

async function runImportWhatsApp(engine: MemoryEngine, rest: string[]): Promise<number> {
  const positional = rest.filter((a) => !a.startsWith("--"));
  const zipPath = positional[0];
  if (!zipPath) {
    process.stderr.write(
      "usage: memrag import-whatsapp <path/to/export.zip> --bot-name=\"NAME\" [--chat-id=ID] [--channel=NAME] [--source-tag=TAG] [--no-bot] [--date-format=auto|DMY|MDY|YMD] [--dry-run]\n",
    );
    return 2;
  }
  const botName = extractFlag(rest, "--bot-name");
  if (!botName) {
    process.stderr.write("memrag import-whatsapp: --bot-name is required (e.g. --bot-name=\"Virinchi Vedhas\")\n");
    return 2;
  }
  const chatId = extractFlag(rest, "--chat-id");
  const channel = extractFlag(rest, "--channel");
  const sourceTag = extractFlag(rest, "--source-tag");
  const dateFormatRaw = extractFlag(rest, "--date-format");
  const dateFormat: "auto" | "DMY" | "MDY" | "YMD" =
    dateFormatRaw === "DMY" || dateFormatRaw === "MDY" || dateFormatRaw === "YMD"
      ? dateFormatRaw
      : "auto";

  const report = await importWhatsAppExport(engine, {
    zipPath,
    botName,
    ...(chatId ? { chatId } : {}),
    ...(channel ? { channel } : {}),
    ...(sourceTag ? { sourceTag } : {}),
    includeBot: !hasFlag(rest, "--no-bot"),
    dateFormat,
    dryRun: hasFlag(rest, "--dry-run"),
    onProgress: (m) => process.stdout.write(`${m}\n`),
  });
  process.stdout.write(`\nImport report: ${JSON.stringify(report, null, 2)}\n`);
  return report.errors > 0 ? 1 : 0;
}

async function runStore(engine: MemoryEngine, rest: string[]): Promise<number> {
  const text = rest.filter((a) => !a.startsWith("--")).join(" ").trim();
  if (!text) {
    process.stderr.write("usage: memrag store <text> [--chat=ID] [--channel=name]\n");
    return 2;
  }
  const chatId = extractFlag(rest, "--chat") ?? "global";
  const channel = extractFlag(rest, "--channel") ?? "cli";
  await engine.store({
    text,
    chatId,
    isGroup: false,
    channel,
    isOwner: true,
    role: "summary",
    source: "live",
  });
  process.stdout.write(`stored.\n`);
  return 0;
}

function printHelp(): void {
  process.stdout.write(`memrag \u2014 OpenClaw hybrid RAG memory CLI

Commands:
  doctor                          Probe Qdrant, Ollama, reranker, WAL.
  backfill [--dry-run]            Ingest ~/.openclaw/agents/main/sessions/*.jsonl
           [--sessions-dir=PATH]
           [--channel=NAME]
           [--sqlite=PATH]        Also ingest sqlite-vec chunks from PATH.
  rebuild-from-wal                Re-upsert every WAL event into Qdrant (idempotent).
  search <query>                  Run hybrid retrieval and print rendered context.
  store <text> [--chat=ID]        Manually store a fact.
                  [--channel=N]
  import-whatsapp <zip>           Ingest a WhatsApp "Export chat" zip.
                  --bot-name=NAME   (required) Display name of the bot in the export.
                  [--chat-id=ID]    Override synthetic chat id.
                  [--channel=NAME]  Default: whatsapp
                  [--source-tag=T]  Default: whatsapp-export
                  [--no-bot]        Skip bot-side messages.
                  [--date-format=auto|DMY|MDY|YMD]
                  [--dry-run]       Parse only.

Flags:
  --config <path>                 Override config json path.
`);
}

void main(process.argv).then(
  (code) => process.exit(code),
  (err) => {
    process.stderr.write(`memrag: ${String(err)}\n`);
    process.exit(1);
  },
);
