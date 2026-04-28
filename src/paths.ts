import { homedir } from "node:os";
import { resolve, dirname } from "node:path";
import { mkdir } from "node:fs/promises";

/** Expand a leading `~` to the user's home directory. Pass-through otherwise. */
export function expandHome(p: string): string {
  if (!p) return p;
  if (p === "~") return homedir();
  if (p.startsWith("~/")) return resolve(homedir(), p.slice(2));
  return p;
}

/** Ensure a directory exists, creating any missing parents. Idempotent. */
export async function ensureDir(path: string): Promise<void> {
  await mkdir(path, { recursive: true });
}

/** Ensure the parent directory of a file path exists. */
export async function ensureParentDir(filePath: string): Promise<void> {
  await ensureDir(dirname(filePath));
}

/** Today's date in YYYY-MM-DD form using the local timezone (used for WAL file naming). */
export function todayStamp(now = new Date()): string {
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}
