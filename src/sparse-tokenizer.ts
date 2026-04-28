import { createHash } from "node:crypto";

/**
 * Sparse vector representation compatible with Qdrant's named sparse vectors:
 * `{indices: u32[], values: f32[]}` parallel arrays of equal length.
 */
export type SparseVector = {
  indices: number[];
  values: number[];
};

/**
 * Tiny BM25-style sparse vector generator. Not as good as a true SPLADE/BM42
 * model, but: (a) zero-dependency, (b) deterministic, (c) good enough to give
 * us a real keyword channel that catches specific entities (names, IDs, model
 * numbers) the dense encoder might soften away.
 *
 * Token strategy:
 *   - lowercased, whitespace + punctuation split
 *   - alphanumeric chunks of length >= 2 (filters single chars and pure punct)
 *   - phone numbers (digit runs of length >= 7) preserved whole
 *   - email addresses preserved whole
 *
 * Each unique token is mapped to a stable u32 index via SHA-1 prefix. Term
 * frequency is BM25-saturated using k1=1.2 (no IDF here \u2014 IDF lives on the
 * Qdrant side via the modifier when we set up the sparse index).
 */
const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "must", "shall", "to", "of", "in", "on", "at", "by",
  "for", "with", "from", "as", "this", "that", "these", "those", "it", "its",
  "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
  "your", "his", "our", "their", "if", "so", "no", "not", "yes", "ok", "okay",
]);

const EMAIL_RE = /[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g;
const PHONE_RE = /\+?\d[\d\s\-().]{6,}\d/g;
const TOKEN_RE = /[a-zA-Z0-9]{2,}/g;

const BM25_K1 = 1.2;

export function tokenize(text: string): string[] {
  if (!text) return [];
  const tokens: string[] = [];

  for (const m of text.matchAll(EMAIL_RE)) {
    tokens.push(m[0].toLowerCase());
  }
  const noEmail = text.replace(EMAIL_RE, " ");

  for (const m of noEmail.matchAll(PHONE_RE)) {
    const digits = m[0].replace(/\D/g, "");
    if (digits.length >= 7) tokens.push(digits);
  }
  const noPhone = noEmail.replace(PHONE_RE, " ");

  for (const m of noPhone.matchAll(TOKEN_RE)) {
    const lower = m[0].toLowerCase();
    if (STOPWORDS.has(lower)) continue;
    tokens.push(lower);
  }
  return tokens;
}

/**
 * Hash a token into a non-negative 30-bit integer index (Qdrant accepts u32,
 * we stay below 2^30 to be safe across language clients).
 */
function tokenIndex(token: string): number {
  const hash = createHash("sha1").update(token).digest();
  return (
    (hash[0]! << 22) |
    (hash[1]! << 14) |
    (hash[2]! << 6) |
    (hash[3]! >> 2)
  ) >>> 0;
}

/**
 * Build a sparse BM25-weighted vector. We omit the IDF factor here; Qdrant can
 * apply the IDF modifier server-side when the sparse index is created with
 * `modifier: "idf"`.
 */
export function buildSparseVector(text: string): SparseVector {
  const tokens = tokenize(text);
  if (tokens.length === 0) return { indices: [], values: [] };

  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);

  const indexToValue = new Map<number, number>();
  for (const [token, count] of tf.entries()) {
    const idx = tokenIndex(token);
    const weight = (count * (BM25_K1 + 1)) / (count + BM25_K1);
    const prev = indexToValue.get(idx) ?? 0;
    indexToValue.set(idx, prev + weight);
  }

  const sortedEntries = Array.from(indexToValue.entries()).sort((a, b) => a[0] - b[0]);
  return {
    indices: sortedEntries.map(([i]) => i),
    values: sortedEntries.map(([, v]) => v),
  };
}
