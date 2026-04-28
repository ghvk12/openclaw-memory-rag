/**
 * Centralized prompt-injection detection patterns. Used by both
 *   - `engine.shouldCapture()`  — refuses to index attacker payloads at write time
 *   - `parent-expansion.sanitize()` — replaces matched text with a suppression
 *     marker at render time, as a defense-in-depth backstop for anything that
 *     slipped past write-time filtering (e.g., legacy WAL replay).
 *
 * Design notes:
 *   - Patterns intentionally allow ANY characters between the verb ("ignore",
 *     "disregard", "forget", "override") and the noun ("instructions", "rules",
 *     "prompt"), bounded to ~80 chars to limit false positives. Earlier
 *     versions required exactly one modifier word in between, which let the
 *     most common attack ("Ignore all previous instructions") through.
 *   - We deliberately keep patterns conservative — false positives suppress
 *     legitimate text. Better to miss a paraphrased attack (which the LLM is
 *     also told to treat as untrusted) than to suppress real conversation.
 *   - Patterns are case-insensitive; \b word boundaries prevent substring
 *     false positives (e.g., "ignored" alone won't trigger).
 */

const VERBS = "(?:ignore|disregard|forget|override|bypass|skip)";
const TARGETS = "(?:instructions?|rules?|prompts?|directives?|guidelines?|system\\s+prompts?)";
const PERSONA_VERBS = "(?:you\\s+are\\s+(?:now\\s+)?|act\\s+as|pretend\\s+to\\s+be|behave\\s+like|roleplay\\s+as)";

export const PROMPT_INJECTION_PATTERNS: RegExp[] = [
  // "ignore/disregard/forget [anything] (previous|prior|above|all|any|all your) instructions/rules/prompt"
  new RegExp(`\\b${VERBS}\\b[^.\\n]{0,80}\\b${TARGETS}\\b`, "i"),

  // High-precision instruction-rewrite preamble: "disregard the above",
  // "ignore everything above", "forget the prior". Catches attacks that omit
  // the explicit "instructions" noun but still rewrite context. Empirically
  // rare in legitimate WhatsApp prose; high signal-to-noise.
  /\b(?:disregard|ignore|forget|nevermind|never\s+mind)\s+(?:the\s+|all\s+(?:the\s+)?|everything\s+(?:that\s+is\s+)?)?(?:above|prior|previous|earlier|preceding|before)\b/i,

  // "do not follow (the) (system|developer) (prompt|instructions)"
  /\b(?:do\s+not|don'?t)\s+follow\b[^.\n]{0,40}\b(?:system|developer|admin|root|prior|previous|above)/i,

  // "you are now <new persona>" — a common jailbreak setup line
  new RegExp(`\\b${PERSONA_VERBS}[^.\\n]{0,80}`, "i"),

  // "from now on, ..." instruction-rewriting preamble
  /\bfrom\s+now\s+on\b[^.\n]{0,40}\b(?:you|act|pretend|respond|reply|answer)/i,

  // "new (system) instructions/prompt are/follow:" — common attack header
  /\bnew\s+(?:system\s+)?(?:instructions?|prompts?|directives?)\b/i,

  // Inline tag injection — must catch raw form (pre-escape) AND the
  // already-escaped &lt;...&gt; form so render-time double-checks are safe.
  /<\s*\/?\s*(?:system|assistant|developer|tool|function|user|relevant-memories)\b/i,
  /&lt;\s*\/?\s*(?:system|assistant|developer|tool|function|user|relevant-memories)\b/i,

  // ChatML / OpenAI special-token injection
  /<\|im_(?:start|end|sep)\|>/i,
  /\|<\|endoftext\|>\|/i,
];

/** Returns true iff the supplied text matches any prompt-injection pattern. */
export function looksLikePromptInjection(text: string): boolean {
  for (const re of PROMPT_INJECTION_PATTERNS) {
    if (re.test(text)) return true;
  }
  return false;
}
