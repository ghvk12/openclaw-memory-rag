/**
 * Minimal logger surface that matches the OpenClaw plugin api.logger shape.
 * Lets us stub a logger in tests/CLI without depending on the full plugin api.
 */
export type Logger = {
  info: (msg: string, data?: unknown) => void;
  warn: (msg: string, data?: unknown) => void;
  error: (msg: string, data?: unknown) => void;
  debug?: (msg: string, data?: unknown) => void;
};

const PREFIX = "[memory-rag]";

export function consoleLogger(): Logger {
  return {
    info: (m, d) => (d !== undefined ? console.log(`${PREFIX} ${m}`, d) : console.log(`${PREFIX} ${m}`)),
    warn: (m, d) => (d !== undefined ? console.warn(`${PREFIX} ${m}`, d) : console.warn(`${PREFIX} ${m}`)),
    error: (m, d) => (d !== undefined ? console.error(`${PREFIX} ${m}`, d) : console.error(`${PREFIX} ${m}`)),
    debug: (m, d) => (process.env.MEMRAG_DEBUG ? console.error(`${PREFIX} [debug] ${m}`, d ?? "") : undefined),
  };
}

/** Wrap an OpenClaw api.logger so missing methods are coerced to no-ops. */
export function adoptLogger(apiLogger: Partial<Logger> | undefined): Logger {
  const fallback = consoleLogger();
  return {
    info: apiLogger?.info?.bind(apiLogger) ?? fallback.info,
    warn: apiLogger?.warn?.bind(apiLogger) ?? fallback.warn,
    error: apiLogger?.error?.bind(apiLogger) ?? fallback.error,
    debug: apiLogger?.debug?.bind(apiLogger) ?? fallback.debug,
  };
}
