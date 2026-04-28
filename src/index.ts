import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";
import { resolveConfig } from "./config.js";
import { adoptLogger } from "./logger.js";
import { MemoryEngine } from "./engine.js";
import { buildMemoryEmbeddingProviderAdapter } from "./ollama-embeddings.js";
import { buildCorpusSupplement } from "./corpus-supplement.js";
import { registerHooks, type PluginApiLike } from "./hooks.js";
import { registerTools, type ToolApi } from "./tools.js";

/**
 * OpenClaw plugin entry. Migrated from the legacy `activate(api)` shape (compat
 * code `plugin-activate-entrypoint-alias`, deprecated) to the canonical
 * `definePluginEntry({ register })` from `openclaw/plugin-sdk/plugin-entry`.
 *
 * Loader contract recap:
 *   - `register(api)` MUST be synchronous. Returning a Promise is rejected with
 *     `Error: plugin register must be synchronous` and the plugin is unloaded.
 *   - All async work (engine.init, qdrant collection ensure, ollama probe) is
 *     deferred to the `gateway_start` hook and to fire-and-forget Promises.
 *   - Plugin shutdown is handled by the `gateway_stop` hook in `hooks.ts`,
 *     so we don't need a separate `deactivate` callback (the legacy
 *     `activate` return shape is no longer wired through `definePluginEntry`).
 *
 * The `api` parameter is the typed `OpenClawPluginApi`. We still cast a few
 * call sites through `apiObj` because `registerHooks` / `registerTools` were
 * written against loose structural types that survive minor SDK shifts.
 */
export default definePluginEntry({
  id: "memory-rag",
  name: "Memory RAG",
  description:
    "Hybrid (dense + BM42 sparse) RAG memory supplement for OpenClaw, backed by Qdrant + Ollama.",
  // `kind: "memory"` advertises slot eligibility so users can set
  // `plugins.slots.memory = "memory-rag"` (the original deployment pattern).
  // The plugin doesn't implement the exclusive memory runtime — recall/store
  // happen through the `memory_recall_rag` / `memory_store_rag` tools and
  // the `before_prompt_build` / `agent_end` hooks instead. See README.
  kind: "memory",
  register(api: OpenClawPluginApi) {
    const apiObj = api as unknown as Record<string, unknown>;
    const cfg = resolveConfig(api.pluginConfig ?? (apiObj.config as Record<string, unknown> | undefined)?.memoryRag ?? {});
    const logger = adoptLogger(api.logger as Parameters<typeof adoptLogger>[0]);
    const engine = new MemoryEngine(cfg, logger);

    const embedAdapter = buildMemoryEmbeddingProviderAdapter(cfg.embeddings, logger);
    try {
      api.registerMemoryEmbeddingProvider(embedAdapter as unknown as Parameters<typeof api.registerMemoryEmbeddingProvider>[0]);
      logger.info(`memory-rag: registered Ollama embedding provider (${embedAdapter.id})`);
    } catch (err) {
      logger.warn(`memory-rag: failed to register embedding provider: ${String(err)}`);
    }

    const corpusSupplement = buildCorpusSupplement(engine);
    try {
      api.registerMemoryCorpusSupplement({
        name: "memory-rag",
        priority: 50,
        ...corpusSupplement,
      } as unknown as Parameters<typeof api.registerMemoryCorpusSupplement>[0]);
      logger.info("memory-rag: registered Qdrant-backed corpus supplement");
    } catch (err) {
      logger.warn(`memory-rag: failed to register corpus supplement: ${String(err)}`);
    }

    try {
      registerTools(apiObj as unknown as ToolApi, engine);
      logger.info("memory-rag: registered tools (memory_recall_rag, memory_store_rag)");
    } catch (err) {
      logger.warn(`memory-rag: failed to register tools: ${String(err)}`);
    }

    registerHooks(apiObj as unknown as PluginApiLike, engine);
  },
});
