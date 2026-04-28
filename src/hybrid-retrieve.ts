import type { Logger } from "./logger.js";
import type { ResolvedConfig, IsolationMode } from "./config.js";
import type { OllamaEmbedClient } from "./ollama-embeddings.js";
import { OllamaEmbeddingError } from "./ollama-embeddings.js";
import { QdrantBackend, type HybridSearchHit, type QdrantFilter, type QdrantFilterClause } from "./qdrant-client.js";
import { buildSparseVector } from "./sparse-tokenizer.js";

export type RetrieveContext = {
  chatId?: string;
  isGroup?: boolean;
  channel?: string;
};

export type RetrieveOptions = {
  query: string;
  context?: RetrieveContext;
  limit?: number;
};

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * Run a hybrid dense + sparse retrieval against Qdrant, applying:
 *   - isolation filter (per_chat | per_user | global_owner | global_all)
 *   - channel whitelist (optional)
 *   - recency half-life score boost (multiplicative)
 *
 * Returns prefetched hits (top reranker.topNIn) ready for the reranker stage.
 * Caller is responsible for downstream rerank + parent expansion + budget trim.
 */
export async function hybridRetrieve(params: {
  cfg: ResolvedConfig;
  qdrant: QdrantBackend;
  embed: OllamaEmbedClient;
  logger: Logger;
  options: RetrieveOptions;
}): Promise<HybridSearchHit[]> {
  const { cfg, qdrant, embed, logger, options } = params;

  let denseQuery: number[];
  try {
    denseQuery = await embed.embedQuery(options.query);
  } catch (err) {
    if (err instanceof OllamaEmbeddingError) {
      logger.warn(`memory-rag: embedding failed, returning empty hits: ${err.message}`);
      return [];
    }
    throw err;
  }

  const sparseQuery = buildSparseVector(options.query);
  const filter = buildIsolationFilter(cfg.isolation, cfg.ownerJids, cfg.channels.whitelist, options.context);
  const fusion = cfg.retrieval.hybridFusion;
  const prefetchLimit = cfg.reranker.topNIn;
  const limit = options.limit ?? cfg.reranker.topNIn;

  let hits: HybridSearchHit[];
  try {
    hits = await qdrant.hybridSearch({
      denseQuery,
      sparseQuery,
      limit,
      prefetchLimit,
      ...(filter ? { filter } : {}),
      fusion,
      ...(cfg.retrieval.minScore > 0 ? { scoreThreshold: cfg.retrieval.minScore } : {}),
    });
  } catch (err) {
    logger.warn(`memory-rag: hybrid search failed, returning empty hits: ${String(err)}`);
    return [];
  }

  if (cfg.retrieval.recencyHalfLifeDays > 0) {
    const halfLifeMs = cfg.retrieval.recencyHalfLifeDays * DAY_MS;
    const now = Date.now();
    for (const h of hits) {
      const age = Math.max(0, now - (h.payload?.timestamp ?? now));
      const boost = Math.pow(0.5, age / halfLifeMs);
      h.score = h.score * (0.7 + 0.3 * boost);
    }
    hits.sort((a, b) => b.score - a.score);
  }

  return hits;
}

function buildIsolationFilter(
  mode: IsolationMode,
  ownerJids: string[],
  channelWhitelist: string[] | null,
  context?: RetrieveContext,
): QdrantFilter | undefined {
  const must: QdrantFilterClause[] = [];
  const should: QdrantFilterClause[] = [];

  if (channelWhitelist && channelWhitelist.length > 0) {
    if (channelWhitelist.length === 1) {
      must.push({ key: "channel", match: { value: channelWhitelist[0]! } });
    } else {
      for (const ch of channelWhitelist) {
        should.push({ key: "channel", match: { value: ch } });
      }
    }
  }

  switch (mode) {
    case "per_chat": {
      if (context?.chatId) {
        must.push({ key: "chatId", match: { value: context.chatId } });
      }
      break;
    }
    case "per_user": {
      for (const jid of ownerJids) {
        should.push({ key: "senderJid", match: { value: jid } });
      }
      break;
    }
    case "global_owner": {
      if (context?.chatId) {
        should.push({ key: "chatId", match: { value: context.chatId } });
      }
      should.push({ key: "isOwner", match: { value: true } });
      break;
    }
    case "global_all":
    default:
      break;
  }

  if (must.length === 0 && should.length === 0) return undefined;
  const filter: QdrantFilter = {};
  if (must.length > 0) filter.must = must;
  if (should.length > 0) filter.should = should;
  return filter;
}
