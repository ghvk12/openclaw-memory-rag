import { QdrantClient } from "@qdrant/js-client-rest";
import type { Logger } from "./logger.js";
import type { QdrantConfig } from "./config.js";
import type { SparseVector } from "./sparse-tokenizer.js";

export class QdrantBackendError extends Error {
  override readonly cause?: unknown;
  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "QdrantBackendError";
    if (cause !== undefined) this.cause = cause;
  }
}

/**
 * Payload schema for memory-rag points. Stable shape \u2014 add fields, never rename.
 * `messageIndex` is a per-`chatId` monotonically increasing integer used for
 * sentence-window parent expansion.
 */
export type MemoryPayload = {
  text: string;
  chatId: string;
  messageIndex: number;
  isGroup: boolean;
  senderJid?: string;
  senderName?: string;
  isOwner: boolean;
  channel: string;
  channelMessageId?: string;
  agentId?: string;
  sessionId?: string;
  role: "user" | "assistant" | "exchange" | "system" | "summary";
  partnerJid?: string;
  timestamp: number;
  embeddingModel: string;
  source: "live" | "backfill" | "wal-replay";
  metadata?: Record<string, unknown>;
};

export type UpsertPoint = {
  id: string;
  dense: number[];
  sparse: SparseVector;
  payload: MemoryPayload;
};

export type HybridSearchHit = {
  id: string;
  score: number;
  payload: MemoryPayload;
};

export type HybridSearchOptions = {
  denseQuery: number[];
  sparseQuery: SparseVector;
  limit: number;
  prefetchLimit?: number;
  filter?: QdrantFilter;
  fusion?: "rrf" | "dense_only" | "sparse_only";
  scoreThreshold?: number;
};

export type QdrantFilter = {
  must?: QdrantFilterClause[];
  should?: QdrantFilterClause[];
  must_not?: QdrantFilterClause[];
};

export type QdrantFilterClause = {
  key: string;
  match?: { value: string | number | boolean };
  range?: { gte?: number; lte?: number; gt?: number; lt?: number };
};

const DENSE_VECTOR_NAME = "dense";
const SPARSE_VECTOR_NAME = "bm42";
const DEFAULT_PREFETCH = 50;

export class QdrantBackend {
  private client: QdrantClient;
  private collection: string;
  private dim: number;
  private logger: Logger;
  private initPromise: Promise<void> | null = null;

  constructor(cfg: QdrantConfig, dim: number, logger: Logger) {
    this.client = new QdrantClient({
      url: cfg.url,
      ...(cfg.apiKey ? { apiKey: cfg.apiKey } : {}),
      checkCompatibility: false,
    });
    this.collection = cfg.collection;
    this.dim = dim;
    this.logger = logger;
  }

  /** Idempotent: ensures the collection exists with dense + sparse named vectors. */
  async ensureCollection(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doEnsureCollection().catch((err) => {
      this.initPromise = null;
      throw err;
    });
    return this.initPromise;
  }

  private async doEnsureCollection(): Promise<void> {
    let exists = false;
    try {
      const collections = await this.client.getCollections();
      exists = collections.collections.some((c) => c.name === this.collection);
    } catch (err) {
      throw new QdrantBackendError(`Failed to list Qdrant collections (is it running?): ${String(err)}`, err);
    }

    if (exists) {
      const info = await this.client.getCollection(this.collection);
      const denseDim = (info.config?.params?.vectors as Record<string, { size?: number }> | undefined)?.[
        DENSE_VECTOR_NAME
      ]?.size;
      if (denseDim && denseDim !== this.dim) {
        throw new QdrantBackendError(
          `Collection "${this.collection}" has dense dim=${denseDim}, plugin expects ${this.dim}. ` +
            `Bump the collection name (e.g. ..._v2) or run \`openclaw memrag rebuild\`.`,
        );
      }
      this.logger.info(`memory-rag: Qdrant collection "${this.collection}" ready (dim=${this.dim})`);
      return;
    }

    this.logger.info(`memory-rag: creating Qdrant collection "${this.collection}" (dim=${this.dim})`);
    await this.client.createCollection(this.collection, {
      vectors: {
        [DENSE_VECTOR_NAME]: {
          size: this.dim,
          distance: "Cosine",
          on_disk: true,
        },
      },
      sparse_vectors: {
        [SPARSE_VECTOR_NAME]: {
          modifier: "idf",
        },
      },
      hnsw_config: { m: 16, ef_construct: 100, on_disk: true },
      optimizers_config: { default_segment_number: 2 },
    });

    await Promise.all([
      this.client.createPayloadIndex(this.collection, {
        field_name: "chatId",
        field_schema: "keyword",
        wait: true,
      }),
      this.client.createPayloadIndex(this.collection, {
        field_name: "isOwner",
        field_schema: "bool",
        wait: true,
      }),
      this.client.createPayloadIndex(this.collection, {
        field_name: "messageIndex",
        field_schema: "integer",
        wait: true,
      }),
      this.client.createPayloadIndex(this.collection, {
        field_name: "timestamp",
        field_schema: "integer",
        wait: true,
      }),
      this.client.createPayloadIndex(this.collection, {
        field_name: "channel",
        field_schema: "keyword",
        wait: true,
      }),
      this.client.createPayloadIndex(this.collection, {
        field_name: "isGroup",
        field_schema: "bool",
        wait: true,
      }),
    ]);
    this.logger.info(`memory-rag: collection "${this.collection}" created with payload indexes`);
  }

  async upsert(points: UpsertPoint[]): Promise<void> {
    if (points.length === 0) return;
    await this.ensureCollection();
    await this.client.upsert(this.collection, {
      wait: true,
      points: points.map((p) => ({
        id: p.id,
        vector: {
          [DENSE_VECTOR_NAME]: p.dense,
          [SPARSE_VECTOR_NAME]: { indices: p.sparse.indices, values: p.sparse.values },
        },
        payload: p.payload as unknown as Record<string, unknown>,
      })),
    });
  }

  async hybridSearch(opts: HybridSearchOptions): Promise<HybridSearchHit[]> {
    await this.ensureCollection();
    const fusion = opts.fusion ?? "rrf";
    const prefetchLimit = opts.prefetchLimit ?? DEFAULT_PREFETCH;

    if (fusion === "dense_only") {
      const res = await this.client.query(this.collection, {
        query: opts.denseQuery,
        using: DENSE_VECTOR_NAME,
        limit: opts.limit,
        with_payload: true,
        ...(opts.filter ? { filter: opts.filter as never } : {}),
        ...(opts.scoreThreshold !== undefined ? { score_threshold: opts.scoreThreshold } : {}),
      });
      return this.normalizeHits(res.points);
    }

    if (fusion === "sparse_only") {
      const res = await this.client.query(this.collection, {
        query: { indices: opts.sparseQuery.indices, values: opts.sparseQuery.values },
        using: SPARSE_VECTOR_NAME,
        limit: opts.limit,
        with_payload: true,
        ...(opts.filter ? { filter: opts.filter as never } : {}),
      });
      return this.normalizeHits(res.points);
    }

    const res = await this.client.query(this.collection, {
      prefetch: [
        {
          query: opts.denseQuery,
          using: DENSE_VECTOR_NAME,
          limit: prefetchLimit,
          ...(opts.filter ? { filter: opts.filter as never } : {}),
        },
        {
          query: { indices: opts.sparseQuery.indices, values: opts.sparseQuery.values },
          using: SPARSE_VECTOR_NAME,
          limit: prefetchLimit,
          ...(opts.filter ? { filter: opts.filter as never } : {}),
        },
      ],
      query: { fusion: "rrf" },
      limit: opts.limit,
      with_payload: true,
    });
    return this.normalizeHits(res.points);
  }

  /** Fetch a window of points around a hit by (chatId, messageIndex range). */
  async fetchWindow(chatId: string, fromIndex: number, toIndex: number, limit = 32): Promise<HybridSearchHit[]> {
    await this.ensureCollection();
    const res = await this.client.scroll(this.collection, {
      filter: {
        must: [
          { key: "chatId", match: { value: chatId } },
          { key: "messageIndex", range: { gte: fromIndex, lte: toIndex } },
        ],
      } as never,
      limit,
      with_payload: true,
      with_vector: false,
    });
    return res.points.map((p) => ({
      id: String(p.id),
      score: 1.0,
      payload: (p.payload ?? {}) as MemoryPayload,
    }));
  }

  async countAll(): Promise<number> {
    try {
      await this.ensureCollection();
      const res = await this.client.count(this.collection, { exact: true });
      return res.count;
    } catch {
      return 0;
    }
  }

  async getMaxMessageIndex(chatId: string): Promise<number> {
    await this.ensureCollection();
    const res = await this.client.scroll(this.collection, {
      filter: { must: [{ key: "chatId", match: { value: chatId } }] } as never,
      limit: 1,
      with_payload: ["messageIndex"],
      order_by: { key: "messageIndex", direction: "desc" } as never,
    });
    if (res.points.length === 0) return -1;
    const first = res.points[0]!;
    const idx = (first.payload as { messageIndex?: number } | undefined)?.messageIndex;
    return typeof idx === "number" ? idx : -1;
  }

  async dropCollection(): Promise<void> {
    try {
      await this.client.deleteCollection(this.collection);
      this.initPromise = null;
      this.logger.info(`memory-rag: dropped collection "${this.collection}"`);
    } catch (err) {
      this.logger.warn(`memory-rag: dropCollection failed (may not exist): ${String(err)}`);
    }
  }

  async probe(): Promise<{ ok: boolean; reason?: string }> {
    try {
      const collections = await this.client.getCollections();
      return {
        ok: true,
        reason: `Qdrant up. ${collections.collections.length} collection(s).`,
      };
    } catch (err) {
      return { ok: false, reason: `Qdrant unreachable: ${String(err)}` };
    }
  }

  private normalizeHits(points: ReadonlyArray<{ id: string | number; score?: number; payload?: unknown }>): HybridSearchHit[] {
    return points.map((p) => ({
      id: String(p.id),
      score: typeof p.score === "number" ? p.score : 0,
      payload: (p.payload ?? {}) as MemoryPayload,
    }));
  }
}
