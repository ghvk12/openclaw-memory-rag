# openclaw-memory-rag

Hybrid (dense + BM42 sparse) Retrieval-Augmented Generation memory plugin for [OpenClaw](https://github.com/openclaw/openclaw).

Backed by **Qdrant** for vector + sparse storage, **Ollama** for local embeddings (`mxbai-embed-large`), and a **TEI-compatible cross-encoder reranker** (`BAAI/bge-reranker-v2-m3` via either HuggingFace TEI or the bundled `tools/reranker-sidecar/`), with sentence-window parent expansion, token-budget trimming, WhatsApp metadata enrichment, and durable JSONL WAL.

## Why

OpenClaw ships a built-in memory subsystem (sqlite-vec + FTS5). It is great for local correctness and small corpora. This plugin adds a **complementary** hybrid retriever for high-recall, multi-channel persistent memory — so your WhatsApp (and other) conversations stay searchable across context-window resets without losing specific entities (names, project IDs, model numbers).

It does **not** replace the built-in memory; it augments it via the plugin SDK's `MemoryCorpusSupplement` interface and registers `memory_recall_rag` / `memory_store_rag` tools.

## Requirements

- OpenClaw >= 2026.4.20
- Node 22.14+ (24 recommended)
- Docker (for Qdrant)
- Ollama (native or Docker) with the embedding model pulled:
  ```bash
  ollama pull mxbai-embed-large
  ```
- Qdrant running on `:6333`. From this repo root one level up:
  ```bash
  cd .. && docker compose up -d qdrant
  ```
- **(Optional but recommended)** A cross-encoder reranker. The plugin supports two backends:
  - **`endpoint: "tei"`** — a [TEI](https://github.com/huggingface/text-embeddings-inference)-compatible `/rerank` server. On amd64 hosts, the official HuggingFace `text-embeddings-inference` Docker image works directly. On Apple Silicon, TEI's amd64 image crashes under Rosetta (Intel MKL incompatibility), so use the bundled `tools/reranker-sidecar/` (a tiny FastAPI service that runs `BAAI/bge-reranker-v2-m3` natively via `sentence-transformers` with MPS acceleration). See [tools/reranker-sidecar/README.md](tools/reranker-sidecar/README.md).
  - **`endpoint: "ollama"`** (legacy default) — calls Ollama's `/api/embeddings`. Most community GGUF ports of cross-encoder rerankers don't actually return real reranking scores through this endpoint; the plugin gracefully falls back to hybrid scores when this happens. Kept for back-compat only — strongly prefer `endpoint: "tei"`.

## Install

```bash
# Once published to ClawHub:
openclaw plugins install openclaw-memory-rag
openclaw plugins enable memory-rag

# Or, from a local checkout:
cd ./openclaw-memory-rag
npm install
npm run build
openclaw plugins install .
openclaw plugins enable memory-rag
```

Then add to `~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    entries: {
      "memory-rag": {
        enabled: true,
        config: {
          qdrant: { url: "http://localhost:6333", collection: "wa_memory_v1_mxbai_1024" },
          embeddings: { url: "http://localhost:11434", model: "mxbai-embed-large", dim: 1024 },
          // For real cross-encoder reranking, run the bundled sidecar
          // (tools/reranker-sidecar/) or HuggingFace TEI on port 8089:
          reranker: {
            enabled: true,
            endpoint: "tei",                    // "ollama" | "tei"
            url: "http://localhost:8089",       // sidecar/TEI port
            model: "BAAI/bge-reranker-v2-m3"    // HF model id under TEI
          },
          retrieval: { topK: 10, parentWindow: 2, tokenBudget: 4000 },
          isolation: "global_owner",
          // Phone numbers without @s.whatsapp.net. Replace with your own.
          ownerJids: ["1XXXXXXXXXX", "1YYYYYYYYYY"],
          // autoCapture defaults to false. Set true to index every (user, assistant)
          // exchange across configured channels into Qdrant. Read the Privacy
          // section before flipping this on.
          filters: { autoCapture: false, autoRecall: true }
        }
      }
    }
  }
}
```

Restart the gateway: `openclaw gateway --port 18790 --verbose` (or whatever port your config uses).

> **Security note:** non-loopback `qdrant.url` / `embeddings.url` / `reranker.url` MUST use `https://`. The plugin refuses to start with plaintext `http://` against a remote host so vector payloads (which contain raw conversation text) and `qdrant.apiKey` are not exposed in transit.

## Privacy

This plugin reads agent input/output from the `before_prompt_build` and `agent_end` hooks and persists the latest (user, assistant) exchange to Qdrant when `filters.autoCapture` is `true`. By default, `autoCapture` is **off** — you must explicitly enable it, ideally after restricting `channels.whitelist` to the channels you actually want indexed.

Captured payload includes message text, derived `chatId` (hashed session key), `senderJid` (when present in the message metadata), and the configured `agentId` / `sessionId`. Nothing is sent off your machine when Qdrant and Ollama are running on `localhost`. Pointing at a remote Qdrant or Ollama means those operators see the same payload — pick endpoints accordingly.

## CLI

```bash
openclaw memrag status                   # Qdrant + Ollama health, doc count, WAL lag
openclaw memrag doctor                   # full preflight: collection schema, models pulled, dim match
openclaw memrag backfill --source=both   # ingest existing sessions + sqlite chunks into Qdrant
openclaw memrag rebuild                  # drop + rebuild Qdrant collection from WAL
openclaw memrag search "ship deadline"   # debug retrieval with score breakdown
```

## Architecture

```
WhatsApp → Baileys → OpenClaw gateway
                          │
         ┌────────────────┴─────────────────┐
         │  before_prompt_build hook         │
         │   1. embed query (Ollama)         │
         │   2. hybrid query Qdrant          │
         │      (dense + BM42 sparse, RRF)   │
         │   3. cross-encoder rerank         │
         │   4. parent-window expand         │
         │   5. token-budget trim            │
         │   6. inject as prependContext     │
         └────────────────┬─────────────────┘
                          │
                  LLM (Gemini 3 Pro / DeepSeek V3 fallback)
                          │
         ┌────────────────┴─────────────────┐
         │  agent_end hook                   │
         │   1. WAL append (durable, first)  │
         │   2. embed exchange (Ollama)      │
         │   3. upsert to Qdrant w/ metadata │
         └────────────────────────────────────┘
```

## Failure handling

| Failure | Behavior |
|---|---|
| Qdrant unreachable | Auto-recall returns empty; WAL still writes; `doctor` shows red. Bridge keeps running. |
| Ollama unreachable | Same. Pending embed jobs queue; retried on reconnect. |
| Embedding model swapped | Collection is namespaced (`wa_memory_v1_mxbai_1024`). Run `memrag rebuild` to re-embed from WAL. |
| Both down | Built-in sqlite-vec memory still answers. Bridge runs. |

## Slot configuration

OpenClaw distinguishes between two ways a memory plugin can participate:

- **Additive (default for this plugin):** the plugin registers a `MemoryCorpusSupplement` and the `memory_recall_rag` / `memory_store_rag` tools, plus `before_prompt_build` / `agent_end` hooks. It runs *alongside* whatever active memory plugin you already have (`memory-core`, `memory-lancedb`, `memory-wiki`, etc.). This is the recommended setup for most users.

  No special config required — just enable it:

  ```json5
  {
    plugins: {
      entries: {
        "memory-rag": { enabled: true, config: { /* ... */ } }
      }
    }
  }
  ```

- **Slotted as the active memory backend:** the plugin's manifest declares `kind: "memory"`, so OpenClaw will accept it as the value of `plugins.slots.memory`. Some users prefer this when they want `memory-rag` to be the *only* memory layer in the gateway and want to disable `memory-core` / `memory-lancedb` to avoid duplicate indexing.

  ```json5
  {
    plugins: {
      slots: { memory: "memory-rag" },
      entries: {
        "memory-rag":     { enabled: true, config: { /* ... */ } },
        "memory-core":    { enabled: false },
        "memory-lancedb": { enabled: false }
      }
    }
  }
  ```

  > **Caveat:** `memory-rag` does not implement the exclusive memory-runtime contract (`registerMemoryCapability`'s `runtime` / `flushPlanResolver` / `promptBuilder` fields). When slotted, recall and storage still go through this plugin's own tools and hooks (which is the intended path), but core memory plumbing that expects a slot owner with a full runtime may degrade or no-op. If that matters for your deployment, prefer the additive setup above.

## License

MIT
