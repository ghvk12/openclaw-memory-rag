#!/usr/bin/env bash
# tools/install/setup-qdrant.sh
#
# One-shot Qdrant setup for openclaw-memory-rag. Brings up the storage
# backend with the security posture the plugin expects:
#
#   - bound to 127.0.0.1 only (no LAN exposure)
#   - api-key authentication on (defense-in-depth, in case the loopback
#     bind ever breaks — and so the same setup works unchanged if you
#     later move Qdrant behind ssh/tailscale/HTTPS)
#   - data persisted via bind mount (survives container recreation)
#
# Idempotent: detects an existing container by name and bails out instead
# of clobbering it. Prints the api-key + the openclaw.json snippet to copy
# on success.
#
# Usage:
#   ./tools/install/setup-qdrant.sh                          # auth on, fresh key
#   QDRANT_API_KEY=<value>    ./tools/install/setup-qdrant.sh  # auth on, given key
#   QDRANT_API_KEY=disabled   ./tools/install/setup-qdrant.sh  # auth off (loopback only)
#   QDRANT_DATA_DIR=/path     ./tools/install/setup-qdrant.sh  # custom storage dir
#   QDRANT_VERSION=v1.12.4    ./tools/install/setup-qdrant.sh  # pinned image
#   CONTAINER_NAME=my-qdrant  ./tools/install/setup-qdrant.sh  # custom name
#   QDRANT_PORT=6333          ./tools/install/setup-qdrant.sh  # REST port (host)
#   QDRANT_GRPC_PORT=6334     ./tools/install/setup-qdrant.sh  # gRPC port (host)
#
# The api-key is the SAME value Qdrant validates server-side AND that you
# put in `~/.openclaw/openclaw.json` under
# plugins.entries."memory-rag".config.qdrant.apiKey. Keep them in sync.

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-openclaw-qdrant}"
QDRANT_VERSION="${QDRANT_VERSION:-v1.12.4}"
QDRANT_DATA_DIR="${QDRANT_DATA_DIR:-$HOME/openclaw-data/qdrant}"
QDRANT_API_KEY_INPUT="${QDRANT_API_KEY:-}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"

# --- preflight ------------------------------------------------------------
command -v docker >/dev/null 2>&1 || {
  echo "ERROR: docker is not on PATH. Install Docker Desktop first." >&2
  exit 2
}
command -v curl >/dev/null 2>&1 || {
  echo "ERROR: curl is not on PATH." >&2
  exit 2
}

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: a container named '$CONTAINER_NAME' already exists.

  This script refuses to clobber existing state. To recreate it
  (your data is safe — it lives on the bind mount, not in the container):

    docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME
    $0    # then re-run

  To use a different name instead, set CONTAINER_NAME=<new-name>.
EOF
  exit 1
fi

# --- resolve api-key ------------------------------------------------------
AUTH_ENABLED=1
if [[ "$QDRANT_API_KEY_INPUT" == "disabled" ]]; then
  AUTH_ENABLED=0
  QDRANT_API_KEY=""
elif [[ -n "$QDRANT_API_KEY_INPUT" ]]; then
  QDRANT_API_KEY="$QDRANT_API_KEY_INPUT"
  KEY_SOURCE="from QDRANT_API_KEY env var"
else
  QDRANT_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null) || {
    echo "ERROR: python3 not available to generate api-key. Set QDRANT_API_KEY=<value> manually." >&2
    exit 2
  }
  KEY_SOURCE="freshly generated (32 random bytes)"
fi

# --- storage dir ----------------------------------------------------------
mkdir -p "$QDRANT_DATA_DIR"

# --- run ------------------------------------------------------------------
DOCKER_ARGS=(
  -d
  --name "$CONTAINER_NAME"
  --restart unless-stopped
  -p "127.0.0.1:${QDRANT_PORT}:6333"
  -p "127.0.0.1:${QDRANT_GRPC_PORT}:6334"
  -v "$QDRANT_DATA_DIR:/qdrant/storage"
)
if [[ $AUTH_ENABLED -eq 1 ]]; then
  DOCKER_ARGS+=( -e "QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY" )
fi

echo "Starting $CONTAINER_NAME (qdrant/qdrant:$QDRANT_VERSION) ..."
docker run "${DOCKER_ARGS[@]}" "qdrant/qdrant:$QDRANT_VERSION" >/dev/null

# --- wait for ready -------------------------------------------------------
HEALTH_HEADER=()
[[ $AUTH_ENABLED -eq 1 ]] && HEALTH_HEADER=(-H "api-key: $QDRANT_API_KEY")

READY=0
for i in $(seq 1 20); do
  if curl -sf -m 1 "${HEALTH_HEADER[@]}" "http://127.0.0.1:${QDRANT_PORT}/healthz" >/dev/null 2>&1; then
    READY=1
    echo "  ready in ${i}s"
    break
  fi
  sleep 1
done

if [[ $READY -ne 1 ]]; then
  echo "ERROR: Qdrant did not become healthy within 20s. Check 'docker logs $CONTAINER_NAME'." >&2
  exit 3
fi

# --- print the snippet ----------------------------------------------------
cat <<EOF

================================================================
Qdrant is up at http://127.0.0.1:${QDRANT_PORT}  (loopback only, not on LAN)
  data dir : $QDRANT_DATA_DIR
  container: $CONTAINER_NAME
EOF

if [[ $AUTH_ENABLED -eq 1 ]]; then
  cat <<EOF
  auth     : api-key required ($KEY_SOURCE)

api-key (save this in your password manager — it is NOT recoverable
from the running container without docker inspect, and you'll need it
to recreate the container later):

  $QDRANT_API_KEY

Add this block to ~/.openclaw/openclaw.json under
plugins.entries."memory-rag".config:

  qdrant: {
    url: "http://localhost:${QDRANT_PORT}",
    collection: "wa_memory_v1_mxbai_1024",
    apiKey: "$QDRANT_API_KEY"
  }
EOF
else
  cat <<EOF
  auth     : DISABLED (loopback bind is your only protection)

Add this block to ~/.openclaw/openclaw.json under
plugins.entries."memory-rag".config:

  qdrant: {
    url: "http://localhost:${QDRANT_PORT}",
    collection: "wa_memory_v1_mxbai_1024"
  }
EOF
fi

cat <<EOF
================================================================

Next steps:
  1. Save the api-key above in your password manager.
  2. Paste the qdrant block into ~/.openclaw/openclaw.json
     (chmod 600 — keep it owner-only).
  3. Make sure ollama is running (\`ollama serve\` + \`ollama pull mxbai-embed-large\`).
  4. (Optional) Start the reranker sidecar — see tools/reranker-sidecar/README.md.
  5. Restart the gateway so memory-rag picks up the new config.

EOF
