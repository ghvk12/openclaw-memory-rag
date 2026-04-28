# Security policy

## Supported versions

Only the latest release of openclaw-memory-rag receives security fixes.

## Reporting a vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

Use the GitHub private vulnerability reporting channel instead:

1. Go to the **Security** tab of this repository on GitHub.
2. Click **"Report a vulnerability"**.
3. Fill in the advisory form (title, description, severity estimate, steps to reproduce).
4. Submit — this opens a private draft advisory visible only to you and the maintainer.

Alternatively, you can navigate directly to:
https://github.com/ghvk12/openclaw-memory-rag/security/advisories/new

### What to include

- A clear description of the vulnerability and its impact.
- Step-by-step reproduction instructions.
- The plugin version and OS you tested on.
- A suggested severity (low / medium / high / critical) with reasoning.
- If you have a proof-of-concept, attach it to the advisory.

### What happens next

- You will receive an acknowledgment within **3 business days**.
- If the report is confirmed, a fix will be developed privately and coordinated with you before any public disclosure.
- You will be credited in the release notes and the security advisory unless you prefer to remain anonymous.
- Fixes for critical or high-severity issues target a **7-day** patch window from confirmation.

### Scope

Items in scope:

- Prompt-injection bypasses that let attacker-controlled memory payloads reach the LLM as trusted instructions.
- HTTPS-enforcement bypasses that expose conversation data or the Qdrant api-key in plaintext over a network.
- Path traversal or arbitrary file write via plugin configuration fields.
- SSRF via the reranker sidecar's `/rerank` endpoint.
- Credential or PII leakage from logs, WAL files, or IPC surfaces.

Out of scope:

- Attacks that require physical access to the machine or root/admin privileges.
- Denial of service against the local sidecar or Qdrant (both are single-user, local services).
- Vulnerabilities in upstream dependencies (Qdrant, Ollama, sentence-transformers) — report those upstream; we will pin or patch defensively on our side.

## Security hardening notes

See the audit commentary in recent commits and the inline documentation in
`src/prompt-injection.ts`, `src/config.ts` (`assertSecureUrl`), and
`tools/install/setup-qdrant.sh` for a description of the current security
controls and their design rationale.
