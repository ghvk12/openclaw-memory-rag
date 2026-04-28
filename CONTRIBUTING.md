# Contributing to openclaw-memory-rag

Thanks for considering a contribution. This is a small, opinionated plugin and the bar for changes is high — but bug fixes, doc improvements, and well-scoped features are all welcome.

## Where to start

- **Bug reports / feature requests** → [GitHub Issues](https://github.com/ghvk12/openclaw-memory-rag/issues) using the templates.
- **Questions / how-to / "is this supposed to work?"** → [GitHub Discussions](https://github.com/ghvk12/openclaw-memory-rag/discussions). Don't open issues for these — they clutter the bug tracker.
- **Security vulnerabilities** → **Do NOT open a public issue.** See [`SECURITY.md`](SECURITY.md) for the private disclosure channel.

## Development setup

```bash
# 1. Fork on GitHub, then clone your fork
git clone git@github.com:<your-username>/openclaw-memory-rag.git
cd openclaw-memory-rag

# 2. Install JS deps (Node 22.14+ required, 24 recommended)
npm install

# 3. Set up Qdrant + the reranker sidecar (one-shot)
./tools/install/setup-qdrant.sh
cd tools/reranker-sidecar && ./setup.sh && cd ../..

# 4. Build + run the inspector (offline compat check)
npm run build
npm run plugin:check    # same thing CI runs

# 5. Optional: link into your local OpenClaw install for end-to-end testing
openclaw plugins install .
openclaw plugins enable memory-rag
```

## What to send

Small, focused PRs. One PR per logical change. If your change touches multiple subsystems (capture path AND retrieval path AND config schema), split it up.

Before opening a PR, please:

1. `npm run build` — must compile without errors
2. `npm run plugin:check` — must pass (CI runs this and will fail the PR otherwise)
3. Manually verify the change with `openclaw memrag status` / `memrag search` / however the affected path is reached
4. If you touched the prompt-injection filter, the HTTPS-enforcement code, the WAL format, or the Qdrant collection schema, **call it out explicitly in the PR description** — those areas need extra review

## Coding style

- **TypeScript strict mode** is on. No `any` unless there's a comment explaining why.
- **JSDoc on every exported symbol.** Internal helpers can skip it.
- **No `console.log`** in committed code — use the `logger` parameter (Winston-style) so output gets routed properly through the gateway.
- Prefer **named exports** over default exports.
- Keep functions short. If a function exceeds ~50 lines, split it.
- **No comments that just narrate what the code does.** Comments should explain *why* a non-obvious choice was made (trade-offs, references to upstream issues, etc.).

## Commit style

Imperative mood, one-line subject under 72 chars, optional body explaining the why. Examples from the existing log:

```
Strengthen prompt-injection filter; bump starlette CVEs; add Qdrant install helper
Add TEI reranker backend + Apple Silicon Python sidecar
```

If your change is part of a larger sequence, put the broader context in the body.

## PR review process

- CI (`plugin-inspector`) must pass.
- The maintainer will review and either merge, request changes, or explain why it's a no.
- For security-sensitive changes (anything in `src/prompt-injection.ts`, `src/config.ts`'s `assertSecureUrl`, `src/wal.ts`, or the sidecar) the bar is "convince me with a test that proves the new behavior is correct AND doesn't regress the existing security tests".

## What we won't merge

- Changes that hardcode a specific phone number, JID, or other PII.
- New dependencies without a clear justification (each new dep is a future CVE).
- Use of `eval`, `Function()`, or other dynamic-code paths.
- Changes that downgrade `assertSecureUrl` to allow plaintext `http://` against non-loopback hosts.
- Changes that add a network listener bound to anything other than `127.0.0.1` by default.
- "Cleanup" PRs that touch dozens of files for no functional reason.

## License

By contributing, you agree your contributions are licensed under the same MIT license as the rest of the repo.
