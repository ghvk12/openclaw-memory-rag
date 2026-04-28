<!--
Thanks for contributing! Fill in the sections below. Delete any that don't apply.
For trivial fixes (typos, comments) you can shorten this to one line in Summary.
-->

## Summary

<!-- One or two sentences: what does this PR do, and why? -->

## Linked issue

<!-- e.g., Closes #42, Fixes #17. Skip if there's no tracking issue. -->

## How was this tested?

<!--
Describe the manual testing you did. Examples:
  - "Ran ./tools/install/setup-qdrant.sh and verified `memrag status` reports qdrant=ok"
  - "Stored 10 messages, queried with `memrag search`, confirmed all 10 returned"
  - "Re-ran `npm run build && node ./.tmp-verify/security-tests.mjs` — all PASS"
CI runs `plugin-inspector ci` automatically.
-->

## Risk / scope

<!--
- What could this break? (e.g., changes default config, requires a Qdrant rebuild, modifies WAL format)
- Is it backward-compatible with existing installs?
- Any rollback plan if it goes wrong in production?
-->

## Checklist

- [ ] `npm run build` passes locally
- [ ] `npm run plugin:check` (plugin-inspector) passes locally — CI also runs this
- [ ] No secrets, api-keys, or personal phone numbers committed
- [ ] README / inline JSDoc updated for any user-visible behavior change
- [ ] Backward-compatible with the existing `~/.openclaw/openclaw.json` schema (or migration documented)
- [ ] If touching the prompt-injection or HTTPS-enforcement code paths, the security-test suite still passes
