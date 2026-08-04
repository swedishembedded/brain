---
description: Refresh vendored upstream OpenAPI specs and reconcile brain's API surface to any provider drift
---

Keep brain's HTTP API surface in sync with the upstream providers (Anthropic / OpenAI /
OpenRouter), maintaining **at most two sources of truth**: brain's code, and the
*vendored upstream* specs under `crates/apiserve/tests/specs/` (a cached copy of what
the providers publish). The jsonschema conformance tests validate brain against those
vendored specs — there is no separate hand-maintained brain spec.

Do this:

1. **See the drift** (does not write):
   `python3 scripts/api/api_sync.py --check`
   It fetches each provider's current upstream OpenAPI doc and prints added/removed/
   changed `paths` and `components/schemas` vs the vendored copies.

2. **If there is drift**, review it and decide what matters for the endpoints brain
   actually implements (chat/messages, embeddings, images, models). Then update the
   vendored specs:
   `python3 scripts/api/api_sync.py`  (optionally `--provider openai`)

3. **Re-run conformance**: `cargo test -p brain-apiserve`. Failures show exactly where
   brain's accepted requests / emitted responses no longer match the refreshed spec.

4. **Reconcile the code** (`crates/apiserve/src/{anthropic,openai,openrouter,models,
   catalog,error}.rs`): add/adjust request fields brain now accepts, response fields it
   emits, new/removed endpoints, and error shapes — only as far as brain's supported
   capabilities allow (a field for a capability brain doesn't have is accepted-and-
   ignored or 400/501, documented, never silently wrong). Re-green the tests.

5. **Security audit** (mandatory — new accepted input is a surface change): run the
   `security-review` skill over `crates/apiserve` (+ `crates/dbus`) against
   `docs/api-security-audit.md` and fix findings.

6. Commit the vendored-spec update + code reconciliation together, noting what upstream
   changed.

Report: what drifted per provider, what brain changed to match (or deliberately did
not, and why), and the test + audit results.
