# HTTP inference APIs — Anthropic / OpenAI / OpenRouter

brain can act as a **backend inference server for external agents** (Claude Code, the
OpenAI/OpenRouter SDKs) by speaking three provider dialects over HTTP, each on its own
localhost port behind its own key, all dispatching to the ONE shared
`residency::Executor` (`crates/apiserve`). Sibling of the D-Bus surface (`crates/dbus`)
and the JSONL server (`crates/server`) — same models, same scheduling/residency/
batching, different wire protocol.

## Running

```bash
brain serve --dbus --openai[:PORT] --anthropic[:PORT] --openrouter[:PORT] \
            [--models-dir DIR] [--api-keys-out FILE]
```

- Each selected surface binds `127.0.0.1` (localhost only) on its default port —
  **Anthropic 8787 / OpenAI 8788 / OpenRouter 8789** — or the `:PORT` you pass.
- **Access control is always on.** A fresh per-provider API key is generated at startup
  and printed as `APIKEY <provider> <key>` (stderr); `--api-keys-out FILE` also writes
  `{provider: key}` JSON (mode 0600). Anthropic reads `x-api-key`; OpenAI/OpenRouter read
  `Authorization: Bearer`. No/blank/wrong key → 401 on every route (incl. the 404
  fallback — no route enumeration).
- D-Bus and the HTTP surfaces share one executor; D-Bus runs on its own thread.

Example: [`examples/api/claude-with-brain.sh`](../../../examples/api/claude-with-brain.sh)
launches a local qwen3 Anthropic surface and points Claude Code at it.

## Endpoints (the implemented subset)

| Endpoint | Anthropic | OpenAI | OpenRouter |
|---|---|---|---|
| `GET /models`, `/models/{id}` | ✓ | ✓ | ✓ (rich card) |
| chat (non-stream + SSE) | `POST /v1/messages` | `POST /chat/completions` | `POST /chat/completions` |
| token count | `POST /v1/messages/count_tokens` | — | — |
| embeddings | — | `POST /embeddings` | `POST /embeddings` |
| image generation | — | `POST /images/generations` | `POST /images/generations` |

- **Streaming** (`stream:true`): Anthropic emits `message_start → content_block_start →
  content_block_delta* → content_block_stop → message_delta → message_stop`; OpenAI/
  OpenRouter emit `chat.completion.chunk`s ending in `data: [DONE]`. Client disconnect
  cancels the running job (frees the lane).
- **Admission / backpressure:** if a request can't start on a lane within
  `BRAIN_ADMIT_DEADLINE_MS` (default 10 s) → **429** (`Retry-After`); the edge concurrency
  limit sheds overflow as **503**.
- **OpenRouter** reuses the OpenAI handlers, strips a `provider/` model prefix
  (`anything/qwen3-4b` → `qwen3-4b`), honors a `models[]` fallback list, tolerates its
  extra fields, and adds `native_finish_reason`.
- Unimplemented OpenAI surfaces (files/fine-tuning/responses/batch/…) → 501/404.

## Which models each provider exposes

`/models` per provider lists only the loaded models whose **capability fits** that
provider (derived from each model's `capability::Manifest` action shape,
`catalog::api_caps`): OpenAI/OpenRouter = chat ∪ embeddings ∪ image-gen; Anthropic =
chat. A model that can't satisfy an endpoint → `model_not_found` (404).

Models come from the **global model directory** (`--models-dir` / `BRAIN_MODELS_DIR`,
default `$XDG_DATA_HOME/brain/models`), scanned for `*.safetensors` and `*.gguf` — each
file a distinct catalog entry keyed by its model-card id (a base and a finetune are two
entries) — plus the env-gated residents (`BRAIN_QWEN_WEIGHTS`, `BRAIN_GLM_WEIGHTS`, …)
and, for testing, the `BRAIN_MOCK` model.

Import a HuggingFace checkpoint into brain's format:
`brain qwen import --hf <hf_dir> --out qwen3.safetensors`.

## Model card

brain's safetensors containers carry a **ModelCard** in the `__metadata__`
(`brain.card`) — id, family, architecture, variant_of, capabilities, context_length,
param_count, license, … (`crates/checkpoint/src/st.rs`). GGUF cards are synthesized from
the KV store. The card drives `/models` and capability filtering; `checkpoint::st::
read_card` reads it without loading tensors.

## Specs & staying in sync

The three providers' upstream OpenAPI specs are vendored (cached) under
`crates/apiserve/tests/specs/` — the **single source of truth** the conformance tests
validate against. There is no separately hand-maintained brain spec. Refresh from
upstream and see drift with the **`/api-sync`** command (`.claude/commands/api-sync.md`,
`scripts/api/api_sync.py`).

## Testing & security

- **`make test/e2e/api-conformance`** — a 24-test socket-level harness driving a real
  `brain serve` with the deterministic `BRAIN_MOCK` model (no real model, no claude):
  every endpoint × every provider (models / chat non-stream + SSE / embeddings / images /
  count_tokens) **plus the security requirements** (413 body limit, no route enumeration,
  no key leak in any body, input-bound 400s, error hygiene, admission 429).
- **`make test/e2e/claude-code`** — drives the real `claude` CLI against the mock (our
  key only, local-only routing, hard timeouts).
- **`crates/apiserve/tests/api.rs`** — in-process router tests validated against the
  vendored schemas via `jsonschema`.
- **Security is mandatory and continuous:** any change to `crates/apiserve` OR
  `crates/dbus` requires a full audit against **`docs/api-security-audit.md`** (run the
  `security-review` skill) AND the matching automated tests above. The surface stays
  watertight — see the invariant in `AGENTS.md`.
