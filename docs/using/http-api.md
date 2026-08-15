# HTTP inference APIs - Anthropic / OpenAI / OpenRouter

brain can act as a **backend inference server for external agents** (Claude Code, the
OpenAI/OpenRouter SDKs) by speaking three provider dialects over HTTP, each on its own
localhost port behind its own key, all dispatching to the same shared scheduler. It's
a sibling of the D-Bus surface ([`docs/using/dbus-api.md`](dbus-api.md)) - same models,
same scheduling/residency/batching, different wire protocol.

## Running

```bash
brain serve --dbus --openai[:PORT] --anthropic[:PORT] --openrouter[:PORT] \
            [--models-dir DIR] [--api-keys-out FILE]
```

- Each selected surface binds `127.0.0.1` (localhost only) on its default port -
  **Anthropic 8787 / OpenAI 8788 / OpenRouter 8789** - or the `:PORT` you pass.
- **Access control is always on.** A fresh per-provider API key is generated at startup
  and printed as `APIKEY <provider> <key>` (stderr); `--api-keys-out FILE` also writes
  `{provider: key}` JSON (mode 0600). Anthropic reads `x-api-key`; OpenAI/OpenRouter read
  `Authorization: Bearer`. No/blank/wrong key → 401 on every route (incl. the 404
  fallback - no route enumeration).
- D-Bus and the HTTP surfaces share one scheduler; D-Bus runs on its own thread.

Example: [`examples/api/claude-with-brain.sh`](../../examples/api/claude-with-brain.sh)
launches a local qwen3 Anthropic surface and points Claude Code at it.

## Endpoints (the implemented subset)

| Endpoint | Anthropic | OpenAI | OpenRouter |
|---|---|---|---|
| `GET /models`, `/models/{id}` | ✓ | ✓ | ✓ (rich card) |
| chat (non-stream + SSE) | `POST /v1/messages` | `POST /chat/completions` | `POST /chat/completions` |
| token count | `POST /v1/messages/count_tokens` | - | - |
| embeddings | - | `POST /embeddings` | `POST /embeddings` |
| image generation | - | `POST /images/generations` | `POST /images/generations` |

- **Streaming** (`stream:true`): Anthropic emits `message_start → content_block_start →
  content_block_delta* → content_block_stop → message_delta → message_stop`; OpenAI/
  OpenRouter emit `chat.completion.chunk`s ending in `data: [DONE]`. Client disconnect
  cancels the running job (frees the lane).
- **Admission / backpressure:** a request that can't start on a lane within
  `BRAIN_ADMIT_DEADLINE_MS` gets **429** (`Retry-After`) - unless the request's own
  model is still cold-building (its first-ever activation, which can take well over a
  minute for a real model), in which case it gets the longer
  `BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS` grace window instead - a legitimate cold start is
  not overload. Genuine overload still sheds as **503**. See
  [`docs/using/serving.md`](serving.md#admission-and-backpressure) for the general
  admission story and [`docs/using/configuration.md`](configuration.md#serving--admission)
  for both variables and their defaults.
- **OpenRouter** reuses the OpenAI handlers, strips a `provider/` model prefix
  (`anything/qwen3-4b` → `qwen3-4b`), honors a `models[]` fallback list, tolerates its
  extra fields, and adds `native_finish_reason`.
- Unimplemented OpenAI surfaces (files/fine-tuning/responses/batch/…) → 501/404.

## Which models each provider exposes

`/models` per provider lists only the loaded models whose capability fits that
provider: OpenAI/OpenRouter = chat ∪ embeddings ∪ image-gen; Anthropic = chat. A model
that can't satisfy an endpoint → `model_not_found` (404).

Models come from the **global model directory** (`--models-dir` / `BRAIN_MODELS_DIR`,
default `$XDG_DATA_HOME/brain/models`), scanned for `*.safetensors` and `*.gguf` - each
file a distinct catalog entry keyed by its model-card id (a base and a finetune are two
entries) - plus whichever models are enabled via their `BRAIN_*_WEIGHTS`-style
environment variables (see [`docs/using/configuration.md`](configuration.md)), and, for
testing, the `BRAIN_MOCK` model.

Import a HuggingFace checkpoint into brain's format:
`brain qwen3 import --hf <hf_dir> --out qwen3.safetensors`.

## Model card

brain's safetensors containers carry a model card in their metadata - id, family,
architecture, variant_of, capabilities, context_length, param_count, license, … GGUF
cards are synthesized from the file's own key/value store. The card drives `/models`
and capability filtering.

## Security

Every route above is key-gated, request bodies are size/depth-bounded, and servers
default to localhost-only. See [`docs/using/security.md`](security.md) for the full
"know before you expose this" rundown.
