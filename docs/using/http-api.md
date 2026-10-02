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

Example: [`samples/shell/api/claude-with-brain/claude-with-brain.sh`](../../samples/shell/api/claude-with-brain/claude-with-brain.sh)
launches a local qwen3 Anthropic surface and points Claude Code at it.

## Endpoints (the implemented subset)

| Endpoint | Anthropic | OpenAI | OpenRouter |
|---|---|---|---|
| `GET /models`, `/models/{id}` | ✓ | ✓ | ✓ (rich card) |
| chat (non-stream + SSE) | `POST /v1/messages` | `POST /chat/completions` | `POST /chat/completions` |
| raw completion + fill-in-the-middle (non-stream + SSE) | - | `POST /completions` | - |
| token count | `POST /v1/messages/count_tokens` | - | - |
| embeddings | - | `POST /embeddings` | `POST /embeddings` |
| image generation | - | `POST /images/generations` | `POST /images/generations` |

- **Streaming** (`stream:true`): Anthropic emits `message_start`, then per content
  block (the thinking block, when the model reasons, then the text block)
  `content_block_start → content_block_delta* → content_block_stop`, then
  `message_delta → message_stop`; OpenAI/OpenRouter emit `chat.completion.chunk`s
  (`text_completion` objects on `/completions`) ending in `data: [DONE]`. Client disconnect
  cancels the running job (frees the lane).
- **Reasoning.** A reasoning model's thinking is kept apart from its answer:
  `message.reasoning_content` (and `delta.reasoning_content` when streaming) on the
  OpenAI surfaces, and a `thinking` content block ahead of the `text` block on the
  Anthropic surface, streamed as `thinking_delta`s and closed by a
  `signature_delta`. brain signs nothing, so `signature` is an empty string; a
  thinking block sent back in the history reaches the model as that turn's
  reasoning. Anthropic's `thinking` config turns reasoning on
  (`{"type": "enabled", "budget_tokens": N}`, with Anthropic's own bound
  1024 <= N < `max_tokens`) or off (`{"type": "disabled"}`). The budget is a
  target, as Anthropic defines it: reasoning counts against `max_tokens`, and
  that is what bounds it.
- **Completions** (`POST /v1/completions`): the `prompt` is continued as it is,
  with no chat template, and the `text` is the raw generation. `prompt` is one
  string (or an array holding one); several prompts and token-id prompts are a 400.
  `suffix` makes it fill-in-the-middle: the prompt is the code before the
  insertion point, the suffix the code after it, and the generation what goes
  between, framed with the checkpoint's own FIM tokens (DeepSeek-Coder's
  `<｜fim▁begin｜>`/`<｜fim▁hole｜>`/`<｜fim▁end｜>`, or the Qwen vocabulary's
  `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`). A model whose vocabulary
  has neither set answers `suffix` with a 400. `echo`, `best_of` > 1 and
  `logprobs` are refused by name. Streamed frames are `text_completion` objects;
  every frame before the last has `finish_reason: null`.
- **Token counts.** `usage` on a non-streaming reply is the served model's own
  count: `input_tokens`/`prompt_tokens` is the tokenizer's count of the fully
  rendered prompt (chat template and tool schemas included), and the output
  count is the tokens the model actually generated. Two places are NOT that,
  and a client sizing requests or billing on them should know it:
  `POST /v1/messages/count_tokens` answers with a heuristic (content
  characters / 4, no tokenizer), and the `input_tokens` in an Anthropic
  stream's `message_start` is the same heuristic, because the real count is
  not known until the model has rendered and encoded the prompt. The
  non-streaming reply and the OpenAI-shaped `usage` chunk carry the real
  numbers.
- **Sampling parameters.** `temperature` and `top_p` default to 1.0, as in both
  upstream APIs. `top_k` (an integer in 0..=1000) and `seed` are passed on only
  when the request sets them; otherwise the model applies its own defaults, and an
  unseeded request is sampled with a fresh random seed. OpenAI parameters brain
  cannot honour are refused with 400 `invalid_request_error` naming the
  parameter rather than ignored: `n` > 1, a non-zero `presence_penalty` or
  `frequency_penalty`, a non-empty `logit_bias`, `logprobs: true`, a non-zero
  `top_logprobs`, and a `response_format` other than `text`. Their neutral values
  are accepted.
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

## Every model class: capabilities, run and jobs

The dialect routes above exist for what a chat, embeddings or image client already
speaks. Everything else brain serves (speech in and out, vision tasks, detection,
segmentation, music, video, 3D) is reached through three dialect-neutral routes,
present on every surface and behind the same key, hooks and admission:

- `GET /v1/capabilities` lists every model's every action with its contract: params
  (type, default, range, allowed values), blob inputs and blob outputs. Clients are
  generated from it.
- `POST /v1/run` runs one action and answers with its outputs and blobs.
- `POST /v1/jobs` starts one in the background, for work too long to hold a request
  open. `GET /v1/jobs/{id}` reports `running` (with `progress.step` / `progress.total`),
  `succeeded`, `failed` or `cancelled`; `GET /v1/jobs/{id}/result` returns the result
  once it exists (409 while it is running); `DELETE /v1/jobs/{id}` cancels it, and the
  running action sees the cancellation.

A call is `{"model", "action", "params", "blobs": {name: {"media", "data", "meta"}}}`
with `data` base64 and `media` one of `image`, `mask`, `audio`, `video`, `text`,
`bytes`. A result is `{"outputs", "blobs"}` in the same blob shape. The call is
checked against the action's own spec first (unknown params, wrong types, values
out of range, a missing or mistyped blob are all a 400), and the request body is
bounded at 64 MiB.

A job belongs to the caller that started it: with an `Authenticator` that scopes
callers, another caller gets 404 for an id that exists. Jobs live in memory, at most
32 per caller and 1024 per surface, finished results are forgotten after an hour or
when more than 1 GiB is retained, and a restart forgets them all.

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

## Embedding the surfaces in an application

`brain serve` gives every surface one static key and serves everything it is
asked. An application that links the `apiserve` crate can replace both, with two
traits on `AppState`:

- `Authenticator` (`AppState::with_authenticator`) decides who a request is from
  and returns a `Principal`, an opaque value the application downcasts later. The
  surface's static key stops opening anything once it is replaced. A refusal is an
  `ApiError`, so it keeps the dialect's shape; besides `Unauthorized` there are
  `Forbidden` (403), `PaymentRequired` (402, not retryable) and `RateLimited` (429
  with `Retry-After`, distinct from `Overloaded`, which means the server is full).
- `RequestHooks` (`AppState::with_hooks`) gets a say before each call and a record
  after it. `begin` sees the caller, the model, the action and the invocation, and
  may refuse; what it returns is a `Ticket`, and `Ticket::finish` settles it with
  the outcome (token counts, output blobs), a failure, or `Refused` when the call
  never ran. A ticket is settled exactly once on every path out, including a
  streamed answer whose client disconnected. A settlement that returns an error
  fails the request: an answer nobody is accountable for is not served.

Both are optional, and a surface without them behaves exactly as before.

## Security

Every route above is key-gated, request bodies are size/depth-bounded, and servers
default to localhost-only. See [`docs/using/security.md`](security.md) for the full
"know before you expose this" rundown.
