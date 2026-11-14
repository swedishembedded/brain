# HTTP inference APIs - Anthropic / OpenAI / OpenRouter

brain can act as a **backend inference server for external agents** (Claude Code, the
OpenAI/OpenRouter SDKs) by speaking three provider dialects over HTTP, each on its own
localhost port behind its own key, all dispatching to the same shared scheduler. It's
a sibling of the D-Bus surface ([`docs/reference/api/dbus.md`](dbus.md)) - same models,
same scheduling/residency/batching, different wire protocol.

## Running

```bash
brain serve --dbus --openai[:PORT] --anthropic[:PORT] --openrouter[:PORT] \
            [--models-dir DIR] [--api-keys-out FILE]
```

- Each selected surface binds `127.0.0.1` (localhost only) on its default port -
  **Anthropic 8787 / OpenAI 8788 / OpenRouter 8789** - or the `:PORT` you pass.
- **Access control is always on.** A fresh per-provider API key is generated at startup
  and printed on stdout as `APIKEY <provider> <key>`; `--api-keys-out FILE` writes
  `{provider: key}` JSON (mode 0600) and the line then reads `APIKEY <provider> written to FILE`
  - the key itself stays out of logs. Anthropic reads `x-api-key`; OpenAI/OpenRouter read
  `Authorization: Bearer`. No/blank/wrong key → 401 on every route (incl. the 404
  fallback - no route enumeration).
- D-Bus and the HTTP surfaces share one scheduler; D-Bus runs on its own thread.

Example: [`samples/shell/api/claude-with-brain/claude-with-brain.sh`](../../../samples/shell/api/claude-with-brain/claude-with-brain.sh)
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
| transcription (speech to text) | - | `POST /audio/transcriptions` | `POST /audio/transcriptions` |
| speech synthesis (text to speech) | - | `POST /audio/speech` | `POST /audio/speech` |
| video (generate and edit) | - | `POST /videos`, `GET /videos/{id}`, `GET /videos/{id}/content`, `GET /videos/models` | same |

- **Chat content parts**: a message's `content` may be a string or a list of
  parts. OpenAI parts are `text`, `refusal`, `image_url` (an inline `data:` URL)
  and `input_audio` (one clip per request, WAV, Ogg/Opus or WebM/Opus);
  Anthropic blocks are `text`, `image` (a `base64` source), `tool_use`,
  `tool_result`, `thinking` and `redacted_thinking`. Every part is read: any
  other type, an external URL, a part without its payload, a second audio
  clip, more images than the model's card states in `max_images_per_request`,
  or a medium the model's `generate` action does not take is a 400 naming it,
  never a reply from the rest of the request.
- **Transcription** (`POST /v1/audio/transcriptions`, also without `/v1`): an
  OpenAI multipart form (`file`, `model`, optional `language` as an ISO-639-1
  code; `response_format` may only be `json`) or OpenRouter's JSON
  (`{"model", "input_audio": {"data": <base64>, "format"}, "language"?}`). The
  recording may be WAV (any rate, mono or stereo), Ogg/Opus or WebM/Opus,
  including the unknown-length WebM a live browser `MediaRecorder` writes; the
<!-- perf-number: the endpoint's own length limit, an API constant -->
  container is read from the bytes, not from `format`. At most 300 s and the
  8 MiB body limit per request. A malformed, unsupported, empty or over-long
  recording is a 400 before any model runs. The answer is
  `{"text", "usage": {"type": "duration", "seconds"}}`, where `seconds` is the
  decoded duration. `language` reaches a model only if its `transcribe` action
  takes one.
- **Speech** (`POST /v1/audio/speech`, also without `/v1`): `{"model", "input", "response_format"?}` and nothing
  else. `input` is 1 to 4096 characters; `response_format` is `wav` (default, 16-bit mono) or `pcm` (raw 16-bit
  signed little-endian mono); the reply is the audio itself, with `Content-Type` `audio/wav` or `audio/pcm` and
  `X-Sample-Rate` (the model's own rate, 24000 for Qwen3-TTS). A compressed format, `voice`, `speed`,
  `instructions`, `stream_format` and any unknown field are a 400 naming it: brain's speech models take the text and
  nothing that steers the voice (Qwen3-TTS's voice is fixed by the operator's configuration). The model is any one
  whose `speak` action takes `text` and returns `audio`, listed as capability `speech`; the call is held to that
  action's own contract before it runs and its length is capped at 1024 codec frames (about a minute and a half).
- **Video** (`/v1/videos`, also without `/v1`), OpenRouter's asynchronous
  video contract. `POST /videos` takes `{"model", "prompt", "duration"?,
  "resolution"?, "aspect_ratio"?, "size"?, "generate_audio"?, "seed"?,
  "frame_images"?, "input_references"?}` and answers `202 {"id",
  "polling_url", "status": "pending"}`. `frame_images` anchors the clip's
  `first_frame` and/or `last_frame` to an image
  (`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."},
  "frame_type": "first_frame"}`); `input_references` are image, video
  (MP4/WebM) or WAV audio content parts the clip draws its subjects, look
  and sound from. Each is sent inline as a `data:` URL; the request goes to
  the model's action that takes them (MiniMax-H3's `fl2va` for frames,
  `ref2va` for references; LTX-2.5's `t2v` for frames). `GET /videos/{id}`
  reports `pending`, `in_progress` (with `progress {step, total}`),
  `completed` (with `unsigned_urls`, `usage {seconds, frames, fps, width,
  height, audio}` and `content [{media_type, bytes, sha256}]`), `failed`
  (`error`), `cancelled` or `expired`. `GET /videos/{id}/content?index=0` is
  the clip as an MP4 (AV1 video, Opus sound), with `Range` support; it needs
  the same key. `GET /videos/models` lists each video model with
  `supported_durations`, `supported_resolutions`, `supported_aspect_ratios`,
  `supported_sizes`, `supported_frame_images`, `generate_audio`, `seed` and,
  as extensions, `supported_reference_types` and its `actions`. Each model maps these fields onto its own frame grid, sizes and
  rate, and a value it cannot render is a 400 naming the field; nothing is
  silently changed beyond snapping the frame count to the model's grid.
  Extensions: an `Idempotency-Key` header returns the job it already started
  for the same key and body (a different body is a 422);
  `DELETE /videos/{id}` cancels a running job or deletes a finished one and
  its file; an edit names one of the model's edit actions in
  `provider.options.brain.action`, with its own params in
  `provider.options.brain.params`, and sends the clip to edit as
  `input_video: {"type": "video_url", "video_url": {"url":
  "data:video/mp4;base64,..."}}` (MP4 or WebM, AV1, read by brain itself:
  a clip in another codec is refused, never handed to ffmpeg), at most 2000
  frames and 64 million pixels across them. A remote URL is refused. Finished clips are kept for an hour, within the server's spool
  budget; a restart forgets every job.
- **Streaming** (`stream:true`): Anthropic emits `message_start`, then per content
  block (the thinking block, when the model reasons, then the text block, then
  one `tool_use` block per tool call)
  `content_block_start → content_block_delta* → content_block_stop`, then
  `message_delta → message_stop`; OpenAI/OpenRouter emit `chat.completion.chunk`s
  (`text_completion` objects on `/completions`) ending in `data: [DONE]`. Client disconnect
  cancels the running job (frees the lane). Every SSE stream emits a comment keepalive
  (`: keepalive`) roughly every 15 s while the model is silent - a long local prefill can <!-- perf-number: keepalive interval constant of the server -->
  be minutes without a real event, and idle-watchdog clients need bytes to stay attached.
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
  `{"type": "adaptive"}` leaves the choice to the model's own default.
- **Tool calling.** A model whose card lists `tools` takes them on every
  surface. On the Anthropic surface a client tool (`name`, `description`,
  `input_schema`) reaches the model as a function tool, an assistant turn's
  `tool_use` blocks as its calls, and a user turn's `tool_result` blocks as the
  results answering them (an `is_error` result is marked as an error). The
  model's calls come back as `tool_use` blocks whose `input` is the JSON object
  it wrote, streamed as `input_json_delta` fragments, with
  `stop_reason: "tool_use"`. `tool_choice` takes `auto`, `any`, `none` and
  `tool`. Refused by name: server tools (a `type` other than `custom`, such as
  hosted web search), `disable_parallel_tool_use: true` (the model may call
  several tools in one turn), and a call whose arguments are not a JSON object.
  A request must fit the model's context, prompt plus `max_tokens`, as on
  Anthropic's own API: Claude Code asks for 32000 output tokens unless
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS` lowers it, which
  [`claude-with-brain.sh`](../../../samples/shell/api/claude-with-brain/claude-with-brain.sh)
  does (`--max-output-tokens`).
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
- **Sampling parameters.** `temperature`, `top_p`, `top_k` (an integer in
  0..=1000) and `seed` are passed on only when the request sets them; otherwise
  the model applies its own defaults - for a checkpoint that declares them
  (`generation_config.json`, or `general.sampling.*` in a GGUF) the declared
  `temperature`, `top_k` and `top_p`, shown as the `generate` action's param
  defaults - and an unseeded request is sampled with a fresh random seed. OpenAI parameters brain
  cannot honour are refused with 400 `invalid_request_error` naming the
  parameter rather than ignored: `n` > 1, a non-empty `logit_bias`, a non-zero
  `top_logprobs`, and a `response_format` other than `text`; on the legacy
  completions endpoint also `logprobs: true`. Their neutral values are accepted.
- **Anti-repetition sampling and the thinking budget.** For the text models that
  serve through the shared scheduler, a request may add (all off unless sent;
  a model without them refuses the field as an unknown parameter):
  `presence_penalty` and `frequency_penalty` (-2..=2, subtracted from the logit
  of tokens already generated, once and per occurrence), `repetition_penalty`
  (vLLM; divides the positive logit of every token in the prompt and the
  generated text), `min_p` (drop tokens below that fraction of the best token's
  probability), and DRY as llama.cpp defines it over the whole context:
  `dry_multiplier`, `dry_base` (default 1.75), `dry_allowed_length` (default 2)
  and `dry_sequence_breakers` (an array of strings, each reduced to its first
  token; default newline, colon, quote, asterisk). Processors run on the
  decoder's whole logits row before temperature, top-k, top-p and min-p, and
  also steer a greedy request (`temperature: 0` then takes the best penalised
  token). Sampling is exact over the full vocabulary: a request whose draw the
  decoder's on-device candidate list cannot answer exactly (`top_p`, `min_p`, a
  penalty, or no `top_k` within the list) is sampled from the whole row, and a
  decoder that cannot read its row back refuses such a request with 400 rather
  than draw from a truncated distribution. `thinking_budget` (also accepted as
  `thinking_token_budget` and `max_thinking_tokens`; Anthropic's
  `thinking.budget_tokens` sets it too) limits the tokens inside a think block:
  when it is spent the closing tag and the blank line after it are forced and
  the model answers. A reply reports the tokens spent thinking in
  `usage.completion_tokens_details.reasoning_tokens`.
- **Prompt cache.** `usage.prompt_tokens_details.cached_tokens` is how many of
  `prompt_tokens` the model's prefix cache served instead of computing (0 for a
  cold prompt); a model that does not report it omits the field.
- **Sampled tokens (chat).** `logprobs: true` makes the reply carry
  `choices[0].logprobs.content[]` (`{token, logprob, bytes, top_logprobs}`; the
  server relays ids, so `token` is `token_id:<id>`, `top_logprobs` is empty) and
  `choices[0].token_ids`; `return_token_ids: true` (the vLLM extension name) adds
  the completion ids and the top-level `prompt_token_ids` the model read. A
  streamed reply carries them on the terminal chunk. The `logprob` of a token is
  its log-probability under the distribution it was decoded from (the
  temperature-scaled, top-k and top-p truncated softmax), `0.0` for greedy
  decoding where the choice is deterministic; it is not the untruncated model
  distribution. A request that sets neither flag pays for neither.
- **Embeddings extensions.** `POST /v1/embeddings` also takes `instruction` (a non-empty string rendered as `Instruct: {instruction}\nQuery: {text}`, the Qwen3-Embedding query convention; omit for a passage) and `max_tokens` (truncate each input to this many tokens; 0 or absent: the model's own limit). The call is held to the model's own `embed` contract: a malformed value, a value past the model's bound, or a field the model does not take (an encoder without a query convention takes no `instruction`) is a 400 naming it.
- **Policy trace (chat, for training).** `policy_trace: true` (a boolean; anything
  else is a 400) makes the reply carry a top-level `policy_trace` object (a
  streamed reply: on the terminal chunk) holding `prompt_token_ids`,
  `generation_token_ids` (the stop token that ended the generation included, so a
  trainer can train the stop) and `generation_log_probs`: for each generated token,
  `log softmax(logits / log_prob_temperature)` over the WHOLE vocabulary, with no
  top-k, top-p or min-p, however the token was drawn. `log_prob_temperature` is the
  request's temperature (1.0 for a greedy request). The log-probabilities are
  scored after the generation by the serving model itself, as one
  teacher-forced pass under the adapter the request ran with (the same weights,
  quantisation and KV precision that served it), so they are what a trainer's
  teacher-forced forward computes for those tokens, up to the serve/train numeric
  gap a numerics gate measures. The object also names the `policy` that generated
  it (`base_digest`, and `adapter: {id, digest, version}` or null), `finish_reason`, and
  `truncated` (the token budget ended the generation, which a trainer masks).
  A streamed reply carries the trace, the `brain` object (served identity) and the sampled-token fields together on the terminal
  chunk (the one with `finish_reason`); the usage chunk after it carries none of them.
  The request is refused (400, `invalid-request`) when it sets penalties, DRY,
  no-repeat bans or a thinking budget, which reshape the distribution the tokens
  are drawn from, and fails with `unsupported-capability` on a model that cannot
  score its own generation. The cost is one extra prefill of prompt plus
  generation on the serving thread. Train rollouts at temperature 1.0 without
  top-k or top-p so the sampler's distribution and the trace's coincide; `logprobs`
  (above) is the sampler's own, truncated figure and is a different quantity.
- **Scoring given text (`score`).** A served Qwen3 or Qwen3.8 chat model (and each adapter id riding on it) has a
  `score` action, called through `POST /v1/run` with `model` the base or adapter id and `blobs.records` a
  `generic-messages-v2` JSONL text blob. Each message marked `"train": true` is an answer, scored teacher-forced
  under the model's full distribution at temperature 1, as the model is asked for it: the prompt is what a `generate`
  call renders for the messages before it (client text cannot spell a control token), the answer's tokens are its
  content plus the end-of-turn token that closes it (`include_stop`, default true). Params: `per_token` (also return
  each answer's `token_ids` and `token_logprobs`), `include_stop`, `enable_thinking` (render the prompts as a request
  that allows reasoning would). The reply's `outputs` hold `records` (each with its `answers`: `message`, `tokens`,
  `logprob`, `mean_logprob`), the totals `tokens`, `logprob`, `mean_logprob`, the `tokenizer_digest`,
  `chat_template_digest` and `base_digest`, and the `adapter` that scored. At most 4096 records of at most 64
  supervised answers each, and 8192 answers in a call; a record with tool calls, tool results, tool schemas or reasoning content, one with no
  supervised message, a malformed one, or one longer than the model holds fails the whole call by record number
  before any pass (400). A model that cannot score a continuation does not declare the action. Scoring is one
  prefill per answer on the serving instance, between its decode steps; it does not read or write the prefix cache.
- **Seed and served identity.** An OpenAI or OpenRouter chat or completions
  reply (and the terminal chunk of a stream) carries a non-standard
  `brain: {"seed": <int>, "adapter": {"id", "digest", "version"}, "base_digest", "backend",
  "engine"}` object. `seed` is the
  seed the decode used, drawn fresh for an unseeded request, so any reply can
  be reproduced by sending it back; `adapter` appears only when an adapter
  answered and names the release that did, the same triple its model card carries. `version` is the policy version of that release under its id: 1 for the first release the server offered, +1 each time a release with different weights replaces it, never repeated while the process runs. A sequence runs on the release it started on and never spans a refit, so every token of a reply carries the one version; a trainer that records it with the tokens can mask a sample whose version is too far behind the weights it trains.
  `base_digest` is the `sha256:<hex>` of the base checkpoint (hashed in the
  background after load, so absent on the first replies of a large model),
  `backend` the compute backend that ran the decode (`cuda`, `cpu`, ...) and
  `engine` the build (`brain <version>`). `policy_version` is the version of the
  adapter release that answered (1 for the first release under its id, one more
  for each release with different weights; 0 for the base). `logprob_kind` says
  what the reply's log-probabilities are: `policy` (the full distribution of a
  `policy_trace`) or `sampler` (the truncated figure of `logprobs`); it is
  present when either was asked. A policy trace also names the
  `tokenizer_digest` (`sha256:<hex>` of the tokenizer file, when the model has
  one) and the `chat_template_digest` (`sha256:<hex>` of the token ids the
  server renders for a fixed probe conversation, so it changes when the template
  or the tokenizer does), here and under `policy_trace.policy`. Each key is
  absent when the model does
  not report it. Over D-Bus the same values are keys of the `result` JSON of `Run`.
- **Admission / backpressure:** a request that can't start on a lane within
  `--admit-deadline-ms` gets **429** (`Retry-After`) - unless the request's own
  model is still cold-building (its first-ever activation, which can take well over a
  minute for a real model), in which case it gets the longer
  `--cold-build-admit-deadline-ms` grace window instead - a legitimate cold start is
  not overload. Genuine overload still sheds as **503**. See
  [`docs/guides/serving/serving.md`](../../guides/serving/serving.md#admission-and-backpressure) for the general
  admission story and both flags' defaults.
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
- `GET /v1/devices` lists the devices the server schedules work on, as `{"object": "list", "host": {...}, "data":
  [...]}`. Each entry has `id` (`cpu`, `gpu<i>`, `npu<i>`: the names `--device` takes), `kind` and `memory`
  (`total_bytes`, `reserved_bytes` held back by the operator, `used_bytes` held by resident models and runs,
  `free_bytes`), and, for a card, `name`, `vendor_id`, `class` (`discrete`, `integrated`) and `backends` (which of
  `cuda`, `vulkan`, `wgpu` can drive it). `host` has the backend in use, the core count and the card and NPU counts.
  It names no path, bus address or UUID of the machine. Per-card compute capability is not reported: it is known
  only by opening the card, which the serving process does once, for the cards it serves on.
- `POST /v1/run` runs one action and answers with its outputs and blobs.
- `POST /v1/jobs` starts one in the background, for work too long to hold a request
  open. `GET /v1/jobs/{id}` reports `running` (with `progress.step` / `progress.total`),
  `succeeded`, `failed` or `cancelled`; `GET /v1/jobs/{id}/result` returns the result
  once it exists (409 while it is running); `DELETE /v1/jobs/{id}` cancels it, and the
  running action sees the cancellation. `GET /v1/jobs/{id}/events` follows it as
  server-sent events: a `progress` event per step tick, a `frame` event per structured
  frame the action reports (a training step's loss and statistics), and one `done`
  event with the final status. The last 4096 frames are kept, and `Last-Event-ID`
  resumes after the last event received; at most 8 streams follow one job.

A call is `{"model", "action", "params", "blobs": {name: {"media", "data", "meta"}}}`
(no other field is accepted) with `data` base64 and `media` one of `image`, `mask`, `audio`, `video`, `text`,
`bytes`. A result is `{"outputs", "blobs"}` in the same blob shape. The call is
checked against the action's own spec first (unknown params, wrong types, values
out of range, a missing or mistyped blob are all a 400), and the request body is
bounded at 64 MiB.

A param the host resolves for itself, such as a weights path, is never offered by
`/v1/capabilities` and is refused (400, unknown param) if a call names it: a caller
cannot point a model at a file on the machine that serves it. The host supplies its
own answer.

A job belongs to the caller that started it: with an `Authenticator` that scopes
callers, another caller gets 404 for an id that exists. Jobs live in memory, at most
32 per caller and 1024 per surface, finished results are forgotten after an hour or
when more than 1 GiB is retained, and a restart forgets them all.

## Typed decisions: `/v1/systemone`

`POST /v1/systemone` asks a decision model typed questions (`choice`, `score`,
`noul`) about one state, in the System One wire shape, and is a thin layer over
the `decide` action every decision model serves. The request is
`{"model", "state", "questions": {name: {"type", "instructions", "criteria"}}}`;
the response is `{"model", "answers": {name: ...}, "usage": {"output_tokens": 0},
"brain": {...}}`, with probabilities keyed in the caller's option order.
`"model"` is a decision model id or `"auto"` (the default): the configured
model, else a validated Laya model, else a validated `decide` model, and a
`not_installed` error with the reason when none qualifies. A request outside the
published limits (255 options, 2 to 10 levels) is a 422 naming the question.
See [Decision models](../models/decision.md) for the full contract.

Any model that classes its own failure reaches a client as that class, in the
error `code`: `not_installed` (404), `failed_to_load` (500),
`unsupported_capability` (422), `timeout` (504), `invalid_request` (422) or
`backend_failure` (500) or `unavailable` (503, with `Retry-After`).

A language model whose card runs out of memory does not fail the requests it
admitted: the sequences of the failing step go back to the queue and run again
once the card has shown it can hold them, so they complete with the same tokens.
Only a request that fails again while it is alone on the card, after the prefix
cache has been given back, is answered `unavailable` (503): retry later, or with
a shorter prompt.

A request whose job the server cancels while draining at shutdown is answered
`unavailable` (503) with the message `the request was cancelled`: retry it
against another server. Over D-Bus a cancelled call fails with the message
`cancelled`, whoever cancelled it.

## Which models each provider exposes

`/models` per provider lists only the loaded models whose capability fits that
provider: OpenAI/OpenRouter = chat ∪ embeddings ∪ image-gen ∪ transcription;
Anthropic = chat. A model
that can't satisfy an endpoint → `model_not_found` (404).

Models come from the **global model directory** (`--models-dir` / `BRAIN_MODELS_DIR`,
default `$XDG_DATA_HOME/brain/models`), scanned for `*.safetensors` and `*.gguf` - each
file a distinct catalog entry keyed by its model-card id (a base and a finetune are two
entries) - plus whichever models are enabled via their `BRAIN_*_WEIGHTS`-style
environment variables (see [`docs/reference/configuration.md`](../configuration.md)), and, for
testing, the mock model (`brain serve --mock`).

Import a HuggingFace checkpoint into brain's format:
`brain qwen3 import --source <hf_dir> --dest qwen3.safetensors`.

### Discovery fields on a model entry

On the OpenAI and OpenRouter dialects every entry of `GET /models` and
`GET /models/{id}` may carry these optional fields, in addition to the dialect's own.
A field is absent when its value is unknown; it is never `null` or `0` as a stand-in.
The Anthropic entry is unchanged.

| Field | Meaning |
|---|---|
| `capabilities` | Names derived from the model's manifest: `chat`, `tools` (chat whose `generate` accepts `tools`), `vision` (chat whose `generate` reads `image` inputs; image parts sent to any other model are refused with 400), `embeddings`, `images`, `transcription` (a `transcribe` action taking `audio`), `speech` (a `speak` action taking `text` and returning `audio`). On OpenRouter the card's `architecture` modalities follow from the same set, e.g. `audio->text` for a speech model. |
| `max_images_per_request`, `max_image_pixels_per_request` | On a `vision` model: the most images (the `image`, `image1`, ... inputs its `generate` action declares), and pixels over all of them, one request may carry; a request past either is refused with 400. |
| `architecture.input_modalities` | On a chat model: `"text"`, then `"image"` for a `vision` model and `"audio"` for one whose `generate` takes an `audio` clip (the field an agent runtime reads to decide what it may attach). |
| `max_output_tokens` | The most new tokens one request may ask for, when the model's `generate` action declares an upper bound. On OpenRouter it is also `top_provider.max_completion_tokens`. |
| `adapter` | Present on an entry that IS a LoRA adapter: `{ "id": "<adapter card id>", "digest": "sha256:<hex>", "version": <int> }`. The id is the adapter's name and the digest is that of its file; no file path is ever included. A release that replaces an adapter under the same id changes the digest and raises the policy version on the next read. |
| `zero_data_retention` | Always `true`: the server keeps nothing of a request (no prompt, answer or upload is written anywhere or logged). A client that routes by data retention reads it to tell this server from one that does not say. |

### Adapters are models

A LoRA adapter served beside its base (see
[`docs/guides/serving/serving.md`](../../guides/serving/serving.md#serving-lora-adapters)) has an entry of its
own: `id` is the adapter's card id, `<vendor>/<repo>:<owner>:<name>:<tag>`, and
it is selected by putting that id in `model` on any route (chat, completions,
messages, `/v1/run`). Its base's entry is the same as without adapters and has
no `adapter` field. Adapter and base requests are batched together on one copy
of the weights.

An adapter id needs the same API key as any model. An id that is not served -
never loaded, withdrawn after a release, or merely adapter-shaped - is an
unknown model: the same 404 and the same body, and nothing in a request can
make the server load or fetch an adapter. The selection travels in a reserved
parameter the server sets itself; `/v1/run`, `/v1/jobs` and D-Bus `Run` refuse
`__adapter` (and `adapter`) as an unknown param like any other undeclared name.

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

`RequestHooks::listed` decides which models `/models`, `/models/{id}` and
`/v1/capabilities` show (everything, by default), so an application can advertise only
what it sells.

Both are optional, and a surface without them behaves exactly as before.

## Security

Every route above is key-gated, request bodies are size/depth-bounded, and servers
default to localhost-only. See [`docs/guides/serving/security.md`](../../guides/serving/security.md) for the full
"know before you expose this" rundown.
