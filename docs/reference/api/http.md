# HTTP inference APIs - Anthropic / OpenAI / OpenRouter

brain can act as a **backend inference server for external agents** (Claude Code, the
OpenAI/OpenRouter SDKs) by speaking three provider dialects over HTTP, each on its own
localhost port behind its own key, all dispatching to the same shared scheduler. It's
a sibling of the D-Bus surface ([`docs/reference/api/dbus.md`](dbus.md)) - same models,
same scheduling/residency/batching, different wire protocol.

## Running

```bash
brain serve --dbus --openai [ADDR] --anthropic [ADDR] --openrouter [ADDR] \
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
  never a reply from the rest of the request. An image may stand in any turn the
  model's chat template places one, a tool result included (an OpenAI `tool`
  message's `image_url` parts, an Anthropic `tool_result`'s `image` blocks), and
  is read where it stands; one in a turn the template has no place for (a system
  message, or a tool result for a model without tool turns) is a 400 naming the
  message (`messages[i]`).
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
- **Speech** (`POST /v1/audio/speech`, also without `/v1`): `{"model", "input", "response_format"?,
  "input_references"?, "instructions"?, "voice"?, "seed"?, "language"?}`. `input` is 1 to 4096 characters; `response_format` is
  `wav` (default, 16-bit mono) or `pcm` (raw 16-bit signed little-endian mono); the reply is the audio itself, with
  `Content-Type` `audio/wav` or `audio/pcm`, `X-Sample-Rate` (the model's own rate, 24000 for Qwen3-TTS and
  CosyVoice) and `X-Seed` (the seed the take was drawn with; send it back as `seed` for the same take).
  - `input_references` is the voice to imitate: one `input_audio` part (`{"type": "input_audio", "input_audio":
    {"data": "data:audio/wav;base64,..."}}`, bare base64 and a `format` of `wav`, `ogg`, `opus` or `webm` also
    accepted) and, optionally, one `text` part with what the recording says. The recording may be WAV (any rate),
<!-- perf-number: the endpoint's own reference length limit, an API constant -->
    Ogg/Opus or WebM/Opus, at most 30 s; it reaches the model at its own rate (Opus at 48 kHz), never cut to 16 kHz.
    With a transcript, Qwen3-TTS clones in context (timbre, pitch, pace and room); without, from the speaker vector
    alone. CosyVoice needs the transcript.
  - `instructions` (1 to 4096 characters) describes the voice and delivery, and `voice` names a preset voice; both
    need a checkpoint that has them (Qwen3-TTS VoiceDesign and CustomVoice). A custom voice object
    (`{"id": ...}`) is refused: brain keeps no stored voices. A trained Qwen3-TTS checkpoint's named voices are its
    preset voices.
  - `language` (brain's extension, 1 to 64 characters) names the language the input is spoken in, as the model's
    checkpoint names its languages (Qwen3-TTS: a `codec_language_id` key such as `english`, a trained one's new
    languages included, or `auto`); omitted, the model's default. A language the checkpoint does not name is a 400
    listing the ones it does.
  - The request goes to the model's speech action whose declared contract takes everything it carries; anything no
    action of the model takes (`voice` on a model with no presets, a reference on one that does not clone, a missing
    transcript where one is required), a compressed `response_format`, `speed`, `stream_format` and any unknown field
    are a 400 naming it. The model is any one with such an action, listed as capability `speech`; the call is held to
    the action's own contract before it runs and its length is capped at 1024 codec frames (about a minute and a
    half) where the model counts frames.
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
  budget (`brain serve --video-spool-gb`). Jobs outlive a restart: `brain serve`
  records each one beside its clip under `--video-jobs-dir` (default
  `<data root>/video-jobs`), so a finished clip is still served by its id and its
  `Idempotency-Key`, a clip whose file is gone reads `expired`, and a job that was
  running when the server stopped reads `failed` with that reason. Such a job is
  never rendered again on its own; submit it with a new key to render it.
- **Streaming** (`stream:true`): Anthropic emits `message_start`, then per content
  block (the thinking block, when the model reasons, then the text block, then
  one `tool_use` block per tool call)
  `content_block_start → content_block_delta* → content_block_stop`, then
  `message_delta → message_stop`; OpenAI/OpenRouter emit `chat.completion.chunk`s
  (`text_completion` objects on `/completions`) ending in `data: [DONE]`. Client disconnect
  cancels the running job (frees the lane). A stream holds at most 64 queued events: a
  client that reads more slowly than the model generates holds the generation back
  rather than growing server memory, and text deltas waiting on a full queue are merged
  into the newest queued one (up to 1 KiB), so chunks get larger but the concatenated
  text and the order of events are unchanged. A client that reads nothing for 30 s <!-- perf-number: stall deadline default of the server -->
  (`ConnectionLimits::stream_stall`) while the generation waits has it cancelled; its
  stream ends with an `error` event whose code is `timeout`. Every SSE stream emits a comment keepalive
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
  `output_config.effort` maps to the model's `reasoning_effort`: `high` (Anthropic's
  default) and `max` leave the model's default, `medium` and `low` ask for less
  reasoning and are refused, naming `reasoning_effort`, by a model without that
  setting; any other `output_config` member (`format`) is refused by name.
  A model has that setting only where its chat template reads it (Qwen3.8's):
  a Qwen3 model refuses `reasoning_effort` on every surface rather than serve a
  prompt with an instruction its own template never writes, and the prompt it
  is served is its template's rendering of the request, tool schemas in the
  order the request wrote them.
  `context_management` is accepted when it edits nothing (`{"type":
  "clear_thinking_20251015", "keep": "all"}`); an edit that would drop part of the
  prompt is refused by name, since brain edits no context.
- **Tool calling.** A model whose card lists `tools` takes them on every
  surface. On the Anthropic surface a client tool (`name`, `description`,
  `input_schema`) reaches the model as a function tool (one marked
  `defer_loading: true` does not: it waits for a server-side tool search brain does not run), an assistant turn's
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
  non-streaming reply, the Anthropic stream's closing `message_delta` (which
  repeats `input_tokens`) and the OpenAI-shaped `usage` chunk carry the real
  numbers.
- **The request is held to the model's served spec.** Every chat route (OpenAI
  and OpenRouter chat, OpenAI completions, Anthropic messages; streamed or not)
  validates the call it builds against the model's `generate` action as
  `/v1/capabilities` lists it, the same check `/v1/run` and D-Bus `Run` make,
  before anything is queued (a streamed request is refused with a plain 400,
  never an event stream). A setting the request leaves out takes the default the
  action declares - the sampler's and the chat template's alike
  (`enable_thinking`, `template_flavor`, `preserve_thinking` where the model
  declares them; `reasoning_effort` has none of brain's own, so a template that
  reads it applies its own default) - and a field the model's action does not take
  is refused with 400 `unsupported_parameter` naming the field and the model
  (OpenAI's `error.param` is the field), never dropped. A top-level field the
  dialect does not define, or one brain does not read (`functions`,
  `verbosity`, `audio`, OpenRouter's `plugins` and `reasoning`, Anthropic's
  `container` and `mcp_servers`, ...), is refused the same way. These are
  accepted and ignored, each because ignoring it answers the request the client
  sent (`crates/apiserve/src/request_fields.rs` holds the same table):

  | Field | Dialects | Accepted values | Why ignoring it is correct |
  |---|---|---|---|
  | `user` | OpenAI chat, OpenRouter, completions | any | an end-user id for the provider's abuse monitoring; it does not change the reply |
  | `safety_identifier` | OpenAI chat, OpenRouter | any | the successor of `user` |
  | `prompt_cache_key` | OpenAI chat, OpenRouter | any | a hint for routing to a warm prompt cache; brain's prefix cache matches prompts by their tokens |
  | `metadata` | OpenAI chat, OpenRouter, Anthropic | any | stored-completion tags (OpenAI) or an end-user id (Anthropic); brain stores nothing |
  | `store` | OpenAI chat, OpenRouter | `false` | brain keeps no copy; `true` asks it to, and is refused |
  | `service_tier` | OpenAI chat, OpenRouter | `auto`, `default` | brain has one processing tier; `flex` or `priority` is refused |
  | `service_tier` | Anthropic | `auto`, `standard_only` | brain has one capacity tier |
  | `cache_control` | Anthropic | any | a request-wide prompt-caching breakpoint (block-level ones are ignored too); brain's prefix cache reuses any prefix it holds, matched by its tokens |
  | `parallel_tool_calls` | OpenAI chat, OpenRouter | `true` | the default: the model may call several tools in a turn; `false` is not enforced, so it is refused |
  | `modalities` | OpenAI chat, OpenRouter | `["text"]` | a chat reply is text; audio output is refused |
  | `provider` | OpenRouter | any | upstream-provider preferences; brain is the provider |
  | `route` | OpenRouter | any | brain resolves `model`, then `models`, in order |
  | `transforms` | OpenRouter | any | a prompt that fits is not transformed; one that does not is refused with `context_length_exceeded`, never trimmed |
  | `usage` | OpenRouter | any | brain always reports usage |

  `n: 1`, `logprobs: false`, `top_logprobs: 0`, an empty `logit_bias`,
  `response_format: {"type": "text"}` and `stream_options` are read, not
  ignored (below).
- **Sampling parameters.** `temperature`, `top_p`, `top_k` (an integer in
  0..=1000) and `seed` are passed on only when the request sets them; otherwise
  the request is sampled with the model's own defaults, the `generate` action's
  param defaults - for a checkpoint that declares them (`generation_config.json`,
  or `general.sampling.*` in a GGUF) the declared `temperature`, `top_k`, `top_p`,
  `min_p` and `repetition_penalty` - and an unseeded request is sampled with a
  fresh random seed. A neutral value a client sends (`min_p: 0`,
  `repetition_penalty: 1`, a zero penalty) overrides a default the model declares
  and is dropped for a model without that sampler. The reply's `brain.sampler`
  says what was applied (below). OpenAI parameters brain
  cannot honour are refused with 400 `invalid_request_error` naming the
  parameter rather than ignored: `n` > 1, a non-empty `logit_bias`, a non-zero
  `top_logprobs`, and a `response_format` other than `text`; on the legacy
  completions endpoint also `logprobs: true`. Their neutral values are accepted.
- **Anti-repetition sampling and the thinking budget.** For the text models that
  serve through the shared scheduler, a request may add (all off unless sent;
  a model without them refuses the field with 400 `unsupported_parameter`):
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
  `thinking.budget_tokens` sets it too) limits the tokens the reply (one turn)
  may spend inside think blocks, summed over every block it opens. When it is
  spent inside a block, the model's thinking-budget lead-in (Qwen's "Considering
  the limited time..." sentence, once a reply), the closing tag and the blank
  line after it are forced and the model answers; a block it opens again after
  the budget is spent is closed at once. A tool call the model writes inside
  the block (`<tool_call>` ... `</tool_call>`) is neither counted nor cut, and
  if the budget runs out after such a call the reply ends there, as if the
  model had ended its turn. The model's own tokens inside think blocks, outside
  such calls, never exceed the budget. A reply reports the tokens spent thinking
  in `usage.completion_tokens_details.reasoning_tokens` (absent when there were
  none): exactly the generated tokens the reply's parse reads as reasoning (the
  tags excluded, a call that ended the reasoning excluded, a call that stayed
  reasoning included), with or without a budget; `usage.completion_tokens` is
  every generated token.
- **Prompt cache.** `usage.prompt_tokens_details.cached_tokens` (Anthropic:
  `usage.cache_read_input_tokens`) is how many of the prompt tokens the model's
  prefix cache served instead of computing (0 for a cold prompt); a model that
  does not report it omits the field.
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
  the serving model's own: from the decode steps that drew the tokens, or scored after the generation as one
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
  score its own generation. A reply sampled with a temperature above 0 on a model that
  draws over its whole logits row, with or without a top-k, top-p or min-p cut, carries the
  decode steps' own log-probabilities under the whole distribution at the request temperature,
  `log_softmax(logits / T)[token]` (`log_prob_source: "decode"`): the exact behaviour policy,
  with no extra prefill. Any other reply (a greedy draw, a speculated step, a token sampled on
  the host) is scored as above, at the cost of one extra prefill of prompt plus generation on
  the serving thread (`log_prob_source: "score"`); on a GPU that pass runs other kernels than
  the decode did and differs from the decode-time distribution by a few thousandths of a nat
  per token. `logprobs` (above) is the sampler's own, truncated figure and is a different
  quantity.
  `policy_trace` may instead be the object `{"top_log_probs": K}` (an integer `1..=20`; no other key or type, each a 400
  naming `policy_trace.<key>`; a K past 20 or below 1 is a 400 naming `policy_trace.top_log_probs`, and a model that does not
  declare it refuses it by that name). The trace then also carries `top_log_probs` (K) and, for each generated token
  including the stop token, `generation_top_ids[i]` and `generation_top_log_probs[i]`: the K most probable token ids of
  the row position `i` was drawn from (most probable first, ties to the lower id) and their log-probabilities under the
  whole row at `log_prob_temperature`, with the same arithmetic as `generation_log_probs[i]`, which is the entry of the
  sampled token bit for bit when it is among them. For a draw over the whole row (no `top_k`, nucleus or min-p cut, no penalty or thinking budget, temperature above 0) the
  figures and the lists come from the decode steps that drew the tokens (`log_prob_source: "decode"`, no extra
  prefill); any other request (a cut, greedy or speculated draw) gets them from one scoring pass (`log_prob_source:
  "score"`, the extra prefill above), under the adapter the request ran with, and a model whose scoring pass cannot report a
  top list fails with `unsupported-capability` naming `policy_trace_top_log_probs`. The reply grows by at most 20 ids and 20
  figures per generated token. Over D-Bus the same two params are `policy_trace: true` and `policy_trace_top_log_probs: K`.
  The object may also carry `"support": N` (an integer `1..=256`, alone or beside `top_log_probs`; a value outside is a
  400 naming `policy_trace.support`): the sampler's support, which a trainer needs to replay a cut draw without bias. The
  trace then also carries `support_cap` (N), `generation_sampler_log_probs[i]`, the log-probability of generated token
  `i` (stop token included) under the distribution it was drawn from (the survivors of `top_k`, `top_p` and `min_p` at
  the request temperature; `generation_log_probs[i]` stays the whole row's), and `generation_support_offsets` (one more
  than the generated tokens, starting at 0) with `generation_support_ids`: token `i`'s support is
  `generation_support_ids[offsets[i]..offsets[i + 1]]`, the token ids that survived the cut, best first (ties to the lower
  id), the token among them. They come from the decode steps that drew the tokens (`log_prob_source: "decode"`, a top
  list asked beside them included); a draw that cuts nothing has the whole row as its support, so every range is empty
  and the sampler's figures are `generation_log_probs`. It is refused (400, `invalid-request`) for a greedy request and
  for a `top_k` above N, and fails with `unsupported-capability` on a model whose decode does not draw from rows it reads
  back; a request whose draw at some position keeps more than N survivors fails (`invalid-request`, naming the position
  and `policy_trace_support`) rather than return a shortened support. The reply grows by at most N ids, one offset and
  one figure per generated token. Over D-Bus the param is `policy_trace_support: N`.
- **Scoring given text (`score`).** A served Qwen3 or Qwen3.8 chat model (the Qwen3.8-27B served from a released FP8
  or GGUF checkpoint included), and each adapter id riding on it, has a
  `score` action, called through `POST /v1/run` with `model` the base or adapter id and `blobs.records` a
  `generic-messages-v2` JSONL text blob. Each message marked `"train": true` is an answer, scored teacher-forced
  under the model's full distribution at temperature 1, as the model is asked for it: the prompt is what a `generate`
  call renders for the messages before it (client text cannot spell a control token), the answer's tokens are its
  content plus the end-of-turn token that closes it (`include_stop`, default true). A line may instead be a token
  record, `{"prompt_token_ids": [...], "token_ids": [...]}`: its `token_ids` are scored exactly as given after its
  `prompt_token_ids` as given, with nothing rendered or added. That is the form for a served generation whatever it
  holds (reasoning, tool calls, tokens a thinking budget forced): send back the `prompt_token_ids` and
  `completion_token_ids` (or a policy trace's `generation_token_ids`) a reply reported, under the model or adapter
  whose log-probabilities you want, for an importance ratio against the policy that sampled them or a teacher's
  log-probabilities for distillation (a training record's `teacher_logprobs` are a token record's `token_logprobs`, its
  `teacher_logprob` a chat record's answer `logprob`; see the train job's `distill_teacher` in
  `docs/guides/training/served-training.md`). Params: `per_token` (also return
  each answer's `token_ids` and `token_logprobs`), `include_stop`, `enable_thinking` (render the prompts as a request
  that allows reasoning would), `temperature` (default 1, within 0.001..=100: the log-probabilities are
  `log_softmax(logits / temperature)` over the whole vocabulary, so a ratio against a policy sampled at temperature
  T scores at T). The reply's `outputs` hold `records` (each with its `answers`: `message` (absent for a token
  record), `tokens`, `logprob`, `mean_logprob`), the totals `tokens`, `logprob`, `mean_logprob`,
  `log_prob_temperature`, the `tokenizer_digest`,
  `chat_template_digest` and `base_digest`, and the `adapter` that scored. At most 4096 records of at most 64
  supervised answers each, and 8192 answers in a call; a chat record with tool calls, tool results, tool schemas or
  reasoning content, one with no supervised message, a token record with an empty context or nothing to score, an id
  outside the tokenizer's vocabulary or a field it does not have, a malformed record, or one longer than the model
  holds fails the whole call by record number before any pass (400). A model that cannot score a continuation does not declare the action. Scoring is one
  prefill per answer on the serving instance, between its decode steps; it does not read or write the prefix cache.
- **Seed and served identity.** An OpenAI or OpenRouter chat or completions
  reply (and the terminal chunk of a stream) and an Anthropic `Message` (and a
  stream's closing `message_delta`) carry a non-standard
  `brain: {"seed": <int>, "sampler": {...}, "adapter": {"id", "digest", "version"}, "base_digest", "backend",
  "engine"}` object. `seed` is the
  seed the decode used, drawn fresh for an unseeded request, so any reply can
  be reproduced by sending it back. `sampler` holds the settings the tokens were
  drawn with, exactly as the sampler applied them after the model's defaults:
  `temperature` (0 is greedy), `top_k` (0 keeps every token), `top_p` (1 keeps
  every token) and `min_p` (0 keeps every token), named as a request sets them;
  a trainer that reads the sampler's `logprobs` or checks that rollouts were
  drawn untruncated (`policy_trace`) reads them here. Over D-Bus the result's
  `sampler` key holds the same values under the `generate` params' names
  (`temp`, `top_k`, `top_p`, `min_p`). `adapter` appears only when an adapter
  answered and names the release that did, the same triple its model card carries. `version` is the policy version of that release under its id: 1 for the first release the server offered, +1 each time a release with different weights replaces it, never repeated while the process runs. A sequence runs on the release it started on and never spans a refit, so every token of a reply carries the one version; a trainer that records it with the tokens can mask a sample whose version is too far behind the weights it trains. The version is local to the serving process; a release a training run published also carries `lineage`, `{"run", "epoch", "update", "base_digest", "adapter_digest"}`: the run (its `resume_key`, or `job:<hex>` drawn for a run without one), the job of it (`epoch`, one more for each job that continued the run), the optimiser update the weights hold, the base it trained on and the adapter file. It is read from the adapter's own card, so every server that serves the file (a restarted one, a worker) names the same weights, and a reply names the lineage of the release it was decoded by, as its model card and `policy_trace.policy.adapter` do. An adapter no run published has no `lineage`.
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
  not overload. Genuine overload still sheds as **503**. A job still waiting when
  the scheduler's own `--queue-deadline-ms` passes (longer than both by default)
  is answered **504** with a `timeout` error naming that flag. See
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
  (type, default, range, allowed values), blob inputs and blob outputs, whether it
  streams progress (`streaming`) and whether it stops when cancelled (`cancellable`).
  A param whose absence itself means something (a setting of one training objective,
  a LoRA rank a continued adapter brings) has no `default`, which the server would
  fill in, but may carry `implied_default`: the value the action applies when the
  param is omitted, so a caller can record a run's full settings. Clients are
  generated from it.
  Beside the list, `server` says what serves it: `engine` (`version`; `commit`, the full
  git commit the build was made from, and `dirty`, whether its tree had uncommitted
  changes, when the build knew them), `backend` (`cpu`, `cuda`, ...) and
  `training_records` (`contract` `training-records-v2`, `version` 2 and `digest`, the
  `sha256:` digest of the contract's example set in brain's source tree,
  `crates/data/tests/fixtures/training-records/v2`, computed as
  `data::chat_rollout::contract_digest` does: the SHA-256 of one line
  `<path>\0<sha256 hex of the file>\n` per file but `README.md`, sorted by path). A
  producer that keeps a copy of the set compares that digest with its copy's to know it
  writes the records this server's trainer checks. Each key is absent when the server
  does not know it.
- `GET /v1/devices` lists the devices the server schedules work on, as `{"object": "list", "host": {...}, "data":
  [...]}`. Each entry has `id` (`cpu`, `gpu<i>`, `npu<i>`: the names `--device` takes), `kind` and `memory`
  (`total_bytes`, `reserved_bytes` held back by the operator, `used_bytes` held by resident models and runs,
  `free_bytes`), and, for a card, `name`, `vendor_id`, `class` (`discrete`, `integrated`) and `backends` (which of
  `cuda`, `vulkan`, `wgpu` can drive it). `host` has the backend in use, the core count and the card and NPU counts.
  It names no path, bus address or UUID of the machine. The `cpu` entry is the host's RAM, budgeted for host-side
  work even when `--device` names only accelerators. Per-card compute capability is not reported: it is known
  only by opening the card, which the serving process does once, for the cards it serves on.
- `POST /v1/run` runs one action and answers with its outputs and blobs.
- `POST /v1/jobs` starts one in the background, for work too long to hold a request
  open. `GET /v1/jobs` lists the caller's own jobs, running or finished and not yet
  forgotten, oldest first, as `{"object": "list", "data": [{"id", "status", "model",
  "action", "created", "progress"?, "error"?}]}` (`created` in seconds since the Unix
  epoch). `GET /v1/jobs/{id}` reports `running` (with `progress.step` / `progress.total`),
  `succeeded`, `failed` or `cancelled`; `GET /v1/jobs/{id}/result` returns the result
  once it exists (409 while it is running); `DELETE /v1/jobs/{id}` cancels it, and a
  `cancellable` action stops at its next step. `GET /v1/jobs/{id}/events` follows it as
  server-sent events: a `progress` event per step tick, a `frame` event per structured
  frame the action reports (a training step's loss and statistics), and one `done`
  event with the final status. The last 4096 frames are kept, and `Last-Event-ID`
  resumes after the last event received; at most 8 streams follow one job.
- `POST /v1/jobs/{id}/records[?epoch=N]` and `POST /v1/jobs/{id}/finish` send a running job input: the JSONL
  rollout records of a streaming training job and the end of them (body at most 64 MiB; `finish` takes none). The
  job answers each: `200` with its receipt, `400` for input it refuses, `409` for input that disagrees with its state
  (another run epoch, records after `finish`), `429` with `Retry-After` for a full queue (`detail.allowance`), `503`
  with `Retry-After` for a job not ready to answer yet, `410` for a job that has ended or reads no input, `404` for
  an id the caller does not hold. See
  [Training while serving](../../guides/training/served-training.md#training-while-serving-a-streaming-job).

  A job is detached from the request that started it and from its event streams: a
  client that hangs up does not end it. Ending one is a `DELETE /v1/jobs/{id}`, with the
  id from the `POST` or from `GET /v1/jobs` (a client that lost it, after a restart or
  an interrupt, finds it there). A training run given a `resume_key` continues from its
  last checkpoint when it is submitted again after the server itself stopped.

A call is `{"model", "action", "params", "blobs": {name: {"media", "data", "meta"}}}`
(no other field is accepted) with `data` base64 and `media` one of `image`, `mask`, `audio`, `video`, `text`,
`bytes`. A result is `{"outputs", "blobs"}` in the same blob shape. The call is
checked against the action's own spec first (wrong types, values out of range, a
missing or mistyped blob are a 400; a param or blob the action does not declare is a
400 `unsupported_parameter` naming it and the model), and the request body is
bounded at 64 MiB. The OpenAI-shaped embeddings, images, transcription and speech
routes hold the call they build to the same spec.

A param the host resolves for itself, such as a weights path, is never offered by
`/v1/capabilities` and is refused (400, `unsupported_parameter`) if a call names it: a caller
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
against another server. That holds for a chat or completion request too, as long
as its generation had not streamed text yet; one that had ends with its partial
text and `finish_reason` `"cancelled"`. Over D-Bus a cancelled call fails with the
message `cancelled`, whoever cancelled it, with the same exception for a text
generation that already streamed.

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
The Anthropic entry carries only `supported_parameters` of them.

| Field | Meaning |
|---|---|
| `capabilities` | Names derived from the model's manifest: `chat`, `tools` (chat whose `generate` accepts `tools`), `vision` (chat whose `generate` reads `image` inputs; image parts sent to any other model are refused with 400), `embeddings`, `images`, `transcription` (a `transcribe` action taking `audio`), `speech` (an action that declares a speech contract and returns `audio`). On OpenRouter the card's `architecture` modalities follow from the same set in OpenRouter's names: `audio->transcription` for a transcription model, `text->speech` for a speech model. |
| `speech` | On a `speech` model: `response_formats` (`wav`, `pcm`), `max_reference_seconds`, and `modes`, one per speech action, each saying whether it imitates a recording (`reference_voice`), whether that recording's transcript is `none`, `optional` or `required`, and whether it takes `instructions`, preset voices (`preset_voices`), a `seed` and a `language`. A request that needs a mode the model lacks is refused with 400. |
| `max_images_per_request`, `max_image_pixels_per_request` | On a `vision` model: the most images (the `image`, `image1`, ... inputs its `generate` action declares), and pixels over all of them, one request may carry; a request past either is refused with 400. |
| `architecture.input_modalities` | On a chat model: `"text"`, then `"image"` for a `vision` model and `"audio"` for one whose `generate` takes an `audio` clip (the field an agent runtime reads to decide what it may attach). |
| `supported_parameters` | On a model that generates text, on every dialect: the request fields its `generate` action honours - `max_tokens`, then the sampling parameters among `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty`, `min_p`, `repetition_penalty`, the DRY fields, the thinking-budget names and `ignore_eos`, then the token-capture fields (`logprobs`, `return_token_ids`, `policy_trace`). A field not listed is refused with 400 naming it. |
| `max_output_tokens` | The most new tokens one request may ask for, when the model's `generate` action declares an upper bound. On OpenRouter it is also `top_provider.max_completion_tokens`. |
| `adapter` | Present on an entry that IS a LoRA adapter: `{ "id": "<adapter card id>", "digest": "sha256:<hex>", "version": <int> }`. The id is the adapter's name and the digest is that of its file; no file path is ever included. A release that replaces an adapter under the same id changes the digest and raises the policy version on the next read. |
| `support` | Always present: `"production"` when the model's architecture is certified for serving (forward parity, serving, cancellation and batching each backed by tests in the [serving certification](../training-capabilities.md#serving-certification) table), `"preview"` otherwise. A model whose architecture the server does not know, or that the table does not list, is `"preview"`. An adapter reports its base's architecture. |
| `zero_data_retention` | Always `true`: the server keeps nothing of a request (no prompt, answer or upload is written anywhere or logged). A client that routes by data retention reads it to tell this server from one that does not say. |
| `decode_slots` | The sequences of this model the server decodes at once: its batch on each replica (1 for a model that declares none). `/health`'s `decode_slots` is the sum over every model the server could load, so a router in front of several servers (`brain serve --upstream`) bounds each model's requests per server by this field instead. Absent for a multi-device model. |
| `resident` | Whether the model is built on a device now (`true`), or would have to be loaded first, evicting what the device holds if memory is short (`false`). The router sends a model's requests to servers where it is resident while one of them has a free slot for it, and spills them onto a completely idle server where it is not once every server where it is carries at least 8 requests for it; a server busy with another model is never spilled onto. The router's own listing leaves `decode_slots` and `resident` out: they describe one server. |

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
parameter the server sets itself; every route and D-Bus `Run` refuse
`__adapter` (and `adapter`) like any other undeclared name.

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
