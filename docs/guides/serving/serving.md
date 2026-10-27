# Serving - `brain serve`

`brain serve` loads whichever models are configured (see
[`docs/reference/configuration.md`](../../reference/configuration.md)), makes them resident under
one scheduler, and exposes them over one or more transports at once.

```
brain serve [--openai[:PORT]] [--anthropic[:PORT]] [--openrouter[:PORT]] \
            [--dbus] [--models-dir DIR] [--api-keys-out FILE] \
            [--api-key-file FILE] [--runtime-dir DIR] [--ready-file PATH] \
            [--checkpoint PATH [--tokenizer PATH]] \
            [--adapter FILE | --adapter-manifest FILE | --watch-adapters DIR]
```

With no surface flag, `brain serve` serves the OpenAI-compatible API on
`127.0.0.1:8788`; `brain serve --stdio` is the event-driven stdio controller.

- `--openai[:PORT]` - serve an OpenAI-compatible HTTP API (default port if
  omitted). Every route is served both with and without the `/v1` prefix -
  either works as a client's `base_url`.
- `--anthropic[:PORT]` - serve an Anthropic-compatible HTTP API (`/v1` only).
- `--openrouter[:PORT]` - serve an OpenRouter-compatible HTTP API (with and
  without `/v1`, same as OpenAI).
- `--dbus` - serve the D-Bus control surface (`com.swedishembedded.Brain1`).
- `--models-dir DIR` - the model store: scanned at startup, and the one store
  every served model's weights are resolved from (overrides
  `BRAIN_MODELS_DIR`).
- `--api-keys-out FILE` - write the generated API keys to `FILE` (mode 0600). The stdout
  `APIKEY` lines then name the file and omit the keys, so server logs on a shared
  machine do not hold them.
- `--api-key-file FILE` - serve every dialect with the one key `FILE` holds
  (surrounding whitespace trimmed) instead of a fresh key per dialect per launch.
- `--runtime-dir DIR` - where the pid file, log and connection record live
  (default `$XDG_RUNTIME_DIR/brain`, else `<tmpdir>/brain-<uid>`). A server under
  its own `DIR` is not found by clients looking in the default one.
- `--checkpoint PATH`, `--tokenizer PATH` - also serve this Qwen3 checkpoint
  (file or repo directory), with this `tokenizer.json` when none is beside it.
- `--ready-file PATH` - see **Readiness** below.
- `--adapter FILE`, `--adapter-manifest FILE`, `--watch-adapters DIR`,
  `--adapter-slots N`, `--adapter-rank-cap R`, `--adapter-targets LIST` - see
  **Serving LoRA adapters** below. At most one of the first three may be given.
- `--decision-model REF|DIR` (repeatable), `--decision-default ID` - serve
  decision models beyond the ones found in the model store, and name the one
  `auto` resolves to. See [Decision models](../../reference/models/decision.md).

An unknown flag is a hard error (exit 2), not a warning; so is a stdio-only
flag (`--gpt`, `--yolo`, `--conf`) without `--stdio`. Run `brain serve --help` for the
authoritative flag list.

## The serving policy

How a model is stored and sized is one model-agnostic policy, the same for
every model that serves it; no flag is spelled with an architecture name.

| flag | what it chooses |
|---|---|
| `--checkpoint PATH` | the LLM checkpoint to serve (file or repo directory) |
| `--weight-format SPEC` | weight storage: `f32`, `bf16`, `f16`, `int8`, `q4`, `nf4`, or per leaf `int8,blk.0.attn_q=f32` where the model supports rules |
| `--kv-cache TIER` | KV-cache storage: `f32`, `bf16`, `int8`, `fp8` |
| `--gemm auto\|fp32` | `fp32` pins exact fp32 arithmetic (the reference tier); `auto` uses tensor cores where the device has them |
| `--kv-calibration` | use a `kv_calib.json` beside the checkpoint |
| `--exact-decode` | make a decode row's result independent of which other requests share its batch: int8 projections (and the head) run in groups small enough to stay on the bit-identical kernel and every row's attention is sized from `--ctx`, at the cost of one extra pass over the weights per group; speculation is off in this regime. Default off |
| `--ctx N` | cap the context window below the model's declared one |
| `--max-batch N` | the concurrency the server must carry |
| `--kv-offload-gb N` | host RAM granted for preempted sessions' KV |

Each model declares which formats it supports. A choice a model cannot honour
is refused by name with the supported set (for example `qwen3: KV cache bf16
is not supported; supported: f32, int8, fp8`); the model does not serve, and
the server refuses to start only when no served family can honour the policy.
A limit that has no meaning for a model is reported as a note at startup.

Only real choices are flags. The prefix-cache slots follow `--max-batch` (one warm conversation per concurrent sequence, never fewer than four). The KV pool size, the prefix-cache slots, the
prefill slice, batch and row cap, and the residual-add and norm fusion are
derived from the model's metadata, the device and the memory plan (the context
window from the model's declared window and the free memory) and are not
options; the values a server runs with are reported at startup. The old
architecture-prefixed spellings (`--qwen-ctx`, `--qwen35-kv`,
`--qwen-weights-int8`, `--kronos-decoder`, ...) still work with a deprecation
warning until their callers have moved; the ones for derived settings still
set them so existing launch lines behave as before.

## Sampling defaults

A request that omits `temp`, `top_k`, `top_p`, `min_p` or `repetition_penalty` is
sampled with what the checkpoint declares: `general.sampling.*` in a GGUF,
`generation_config.json` beside Hugging Face weights. The values are the
`generate` action's param defaults, so `Manifests` (and `GET /v1/capabilities`)
show what an omitted setting will be. A chat model whose checkpoint declares
nothing (or only some of them) is sampled with temperature 0.7, top-p 0.95 and
top-k 20 for the rest, not greedily; a value the request sets always wins.

## Serving LoRA adapters

One resident copy of a Qwen3's weights serves the base and any number of LoRA
adapters **at the same time, in the same batches**. An adapter is a model: it
is selected by its card id, `<vendor>/<repo>:<owner>:<name>:<tag>` (for
example `Qwen/Qwen3-0.6B:lab:support:v3`), in the `model` field of any request
over HTTP or D-Bus. The base keeps answering under its own id with no adapter.
`GET /models` lists one entry per served adapter id, each with
`adapter: {id, digest}` (see [`docs/reference/api/http.md`](../../reference/api/http.md)), and the
base's entry unchanged.

Adapters are loaded by the operator only. No request parameter can load an
adapter, name a file, or make the server fetch one: a model id that is not
served, adapter-shaped or not, is an unknown model (404), the same answer and
the same body as any other unknown id.

Three things put an adapter in front of the base:

- a model directory (`--models-dir`) that holds adapters under
  `<vendor>/<repo>/adapters/<owner>/<name>/<tag>/` for a served Qwen3 base
  offers each as `<vendor>/<repo>:<owner>:<name>:<tag>`;
- `--adapter FILE`, `--adapter-manifest FILE` or `--watch-adapters DIR` (below)
  for the Qwen3 selected with `--checkpoint PATH` (the checkpoint file or its repo directory), or, when there is none, for
  the one released Qwen3.8 GGUF the model directory serves (several are
  ambiguous and refused by name: put the adapter in the model directory under
  the id of the one it rides on).

A released Qwen3.8 GGUF takes adapters the same way (card family `qwen35`, as
`brain qwen35 finetune` writes them), on its multi-card INT8 weights: the
adapter is found under the GGUF's repo directory, or by the full-precision
release its card names as its base (`Qwen/Qwen3.8-27B-FP8`), which the GGUF
quantises - an adapter is trained on the release and served on the
quantisation, so the digest of the GGUF is not the one the card records. An
adapter of the other family is refused by family.

### The adapter bank

The adapters run from a fixed-size bank of device memory built beside the base
weights. It is built only when adapters are asked for - by any adapter source
above, or by one of these flags - and costs nothing otherwise:

- `--adapter-slots N` - adapters resident on the device at once (default 4).
  More adapters than slots are still served: the least recently used idle one
  is swapped out of its slot when another is needed.
- `--adapter-rank-cap R` - the highest LoRA rank a slot holds (default 16).
- `--adapter-targets LIST` - the linears the bank corrects: `attention`
  (the default; `wq,wk,wv,wo` on Qwen3, what the default LoRA training targets;
  the Gated-DeltaNet and attention projections on Qwen3.8 - an adapter that also
  trains the MLP needs `all`),
  `all` (every adapter site the model has), or a comma-separated list of the
  model's site names (`wq wk wv wo gate up down` on Qwen3). A name the model
  has no site for is refused when the bank is built, listing the ones it has.

An adapter above the rank cap, or correcting a linear outside the targets, is
refused when it is loaded, with the flag that would admit it. For adapters in a
model directory the bank is widened to hold them unless these flags were given:
`--adapter-slots` and `--adapter-rank-cap` keep what they say but leave the
targets to the adapters (widened to cover them); only `--adapter-targets` makes
the targets the operator's, and an adapter outside them is skipped with that flag named.
The bank's device bytes (`slots x rank-cap x (k + n) x 4` per targeted linear
per layer) are part of the model's footprint, so placement budgets them.

### Following releases

- `--adapter FILE` offers exactly one release and nothing replaces it while the
  server runs. Before any surface binds, the adapter is read once and bound to
  the served base: when its card records the digest of the base it was trained
  against (`brain::ChatFineTune` records it), the served base must have that
  digest; an adapter without one must name the served base's id. A mismatch
  stops the server with both digests in the message. The startup line says
  what is served: `brain serve: <id> adapter=<id> digest=sha256:<hex>`. The
  digest is the SHA-256 of the adapter file, the same value
  `ChatFineTuneOutcome::adapter_digest` reports for it.
- `--adapter-manifest FILE` follows a release manifest,
  `{ "adapter": "releases/support-3.safetensors", "digest": "sha256:<hex>" }`
  (a relative path is relative to the manifest). A new release is offered only
  once the file hashes to `digest` and binds to the served base; a manifest
  that fails any check keeps the previous adapter and prints why. Replace the
  manifest atomically; a release step in Rust does this with
  `brain::AdapterManifest { adapter, digest }.write_atomically(path)`. At
  startup a manifest that exists must verify; one that does not exist yet
  offers the base alone until it is published.
- `--watch-adapters DIR` offers the highest-versioned
  `adapter-NNNNNN.safetensors` in `DIR` (the version, never the modification
  time). Files there are read once and checked to be Qwen3 LoRA adapters; they
  are not bound to the base, since a gated promotion put them there.
- The three are mutually exclusive: each names the adapter it offers by its own
  card id, and with two sources "which id is current" would depend on which
  moved last.

Each adapter file is read once: the digest reported is of the very bytes the
engine was given, so there is no window in which the file can be rewritten
between verification and serving. Publish each release under a new file name.

A release has a new card id (a new tag) and a new digest. It is offered beside
the old one first, then the old id is withdrawn: it leaves `GET /models` and
takes no new request (404), requests already running or queued for it finish on
it, and the engine unloads it afterwards. A swap interrupts nothing. A client
that follows "the latest" reads the model list for the current id. A release
that keeps its card id under a new digest replaces the adapter under that id.

The adapter flags need a served Qwen3 (`--checkpoint`) behind an HTTP or
D-Bus surface; `--adapter` and `--adapter-manifest` without one stop the
server, and the stdio controller refuses all three.

Adapters are served in fp32 on the fp32 activation, so they are exact on int8
weights too. Clip ceilings calibrated on the base (`--kv-calibration`) are
applied to adapter sequences as well, and the server says so once.

## Access control is always on

There is no way to run `brain serve` without authentication. Every requested
surface gets its own freshly generated API key at startup: printed to
stdout as `APIKEY <provider> <key>` lines, or - if `--api-keys-out FILE` is given -
written to `FILE` (the lines then read `APIKEY <provider> written to FILE`). Both the keys and the file are written before any socket
binds, so a client can never race the server up.

## Local clients

A client on the same machine needs neither the key nor the port. Once every
HTTP surface is bound, the server writes its connection record,
`<runtime dir>/serve-<pid>.json`, owner-only (mode 0600), and removes it when
it stops:

```json
{"pid": 4242, "surfaces": [{"provider": "openai", "url": "http://127.0.0.1:8788", "key": "sk-brain-..."}]}
```

There is one entry per bound address; `url` is the address the listener
actually holds (a `:0` request is reported as the port it got, an unspecified
bind as loopback). The OpenAI and OpenRouter routes answer under `url` and
`url/v1`, the Anthropic ones under `url/v1`. Several servers may run at once,
each with its own record; one whose process is gone (a crash leaves its record
behind) is to be skipped. A server started with its own `--runtime-dir` is
private to whoever reads that directory.

## Admission and backpressure

A request that can't be admitted to run promptly gets an HTTP 429 rather than
queuing indefinitely. The deadline is `--admit-deadline-ms N` (default
900000, fifteen minutes), with one exception: a model's first-ever activation
("cold start" - building the model on device for the first time) gets a longer
grace window, `--cold-build-admit-deadline-ms N` (default 1800000), because
that one activation is inherently slower than every request after it. Both
must be positive; a value that is not is refused at startup.

A request the engine can never serve - a prompt plus `max_tokens` longer than the
model's context (`context_length` in `/v1/models`), a token outside the vocabulary, an
adapter the server does not hold - is refused at once, whatever the running requests
are doing: it does not wait for a free slot to be told no. The refusal is an HTTP 400
with an OpenAI-style error body (`context_length_exceeded` for length), streamed or not:
a streamed request is given 300 ms to fail before its response headers go out, so an <!-- perf-number: the configured grace period before response headers go out, not a measurement -->
early refusal is a plain status rather than a 200 stream holding one `error` event. A
failure after the response has begun is an `error` event in the stream.

The default is long on purpose. A request waiting behind sequences that are
already running will start when they finish, and an agent turn runs for many
minutes, so that wait is queueing, not overload: with a seconds-long limit the
third and fourth concurrent long request were refused with 429 while the first
two were still making progress. Lower `--admit-deadline-ms` if you would
rather shed fast and let the client retry elsewhere (a front end with many
short requests, or a load balancer in front of several servers).

Every refusal is also logged on stderr, at any verbosity, as one `[warn]`
line naming the model, the queue depth, the deadline that applied and the
time waited. The 429 body repeats the wait and the limit and names the flag
that raises it.

Genuine overload - the server has more concurrent work than it can ever get
to, not just a momentary queue - sheds requests as HTTP 503.

## One model, every transport

A served model is uniformly discoverable and callable no matter how you
reach it:

- `brain caps` lists every model currently loaded and what it can do.
- `brain <architecture> <action>` invokes any action from the CLI; the same
  action is reachable from HTTP or D-Bus too - all three paths go through the
  same scheduler, the same batching, and the same cancellation semantics.
  There is no separate code path or separate behavior per transport.

## Cancellation, deadlines and concurrency

Long-running actions - multi-minute generation, training - are cancellable
mid-flight from any transport that supports it (D-Bus `Cancel`, or closing
the client connection over HTTP). What that guarantees, precisely:

- **Cancellation is cooperative and polled.** A model checks the request's
  cancel flag between its own steps: one decode token, one denoise step, one
  item of a batch. Nothing checks it *inside* a step, so a step already in
  flight always completes. On a healthy device that is milliseconds; a step
  wedged in a driver call is not interruptible at all from here.
- **Not every model honours it.** The flag rides on every request, so an
  action that ignores it looks exactly like one that honours it, and a client
  cancelling such a generation simply waits for it to finish. The models that
  ignore it are listed in `scripts/gates/check-cancellable-actions.sh`, which
  fails the build if a new one joins them. The Qwen3 family (including the
  batched serving engine), the image/video generators, the SUPIR/splat
  pipelines and every LoRA trainer honour it; several of the vision-language
  and audio models on that list do not.
- **A cancelled generation says so.** It returns the text it had produced,
  with `finish_reason: "cancelled"` - never `"stop"`, which would make a
  truncated answer indistinguishable from a finished one. Over the
  OpenAI-shaped surfaces that value is collapsed into OpenAI's own `stop`
  (their enum has no other slot for it) and survives verbatim under
  `native_finish_reason`. A non-generating action that had produced nothing
  yet fails with `cancelled` instead.
- **There is no request timeout.** `max_tokens` bounds the WORK a request
  does, not its duration; admission deadlines bound the wait to *start*, not
  the run itself (see **Admission and backpressure** above). A caller that
  needs a wall-clock bound cancels from its own timer, within the limits
  above. The longest uninterruptible stretch is not a decode step but a
  model's FIRST activation: reading the checkpoint and building it on the
  device happens before any step loop exists to poll a cancel from, and takes
  minutes for a real model. That is what
  `--cold-build-admit-deadline-ms` exists for.
- **One model serves one request at a time** unless it is on the paged-KV
  batching engine, which explicitly batches concurrent sequences. Concurrent
  requests for the same non-batched model are serialized by that model, so
  the second one's latency includes the first one's. This is a property of
  the model, not of the transport: it is the same in-process, over HTTP and
  over D-Bus.

The same applies to a host that skips the transports entirely and drives
`capability::Registry` in-process. That interface does not serialize
anything of its own: an action may be entered concurrently and is responsible
for its own state. `crates/capability/src/lib.rs` documents the contract,
and `crates/capability/tests/contracts.rs` pins it.

The one promise brain does NOT make here is a typed failure. An action
returns `Result<Outcome, String>`, so "unknown model", "context exceeded",
"out of memory" and "the device fell over" are distinguishable only by their
wording, and the HTTP layer separates the cases it must (a 400 for a prompt
past capacity, a 404 for an unknown model) by matching that wording. A caller
outside brain should treat any other error as opaque and retryable at its own
discretion, not parse it.

## Readiness for scripted startup

`--ready-file PATH` touches an empty marker file once **every** requested
surface has actually bound - every HTTP dialect plus D-Bus - and never at
all if any of them fails to come up. Because the API keys are written before
any socket binds, the marker appearing implies both that the keys are on
disk and that the socket(s) are accepting connections, so a launcher script
can simply poll for the file's existence instead of polling a port or
grepping logs.

There is deliberately no `/healthz`-style unauthenticated route - readiness
here is a process-level property that also has to cover D-Bus, which has no
HTTP route to answer for it. Use `--ready-file` for scripted startup instead.

## Per-protocol detail

This page covers the `brain serve` command itself. For the wire-level detail
of each transport, see:

- [`docs/reference/api/http.md`](../../reference/api/http.md) - the OpenAI/Anthropic/OpenRouter
  HTTP surfaces.
- [`docs/reference/api/dbus.md`](../../reference/api/dbus.md) - the D-Bus control surface.
