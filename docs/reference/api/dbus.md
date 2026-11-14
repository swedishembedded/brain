# brain over D-Bus (`com.swedishembedded.Brain1`)

brain exposes the same served models over D-Bus as it does over the CLI and the HTTP
APIs — same scheduling, same batching, same cancellation, just a different transport.
It's a control surface for local Linux apps: discover models, run actions, and exchange
images/audio/results as file descriptors instead of bytes marshalled through D-Bus.

## Enable & run

Compiled into the default build; the service only runs when you pass `--dbus`:

```bash
brain serve --dbus                 # session bus (default), name com.swedishembedded.Brain1
brain serve --dbus --dbus-system   # system bus (needs a system.d policy)
brain serve --dbus --dbus-name com.example.MyBrain
```

On the system bus the shipped policy
(`scripts/build/com.swedishembedded.Brain1.conf`) lets only the system user
`brain` own the name, and only root and the `brain` group call it: run the
server as that user, never as root (`useradd --system --no-create-home
--user-group brain`, then `sudo -u brain brain serve --dbus --dbus-system`).

Without a session bus of your own, wrap it in a private one:

```bash
dbus-run-session -- bash -c 'brain serve --dbus & sleep 1; ...client...'
```

## Interface

Bus `com.swedishembedded.Brain1`, object `/com/swedishembedded/Brain1`, interface
`com.swedishembedded.Brain1.Manager`:

| Member | Purpose |
|---|---|
| `Manifests()` | JSON of every model/action/param (discovery) |
| `ListModels()` | served model names |
| `Devices()` | JSON of the devices the server schedules on: each one's `id` (`cpu`, `gpu<i>`, `npu<i>`), `kind`, `memory` (`total_bytes`, `reserved_bytes`, `used_bytes`, `free_bytes`) and, where the host can say, a card's `name`, `vendor_id`, `class` and `backends`, plus `host` (backend, core and card counts); the same document as HTTP `GET /v1/devices`, with no path, bus address or UUID |
| `Run(model, action, params, in_fds, in_meta, transport)` | one-shot request → result + output data |
| `Subscribe(model, action, params, in_fds, in_meta)` | a job that streams progress/result frames back — for long-running generation |
| `Cancel(job)` | cooperative cancel of an in-flight `Subscribe` job |
| `StreamTranscribe(model, params, pcm_fd)` | continuous-input method for live audio, e.g. streaming ASR |
| props | `Version`, `ActiveJobs` (in-flight `Run`/`Subscribe`/stream jobs), `Models` |

Every model's actions are discoverable the same way as on the CLI (`brain caps`) —
see [`docs/reference/cli.md`](../cli.md) if that page exists yet.

The bus is a serving surface like HTTP: `Manifests()` lists only the params a
caller may set, and `Run`, `Subscribe` and `Plan` refuse any other as an unknown
param. That leaves out every param that names something on the serving
machine: a checkpoint (`weights`), a training dataset or output path
(`data`, `save`), an adapter file (`adapter`), a conditioning image file
(`start_frame`, `end_frame`, `mid_frame`). The host answers those itself, or
they are set by someone on the machine with `brain do`.

### `Run` — one-shot

A single request/response call: input blobs (images, audio) go in as file
descriptors, the result comes back as JSON plus output blobs, also as file
descriptors. Nothing is inlined onto the bus — blob data always travels by fd.

### `Subscribe` — streaming

For actions that run for a while (generation, training), `Subscribe` returns a job id
and a stream you read progress/blob/done frames from as they happen, e.g.:

```json
{"type":"progress","step":3,"total":10,"message":"sampling"}
{"type":"blob","name":"image","media":"image","meta":{"w":256,"h":256,"c":3}}
{"type":"done","result":{"width":256,"height":256}}
{"type":"error","message":"..."}
```

A slow subscriber's frames are dropped rather than stalling inference for everyone
else. A `progress` frame carries the action's structured frame in `event`: the training
job (`brain/train`, see [Training as a served job](../../guides/training/served-training.md))
reports each optimiser step there, takes its datasets as input descriptors and returns
the adapter as an output descriptor. Its other actions (`full_train`, `merge`, `check_dataset`)
are reached the same way and take their inputs as descriptors; a checkpoint they produce leaves
by manifest digest in the artifact store, never as a descriptor.

### `Cancel` — cooperative cancellation

A text `generate` result also carries `cached_tokens` (the prompt tokens the
prefix cache served), the `seed` the decode sampled with (drawn or given: pass
it back to reproduce the decode), and `scan_anomalies` when the chat scanner
could not read the output cleanly (a tool call written inside the reasoning,
a call in the wrong syntax or cut off, text dropped from a call). Passing
`return_raw: true` adds `raw_output` - the completion exactly as the model wrote
it, every reasoning and tool-call marker included, ending with the stop token
(`stop_token`) when one ended it - so a client can tell a turn that ended in
prose from one whose tool call was lost.

`Cancel(job)` takes the job id `Subscribe` returned and asks that job to stop.
Cancellation is cooperative: an action whose manifest entry says `"cancellable": true`
checks for it on entry and between steps (decode tokens, denoising steps, training
steps) and aborts, which arrives as the stream's terminal `error` frame
(`"cancelled"`). A text generation that has already streamed text ends instead with
its partial text and `finish_reason` `"cancelled"`. Returns `true` if the job was still in flight, `false`
for an unknown or already-finished id.

### `StreamTranscribe` — continuous input

For live-input models (streaming ASR today, and the pattern for any future live
audio/video model), the client keeps a file descriptor open and writes to it
continuously; brain windows that input into jobs internally and answers with
`segment`/`done` frames as transcription becomes available, over the same kind of
stream `Subscribe` uses.

## Blob format conventions

Blobs use fixed, model-independent conventions — a client encodes/decodes against
these, not a per-model format:

- **Images** — raw interleaved HWC pixel data as 32-bit floats in `[0, 1]`, with
  width/height/channel-count carried alongside as metadata.
- **Audio** — raw mono 32-bit float PCM at 16 kHz, with the sample rate carried
  alongside as metadata.

## Client examples

A reusable Python client lives in the `brain-py` package (`brain_py.dbus.BrainDBus`),
and [`samples/python/dbus/brain-dbus/`](../../../samples/python/dbus/brain-dbus/README.md)
has a runnable example (`brain_dbus.py`) showing discovery, a one-shot image action
read back over a file descriptor, and a streaming request - plus
[`samples/shell/dbus/busctl-smoke/`](../../../samples/shell/dbus/busctl-smoke/README.md),
a `busctl`-based smoke test (`busctl_smoke.sh`) that needs no Python at all. See
those directories' READMEs for the exact commands.

The token facts of a text `generate` are params of the same action over D-Bus as over HTTP: `logprobs` and `return_token_ids`
(the sampler's figure, `completion_logprobs` and the ids in the result) and `policy_trace: true`, which adds a `policy_trace`
object to the result (prompt ids, generated ids with the stop token, and their log-probabilities under the model's full
distribution at the request temperature, with `tokenizer_digest` and `chat_template_digest`); the served identity is the
result's `base_digest` and `adapter` (`id`, `digest`, `version`). The validation is the generate spec's, the same on both
surfaces; a request that sets penalties, DRY, no-repeat bans or a thinking budget, or a model that cannot score its own
generation, fails with the `invalid-request` or `unsupported-capability` class. The HTTP reference
([HTTP API](http.md)) describes the object.
