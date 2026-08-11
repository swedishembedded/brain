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
| `Run(model, action, params, in_fds, in_meta, transport)` | one-shot request → result + output data |
| `Subscribe(model, action, params, in_fds, in_meta)` | a job that streams progress/result frames back — for long-running generation |
| `Cancel(job)` | cooperative cancel of an in-flight `Subscribe` job |
| `StreamTranscribe(model, params, pcm_fd)` | continuous-input method for live audio, e.g. streaming ASR |
| props | `Version`, `ActiveJobs` (in-flight `Run`/`Subscribe`/stream jobs), `Models` |

Every model's actions are discoverable the same way as on the CLI (`brain caps`) —
see [`docs/using/cli.md`](cli.md) if that page exists yet.

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
else.

### `Cancel` — cooperative cancellation

`Cancel(job)` takes the job id `Subscribe` returned and asks that job to stop.
Cancellation is cooperative: a long-running action checks for it between steps
(denoising steps, training steps) and aborts, which arrives as the stream's terminal
`error` frame (`"cancelled"`). Returns `true` if the job was still in flight, `false`
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
and [`examples/dbus/`](../../examples/dbus/) has a runnable example
(`brain_dbus.py`) showing discovery, a one-shot image action read back over a file
descriptor, and a streaming request — plus a `busctl`-based smoke test
(`busctl_smoke.sh`) that needs no Python at all. See that directory's README for the
exact commands.
