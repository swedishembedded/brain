# brain over D-Bus (`com.swedishembedded.Brain1`)

The canonical protocol reference for brain's D-Bus control surface (compiled
into the default build, opt-in at runtime via `--dbus`), plus `brain_dbus.py`,
the minimal client that exercises it: discover the served models, pull a real
image back over a file descriptor, and stream a z-image generation - all
through the reusable client in the `brain-py` package.

```bash
pip install -e brain-py   # jeepney with fd passing
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/dbus/brain-dbus/brain_dbus.py
```

With brain's z-image weights exported (`BRAIN_S3DIT_*`) it also streams an
image generation; otherwise it runs the no-GPU `imageops` path only.

## What it demonstrates

* `brain_py.dbus.BrainDBus` (context manager + `read_fd`/`sealed_memfd`) as the
  reusable client every other D-Bus sample in this tree builds on.
* `Run` - a one-shot call (`imageops.gradient`) that returns a real image
  already materialised into bytes.
* `Subscribe` - a streaming call (`s3dit.text2image`) with a progress
  callback and a final blob frame.

## What it needs

- `pip install -e brain-py` (jeepney with fd passing).
- `BRAIN_S3DIT_*` exported, to exercise the streaming generation demo; without
  it that demo prints a note and is skipped, everything else still runs.

## The interface

Bus `com.swedishembedded.Brain1`, object `/com/swedishembedded/Brain1`,
interface `com.swedishembedded.Brain1.Manager` (plus the standard
`Introspectable` / `Properties` / `Peer`):

| Member | Signature | Purpose |
|---|---|---|
| `Manifests()` | `→ s` | JSON of every model/action/param (discovery) |
| `ListModels()` | `→ as` | served model names |
| `Run(model, action, params, in_fds, in_meta, transport)` | `sssa{sh}ss → sa{sh}s` | one-shot: `(result_json, out_fds, out_meta_json)` |
| `Subscribe(model, action, params, in_fds, in_meta)` | `sssa{sh}s → th` | streaming: `(job, event_fd)` |
| `Cancel(job)` | `t → b` | cooperative cancel of a `Subscribe` job (`true` iff found in flight) |
| props | `Version s`, `ActiveJobs u` (in-flight `Run`/`Subscribe` jobs), `Models as` | |

- `params` / `in_meta` / `out_meta` are JSON strings.
- `in_fds` / `out_fds` are `a{sh}` - a map from **blob name** to a Unix fd.
- `in_meta` describes each input fd: `{"image": {"media":"image","w":512,"h":512,"c":3}}`.
  `media` is one of `image|mask|audio|video|text|bytes`; the whole object is passed
  to the action as the blob's metadata. `video` is a whole clip in one blob:
  N interleaved-HWC f32 RGB frames concatenated, meta `{"frames","w","h","c"}`
  (plus `fps` on a generated one).
- `transport` is `"memfd"` (default) or `"dmabuf"` (best-effort - falls back to
  memfd where no DMA-heap is available; the actual choice is reported in `out_meta`).

### `Run` - one-shot

Input blobs arrive as fds (mmap-read into the action); output blobs come back as
fds (a sealed memfd per output). `out_meta` records each output's `media`,
`transport`, `bytes`, and blob `meta` (e.g. `{w,h,c}` for an image).

### `Subscribe` - streaming

Returns a `SOCK_SEQPACKET` fd. Each datagram is one JSON frame; a `blob` frame
carries its payload as an out-of-band memfd via `SCM_RIGHTS`:

```json
{"type":"progress","step":3,"total":10,"message":"sampling"}
{"type":"blob","name":"image","media":"image","meta":{"w":256,"h":256,"c":3}}   + fd
{"type":"done","result":{"width":256,"height":256}}
{"type":"error","message":"..."}
```

Sends are non-blocking: a slow subscriber's frames are dropped rather than stalling
inference (`SEQPACKET` preserves message boundaries).

### `Cancel` - cooperative cancellation

`Cancel(job)` takes the job id `Subscribe` returned and flips the cancel token the
server armed in that job's invocation. A long-running action polls the token between
steps (denoising steps, training steps) and aborts with `"cancelled"`, which arrives
as the stream's terminal `error` frame. Returns `true` if the job was still in
flight, `false` for an unknown or already-finished id. Cancellation is cooperative:
an action that never polls (or a step already on the GPU) finishes its current step
first. Python: `BrainDBus.cancel(job)`.

## More D-Bus samples

- **`samples/shell/dbus/busctl-smoke/`** - validate the raw surface with
  systemd's `busctl` (introspect, properties, `ListModels`, `Manifests`, and a
  FD-returning `Run demo.echo`), no Python needed.
- **`samples/python/dbus/detect-pipeline/`** - a full multi-model pipeline
  (generate → detect → draw) built on the same client this directory ships.

## Design notes

- **Separation of concerns**: all D-Bus/async code lives in `crates/dbus`
  (`brain-dbus`), which depends only on `capability`. The CLI builds the registry and
  hands it to `brain_dbus::serve`; no model code knows about D-Bus.
- **No inference on the bus thread**: a dedicated worker thread owns the registry and
  runs the blocking `Registry::run`; D-Bus methods only validate, enqueue, and reply.
  One worker => jobs serialize, which is correct for a single-GPU engine.
- **Automated test**: `crates/dbus/tests/roundtrip.rs` (run under `dbus-run-session
  -- cargo test -p brain-dbus --test roundtrip`) round-trips a result
  and an input both through fds.
