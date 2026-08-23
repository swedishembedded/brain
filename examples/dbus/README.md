# brain over D-Bus (`com.swedishembedded.Brain1`)

A D-Bus control surface (compiled into the default build, opt-in at runtime via
`--dbus`) lets local Linux apps use brain as a service: discover models, run
actions, and exchange images / streams / results as **file descriptors** (memfd/mmap,
and dmabuf where the kernel supports it) instead of bytes marshalled through D-Bus.

It's a thin front-end over brain's `capability::Registry` — the same models and
actions as the CLI (`brain <arch> <verb>`) or `brain serve --stdio`, now
reachable over the bus.

## Enable & run

Compiled into the default build; the service only runs when you pass `--dbus`:

```bash
cargo build -p brain-cli
brain serve                        # stdio JSONL loop (no D-Bus)
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
`com.swedishembedded.Brain1.Manager` (plus the standard `Introspectable` /
`Properties` / `Peer`):

| Member | Signature | Purpose |
|---|---|---|
| `Manifests()` | `→ s` | JSON of every model/action/param (discovery) |
| `ListModels()` | `→ as` | served model names |
| `Run(model, action, params, in_fds, in_meta, transport)` | `sssa{sh}ss → sa{sh}s` | one-shot: `(result_json, out_fds, out_meta_json)` |
| `Subscribe(model, action, params, in_fds, in_meta)` | `sssa{sh}s → th` | streaming: `(job, event_fd)` |
| `Cancel(job)` | `t → b` | cooperative cancel of a `Subscribe` job (`true` iff found in flight) |
| props | `Version s`, `ActiveJobs u` (in-flight `Run`/`Subscribe` jobs), `Models as` | |

- `params` / `in_meta` / `out_meta` are JSON strings.
- `in_fds` / `out_fds` are `a{sh}` — a map from **blob name** to a Unix fd.
- `in_meta` describes each input fd: `{"image": {"media":"image","w":512,"h":512,"c":3}}`.
  `media` is one of `image|mask|audio|video|text|bytes`; the whole object is passed
  to the action as the blob's metadata. `video` is a whole clip in one blob:
  N interleaved-HWC f32 RGB frames concatenated, meta `{"frames","w","h","c"}`
  (plus `fps` on a generated one) - see `examples/videogen/`.
- `transport` is `"memfd"` (default) or `"dmabuf"` (best-effort — falls back to
  memfd where no DMA-heap is available; the actual choice is reported in `out_meta`).

### `Run` — one-shot

Input blobs arrive as fds (mmap-read into the action); output blobs come back as
fds (a sealed memfd per output). `out_meta` records each output's `media`,
`transport`, `bytes`, and blob `meta` (e.g. `{w,h,c}` for an image).

### `Subscribe` — streaming

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

### `Cancel` — cooperative cancellation

`Cancel(job)` takes the job id `Subscribe` returned and flips the cancel token the
server armed in that job's invocation. A long-running action polls the token between
steps (denoising steps, training steps) and aborts with `"cancelled"`, which arrives
as the stream's terminal `error` frame. Returns `true` if the job was still in
flight, `false` for an unknown or already-finished id. Cancellation is cooperative:
an action that never polls (or a step already on the GPU) finishes its current step
first. Python: `BrainDBus.cancel(job)`.

## Examples in this directory

- **`busctl_smoke.sh`** — validate the surface with systemd's `busctl` (introspect,
  properties, `ListModels`, `Manifests`, and a FD-returning `Run demo.echo`):

  ```bash
  dbus-run-session -- bash examples/dbus/busctl_smoke.sh target/debug/brain
  ```

- The reusable client lives in the **brain-py** package: `brain_py.dbus.BrainDBus`
  (context manager + `read_fd`/`sealed_memfd`; jeepney with `enable_fds=True`, install
  with `pip install -e brain-py`) and `brain_py.image` (PPM save + box drawing for
  brain's HWC-f32 image blobs, no third-party image lib).
- **`brain_dbus.py`** — an example using the client: discovery, `imageops.gradient`
  (a real image via fd → PPM), and - with `BRAIN_S3DIT_*` exported - a streaming
  `z-image text2image`:

  ```bash
  dbus-run-session -- bash -c 'brain serve --dbus & sleep 1; python3 examples/dbus/brain_dbus.py'
  ```

## Design notes

- **Separation of concerns**: all D-Bus/async code lives in `crates/dbus`
  (`brain-dbus`), which depends only on `capability`. The CLI builds the registry and
  hands it to `brain_dbus::serve`; no model code knows about D-Bus.
- **No inference on the bus thread**: a dedicated worker thread owns the registry and
  runs the blocking `Registry::run`; D-Bus methods only validate, enqueue, and reply.
  One worker ⇒ jobs serialize, which is correct for a single-GPU engine.
- **Automated test**: `crates/dbus/tests/roundtrip.rs` (run under `dbus-run-session
  -- cargo test -p brain-dbus --test roundtrip`) round-trips a result
  and an input both through fds.

---

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements native Linux system integration for products
with AI inside them: bus services, zero-copy fd passing, and applications that
talk to a local model as an ordinary system service. If your team needs
expertise in wiring inference into a Linux system or an embedded device, you
can procure our services by sending an email to **info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
