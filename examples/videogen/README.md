# Video generation over D-Bus (Wan2.1)

These scripts drive brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, protocol in `examples/dbus/README.md`) against
the `brain/wan` model: streaming text-to-video and cooperative cancellation.
The clip travels as a **file descriptor** (sealed memfd), never as bytes
marshalled through D-Bus; progress arrives per denoise step over the
`Subscribe` SEQPACKET stream, which for this model is the difference between a
visible run and an hour of silence.

```
generate_video.py / cancel_generation.py
        | jeepney (session bus, fd passing)          brain-py/brain_py/dbus.py
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames (per denoise step)
        v                                      | blob frame + memfd (the clip)
residency executor --instance key--> resident wan DiT (variant:frames:WxH)
        |                                      ^
        Cancel(job) --> cancel token --> polled per denoise step
```

The clip blob is `capability::Media::Video`: N interleaved-HWC f32 RGB frames
concatenated into one payload, meta `{frames, w, h, c, fps}`. That is the same
wire format `capability::blob::video_blob`/`decode_video` use everywhere else
in brain, so `generate_video.py` needs no video library to read it -- it
writes numbered PPMs and shells out to `ffmpeg` only to mux them.

## Run

```bash
# deps: pip install -e brain-py   (jeepney with fd passing)
dbus-run-session -- bash -c '
  BRAIN_WAN_DIT=/path/to/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  BRAIN_WAN_VAE=/path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  BRAIN_WAN_T5=/path/to/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
  BRAIN_WAN_TOKENIZER=/path/to/Wan2.1-T2V-1.3B/google/umt5-xxl \
  BRAIN_GPU_WAIT_S=1200 \
  ./target/release/brain serve --dbus & sleep 2

  # a clip you can watch finish: the scripts default to 9 frames at 416x240
  python3 examples/videogen/generate_video.py --prompt "a cat walking on a beach" \
      --out cat.mp4

  # cancel a running generation after the 2nd progress frame
  python3 examples/videogen/cancel_generation.py'
```

`BRAIN_GPU_WAIT_S` is not optional on a GPU: one Wan forward is the whole
30-block stack in a single submit, which at any real size is far past the
backend's 30 s deadlock guard. `brain wan t2v` raises it automatically;
`brain serve` does not, because the same process may be serving models for
which that guard is doing its job.

## Sizes, and what they cost

**These scripts do not default to the model's defaults, deliberately.** The
`t2v` action's own defaults are upstream's (81 frames at 832x480, 50 UniPC
steps at guidance 5.0), which measured **57.5 minutes** on a P40 -- text encode
246 s, DiT load 20 s, denoise 2308 s, VAE decode 876 s. An example that takes an
hour is an example nobody runs to the end, so the scripts pass 9 frames at
416x240 over 20 steps instead: the served run below reported **382.3 s**, and
`brain wan t2v` at the same size breaks that down as text encode 241 s, DiT load
19 s, denoise 81 s, VAE decode 41 s. **The text encode is most of it**, it runs
on the CPU, and it costs the same at every size - which is why shrinking further
buys very little. Pass `--frames 81 --width 832 --height 480 --steps 50` for the
real thing.

A step is TWO forwards whenever guidance > 1.0, and 81 frames at 480p is 32,760
tokens per forward against 1,170 at the scripts' default.

Only `--prompt` is required by the ACTION; every parameter the scripts do not
name defaults from the action's own schema, so they send nothing they were not
asked for.

## Expected output

Real output from the run above, trimmed in the middle:

```
t2v 'a cat walking on a beach':
  [0/23] text encode
  [1/23] load transformer
  [2/23] denoise t=999 4.8s/step, ~91s left
  ...
  [21/23] denoise t=208 4.1s/step, ~0s left
  [22/23] vae decode
  [23/23] done
  done: {'fps': 16, 'frames': 9, 'height': 240, 'seconds_per_forward': 2.0384888648986816, 'width': 416} (382.3s)
wrote cat.mp4 (416x240, 9 frames at 16 fps)
```

and, for the cancel path:

```
job 2 started; cancelling after the second progress frame
  [0/53] text encode
  [2/53] denoise t=999 4.2s/step, ~206s left
  Cancel(2) -> True
  [3/53] denoise t=995 4.1s/step, ~198s left
  error frame: 'cancelled' (expected)
```

Two things that output is showing. The `[1/53] load transformer` frame is
**missing** because the previous generation left the transformer resident at
the same instance key - that is residency working, and it is why the cancel
example is cheap to run second. And the abort lands one step LATE (`[3/53]`
arrives after `Cancel`) because the denoise loop polls the token once per step
and the forward already in flight is a single submit of the whole block stack,
which the host cannot interrupt. `/53` is the action's default step count: the
cancel example never sets `--steps`, because it never gets that far.

## The CLI, for comparison

The same generation without a server, one command, one playable file:

```bash
brain wan t2v --prompt "a cat walking on a beach" --seed 42 --output-path cat.mp4
```

With no `--dit/--vae/--t5/--tokenizer` and no `BRAIN_WAN_*` exported, that
command auto-fetches `Wan-AI/Wan2.1-T2V-1.3B` (17.6 GB) into
`$BRAIN_MODELS_DIR` first. Naming all four paths (or exporting all four
variables) skips the fetch entirely.
