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

  # a smoke-test-sized clip (seconds-to-minutes, not an hour)
  python3 examples/videogen/generate_video.py --prompt "a cat walking on a beach" \
      --frames 9 --width 256 --height 256 --steps 4 --out cat.mp4

  # cancel a running generation after the 2nd progress frame
  python3 examples/videogen/cancel_generation.py --frames 9 --width 256 --height 256'
```

`BRAIN_GPU_WAIT_S` is not optional on a GPU: one Wan forward is the whole
30-block stack in a single submit, which at any real size is far past the
backend's 30 s deadlock guard. `brain wan t2v` raises it automatically;
`brain serve` does not, because the same process may be serving models for
which that guard is doing its job.

## Sizes, and what they cost

The action's defaults are upstream's own (81 frames at 832x480, 50 UniPC steps
at guidance 5.0). Measured on a P40 that is **57.5 minutes** -- text encode
246 s, DiT load 20 s, denoise 2308 s, VAE decode 876 s. A step is TWO forwards
whenever guidance > 1.0, and 81 frames at 480p is 32,760 tokens per forward.
Start with `--frames 9 --width 256 --height 256 --steps 4` to check the whole
path, then scale up.

Only `--prompt` is required; every other parameter defaults from the action's
own schema, so the scripts send nothing they were not asked for.

## Expected output

```
t2v 'a cat walking on a beach':
  [0/7] text encode
  [1/7] load transformer
  [2/7] denoise t=999 12.4s/step, ~37s left
  ...
  [6/7] vae decode
  done: {'frames': 9, 'width': 256, 'height': 256, 'fps': 16, 'seconds_per_forward': 6.2} (94.1s)
wrote cat.mp4 (256x256, 9 frames at 16 fps)
```

and, for the cancel path:

```
job 1 started; cancelling after the second progress frame
  [0/7] text encode
  [1/7] load transformer
  Cancel(1) -> True
  error frame: 'cancelled' (expected)
```

## The CLI, for comparison

The same generation without a server, one command, one playable file:

```bash
brain wan t2v --prompt "a cat walking on a beach" --seed 42 --output-path cat.mp4
```

With no `--dit/--vae/--t5/--tokenizer` and no `BRAIN_WAN_*` exported, that
command auto-fetches `Wan-AI/Wan2.1-T2V-1.3B` (17.6 GB) into
`$BRAIN_MODELS_DIR` first. Naming all four paths (or exporting all four
variables) skips the fetch entirely.
