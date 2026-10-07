# sample: imagegen/flux2-klein (D-Bus)

Three Python clients that drive brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, protocol in `samples/python/dbus/brain-dbus/README.md`)
against the `flux2-klein` model: streaming text-to-image, reference-image
editing, and cooperative cancellation. Images travel as **file descriptors**
(sealed memfd), never as bytes marshalled through D-Bus; progress arrives per
denoise step over the `Subscribe` SEQPACKET stream, so a multi-minute
generation is never a black box.

LoRA fine-tuning is not among them: `lora_train` reads a dataset folder and
writes an adapter file on the serving machine, and the bus accepts no param
that names a file there. Train on the machine that holds the dataset with
`samples/shell/imagegen/lora/train_lora.sh` (or `brain do`).

```
generate.py / edit_image.py / cancel_generation.py
        | jeepney (session bus, fd passing)          brain-py/brain_py/dbus.py
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames (per step)
        v                                      | blob frame + memfd (image)
residency executor --instance key--> resident flux2::Pipeline (DiT+TE+VAE)
        |                                      ^
        Cancel(job) --> cancel token --> polled per denoise step
```

## Run

```bash
# deps: pip install -e brain-py   (jeepney with fd passing)
BRAIN_FLUX2_DIT=/path/to/flux2-klein/transformer \
BRAIN_FLUX2_VAE=/path/to/flux2-klein/vae \
BRAIN_FLUX2_TE=/path/to/Qwen3-4B \
BRAIN_FLUX2_TOKENIZER=/path/to/Qwen3-4B/tokenizer.json \
tools/dbus-session.sh --serve "--dbus" -- bash -c '
  # text-to-image (progress = one line per denoise step)
  python3 samples/python/imagegen/flux2-klein/generate.py --prompt "a red fox in the snow" --out fox.ppm

  # editing: the input PPM rides as a memfd blob; extra refs via --ref
  python3 samples/python/imagegen/flux2-klein/edit_image.py --image fox.ppm \
      --prompt "the same scene at night" --out fox_night.ppm

  # cancel a running generation after the 2nd progress frame
  python3 samples/python/imagegen/flux2-klein/cancel_generation.py'
```

## Expected output

```
text2image 512x512 (klein-4b):
  [0/6] encoding prompt
  [1/6] denoising
  ...
  [6/6] decoding
  done: {'width': 512, 'height': 512} (410.3s)
wrote fox.ppm (512x512)
```

`cancel_generation.py` ends with `error frame: 'cancelled' (expected)`.

## What it needs

- **Weights** come only from the four `BRAIN_FLUX2_*` env vars (diffusers
  `transformer/`+`vae/` dirs, BFL single-file safetensors, or BF16 GGUF for
  the DiT). Without them the model is discoverable (`brain caps brain/flux2-klein`,
  static manifest) but not served.
- **Variants**: `klein-4b` (default, 4-step distilled), `klein-9b`,
  `base-4b`/`base-9b` (50 steps + CFG via `guidance`). The **9B weights are
  NC-licensed** (FLUX.2 [Non-Commercial] License, Black Forest Labs): the
  server refuses them unless `BRAIN_FLUX2_ALLOW_NC=1` is set, and prints the
  attribution notice once when enabled.
- `pip install -e brain-py` (jeepney with fd passing).

## What it demonstrates

- `run_streaming()` (in `generate.py`, reused by `edit_image.py`) is the
  standard progress-frame loop: `on_progress` per denoise step, a
  `blob` frame with the finished artifact, `error` on failure.
- `edit_image.py` shows the multi-reference input path (`--ref`, up to 3) and
  that a reference is just another memfd blob.
- `cancel_generation.py` shows the low-level `stream_frames_with_job` API
  (needed to get the job id mid-stream) and cooperative cancellation: the
  current step finishes before the `cancelled` error frame arrives.

## Notes

- Width/height must be multiples of 16; reference images are center-cropped
  to /16 server-side.
- Generation time is hardware-dependent (a 512x512 klein-4b run is roughly
  15+ min on CPU); cancellation is cooperative, not immediate.
- Same-key concurrent requests are grouped by the scheduler but execute
  sequentially for now (`resident_flux2.rs::run_batch`'s module docs).
- The CLI twin of these scripts: `brain flux2 generate --prompt ... --out out.ppm`.
- All three also run against the weight-free mock model (`--model brain/mock`);
  `cancel_generation.py`'s own header explains the `BRAIN_MOCK_DELAY_MS` knob
  needed to make that timing-sensitive.
