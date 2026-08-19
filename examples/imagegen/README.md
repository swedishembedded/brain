# Image generation over D-Bus (FLUX.2 Klein)

These scripts drive brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, protocol in `examples/dbus/README.md`) against
the `flux2-klein` model: streaming text-to-image, reference-image editing,
LoRA fine-tuning with a live loss feed, and cooperative cancellation. Images
and adapters travel as **file descriptors** (sealed memfd), never as bytes
marshalled through D-Bus; progress arrives per denoise / training step over
the `Subscribe` SEQPACKET stream, so a multi-minute generation is never a
black box.

```
generate.py / edit_image.py / lora_finetune.py / cancel_generation.py
        | jeepney (session bus, fd passing)          brain-py/brain_py/dbus.py
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames (per step)
        v                                      | blob frame + memfd (image/adapter)
residency executor --instance key--> resident flux2::Pipeline (DiT+TE+VAE)
        |                                      ^
        Cancel(job) --> cancel token --> polled per denoise/training step
```

## Run

```bash
# deps: pip install -e brain-py   (jeepney with fd passing)
dbus-run-session -- bash -c '
  BRAIN_FLUX2_DIT=/path/to/flux2-klein/transformer \
  BRAIN_FLUX2_VAE=/path/to/flux2-klein/vae \
  BRAIN_FLUX2_TE=/path/to/Qwen3-4B \
  BRAIN_FLUX2_TOKENIZER=/path/to/Qwen3-4B/tokenizer.json \
  ./target/release/brain serve --dbus & sleep 2

  # text-to-image (progress = one line per denoise step)
  python3 examples/imagegen/generate.py --prompt "a red fox in the snow" --out fox.ppm

  # editing: the input PPM rides as a memfd blob; extra refs via --ref
  python3 examples/imagegen/edit_image.py --image fox.ppm \
      --prompt "the same scene at night" --out fox_night.ppm

  # LoRA fine-tune (server-side dataset), live loss, adapter back as an fd,
  # then a test generation with the adapter applied
  python3 examples/imagegen/lora_finetune.py --data /srv/dataset \
      --save out/my.lora --steps 200 --prompt "a photo of sks dog"

  # cancel a running generation after the 2nd progress frame
  python3 examples/imagegen/cancel_generation.py'
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

`lora_finetune.py` streams `step 3/200  loss 0.41231  (95.2 s)` lines;
`cancel_generation.py` ends with `error frame: 'cancelled' (expected)`.

## Notes

- **Weights** come only from the four `BRAIN_FLUX2_*` env vars (diffusers
  `transformer/`+`vae/` dirs, BFL single-file safetensors, or BF16 GGUF for
  the DiT). Without them the model is discoverable (`brain caps brain/flux2-klein`,
  static manifest) but not served.
- **Variants**: `klein-4b` (default, 4-step distilled), `klein-9b`,
  `base-4b`/`base-9b` (50 steps + CFG via `guidance`). The **9B weights are
  NC-licensed** (FLUX.2 [Non-Commercial] License, Black Forest Labs): the
  server refuses them unless `BRAIN_FLUX2_ALLOW_NC=1` is set, and prints the
  attribution notice once when enabled.
- Width/height must be multiples of 16; reference images are center-cropped
  to /16 server-side.
- Generation time is hardware-dependent (a 512×512 klein-4b run is ≈15+ min
  on CPU); cancellation is cooperative — the current step finishes before the
  `cancelled` error frame arrives.
- Same-key concurrent requests are grouped by the scheduler but execute
  sequentially for now (documented in `resident_flux2.rs::run_batch`; a true
  batched DiT forward is a planned follow-up).
- The CLI twin of these scripts: `brain flux2 generate --prompt … --out out.ppm`.

---

## SDXL (`brain sdxl text2image`)

`sdxl_generate.py` drives the SDXL `text2image` action - dual CLIP-L/OpenCLIP-bigG
conditioning, a discrete Euler scheduler, and classifier-free guidance
(`crates/sdxlunet/src/pipeline.rs`). Unlike FLUX.2's action, this one is a
plain `Run` (no per-step progress hook yet), so it is one blocking call:

```bash
BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 dbus-run-session -- bash -c '
  ./target/release/brain serve --dbus & sleep 2
  python3 examples/imagegen/sdxl_generate.py --prompt "a red fox in the snow"'
```

`BRAIN_SDXL_DIR` is a released diffusers SDXL checkpoint root
(`unet/`, `vae/`, `text_encoder/`, `text_encoder_2/`, `tokenizer/`,
`tokenizer_2/`). No batching: every request runs its own denoising loop
(`resident_sdxl.rs`'s module docs explain why grouping would not help).
