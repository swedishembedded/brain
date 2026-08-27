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

## A person in a target pose (`portrait_from_refs.sh`)

The only example here that is not a D-Bus client: a short shell wrapper around
`brain flux2 generate` that takes **one folder** and produces a portrait.

```bash
BRAIN_FLUX2_DIT=… BRAIN_FLUX2_VAE=… BRAIN_FLUX2_TE=… BRAIN_FLUX2_TOKENIZER=… \
  examples/imagegen/portrait_from_refs.sh ~/photos/alice
```

No device variables: with none given, brain places the DiT, the text encoder
and the VAE itself - on two cards when the machine has two and they do not fit
one, on one card when it has one - and prints the placement it chose.
`BRAIN_DEVICE` / `BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8]` still override it.

The folder holds numbered photographs of the person (`ref-01.jpeg`, `02.png`,
`3.webp` - any format brain decodes) plus one `target.*`, the photograph whose
pose you want; the script globs both. Unprepared camera photographs are the
expected input: `--ref-size` bounds each reference before it is tokenized, so
you never have to work out a token budget by hand.

The target is passed as a reference by default - that is what carries its pose
and framing across literally, at the cost of pulling some of the target
person's face along with it. `TARGET_REF=0` drops it and takes the pose from
the `POSE` prompt text instead, which keeps identity coming from the numbered
references alone. Read the script's header for the rest of the knobs; there
are only a handful and they are all environment variables with defaults.

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

---

## SDXL + ControlNet (`brain sdxl-controlnet text2image`)

`controlnet_generate.py` adds a conditioning image (edge map, depth map,
pose, ...) to the same SDXL loop: `crates/controlnet/src/caps.rs` builds the
backbone with `Unet::new_controlled` instead of `Unet::new` and runs the
`ControlNet` once per denoising step, threading its residuals in via
`Unet::run_with_control` (`crates/controlnet/src/adapter.rs`'s
`ControlAdapter`/`ControlSource` seam - backbone-agnostic by design, so a
FLUX ControlNet would plug into the same seam without touching it).

```bash
BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \
BRAIN_CONTROLNET_DIR=/path/to/controlnet-canny-sdxl-1.0 \
dbus-run-session -- bash -c '
  ./target/release/brain serve --dbus & sleep 2
  python3 examples/imagegen/controlnet_generate.py \
    --prompt "a red fox in the snow" --control canny_edges.ppm'
```

The conditioning image is resized on the device to the output size, so it
need not be pre-sized to match `--width`/`--height`.

---

## FLUX.1 (`brain flux1 text2image`)

`flux1_generate.py` drives FLUX.1's `text2image` action - T5-XXL context +
CLIP-L pooled conditioning, `crates/flux1/src/pipeline.rs`'s own rectified-flow
schedule (BFL's linear `calculate_shift`, not FLUX.2 Klein's empirical fit -
the two use different constants and the module docs explain why reusing
Klein's would be silently wrong), 16-channel VAE decode. `dev`/`kontext-dev`
are guidance-distilled (`--guidance`); `schnell` is timestep-distilled and
ignores it.

```bash
BRAIN_FLUX1_DIR=/path/to/FLUX.1-dev dbus-run-session -- bash -c '
  ./target/release/brain serve --dbus & sleep 2
  python3 examples/imagegen/flux1_generate.py --prompt "a red fox in the snow"'
```

**Scope**: text-to-image only - no Kontext reference-image editing, img2img,
or LoRA yet (`flux2::pipeline` is the fuller reference for what each needs
when they land here). No batching, same reasoning as plain SDXL.

**On verification**: every piece this composes (the DiT forward, the T5/CLIP
towers, the VAE) is independently parity-gated elsewhere in this workspace.
The pipeline glue - patchify layout, position ids, the schedule, the affine
latent normalization - has not been run against a real FLUX.1 checkpoint in
the environment that wrote it; there is no fixture here to verify it end to
end. Treat a first real generation as the actual test of this file.

---

## PuLID identity conditioning (`brain flux1-pulid text2image`)

`pulid_generate.py` adds a face photo to the FLUX.1 loop: ArcFace (raw
embedding) + EVA-CLIP-L/336 (CLS + 5 tapped hidden states) compose into
`id_cond`, `crate::model::IdFormer` projects 32 ID tokens, and
`crate::adapter::PulidAdapter` cross-attends them into the DiT at 20 points
through `flux1::pipeline::Flux1::generate_injected` -
`crates/pulid/src/caps.rs`'s module docs are the full account of what
composes what, including the one real preprocessing gap (a plain resize
where the reference uses face-parsing alignment brain does not have).

```bash
BRAIN_FLUX1_DIR=/path/to/FLUX.1-dev \
BRAIN_PULID_DIR=/path/to/pulid_flux_v0.9.1.safetensors \
BRAIN_ARCFACE_DIR=/path/to/antelopev2 \
BRAIN_CLIP_DIR=/path/to/eva-clip-dir \
dbus-run-session -- bash -c '
  ./target/release/brain serve --dbus & sleep 2
  python3 examples/imagegen/pulid_generate.py \
    --prompt "a photo of a person hiking in the mountains" --face portrait.ppm'
```

Only `dev` is validated against a PuLID reference (the reference is built on
FLUX.1-dev, not Kontext or schnell). Same scope/verification caveats as
plain FLUX.1, doubled: this also has no end-to-end fixture for the
ID-conditioning wiring itself.

---

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements image-generation pipelines that run on hardware
you own, for teams that need generation to be private, unmetered, or offline.
If your team needs expertise in diffusion models, LoRA fine-tuning, or fitting
a generative model onto the card you actually have, you can procure our
services by sending an email to **info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
