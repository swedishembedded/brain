# sample: imagegen/flux1 (D-Bus)

`flux1_generate.py` drives FLUX.1's `text2image` action - T5-XXL context +
CLIP-L pooled conditioning, `crates/flux1/src/pipeline.rs`'s own rectified-flow
schedule (BFL's linear `calculate_shift`, not FLUX.2 Klein's empirical fit -
the two use different constants and the module docs explain why reusing
Klein's would be silently wrong), 16-channel VAE decode. `dev`/`kontext-dev`
are guidance-distilled (`--guidance`); `schnell` is timestep-distilled and
ignores it. Like SDXL, `text2image` here is a plain `Run` call - no per-step
progress hook yet.

```bash
BRAIN_FLUX1_DIR=/path/to/FLUX.1-dev \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/imagegen/flux1/flux1_generate.py --prompt "a red fox in the snow"
```

## What it demonstrates

- The plain `Run` call path against a third distinct backbone
  (`brain/flux1`), reusing the same `BrainDBus` client as SDXL and
  ControlNet.
- `--variant` selecting between `dev`, `kontext-dev` and `schnell`, and how
  `--guidance` is ignored for the timestep-distilled `schnell`.

## What it needs

`BRAIN_FLUX1_DIR` is a released diffusers FLUX.1 checkpoint root
(`transformer/`, `vae/`, `text_encoder/`+`tokenizer/` for CLIP-L,
`text_encoder_2/`+`tokenizer_2/` for T5-XXL).

`pip install -e brain-py` (jeepney with fd passing).

## Options

| flag | default |
|---|---|
| `--prompt TEXT` | required |
| `--variant` | `dev` (`dev`\|`kontext-dev`\|`schnell`) |
| `--out PATH` | `flux1.ppm` |
| `--width N` | `1024` (multiple of 16) |
| `--height N` | `1024` (multiple of 16) |
| `--steps N` | `0` (variant default) |
| `--guidance F` | `3.5` (`dev`/`kontext-dev` only) |
| `--max-len N` | `512` (T5-XXL context length) |
| `--seed N` | `0` |

**Scope**: text-to-image only - no Kontext reference-image editing, img2img,
or LoRA yet (`../flux2-klein/`'s pipeline is the fuller reference for what
each needs when they land here). No batching, same reasoning as plain SDXL.

**On verification**: every piece this composes (the DiT forward, the T5/CLIP
towers, the VAE) is independently parity-gated elsewhere in this workspace.
The pipeline glue - patchify layout, position ids, the schedule, the affine
latent normalization - has not been run against a real FLUX.1 checkpoint in
the environment that wrote it; there is no fixture here to verify it end to
end. Treat a first real generation as the actual test of this sample.
