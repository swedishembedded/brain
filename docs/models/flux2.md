# FLUX.2 Klein

FLUX.2 Klein is a text-to-image and reference-image editing model: give it a
prompt to generate an image from scratch, or a prompt plus one or more
reference images to edit them in place (recolor a photo, add weather, change
style, and similar edits). The Klein variant is a fast, distilled variant
requiring only a few generation steps, trading the slower full ("base")
model's extra flexibility for speed.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [x] |
| INT8                   | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [x] |

## Getting the weights

Model id: `brain/flux2-klein`. Weights are not auto-fetched - point brain at
a local diffusers-layout checkout:

```bash
export BRAIN_FLUX2_DIT=…/FLUX.2-klein-4B/transformer
export BRAIN_FLUX2_VAE=…/FLUX.2-klein-4B/vae
export BRAIN_FLUX2_TE=…/FLUX.2-klein-4B/text_encoder
export BRAIN_FLUX2_TOKENIZER=…/FLUX.2-klein-4B/tokenizer/tokenizer.json
```

## Running it

```bash
brain flux2 generate --prompt "a red fox on a mossy rock" --out fox.ppm \
    --width 512 --height 512 --seed 7

# reference-image editing: pass the source image as --ref and an instruction as the prompt
brain flux2 generate --prompt "make it snow" --ref fox.ppm --out snow.ppm
```

The same model is also reachable through the generalized capability
interface's `text2image`/`edit`/`lora_train` actions (`brain caps
brain/flux2-klein` lists them) over D-Bus and over HTTP at
`/v1/images/generations` - those three are not CLI-reachable today, only
`generate`/`infer` above are.

## Options

- `--width` / `--height`
- `--steps`, `--seed`
- `--guidance` - classifier-free guidance strength; only meaningful on the
  undistilled `base` variants, since Klein is guidance-distilled
- `--variant klein-4b|klein-9b|base-4b|base-9b`
- `--precision fp32|int8` - supports INT8 inference for lower VRAM use
- `--ref <image>` (repeatable) - supplying one or more reference images
  switches generation into editing mode
- `--adapter <path>` - fold a LoRA adapter into the DiT before generating.
  Two families are accepted, told apart by extension: brain's own `finetune`
  checkpoint, or a third-party `.safetensors` adapter in the ai-toolkit /
  ComfyUI / diffusers convention (`diffusion_model.<module>.lora_A/B.weight`).
  The adapter must have been trained for the `--variant` you select; a key
  that does not match a tensor of that variant is a hard error naming the
  tensor, never a silent skip.
- `--lora-scale <S>` - LoRA strength, ComfyUI's `strength_model`. Default
  `1.0`. Third-party adapter files usually carry no alpha, in which case both
  ai-toolkit and ComfyUI resolve the alpha multiplier to exactly 1.0 and this
  flag is the only dial; `0.0` reproduces the base model, which is the way to
  check what the adapter is actually contributing.
- `--strength` - for image-to-image editing, how much the output may drift
  from the reference: low values (around 0.1) preserve structure/texture
  closely, high values (around 0.9) allow more freedom, and `1.0` starts the
  denoise from pure noise. It does not add color/hue changes on its own.
  It controls **how much denoising starts from the init latent, not whether
  the model can see the reference**: a supplied reference contributes
  conditioning tokens at every value of `--strength`. Below `1.0` the first
  reference does double duty - it is the init latent *and* it is attended to.
- `--ref-cond-scale <S>` - linear size of the **conditioning copy** of that
  init reference, in `0..=1`, default `0.75`. Only the first `--ref` under
  `--strength < 1` is affected: the init-latent role pins it to the output
  size, so it is the one reference whose resolution you cannot choose by
  picking a different file. Reference tokens cost joint attention
  quadratically, so a full-size copy of a same-size reference doubles the
  image half of the sequence; the default downscale keeps an edit at roughly
  the cost of a moderate second reference. `1.0` conditions at full size -
  identical token cost to `--strength 1.0`. `0` switches the conditioning
  copy off, which is the explicit opt-in to the cheap mode where the
  reference reaches the denoiser only through the init latent.
- `--mask <image>` - a spatial preservation mask over the output canvas.
  **White regenerates, black preserves, greys blend.** See
  [Masked editing](#masked-editing-blended-latent-diffusion) below.

For edits, short imperative instructions that name the change
(`"Colorize this photograph."`, `"Make it snow."`) work much better than a
descriptive scene prompt - the model treats a description as a
text-to-image prompt rather than an edit instruction. If you need a strong
color or hue change with guaranteed structural fidelity, use the
undistilled `base` variant (which supports CFG) or train a LoRA for it.

## Masked editing (blended latent diffusion)

`--strength` is a **global** dial. Some edits need a **spatial** one: redraw
the middle of a room, keep the walls and the windows. Virtual staging is the
motivating case - there is no single `--strength` that both replaces the
furniture and leaves the architecture where it was, because the two demands
pull the same knob in opposite directions.

`--mask <image>` takes a greyscale image over the **output canvas**:

* **white (255) = regenerate** - the denoiser owns these pixels;
* **black (0) = preserve** - these track the first `--ref` image exactly;
* **grey = blend** - the value is used verbatim as a linear weight, so
  mid-grey is an even mix of source and generation. There is no threshold, and
  soft edges are the point: a hard latent-cell boundary between "kept" and
  "redrawn" shows up as a seam.

After every Euler step the masked-out region is replaced by the source latent
renoised to that step's own sigma:

```text
x = m·x_denoised + (1 − m)·((1 − σ)·x₀ + σ·ε)
```

`(1 − σ)·x₀ + σ·ε` is the rectified-flow forward process - the same one
`--strength` uses to build its init latent - so the preserved region is always
a legal point on the source's own trajectory rather than an out-of-distribution
paste, and at the terminal sigma it is the source latent exactly. That is the
difference from `--strength`: preserved regions *track* the source at every
step instead of being softly guided toward it and drifting a little with each
forward.

```bash
brain flux2 generate --variant klein-9b --precision int8 \
    --prompt "..." --ref room.png --mask arch-mask.png \
    --strength 0.999 --width 1024 --height 768 --steps 12 --out staged.png
```

### Rules the implementation guarantees

* The first `--ref` is the preserved source and must be at the output size -
  the same rule `--strength` already imposes. `--mask` does **not** consume
  it and does not change the token budget either way: that reference
  contributes conditioning tokens regardless, at the size
  `--ref-cond-scale` picks.
* An **all-white** mask is bit-for-bit identical to no mask at all, and an
  **all-black** one reproduces the source latent exactly. Both are asserted,
  and both are exact rather than approximate - the mask resampler accumulates
  integer area overlaps so that a mask which is constant over a latent cell
  resamples to exactly that constant.
* The mask is resampled to the latent grid by an exact **area average**
  (box filter), each axis independently, so a non-square canvas is not a
  special case. It may be supplied at any resolution.
* One weight per latent **token** (a 16x16 pixel block), applied to all 128
  latent channels. That is the finest granularity that exists - latent
  channels are not spatial - so a mask edge is quantised to 16 pixels before
  the VAE decoder's own receptive field smears it a little further. Preserved
  regions are exact in *latent* space; in *pixel* space the guarantee softens
  within a few pixels of a mask boundary.

### Producing a mask

There is no automatic mask generator in brain, and the honest reason is that
the obvious ones do not work. A monocular-depth mask (`brain zipdepth --view
depth --colormap gray --headless`, threshold the near field) segments a bedroom
where the bed is the only near object, but on a living room it marks the
*ceiling* as foreground - the ceiling directly above the camera is genuinely
the nearest surface in the frame - while leaving a sofa against the far wall
as background. "Near" is not "furniture". A depth top-hat (opening by
reconstruction) fixes the ceiling but then misses any object larger than the
structuring element, which for a bed filling half the frame is all of it.

Two things make the segmentation genuinely hard, and both are worth knowing
before trusting any automatic mask:

1. Staging must **add** furniture where there is none - a rug on bare floor, a
   plant in an empty corner. A mask that covers only the existing furniture
   cannot stage. What has to be described is the enclosing shell to keep, not
   the objects to remove.
2. The floor is simultaneously architecture (its perspective and material must
   not drift) and the place new objects go. That is what the grey levels are
   for: a mid-weight floor band lets objects appear while keeping the plane
   half-anchored.

So today a mask is authored, not inferred. Any greyscale image works - a paint
program, a few polygons rasterised over the frame, or a segmentation model's
output thresholded by hand. A working recipe: mark the interior volume white
(existing furniture plus the space new furniture must occupy), leave the
ceiling, walls, windows and doors black, put the floor band at ~0.45 grey,
and feather the whole thing with a few pixels of blur so the seams blend.

## Hardware and limits

INT8 precision lowers resident VRAM use for the transformer. Klein does not
use classifier-free guidance, so it cannot be steered as strongly as the
`base` variants for edits that need to fight the source image's content - use
`base-4b`/`base-9b` or a LoRA for those.

Licensing: Klein 4B, the FLUX.2 VAE, and the Qwen3 text encoders are Apache
2.0 (commercial use OK). The 9B variants (`klein-9b`, `base-9b`) are under the
FLUX Non-Commercial License - research/non-commercial use only, and require
setting `BRAIN_FLUX2_ALLOW_NC=1` to opt in.
