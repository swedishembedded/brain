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
- `--strength` - for image-to-image editing, how much the output may drift
  from the reference: low values (around 0.1) preserve structure/texture
  closely, high values (around 0.9) allow more freedom. It does not add
  color/hue changes on its own.

For edits, short imperative instructions that name the change
(`"Colorize this photograph."`, `"Make it snow."`) work much better than a
descriptive scene prompt - the model treats a description as a
text-to-image prompt rather than an edit instruction. If you need a strong
color or hue change with guaranteed structural fidelity, use the
undistilled `base` variant (which supports CFG) or train a LoRA for it.

## Hardware and limits

INT8 precision lowers resident VRAM use for the transformer. Klein does not
use classifier-free guidance, so it cannot be steered as strongly as the
`base` variants for edits that need to fight the source image's content - use
`base-4b`/`base-9b` or a LoRA for those.

Licensing: Klein 4B, the FLUX.2 VAE, and the Qwen3 text encoders are Apache
2.0 (commercial use OK). The 9B variants (`klein-9b`, `base-9b`) are under the
FLUX Non-Commercial License - research/non-commercial use only, and require
setting `BRAIN_FLUX2_ALLOW_NC=1` to opt in.
