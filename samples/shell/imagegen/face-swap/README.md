# sample: imagegen/face-swap

Put someone's face into someone else's photograph. A direct CLI wrapper (no
server) around `brain flux2 generate`, using the colocated helper
`head_mask.py` to derive a feathered head mask from face landmarks - it's an
internal helper for this sample, not a separate sample of its own.

```bash
BRAIN_SCRFD_DIR=... BRAIN_FLUX2_DIT=... BRAIN_FLUX2_VAE=... BRAIN_FLUX2_TE=... BRAIN_FLUX2_TOKENIZER=... \
  samples/shell/imagegen/face-swap/face_swap.sh <dir-of-images> [count]
```

The folder holds numbered images - `ref-01.jpg`, `ref-02.png`, `03.jpeg`, any
name that is (optionally `ref-`) a number, in any format brain decodes.
**The first one is the target**: the photograph whose pose, framing,
clothing, background and lighting are kept. Every one after it is a
photograph of the person whose face goes in. Results are written as
`result-01.jpg`...

## What it demonstrates

- **Why the target has to be first**: under `--strength` and under `--mask`,
  brain treats the first `--ref` as the init latent - it is VAE-encoded and
  the denoise starts from it, which is the only reason the pose survives at
  all. Every later reference is conditioning.
- **`--ref-cond-scale 0`**: the target contains a face - the one being
  replaced - so conditioning on it as well as seeding from it feeds the
  model the wrong identity, which then competes with the face references and
  wins (it's also what every pixel is pulled back toward). Measured on a
  deliberately hard swap, switching this off moved ArcFace identity from
  0.018 to 0.198 with pose preservation unchanged. Leave it on by mistake and
  the script appears to run correctly and quietly returns the original
  person.
- **The mask**: a `mask.*` in the folder is used as-is (white regenerates,
  black keeps, greys blend) - the escape hatch for a target the detector
  can't read. Otherwise `head_mask.py` derives one from SCRFD's five facial
  landmarks: an ellipse over the head, rotated to the face axis and
  feathered. `MASK_GROW` scales it; large enough to include the hairline
  matters, since whatever the mask does not cover is kept bit-for-bit.
  `MASK=0` switches to the plain `--strength` route instead: no seam, but
  pose is only approximately preserved and identity transfer measures
  weaker.
- `--adapter`/`ADAPTER` composes a per-identity LoRA
  (`../lora/train_identity_lora.sh`) with the face references - matches
  `../portrait-from-refs/portrait_from_refs.sh`'s convention.

Grade it, don't eyeball it - a face swap is exactly the case where the eye is
easiest to fool, since the picture is a real photograph almost everywhere:

```bash
samples/shell/imagegen/identity-score/identity_score.sh <dir> <dir>
```

## What it needs

`BRAIN_SCRFD_DIR` for the landmarks, `BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}` for
generation. References are sized by brain and the output takes the target's
own size, so pass photographs exactly as they are.

## Options

```
face_swap.sh <dir-of-images> [count] [--adapter P] [--lora-scale S] [--text-encoder P]
```

| flag / env | default | meaning |
|---|---|---|
| `--adapter` / `ADAPTER` | none | per-identity LoRA, composes with face refs |
| `--lora-scale` / `LORA_SCALE` | `0.5` | its strength |
| `--text-encoder` / `TEXT_ENCODER` | none | HF dir or single `.safetensors`/`.gguf` |
| `MASK` | `1` | `0` disables mask-based inpainting |
| `MASK_GROW` | `2.0` | derived-mask size dial |
| `STRENGTH` | `0.99` masked / `0.9` unmasked | denoise strength |
| `SEED` | `101` | first seed; each result adds 1 |
| `STEPS` | `12` | denoise steps |
| `PROMPT` | (built-in) | override the generation prompt |
| `VARIANT` | `klein-4b` | FLUX.2 variant |
| `PRECISION` | `int8` | DiT numeric tier |
| `BRAIN` | `./target/release/brain` | binary path |

Count defaults to 2 results. Every way of not having a mask (`MASK=0`, no
`mask.*` in the folder, no detector weights, no face found, no python) falls
back to the plain-`--strength` route rather than stopping the run.
