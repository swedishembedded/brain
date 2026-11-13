# sample: imagegen/portrait-from-refs

A person in a target pose, from a folder of their photographs. The only
imagegen sample that isn't a D-Bus client: a short shell wrapper around
`brain flux2 edit` and `brain flux2 text2image` (direct CLI invocation, no server involved).

```bash
samples/shell/imagegen/portrait-from-refs/portrait_from_refs.sh ~/photos/alice
```

No device variables: with none given, brain places the DiT, the text encoder
and the VAE itself - on two cards when the machine has two and they do not fit
one, on one card when it has one - and prints the placement it chose.
`DEVICE` (`gpu0`, `gpu1`, `cpu`; passed to brain as `--device`) restricts it.

## What it demonstrates

- The folder holds numbered photographs of the person (`ref-01.jpeg`,
  `02.png`, `3.webp` - any format brain decodes) plus one `target.*`, the
  photograph whose pose you want; the script globs both. Unprepared camera
  photographs are the expected input: `--ref-size` bounds each reference
  before it is tokenized, so you never have to work out a token budget by
  hand.
- `MASK=1` switches on inpainting instead: `target.*`'s WHITE region is the
  hole to fill, regenerated while everything else is preserved exactly
  (output takes the target's own size). Keeps the background and body
  bit-for-bit at the cost of a visible seam where the blend happens in latent
  space. To swap a face INTO another photograph, use
  `../face-swap/face_swap.sh` instead - it's built for that and masks the
  head for you.
- `ADAPTER` composes a per-identity LoRA (see `../lora/train_identity_lora.sh`)
  with the reference images: the adapter carries identity even in poses no
  reference shows, which references alone cannot do. `LORA_SCALE` (default
  0.5) dials its strength - start low, identity plateaus well before image
  quality gives out.
- Grade the result with a number, not an opinion:
  `../identity-score/identity_score.sh <dir> <dir>`.

## What it needs

Weights from the model store, or named with `--dit`, `--vae`,
`--text-encoder` and `--tokenizer`; brain picks a card with room unless
`DEVICE` says otherwise.

## Options

```
portrait_from_refs.sh <dir-of-images> [count] [--adapter P] [--lora-scale S] [--text-encoder P]
    [--dit P] [--vae P] [--tokenizer P]
```

| flag / env | default | meaning |
|---|---|---|
| `--adapter` / `ADAPTER` | none | per-identity LoRA path |
| `--lora-scale` / `LORA_SCALE` | `0.5` | adapter strength |
| `--text-encoder` / `TEXT_ENCODER` | none | HF dir or single `.safetensors`/`.gguf` |
| `MASK` | `0` | `1` = inpaint mode (needs `target.*`) |
| `SEED` | random, printed | first seed; each result adds 1 |
| `STRENGTH` | `0.99` | mask mode only |
| `STEPS` | `12` | denoise steps |
| `REF_PX` | `512` | bound on each reference's encoded long edge; `0` = native |
| `SIZE` | `768x1024` | `WxH`, default (non-mask) mode only |
| `POSE` | (built-in) | prompt text describing pose/lighting |
| `VARIANT` | `klein-4b` | FLUX.2 variant |
| `PRECISION` | `int8` | DiT numeric tier |
| `BRAIN` | `./target/release/brain` | binary path |

Every flag also has an environment variable of the same name in caps, and the
flag wins when both are given. Count defaults to 4 results, written as
`result-01.jpg`...
