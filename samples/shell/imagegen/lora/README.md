# sample: imagegen/lora

A train -> generate workflow around FLUX.2 Klein LoRA adapters, direct CLI
invocations (no D-Bus, no server) of `brain label`, `brain flux2 finetune`
and `brain flux2 generate`. Three scripts:

- `train_lora.sh` - caption a folder, then train an adapter on it.
- `run_lora.sh` - generate with an adapter `train_lora.sh` produced.
- `train_identity_lora.sh` - `train_lora.sh` specialised for the case where
  the concept is a specific *person*.

## train_lora.sh

A folder of images in, a LoRA adapter out, in two steps:

```bash
BRAIN_FLUX2_DIT=... BRAIN_FLUX2_VAE=... BRAIN_FLUX2_TE=... BRAIN_FLUX2_TOKENIZER=... \
  samples/shell/imagegen/lora/train_lora.sh <image-dir> [adapter.brain] [trigger phrase]
```

Step 1 captions every image with a vision-language model and writes
`<dir>/captions.yaml` - editable YAML with block scalars, so you can read what
the model saw and fix it before spending GPU time on it. Captions ARE the
training signal; a vague caption set caps the adapter's ceiling, so reviewing
them is the highest-leverage minute in this script. Step 2 trains the adapter
on those captions.

The trigger phrase is what you type later to invoke the style. A rare token
binds cleanly but reads oddly; a natural phrase composes well but shifts a
concept the base model already owns, so it leaks into every prompt that
mentions it.

Options (env only): `MODEL` (`qwen3vl`\|`fastvlm`), `STEPS`, `RANK`, `LR`,
`SIZE`, `VARIANT`, `TRAINER` (`device`\|`host` - device is the GPU trainer and
the default), `BRAIN`.

## run_lora.sh

Generate with the adapter:

```bash
samples/shell/imagegen/lora/run_lora.sh ~/photos/bohemian "a bohemian style bedroom"
```

The first argument is either the adapter file or the folder you trained on -
`train_lora.sh` writes `adapter.brain` inside it, so passing the folder is
enough. Whatever trigger phrase you trained with has to appear in the prompt,
or the adapter has little to fire on. Results are `<name>-01.jpg`... one per
seed, beside the adapter.

`REF` restyles an existing photograph instead of generating from nothing, and
`STRENGTH` is the dial: `1.0` generates a new image conditioned on it, lower
values keep progressively more of it, `0` returns it unchanged. Below 1.0 the
reference IS the starting latent and so must be at the output size; the
script resizes it for you.

```bash
REF=room.jpg STRENGTH=0.96 samples/shell/imagegen/lora/run_lora.sh ~/photos/boho "..."
```

Options (env only): `REF`, `STRENGTH`, `SEED` (first; each result adds 1),
`STEPS`, `SIZE` (`WxH`), `SCALE` (adapter strength, `0` reproduces the base
model - the way to see what the adapter is actually contributing), `VARIANT`,
`PRECISION`, `BRAIN`.

## train_identity_lora.sh

A few photographs of one person in, a LoRA that knows their face out:

```bash
BRAIN_SCRFD_DIR=... BRAIN_FLUX2_DIT=... BRAIN_FLUX2_VAE=... BRAIN_FLUX2_TE=... BRAIN_FLUX2_TOKENIZER=... \
  samples/shell/imagegen/lora/train_identity_lora.sh <photo-dir> "<their name>"
```

This is `train_lora.sh` specialised for the case where the concept is a
**person**, which changes three things that matter more than the
hyperparameters:

1. **The dataset is built by the face detector, not by the crop tool.**
   Holiday snaps and selfies are mostly not-the-person: background, other
   people, half a restaurant. Every photograph is cropped square around the
   detected primary face, and the crop window SLIDES to push bystanders out
   of frame before it gives up and shrinks. A photograph whose face cannot be
   framed without a bystander is dropped rather than poisoned in.
2. **The captions describe everything EXCEPT the face.** Framing, pose,
   clothing, background, light - yes. Eyes, skin, head shape - never. An
   adapter binds a concept to whatever the caption does not already explain,
   so describing the face hands the identity to those words instead of to
   the name.
3. **The trigger is a name and is captioned as a name** (`brain label
   --trigger-role`), not a style.

Each source photograph also enters mirrored. Three photographs is thin for
this - expect the adapter to know the face and to have opinions about pose
and lighting that it should not have.

Grade the result, always, with a number:

```bash
samples/shell/imagegen/identity-score/identity_score.sh <photo-dir> <generated-dir>
```

then generate with
`ADAPTER=<out> samples/shell/imagegen/portrait-from-refs/portrait_from_refs.sh`.

Options (env only): `OUT` (adapter path), `STEPS`, `RANK`, `LR`, `SIZE`,
`VARIANT`, `TRAINER`, `CKPT_EVERY`, `MODEL` (captioner), `BRAIN`.

## What it needs

- `train_lora.sh` / `run_lora.sh`: `BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}`, plus
  the captioner's own weights (fetched on demand or pointed at with
  `--weights`).
- `train_identity_lora.sh`: additionally `BRAIN_SCRFD_DIR` (the directory
  holding `scrfd_10g_bnkps.onnx`) for the face detector, and
  `$BRAIN_QWEN3VL_WEIGHTS` (default) for the captioner.

Direct CLI invocations throughout - no `brain serve`, no D-Bus.
