# Restore (blind face restoration)

Blind face restoration for degraded photos: feed it a low-quality, blurry,
compressed, or otherwise damaged face image - ideally already cropped/aligned
to the face - and it returns a restored 512x512 version. A fidelity dial lets
you choose how much the model should trust the input pixels versus regenerate
detail from its own learned prior.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [ ] `/v1/images/generations` is text-to-image only; `restore_face` takes a REQUIRED input image, which `api_caps` excludes by design |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/codeformer`. Put a `codeformer.pth` checkpoint under the
models directory (`--models-dir` / `BRAIN_MODELS_DIR`) and brain finds it by
scanning - `brain codeformer restore_face` needs no variable set.
`BRAIN_CODEFORMER_WEIGHTS` still works as an explicit pin to one file.

## Running it

```bash
brain caps brain/codeformer
brain codeformer restore_face --w 0.5 \
    --in image=face.ppm --out image=restored.ppm --json
```

Over D-Bus, the single action is `restore_face`: input `image` (ideally an
aligned 512x512 face), one float param `w`, output `image` (the restored
512x512 face).

## Options

- `w` (`0.0..=1.0`, default `0.5`) - the identity-fidelity dial. `0` favors
  maximum restored quality (the model leans on its own prior, more
  hallucination); `1` favors maximum fidelity to the original degraded input
  (less hallucination, closer to the source).

## What the dial costs, measured

`w` is worth setting deliberately, because the default is not neutral. These
are ArcFace cosines against the undamaged original, for one aligned 512x512
face put through three levels of damage and restored at four settings
(`brain arcface embed --align false` on each output, cosine against the
original's embedding):

| input | degraded | `w=0.0` | `w=0.5` | `w=0.7` | `w=1.0` |
|---|---|---|---|---|---|
| undamaged original | +1.0000 | +0.3064 | +0.8008 | +0.8911 | +0.9498 |
| 256px, JPEG q60 | +0.9765 | +0.2921 | +0.7759 | +0.8804 | +0.9394 |
| 160px, JPEG q40 | +0.9159 | +0.3378 | +0.6991 | +0.8137 | +0.8806 |
| 112px, JPEG q30 + blur | +0.7893 | +0.3204 | +0.6487 | +0.7252 | +0.7747 |

Three things this says, none of which is obvious from the flag's help text:

- **`w=0` discards the identity outright**, and the damage has nothing to do
  with it. Restoring the *undamaged* original at `w=0` still scores +0.31 -
  within noise of what the heavily damaged one scores. At `w=0` the output is
  CodeFormer's prior wearing the input's pose: a clean, attractive,
  well-lit face belonging to someone who was never in the photo.
- **Restoration never raises the cosine above the degraded input.** ArcFace is
  trained to be invariant to exactly these degradations, so a blurred, crushed
  face still embeds close to its original: +0.79 even at 112px. The number is
  therefore not a quality score and must not be read as one. It measures
  identity DRIFT, which is the failure mode blind restoration actually has,
  and it is worth having precisely because the eye cannot see it - the drifted
  output looks better, not worse.
- **`w=1.0` costs 0.015 to 0.05 of cosine** and buys back the detail. That is
  the setting for anything where the output stands in for a person: evidence,
  identity documents, archive restoration.

The default `w=0.5` sits closer to the prior than most callers expect - it
gives up roughly 0.20 of cosine even on a perfectly clean input.

## Hardware and limits

The action expects an already-aligned face - pair it with a face-detection
step to locate and align a face within a full photo first if you don't
already have one cropped. No LoRA/fine-tune path is exposed on the CLI, no
batching beyond one image per request, and no HTTP endpoint - use `brain do`
or D-Bus.
