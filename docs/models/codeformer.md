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

## Getting a usable crop in the first place

Two things about the input decide more about the result than any flag on this
page, and both are easy to get wrong silently.

**Use facexlib's 512² template, not `arcface::ARCFACE_DST_112` rescaled.** They
are not the same crop: the rescaled ArcFace template puts the eyes 161px apart
in a 512² crop, facexlib's puts them 126px apart, so the ArcFace one hands
CodeFormer a face 28% larger than its prior was trained on. Nothing fails; the
restoration is just worse, and worst around the eyes, where the prior then
reconstructs a differently-shaped eye because it believes the face is closer
than it is.

```text
facexlib 512²  [[192.98, 239.95], [318.90, 240.19], [256.63, 314.02],
                [201.26, 371.41], [313.09, 371.15]]
```

**Do not feed it an upscaled face.** The crop is 512², so if the face occupies
150px in the source photo, the aligned crop is a 3.4x upscale carrying no
detail above 150px, and CodeFormer will invent all of it from its prior - which
looks exactly like a restoration failure and is not one. Aim for an eye-to-eye
distance of at least ~126px in the SOURCE, so the warp scales down rather than
up.

## What the dial costs, measured

`w` is worth setting deliberately, because the default is not neutral. One
aligned 512² face, put through three levels of the degradation CodeFormer
trains against (gaussian blur, downsample, gaussian noise, JPEG - in that
order) and restored at four settings. Two measures, because either one alone
is misleading: PSNR against the original in dB, and the ArcFace cosine to the
original's identity embedding.

ArcFace cosine to the original:

| input | the input itself | `w=0.0` | `w=0.5` | `w=0.7` | `w=1.0` |
|---|---|---|---|---|---|
| undamaged original | +1.0000 | +0.6572 | +0.8487 | +0.8884 | +0.9313 |
| 256px, blur 1.0, noise 2, q70 | +0.9637 | +0.6770 | +0.8572 | +0.8916 | +0.9183 |
| 160px, blur 2.0, noise 4, q50 | +0.9210 | +0.6527 | +0.8176 | +0.8515 | +0.8878 |
| 112px, blur 3.0, noise 6, q35 | +0.7316 | +0.6239 | +0.7294 | +0.7669 | **+0.7929** |

PSNR against the original, dB:

| input | the input itself | `w=0.0` | `w=0.5` | `w=0.7` | `w=1.0` |
|---|---|---|---|---|---|
| undamaged original | - | 27.87 | 30.71 | 31.39 | 32.10 |
| 256px, blur 1.0, noise 2, q70 | 35.03 | 27.86 | 30.28 | 30.82 | 31.56 |
| 160px, blur 2.0, noise 4, q50 | 31.10 | 27.90 | 29.34 | 29.72 | 30.29 |
| 112px, blur 3.0, noise 6, q35 | 28.47 | 27.43 | 28.66 | 28.93 | **29.24** |

What the two tables say together:

- **There is a damage threshold, and below it restoring makes things worse.**
  On the top three rows every setting scores below the input it was given, on
  both measures. Only the bottom row - a face genuinely destroyed - beats its
  input, and there `w=1.0` wins on both at once. Restoration is a repair, not
  an enhancement: run it on something that is not broken and you pay for it.
- **`w` is monotonic on both measures, at every damage level.** Higher `w`
  is closer to the input pixels, and that is closer to the truth whenever the
  input still carries any. There is no setting where the prior helps.
- **The default `w=0.5` gives up about 0.15 of cosine** relative to `w=1.0`,
  and looks no worse doing it. If the output stands in for a person -
  evidence, identity documents, archive restoration - the number to use is
  `1.0`, and the reason to believe that is this table rather than the flag's
  help text.
- **The eye cannot arbitrate this.** A `w=0.0` restoration of a destroyed face
  is a clean, plausible, good-looking face that scores below the smear it came
  from. That is why the ArcFace step is in the pipeline at all.

## Hardware and limits

The action expects an already-aligned face - pair it with a face-detection
step to locate and align a face within a full photo first if you don't
already have one cropped. No LoRA/fine-tune path is exposed on the CLI, no
batching beyond one image per request, and no HTTP endpoint - use `brain do`
or D-Bus.
