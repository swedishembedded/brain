# Restore (blind face restoration)

Blind face restoration for degraded photos: feed it a low-quality, blurry,
compressed, or otherwise damaged face image — ideally already cropped/aligned
to the face — and it returns a restored 512x512 version. A fidelity dial lets
you choose how much the model should trust the input pixels versus regenerate
detail from its own learned prior.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI (`brain do`)       | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/restore`. Set `BRAIN_RESTORE_WEIGHTS` to a `codeformer.pth`
checkpoint file, or to a directory containing one.

## Running it

```bash
brain caps brain/restore
brain do brain/restore restore_face --w 0.5 \
    --in image=face.ppm --out image=restored.ppm --json
```

Over D-Bus, the single action is `restore_face`: input `image` (ideally an
aligned 512x512 face), one float param `w`, output `image` (the restored
512x512 face).

## Options

- `w` (`0.0..=1.0`, default `0.5`) — the identity-fidelity dial. `0` favors
  maximum restored quality (the model leans on its own prior, more
  hallucination); `1` favors maximum fidelity to the original degraded input
  (less hallucination, closer to the source).

## Hardware and limits

The action expects an already-aligned face — pair it with a face-detection
step to locate and align a face within a full photo first if you don't
already have one cropped. No LoRA/fine-tune path is exposed on the CLI, no
batching beyond one image per request, and no HTTP endpoint — use `brain do`
or D-Bus.
