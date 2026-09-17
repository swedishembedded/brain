# sample: imagegen/identity-score

Is the generated face actually THEIR face? Answer as a number, not an
opinion. A direct CLI wrapper (no server) around `brain arcface embed`.

```bash
BRAIN_ARCFACE_DIR=/path/to/antelopev2 \
  samples/shell/imagegen/identity-score/identity_score.sh <ref-dir> <generated-dir>...
```

Prints, per generated folder, the mean ArcFace cosine between every image in
it and the reference photographs - the same 512-d identity embedding face
recognition itself is built on, so the number means what it says.

## What it demonstrates

- A grading step meant to sit at the end of every identity-conditioning
  workflow in this tree: `../portrait-from-refs/`, `../face-swap/`, and
  `../lora/train_identity_lora.sh` all point here.
- Reading these anchors, which this repo measures rather than assumes:

  ```
  ~0.0   a stranger. Text-only generation of a named person lands here,
         because the model has never seen them.
  ~0.3   recognisably influenced - family-resemblance territory.
  ~0.5   the same person to a human, most of the time.
  ~0.7+  what two genuine photographs of one person score against each
         other. This is the ceiling, not 1.0: pose, lighting and age move it.
  ```

- A conditioning path that does not MOVE this number is not working, however
  good the pictures look. That is the whole point of running it: identity is
  the one property of a portrait you cannot grade by eye without fooling
  yourself, because the eye grades "plausible person" and this grades "that
  person".

## What it needs

`BRAIN_ARCFACE_DIR` and `BRAIN_SCRFD_DIR` (defaults to `BRAIN_ARCFACE_DIR`),
both the directory holding the insightface antelopev2 `glintr100.onnx` and
`scrfd_10g_bnkps.onnx`.

## Notes

- The reference directory is the folder of their photographs (any name, any
  format brain decodes); the generated directories are what you are grading.
- Faces that fill the entire frame are NOT detectable - the detector needs
  context around the head - so generate head-and-shoulders framing if you
  intend to measure it. Undetectable images are reported, never scored as 0.
- Output also prints the references' own mean pairwise cosine as "the
  ceiling", so a generated score is never compared against an imagined 1.0.
