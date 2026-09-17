# sample: vision/face-id (D-Bus)

`face_id.py` drives two models of the insightface antelopev2 stack over
D-Bus: `brain/scrfd` `detect` (boxes, scores, 5 landmarks, in source-image
pixels) and `brain/arcface` `embed` (one L2-normalised 512-d identity
vector for the primary face - it runs the detector itself unless
`align=false`). Given several photos it prints the detections, writes a
box overlay per photo, then prints the full cosine similarity matrix over
the embeddings - since the vectors are already unit-norm, cosine is a
plain dot product.

```bash
BRAIN_SCRFD_DIR=$BRAIN_TESTDATA/face/antelopev2 \
BRAIN_ARCFACE_DIR=$BRAIN_TESTDATA/face/antelopev2 \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/face-id/face_id.py a.ppm b.ppm c.ppm
```

## What it demonstrates

```
            face0.ppm face1.ppm
face0.ppm      1.0000   -0.0722
face1.ppm     -0.0722    1.0000
```

Pass `--align false` (or `{"align": false}` over the bus) when the input is
already an aligned face crop; then no detector runs and the embedding is
the reference one bit for bit (cosine 1.000000 against the insightface
goldens). Both face graphs are built for a single image, so their
`run_batch` is the serial default (stated, with the reason, in
`crates/cli/src/resident_scrfd.rs`).

## What it needs

`BRAIN_SCRFD_DIR` and `BRAIN_ARCFACE_DIR`, both pointing at the antelopev2
checkpoint directory. Inputs are binary PPM (P6) - brain's image
convention; `brain_py.image.load_ppm`/`save_ppm` handle the conversion.

## Options

| flag | default |
|---|---|
| `photos...` | *required* - one or more binary PPMs |
| `--out DIR` | `/tmp` (box overlays) |

## The CLI, for comparison

```bash
BRAIN_SCRFD_DIR=… brain scrfd detect --in image=photo.ppm --json
BRAIN_ARCFACE_DIR=… brain arcface embed --align false \
    --in image=aligned112.ppm --out embedding=id.bin
```
