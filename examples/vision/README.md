# Segmentation and face identity over D-Bus

Two imaging models driven through `com.swedishembedded.Brain1` — no model-specific
code in the transport, no bespoke subcommand. Both go through the same
`Run(model, action, params, in_fds, in_meta, transport)` method `examples/dbus`
documents, exchanging images as file descriptors (memfd/dmabuf) rather than bytes
marshalled through D-Bus.

| model | actions | weights env |
|---|---|---|
| `sam2` | `segment` — point/box prompts → a mask | `BRAIN_SAM2_WEIGHTS` (+ optional `BRAIN_SAM2_VARIANT=tiny\|large`) |
| `facenet` | `detect` → boxes/scores/landmarks, `embed` → a 512-d identity vector | `BRAIN_FACENET_DIR` (the antelopev2 directory) |

Discover them the same way as everything else:

```bash
brain caps sam2
brain caps facenet
```

## Run it

A private session bus needs no system configuration:

```bash
BRAIN_SAM2_WEIGHTS=$BRAIN_TESTDATA/sam2/hiera-tiny/sam2.1_hiera_tiny.pt \
BRAIN_FACENET_DIR=$BRAIN_TESTDATA/face/antelopev2 \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/vision/segment_image.py --image photo.ppm --point 614,430 --concurrent 4
    python3 examples/vision/face_id.py a.ppm b.ppm c.ppm'
```

Inputs are binary PPM (P6) — brain's image convention; `brain_py.image.load_ppm`
turns one into the HWC-f32 blob the wire format carries, and `save_ppm` turns a
result back. Everything the scripts write is a PPM too.

The same actions work with no bus at all:

```bash
BRAIN_SAM2_WEIGHTS=… brain do sam2 segment --points "614,430" \
    --in image=photo.ppm --out mask=mask.ppm --json
BRAIN_FACENET_DIR=… brain do facenet detect --in image=photo.ppm --json
BRAIN_FACENET_DIR=… brain do facenet embed --align false \
    --in image=aligned112.ppm --out embedding=id.bin
```

## `segment_image.py` — SAM 2.1

Prompts are given in **source-image pixels**; the action scales them into the
model's 1024² frame and scales the mask back, so a client never sees the model's
geometry. Points are `--point x,y` (repeatable), a box is
`--box x1,y1,x2,y2`, and `--labels` (over the wire) marks background clicks with
`0`.

The mask comes back as `sigmoid(logits)` on the source grid, tagged
`Media::Mask`: **threshold at 0.5** for a binary mask (which is exactly
`logit > 0`, the reference's rule).

What the timings show:

```
  {'points': '614.4,430.08'}   iou 0.9924  area 73748   16409.5 ms  <- trunk + decoder
  {'points': '500,375'}        iou 0.9966  area 73650     946.9 ms  <- decoder only
  {'box': '122.88,716.8,…'}    iou 0.9949  area 53487     514.2 ms  <- decoder only
```

SAM 2 is encode-once / prompt-many: the resident instance caches the image
encoding keyed by a hash of the image blob, so only the first prompt on a frame
pays the Hiera trunk. That is also the model's real batching axis —
`--concurrent 4` submits four prompts at once and the residency `Executor`
groups them into one batch (`max_batch: 3` in `brain.stats()`), which
`resident_sam2::run_batch` answers with **one** trunk pass and N decoder passes.

> Numbers above were measured on a Tesla P40 with a **debug** build of `brain`
> and the `hiera_tiny` checkpoint; the first call also includes the 156 MB
> checkpoint import. Use `make release` for representative latency.

## `face_id.py` — SCRFD + ArcFace

`detect` returns every face with its score and five landmarks, in source pixels.
`embed` runs the same detector, picks the primary face (`select = largest`, or
`score`), similarity-aligns it to the 112² ArcFace template and returns the
512-d vector — **already L2-normalised**, so a cosine is a plain dot product:

```
            face0.ppm face1.ppm
face0.ppm      1.0000   -0.0722
face1.ppm     -0.0722    1.0000
```

Pass `--align false` (or `{"align": false}` over the bus) when the input is
already an aligned face crop; then no detector runs and the embedding is the
reference one bit for bit (cosine 1.000000 against the insightface goldens).

## What is NOT here

* **Batched faces.** `facenet`'s two graphs are built for a single image, so its
  `run_batch` is the serial default — stated, with the reason, in
  `crates/cli/src/resident_facenet.rs`. SAM 2 batches by image, as above.
* **A mask prompt for SAM 2.** The reference downsamples one with
  `interpolate(antialias=True)` and brain has no antialiased resize kernel, so
  the wire surface does not offer what it cannot compute exactly.
