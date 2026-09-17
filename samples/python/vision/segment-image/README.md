# sample: vision/segment-image (D-Bus)

`segment_image.py` drives `brain/sam2`'s `segment` action (SAM 2.1) over
D-Bus: one image, several point/box prompts, a mask plus a cut-out per
prompt back.

```bash
BRAIN_SAM2_WEIGHTS=$BRAIN_TESTDATA/sam2/hiera-tiny/sam2.1_hiera_tiny.pt \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/segment-image/segment_image.py --image photo.ppm --point 614,430 --concurrent 4
```

## What it demonstrates

SAM 2 is encode-once / prompt-many: the resident instance caches the image
encoding keyed by a hash of the image blob, so only the first prompt on a
frame pays the Hiera trunk. The timings make that visible:

```
  {'points': '614.4,430.08'}   iou 0.9924  area 73748   16409.5 ms  <- trunk + decoder
  {'points': '500,375'}        iou 0.9966  area 73650     946.9 ms  <- decoder only
  {'box': '122.88,716.8,…'}    iou 0.9949  area 53487     514.2 ms  <- decoder only
```

`--concurrent N` submits N prompts at once and the residency `Executor`
groups them into one batch (`max_batch: N` in `brain.stats()`), which
`resident_sam2::run_batch` answers with **one** trunk pass and N decoder
passes - the model's real batching axis.

Prompts are given in **source-image pixels**; the action scales them into
the model's 1024^2 frame and scales the mask back, so a client never sees
the model's geometry. The mask comes back as `sigmoid(logits)`, tagged
`Media::Mask`: **threshold at 0.5** for a binary mask (exactly `logit > 0`,
the reference's rule).

## What it needs

`BRAIN_SAM2_WEIGHTS` (+ optional `BRAIN_SAM2_VARIANT=tiny|large`). Input is
a binary PPM (P6).

**Not supported**: a mask prompt. The reference downsamples one with
`interpolate(antialias=True)` and brain has no antialiased resize kernel,
so the wire surface doesn't offer what it can't compute exactly.

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* |
| `--point x,y` | repeatable; defaults to the image centre if no point/box is given |
| `--box x1,y1,x2,y2` | none |
| `--concurrent N` | `0` (batching demo, N prompts submitted at once) |
| `--out DIR` | `/tmp` |
