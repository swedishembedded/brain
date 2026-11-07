<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 103. A detector head answers "what kind of thing", so ask something else "which one"
**Lesson:** A detector head answers "what kind of thing", so ask something else "which one":
decide identity where the signal lives, not in a class logit. Before explaining a bad measurement
with a theory, check that the artifact you measured was produced by the code you are theorising
about.

## What happened

Adding a class for a specific PERSON to a COCO-preserving YOLOv8 head is a well-posed engineering
task with no good answer, and the reason is in the features, not the recipe. Both ends of the trade
below were measured, and neither is a tuning problem.

## Why

The shared backbone was trained to separate CATEGORIES. Everything that distinguishes one person
from another is within-category variation, which is exactly what such a representation is rewarded
for discarding.

A frozen head is then a linear probe on a space where the answer is not present, and it learns the
only thing that IS present: "a person, in this one's typical scenes". Unfreezing the features so
the head can learn identity works by destroying the preservation the freeze existed for.

## What to do

Decide identity where the signal lives: the detector supplies a person BOX, and a face embedder
decides WHO is in it. That is `yolov8::identity`, with a `BoxEmbedder` seam so `crates/yolov8` gains
no face-model dependency, wired to SCRFD + ArcFace by `brain yolov8 detect --identity-ref`.

Measured on one held-out set, strangers' faces score at most 0.12 cosine against the reference, and
every target face big enough to identify scores 0.27 or better, a gap a class logit never
produced. The composed system's misses are also of a different kind: it abstains, with a reason
("no face in this box", "cosine 0.06"), where the class head guessed.

Two things this design gets wrong if built the obvious way.

**Do not crop the box and hand the crop to the face detector.** A face detector resizes whatever it
is given onto its own square canvas, so a small crop is UPSCALED, and a marginal face does not
survive that. Interpolated detail is not detail.

Run the face search ONCE over the frame at native scale and attribute each face to the box its
centre falls inside. It is more accurate AND N times cheaper.

**Do not let a fine-tune's training resolution become the pretrained classes' inference
resolution.** The checkpoint records the `--input` it was trained at, so a cheap 256 fine-tune
silently becomes the geometry for all 80 inherited COCO classes, at a scale they never saw. The
weights were preserved perfectly and the model was still wrong, which is why `Yolo::load_at`
separates the two.

A corollary worth its own line, because it cost a wrong root-cause first: before explaining a bad
measurement with a theory, check that the artifact you measured was produced by the code you are
theorising about.

## Evidence

- Crop upscaling: the same face of 17 x 24 source pixels, same detector, was found at score 0.595
  in the full 512² frame (1.25x onto its 640² canvas), and found by NOTHING at 400², 300² or 235²
  crops around it (1.6x, 2.1x, 2.7x).
- Inference resolution, same bit-identical weights: at 640 every base COCO detection reproduces to
  the printed digit. At 256 chair becomes bench, three spurious cars appear, and the added class
  fires on background at 0.21.
- Stale artifact: the checkpoint under test predated the fine-tune commit entirely. It was a
  from-scratch `tiny(nc=1)` model sharing zero tensors with any COCO release, and every conclusion
  drawn from it was about nothing.
