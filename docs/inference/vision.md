# Vision: detection, depth, segmentation, and faces

brain runs a set of models for *understanding* an image - finding objects,
estimating depth, cutting out a region, or recognizing a face - as distinct
from *editing* one. For cropping, restoring, or upscaling an existing image,
see [imaging.md](imaging.md); this page is about detection and
understanding.

## Capabilities

### Object detection - `Ultralytics/YOLOv8`

An anchor-free, single-stage real-time object detector, byte-compatible with
the official Ultralytics YOLOv8n graph - pretrained weights import and run
unchanged, and it can also be trained or fine-tuned on your own data. Reach
for it for bounding-box detection, whether as a one-shot call or wired into
a live event-driven pipeline. See
[the YOLOv8 page](../models/yolo/readme.md).

### Monocular depth - `brain/depth`

Point it at a single image or a live camera feed and it produces a
per-pixel depth map, no stereo rig needed. It's small and fast enough for
realtime use - live webcam preview, depth-of-field effects, or
autostereograms. See [the ZipDepth page](../models/depth.md).

### Promptable segmentation - `brain/sam2`

Give it an image plus a point or box prompt and it returns a pixel-accurate
mask for the object you pointed at - background removal, region selection,
or a mask to feed into another pipeline, without training a detector for
that specific object class. See [the SAM 2.1 page](../models/sam2.md).

### Face detection - `brain/scrfd`

Finds every face in an image: boxes, scores and five landmarks, in
source-image pixels. See [the SCRFD page](../models/scrfd.md).

### Face identity - `brain/arcface`

Turns a face into an identity embedding for matching or search - "are these
two photos the same person?" By default it detects and aligns the face
itself (via SCRFD); pass `--align false` for an already-aligned crop. It's
also the identity input some generative pipelines condition on. See
[the ArcFace page](../models/arcface.md).

## Detection vs. editing

These models look at an image and tell you something about it, or cut
a piece out of it. If what you actually want is to *change* pixels -
restore a degraded face, upscale a result, or run a segment-refine-restore-
upscale pipeline in one call - see [imaging.md](imaging.md), which covers
that side of brain's image tooling.
