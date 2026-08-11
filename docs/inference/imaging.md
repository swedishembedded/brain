# Imaging: segmentation, restoration, and upscaling

Beyond generating images from a prompt, brain can operate *on* an existing
image — cut out a region, find and embed a face, clean up a degraded face,
upscale a result, or move an image in and out of a discrete codebook. These
are separate model ids from image generation, each with its own `brain do`
action, and they compose into one pipeline call so a multi-step edit doesn't
need a round-trip per stage.

## Capabilities

### Promptable segmentation — `brain/sam2`

Point or box prompts on an image produce a pixel-accurate mask (SAM 2.1-style).
Use it whenever an edit needs to touch *only* one region — a mask from this
model is what turns "only change the sky" into an exact operation instead of a
prompt-engineering hope. See the [SAM 2.1 page](../models/sam2.md).

### Face detection and identity embedding — `brain/facenet`

Detects faces in a photo (boxes, scores, landmarks) and produces a normalized
identity embedding for a face crop. Use it for face search/verification, or to
locate and align a face before restoring it. See
the [face recognition page](../models/face.md).

### Blind face restoration — `brain/restore`

Takes a degraded (ideally aligned) face and produces a restored version, with
a continuous identity-fidelity dial: one end favors visual quality, the other
favors staying close to the input. Use it after detection/alignment to clean
up compression artifacts, blur, or low resolution on a face crop. See
the [face restoration page](../models/restore.md).

### Image upscaling — `brain/upscale`

Super-resolves an image (4x on the released checkpoint). Use it as a final
step after generation, editing, or restoration to raise output resolution.
See the [upscaling page](../models/upscale.md).

### Image / codebook encode-decode — `brain/vqgan`

Encodes an image to a grid of discrete codebook indices, and decodes indices
back to an image. This is the same discrete-codebook mechanism face
restoration builds on; use it directly when you need the codes themselves
(e.g. for downstream analysis) rather than a restored image. See
the [VQGAN page](../models/vqgan.md).

### Composed pipeline — `brain/imgpipe`

Chains segment -> refine mask -> restore -> upscale as **one** call instead of
four separate round-trips, so intermediates never cross the process/bus
boundary and pixels outside the mask come back unchanged. Use it whenever an
edit needs more than one of the above stages — it's the same providers `brain
do` uses for each model, just composed server-side. See
the [imgpipe page](../models/imgpipe.md).

## Approximate VRAM needed

Measured configuration, not a promise — actual usage depends on image size and
batching:

| Model | Approximate VRAM |
|---|---|
| `brain/sam2` | under 1 GB |
| `brain/facenet` | under 1 GB |
| `brain/restore` | under 1 GB |
| `brain/vqgan` | under 1 GB |
| `brain/upscale` | under 1 GB |

All five are small enough to stay resident together on one GPU.
`brain/imgpipe` holds no weights of its own — it composes whichever of the
above stages a call asks for, so its cost is the sum of the stages it runs.
