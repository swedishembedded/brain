# SAM 2.1 (segmentation)

Promptable segmentation. On an image: give it a point, a box, or both, and it
returns a mask for the object you pointed at. On a video: click the object
once and the mask FOLLOWS it, frame by frame, through the clip.

Reach for it whenever you need to cut an object out - background removal,
region selection for editing, building a mask for another pipeline - without
training a detector for that specific object class. The video path is what a
character swap needs: a per-frame mask that tracks a moving subject rather
than a rectangle held still.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [x] |
| Video tracking (`brain sam2 track`) | [x] |

## Getting the weights

Model id: `brain/sam2`. `facebook/sam2.1-hiera-tiny` auto-fetches (⤓, opt-in `--autofetch`) on
first CLI use - no env var or manual download needed. For the `large`
variant, or to point at a checkpoint you already have, set
`BRAIN_SAM2_WEIGHTS` to a `sam2.1_hiera_{tiny,large}.pt` release checkpoint
(or an equivalent `.safetensors`) yourself; `BRAIN_SAM2_VARIANT` picks
`tiny` (default) or `large`, and a per-request `variant` param overrides it.

## Running it: one image

`segment` is a generic capability action, over the CLI or D-Bus. It takes a required `image` input plus
prompt params, and returns a `mask` (sigmoid probability on the source-image
grid; threshold at 0.5):

```bash
brain caps brain/sam2

brain sam2 segment --points "614,430" \
  --in image=photo.ppm --out mask=mask.ppm --json
```

Over D-Bus, the same action takes the same params:

```bash
BRAIN_SAM2_WEIGHTS=<ckpt>/sam2.1_hiera_tiny.pt \
  dbus-run-session -- bash -c 'brain serve --dbus & sleep 3
    python3 examples/vision/segment_image.py --image photo.ppm --point 614,430 --concurrent 4'
```

Reference client: `examples/vision/segment_image.py` - prompts are given in
source-image pixels; `--point x,y` is repeatable, `--box x1,y1,x2,y2` sets a
box prompt, `--concurrent N` submits N prompts at once to exercise the
image-batched path.

## Running it: a video

`brain sam2 track` is the video path. One click on one frame in, a per-frame
mask sequence out:

```bash
brain sam2 track --video clip.mp4 --point 640,300 --out masks/
```

That writes a mask-sequence DIRECTORY:

```text
masks/masks.json          the manifest
masks/mask_000000.png     frame 0, 8-bit, source resolution
masks/mask_000001.png     ...
```

**Polarity is declared in `masks.json`, never inferred.** By default `255` is
the tracked object (SAM 2's own meaning). `--invert` writes the exact inverse
instead, which is what a consumer whose `1` means "keep this region, do not
regenerate" wants - LTX-2.5's masked conditioning, where replacing a character
means masking the BACKGROUND white. A consumer must read the `polarity` key
and hard-error if it is missing: the two are exact inverses, and getting it
backwards regenerates the whole background while preserving the subject.

`masks.json` also carries a per-frame `object_score`. **Negative means SAM 2
believes the object is absent or fully occluded on that frame**, so an empty
mask there is the model's answer, not a bug - worth surfacing in a stunt
sequence, which is full of occlusion.

| Flag | Effect |
|---|---|
| `--video` / `--frames` | a clip (needs ffmpeg), or a directory of numbered images |
| `--point x,y` | the click that picks the object, in source pixels |
| `--label 0\|1` | 0 = background click, 1 = foreground (default) |
| `--prompt-frame n` | which frame the click sits on (default 0) |
| `--max-frames n` / `--fps f` | trim or resample the clip before tracking |
| `--object-id n` | recorded in `masks.json`, for multi-character work |
| `--invert` | emit `object=0` polarity |
| `--soft` | emit the sigmoid ramp instead of a hard 0/255 mask |

Masks come out at the SOURCE frame size, one PNG per source frame, contiguous
from 0. Any spatial or temporal downsampling belongs to the consumer, against
its own grid.

## Options

| Param | Effect |
|---|---|
| `variant` | `tiny` (default) or `large` - overrides `BRAIN_SAM2_VARIANT` for this request |
| `points` | `"x,y;x,y;…"` in source-image pixels |
| `labels` | `"1;0;…"` - 1 = foreground, 0 = background (default: all-foreground) |
| `box` | `"x1,y1,x2,y2"` |
| `multimask` | bool, default `true` |

Multiple prompts on the same image are cheap: N prompts on one frame cost one
image-encoder pass plus N decoder passes, so batching prompts (not requests)
is the efficient way to segment many regions of one photo.

## Hardware and limits

The video path tracks ONE object per run: `--object-id` names it in the
manifest, and several characters means several runs (they share nothing but
the clip). Tracking is forward only - frames BEFORE the clicked one take the
clicked frame's mask rather than being tracked backwards - and a correction
click on an already-tracked frame is not implemented, so re-prompting means
re-running. A video mask PROMPT (segment from an existing mask rather than a
point) is likewise not wired up.

There's no training or fine-tune verb and no HTTP surface; `segment` is
reached through the CLI's generic capability dispatch or D-Bus, and `track`
through its own CLI verb (a mask sequence is a directory, which no single
capability blob can carry).
