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
| Video tracking (`track`) | [x] |

## Getting the weights

Model id: `brain/sam2`. Both variants brain runs pull into the model store:

```bash
brain pull facebook/sam2.1-hiera-tiny     # also fetched on first use with --autofetch
brain pull facebook/sam2.1-hiera-large
```

The checkpoint resolves from the store; with more than one there, name it
with `--weights <file>` (a `sam2.1_hiera_{tiny,large}.pt` release
checkpoint, or an equivalent `.safetensors`). The variant is read from the
checkpoint's own trunk width; `--variant tiny|large` states it, and is
checked against the file.

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
dbus-run-session -- bash -c 'brain serve --dbus & sleep 3
    python3 samples/python/vision/segment-image/segment_image.py --image photo.ppm --point 614,430 --concurrent 4'
```

Reference client: `samples/python/vision/segment-image/segment_image.py` - prompts are given in
source-image pixels; `--point x,y` is repeatable, `--box x1,y1,x2,y2` sets a
box prompt, `--concurrent N` submits N prompts at once to exercise the
image-batched path.

## Running it: a video

`track` is the video path. Clicks on one frame in, the object's mask on
every frame out:

```bash
brain sam2 track --video clip.mp4 --points 640,300 --out masks=masks/
```

`--points` takes several clicks, `"x,y;x,y"`, which pick one object together
(a person whose coat and trousers a single click would take apart).
`--exclude "x,y;x,y"` are clicks on things that are not the object. With one
click SAM 2 picks the best of its three ambiguity masks; with several it
asks for one mask, as the reference does.

The result is a mask sequence. Over the served API it is one blob (a
one-channel clip whose `meta.maskseq` is the manifest); written to a path it
is a directory:

```text
masks/masks.json          the manifest
masks/mask_000000.png     frame 0, 8-bit, source resolution
masks/mask_000001.png     ...
```

That directory is what `brain ltxv v2v --mask` takes.

**Polarity is declared in the manifest, never inferred.** By default `255`
(1) is the tracked object, SAM 2's own meaning. `--polarity object=0` writes
the exact inverse, for a consumer whose `1` means "keep this region"
(LTX-2.5's masked conditioning). A reader acts on the `polarity` key and
refuses a sequence without it: the two are exact inverses, and getting it
backwards regenerates the whole background while preserving the subject.

The manifest also carries a per-frame `object_score`. **Negative means SAM 2
believes the object is absent or fully occluded on that frame**, so an empty
mask there is the model's answer, not a bug; the run lists those frames as
`occluded_frames`.

| Param | Effect |
|---|---|
| `--video` | a clip (AV1 read in process, other codecs need ffmpeg), or a directory of numbered images |
| `points` | clicks on the object, `"x,y;x,y"` in source pixels (required) |
| `exclude` | clicks on what is not the object |
| `prompt_frame` | which frame the clicks sit on (default 0) |
| `max_frames` / `fps` | trim or resample the clip before tracking; `fps` is required for a directory of frames |
| `object_id` | recorded in the manifest, for multi-character work |
| `polarity` | `object=255` (default) or `object=0` |
| `soft` | keep the sigmoid ramp instead of hard masks |

Masks come out at the SOURCE frame size, one per source frame, contiguous
from 0. Any spatial or temporal downsampling belongs to the consumer, against
its own grid. `brain sam2 track --help` lists every param with its default.

## Options (`segment`)

| Param | Effect |
|---|---|
| `variant` | `tiny` (default) or `large` - the configuration the served checkpoint is loaded with; it has to be the checkpoint's own |
| `points` | `"x,y;x,y;…"` in source-image pixels |
| `labels` | `"1;0;…"` - 1 = foreground, 0 = background (default: all-foreground) |
| `box` | `"x1,y1,x2,y2"` |
| `multimask` | bool, default `true` |

Multiple prompts on the same image are cheap: N prompts on one frame cost one
image-encoder pass plus N decoder passes, so batching prompts (not requests)
is the efficient way to segment many regions of one photo.

## Hardware and limits

The video path tracks ONE object per run: `object_id` names it in the
manifest, and several characters means several runs (they share nothing but
the clip). Tracking is forward only - frames BEFORE the clicked one take the
clicked frame's mask rather than being tracked backwards - and a correction
click on an already-tracked frame is not implemented, so re-prompting means
re-running. A video mask PROMPT (segment from an existing mask rather than a
point) is likewise not wired up.

There's no training or fine-tune action and no HTTP surface; `segment` and
`track` are reached through `brain sam2 <action>` and D-Bus.
