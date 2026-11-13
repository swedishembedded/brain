<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/sam2

SAM 2.1 promptable segmentation as a standalone program: cut an object out
of an image from a click or a box, or follow one object through a whole clip
and get its mask on every frame.

The program is SAM 2's served actions with a command line built from their
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults, and nothing is reimplemented.

## Build

```bash
make samples/models/sam2/build
```

## Weights

```bash
brain pull facebook/sam2.1-hiera-tiny       # the default
brain pull facebook/sam2.1-hiera-large      # optional, larger and slower
```

The checkpoint is found in the model store. With both variants there, the
action's `--variant` (default `tiny`) picks one; `--weights PATH` names a
file outright.

## Run

Track the person you click on through a clip:

```bash
target/release/sample-models-sam2 track --video street.mp4 --points 405,195 --out masks=masks/
```

writes a mask-sequence directory (`masks/masks.json` and one PNG per frame)
that `brain ltxv v2v --mask masks/` takes. Several clicks pick one object
together, and clicks on things that are not the object narrow it:

```bash
target/release/sample-models-sam2 track --video street.mp4 \
  --points "405,100;405,195;405,368" --exclude "40,40" \
  --variant large --polarity object=0 --soft --out masks=masks/
```

Segment an image:

```bash
target/release/sample-models-sam2 segment --image photo.jpg --points 320,240 --out mask=mask.png
target/release/sample-models-sam2 segment --image photo.jpg --box 100,80,420,400 --out mask=mask.png
```

`sample-models-sam2 --help` lists the actions, `sample-models-sam2 <action>
--help` every parameter with its default. `--device gpu0` places the run.

## What it shows

How a program embeds one brain model through its served actions:
`invoke::Model` names the manifest, how the weights are resolved and how the
provider is built, and `invoke::main` does the rest - help from the static
manifest, the weights from the store, input files to blobs, outputs to files.
