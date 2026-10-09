# sample: imagegen/rgb-to-ir

Can RGB images be translated into thermal-IR images realistic enough to
fine-tune a YOLOv8 detector for IR input? The study fine-tunes detectors with
`brain yolov8 fine-tune` on different training images and scores every one of
them on real held-out IR frames of the supplied datasets. This sample is its
data side and works with ANY paired RGB / IR detection dataset: a generic input
contract (the pairs manifest), readers for common label layouts, sequence-only
splits, a packer for the detector's flat dataset format, and the training-image
arms that need no generative model.

**What is here, and what is not.** The manifest contract and validator, the
readers, the splits, the packer, the arms below and the training data of the
instruction-conditioned translator (tiles, region contrast, instruction
captions) are implemented and tested. The translator itself is trained with
`brain flux2 finetune` and is not part of this sample; neither is the
evaluation harness (fine-tune per arm, score on real IR, paired statistics).

## Study design

| split | meaning |
|---|---|
| **T** | translator-training pairs (RGB + real IR); also where the sensor model is fitted |
| **S** | detector-training frames: RGB + GT boxes, plus its real IR twin used only by the upper-bound arm |
| **V** | validation; all tuning happens here |
| **Test** | held out, touched once at the end |

Splits are by SEQUENCE, never by frame: neighbouring video frames are
near-duplicates, so a frame-level split would train on what it is scored on.

| arm | training image for frames of S |
|---|---|
| A1 | RGB as is |
| A2 | real IR twin (upper bound) |
| B1 | grayscale of the RGB |
| B2 | inverted grayscale |
| B3 | grayscale, CLAHE, then a sensor model fitted on T: extra blur, contrast match, noise, stripe noise |
| B4 | semantic renderer: class-prior intensities painted from the GT boxes on a cold-sky gradient, then the same sensor model |

IR images are 8-bit, single channel, white-hot; the packer replicates a single
channel to three. Classes are chosen by name (`--classes`, default
person,car,bicycle) and boxes of other classes are dropped.

## Run it

```bash
S=samples/shell/imagegen/rgb-to-ir
$S/rgb_to_ir.sh manifest --dataset d1 --rgb-glob 'DIR/rgb/*.jpg' --ir-glob 'DIR/ir/*.png' \
    --labels-dir DIR/labels --label-format voc --sequence-rule dir
$S/rgb_to_ir.sh validate --check-files           # every manifest against the contract
$S/rgb_to_ir.sh splits                           # splits.json + leaks.json
$S/rgb_to_ir.sh arms --limit 64                  # sensor-model.json + arms/<arm>/ for 64 frames of S
$S/rgb_to_ir.sh pack a2 --size 512               # packed/a2/{images.f32,boxes.bin,meta.json}
brain yolov8 fine-tune "$RIR_WORK/packed/a2" --nc 3 --input 512 \
  --weights <yolov8 checkpoint> --out ft-a2.safetensors
$S/rgb_to_ir.sh tiles --limit 1100               # lora-set/: aligned RGB / IR tiles of split T + pairs.yaml
$S/rgb_to_ir.sh measure                          # lora-set/regions/*.json (needs a resident SAM 2, see below)
$S/rgb_to_ir.sh captions                         # lora-set/captions.yaml + captions-report.json
$S/rgb_to_ir.sh sheet                            # lora-set/sheet.png, 12 random tiles to look at
brain flux2 finetune "$RIR_WORK/lora-set" --out rgb2ir.brain --size 512
$S/rgb_to_ir.sh test                             # unit tests, no data needed
```

`RIR_WORK` (default `${TMPDIR:-/tmp}/rgb-to-ir`) holds every output; nothing
is written into the repository. `SEED` (default 1) seeds splits, sensor fit,
frame selection and packing order. Run `manifest` once per dataset (each with
its own `--dataset` id); `splits` and `arms` use all manifests. Each stage is
also a script with `--help`: `rir_readers.py`, `rir_manifest.py validate`,
`rir_splits.py`, `rir_arms.py {fit-sensor,render}`, `rir_pack.py`.

## The pairs manifest

One JSON-lines file, one record per aligned RGB / IR frame; paths are absolute
or relative to the manifest. The validator reports every violation with its
line and field (`rir_manifest.py validate FILE [--check-files]`), including
duplicate ids and a sequence that straddles the official split.

```
required
  id            str    unique within `dataset`
  dataset       str    your dataset id (sensor models and splits are per dataset)
  rgb           str    RGB image path (8-bit; 16-bit is reduced by >> 8)
  ir            str    thermal image path; may be the same file as `rgb`
  boxes         list   [{"class": str, "x1": n, "y1": n, "x2": n, "y2": n}], pixels, x1<x2, y1<y2; may be []
  sequence_id   str    frames that may be near-duplicates share one
optional
  official_split  "train" | "test"   the dataset's own held-out hint; whole sequences must agree
  capture_time    str                free-form
  day_night       "day" | "night"
  tags            [str]
  width, height   int
  ir_read         {"channel": "gray" | "alpha", "invert": bool}
                  gray: any 1- or 3-channel image read as gray; alpha: IR in the 4th channel of the file
                  (also `rgb`'s file); invert: black-hot source, read as white-hot
  unusable        str                a reason this frame must not be used
```

### What to look for in a dataset

Aligned pairs (the same size and viewpoint, no per-frame offset: the readers
cannot fix registration, `rir_data.edge_alignment` measures it), pixel boxes
for the classes you will train, some way to tell frames of one video from
another (a scene or clip id, or consecutive frame numbers), real thermal frames
from a scene distribution like your target, and a licence that permits the
use. Check the licence before downloading or training: several public
RGB / IR sets are non-commercial and some terms reach derivative work such as
generated images or trained weights. There is deliberately no downloader here;
obtain the data yourself and describe it to the readers.

## Readers (`rir_readers.py`)

`--rgb-glob`, `--ir-glob` (omit for IR in the RGB file, with `--ir-channel
alpha`), `--labels-dir` / `--coco-json`, `--label-format voc|yolo|coco`,
`--yolo-names`, `--class-map src=dst,...`. RGB, IR and label files are paired
by a key, group 1 of `--pair-regex` on the base name (default: the name without
its extension); unpaired frames and frames without labels are counted in the
report.

`--sequence-rule` decides the sequence id:

| rule | meaning |
|---|---|
| `dir` | the RGB file's directory name |
| `regex:PATTERN` | group 1 of PATTERN searched in the RGB path (a scene prefix, say) |
| `similarity` | frames carry a number (`--numeric-id-regex`); a break when the number jumps by more than `--max-gap` (3) or the correlation of neighbouring 16x16 grayscale thumbnails is below `--min-corr` (0.5). A pair touching an RGB shrunk into a black sub-rectangle is judged on IR thumbnails, so a padded frame does not cut a video |
| `block:WIDTH` | numbers cut into contiguous blocks; `--block-guard G` marks frames within G of a block edge unusable so blocks never touch |

Official hint, optional: `--official-split-regex` (a path word, group 1; words
in `--test-words`, default test,val, are test) or `--test-list FILE` (pair keys
that are test). IR: 16-bit reads as 8-bit; `--ir-polarity auto|white-hot|black-hot`,
where auto compares the mean IR in boxes of `--hot-classes` (person) with the
frame median over up to 200 frames and inverts when they are mostly darker.

## Splits

`splits.json` maps every frame to its record fields plus `{split, usable,
reason, day_night, luminance}`. A dataset with an official hint (records, or
`--test-sequences FILE`) holds out exactly those sequences as Test; one
without holds out a seeded `--holdout-fraction` (0.2) of whole sequences.
Of the rest V takes 10% of the sequences, S takes sequences while they fit
under a frame cap (`--default-s-cap` 800, `--s-cap id=N`; set it below the
dataset size on small sets), and T gets the rest. Unusable frames carry
`split: null` and a reason: the record's own, or `rgb_inset_padding` when the
RGB occupies under 98% of the frame once black padding is ignored (a
field-of-view mismatch with the IR). Day/night comes from the record, else
Otsu on each sequence's median RGB luminance (left unset when the spread is
under 20 levels; on a dataset with underexposed RGB treat it as unreliable).

`leaks.json` lists held-out frames (V, Test) with a near-duplicate in T or S
(thumbnail correlation of at least 0.95, or a 6-bit dHash match with
correlation of at least 0.8). A sequence split cannot remove repeated scenery
across DIFFERENT sequences (two scenes shot at one place, say); the list is
how to see it, and a few leaks in V mean its scores need care.

## Pack and arms

`rir_pack.py`: letterbox to a square canvas (scale to fit, centre, grey 114
padding), boxes transformed with the same scale and padding and clipped (a box
thinner than 1 px is dropped), RGB CHW f32 in [0, 1]. The detector trainer
neither shuffles nor augments and consumes batches round-robin, so the item
order is shuffled with the seed; `order.json` records which frame sits at each
index.

`rir_arms.py fit-sensor` fits, per dataset and ONLY on split-T frames (it
refuses anything else), single-frame estimators: noise sigma (Immerkaer's
Laplacian difference, which stripes do not inflate), stripe amplitude (column
mean residual, an upper bound on real scenes), extra blur (edge-spread ratio at
two scales, in quadrature against the RGB's own blur) and the spread of the
IR's per-frame mean and standard deviation. Noise estimates below about 0.3
grey levels are the 8-bit quantisation floor, not a measured level. `render`
writes `<arm>/<dataset>_<id>.png` and `<arm>/manifest.jsonl` for the same
seeded frames in every arm; A1 is a 3-channel PNG, the other arms single
channel. B4's class priors are documented constants in `rir_arms.py`: a
deliberately simple version using GT boxes as regions.

## Instruction captions for the RGB-to-IR translator

The translator is a FLUX.2 Klein LoRA trained with `brain flux2 finetune` in its
paired mode: the RGB tile is the reference, the real IR tile the target, and
the caption is the INSTRUCTION. Its data comes from split T only, in three
steps; each is a stage of the driver and a module with `--help`.

**1. Tiles (`rir_tiles.py`).** `brain flux2 finetune <dir>` reads `pairs.yaml`
(`target: reference`, one per line), `captions.yaml` (targets only: a reference
is an input, not a sample) and the images, center-crops each to a square and
resizes to `--size` identically for target and reference. Tiles are therefore
cut square, with one crop box for both images of a pair; the IR target is
written as 3 channels. Along each axis `ceil(extent / size)` tiles are spread
evenly over the frame, so every pixel is covered (a 640x512 frame at 512 gives a
left and a right tile; a frame smaller than the tile is an error, not a resize).
A tile is rejected, with its reason in `tiles.jsonl`, when its RGB has black
padding (under 98 percent valid) or when the RGB / IR edge correlation shows
a shift gain over 0.08 or no correlation (`rir_data.edge_alignment`). Only
usable split-T frames are read.

**2. Region contrast (`rir_regions.py`, `rir_sam2.py`).** Each ground-truth box
of a tile prompts SAM 2 on the RGB tile for an object mask. On the real IR tile
the contrast is `c = (median IR in the eroded mask - median IR in a ring around
it) / MAD of the IR frame`. The mask is eroded and the ring starts a gap away
from it (the thermal point-spread skirt belongs to neither), the ring excludes
every other object's mask, and the MAD is the frame's own robust spread, so no
radiometric calibration is needed (the IR is 8-bit AGC output, not
temperature). The noise floor `tau` is measured per object: the 75th percentile
of `|c|` between random pairs of background patches of the object's size,
lying outside every object and at most two patch sides apart (the ring
comparison is local, so its null is). A statement is made only when
`|c| >= tau`; below it the object is "about the same temperature". Pairs drawn
from anywhere in the frame (`pair_reach=None`) compared sky with ground and set
a floor of about 1.4 MAD that almost no object reached, which is why they are
not the default.

SAM 2 through the CLI loads the checkpoint in every process: measured at 37 to
43 s per box on a GPU (14 s on the CPU backend), unusable for thousands of
boxes. `--backend dbus` (default) uses one resident `brain serve --dbus` and
brain-py (needs `pip install jeepney`; `BRAIN_PY` overrides the brain-py
location): the first box of a tile costs the image encoder (about 2.5 s), the
rest about 0.1 s each. Start the server with `BRAIN_SAM2_WEIGHTS=<ckpt> brain
--device gpu1 --backend cuda serve --dbus --dbus-address <addr>` and pass
`--dbus-address <addr>`; `--backend cli --brain-args '--device gpu1 --backend
cuda'` is the dependency-free fallback.

Part-level regions (bonnet, windows) need a part grounder. `--grounder
florence2 --parts 'car=bonnet,windows'` boxes the parts with `brain florence2
ground` on a crop of the object, segments each part box with SAM 2 and
measures it against a ring that excludes other objects and sibling parts. The
code path is tested with a stub grounder and a fake `brain` executable
only: Florence-2 weights were not available where this was built, so it has
NOT been run against the real model, and the phrasing of the grounding request
and the JSON shape it parses are unverified. The default is object-level only,
and no part is ever invented.

**3. Captions (`rir_captions.py`).** Every caption opens with `Convert to
thermal infrared, white-hot.` and then, for up to three classes of the tile,
a clause from a small seeded grammar: a subject ("The car", "The visible car")
and a predicate of the measured polarity ("is warmer than its surroundings",
"is about the same temperature as its surroundings", and three paraphrases of
each). A class is stated with the polarity two thirds of its instances in the
tile share and not stated when they split. 30 percent of all tiles keep the
neutral caption alone (tiles with nothing measured count toward it). Two
predicates, one warmer and one cooler, are HELD OUT: they are never written to
`captions.yaml`, only to `heldout-captions.jsonl` for evaluating whether the
adapter follows words it did not train on. `paraphrase=` is a hook for an LLM
paraphraser; its output is accepted only if it is one line that still names
every object. No counterfactual target is synthesised: a caption describes the
IR next to it.

`captions-report.json` gives, per class, the tiles stating warmer, cooler and
same and flags a class whose rarest polarity is under 10 percent of its
statements as "not controllable from natural data": the adapter cannot learn
to switch an outcome it almost never saw, and nothing here manufactures it.

### Measured on one real dataset (split T of a pairs manifest, 640x512 frames)

1100 random frames gave 2200 candidate tiles at 512; 2015 were accepted (185
rejected for edge misalignment, none for padding, the padded frames having been
dropped by the splits) and 1978 of them contain boxes. 13825 boxes were
prompted: 910 skipped as under 10 px on a side, 302 whose mask left the box
(box prompt failures), 36 with no usable ring or no room for background
patches, leaving 12577 measured objects: 3773 warmer, 6608 same, 2196 cooler.
`tau` has median 0.23 MAD (10th to 90th percentile 0.13 to 0.50) against a
median `|c|` of 0.22: about half of all objects are indistinguishable from
their surroundings at this noise floor. SAM 2 hiera-tiny, resident, on a GPU
shared with other jobs: 0.48 s per box and 3.3 s per tile over the whole run.
Captions: 30.0 percent neutral, 1147 name one object, 227 two, 37 three.

| class | tiles stating warmer / cooler / same | rarest polarity | flag |
|---|---|---|---|
| person | 442 / 52 / 462 | cooler, 5.4% | not controllable |
| car | 304 / 45 / 827 | cooler, 3.8% | not controllable |
| bicycle | 38 / 203 / 145 | warmer, 9.8% | not controllable |
| dog | 5 / 9 / 11 | warmer, 20% | none (n = 25: too few to say anything) |

### Limits

- The data is overwhelmingly daytime (split T is 92 percent day), where most
  objects sit at ambient temperature: "warmer" for people and "cooler" for
  cars are what natural data offers, the opposite polarities are rare, and
  the adapter will follow an instruction only in the directions it saw.
- The IR is 8-bit AGC output scaled per frame. A contrast is relative to the
  frame, not a temperature difference, and the same object can read warmer in
  one frame and cooler in the next when the AGC range moves.
- The pairs are registered to a few pixels, not exactly: the best-aligning IR shift
  over the accepted tiles is a consistent 2 to 4 px in x, which the
  translator will learn as part of the mapping. The tile check rejects visible
  offsets (the 185 above), not this residual.
- Masks are SAM 2 outputs for box prompts, not hand labels. 302 of 13825 left
  their box and were dropped, nothing else audits them (`sheet` shows 12).
  Thin or crowded objects (bicycles, groups of people) get ragged masks and
  rings that may include their neighbours' edges; the bicycle row above, 53
  percent cooler, is probably that and not physics.
- A statement that holds for two thirds of a class's instances in a tile is
  written for the class; the caption does not say which.
- Part-level captions need a grounder that was not available: object level only.

## What is verified

`rgb_to_ir.sh test` runs the spec tests, all on tiny synthetic pairs written by
the tests in three layouts (Pascal VOC with a separate single-channel IR, YOLO
txt with IR in an alpha channel, COCO json with a replicated 3-channel IR):

- manifest: every violation is reported with line and field;
- readers: pairing, labels in the three formats, the four sequence rules (a
  padded frame does not cut a video), official hints, 16-bit IR, alpha IR,
  polarity detection and override, the padding and misalignment checks;
- splits: no sequence in two splits, hinted frames only in Test, a held-out
  share by seed without a hint, determinism, the padded-frame drop, the S cap,
  and a planted near-duplicate being flagged;
- pack: the box round trip through the letterbox, byte layout, determinism;
- arms: B2 is the exact inverse of B1, the sensor fit recovers a known
  blur / noise / stripe within stated tolerances and reads only split-T
  frames, B4 paints boxes brighter than their surroundings and sky darker than
  ground, and a manifest -> splits -> fit -> render -> pack round trip.

- tiles: the grid covers any aspect, only split T is read, one crop box for both
  images, the pairs.yaml / captions.yaml format, rejection of padded and
  misregistered tiles with a reason, determinism;
- regions: contrast of a planted warm or cool blob in units of the frame MAD, the
  ring excluding neighbours, the noise floor of pure noise against its order
  statistics, a planted object not inflating tau, local against frame-wide pairs,
  polarity only at `|c| >= tau`, part measurement only through a grounder
  (stub), resumable measurement with a throughput summary;
- sam2: command lines, mask decoding and the grounder's coordinate mapping
  against a fake `brain` executable (the models are exercised by running the
  stage);
- captions: polarity matching planted blobs end to end, held-out wording never
  in captions.yaml, the 30 percent neutral share, at most three objects, the
  majority rule, the paraphraser hook, determinism, the balance flag;
- sheet: grid size and sampling from measured tiles only.

## Dependencies

`python3` with `numpy`, `opencv-python` and `Pillow`; the `brain` binary for the
fine-tune steps and for `measure`, which also needs `jeepney` for the default
D-Bus backend.
