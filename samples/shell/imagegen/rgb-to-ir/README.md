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
readers, the splits, the packer and the arms below are implemented and tested.
The generative arms (an RGB-to-IR translator) and the evaluation harness (fine-tune
per arm, score on real IR, paired statistics) are added by later work and are
NOT part of this sample or claimed by it.

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

## Dependencies

`python3` with `numpy`, `opencv-python` and `Pillow`; the `brain` binary only
for the fine-tune step.
