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
readers, the splits, the packer, the arms below, the training data of the
instruction-conditioned translator (tiles, region contrast, instruction
captions) and the evaluation and decision harness (offline mAP scoring, the
sequence-clustered bootstrap, the pre-registered decision rules, the
label-preservation gates, the instruction-obedience measurement) are
implemented and tested on synthetic inputs. The translator itself is trained
with `brain flux2 finetune` and the detectors with `brain yolov8 fine-tune`;
neither is run by this sample, and no real result is claimed here: the numbers
that decide the study come from those runs, scored by the stages below. The
synthetic-IR arms (a driver that runs the translator over the frames of split S,
ingests its outputs as an arm and gates them) are in `rir_synth.py`.

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

The synthetic arms are made from the same frames of S by `rir_synth.py`:

| arm | training image for frames of S |
|---|---|
| C1 | zero-shot edit: the base model (no adapter) told the neutral instruction |
| C2 | the translator (adapter), neutral instruction |
| C3 | the translator, instructions built from class priors (no IR measured) |
| V0 | the REAL IR passed through the autoencoder alone: a control for whether a detector learns the autoencoder's artefacts |

IR images are 8-bit, single channel, white-hot; the packer replicates a single
channel to three. Classes are chosen by name (`--classes`, default
person,car,bicycle) and boxes of other classes are dropped.

## Run it

```bash
WORK=DIR                                  # any directory outside the checkout: every output goes there
R="samples/shell/imagegen/rgb-to-ir/rgb_to_ir.sh --work $WORK"
$R manifest --dataset d1 --rgb-glob 'DIR/rgb/*.jpg' --ir-glob 'DIR/ir/*.png' \
    --labels-dir DIR/labels --label-format voc --sequence-rule dir
$R validate --check-files           # every manifest against the contract
$R splits                           # splits.json + leaks.json
$R arms --limit 64                  # sensor-model.json + arms/<arm>/ for 64 frames of S
$R pack a2 --size 512               # packed/a2/{images.f32,boxes.bin,meta.json}
brain yolov8 fine-tune "$WORK/packed/a2" --nc 3 --input 512 \
  --weights <yolov8 checkpoint> --out ft-a2.safetensors
$R tiles --limit 1100               # lora-set/: aligned RGB / IR tiles of split T + pairs.yaml
$R measure                          # lora-set/regions/*.json (needs a resident SAM 2, see below)
$R captions                         # lora-set/captions.yaml + captions-report.json
$R sheet                            # lora-set/sheet.png, 12 random tiles to look at
brain flux2 finetune "$WORK/lora-set" --out rgb2ir.brain --size 512
$R generate c2 --config synth.json --modes neutral,prior   # generated/c2/<mode>/: synthetic IR for the frames of S
$R ingest c2 c2/neutral --sensor-model "$WORK/sensor-model.json"   # arms/c2/ (add --no-sensor to skip the sensor model)
$R gate c2                                       # rejects.jsonl, gate-stats.json; rejected frames leave the manifest
$R pack c2 --size 512                            # packed/c2/ like any other arm
$R evalset --split Test --size 512  # packed/eval-Test/: real IR frames of split Test + sequences.json
$R evaluate a2 1 ft-a2.safetensors  # results/Test/a2/seed1.jsonl: brain yolov8 eval --dump-preds
$R decide --config decision-config.json   # decision-Test/decision.md and decision.json
$R test                             # unit tests, no data needed
```

`--work DIR` (required) holds every output; nothing is written into the
repository. The driver's settings are flags placed before the stage and the
environment configures nothing: `--seed N` (default 1) seeds splits, sensor fit,
frame selection and packing order, `--brain PATH` (default `brain` on `PATH`)
is the binary the stages that run brain use and `--device gpuN` names the card
for them. Run `manifest` once per dataset (each with
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
index and `sequences.json` (written when the items carry a sequence) its
`dataset:sequence` - a prediction dump names images by index, and held-out images
are scored in clusters of sequences.

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
brain-py (needs `pip install jeepney`; `--brain-py` names the brain-py
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

## Synthetic IR arms (`rir_synth.py`)

Three subcommands, one driver stage each, turn the translator into training
arms for the same frames of S as the arms above.

**`generate`** runs `brain flux2 generate` once per frame and mode. The CLI takes
one prompt and one output per process, so the model is loaded for every image;
`--shard i/n` splits the selected frames over n processes (one per card) and
`--limit N` takes a seeded subset first. A frame is cut to the largest centred
window whose sides are multiples of 16 (a 640x512 frame is not cut); the crop
(offset, size, source size) is recorded with the output so that boxes and pixels
map back to the source exactly. Nothing is padded or resampled, and a frame
under 16 px on a side is an error.

```bash
$R generate c2 --config synth.json --modes neutral,prior --shard 0/2 --device gpu1   # and 1/2 on another card
```

The config names the components, the adapter and the sampler settings, with
paths relative to the file; `adapter` is omitted for the zero-shot arm:

```json
{"device": "gpu1", "backend": "cuda", "variant": "klein-4b",
 "dit": "models/dit.gguf", "vae": "models/vae.safetensors",
 "text_encoder": "models/text_encoder", "tokenizer": "models/tokenizer.json",
 "adapter": "rgb2ir.brain", "strength": 1.0, "seed": 1}
```

The generation `seed` is the same for every frame and mode, so the modes are
paired; the driver's `--seed` selects frames and words the prior instructions.

| mode | instruction | source image |
|---|---|---|
| `neutral` | the neutral caption | RGB |
| `prior` | the neutral caption plus one clause per class of the frame that has a prior | RGB |
| `vae-roundtrip` | the neutral caption (unused at strength 0) | the REAL IR, replicated to 3 channels; strength 0, no adapter |

Priors come from `lora-set/captions-report.json` (the driver passes it when it
exists). For each class the prior is the commonest of warmer and cooler among
those that at least a tenth of the class's statements state, which is what the
adapter has seen enough to follow; the wording is a training template of
`rir_captions.py` and never a held-out one. A polarity that natural data almost
never shows (here, for instance, a cooler person) is never asked for, and a
frame without a class that has a prior gets the neutral caption. No IR is
measured for it: a prior instruction uses the class and nothing else about the
frame. `vae-roundtrip` is the exact codec round trip that `--strength 0` is
documented as in `brain flux2 generate --help`, applied to real IR.

Each output is `<mode>/<frame>.ppm` with a `<frame>.json` record (frame, mode,
instruction, seed, strength, crop, source, the adapter path and its sha256, the
command line, duration and attempts); `progress.log` appends one line per
frame. A rerun skips frames whose output is a complete PPM of the recorded size;
a damaged output is regenerated. A record made with another instruction, seed,
strength, crop or adapter hash aborts the run (`ConfigMismatch`): one output
directory holds one configuration. A run that fails because the card has no
room (the placement or out-of-memory messages) is retried after `--retry-sleep`
seconds, at most `--max-retries` times, then fails saying so; any other
failure, a timeout, or an exit 0 without a valid output stops the run at once
with the stderr. Unfinished frames carry no record, so a stopped run resumes.

**`ingest`** turns a directory of outputs (a mode directory of `generate`, or any
directory of `<dataset>_<frame>.ppm` / `.png`) into an arm in the format
`rir_arms.py render` writes: single-channel 8-bit white-hot PNGs and a
`manifest.jsonl` whose boxes went through the recorded crop (a box with under
half its area inside the crop is dropped and counted). RGB output becomes luma.
`--sensor-model sensor-model.json` applies the fitted sensor model with
`rir_arms.apply_sensor` and the same per-frame noise stream as B3 and B4;
`--no-sensor` leaves the luma alone, and one of the two must be given. Outputs
without a record are mapped by the default crop, or taken as the whole frame,
whichever has the output's size. Frames without an output are counted in
`ingest-stats.json`, never invented.

**`gate`** runs the label-preservation gates over the arm. For every frame: the
edge correlation inside each ground-truth box must reach the 10th percentile of
the same quantity over real pairs of split T (the threshold is read off at most
`--reference-limit` seeded split-T frames, and it is an error if split T has
none), and the whole image must be no more misaligned with the RGB than real
pairs are. The shift is the phase correlation of the edge maps; the real pairs of
split T are themselves registered a few pixels apart and the translator learns
that offset, so a fixed limit from zero would fail every translator. With
`--max-shift auto` (the default) the centre is the median (dx, dy) of the real
pairs and the limit the 95th percentile of their own distance from it plus a
half-pixel floor; a number is a fixed limit from zero. `gate-stats.json` records
the calibration (`shift_calibration`: mode, median, p95, floor, limit, n) and the
percentiles of the real edge-correlation distribution (`edge_reference`).
With `--sam2 --dbus-address ADDR [--brain-py DIR]` the SAM 2
mask IoU of each box on the RGB and on the synthetic IR is gated too, through the
resident D-Bus segmenter of `rir_sam2.py`, only for boxes that passed the
model-free gates (a segmenter call is the expensive part). Failing frames are
written to `rejects.jsonl` with their reasons and, by default, leave
`manifest.jsonl`, the file `pack` reads; `--keep-rejected` keeps them there.
`manifest.all.jsonl` is the whole arm and a rerun starts from it.
`gate-stats.json` is the per-arm rejection statistics (`n_images`, `n_failed`,
per reason, the threshold and counts) and the file `gate_statistics` of the
decision config points at for K4.

Filling the arms of the study, with the ingest of C2 and C3 using the sensor
model fitted on T and V0 left as the autoencoder made it:

```bash
$R generate c1 --config zero-shot.json --modes neutral                    # config without "adapter"
$R generate c2 --config synth.json --modes neutral,prior                   # C2 and C3
$R generate v0 --config synth.json --modes vae-roundtrip
$R ingest c1 c1/neutral --sensor-model "$WORK/sensor-model.json"
$R ingest c2 c2/neutral --sensor-model "$WORK/sensor-model.json"
$R ingest c3 c2/prior   --sensor-model "$WORK/sensor-model.json"
$R ingest v0 v0/vae-roundtrip --no-sensor
for a in c1 c2 c3; do $R gate $a; done
for a in c1 c2 c3 v0; do $R pack $a --size 512; done
```

The gates are model-free proxies for "the boxes still describe the image"; the
edge gate checks that edges line up, not that an object looks like itself, and
the hallucination gate (a reference detector on the synthetic image) is still a
function of `rir_gates.py` that this stage does not run. V0 is not gated: it is
a real frame through a codec and its boxes are the real ones.

## Evaluation and decision

Every arm is fine-tuned (`brain yolov8 fine-tune`) from one or more seeds and
scored on the REAL IR frames of a held-out split. `evalset` renders the real-IR
twins of split Test (or V, for tuning) and packs them; `evaluate` runs
`brain yolov8 eval --split all --conf 0.001 --dump-preds` for one arm and seed
into `results/<split>/<arm>/seed<N>.jsonl`; `decide` reads the whole results
directory. The low confidence floor matters: the default 0.25 is a detection
operating point and truncates the precision-recall curve.

**Scoring (`rir_eval.py`).** The dump has one line per image (`image` index,
`gts`, `preds`). `rir_eval.py preds.jsonl [--nc N]` reproduces
`eval::detection_report` in numpy: greedy class-aware matching by score within
each image (never across images), the all-points area under the precision
envelope, ten IoU thresholds 0.50 to 0.95, classes without ground truth left
out of the mean. Matching depends on one image only, so a run is matched once
and then re-scored under any image weights, which is what the bootstrap needs.
The sample's test checks it against the real binary: a detector is trained and
evaluated on CPU, and every number the binary prints (both mAPs, precision,
recall, per-class AP, counts) is reproduced to the four printed decimals. The
binary prints four decimals, so that is the resolution of the comparison; it
sums the area in float32 where this code sums in float64, which differs by
rounding (about 1e-7). The test is skipped, with its reason, when no binary is
found (`brain` on `PATH`, or a built `target/release/brain`
above the checkout).

**Statistics (`rir_bootstrap.py`).** The resampling unit is the SEQUENCE: a
resample draws as many test sequences as there are, with replacement, and
counts every image of a drawn sequence as often as the sequence was drawn
(2000 resamples, seeded). A frame-level bootstrap would count near-duplicate
frames as independent evidence; a test plants a leaked copy of every frame and
checks the interval does not shrink. One set of resamples scores EVERY run of
EVERY arm, so arms are paired, and an arm's value on a resample is the mean of
its seeds' mAP (the alternative, bootstrapping each seed separately, breaks the
pairing). Seed-to-seed variation is reported apart as the sample standard
deviation of the seeds' mAP on the full test set. A difference COUNTS only if
its 95 percent percentile interval excludes 0, its size exceeds twice the seed
noise of a difference of two single trainings, 2 x sqrt(sd_a^2 + sd_b^2), and
its two-sided bootstrap p survives Holm's correction over the pre-registered
family of three comparisons (a comparison that cannot be tested still counts
toward the family size). With one seed the seed noise is not measured, and the
rule is reported as not evaluable rather than passed.

**Decision (`rir_decide.py`).** The config (JSON, paths relative to it):

```json
{"sequences": "packed/eval-Test/sequences.json",
 "roles": {"A1": "a1", "A2": "a2", "B3": "b3", "C2": "c2",
           "REAL_K": "real-k", "REAL_K_PLUS_SYNTHETIC": "real-k+c2"},
 "resamples": 2000, "seed": 1, "nc": 3,
 "instruction_model": true,
 "gate_statistics": "arms/c2/gate-stats.json", "obedience": "obedience.json"}
```

A role names the results directory (`results/<split>/<arm>/`) of the arm that
plays it: the arm names of `generate` / `ingest` / `gate` / `pack` are the
directory names (`c2` above is the C2 arm ingested from `neutral` outputs, and
`gate_statistics` is its `gate-stats.json`). C1, C3 and V0 take no role in the
rules: `evaluate` them like any arm and they appear in the per-arm tables next
to the others by their directory names (`c1`, `c3`, `v0`), as do all arms found
in the results. A role set to C3's directory (`"C2": "c3"`) decides the study on
the prior-instruction arm instead.

`decision.md` (and `decision.json`) holds per-arm mAP@0.5:0.95 and mAP@0.5 with
intervals and seed std; the three pre-registered comparisons, P1 (C2 against
the trivial-transform arm B3), P2 (real-k plus synthetic against real-k alone)
and P3, the gap closure g = (C2 - A1) / (A2 - A1) with its bootstrap interval;
and the kill criteria. K1: A2 - A1 under 3 mAP points (the premise fails). K2:
g under 0.25 and P2 not positive. K3: C2 not better than B3. K4: more than 30
percent of C2's images fail the geometry gates (from the gate-statistics JSON
of `rir_gates.py`). K5, instruction model only: the controllability (or, with
no counterfactuals, the direction margin) interval includes 0 for every region
type (from `rir_obey.py`). Verdict: "reliable transform" iff P1 favours C2 AND
g >= 0.5 AND P2 is positive; "useful but not reliable" iff P2 is positive and
g < 0.5; otherwise "no reliable transform", naming the triggered kill
criteria. Kill criteria that fire next to one of the first two verdicts are
listed beside it and do not change it. An arm without predictions is NOT
measured: it is absent from the tables, never a 0, and every rule that needs it
is "not evaluable"; when that leaves the verdict open the verdict is "not
evaluable" (when the known parts already settle it, it stands). A K1 verdict
with a gap under 3 points makes g undefined and the transform cannot be
reliable. All runs must have been scored on the same images; others are refused.

**Label-preservation gates (`rir_gates.py`).** For a synthetic IR image made
from a labelled RGB image: (a) inside each box the edge correlation of the RGB
luma and the synthetic IR must reach the 10th percentile of the same quantity
over REAL pairs (`reference_distribution` on `manifest_pairs`); (b) the whole
image must be within 2 px of the RGB by phase correlation of the edge maps;
(c) detections of a reference detector on the synthetic image with no ground
truth of their class at IoU 0.5 and confidence of at least 0.5 are
hallucinations; (d) the SAM 2 mask-IoU gate is a pluggable callable
`mask_iou(rgb, ir, box) -> float | None`, wired to the resident segmenter by
`rir_synth.py gate --sam2` (`segmenter_mask_iou`), and run only on boxes that
passed the model-free gates; without it that gate is absent, not passed.
`RejectionStats` counts rejections per reason and writes the JSON K4 reads.
The edge gate is model-free and modest: it checks that edges line up, not that
the object looks like itself.

**Instruction obedience (`rir_obey.py`).** For generated IR tiles, the region
mask on the source RGB tile and the instruction (warmer, cooler, same), it
measures `c_gen` with `rir_regions.object_contrast`, direction accuracy
(`polarity(c_gen, tau) == instruction`, "same" being inside the noise floor;
tau is the 75th percentile of the null, so a perfect "same" is read as same
only about three times in four), controllability s x (c(a) - c(b)) against a
counterfactual generation of the same tile with the flipped instruction, the
luminance shortcut (partial correlation of `c_gen` with the RGB luma contrast
of the region, holding the instruction fixed) and fidelity |c_gen - c_real|,
with sequence-clustered intervals, for oracle instructions (measured from the
real IR) and prior instructions (`class_priors` of the caption report)
separately.

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
- eval: Rust goldens (perfect, shifted, cross-image, ties at the threshold, class
  handling), the ranking across images, weights equal to repeated images, the
  image-index to sequence map, and parity with the table printed by the built
  binary on a CPU-trained detector;
- bootstrap: whole sequences get one weight, seeded resamples, a planted
  improvement detected and equal arms not, about 95 percent coverage of the
  interval on null data over 120 seeded simulations (116 covered), a planted
  leak of one frame per sequence leaving the interval unchanged while frame-level
  resampling narrows it, seed std apart, the 2 x seed-std rule, Holm, gap closure;
- decide: every branch of the verdict (reliable, useful, none naming K2 and K3,
  not evaluable, settled despite a missing arm), K1 to K5 each way, unmeasured
  arms absent rather than 0, mismatched test sets refused, the written files;
- gates: planted shifts, planted objects, a planted mismatching texture in one
  box, the 10th percentile of real pairs, the pluggable mask callable, the
  rejection statistics JSON;
- obey: fake generators that obey, ignore the instruction or copy luminance,
  the "same" band, oracle against prior, the fall-back to direction without
  counterfactuals, duplicate pairs not narrowing the interval, the CLI.

- synth: against a fake `brain` executable, resume (a damaged output is redone),
  bounded retry on memory pressure and its exhaustion, a non-memory failure and an
  exit 0 without an output aborting with the stderr, the crop record mapping
  outputs and boxes back to source pixels, prior instructions from the report's
  controllable polarities with training templates only, the adapter's sha256 in
  every record, a changed configuration refused, shards partitioning the frames,
  the vae-roundtrip source and strength; ingest matching `rir_arms.apply_sensor`
  bit for bit and packing with the unchanged packer; gate with planted shifts and
  planted textures, the split-T threshold, exclusion from the manifest, and the
  mask gate through a fake segmenter. The driver is tested for its flags (the
  environment configures nothing) and the generate, ingest, gate, pack chain.

All of these use synthetic inputs. They show that the harness computes what it
says; they say nothing about whether a translator works. The detector-in-the-loop
hallucination gate is an interface here, not a wired run, the SAM 2 mask gate was
exercised against a fake segmenter only, the generation, ingest and model-free
gate stages were run on a few real frames, and the full pipeline (training the detectors and the translator on real
data, then `evaluate` and `decide`) has not been run. The first real run
should check the sequence map of its evaluation set and that every arm and seed
was scored on the same `packed/eval-<split>`.

## Dependencies

`python3` with `numpy`, `opencv-python` and `Pillow`; the `brain` binary for the
fine-tune steps, for `evaluate`, for `generate` and for `measure`, which also needs `jeepney` for the default
D-Bus backend (as does `gate --sam2`).
