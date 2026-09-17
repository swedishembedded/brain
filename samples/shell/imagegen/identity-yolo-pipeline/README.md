# sample: imagegen/identity-yolo-pipeline

One photo of a person in, a validated YOLOv8 detector for THAT person out.
An eight-stage, restart-safe pipeline that generates its own training data
(PuLID identity-conditioned positives, plain-FLUX.2 hard negatives and
backgrounds), fine-tunes a COCO-pretrained YOLOv8n with the person appended
as a new class, and validates the result with a genuinely independent
two-stage recognizer.

```bash
BRAIN_FLUX2_ALLOW_NC=1 \
  samples/shell/imagegen/identity-yolo-pipeline/identity_yolo_pipeline.sh <photo-or-url> [out-dir]
```

## What it demonstrates

1. Fetch weights (idempotent). 2. Verify one face; canary-check identity
   conditioning before committing to a full generation budget. 3. Generate
   `N_TARGET` identity-preserving images (PuLID), gated on ArcFace similarity
   to the source photo. 4. Generate hard negatives + backgrounds (plain
   FLUX.2, no identity conditioning). 5. Auto-label person boxes with COCO
   YOLOv8n and pack `brain yolov8 train`'s flat binary format. 6. Fine-tune
   the COCO YOLOv8n, appending the target as class `TARGET_CLASS` with all 80
   COCO classes preserved exactly. 7. Generate fresh held-out target +
   stranger images. 8. Validate the two-stage recognizer and render labelled
   demo images into `<out-dir>/demo`. Exits non-zero if any gate fails.

- **Identity is a face-recognition problem, not a detection-class one.**
  Stage 8 is where that shows: a detection class head reads features trained
  to separate categories and cannot express *which* person - so the class
  head only finds "person", and a second stage (SCRFD + ArcFace, the same
  stack `../identity-score/identity_score.sh` uses) decides which one, over
  a `brain yolov8 detect --identity-ref` call.
- **One resident `brain serve --dbus` daemon**, driven by the colocated
  helper `identity_yolo_gen.py` (an internal helper for this sample, not a
  separate sample of its own) instead of a fresh subprocess per image, so
  PuLID's ~27 GB and FLUX.2's ~13 GB weight sets load once per *backend*
  instead of once per image. The pipeline restarts the daemon when it
  switches backend (`serve_for` in the script) rather than holding both
  resident at once - they don't fit together on a single 24 GiB card, and
  `BudgetPlacer` snapshots free VRAM once at process start, so a second model
  built inside an already-running daemon would plan against stale numbers
  anyway.
- Every stage is restart-safe: an image already on disk is never
  regenerated, and every seed is drawn fresh (`rand_seed`), not derived from
  a loop index, so a resumed run explores real seeds rather than a narrowed
  subset.
- Preservation of the base detector is exact, via `--freeze-backbone
  --freeze-reg-head --freeze-cls-hidden --train-classes <TARGET_CLASS> --wd
  0` on `brain yolov8 fine-tune` - verified in this pipeline's own run to
  99/99 COCO detections preserved, 0 lost, 0 gained. The script's own stage-6
  comment block is the full account of what each flag protects and why
  disabling `FREEZE_CLS_HIDDEN` measures *worse* on both axes rather than
  trading one for the other.

## identity_yolo_gen.py

The pipeline's image generator: one generated PNG from the resident daemon,
replacing what used to be two fresh CLI subprocesses per image. Can also be
run standalone, or used as a readiness probe (`--wait SECONDS`, which is how
the pipeline script fails in seconds when the daemon comes up without the
right `BRAIN_*` weight variables, instead of at the first two-minute
generation):

```bash
BRAIN_FLUX1_DIR=... BRAIN_PULID_DIR=... BRAIN_ARCFACE_DIR=... \
BRAIN_CLIP_DIR=... BRAIN_BISENET_DIR=... \
BRAIN_FLUX2_DIT=... BRAIN_FLUX2_VAE=... BRAIN_FLUX2_TE=... \
BRAIN_FLUX2_TOKENIZER=... BRAIN_FLUX2_ALLOW_NC=1 \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/shell/imagegen/identity-yolo-pipeline/identity_yolo_gen.py pulid \
    --prompt "a photo of a person in a park" --face portrait.jpg --out img.png
```

Both backends key their residency on exactly what a pipeline run holds
constant, so every call after the first reuses the loaded weights: PuLID
(`brain/flux1-pulid`) on a single `"default"` instance key with a
`(variant, height, width, precision)`-keyed session cache, FLUX.2 Klein
(`brain/flux2-klein`) on `"{variant}:{precision}:{w}x{h}:{nref}"` - `variant`
is bound to the real weights at daemon startup, and `--precision` is always
sent explicitly because the server's own default there is `fp32`, not the
`int8` this pipeline wants.

Note the top-level script does not use `tools/dbus-session.sh`: it manages
its own resident daemon across four restarts per run (switching between the
PuLID and FLUX.2 backends), which the single-shot `--serve` helper doesn't
fit - see `serve_for`/`stop_serve` in `identity_yolo_pipeline.sh` for that
logic. It re-execs itself under `dbus-run-session` when no session bus is
already present.

## What it needs

- `hf` (authenticated - FLUX.1-dev is gated), `python3` with
  numpy+Pillow+jeepney (`pip install -e brain-py`), `dbus-run-session` (only
  without an existing session bus), and `make build/release`.
- Weights, fetched idempotently by stage 1 where possible: FLUX.1-dev
  (gated, needs `hf auth login`), the PuLID adapter, a COCO YOLOv8n
  checkpoint (`brain pull Ultralytics/YOLOv8`), FLUX.2 Klein-9B (`brain pull
  black-forest-labs/FLUX.2-klein-9B`). ArcFace/SCRFD (antelopev2), EVA-CLIP
  and BiSeNet weights are **not** auto-fetched - the script `die`s with the
  exact path each is expected at (BiSeNet in particular has no fetch recipe;
  its header explains the one-time conversion from facexlib's torch
  pickle).
- `BRAIN_FLUX2_ALLOW_NC=1`, set explicitly by the caller to confirm
  non-commercial use: stage 4 uses FLUX.2 Klein-9B, which is
  NC-licensed (FLUX.2 [Non-Commercial] License, Black Forest Labs).

## Options (env-overridable, all optional)

| env | default | meaning |
|---|---|---|
| `GEN_SIZE` / `GEN_STEPS` | `512` / `8` | generation resolution / denoise steps |
| `TRAIN_SIZE` | `256` | fine-tune input resolution |
| `TRAIN_STEPS` / `TRAIN_BATCH` / `TRAIN_LR` / `TRAIN_SEED` | `1500` / `4` / `3e-3` / `1337` | fine-tune hyperparameters |
| `N_TARGET` / `N_NEGATIVE` / `N_BACKGROUND` / `N_HOLDOUT` | `50` / `40` / `10` / `5` | dataset sizes per split |
| `TARGET_CLASS` | `80` | appended class index (COCO keeps 0..79) |
| `TARGET_NAME` | `einstein` | label for the added class |
| `IDENTITY_FLOOR` | `0.05` | ArcFace floor gating stage-3 generations (catches total conditioning failure) |
| `IDENTITY_THRESHOLD` | `0.20` | ArcFace floor for stage-8's "is this them" gate |
| `DETECT_CONF` / `DETECT_IOU` / `DETECT_INPUT` | `0.25` / `0.45` / `640` | stage-8 detector settings (640 matches COCO's own trained resolution, not `TRAIN_SIZE`) |
| `FREEZE_CLS_HIDDEN` | `1` | trade capacity for exact COCO preservation (see above) |
| `TRAIN_CLASSES` | `TARGET_CLASS` | which class indices the loss trains |
| `SERVE_RESERVE_GB` | `3` | VRAM headroom the resident daemon reserves |
| `FLUX1_DIR`, `PULID_FILE`, `ARCFACE_DIR`, `CLIP_DIR`, `YOLOV8_COCO_DIR`, `BISENET_DIR`, `FLUX2_DIT`, `FLUX2_VAE`, `FLUX2_TE`, `FLUX2_TOKENIZER`, `FLUX2_VARIANT` | under `~/.local/share/brain/models/...` | weight paths; each still wins unconditionally when set, so a run stays reproducible on a box whose model store holds something else |
| `BRAIN` | `./target/release/brain` | binary path |

## Output

`<out-dir>/{train,holdout}/...` the generated + gated dataset,
`<out-dir>/<TARGET_NAME>-detector.safetensors` the fine-tuned weights,
`<out-dir>/demo/` labelled validation renders, plus a scorecard on stderr and
the ready-to-run `brain yolov8 detect --identity-ref ...` invocation on
success.
