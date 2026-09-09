#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end edge-AI model training and
# validation pipelines like this one for its clients. If your team needs
# expertise in identity-preserving generation, dataset auto-labeling, or
# on-device object detection, you can procure our services by sending an
# email to info@swedishembedded.com.

#
# One photo of a person in, a validated YOLOv8 detector for THAT person out.
#
#   examples/imagegen/identity_yolo_pipeline.sh <photo-or-url> [out-dir]
#
# The whole loop, driven only through `brain` (plus `hf download` for the two
# checkpoints `brain pull` cannot fetch today - see the module docs of
# `crates/pulid/src/caps.rs` and `crates/flux1/src/pipeline.rs` for why):
#
#   1. Fetch FLUX.1-dev + PuLID weights (idempotent - skipped if present).
#   2. Verify the source photo has one face; generate 3 canary images and
#      check identity is actually being conditioned before committing to 50.
#   3. Generate 50 identity-preserving training images (varied settings and
#      angles) with `brain pulid text2image`, gated per-image on ArcFace
#      cosine similarity to the source photo.
#   4. Generate hard negatives (other people) and pure backgrounds with plain
#      `brain flux2 generate` - no identity conditioning - so the trained
#      class learns THIS person, not "a person" or "the settings".
#   5. Auto-label every survivor's person bounding box with COCO YOLOv8n
#      (`brain yolov8 detect`, class 0), and pack the whole pool into the
#      three flat files `brain yolov8 train` actually reads (`images.f32` +
#      `boxes.bin` + `meta.json` - NOT `images/`+`labels/`+`data.yaml`; see
#      `crates/data/src/{gen_detect,binio}.rs`, which nothing in this repo
#      produces from real photos before this script).
#   6. Train a from-scratch YOLOv8-tiny detector on the pool (`brain yolov8
#      fine-tune`-from-COCO does NOT work - the tiny graph's channels don't
#      match yolov8n's, see this script's own preflight check).
#   7. Generate 5 FRESH held-out images (new settings/angles, never seen in
#      training) plus 5 fresh strangers for a false-positive control.
#   8. Validate: run `brain yolov8 detect` on every one of the 10 held-out
#      images and assert person-class detections land exactly where they
#      should (the target's 5) and nowhere else (the strangers' 5, plus the
#      backgrounds). Exits non-zero if any gate fails - this is the actual
#      "does it work" test, not just "the commands didn't crash".
#
# Fixed choices (fastest-that-validates, not maximum quality): 512x512
# generation canvas, 8 PuLID denoise steps, 256x256 YOLO training canvas,
# 1500 training steps. See the CONFIG block below to change any of them.
#
# Required env: none - weights are fetched into brain's own model store
# (`~/.local/share/brain/models`, wherever that resolves to) and every
# BRAIN_* weights variable is exported by this script once fetched.
# Optional env: BRAIN (binary path), OUT_DIR, plus everything CONFIG below
# reads as an override.
#
# Needs: `hf` (huggingface_hub CLI, already authenticated - FLUX.1-dev is a
# gated repo, so `hf auth login` with an account that accepted its license is
# a real prerequisite), `python3` with `numpy`+`Pillow`, `curl` (only if the
# first argument is a URL), and a build of `brain` with PuLID CLI-reachable
# (this script's own Stage 0 checks that and tells you which `make
# build/release` to run if not).

set -euo pipefail

# ============================================================= arguments

SRC="${1:?usage: identity_yolo_pipeline.sh <photo-or-url> [out-dir]}"
OUT_DIR="${2:-out/identity-yolo-$(date +%Y%m%d-%H%M%S)}"
BRAIN="${BRAIN:-./target/release/brain}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
log() { printf '\n=== %s ===\n' "$*" >&2; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# ============================================================= config

GEN_SIZE=512            # PuLID/flux2 generation canvas (square)
GEN_STEPS=8              # PuLID denoise steps (dev variant default is 50)
TRAIN_SIZE=256           # YOLO tiny training/eval canvas (square, --input)
TRAIN_STEPS=1500
TRAIN_BATCH=4
TRAIN_LR=3e-3
TRAIN_SEED=1337
N_TARGET=50               # training images of the person (the request's "50")
N_NEGATIVE=15             # hard negatives: other people, zero boxes
N_BACKGROUND=5            # pure backgrounds, zero boxes
                          # (the 5 held-out target images and 5 stranger
                          # controls in stage 7 are a curated prompt list, not
                          # a count - see HOLDOUT_PROMPTS/STRANGER_PROMPTS)
IDENTITY_FLOOR=0.05        # ArcFace cosine floor below which a generation is
                            # dropped/retried - a generous floor: this is the
                            # first real PuLID run in this workspace (see
                            # crates/pulid/src/caps.rs's module docs), so the
                            # exact "same person" number is unmeasured here;
                            # this floor only needs to catch total failures.
DETECT_CONF=0.1            # matches the repo's own proven Makefile combo for
                            # this tiny/overfit-prone architecture, not the
                            # library default of 0.25
DETECT_IOU=0.45

FLUX1_DIR="${FLUX1_DIR:-$HOME/.local/share/brain/models/black-forest-labs/FLUX.1-dev}"
PULID_FILE="${PULID_FILE:-$HOME/.local/share/brain/models/guozinan/PuLID/pulid_flux_v0.9.1.safetensors}"
ARCFACE_DIR="${ARCFACE_DIR:-$HOME/.local/share/brain/models/DIAMONIK7777/antelopev2}"
CLIP_DIR="${CLIP_DIR:-$HOME/.local/share/brain/models/QuanSun/EVA-CLIP}"
YOLOV8_COCO_DIR="${YOLOV8_COCO_DIR:-$HOME/.local/share/brain/models/Ultralytics/YOLOv8}"

export BRAIN_FLUX1_DIR="$FLUX1_DIR" BRAIN_PULID_DIR="$PULID_FILE" \
       BRAIN_ARCFACE_DIR="$ARCFACE_DIR" BRAIN_SCRFD_DIR="$ARCFACE_DIR" \
       BRAIN_CLIP_DIR="$CLIP_DIR"

PY_TOOLS="$OUT_DIR/pytoolkit.py"

# ============================================================= stage 0: preflight

log "Stage 0: build + preflight"

[ -x "$BRAIN" ] || die "no $BRAIN binary - run: make build/release"
"$BRAIN" pulid text2image --prompt x 2>&1 | grep -q "registered but not reachable" && \
  die "brain pulid text2image is not CLI-reachable - rebuild after the ARCH_TO_MODEL wiring change (crates/cli/src/resolve.rs) lands"
command -v hf >/dev/null || die "the 'hf' CLI (huggingface_hub) is required - pip install -U huggingface_hub[cli]"
command -v python3 >/dev/null || die "python3 is required"
python3 -c "import numpy, PIL" 2>/dev/null || die "python3 needs numpy and Pillow (pip install numpy Pillow)"
[ -f "$HOME/.cache/huggingface/token" ] || [ -n "${HF_TOKEN:-}" ] || \
  die "no HuggingFace token found (~/.cache/huggingface/token or \$HF_TOKEN) - FLUX.1-dev is a gated repo: hf auth login after accepting its license at https://huggingface.co/black-forest-labs/FLUX.1-dev"

"$BRAIN" devices >&2 || true

# ============================================================= stage 1: weights

log "Stage 1: weight provisioning"

fetch_flux1() {
  [ -f "$FLUX1_DIR/transformer/config.json" ] && { echo "flux1-dev: already present at $FLUX1_DIR" >&2; return; }
  hf download black-forest-labs/FLUX.1-dev --local-dir "$FLUX1_DIR" \
    --include "model_index.json" \
    --include "transformer/*" --include "vae/*" \
    --include "text_encoder/*" --include "text_encoder_2/*" \
    --include "tokenizer/*" --include "tokenizer_2/*" \
    || die "FLUX.1-dev download failed - if this is a 401/403, accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev with the account behind your HF token"
}

fetch_pulid() {
  [ -f "$PULID_FILE" ] && { echo "pulid: already present at $PULID_FILE" >&2; return; }
  local dir; dir="$(dirname "$PULID_FILE")"
  hf download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir "$dir"
}

fetch_yolo_coco() {
  find "$YOLOV8_COCO_DIR" -name '*.safetensors' 2>/dev/null | grep -q . && { echo "yolov8-coco: already present" >&2; return; }
  "$BRAIN" pull Ultralytics/YOLOv8 || die "brain pull Ultralytics/YOLOv8 failed"
}

fetch_flux1
fetch_pulid
fetch_yolo_coco

[ -f "$ARCFACE_DIR/glintr100.onnx" ] || die "ArcFace weights missing at $ARCFACE_DIR (expected already present on this box)"
[ -f "$ARCFACE_DIR/scrfd_10g_bnkps.onnx" ] || die "SCRFD weights missing at $ARCFACE_DIR (expected already present on this box)"
[ -f "$CLIP_DIR/EVA02_CLIP_L_336_psz14_s6B.pt" ] || die "EVA-CLIP weights missing at $CLIP_DIR (expected already present on this box)"

YOLO_COCO_WEIGHTS="$(find "$YOLOV8_COCO_DIR" -name '*.safetensors' | head -1)"
[ -n "$YOLO_COCO_WEIGHTS" ] || die "no COCO YOLOv8 checkpoint found under $YOLOV8_COCO_DIR after pull"
echo "yolov8-coco checkpoint: $YOLO_COCO_WEIGHTS" >&2

# ============================================================= the python toolkit

# One embedded toolkit (identity scoring, face/person-box parsing, dataset
# packing) rather than scattering python heredocs per stage - matches
# `identity_score.sh`/`train_identity_lora.sh`'s own inline-python convention
# in this directory, just consolidated since this script calls it from many
# stages.
cat > "$PY_TOOLS" <<'PY'
import json, os, struct, subprocess, sys
import numpy as np
from PIL import Image

BRAIN = os.environ["BRAIN"]


def arcface_embed(img_path):
    """L2-normalised 512-d ArcFace vector, or None if no face is found."""
    out = img_path + ".arcembed.bin"
    try:
        subprocess.run([BRAIN, "arcface", "embed", "--in", f"image={img_path}", "--out", f"embedding={out}"],
                        capture_output=True, text=True, timeout=120)
        if not os.path.exists(out):
            return None
        v = np.fromfile(out, dtype="<f4")
        n = np.linalg.norm(v)
        return v / n if n else None
    finally:
        if os.path.exists(out):
            os.remove(out)


def cmd_identity_score(argv):
    """identity-score <ref-embed.bin> <candidate-image> -> prints a float or NA"""
    ref = np.fromfile(argv[0], dtype="<f4")
    cand = arcface_embed(argv[1])
    print("NA" if cand is None else f"{float(np.dot(ref, cand)):.4f}")


def cmd_save_ref_embed(argv):
    """save-ref-embed <photo> <out.bin> - the one source-photo embedding."""
    e = arcface_embed(argv[0])
    if e is None:
        sys.exit(f"no face found in {argv[0]}")
    e.astype("<f4").tofile(argv[1])


def cmd_face_count(argv):
    """face-count <photo> -> prints 'N x1 y1 x2 y2' for the largest face, or 'NONE'."""
    r = subprocess.run([BRAIN, "scrfd", "detect", "--in", f"image={argv[0]}", "--max_faces", "16"],
                        capture_output=True, text=True, timeout=120)
    faces = []
    for line in r.stdout.splitlines():
        if line.startswith("faces:"):
            faces = json.loads(line[len("faces:"):])
    if not faces:
        print("NONE")
        return
    def area(f):
        b = f["bbox"]
        return (b[2] - b[0]) * (b[3] - b[1])
    best = max(faces, key=area)
    b = best["bbox"]
    print(f"{len(faces)} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}")


def _parse_detect_lines(text):
    """[x1,y1,x2,y2,conf,class] per stdout line, per `yolo_cli::print_dets`."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        x1, y1, x2, y2, conf, cls = json.loads(line)
        out.append({"bbox": [x1, y1, x2, y2], "conf": conf, "class": int(cls)})
    return out


def cmd_person_box(argv):
    """person-box <weights> <image> <conf> [<class-id>] -> the highest-confidence
    matching-class detection as 'x1 y1 x2 y2 conf', or NONE. class-id defaults
    to 0 (COCO 'person'; the trained detector's own single class is also 0)."""
    weights, img, conf = argv[0], argv[1], argv[2]
    cls_id = int(argv[3]) if len(argv) > 3 else 0
    r = subprocess.run([BRAIN, "yolov8", "detect", "--weights", weights, "--image", img,
                         "--conf", conf, "--iou", "0.45"], capture_output=True, text=True, timeout=300)
    dets = [d for d in _parse_detect_lines(r.stdout) if d["class"] == cls_id]
    if not dets:
        print("NONE")
        return
    best = max(dets, key=lambda d: d["conf"])
    b = best["bbox"]
    print(f"{b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} {best['conf']:.4f}")


def _letterbox(img, size):
    """Resize `img` (PIL) to fit `size`x`size` with grey padding, centered.
    Returns (canvas, scale, pad_x, pad_y) - the transform a caller applies to
    box coordinates in the SAME direction (original px -> canvas px)."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def cmd_letterbox(argv):
    """letterbox <src-image> <size> <out-image> - no box, for inference inputs."""
    img = Image.open(argv[0])
    size = int(argv[1])
    canvas, _, _, _ = _letterbox(img, size)
    canvas.save(argv[2])


def cmd_pack_dataset(argv):
    """pack-dataset <manifest.jsonl> <out-dir> <size>

    manifest.jsonl: one JSON object per line, in the FINAL (already-shuffled)
    dataset order: {"image": path, "box": [x1,y1,x2,y2] or null} - box is in
    the SOURCE image's own pixel coordinates (as `brain yolov8 detect`/scrfd
    report it), remapped here through the same letterbox transform used to
    build the canvas, then normalized center-xywh in [0,1] - the exact
    `data::gen_detect`/`data::binio` on-disk format `brain yolov8 train`
    reads: images.f32 (raw LE f32, N*3*size*size, CHW, RGB planes, [0,1]),
    boxes.bin ([u32 num] then num*(u32 class, f32 cx,cy,w,h) per image),
    meta.json ({"n","c":3,"h","w","nc":1}).
    """
    manifest_path, out_dir, size = argv[0], argv[1], int(argv[2])
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(manifest_path) if l.strip()]

    images_f = open(os.path.join(out_dir, "images.f32"), "wb")
    boxes_f = open(os.path.join(out_dir, "boxes.bin"), "wb")
    for row in rows:
        img = Image.open(row["image"])
        w0, h0 = img.size
        canvas, scale, pad_x, pad_y = _letterbox(img, size)
        arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC, RGB, [0,1]
        chw = np.transpose(arr, (2, 0, 1))  # CHW
        images_f.write(chw.astype("<f4").tobytes())

        box = row.get("box")
        if box is None:
            boxes_f.write(struct.pack("<I", 0))
            continue
        x1, y1, x2, y2 = box
        # source px -> letterboxed-canvas px -> normalized center-xywh.
        x1, x2 = x1 * scale + pad_x, x2 * scale + pad_x
        y1, y2 = y1 * scale + pad_y, y2 * scale + pad_y
        x1, x2 = max(0.0, min(size, x1)), max(0.0, min(size, x2))
        y1, y2 = max(0.0, min(size, y1)), max(0.0, min(size, y2))
        if x2 <= x1 or y2 <= y1:
            boxes_f.write(struct.pack("<I", 0))
            continue
        cx, cy = (x1 + x2) / 2.0 / size, (y1 + y2) / 2.0 / size
        bw, bh = (x2 - x1) / size, (y2 - y1) / size
        boxes_f.write(struct.pack("<I", 1))
        boxes_f.write(struct.pack("<Iffff", 0, cx, cy, bw, bh))
    images_f.close()
    boxes_f.close()

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"n": len(rows), "c": 3, "h": size, "w": size, "nc": 1}, f)
    print(f"packed {len(rows)} images -> {out_dir}", file=sys.stderr)


COMMANDS = {
    "identity-score": cmd_identity_score,
    "save-ref-embed": cmd_save_ref_embed,
    "face-count": cmd_face_count,
    "person-box": cmd_person_box,
    "letterbox": cmd_letterbox,
    "pack-dataset": cmd_pack_dataset,
}

if __name__ == "__main__":
    COMMANDS[sys.argv[1]](sys.argv[2:])
PY

pytool() { BRAIN="$BRAIN" python3 "$PY_TOOLS" "$@"; }

# ============================================================= stage 2: source photo

log "Stage 2: source photo -> verified single face"

PHOTO="$OUT_DIR/source.jpg"
case "$SRC" in
  http://*|https://*) curl -fsSL "$SRC" -o "$PHOTO" || die "could not download $SRC" ;;
  *) cp "$SRC" "$PHOTO" || die "no such file: $SRC" ;;
esac

read -r FACE_N FX1 FY1 FX2 FY2 < <(pytool face-count "$PHOTO")
[ "$FACE_N" != "NONE" ] || die "no face detected in $PHOTO"
if [ "$FACE_N" -gt 1 ]; then
  echo "WARNING: $FACE_N faces detected in $PHOTO - using the largest" >&2
fi
echo "source face: bbox [$FX1,$FY1,$FX2,$FY2] ($FACE_N face(s) detected)" >&2

REF_EMBED="$OUT_DIR/ref_embed.bin"
pytool save-ref-embed "$PHOTO" "$REF_EMBED"

mkdir -p "$OUT_DIR/train/target" "$OUT_DIR/train/negative" "$OUT_DIR/train/background" \
         "$OUT_DIR/holdout/target" "$OUT_DIR/holdout/stranger"

pulid_gen() {
  # pulid_gen <prompt> <out.png> <seed>
  "$BRAIN" pulid text2image --prompt "$1" --in face_image="$PHOTO" --out image="$2" \
    --width "$GEN_SIZE" --height "$GEN_SIZE" --steps "$GEN_STEPS" --seed "$3" --precision int8
}

flux2_gen() {
  # flux2_gen <prompt> <out.png> <seed>
  "$BRAIN" flux2 generate --prompt "$1" --out "$2" --width "$GEN_SIZE" --height "$GEN_SIZE" \
    --seed "$3" --variant klein-4b --precision int8
}

log "Stage 2b: canary check (3 images) before committing to $N_TARGET generations"

CANARY_SCORES=()
for i in 1 2 3; do
  out="$OUT_DIR/canary-$i.png"
  pulid_gen "a photo of a person standing in a park, front view" "$out" "$i"
  score="$(pytool identity-score "$REF_EMBED" "$out")"
  echo "canary $i: identity score $score" >&2
  CANARY_SCORES+=("$score")
done
if ! printf '%s\n' "${CANARY_SCORES[@]}" | awk -v floor="$IDENTITY_FLOOR" '$1!="NA" && $1+0>floor{f=1} END{exit !f}'; then
  die "all 3 canary images scored at or below the stranger floor ($IDENTITY_FLOOR) - PuLID conditioning does not appear to be taking effect (check BRAIN_FLUX1_I8_KEEP_F32 and that the DiT and PulidCa are sharing one device/kernel handle) rather than burning GPU-hours on 47 more generations."
fi

# ============================================================= stage 3: 50 target images

log "Stage 3: generating $N_TARGET target-identity training images"

SETTINGS=(park office beach "city street" kitchen library "mountain trail" cafe "studio backdrop" garden)
ANGLES=("front view" "3/4 left view" "3/4 right view" "profile view" "from above" "from below")

MANIFEST="$OUT_DIR/manifest.jsonl"
: > "$MANIFEST"

survive=0
idx=0
for setting in "${SETTINGS[@]}"; do
  for angle in "${ANGLES[@]}"; do
    idx=$((idx + 1))
    [ "$idx" -gt "$N_TARGET" ] && break 2
    out="$OUT_DIR/train/target/img-$(printf '%03d' "$idx").png"
    prompt="a photo of a person in a $setting, $angle"
    seed=$((1000 + idx))
    pulid_gen "$prompt" "$out" "$seed"
    score="$(pytool identity-score "$REF_EMBED" "$out")"
    ok=0
    if [ "$score" != "NA" ] && awk -v s="$score" -v f="$IDENTITY_FLOOR" 'BEGIN{exit !(s+0>f)}'; then
      ok=1
    else
      echo "img-$idx: score $score <= floor, retrying with a different seed" >&2
      pulid_gen "$prompt" "$out" "$((seed + 50000))"
      score="$(pytool identity-score "$REF_EMBED" "$out")"
      if [ "$score" != "NA" ] && awk -v s="$score" -v f="$IDENTITY_FLOOR" 'BEGIN{exit !(s+0>f)}'; then
        ok=1
      fi
    fi
    if [ "$ok" = 1 ]; then
      echo "img-$idx [$setting/$angle]: identity score $score - kept" >&2
      echo "$out" >> "$OUT_DIR/target_survivors.txt"
      survive=$((survive + 1))
    else
      echo "img-$idx [$setting/$angle]: identity score $score - dropped after retry" >&2
      rm -f "$out"
    fi
  done
done

echo "target survivors: $survive / $N_TARGET" >&2
[ "$survive" -ge 40 ] || die "only $survive/$N_TARGET target images survived the identity gate - too few to trust training (floor is 40)"

# ============================================================= stage 4: negatives + backgrounds

log "Stage 4: $N_NEGATIVE hard negatives + $N_BACKGROUND backgrounds"

PEOPLE_DESC=("a young woman with short hair" "an elderly man with a beard" "a teenage boy" "a woman wearing glasses" "a man with curly hair")
neg_idx=0
while [ "$neg_idx" -lt "$N_NEGATIVE" ]; do
  neg_idx=$((neg_idx + 1))
  setting="${SETTINGS[$((neg_idx % ${#SETTINGS[@]}))]}"
  angle="${ANGLES[$((neg_idx % ${#ANGLES[@]}))]}"
  person="${PEOPLE_DESC[$((neg_idx % ${#PEOPLE_DESC[@]}))]}"
  out="$OUT_DIR/train/negative/img-$(printf '%03d' "$neg_idx").png"
  flux2_gen "a photo of $person in a $setting, $angle" "$out" "$((5000 + neg_idx))"
  echo "$out" >> "$OUT_DIR/negative_pool.txt"
done

bg_idx=0
while [ "$bg_idx" -lt "$N_BACKGROUND" ]; do
  bg_idx=$((bg_idx + 1))
  setting="${SETTINGS[$((bg_idx % ${#SETTINGS[@]}))]}"
  out="$OUT_DIR/train/background/img-$(printf '%03d' "$bg_idx").png"
  flux2_gen "an empty $setting scene, no people" "$out" "$((6000 + bg_idx))"
  echo "$out" >> "$OUT_DIR/background_pool.txt"
done

# ============================================================= stage 5: auto-label + pack

log "Stage 5: auto-labeling person boxes + packing the dataset"

kept=0
while IFS= read -r img; do
  read -r bx1 by1 bx2 by2 _ < <(pytool person-box "$YOLO_COCO_WEIGHTS" "$img" 0.25 0 || echo NONE)
  if [ "$bx1" = "NONE" ]; then
    echo "$(basename "$img"): no COCO person detection - dropped" >&2
    continue
  fi
  printf '{"image": %s, "box": [%s, %s, %s, %s]}\n' "$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$img")" "$bx1" "$by1" "$bx2" "$by2" >> "$MANIFEST"
  kept=$((kept + 1))
done < "$OUT_DIR/target_survivors.txt"

echo "auto-labeled target images: $kept" >&2
[ "$kept" -ge 40 ] || die "only $kept target images got a COCO person box - too few to trust training (floor is 40)"

while IFS= read -r img; do
  printf '{"image": %s, "box": null}\n' "$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$img")" >> "$MANIFEST"
done < "$OUT_DIR/negative_pool.txt"
while IFS= read -r img; do
  printf '{"image": %s, "box": null}\n' "$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$img")" >> "$MANIFEST"
done < "$OUT_DIR/background_pool.txt"

SHUFFLED="$OUT_DIR/manifest.shuffled.jsonl"
python3 -c "
import random, sys
random.seed($TRAIN_SEED)
rows = open('$MANIFEST').readlines()
random.shuffle(rows)
open('$SHUFFLED', 'w').writelines(rows)
"

POOL_DIR="$OUT_DIR/data/pool"
pytool pack-dataset "$SHUFFLED" "$POOL_DIR" "$TRAIN_SIZE"

# ============================================================= stage 6: train

log "Stage 6: training the detector"

WEIGHTS="$OUT_DIR/person-detector.safetensors"
"$BRAIN" yolov8 train "$POOL_DIR" --out "$WEIGHTS" --device gpu \
  --steps "$TRAIN_STEPS" --batch "$TRAIN_BATCH" --lr "$TRAIN_LR" --nc 1 --input "$TRAIN_SIZE" --seed "$TRAIN_SEED"

echo "--- informational eval over the internal 90/10 split (not the real validation - stage 8 is) ---" >&2
"$BRAIN" yolov8 eval --weights "$WEIGHTS" --data "$POOL_DIR" --conf "$DETECT_CONF" --iou "$DETECT_IOU" >&2 || true

# ============================================================= stage 7: fresh held-out images

log "Stage 7: 5 fresh held-out images (never-used settings/angles)"

HOLDOUT_PROMPTS=(
  "a photo of a person on a rooftop at night, city lights behind"
  "a photo of a person walking down a snowy street"
  "a photo of a person standing in an art gallery"
  "a photo of a person riding a bicycle"
  "a photo of a person sitting for a seated interview"
)
h=0
for prompt in "${HOLDOUT_PROMPTS[@]}"; do
  h=$((h + 1))
  out="$OUT_DIR/holdout/target/img-$h.png"
  pulid_gen "$prompt" "$out" "$((9000 + h))"
  echo "$out" >> "$OUT_DIR/holdout_target.txt"
done

STRANGER_PROMPTS=(
  "a photo of a person on a rooftop at night, city lights behind"
  "a photo of a person walking down a snowy street"
  "a photo of a person standing in an art gallery"
  "a photo of a person riding a bicycle"
  "a photo of a person sitting for a seated interview"
)
s=0
for prompt in "${STRANGER_PROMPTS[@]}"; do
  s=$((s + 1))
  out="$OUT_DIR/holdout/stranger/img-$s.png"
  person="${PEOPLE_DESC[$((s % ${#PEOPLE_DESC[@]}))]}"
  flux2_gen "a photo of $person, ${prompt#a photo of a person }" "$out" "$((9500 + s))"
  echo "$out" >> "$OUT_DIR/holdout_stranger.txt"
done

# ============================================================= stage 8: validate

log "Stage 8: end-to-end validation"

PASS=1
report() { printf '%-46s %-8s %s\n' "$1" "$2" "$3"; }
report "image" "result" "detail"

while IFS= read -r img; do
  lb="$img.lb.png"
  pytool letterbox "$img" "$TRAIN_SIZE" "$lb"
  det="$(pytool person-box "$WEIGHTS" "$lb" "$DETECT_CONF" 0)"
  score="$(pytool identity-score "$REF_EMBED" "$img")"
  if [ "$det" = "NONE" ]; then
    report "$(basename "$(dirname "$img")")/$(basename "$img")" "FAIL" "no detection (identity score $score)"
    PASS=0
  else
    report "$(basename "$(dirname "$img")")/$(basename "$img")" "PASS" "$det (identity score $score)"
  fi
done < "$OUT_DIR/holdout_target.txt"

while IFS= read -r img; do
  lb="$img.lb.png"
  pytool letterbox "$img" "$TRAIN_SIZE" "$lb"
  det="$(pytool person-box "$WEIGHTS" "$lb" "$DETECT_CONF" 0)"
  if [ "$det" = "NONE" ]; then
    report "$(basename "$(dirname "$img")")/$(basename "$img")" "PASS" "correctly no detection"
  else
    report "$(basename "$(dirname "$img")")/$(basename "$img")" "FAIL" "false-positive detection $det"
    PASS=0
  fi
done < "$OUT_DIR/holdout_stranger.txt"

for img in "$OUT_DIR"/train/background/*.png; do
  [ -e "$img" ] || continue
  lb="$img.lb.png"
  pytool letterbox "$img" "$TRAIN_SIZE" "$lb"
  det="$(pytool person-box "$WEIGHTS" "$lb" "$DETECT_CONF" 0)"
  if [ "$det" = "NONE" ]; then
    report "background/$(basename "$img")" "PASS" "correctly no detection"
  else
    report "background/$(basename "$img")" "FAIL" "false-positive detection $det"
    PASS=0
  fi
done

echo >&2
if [ "$PASS" = 1 ]; then
  echo "ALL GATES PASSED." >&2
  echo "Trained detector: $WEIGHTS" >&2
  echo "Detect the person on any new image with:" >&2
  echo "  $BRAIN yolov8 detect --weights $WEIGHTS --image <letterboxed-to-${TRAIN_SIZE}x${TRAIN_SIZE}.png> --conf $DETECT_CONF --iou $DETECT_IOU" >&2
  echo "  (letterbox first: python3 $PY_TOOLS letterbox <src> $TRAIN_SIZE <out.png>)" >&2
  exit 0
else
  echo "ONE OR MORE GATES FAILED - see the scorecard above. The trained weights are still at $WEIGHTS if you want to inspect them, but detection is not validated." >&2
  exit 1
fi
