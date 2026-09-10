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
#   identity_yolo_pipeline.sh <photo-or-url> [out-dir]
#
# 1. Fetch weights (idempotent). 2. Verify one face; canary-check identity
# conditioning. 3. Generate N_TARGET identity-preserving images (PuLID),
# gated on ArcFace similarity to the source photo. 4. Generate hard
# negatives + backgrounds (plain FLUX.2, no identity conditioning).
# 5. Auto-label person boxes with COCO YOLOv8n and pack `brain yolov8
# train`'s flat binary format. 6. Train a from-scratch YOLOv8-tiny detector.
# 7. Generate fresh held-out target + stranger images. 8. Validate: the
# detector must fire on every held-out target and nowhere else. Exits
# non-zero if any gate fails.
#
# All generation runs against ONE resident `brain serve --dbus` daemon
# (stage 1b) instead of a fresh subprocess per image, so PuLID's ~27 GB and
# FLUX.2's ~13 GB weight sets load once instead of once per image. Every
# stage is restart-safe: an image already on disk is never regenerated, and
# every seed is drawn fresh (`rand_seed`), not derived from a loop index.
#
# Needs: `hf` (authenticated - FLUX.1-dev is gated), `python3` with
# numpy+Pillow+jeepney (`pip install -e brain-py`), `dbus-run-session`
# (only without an existing session bus), and `make build/release`.

set -euo pipefail

# brain serves models over a D-Bus SESSION bus; re-exec under a private one
# if this environment has none (matches every other client under examples/dbus).
if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
  command -v dbus-run-session >/dev/null || { printf '\nFATAL: no D-Bus session bus and no dbus-run-session to start one (apt install dbus)\n' >&2; exit 1; }
  exec dbus-run-session -- "$0" "$@"
fi

SRC="${1:?usage: identity_yolo_pipeline.sh <photo-or-url> [out-dir]}"
OUT_DIR="${2:-out/identity-yolo-$(date +%Y%m%d-%H%M%S)}"
BRAIN="${BRAIN:-./target/release/brain}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_CLIENT="$SCRIPT_DIR/identity_yolo_gen.py"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
log() { printf '\n=== %s ===\n' "$*" >&2; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }
# A fresh, unpredictable seed per image - not derived from a loop index, so a
# whole prompt/angle grid doesn't sample nearby, correlated noise, and a
# resumed run explores real seeds rather than a narrowed subset.
rand_seed() { echo $(( (RANDOM << 15) ^ RANDOM )); }

# ============================================================= config (env-overridable)

GEN_SIZE="${GEN_SIZE:-512}" GEN_STEPS="${GEN_STEPS:-8}" TRAIN_SIZE="${TRAIN_SIZE:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-1500}" TRAIN_BATCH="${TRAIN_BATCH:-4}" TRAIN_LR="${TRAIN_LR:-3e-3}" TRAIN_SEED="${TRAIN_SEED:-1337}"
N_TARGET="${N_TARGET:-50}" N_NEGATIVE="${N_NEGATIVE:-40}" N_BACKGROUND="${N_BACKGROUND:-10}" N_HOLDOUT="${N_HOLDOUT:-5}"

# Class layout. This trains an ADDITIONAL class onto a COCO-pretrained
# yolov8n rather than a single-class detector from scratch, so COCO's own 80
# classes (0..79) keep their indices and their meaning, and the target person
# is appended as class 80. `person` stays class 0 - a different real person is
# still a person, which is exactly the distinction being taught.
PERSON_CLASS=0
TARGET_CLASS="${TARGET_CLASS:-80}"
NUM_CLASSES=$((TARGET_CLASS + 1))
TARGET_NAME="${TARGET_NAME:-einstein}"
IDENTITY_FLOOR="${IDENTITY_FLOOR:-0.05}"   # ArcFace cosine floor: catches total conditioning failure, not a quality bar
DETECT_CONF="${DETECT_CONF:-0.1}" DETECT_IOU="${DETECT_IOU:-0.45}"  # this repo's own proven combo for this tiny architecture
SURVIVE_FLOOR=$((N_TARGET * 8 / 10))       # 80% of N_TARGET must pass the identity gate to trust training

# The served path resolves each of these from the model store on its own now
# (`resolver_cli::served_assembly`, the same resolver the one-shot CLI uses),
# so these are overrides rather than the only way to point brain at them: each
# variable still wins unconditionally for its own role when set. They stay
# pinned here so a run is reproducible on a box whose store holds something
# else. `brain pull` still does not fetch FLUX.1/PuLID themselves.
FLUX1_DIR="${FLUX1_DIR:-$HOME/.local/share/brain/models/black-forest-labs/FLUX.1-dev}"
PULID_FILE="${PULID_FILE:-$HOME/.local/share/brain/models/guozinan/PuLID/pulid_flux_v0.9.1.safetensors}"
ARCFACE_DIR="${ARCFACE_DIR:-$HOME/.local/share/brain/models/DIAMONIK7777/antelopev2}"
CLIP_DIR="${CLIP_DIR:-$HOME/.local/share/brain/models/QuanSun/EVA-CLIP}"
YOLOV8_COCO_DIR="${YOLOV8_COCO_DIR:-$HOME/.local/share/brain/models/Ultralytics/YOLOv8}"
BISENET_DIR="${BISENET_DIR:-$HOME/.local/share/brain/models/facexlib/pulid}"
# FLUX.2 is the one backend that genuinely still needs pinning here, and the
# resolver agrees rather than disagrees: this box holds several real klein
# candidates for the `dit` role (third-party re-quantizations alongside the
# official release), so the served path reports all of them and serves
# nothing until one is named - it will not guess. klein-vs-base is not
# recoverable from any tensor shape either. Left explicit for both reasons.
FLUX2_DIT="${FLUX2_DIT:-$HOME/.local/share/brain/models/black-forest-labs/FLUX.2-klein-9B/Flux-2-Klein-9B-KV-Q8_0.gguf}"
FLUX2_VAE="${FLUX2_VAE:-$HOME/.local/share/brain/models/unsloth/flux2-vae.safetensors}"
FLUX2_TE="${FLUX2_TE:-$HOME/.local/share/brain/models/Qwen/Qwen3-8B}"
FLUX2_TOKENIZER="${FLUX2_TOKENIZER:-$HOME/.local/share/brain/models/Qwen/Qwen3-8B/tokenizer.json}"
FLUX2_VARIANT="${FLUX2_VARIANT:-klein-9b}"  # must match FLUX2_DIT's own weights (klein_9b needs Qwen3-8B, not -4B)

export BRAIN_FLUX1_DIR="$FLUX1_DIR" BRAIN_PULID_DIR="$PULID_FILE" BRAIN_ARCFACE_DIR="$ARCFACE_DIR" \
       BRAIN_SCRFD_DIR="$ARCFACE_DIR" BRAIN_CLIP_DIR="$CLIP_DIR" BRAIN_BISENET_DIR="$BISENET_DIR"

PY_TOOLS="$OUT_DIR/pytoolkit.py"

# ============================================================= stage 0/1: preflight + weights

log "Stage 0: preflight"

[ -x "$BRAIN" ] || die "no $BRAIN binary - run: make build/release"
command -v hf >/dev/null || die "the 'hf' CLI is required - pip install -U huggingface_hub[cli]"
command -v python3 >/dev/null || die "python3 is required"
python3 -c "import numpy, PIL, jeepney" 2>/dev/null || die "python3 needs numpy, Pillow and jeepney - pip install numpy Pillow -e brain-py"
[ -f "$GEN_CLIENT" ] || die "missing $GEN_CLIENT"
[ -f "$HOME/.cache/huggingface/token" ] || [ -n "${HF_TOKEN:-}" ] || \
  die "no HuggingFace token (~/.cache/huggingface/token or \$HF_TOKEN) - FLUX.1-dev is gated: hf auth login after accepting its license"
[ "${BRAIN_FLUX2_ALLOW_NC:-}" = "1" ] || \
  die "stage 4 uses FLUX.2 klein-9B (FLUX.2 [Non-Commercial] License) - set BRAIN_FLUX2_ALLOW_NC=1 yourself to confirm non-commercial use"
export BRAIN_FLUX2_ALLOW_NC
"$BRAIN" devices >&2 || true

log "Stage 1: weight provisioning"

fetch() { [ -e "$2" ] && { echo "$1: already present" >&2; return; }; shift 2; "$@"; }
[ -f "$FLUX1_DIR/transformer/config.json" ] || \
  hf download black-forest-labs/FLUX.1-dev --local-dir "$FLUX1_DIR" \
  --include "model_index.json" --include "transformer/*" --include "vae/*" \
  --include "text_encoder/*" --include "text_encoder_2/*" --include "tokenizer/*" --include "tokenizer_2/*" \
  || die "FLUX.1-dev download failed - if this is a 401/403, accept its license at https://huggingface.co/black-forest-labs/FLUX.1-dev with the account behind your HF token"
fetch pulid "$PULID_FILE" hf download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir "$(dirname "$PULID_FILE")"
fetch yolov8-coco "$(find "$YOLOV8_COCO_DIR" -name '*.safetensors' 2>/dev/null | head -1)" "$BRAIN" pull Ultralytics/YOLOv8
fetch flux2-klein "$FLUX2_DIT" "$BRAIN" pull black-forest-labs/FLUX.2-klein-9B

[ -f "$ARCFACE_DIR/glintr100.onnx" ] || die "ArcFace weights missing at $ARCFACE_DIR"
[ -f "$ARCFACE_DIR/scrfd_10g_bnkps.onnx" ] || die "SCRFD weights missing at $ARCFACE_DIR"
[ -f "$CLIP_DIR/EVA02_CLIP_L_336_psz14_s6B.pt" ] || die "EVA-CLIP weights missing at $CLIP_DIR"
[ -f "$FLUX2_VAE" ] || die "FLUX.2 VAE missing at $FLUX2_VAE"
[ -e "$FLUX2_TE" ] || die "FLUX.2 text encoder missing at $FLUX2_TE"
[ -f "$FLUX2_TOKENIZER" ] || die "FLUX.2 tokenizer missing at $FLUX2_TOKENIZER"
[ -f "$BISENET_DIR/parsing_bisenet.safetensors" ] || die \
  "BiSeNet weights missing at $BISENET_DIR - no fetch recipe exists yet (facexlib's release is a legacy torch pickle brain can't read directly). One-time conversion: pip install facexlib torch safetensors opencv-python; get landmarks via 'brain scrfd detect --in image=<photo> --json'; then python3 tools/goldens/pulid_face_parsing_dump_reference.py --photo <photo> --kps x1,y1,...,x5,y5 --testdata \$(dirname $BISENET_DIR)"

YOLO_COCO_WEIGHTS="$(find "$YOLOV8_COCO_DIR" -name '*.safetensors' | head -1)"
[ -n "$YOLO_COCO_WEIGHTS" ] || die "no COCO YOLOv8 checkpoint found under $YOLOV8_COCO_DIR"

# ============================================================= stage 1b: resident daemon

log "Stage 1b: resident serving daemon"

export BRAIN_FLUX2_DIT="$FLUX2_DIT" BRAIN_FLUX2_VAE="$FLUX2_VAE" BRAIN_FLUX2_TE="$FLUX2_TE" BRAIN_FLUX2_TOKENIZER="$FLUX2_TOKENIZER"

SERVE_LOG="$OUT_DIR/brain-serve.log"
SERVE_READY="$OUT_DIR/brain-serve.ready"
SERVE_PID="" SERVE_BACKEND=""

stop_serve() {
  [ -n "$SERVE_PID" ] || return 0
  local pid="$SERVE_PID"; SERVE_PID=""; SERVE_BACKEND=""
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 150); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap stop_serve EXIT
trap 'stop_serve; exit 130' INT
trap 'stop_serve; exit 143' TERM

# One backend resident at a time, not both: a 512x512 int8 PuLID instance is
# ~22 GiB and FLUX.2 klein-9B needs ~25 GiB more, which doesn't fit two
# 24 GiB cards - and `BudgetPlacer` snapshots free VRAM once at process
# start, so a second model built inside an already-running daemon would plan
# against stale numbers anyway. Restarting between backends (4 loads/run,
# not ~75) is what makes each snapshot true when it's taken.
serve_for() {
  local want="$1"
  [ "$SERVE_BACKEND" = "$want" ] && return 0
  stop_serve; rm -f "$SERVE_READY"
  # --reserve-gb 1 (PuLID's ~22 GiB resident estimate against a 24 GiB card)
  # reproducibly OOM'd on a fully idle card: real peak usage during a
  # generation - staging buffers, upload/download scratch - exceeds the
  # steady-state resident figure the estimate is measured from. 3 GiB of
  # headroom is what stopped it in practice on this hardware.
  "$BRAIN" serve --dbus --reserve-gb "${SERVE_RESERVE_GB:-3}" --ready-file "$SERVE_READY" > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!
  for _ in $(seq 1 300); do
    [ -e "$SERVE_READY" ] && break
    kill -0 "$SERVE_PID" 2>/dev/null || die "brain serve --dbus exited during startup - see $SERVE_LOG"
    sleep 1
  done
  [ -e "$SERVE_READY" ] || die "brain serve --dbus never became ready - see $SERVE_LOG"
  python3 "$GEN_CLIENT" "$want" --wait 60 || \
    die "brain serve --dbus does not serve '$want' - check its BRAIN_* weight variables above. See $SERVE_LOG"
  SERVE_BACKEND="$want"
}

# ============================================================= python toolkit

# Real data-format glue nothing else in this repo provides from real photos:
# ArcFace identity scoring, brain's stdout parsing, letterboxing, and packing
# to `brain yolov8 train`'s flat binary format (crates/data/src/{gen_detect,binio}.rs).
cat > "$PY_TOOLS" <<'PY'
import json, os, struct, subprocess, sys
import numpy as np
from PIL import Image

BRAIN = os.environ["BRAIN"]


def arcface_embed(img_path):
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
    ref = np.fromfile(argv[0], dtype="<f4")
    cand = arcface_embed(argv[1])
    print("NA" if cand is None else f"{float(np.dot(ref, cand)):.4f}")


def cmd_save_ref_embed(argv):
    e = arcface_embed(argv[0])
    if e is None:
        sys.exit(f"no face found in {argv[0]}")
    e.astype("<f4").tofile(argv[1])


def cmd_face_count(argv):
    """-> 'N x1 y1 x2 y2' for the largest face, or 'NONE'."""
    r = subprocess.run([BRAIN, "scrfd", "detect", "--in", f"image={argv[0]}", "--max_faces", "16"],
                        capture_output=True, text=True, timeout=120)
    faces = []
    for line in r.stdout.splitlines():
        if line.startswith("faces:"):
            faces = json.loads(line[len("faces:"):])
    if not faces:
        print("NONE")
        return
    best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    b = best["bbox"]
    print(f"{len(faces)} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}")


def cmd_person_box(argv):
    """<weights> <image> <conf> [class=0] -> 'x1 y1 x2 y2 conf' (best match) or NONE."""
    weights, img, conf = argv[0], argv[1], argv[2]
    cls_id = int(argv[3]) if len(argv) > 3 else 0
    r = subprocess.run([BRAIN, "yolov8", "detect", "--weights", weights, "--image", img, "--conf", conf, "--iou", "0.45"],
                        capture_output=True, text=True, timeout=300)
    dets = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("["):
            x1, y1, x2, y2, conf_v, cls = json.loads(line)
            if int(cls) == cls_id:
                dets.append((x1, y1, x2, y2, conf_v))
    if not dets:
        print("NONE")
        return
    x1, y1, x2, y2, conf_v = max(dets, key=lambda d: d[4])
    print(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf_v:.4f}")


def _letterbox(img, size):
    """Resize to size x size with grey padding, centered. Returns the
    transform (scale, pad_x, pad_y) a caller applies to box coords the same
    direction (original px -> canvas px)."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return canvas, scale, (size - nw) // 2, (size - nh) // 2


def cmd_letterbox(argv):
    canvas, _, _, _ = _letterbox(Image.open(argv[0]), int(argv[1]))
    canvas.save(argv[2])


def cmd_pack_dataset(argv):
    """<manifest.jsonl> <out-dir> <size> <nc> - manifest rows are {"image",
    "box": [x1,y1,x2,y2] or null, "class": int} in SOURCE pixel coords, in
    final dataset order. Writes images.f32 (N*3*size*size LE f32 CHW RGB
    [0,1]), boxes.bin ([u32 num] then num*(u32 class,f32 cx,cy,w,h)
    normalized), meta.json.

    The per-row class is what makes this a MULTI-class training set. A
    hard-negative (a different real person) carries a real box at the ordinary
    COCO `person` class, not zero boxes: the network then has a competing
    hypothesis it must actively reject, instead of only an absence of positive
    signal, which it can satisfy by latching onto any incidental correlate of
    the target images (lighting, composition) rather than the subject."""
    manifest_path, out_dir, size = argv[0], argv[1], int(argv[2])
    nc = int(argv[3]) if len(argv) > 3 else 1
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(manifest_path) if l.strip()]
    images_f = open(os.path.join(out_dir, "images.f32"), "wb")
    boxes_f = open(os.path.join(out_dir, "boxes.bin"), "wb")
    for row in rows:
        img = Image.open(row["image"])
        canvas, scale, pad_x, pad_y = _letterbox(img, size)
        chw = np.transpose(np.asarray(canvas, dtype=np.float32) / 255.0, (2, 0, 1))
        images_f.write(chw.astype("<f4").tobytes())
        box = row.get("box")
        if box is None:
            boxes_f.write(struct.pack("<I", 0))
            continue
        x1, y1, x2, y2 = box
        x1, x2 = max(0.0, min(size, x1 * scale + pad_x)), max(0.0, min(size, x2 * scale + pad_x))
        y1, y2 = max(0.0, min(size, y1 * scale + pad_y)), max(0.0, min(size, y2 * scale + pad_y))
        if x2 <= x1 or y2 <= y1:
            boxes_f.write(struct.pack("<I", 0))
            continue
        cls = int(row.get("class", 0))
        if not 0 <= cls < nc:
            raise SystemExit(f"row class {cls} out of range for nc={nc}: {row['image']}")
        boxes_f.write(struct.pack("<I", 1))
        boxes_f.write(struct.pack("<Iffff", cls, (x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size))
    images_f.close()
    boxes_f.close()
    json.dump({"n": len(rows), "c": 3, "h": size, "w": size, "nc": nc}, open(os.path.join(out_dir, "meta.json"), "w"))
    print(f"packed {len(rows)} images -> {out_dir}", file=sys.stderr)


COMMANDS = {"identity-score": cmd_identity_score, "save-ref-embed": cmd_save_ref_embed, "face-count": cmd_face_count,
            "person-box": cmd_person_box, "letterbox": cmd_letterbox, "pack-dataset": cmd_pack_dataset}
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
[ "$FACE_N" -gt 1 ] && echo "WARNING: $FACE_N faces detected - using the largest" >&2
echo "source face: bbox [$FX1,$FY1,$FX2,$FY2] ($FACE_N face(s))" >&2

REF_EMBED="$OUT_DIR/ref_embed.bin"
pytool save-ref-embed "$PHOTO" "$REF_EMBED"
mkdir -p "$OUT_DIR/train/target" "$OUT_DIR/train/negative" "$OUT_DIR/train/background" "$OUT_DIR/holdout/target" "$OUT_DIR/holdout/stranger"

# Both hold width/height/precision (and PuLID's variant) constant, which is
# what the residents key their instances on - only prompt/seed vary, so every
# call after the first reuses the loaded weights.
pulid_gen() { serve_for pulid; python3 "$GEN_CLIENT" pulid --prompt "$1" --out "$2" --seed "$3" --face "$PHOTO" --width "$GEN_SIZE" --height "$GEN_SIZE" --steps "$GEN_STEPS" --precision int8; }
flux2_gen() { serve_for flux2; python3 "$GEN_CLIENT" flux2 --prompt "$1" --out "$2" --seed "$3" --width "$GEN_SIZE" --height "$GEN_SIZE" --precision int8 --variant "$FLUX2_VARIANT"; }
identity_score() { pytool identity-score "$REF_EMBED" "$1"; }
above_floor() { [ "$1" != "NA" ] && awk -v s="$1" -v f="$IDENTITY_FLOOR" 'BEGIN{exit !(s+0>f)}'; }
gen_if_missing() { [ -f "$1" ] || "$2" "$3" "$1" "$(rand_seed)"; }  # <out> <gen-fn> <prompt>
append_once() { grep -qxF "$2" "$1" 2>/dev/null || echo "$2" >> "$1"; }  # <list-file> <line>

# ============================================================= stage 2b: canary check

log "Stage 2b: canary check before committing to $N_TARGET generations"

CANARY_SCORES=()
for i in 1 2 3; do
  out="$OUT_DIR/canary-$i.png"
  gen_if_missing "$out" pulid_gen "a photo of a person standing in a park, front view"
  score="$(identity_score "$out")"
  echo "canary $i: identity score $score" >&2
  CANARY_SCORES+=("$score")
done
if ! printf '%s\n' "${CANARY_SCORES[@]}" | awk -v floor="$IDENTITY_FLOOR" '$1!="NA" && $1+0>floor{f=1} END{exit !f}'; then
  die "all 3 canary images scored at or below the stranger floor ($IDENTITY_FLOOR) - PuLID conditioning does not appear to be working, rather than burning GPU-hours on more generations"
fi

# ============================================================= stage 3: target images

log "Stage 3: generating $N_TARGET target-identity training images"

# Scene diversity matters more than angle diversity for what this detector can
# latch onto: a scene/lighting condition with NO training example in EITHER
# direction is where a false positive lands. The night/low-light entries are
# here because the held-out set (stage 7) probes exactly that.
SETTINGS=(park office beach "city street" kitchen library "mountain trail" cafe "studio backdrop" garden
          "rooftop at night with city lights behind" "dimly lit bar" "street at night under neon signs" "snowy street at dusk")
ANGLES=("front view" "3/4 left view" "3/4 right view" "profile view" "from above" "from below")
SURVIVORS="$OUT_DIR/target_survivors.txt"
[ -f "$SURVIVORS" ] || : > "$SURVIVORS"

# ANGLE-major, not setting-major: the grid is capped at N_TARGET, and iterating
# settings on the OUTER loop spent the whole budget on the first ceil(N_TARGET/
# #ANGLES) settings and never generated the rest at all. Angle-major spreads a
# budget smaller than the full grid across EVERY setting.
idx=0
for angle in "${ANGLES[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    idx=$((idx + 1))
    [ "$idx" -gt "$N_TARGET" ] && break 2
    out="$OUT_DIR/train/target/img-$(printf '%03d' "$idx").png"
    prompt="a photo of a person in a $setting, $angle"
    if [ -f "$out" ]; then
      # Only survives here already recorded as passing (a failed retry is
      # rm -f'd below) - just resume-record it, don't regenerate.
      append_once "$SURVIVORS" "$out"
      echo "img-$idx [$setting/$angle]: already generated - resuming" >&2
      continue
    fi
    pulid_gen "$prompt" "$out" "$(rand_seed)"
    score="$(identity_score "$out")"
    if ! above_floor "$score"; then
      echo "img-$idx: score $score <= floor, retrying with a new seed" >&2
      pulid_gen "$prompt" "$out" "$(rand_seed)"
      score="$(identity_score "$out")"
    fi
    if above_floor "$score"; then
      echo "img-$idx [$setting/$angle]: identity score $score - kept" >&2
      append_once "$SURVIVORS" "$out"
    else
      echo "img-$idx [$setting/$angle]: identity score $score - dropped after retry" >&2
      rm -f "$out"
    fi
  done
done

survive=$(wc -l < "$SURVIVORS" | tr -d ' ')
echo "target survivors: $survive / $N_TARGET" >&2
[ "$survive" -ge "$SURVIVE_FLOOR" ] || die "only $survive/$N_TARGET target images survived the identity gate (floor $SURVIVE_FLOOR)"

# ============================================================= stage 4: negatives + backgrounds

log "Stage 4: $N_NEGATIVE hard negatives + $N_BACKGROUND backgrounds"

# Hard negatives now carry a REAL person-class box, so their job is to be a
# competing hypothesis, not just filler - the more they cover the target's own
# demographic range, the less room class $TARGET_CLASS has to settle for "an
# older man with wild grey hair" instead of this specific one.
PEOPLE_DESC=("a young woman with short hair" "an elderly man with a beard" "a teenage boy" "a woman wearing glasses"
             "a man with curly hair" "an elderly man with wild grey hair and a moustache" "a middle-aged man with glasses and grey hair"
             "an older man in a suit" "a bald man with a moustache" "a woman with long dark hair")
NEG_POOL="$OUT_DIR/negative_pool.txt" BG_POOL="$OUT_DIR/background_pool.txt"
[ -f "$NEG_POOL" ] || : > "$NEG_POOL"
[ -f "$BG_POOL" ] || : > "$BG_POOL"

for ((neg_idx = 1; neg_idx <= N_NEGATIVE; neg_idx++)); do
  out="$OUT_DIR/train/negative/img-$(printf '%03d' "$neg_idx").png"
  person="${PEOPLE_DESC[$((neg_idx % ${#PEOPLE_DESC[@]}))]}"
  setting="${SETTINGS[$((neg_idx % ${#SETTINGS[@]}))]}" angle="${ANGLES[$((neg_idx % ${#ANGLES[@]}))]}"
  gen_if_missing "$out" flux2_gen "a photo of $person in a $setting, $angle"
  append_once "$NEG_POOL" "$out"
done
for ((bg_idx = 1; bg_idx <= N_BACKGROUND; bg_idx++)); do
  out="$OUT_DIR/train/background/img-$(printf '%03d' "$bg_idx").png"
  setting="${SETTINGS[$((bg_idx % ${#SETTINGS[@]}))]}"
  gen_if_missing "$out" flux2_gen "an empty $setting scene, no people"
  append_once "$BG_POOL" "$out"
done

# ============================================================= stage 5: auto-label + pack

log "Stage 5: auto-labeling person boxes + packing the dataset"

stop_serve  # hand the cards back before COCO auto-labeling / training start their own GPU use

MANIFEST="$OUT_DIR/manifest.jsonl"
: > "$MANIFEST"
row() { printf '{"image": %s, "box": %s, "class": %s}\n' "$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$1")" "$2" "${3:-0}" >> "$MANIFEST"; }

# The COCO auto-labeler locates the person the same way for BOTH pools; only
# the class written to the dataset differs. That is the whole "remapped from
# person to einstein" mechanism: the target's box is class TARGET_CLASS, a
# different real person's box stays ordinary COCO class PERSON_CLASS, so the
# network must tell two populations of person-shaped boxes apart rather than
# learning "person-shaped OR nothing".
kept=0
while IFS= read -r img; do
  read -r bx1 by1 bx2 by2 _ < <(pytool person-box "$YOLO_COCO_WEIGHTS" "$img" 0.25 "$PERSON_CLASS" || echo NONE)
  if [ "$bx1" = "NONE" ]; then
    echo "$(basename "$img"): no COCO person detection - dropped" >&2
    continue
  fi
  row "$img" "[$bx1, $by1, $bx2, $by2]" "$TARGET_CLASS"
  kept=$((kept + 1))
done < "$SURVIVORS"
echo "auto-labeled target images (class $TARGET_CLASS): $kept" >&2
[ "$kept" -ge "$SURVIVE_FLOOR" ] || die "only $kept target images got a COCO person box (floor $SURVIVE_FLOOR)"

neg_kept=0
while IFS= read -r img; do
  read -r bx1 by1 bx2 by2 _ < <(pytool person-box "$YOLO_COCO_WEIGHTS" "$img" 0.25 "$PERSON_CLASS" || echo NONE)
  if [ "$bx1" = "NONE" ]; then
    echo "$(basename "$img"): no COCO person detection in a hard negative - dropped" >&2
    continue
  fi
  row "$img" "[$bx1, $by1, $bx2, $by2]" "$PERSON_CLASS"
  neg_kept=$((neg_kept + 1))
done < "$NEG_POOL"
echo "auto-labeled hard negatives (class $PERSON_CLASS): $neg_kept" >&2
[ "$neg_kept" -gt 0 ] || die "no hard negative got a COCO person box - the run would have no competing class"

# Backgrounds genuinely contain neither class, so they stay boxless.
while IFS= read -r img; do row "$img" null; done < "$BG_POOL"

SHUFFLED="$OUT_DIR/manifest.shuffled.jsonl"
python3 -c "
import random
random.seed($TRAIN_SEED)
rows = open('$MANIFEST').readlines()
random.shuffle(rows)
open('$SHUFFLED', 'w').writelines(rows)
"
POOL_DIR="$OUT_DIR/data/pool"
pytool pack-dataset "$SHUFFLED" "$POOL_DIR" "$TRAIN_SIZE" "$NUM_CLASSES"

# ============================================================= stage 6: train

log "Stage 6: fine-tuning the COCO detector with '$TARGET_NAME' as class $TARGET_CLASS"

# Fine-tune, NOT train-from-scratch: the base is the same real COCO yolov8n
# checkpoint stage 5 auto-labels with, so the run inherits all 80 COCO classes
# and appends one. What each flag protects:
#   --nc            one more class than the base; the class head is widened and
#                   the pretrained 80 channels are copied in (yolov8::finetune).
#   --freeze-backbone   the shared feature extractor cannot drift - gradients
#                   AND BatchNorm running stats, since this dataset is people-only.
#   --freeze-reg-head   YOLOv8's box head is class-agnostic; the pretrained one
#                   already bounds the target correctly, so retraining it could
#                   only move the boxes every other class relies on.
#   --freeze-cls-hidden the class branch's two SHARED hidden convs. Without
#                   this the other 80 classes' own weights stay bit-identical
#                   but the features they READ change, and their detections
#                   collapse anyway (measured: 83 of 99 COCO detections lost).
#   --train-classes only the class this dataset adds. The detection loss scores
#                   BCE over ALL classes against an all-zero target for absent
#                   ones, so without this the other 80 COCO classes would be
#                   trained to never fire.
#   --wd 0          decoupled weight decay is not a gradient and would shrink
#                   the gated classes anyway.
#
# Together these make preservation EXACT: every pretrained tensor and every
# pretrained class channel is bit-for-bit unchanged, so all 80 COCO classes
# produce identical detections (verified: 99/99 preserved, 0 lost, 0 gained).
#
# The cost is capacity, and it is a real ceiling, not a tuning problem: class
# $TARGET_CLASS is then a linear probe on features trained to separate COCO's
# OWN categories, which encode object category, not personal identity. It
# learns "a person, weighted toward this one's typical scenes" rather than the
# individual. Set FREEZE_CLS_HIDDEN=0 to trade preservation for capacity - but
# note that measured WORSE on both axes (COCO collapsed AND the new class fired
# on strangers), because the identity signal is not in these features at all.
# A specific person is properly a face-recognition problem: this repo's
# SCRFD + ArcFace stack (which stage 3 already uses to gate identity) is the
# right tool, with the detector supplying the person box.
FREEZE_CLS_HIDDEN="${FREEZE_CLS_HIDDEN:-1}"
CLS_HIDDEN_FLAG=""; [ "$FREEZE_CLS_HIDDEN" = 1 ] && CLS_HIDDEN_FLAG="--freeze-cls-hidden"
TRAIN_CLASSES="${TRAIN_CLASSES:-$TARGET_CLASS}"

WEIGHTS="$OUT_DIR/$TARGET_NAME-detector.safetensors"
"$BRAIN" yolov8 fine-tune "$POOL_DIR" --weights "$YOLO_COCO_WEIGHTS" --out "$WEIGHTS" --device gpu \
  --steps "$TRAIN_STEPS" --batch "$TRAIN_BATCH" --lr "$TRAIN_LR" --wd 0 \
  --nc "$NUM_CLASSES" --input "$TRAIN_SIZE" --seed "$TRAIN_SEED" \
  --freeze-backbone --freeze-reg-head $CLS_HIDDEN_FLAG --train-classes "$TRAIN_CLASSES"
echo "--- informational eval over the internal 90/10 split (stage 8 is the real validation) ---" >&2
"$BRAIN" yolov8 eval --weights "$WEIGHTS" --data "$POOL_DIR" --conf "$DETECT_CONF" --iou "$DETECT_IOU" >&2 || true

# ============================================================= stage 7: held-out images

log "Stage 7: $N_HOLDOUT fresh held-out target + stranger images"

HOLDOUT_PROMPTS=(
  "a photo of a person on a rooftop at night, city lights behind"
  "a photo of a person walking down a snowy street"
  "a photo of a person standing in an art gallery"
  "a photo of a person riding a bicycle"
  "a photo of a person sitting for a seated interview"
)
HOLDOUT_PROMPTS=("${HOLDOUT_PROMPTS[@]:0:$N_HOLDOUT}")
HOLDOUT_TARGETS="$OUT_DIR/holdout_target.txt" HOLDOUT_STRANGERS="$OUT_DIR/holdout_stranger.txt"
[ -f "$HOLDOUT_TARGETS" ] || : > "$HOLDOUT_TARGETS"
[ -f "$HOLDOUT_STRANGERS" ] || : > "$HOLDOUT_STRANGERS"

n=0
for prompt in "${HOLDOUT_PROMPTS[@]}"; do
  n=$((n + 1))
  out="$OUT_DIR/holdout/target/img-$n.png"
  gen_if_missing "$out" pulid_gen "$prompt"
  append_once "$HOLDOUT_TARGETS" "$out"

  out="$OUT_DIR/holdout/stranger/img-$n.png"
  person="${PEOPLE_DESC[$((n % ${#PEOPLE_DESC[@]}))]}"
  gen_if_missing "$out" flux2_gen "a photo of $person, ${prompt#a photo of a person }"
  append_once "$HOLDOUT_STRANGERS" "$out"
done

# ============================================================= stage 8: validate

log "Stage 8: end-to-end validation"

PASS=1
report() { printf '%-46s %-8s %s\n' "$1" "$2" "$3"; }
report "image" "result" "detail"

# gate <image> <class> <want:1=must-detect|0=must-not> <label>
gate() {
  local img="$1" cls="$2" want="$3" label="$4" lb det
  lb="$img.lb.png"
  [ -f "$lb" ] || pytool letterbox "$img" "$TRAIN_SIZE" "$lb"
  det="$(pytool person-box "$WEIGHTS" "$lb" "$DETECT_CONF" "$cls")"
  if { [ "$want" = 1 ] && [ "$det" != NONE ]; } || { [ "$want" = 0 ] && [ "$det" = NONE ]; }; then
    report "$label" PASS "$([ "$det" = NONE ] && echo "correctly no class-$cls detection" || echo "$det")"
  else
    report "$label" FAIL "$([ "$det" = NONE ] && echo "no class-$cls detection" || echo "false-positive class-$cls detection $det")"
    PASS=0
  fi
}

# The target must be found AS the new class.
while IFS= read -r img; do
  gate "$img" "$TARGET_CLASS" 1 "holdout/target/$(basename "$img") [$TARGET_NAME]"
done < "$HOLDOUT_TARGETS"

# A different real person is the actual discrimination test, and it has TWO
# halves: the new class must not claim them, and COCO's `person` must still.
# Checking only the first would pass a model that had simply forgotten people.
while IFS= read -r img; do
  gate "$img" "$TARGET_CLASS" 0 "holdout/stranger/$(basename "$img") [not $TARGET_NAME]"
  gate "$img" "$PERSON_CLASS" 1 "holdout/stranger/$(basename "$img") [still person]"
done < "$HOLDOUT_STRANGERS"

# Backgrounds contain neither class.
for img in "$OUT_DIR"/train/background/*.png; do
  [ -e "$img" ] || continue
  case "$img" in *.lb.png) continue;; esac
  gate "$img" "$TARGET_CLASS" 0 "background/$(basename "$img") [not $TARGET_NAME]"
done

echo >&2
if [ "$PASS" = 1 ]; then
  echo "ALL GATES PASSED." >&2
  echo "Trained detector: $WEIGHTS" >&2
  echo "  $BRAIN yolov8 detect --weights $WEIGHTS --image <letterboxed-to-${TRAIN_SIZE}x${TRAIN_SIZE}.png> --conf $DETECT_CONF --iou $DETECT_IOU" >&2
  echo "  (letterbox first: python3 $PY_TOOLS letterbox <src> $TRAIN_SIZE <out.png>)" >&2
  exit 0
else
  echo "ONE OR MORE GATES FAILED - see the scorecard above. Weights are still at $WEIGHTS but detection is not validated." >&2
  exit 1
fi
