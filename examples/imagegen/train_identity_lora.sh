#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# A few photographs of one person in, a LoRA that knows their face out.
#
#   examples/imagegen/train_identity_lora.sh <photo-dir> "<their name>"
#
# This is `train_lora.sh` specialised for the case where the concept is a
# PERSON, which changes three things that matter more than the hyperparameters:
#
#  1. **The dataset is built by the face detector, not by the crop tool.**
#     Holiday snaps and selfies are mostly not-the-person: background, other
#     people, half a restaurant. Trained on those, the adapter learns the
#     restaurant. Every photograph is therefore cropped square around the
#     detected primary face, and the crop window SLIDES to push bystanders out
#     of frame before it gives up and shrinks. A photograph whose face cannot
#     be framed without a bystander is dropped rather than poisoned in.
#
#  2. **The captions describe everything EXCEPT the face.** Framing, pose,
#     clothing, background, light -- yes. Eyes, skin, head shape -- never. An
#     adapter binds a concept to whatever the caption does not already explain,
#     so describing the face hands the identity to those words instead of to
#     the name, and the trigger ends up meaning "a photo of a man".
#
#  3. **The trigger is a name and is captioned as a name.** `brain label
#     --trigger-role` is what says so; the default calls a trigger a style,
#     which is the wrong binding for a face.
#
# Each source photograph also enters mirrored. Three photographs is thin for
# this -- expect the adapter to know the face and to have opinions about pose
# and lighting that it should not have. More photographs, in more places, is
# worth more than more steps.
#
# Grade the result, always, with a number:
#
#   examples/imagegen/identity_score.sh <photo-dir> <generated-dir>
#
# then generate with `ADAPTER=<out> examples/imagegen/portrait_from_refs.sh`.
#
# Optional, all env: OUT (adapter path), STEPS, RANK, LR, SIZE, VARIANT,
# TRAINER, CKPT_EVERY, MODEL (captioner), BRAIN.
#
# Weights: BRAIN_SCRFD_DIR for the detector, BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}
# for training, and the captioner's own ($BRAIN_QWEN3VL_WEIGHTS by default).

set -euo pipefail

DIR="${1:?usage: train_identity_lora.sh <photo-dir> \"<their name>\"}"; DIR="${DIR%/}"
NAME="${2:?the name of the person -- this is the trigger you will type at generation time}"
BRAIN="${BRAIN:-./target/release/brain}"
OUT="${OUT:-$DIR/identity.brain}"
DATA="$DIR/.identity"
: "${BRAIN_SCRFD_DIR:?set BRAIN_SCRFD_DIR to the directory holding scrfd_10g_bnkps.onnx}"

rm -rf "$DATA"; mkdir -p "$DATA"

BRAIN="$BRAIN" python3 - "$DIR" "$DATA" <<'PY'
import glob, json, os, subprocess, sys
from PIL import Image

BRAIN = os.environ["BRAIN"]
src, out = sys.argv[1], sys.argv[2]
IMG = (".jpg", ".jpeg", ".png", ".ppm", ".webp", ".bmp")


def faces(path):
    r = subprocess.run([BRAIN, "scrfd", "detect", "--in", f"image={path}", "--max_faces", "16"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("faces:"):
            return json.loads(line[6:])
    return []


def overlap(box, face):
    w = max(0.0, min(box[2], face[2]) - max(box[0], face[0]))
    h = max(0.0, min(box[3], face[3]) - max(box[1], face[1]))
    return w * h


def crop_box(size, primary, others, zoom):
    """The widest square <= zoom*face holding the whole primary face and no
    bystander. Slides the window to dodge bystanders before it shrinks it --
    a group photo usually has a clean framing, just not a centred one."""
    W, H = size
    face = max(primary[2] - primary[0], primary[3] - primary[1])
    cx, cy = (primary[0] + primary[2]) / 2, (primary[1] + primary[3]) / 2
    side = min(zoom * face, W, H)
    while side >= 1.3 * face:
        xlo, xhi = max(0, primary[2] - side), min(primary[0], W - side)
        ylo, yhi = max(0, primary[3] - side), min(primary[1], H - side)
        best = None
        if xlo <= xhi and ylo <= yhi:
            for i in range(13):
                for j in range(13):
                    x = xlo + (xhi - xlo) * i / 12
                    y = ylo + (yhi - ylo) * j / 12
                    box = (x, y, x + side, y + side)
                    if any(overlap(box, o) > 0.10 * ((o[2] - o[0]) * (o[3] - o[1])) for o in others):
                        continue
                    d = (x + side / 2 - cx) ** 2 + (y + side / 2 - cy) ** 2
                    if best is None or d < best[0]:
                        best = (d, box)
        if best:
            return tuple(int(round(v)) for v in best[1])
        side *= 0.97
    return None


n = 0
for i, path in enumerate(sorted(p for p in glob.glob(f"{src}/*") if p.lower().endswith(IMG)), 1):
    found = faces(path)
    if not found:
        print(f"  {os.path.basename(path)}: no face - skipped")
        continue
    found.sort(key=lambda f: -((f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])))
    primary, others = found[0]["bbox"], [f["bbox"] for f in found[1:]]
    im = Image.open(path).convert("RGB")
    seen = set()
    for tag, zoom in (("a", 1.55), ("b", 2.20)):
        box = crop_box(im.size, primary, others, zoom)
        if box is None:
            print(f"  {os.path.basename(path)} [{tag}]: no framing avoids {len(others)} bystander(s) - skipped")
            continue
        if box in seen:      # the two zooms collapsed onto one framing
            continue
        seen.add(box)
        square = im.crop(box).resize((768, 768), Image.LANCZOS)
        for mirror, suffix in ((False, ""), (True, "m")):
            view = square.transpose(Image.FLIP_LEFT_RIGHT) if mirror else square
            view.save(f"{out}/{i:02d}-{tag}{suffix}.png")
            n += 1
        print(f"  {os.path.basename(path)} [{tag}]: {box}, {len(others)} bystander(s) excluded")

if n == 0:
    raise SystemExit("no usable face crops - identity training needs photographs with a clear primary face")
print(f"{n} training images (each photograph also enters mirrored)")
PY

# Captions cover framing, pose, clothing, setting and light and deliberately
# omit the face: what the caption does not explain is what the trigger inherits.
"$BRAIN" label images "$DATA" --model "${MODEL:-qwen3vl}" \
  --trigger "$NAME" --trigger-role "the name of the person" --max-new 130 --max-pixels 200704 \
  --prompt "Write ONE sentence describing this photograph of a person, covering only: the framing and camera distance, the direction their head is turned and where they look, their expression, their clothing, the background and setting, and the lighting. Do NOT describe their face, head, skin, eyes, eyebrows or facial hair. Do not mention any other people."

echo "captions written to $DATA/captions.yaml - review them, then press enter" >&2
[ -t 0 ] && read -r _

exec "$BRAIN" flux2 finetune "$DATA" --out "$OUT" \
  --variant "${VARIANT:-klein-4b}" --trainer "${TRAINER:-device}" \
  --steps "${STEPS:-1500}" --rank "${RANK:-16}" --lr "${LR:-1e-4}" \
  --size "${SIZE:-512}" --ckpt-every "${CKPT_EVERY:-100}"
