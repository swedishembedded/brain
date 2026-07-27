#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Fetch a pretrained Ultralytics YOLOv8 checkpoint and export it to brain's native
# `.weights` format, ready to serve as an object detector (e.g. over `brain serve
# --dbus`, or `brain yolo detect`).
#
#   scripts/fetch-yolov8.sh [--variant yolov8n] [--out DIR]
#
# Defaults: variant yolov8n (nano, 80 COCO classes, 640px), out /data/resources/yolo.
# Produces  <out>/<variant>.pt  (the source) and  <out>/<variant>.brain.weights
# (the brain checkpoint). Point brain at the latter:
#
#   export BRAIN_YOLO=<out>/<variant>.brain.weights
#
# Needs Python with torch + ultralytics to READ the .pt (the export writer is pure
# Python). This installs them (CPU torch) if missing — a large one-time download.
set -euo pipefail

VARIANT=yolov8n
OUT=/data/resources/yolo
ASSETS_URL=https://github.com/ultralytics/assets/releases/download/v8.2.0

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORTER="$REPO_ROOT/tools/yolo_export/export_yolov8.py"
[ -f "$EXPORTER" ] || { echo "exporter not found: $EXPORTER" >&2; exit 1; }

# Fall back to a writable dir if the requested one cannot be created.
if ! mkdir -p "$OUT" 2>/dev/null; then
  OUT="$REPO_ROOT/../resources/yolo"
  echo "note: falling back to $OUT" >&2
  mkdir -p "$OUT"
fi

PT="$OUT/$VARIANT.pt"
WEIGHTS="$OUT/$VARIANT.brain.weights"

# 1. download the pretrained .pt if we do not already have it.
if [ ! -s "$PT" ]; then
  echo "== downloading $VARIANT.pt =="
  curl -fSL --retry 3 -o "$PT" "$ASSETS_URL/$VARIANT.pt"
fi
echo "source: $PT ($(du -h "$PT" | cut -f1))"

# 2. ensure torch + ultralytics are importable (CPU torch; large one-time install).
if ! python3 -c "import torch, ultralytics" 2>/dev/null; then
  echo "== installing torch (CPU) + ultralytics =="
  python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  python3 -m pip install --quiet ultralytics
fi

# 3. export -> brain native weights.
echo "== exporting -> $WEIGHTS =="
python3 "$EXPORTER" --weights "$PT" --out "$WEIGHTS"

echo
echo "done. serve it with:"
echo "  export BRAIN_YOLO=$WEIGHTS"
