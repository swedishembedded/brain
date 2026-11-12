#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Fetch the face tracker this sample's webcam mode runs: OpenSeeFace's face
# detector and two landmark networks (code and weights BSD-2-Clause,
# Copyright (c) 2019 Emiliana), at a pinned commit, each checked against its
# sha256 before it is moved into place.
#
# They land in brain's model store as `emilianavt/OpenSeeFace`, the same
# place brain's tests and this sample look: $BRAIN_MODELS_DIR, else
# $XDG_DATA_HOME/brain/models, else ~/.local/share/brain/models - or the
# directory given as the first argument. About 16 MB.
set -euo pipefail

COMMIT=50c09741897010faad721a218b2bbe14ecad6251
BASE="https://raw.githubusercontent.com/emilianavt/OpenSeeFace/$COMMIT/models"
FILES=(
  "mnv3_detection_opt.onnx 0e8e4806766d85ab067a52c7af0dcb59eb7f9dfe580b44f20a8e6ab712d89809"
  "lm_model0_opt.onnx 43c9c58f03701cc15b8423c49b78af348a6397334d348887e0fda59013275c29"
  "lm_model3_opt.onnx 5aa0ab2594acaf2e1d3916a8de592cffeb1f1a7a7f778cad2f37948bff6549d1"
)

if [ -n "${1:-}" ]; then
  root="$1"
elif [ -n "${BRAIN_MODELS_DIR:-}" ]; then
  root="$BRAIN_MODELS_DIR"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
  root="$XDG_DATA_HOME/brain/models"
else
  root="$HOME/.local/share/brain/models"
fi
dest="$root/emilianavt/OpenSeeFace"
mkdir -p "$dest"

for entry in "${FILES[@]}"; do
  read -r name sum <<<"$entry"
  if [ -f "$dest/$name" ] && echo "$sum  $dest/$name" | sha256sum --check --status; then
    echo "ok       $name"
    continue
  fi
  tmp="$(mktemp "$dest/.$name.XXXXXX")"
  trap 'rm -f "$tmp"' EXIT
  echo "fetching $name"
  curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error -o "$tmp" "$BASE/$name"
  if ! echo "$sum  $tmp" | sha256sum --check --status; then
    echo "error: $name does not have the pinned sha256 $sum" >&2
    exit 1
  fi
  mv "$tmp" "$dest/$name"
  trap - EXIT
  echo "ok       $name"
done
echo "the face tracker is in $dest"
