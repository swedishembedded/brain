#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Fetch what this sample's webcam mode runs, each pinned and checked against
# its sha256 before it is moved into place:
#
#  * the face tracker: OpenSeeFace's face detector and two landmark networks
#    (code and weights BSD-2-Clause, Copyright (c) 2019 Emiliana), about
#    16 MB, into brain's model store as `emilianavt/OpenSeeFace`
#    ($BRAIN_MODELS_DIR, else $XDG_DATA_HOME/brain/models, else
#    ~/.local/share/brain/models - or the directory given as the first
#    argument);
#  * the OpenVINO runtime (Apache-2.0), which compiles the tracker for the
#    Intel NPU and GPU: the official wheel from PyPI, of which only the
#    shared libraries are unpacked (about 150 MB), into
#    $XDG_DATA_HOME/brain/runtimes/openvino-<version>/libs. Nothing is
#    installed system-wide and no Python is needed to run the sample. The
#    NPU additionally needs the host's NPU driver (intel-level-zero-npu);
#    without it the sample reports the NPU as absent and uses the GPU.
#
# Idempotent: what is already present and verified is left alone.
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

OV_VERSION=2026.4.1
OV_WHEEL=openvino-2026.4.1-22982-cp312-cp312-manylinux_2_28_x86_64.whl
OV_URL="https://files.pythonhosted.org/packages/ec/3a/2a173ac1ad749ff0b041788eefc1ade0d410231fedfc43f77474f3b806cc/$OV_WHEEL"
OV_SHA256=6b4375c17ddcac83a5180349e2e2bb811185c261066e2a920659892d58ef0e3b
if [ "$(uname -m)" != x86_64 ] || [ "$(uname -s)" != Linux ]; then
  echo "skipping OpenVINO: the NPU/GPU path is built for x86_64 Linux"
  exit 0
fi
ov_dir="${XDG_DATA_HOME:-$HOME/.local/share}/brain/runtimes/openvino-$OV_VERSION"
if ls "$ov_dir"/libs/libopenvino_c.so.* >/dev/null 2>&1; then
  echo "ok       OpenVINO $OV_VERSION"
else
  mkdir -p "$ov_dir"
  wheel="$(mktemp "$ov_dir/.wheel.XXXXXX")"
  stage="$(mktemp -d "$ov_dir/.libs.XXXXXX")"
  trap 'rm -rf "$wheel" "$stage"' EXIT
  echo "fetching OpenVINO $OV_VERSION"
  curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error -o "$wheel" "$OV_URL"
  if ! echo "$OV_SHA256  $wheel" | sha256sum --check --status; then
    echo "error: $OV_WHEEL does not have the pinned sha256 $OV_SHA256" >&2
    exit 1
  fi
  if command -v unzip >/dev/null; then
    unzip -q "$wheel" 'openvino/libs/*' -d "$stage"
  else
    python3 -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); [z.extract(n, sys.argv[2]) for n in z.namelist() if n.startswith("openvino/libs/")]' "$wheel" "$stage"
  fi
  rm -rf "$ov_dir/libs"
  mv "$stage/openvino/libs" "$ov_dir/libs"
  rm -rf "$wheel" "$stage"
  trap - EXIT
  echo "ok       OpenVINO $OV_VERSION"
fi
echo "the OpenVINO runtime is in $ov_dir/libs"
