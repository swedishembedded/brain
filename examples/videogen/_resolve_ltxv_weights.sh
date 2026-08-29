# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Sourced, not run: fills in BRAIN_LTXV_{DIT,VAE,TEXT_ENCODER} (plus the two
# optional roles, when present) from ONE model directory, instead of every
# script separately asking for four env vars. Any of the three that is
# already set in the environment is left alone - a caller who wants to name
# an unusual file directly still can.
#
# The directory is $LTX_MODEL_DIR, or by default
# $BRAIN_MODELS_DIR/Lightricks/LTX-2.5 (brain's own model-directory
# precedence: BRAIN_MODELS_DIR, else $XDG_DATA_HOME/brain/models, else
# ~/.local/share/brain/models - see crates/modelstore's default_root doc),
# matching where `brain models pull Lightricks/LTX-2.5` (or a manual
# download) would put it. It is read FLAT: every file directly inside it, no
# vae/, text_encoders/, diffusion_models/ subfolders the way the official
# https://huggingface.co/Lightricks/LTX-2.5 repo ships them - move the ones
# you need up to the top level once, rather than this script guessing which
# subfolder layout you used.
#
# Filenames matched are the OFFICIAL ones from that repo, with one
# exception: brain's DiT loader reads a GGUF
# (crate::gguf_src::LtxvGgufSource, see crates/ltxv/src/pipeline.rs's
# `Paths::dit` doc), and Lightricks publishes the 22B transformer only as
# .safetensors (bf16 / comfy-int8-convrot / nvfp4). A GGUF quantization of it
# is not something this directory will ever contain from the official repo
# alone - it has to come from a community conversion or from running
# `brain quantize` yourself against the bf16 file. This script says so, out
# loud, before asking where it is.
#
# Any role not found is asked for interactively (a path, or blank to skip
# the optional ones) - never silently left unset, and never hung waiting on
# a prompt when stdin is not a terminal (e.g. driven from CI): that case is
# a clear error instead.

MODELS_DIR="${BRAIN_MODELS_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/brain/models}"
LTX_DIR="${LTX_MODEL_DIR:-$MODELS_DIR/Lightricks/LTX-2.5}"

# $1: role name (for messages). $2: BRAIN_LTXV_* var name. $3: required
# ("1") or optional (""). $@ from $4: candidate basenames, in preference
# order, resolved against $LTX_DIR.
_ltxv_resolve_one() {
  local role="$1" var="$2" required="$3"
  shift 3
  # Already set (a caller's own export, or a real file name/glob) - honour it.
  [ -n "${!var:-}" ] && return 0

  local cand
  for cand in "$@"; do
    # shellcheck disable=SC2086 # intentional glob expansion, one match expected
    local hit
    hit=$(compgen -G "$LTX_DIR/$cand" 2>/dev/null | head -n1) || true
    if [ -n "$hit" ] && [ -f "$hit" ]; then
      export "$var=$hit"
      echo "ltxv weights: $role -> $hit" >&2
      return 0
    fi
  done

  if [ -z "$required" ]; then
    return 0   # optional role, quietly unset
  fi

  echo "ltxv weights: no $role found under $LTX_DIR (tried: $*)" >&2
  if [ "$role" = "DiT" ]; then
    echo "  Lightricks' own repo publishes the 22B transformer only as .safetensors;" >&2
    echo "  brain's DiT loader needs a GGUF quantization of it (Q8_0 or Q4_K_M), which" >&2
    echo "  has to come from a community conversion or from running 'brain quantize'" >&2
    echo "  yourself against the bf16 file - it will not appear here on its own." >&2
  fi
  if [ ! -t 0 ]; then
    echo "  stdin is not a terminal, so this cannot prompt for a path - set $var yourself." >&2
    exit 1
  fi
  read -r -p "  path to $role: " reply
  [ -n "$reply" ] && [ -f "$reply" ] || { echo "ltxv weights: no file at '$reply'" >&2; exit 1; }
  export "$var=$reply"
}

_ltxv_resolve_one "VAE" BRAIN_LTXV_VAE 1 \
  "ltx-2.5-video-vae-bf16.safetensors" "ltx-2.5-video-vae-conv-bf16.safetensors"
_ltxv_resolve_one "DiT" BRAIN_LTXV_DIT 1 \
  "ltx-2.5-22b-distilled-transformer-*.gguf" "ltx-2.5-22b-dev-transformer-*.gguf"
_ltxv_resolve_one "text encoder" BRAIN_LTXV_TEXT_ENCODER 1 \
  "gemma4-*-with-proj-ltx-2.5-bf16.safetensors"
_ltxv_resolve_one "spatial upsampler" BRAIN_LTXV_UPSAMPLER_SPATIAL "" \
  "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
_ltxv_resolve_one "audio VAE" BRAIN_LTXV_AUDIO_VAE "" \
  "ltx-2.5-audio-vae-bf16.safetensors"

unset -f _ltxv_resolve_one
