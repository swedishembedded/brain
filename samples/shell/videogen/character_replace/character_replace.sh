#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for putting a new character into
# existing footage for its clients. If your team needs expertise in
# generative video editing pipelines then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# Replace a person in a clip with the character from a sheet - or run one
# step of that on its own. Every option is a flag; see README.md, or run
# this script with --help.

set -euo pipefail

usage() {
  cat <<'EOF'
character_replace.sh <mode> [options] [-- brain flags]

Modes:
  run     the whole process: track the person, paint the character in on a
          few stills, regenerate the person's region, composite
          (brain ltxv replace)
  track   only the person's track: a mask sequence to check the clicks
          against, or to hand back to `run --actor-mask`
          (brain sam2 track)
  video   only the video pass, over a mask sequence and stills you already
          have (brain ltxv v2v --mask ... --keyframe ...)

Inputs:
  --input <clip>          the source video (.mp4 / .webm / .mkv / .mov); any
                          length and size - `run` trims and crops it onto
                          LTX-2.5's grid and says so
  --character <image>     the character sheet (.png / .jpg)             [run]
  --point <x,y>           a click on the person, in source pixels; repeat it
                          over head, coat and legs                [run, track]
  --exclude <x,y>         a click on something that is not the person
  --prompt-frame <n>      the frame the clicks sit on (default 0)
  --actor-mask <dir>      the person's track instead of clicks          [run]
  --mask <dir>            the region to regenerate                    [video]
  --keyframe <F:image>    a still guiding frame F (repeatable)        [video]
  --prompt <text>         the clip as it should look           [run, video]
  --soundtrack <wav>      sound to carry into the result                [run]

Outputs:
  --out <path>            the result: a clip (run, video) or a mask
                          directory (track). Default:
                          out/character_replace/<input name>-<mode>[.mp4]
  --work-dir <dir>        the intermediates of `run` (default <out>.work); a
                          repeated `run` resumes from them - steps whose
                          inputs and settings are unchanged are not redone
  --fresh                 redo every step of `run`

Models (each defaults to what the model store resolves):
  --video-dit <file>      the LTX-2.5 DiT (a .gguf)
  --video-vae <file>      the LTX-2.5 video VAE
  --text-encoder <file>   the LTX-2.5 Gemma-4 text encoder
  --adapter <file>        a LoRA for the video pass (repeatable;
  --lora-scale <s>          a --lora-scale after it sets its strength)
  --editor-dit <file>     the FLUX.2 DiT that paints the stills
  --editor-variant <v>    klein-4b | klein-9b | dev
  --editor-vae <file>     the FLUX.2 VAE
  --editor-text-encoder <dir>   the FLUX.2 text encoder
  --editor-tokenizer <file>     the FLUX.2 tokenizer.json
  --edit-adapter <file>   a LoRA for the still edits (repeatable;
  --edit-lora-scale <s>     an --edit-lora-scale after it sets its strength)
  --tracker-weights <file>  the SAM 2.1 checkpoint
  --tracker-variant <v>   tiny | large

This script:
  --brain <path>          the brain binary (default ./target/release/brain)
  --dry-run               print the brain command instead of running it
  --help                  this text

Every other flag goes to the brain command unchanged - all of
`brain ltxv replace --help` (guides, overlap threshold, region growth,
feather, seed, steps, device, ...) in `run`, `brain sam2 track --help` in
`track`, `brain ltxv v2v --help` in `video`.
EOF
}

[ $# -gt 0 ] || { usage; exit 2; }
case "$1" in
  run | track | video) MODE="$1"; shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "character_replace.sh: unknown mode '$1' (run, track or video)" >&2; usage >&2; exit 2 ;;
esac

BRAIN=./target/release/brain
DRY=0
INPUT=""
OUT=""
POINTS=""   # track: the clicks, joined the way `brain sam2 track` takes them
EXCLUDE=""
ARGS=()      # flags for the brain command, in order
PASS=()      # flags after a literal --, unchanged

need() { [ $# -ge 2 ] || { echo "character_replace.sh: $1 needs a value" >&2; exit 2; }; }

while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; PASS=("$@"); break ;;
    -h | --help) usage; exit 0 ;;
    --brain) need "$@"; BRAIN="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --input) need "$@"; INPUT="$2"; shift 2 ;;
    --out) need "$@"; OUT="$2"; shift 2 ;;
    --video-dit) need "$@"; ARGS+=(--dit "$2"); shift 2 ;;
    --video-vae) need "$@"; ARGS+=(--vae "$2"); shift 2 ;;
    --editor-dit) need "$@"; ARGS+=(--component "flux2.dit=$2"); shift 2 ;;
    --editor-variant) need "$@"; ARGS+=(--component "flux2.variant=$2"); shift 2 ;;
    --editor-vae) need "$@"; ARGS+=(--component "flux2.vae=$2"); shift 2 ;;
    --editor-text-encoder) need "$@"; ARGS+=(--component "flux2.text_encoder=$2"); shift 2 ;;
    --editor-tokenizer) need "$@"; ARGS+=(--component "flux2.tokenizer=$2"); shift 2 ;;
    --point | --exclude)
      need "$@"
      if [ "$MODE" != track ]; then ARGS+=("$1" "$2")
      elif [ "$1" = --point ]; then POINTS="${POINTS:+$POINTS;}$2"
      else EXCLUDE="${EXCLUDE:+$EXCLUDE;}$2"; fi
      shift 2 ;;
    --tracker-weights)
      need "$@"
      if [ "$MODE" = track ]; then ARGS+=(--weights "$2"); else ARGS+=(--component "sam2.weights=$2"); fi
      shift 2 ;;
    --tracker-variant)
      need "$@"
      if [ "$MODE" = track ]; then ARGS+=(--variant "$2"); else ARGS+=(--component "sam2.variant=$2"); fi
      shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

[ -n "$INPUT" ] || { echo "character_replace.sh: --input <clip> is required" >&2; exit 2; }
[ -f "$INPUT" ] || { echo "character_replace.sh: no clip at $INPUT" >&2; exit 1; }
if [ ! -x "$BRAIN" ]; then
  echo "character_replace.sh: no brain binary at $BRAIN - build it with 'make build/release' in brain's checkout, or name it with --brain <path>" >&2
  exit 1
fi

STEM=$(basename "${INPUT%.*}")
case "$MODE" in
  run)
    OUT="${OUT:-out/character_replace/$STEM-replaced.mp4}"
    CMD=("$BRAIN" ltxv replace --input "$INPUT" --output-path "$OUT") ;;
  track)
    OUT="${OUT:-out/character_replace/$STEM-track}"
    CMD=("$BRAIN" sam2 track --video "$INPUT" --points "$POINTS" --out "masks=$OUT")
    [ -z "$EXCLUDE" ] || CMD+=(--exclude "$EXCLUDE") ;;
  video)
    OUT="${OUT:-out/character_replace/$STEM-video.mp4}"
    CMD=("$BRAIN" ltxv v2v --input "$INPUT" --output-path "$OUT") ;;
esac
CMD+=(${ARGS[@]+"${ARGS[@]}"} ${PASS[@]+"${PASS[@]}"})

if [ "$DRY" = 1 ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
mkdir -p "$(dirname "$OUT")"
"${CMD[@]}"
echo "character_replace.sh: $MODE wrote $OUT"
[ "$MODE" = run ] && echo "character_replace.sh: intermediates in ${OUT}.work (unless --work-dir named another)"
exit 0
