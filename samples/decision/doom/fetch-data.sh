#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Fetch and build everything this sample needs that it cannot ship: the
# modified DOOM engine, a game WAD, and the sentence encoder.
#
# Nothing is committed and nothing is installed system-wide. Everything lands
# under a directory you choose (default ./.data, which is gitignored), and the
# script finishes by printing the exact flags to pass. The sample reads no
# environment variables: what it runs on is what you typed.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${1:-$here/.data}"
mkdir -p "$data"

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- the engine
#
# RESTful-DOOM is Chocolate Doom with an HTTP API inside its game loop. This
# sample needs the fork with the agent endpoints (/api/state, /api/step,
# /api/episode, /api/frame) and lockstep mode; upstream has the original
# human-facing API only.
DOOM_REPO="https://github.com/mkschreder/restful-doom"
doom_src="$data/restful-doom"
if [ ! -x "$doom_src/src/restful-doom" ]; then
  if [ ! -d "$doom_src/.git" ]; then
    say "cloning the engine from $DOOM_REPO"
    git clone --depth 1 "$DOOM_REPO" "$doom_src"
  fi
  say "building the engine in $doom_src"
  # Needs: gcc, make, automake, autoconf, pkg-config, and the SDL2, SDL2_mixer
  # and SDL2_net development packages. On Debian/Ubuntu:
  #   apt-get install build-essential automake autoconf pkg-config \
  #                   libsdl2-dev libsdl2-mixer-dev libsdl2-net-dev
  ( cd "$doom_src" && [ -f configure ] || ./autogen.sh )
  ( cd "$doom_src" && make -j"$(nproc)" )
fi
if [ ! -x "$doom_src/src/restful-doom" ]; then
  echo "   the engine did not build; see the output above" >&2
  exit 1
fi
doom_bin="$doom_src/src/restful-doom"
echo "   engine: $doom_bin"

# ------------------------------------------------------------------ the WAD
#
# The DOOM engine is free software; the game data is not ours to redistribute,
# which is why nothing here is committed. doom1.wad is the SHAREWARE episode id
# Software released for free distribution - it is the one to use.
wad="$data/doom1.wad"
if [ ! -f "$wad" ]; then
  say "fetching the shareware IWAD"
  curl -fsSL -o "$wad" \
    https://raw.githubusercontent.com/Akbar30Bill/DOOM_wads/master/doom1.wad
fi
# An IWAD starts with the four bytes "IWAD". A proxy or a rate limit hands back
# an HTML page with a 200, and the failure then shows up as an incomprehensible
# error from deep inside the engine's WAD loader.
if [ "$(head -c 4 "$wad")" != "IWAD" ]; then
  echo "   $wad is not a WAD (the download returned something else); removing it" >&2
  rm -f "$wad"
  exit 1
fi
echo "   wad: $wad ($(stat -c %s "$wad") bytes)"

# -------------------------------------------------------------- the encoder
#
# The policy reads text, so it needs a sentence encoder. Any BERT-shaped one
# with config.json / model.safetensors / tokenizer.json works; this is the one
# brain's other decision samples use.
encoder="${ENCODER_DIR:-$HOME/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2}"
if [ ! -f "$encoder/config.json" ]; then
  say "the sentence encoder is missing"
  echo "   Run:  brain pull sentence-transformers/all-MiniLM-L6-v2"
  echo "   or point --encoder at any BERT-shaped checkpoint directory."
else
  echo "   encoder: $encoder"
fi

cat <<EOF

== ready. Run the sample with:

  make samples/decision/doom/run ARGS="probe \\
    --doom-bin $doom_bin \\
    --wad $wad \\
    --encoder $encoder"

EOF
