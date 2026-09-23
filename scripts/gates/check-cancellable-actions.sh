#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# check-cancellable-actions.sh - a ratchet over the models whose streaming
# actions accept a cancel token and never look at it.
#
# `capability::Invocation` carries a `CancelToken` on EVERY call, so every
# action accepts one whether or not it honours it. A caller cannot tell the
# difference from the manifest, and it builds real resource management on the
# assumption that it can stop a generation whose consumer went away. An action
# that never polls is therefore worse than one that never offered: the caller
# waits for a generation that will run to `max_new` no matter what.
#
# Until that fact is advertised per action, this gate keeps it visible and
# stops it spreading: a crate with a `.streaming()` capability action must
# reference the invocation's `cancel` somewhere in its sources, unless it is
# on the KNOWN list below. Adding a new streaming action that ignores
# cancellation fails here. Teaching a listed crate to poll is what shrinks the
# list - remove it from KNOWN and the gate holds you to it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Streaming capability actions that do NOT poll the caller's cancel token.
# Every name here is a model whose streaming run cannot be stopped early:
# firing the token changes nothing, and the caller waits for it to finish.
KNOWN="
cosyvoice
deepseek2ocr
deepseekocr2
fastvlm
lfm2
llava
minimaxmusic3
moondream3
qwen3omnimoe
"

fail=0
for caps in crates/*/src/caps.rs; do
  crate=$(basename "$(dirname "$(dirname "$caps")")")
  grep -q "streaming()" "$caps" || continue
  if grep -rq "inv\.cancel\|\.cancel\.is_cancelled()\|cancel: &CancelToken" "crates/$crate/src"; then
    polls=yes
  else
    polls=no
  fi
  listed=no
  for k in $KNOWN; do [ "$k" = "$crate" ] && listed=yes; done

  if [ "$polls" = no ] && [ "$listed" = no ]; then
    echo "check-cancellable-actions: $crate declares a streaming action but never reads the invocation's cancel token."
    echo "  Poll it between generated tokens/steps (see crates/capability/src/lib.rs, CancelToken),"
    echo "  or add '$crate' to KNOWN in this script and say so in the model's own doc."
    fail=1
  fi
  if [ "$polls" = yes ] && [ "$listed" = yes ]; then
    echo "check-cancellable-actions: $crate now polls the cancel token - remove it from KNOWN in this script."
    fail=1
  fi
done

if [ "$fail" = 0 ]; then
  n=$(echo "$KNOWN" | grep -c '[a-z]')
  echo "check-cancellable-actions: OK ($n model(s) knowingly ignore cancellation)"
fi
exit "$fail"
