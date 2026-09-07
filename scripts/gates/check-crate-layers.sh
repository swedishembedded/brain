#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The crates listed in LEAF_CRATES below must stay BELOW the model layer.
#
# `.agents/rules/architecture.md` stacks this workspace in six layers and says
# each may depend only on the layers above it. Nothing enforced that, and one
# violation is not a style problem - it is a hard Cargo dependency cycle that
# only shows up when the crate that needs the code tries to use it:
#
#   brain-qwen3 (layer 4) -> brain-rl -> brain-bench -> brain-qwen3
#
# That cycle is what blocked the `lora_gate` capability action from reusing
# the promote/reject gate at all, and it is why the model-agnostic half of
# that machinery lives in its own leaf crate now. A leaf crate is only worth
# the split for as long as it STAYS a leaf: one `brain-bench.workspace = true`
# added for one convenient helper puts every model crate back on the wrong
# side of the cycle. This gate is the thing that says so, at check time,
# instead of six months later at the next call site.
#
# The rule: a leaf crate's whole normal-dependency closure (its own
# dependencies, transitively - dev-dependencies excluded, since Cargo permits
# cycles through those and they never reach a consumer) may contain only
# crates from layers 1-3.
#
# Checked with --all-features, not at the default feature set: `brain-rl`
# already keeps `brain-qwen3` behind an off-by-default `qwen3` feature, so a
# default-features-only check would wave through exactly the same escape
# hatch on a leaf crate and report a clean closure that no longer holds the
# moment anyone turns the feature on.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 2

# Crates that must remain usable from a layer-4 model crate.
LEAF_CRATES="brain-promote"

# Layers 1-3 of .agents/rules/architecture.md: kernels + the architecture
# registry, the accelerator seam, and the training substrate. Everything else
# in the workspace is layer 4 or below it and may not appear in a leaf crate's
# closure.
LAYER_1_TO_3="
brain-kernels
brain-arch
brain-backend-api
brain-backend-wgpu
brain-backend-cpu
brain-backend-vulkan
brain-wgsl-cpu
brain-gpu-core
brain-paramstore
brain-optim
brain-checkpoint
brain-gguf
brain-data
brain-model
"

fail=0

for leaf in $LEAF_CRATES; do
    if ! tree=$(cargo tree --offline --edges normal --prefix none --all-features -p "$leaf" 2>&1); then
        echo "check-crate-layers: FAIL - cannot resolve $leaf"
        echo "$tree" | sed 's/^/    /'
        fail=1
        continue
    fi
    # `cargo tree` prints "<name> v<version> (<path>)"; the brain-* rows are the
    # only ones this rule is about (a third-party crate carries no layer).
    closure=$(echo "$tree" | awk '{print $1}' | grep '^brain' | sort -u)
    bad=""
    for dep in $closure; do
        [ "$dep" = "$leaf" ] && continue
        case " $(echo "$LAYER_1_TO_3" | tr '\n' ' ') " in
        *" $dep "*) ;;
        *) bad="$bad $dep" ;;
        esac
    done
    if [ -n "$bad" ]; then
        echo "check-crate-layers: FAIL - $leaf must stay below the model layer, but its closure reaches:"
        for dep in $bad; do echo "    $dep"; done
        echo
        echo "    $leaf exists so a layer-4 model crate can depend on it. Anything"
        echo "    at layer 4 or below in its closure re-creates the cycle the split"
        echo "    was made to break - see .agents/rules/architecture.md."
        fail=1
    fi
done

[ "$fail" -ne 0 ] && exit 1
echo "check-crate-layers: exit 0, $(echo "$LEAF_CRATES" | wc -w | tr -d ' ') leaf crate(s), every closure within layers 1-3"
exit 0
