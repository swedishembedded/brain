#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for validating a ported model on the
# accelerator it will be served from, one ordered and repeatable command per
# step, for its clients. If your team needs expertise in bringing a large
# language model from CPU parity to measured GPU serving then you can procure
# our services by sending an email to info@swedishembedded.com.
#
# The Ripple-65B suites of the GPU validation phase, on one chosen device.
# `make test/gpu/ripple/...` is the front door; the ordered plan these serve is the
# Ripple GPU validation ledger under the roadmap.
#
#   scripts/test/ripple-gpu.sh suites      print the cargo suites and what each guards
#   scripts/test/ripple-gpu.sh cargo       run every suite on the device, report per suite
#   scripts/test/ripple-gpu.sh e2e         run tests/e2e/ripple.bats with the device
#
# Settings (make variables or environment of this script, a test harness and not
# of the brain crates):
#   DEVICE   cpu (default), software (lavapipe), or gpu<N>[,gpu<M>...]; the brain
#            --device grammar for the cases below
#   BACKEND  cuda or vulkan, for gpu devices only (default: cuda); with cpu and
#            software there is no backend choice and a value is refused
#   REQUIRE  1: a skip by name is a failure (BRAIN_REQUIRE_FIXTURES), for the real-weight steps
#   BRAIN_BIN  the binary `e2e` serves with (default ./target/debug/brain)
#
# Cards belong to other jobs while the node is shared: a gpu device is refused in
# a shell that hides the cards (CUDA_VISIBLE_DEVICES set empty by
# ~/tmp/no-gpu-env.sh), so a CPU-only session cannot start a card run by accident.
# Taking a card (lock, free memory) is the caller's step: step 0 of the ledger.
set -uo pipefail
cd "$(dirname "$0")/../.."

mode="${1:-}"
DEVICE="${DEVICE:-cpu}"
BACKEND="${BACKEND:-}"
REQUIRE="${REQUIRE:-}"
OUT="${OUT:-build/ripple-gpu}"

die() { echo "ripple-gpu: $*" >&2; exit 2; }

# label | cargo arguments before `--` | test-name filters after it (empty: the whole target)
SUITES=(
    "ricursive|-p brain-ricursive --test it|"
    "dflash2|-p brain-dflash2 --test it|"
    "catalog|-p brain-catalog --test it|ricursive"
    "sdk|-p brain --test it|ricursive"
    "gradcheck|-p brain-gradcheck --test it|ricursive"
    "model-shared|-p brain-model --lib|lsa:: paged_sparse kv_tier head_ce adapter:: shard::"
    "qwen3-lsa|-p brain-qwen3 --test it|lsa serve_conformance"
    "qwen35-dflash2|-p brain-qwen35 --test it|dflash2"
)

describe() {
    case "$1" in
        ricursive) echo "the engine, tiers, KV tiers, stages, LSA, fused attention, training, bank, DFlash2 taps, prefix branches" ;;
        dflash2) echo "the draft model: dynamic convolution over several blocks, compact paged caches" ;;
        catalog) echo "Ripple served through the catalog: chat, tools, adapters, speculation" ;;
        sdk) echo "the SDK pipelines and fine-tuners over Ripple" ;;
        gradcheck) echo "finite-difference gradients of every Ripple parameter" ;;
        model-shared) echo "shared code the port changed: LSA, paged sparse geometry, KV tiers and fused flash, training head, adapter bank, shard pipeline" ;;
        qwen3-lsa) echo "regression: Qwen3's engine on the shared LSA binding" ;;
        qwen35-dflash2) echo "regression: the Qwen3.8 draft through the shared dflash2 crate" ;;
    esac
}

# DEVICE and BACKEND checked here, before a build, so a typo never runs on the wrong thing.
validate() {
    local d
    case "$DEVICE" in
        cpu | software) [ -z "$BACKEND" ] || die "BACKEND=$BACKEND with DEVICE=$DEVICE: the cpu and software devices have no backend choice (cuda and vulkan need a card, DEVICE=gpu0)" ;;
        *)
            for d in ${DEVICE//,/ }; do
                [[ "$d" =~ ^gpu[0-9]+$ ]] || die "DEVICE=$DEVICE: expected cpu, software, or gpu<N>[,gpu<M>...]"
            done
            [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ -n "$CUDA_VISIBLE_DEVICES" ] || die "DEVICE=$DEVICE: this shell hides the cards (CUDA_VISIBLE_DEVICES is empty; ~/tmp/no-gpu-env.sh). Take a free card first (step 0 of the Ripple GPU validation ledger) in a shell that has not sourced it"
            BACKEND="${BACKEND:-cuda}"
            case "$BACKEND" in cuda | vulkan) ;; *) die "BACKEND=$BACKEND: expected cuda or vulkan" ;; esac ;;
    esac
    case "$REQUIRE" in "" | 0 | 1) ;; *) die "REQUIRE=$REQUIRE: expected 1 or empty" ;; esac
    if [ "$DEVICE" = software ] && [ -n "${BRAIN_LAVAPIPE_ENV:-}" ]; then
        # shellcheck disable=SC1090
        source "$BRAIN_LAVAPIPE_ENV"
    fi
}

run_cargo() {
    mkdir -p "$OUT"
    local profile="${PROFILE:-fast}" failed=() entry label args filters rc log
    export BRAIN_DEVICE="$DEVICE"
    [ -z "$BACKEND" ] || export BRAIN_BACKEND="$BACKEND"
    [ "$REQUIRE" != 1 ] || export BRAIN_REQUIRE_FIXTURES=1
    echo "ripple-gpu: cargo suites on --device $DEVICE${BACKEND:+ backend $BACKEND}, logs in $OUT/"
    for entry in "${SUITES[@]}"; do
        IFS='|' read -r label args filters <<<"$entry"
        log="$OUT/$label.log"
        echo "=== $label: $(describe "$label")"
        # shellcheck disable=SC2086
        cargo test --profile "$profile" --offline $args -- --test-threads=1 --nocapture $filters >"$log" 2>&1
        rc=$?
        grep -E "^test result" "$log" | sed 's/^/    /'
        # --nocapture above: a test that skipped by name prints its reason, a passing test would hide it.
        echo "    skipped by name: $(grep -c '^SKIP' "$log")$(grep '^SKIP' "$log" | sort -u | head -3 | sed 's/^/\n      /')"
        if [ "$rc" -ne 0 ]; then
            echo "FAILED ($rc): $label, see $log"
            grep -E "^test .* FAILED|panicked at" "$log" | head -10
            failed+=("$label")
        fi
    done
    if [ "${#failed[@]}" -gt 0 ]; then
        echo "ripple-gpu: failed suites: ${failed[*]}"
        exit 1
    fi
    echo "ripple-gpu: every suite passed on $DEVICE (a test skipped by name did not run: REQUIRE=1 makes it a failure; the reasons are in the logs)"
}

run_e2e() {
    command -v bats >/dev/null || die "bats is not installed"
    # `cpu` is the file's own default (cpu cases on the CPU backend, the barrier-kernel cases on lavapipe), so the
    # variable stays unset; any other device carries every case.
    if [ "$DEVICE" != cpu ]; then
        export RIPPLE_DEVICE="$DEVICE"
        [ -z "$BACKEND" ] || export RIPPLE_BACKEND="$BACKEND"
    fi
    [ "$REQUIRE" != 1 ] || export BRAIN_REQUIRE_FIXTURES=1
    export BRAIN_BIN="${BRAIN_BIN:-./target/debug/brain}"
    echo "ripple-gpu: tests/e2e/ripple.bats on ${RIPPLE_DEVICE:-the default cpu and software split}${RIPPLE_BACKEND:+ backend $RIPPLE_BACKEND}"
    bats tests/e2e/ripple.bats
}

case "$mode" in
    suites)
        for entry in "${SUITES[@]}"; do
            IFS='|' read -r label args filters <<<"$entry"
            printf '%-16s cargo test %s -- %s\n%16s %s\n' "$label" "$args" "${filters:-(all)}" "" "$(describe "$label")"
        done ;;
    cargo) validate; run_cargo ;;
    e2e) validate; run_e2e ;;
    *) die "usage: ripple-gpu.sh suites|cargo|e2e" ;;
esac
