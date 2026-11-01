#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements per-backend performance regression gates for
# GPU inference engines, for its clients. If your team needs expertise in
# keeping CPU, Vulkan and CUDA paths of one engine at their rooflines as the
# code changes then you can procure our services by sending an email to
# info@swedishembedded.com.

# Per-backend performance gate: the hot ops of serving (FP8 GEMV and GEMM,
# prefill and decode attention, the GDN chunk forward), timed by the focused
# test binary crates/gpu-core/tests/perf_floor.rs on each backend this machine
# has, and compared with the floor recorded for that device class in
# scripts/gates/perf-floors/<backend>-<device>.txt.
#
# A line fails when its rate is below  floor * (1 - PERF_FLOOR_TOLERANCE). The
# floors are measured on a rested card of the class and the tolerance absorbs
# the run-to-run spread of a shared one (default 0.25; the measured spread is
# in the roadmap ledger). A device class with no floor file is reported UNGATED
# (the table is still printed) so a different machine is never judged against
# this one's numbers; an op that has a floor and no line fails.
#
# A backend that is not present is skipped with a message naming it. The CUDA
# card is BRAIN_DEVICE=gpuN when set, else the one with least memory in use.
#
# Usage: scripts/gates/perf-floor-gate.sh [--record] [cpu|vulkan|cuda ...]
#   --record   write the measured rates as the floors of each device class
set -u
cd "$(dirname "$0")/../.."
source scripts/gates/cuda-device.sh

record=0
if [ "${1:-}" = "--record" ]; then record=1; shift; fi
backends=("$@")
[ "${#backends[@]}" -gt 0 ] || backends=(cpu vulkan cuda)
tolerance="${PERF_FLOOR_TOLERANCE:-0.25}"
floors_dir=scripts/gates/perf-floors
mkdir -p "$floors_dir"

fail=0
for backend in "${backends[@]}"; do
    device="${BRAIN_DEVICE:-}"
    case "$backend" in
        cpu) device=cpu ;;
        vulkan|cuda)
            if [ "$backend" = cuda ]; then
                device=$(cuda_device)
                [ -n "$device" ] || { echo "=== $backend: SKIP (no NVIDIA device on this machine) ==="; echo; continue; }
            else
                device="${BRAIN_VULKAN_DEVICE:-$(cuda_device)}"
                [ -n "$device" ] || device=gpu
            fi ;;
        *) echo "unknown backend '$backend' (cpu, vulkan or cuda)"; exit 2 ;;
    esac
    echo "=== perf floors: $backend on $device ==="
    log=$(mktemp)
    start=$SECONDS
    if ! BRAIN_DEVICE="$device" BRAIN_BACKEND="$backend" cargo test --release -q -p brain-gpu-core --test perf_floor \
        -- --ignored --nocapture --test-threads=1 >"$log" 2>&1; then
        echo "  FAIL: the timing binary failed on $backend"; sed 's/^/    /' "$log" | tail -20
        fail=1; rm -f "$log"; continue
    fi
    kind=$(awk '$1 == "PERF-DEVICE" { print $2 }' "$log")
    # The device class names the hardware: the CPU model, or the adapter the backend reported.
    if [ "$backend" = cpu ]; then
        adapter="$(lscpu | sed -n 's/^Model name: *//p' | head -1) x$(nproc)"
    else
        adapter=$(sed -n 's/^adapter: \(.*\) (.*$/\1/p' "$log" | head -1)
        [ -n "$adapter" ] || adapter=$(nvidia-smi --query-gpu=name -i "${device#gpu}" --format=csv,noheader 2>/dev/null)
    fi
    if [ "$kind" != "$backend" ]; then
        echo "  SKIP: asked for the $backend backend, got '$kind' ($adapter) - $backend is not usable here"
        echo; rm -f "$log"; continue
    fi
    class="$backend-$(echo "$adapter" | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')"
    floor_file="$floors_dir/$class.txt"
    grep '^PERF-SKIP' "$log" | sed 's/^PERF-SKIP /  not timed: /'
    echo "  device class $class, timed in $((SECONDS - start)) s"
    if [ "$record" -eq 1 ]; then
        {
            echo "# Recorded rates (higher is better) for device class $class; see scripts/gates/perf-floor-gate.sh."
            echo "# op rate unit"
            awk '$1 == "PERF" { print $2, $3, $4 }' "$log"
        } >"$floor_file"
        echo "  recorded $(grep -c '^PERF ' "$log") floors in $floor_file"
        awk '$1 == "PERF" { printf "  %-40s %10.1f %s\n", $2, $3, $4 }' "$log"
        rm -f "$log"; continue
    fi
    if [ ! -f "$floor_file" ]; then
        echo "  UNGATED: no floors recorded for $class (make test/perf/record)"
        awk '$1 == "PERF" { printf "  %-40s %10.1f %s\n", $2, $3, $4 }' "$log"
        echo; rm -f "$log"; continue
    fi
    printf '  %-40s %10s %10s %7s\n' op measured floor ratio
    if awk -v tol="$tolerance" -v logf="$log" '
        BEGIN { while ((getline line < logf) > 0) { split(line, f, " "); if (f[1] == "PERF") got[f[2]] = f[3] } bad = 0 }
        /^#/ || NF < 3 { next }
        { op = $1; floor = $2; unit = $3
          if (!(op in got)) { printf "  %-40s %10s %10.1f %7s  FAIL (no line: the op vanished)\n", op, "-", floor, "-"; bad = 1; next }
          ratio = got[op] / floor
          ok = ratio >= 1 - tol
          printf "  %-40s %10.1f %10.1f %6.2fx  %s %s\n", op, got[op], floor, ratio, unit, ok ? "ok" : "FAIL (below floor - " tol * 100 "%)"
          if (!ok) bad = 1; seen[op] = 1 }
        END { for (op in got) if (!(op in seen)) printf "  %-40s %10.1f %10s %7s  new (no floor yet)\n", op, got[op], "-", "-"; exit bad }
    ' "$floor_file"; then
        echo "  PASS: $backend"
    else
        echo "  FAIL: $backend regressed"; fail=1
    fi
    echo; rm -f "$log"
done

if [ "$fail" -eq 0 ]; then echo "PERF FLOOR GATE: PASS"; else echo "PERF FLOOR GATE: FAIL"; fi
exit "$fail"
