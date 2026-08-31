#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# npu-diagnose.sh — run every NPU check this repo has separately discovered
# and rediscovered, in one pass, and give a single yes/no/inconclusive verdict
# for "is the Intel NPU actually accessible right now".
#
# Unlike scripts/build/setup-npu-runtime.sh (which INSTALLS/upgrades the
# OpenVINO runtime and is meant to be run once per environment), this script
# is read-only diagnosis you can re-run any time something NPU-related looks
# broken, with two exceptions that are safe, additive, and idempotent (see
# "self-heal" below): it does not touch any system directory, install
# packages, or reload/reset the driver.
#
# Layers checked, cheapest/safest first:
#   1. hardware/kernel   -- /dev/accel/accel*, driver binding, kernel version
#   2. firmware          -- host-only signal (container-aware, see below)
#   3. driver health     -- kernel debugfs reset/fault counters (host+root only)
#   4. userspace libs    -- the intel-level-zero-npu package + the
#                            libze_intel_vpu.so.1 compat symlink a stock
#                            libze1 loader needs to find it (see docs/models/
#                            yolov8/npu.md and ~/.claude memory
#                            brain-npu-container-blocked.md for why)
#   5. OpenVINO install  -- the pip wheel's unversioned-symlink gap that
#                            crates/npu/src/openvino/real.rs papers over at
#                            runtime; self-healed here the same way (venv-only)
#   6. functional check  -- openvino.Core().available_devices(), run TWICE to
#                            catch the exact flakiness this repo has hit
#                            before (a clean device-list one run, a SIGSEGV
#                            the next) -- a single run can look fine and still
#                            be sitting on a wedged driver
#   7. brain itself       -- `cargo test -p brain-npu --test npu_live`, the
#                            same crates/npu code path `brain --device npu`
#                            uses, best-effort (skips cleanly without cargo)
#
# Every layer prints one of: OK / WARN / FAIL / SKIP / INCONCLUSIVE. The
# final verdict line is what to grep for in a script:
#   NPU_DIAGNOSE_VERDICT: accessible
#   NPU_DIAGNOSE_VERDICT: not-accessible
#   NPU_DIAGNOSE_VERDICT: inconclusive
#   NPU_DIAGNOSE_VERDICT: no-hardware
#
# Exit codes: 0 = accessible or no-hardware (nothing to fix), 1 = not
# accessible, 2 = inconclusive (need root/host access this run didn't have).
#
# Usage: scripts/build/npu-diagnose.sh [--verbose]
set -uo pipefail
cd "$(dirname "$0")/../.."

VERBOSE=0
[ "${1:-}" = "--verbose" ] && VERBOSE=1

PASS=0
FAIL=0
WARN=0
SKIP=0

ok()   { PASS=$((PASS + 1));  printf '  [ OK ] %s\n' "$*"; }
warn() { WARN=$((WARN + 1));  printf '  [WARN] %s\n' "$*"; }
bad()  { FAIL=$((FAIL + 1));  printf '  [FAIL] %s\n' "$*"; }
skip() { SKIP=$((SKIP + 1));  printf '  [SKIP] %s\n' "$*"; }
sect() { printf '\n=== %s ===\n' "$*"; }
verbose() { [ "$VERBOSE" -eq 1 ] && printf '         %s\n' "$*"; }

IN_CONTAINER=0
if [ -f /.dockerenv ] || grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
  IN_CONTAINER=1
fi

# ---- 1. hardware/kernel ----------------------------------------------------
sect "1/7 hardware + kernel"
shopt -s nullglob
accel_nodes=(/dev/accel/accel*)
shopt -u nullglob
if [ "${#accel_nodes[@]}" -eq 0 ]; then
  warn "no /dev/accel/accel* node -- no NPU on this box (or not passed through to this container)"
  echo
  echo "NPU_DIAGNOSE_VERDICT: no-hardware"
  exit 0
fi
ok "found ${#accel_nodes[@]} device node(s): ${accel_nodes[*]}"

driver_bound="unknown"
accel_name="$(basename "${accel_nodes[0]}")"
sys_driver="/sys/class/accel/${accel_name}/device/driver"
if [ -e "$sys_driver" ]; then
  driver_bound="$(basename "$(readlink -f "$sys_driver")")"
fi
if [ "$driver_bound" = "intel_vpu" ]; then
  ok "kernel driver bound: intel_vpu"
else
  bad "kernel driver bound: '${driver_bound}' (expected intel_vpu)"
fi

kernel_version="$(uname -r | grep -oE '^[0-9]+\.[0-9]+' || true)"
if [ -n "$kernel_version" ] && printf '%s\n%s\n' "6.6" "$kernel_version" | sort -C -V; then
  ok "kernel $(uname -r) meets the >= 6.6 NPU minimum"
else
  bad "kernel ${kernel_version:-unknown} is below the 6.6 NPU minimum"
fi

# ---- 2. firmware ------------------------------------------------------------
sect "2/7 firmware"
# request_firmware() is served by kernel_read_file_from_path_initns() -- by
# design it ALWAYS reads the HOST's initial mount namespace, never whatever
# container triggered the probe. So this directory check is only meaningful
# run directly on the host; inside a container it can say "absent" even when
# the host genuinely has it. Don't let this layer's result alone drive the
# verdict when IN_CONTAINER=1 -- see the debugfs layer for what actually
# proves firmware loaded.
if [ -d /lib/firmware/intel/vpu ] || [ -d /usr/lib/firmware/intel/vpu ]; then
  ok "firmware directory present in this filesystem view"
elif [ "$IN_CONTAINER" -eq 1 ]; then
  skip "firmware directory absent HERE, but this is a container -- inconclusive by design (kernel reads the HOST's copy, not this filesystem's); see layer 3 for the real signal"
else
  bad "firmware directory absent on what looks like bare metal -- install linux-firmware (or the Intel NPU firmware package), then reload intel_vpu or reboot"
fi

# ---- 3. driver health (debugfs) --------------------------------------------
sect "3/7 kernel driver health (debugfs reset/fault counters)"
DBGFS=""
for d in /sys/kernel/debug/accel/*/; do
  [ -e "$d/fw_name" ] && DBGFS="$d" && break
done
if [ -z "$DBGFS" ]; then
  if [ "$EUID" -ne 0 ]; then
    skip "debugfs not readable without root -- rerun with sudo for the most direct wedge signal (reset_pending/reset_counter/firewall_irq_counter)"
  else
    skip "no /sys/kernel/debug/accel/*/fw_name found -- debugfs not mounted, or not exposed to this container"
  fi
else
  verbose "reading $DBGFS"
  fw_name="$(cat "${DBGFS}fw_name" 2>/dev/null || echo unknown)"
  fw_version="$(cat "${DBGFS}fw_version" 2>/dev/null || echo unknown)"
  bootmode="$(cat "${DBGFS}last_bootmode" 2>/dev/null || echo unknown)"
  reset_pending="$(cat "${DBGFS}reset_pending" 2>/dev/null || echo -1)"
  reset_counter="$(cat "${DBGFS}reset_counter" 2>/dev/null || echo -1)"
  engine_resets="$(cat "${DBGFS}engine_reset_counter" 2>/dev/null || echo -1)"
  firewall_irqs="$(cat "${DBGFS}firewall_irq_counter" 2>/dev/null || echo -1)"
  ok "firmware: ${fw_name} (${fw_version}), last boot: ${bootmode}"
  if [ "$reset_pending" != "0" ]; then
    bad "reset_pending=${reset_pending} -- the driver itself has flagged the device as needing a reset. A module reload will NOT clear this reliably; reboot the host."
  else
    ok "reset_pending=0 (device not flagged for reset)"
  fi
  verbose "reset_counter=${reset_counter} engine_reset_counter=${engine_resets} firewall_irq_counter=${firewall_irqs}"
  if [ "${engine_resets}" != "0" ] && [ "${engine_resets}" != "-1" ]; then
    warn "engine_reset_counter=${engine_resets} (nonzero) -- firmware has recovered from at least one job fault since boot; expected after a crashed client, worth a clean reboot if NPU use is otherwise flaky"
  fi
  if [ "${firewall_irqs}" != "0" ] && [ "${firewall_irqs}" != "-1" ]; then
    warn "firewall_irq_counter=${firewall_irqs} (nonzero) -- MMU/access-violation interrupts logged, consistent with an abruptly-killed client leaving in-flight DMA"
  fi
fi

# ---- 4. userspace driver libs -----------------------------------------------
sect "4/7 userspace driver libraries"
if dpkg -l intel-level-zero-npu 2>/dev/null | grep -q '^ii'; then
  ok "intel-level-zero-npu installed ($(dpkg -l intel-level-zero-npu | awk '/^ii/{print $3}'))"
else
  bad "intel-level-zero-npu not installed (apt package providing libze_intel_npu.so.1)"
fi
NPU_SO="$(find /usr/lib /lib -maxdepth 2 -name 'libze_intel_npu.so.1.*' 2>/dev/null | head -1)"
VPU_COMPAT="/usr/lib/x86_64-linux-gnu/libze_intel_vpu.so.1"
if [ -n "$NPU_SO" ]; then
  ok "found driver: $NPU_SO"
else
  bad "no libze_intel_npu.so.1.* found under /usr/lib or /lib"
fi
if [ -e "$VPU_COMPAT" ]; then
  ok "compat symlink present: ${VPU_COMPAT} -> $(readlink "$VPU_COMPAT")"
else
  bad "missing compat symlink ${VPU_COMPAT}. A stock libze1 loader still probes the pre-rename 'vpu' name; without this symlink, OpenVINO's Core() can SEGFAULT during device enumeration (not a permission issue -- see brain-npu-container-blocked.md memory). Fix: ln -sf \$(basename \"$NPU_SO\") $VPU_COMPAT && ldconfig"
fi

# ---- 5. OpenVINO python install (self-heals its own venv only) ------------
sect "5/7 OpenVINO install"
if ! python3 -c "import openvino" 2>/dev/null; then
  bad "python3 -c 'import openvino' failed -- run 'make requirements' / 'make environment'"
else
  ov_version="$(python3 -c "import openvino; print(openvino.__version__)" 2>/dev/null)"
  ok "openvino importable, version ${ov_version:-unknown}"
  # Same gap crates/npu/src/openvino/real.rs's ensure_unversioned_solinks()
  # papers over for the Rust loader: the pip wheel ships only versioned
  # libs. This only ever touches files inside the venv's own libs/ dir (never
  # a system directory), mirroring what the runtime already does silently on
  # every `brain --device npu` invocation -- reproducing that here is not a
  # surprise mutation, it is the same self-heal running one layer earlier.
  python3 - <<'PYEOF'
import os, glob
import openvino
libs = os.path.join(os.path.dirname(openvino.__file__), "libs")
if os.path.isdir(libs):
    for base in ("libopenvino_c", "libopenvino", "libopenvino_onnx_frontend", "libopenvino_ir_frontend"):
        target = os.path.join(libs, base + ".so")
        if not os.path.exists(target):
            versioned = sorted(glob.glob(os.path.join(libs, base + ".so.*")))
            if versioned:
                try:
                    os.symlink(os.path.basename(versioned[0]), target)
                    print(f"  [ OK ] self-healed missing symlink: {target} -> {os.path.basename(versioned[0])}")
                except OSError as e:
                    print(f"  [WARN] could not create {target}: {e}")
PYEOF
fi

# ---- 6. functional check: run TWICE (catch flaky/wedged behavior) ---------
sect "6/7 functional check (openvino.Core().available_devices(), run twice)"
PYCHECK='
import sys
try:
    from openvino import Core
    devices = Core().available_devices
except Exception as e:
    print(f"EXC:{e}")
    sys.exit(0)
print(f"DEVICES:{devices}")
'
run_check() {
  local errfile="${TMPDIR:-/tmp}/npu-diagnose-stderr.$$"
  python3 -c "$PYCHECK" 2>"$errfile"
  local rc=$?
  cat "$errfile" >&2 2>/dev/null || true
  rm -f "$errfile"
  return $rc
}
attempt1_out="$(run_check)"; attempt1_rc=$?
attempt2_out="$(run_check)"; attempt2_rc=$?
verbose "attempt 1: rc=${attempt1_rc} ${attempt1_out}"
verbose "attempt 2: rc=${attempt2_rc} ${attempt2_out}"

npu_seen=0
if [ "$attempt1_rc" -eq 139 ] || [ "$attempt2_rc" -eq 139 ]; then
  bad "device enumeration SEGFAULTED on at least one of two consecutive runs -- this is the exact flakiness pattern seen before on a wedged NPU (clean result one run, SIGSEGV the next). Do not keep retrying: reboot the host, do not just reload the kernel module."
elif [ "$attempt1_rc" -ne 0 ] || [ "$attempt2_rc" -ne 0 ]; then
  bad "device enumeration crashed or was killed (rc1=${attempt1_rc} rc2=${attempt2_rc})"
elif [[ "$attempt1_out" == EXC:* ]] || [[ "$attempt2_out" == EXC:* ]]; then
  bad "OpenVINO raised: ${attempt1_out#EXC:}${attempt2_out#EXC:}"
else
  ok "no crash across two runs"
  echo "         attempt 1: ${attempt1_out}"
  echo "         attempt 2: ${attempt2_out}"
  if [[ "$attempt1_out" == *"'NPU'"* ]] && [[ "$attempt2_out" == *"'NPU'"* ]]; then
    ok "NPU present in available_devices() on both runs"
    npu_seen=1
  elif [ "$attempt1_out" != "$attempt2_out" ]; then
    bad "available_devices() DIFFERED between two consecutive runs (${attempt1_out} vs ${attempt2_out}) -- nondeterministic device enumeration, treat as unreliable regardless of which run happened to list NPU"
  else
    bad "NPU not in available_devices() (consistently: ${attempt1_out})"
  fi
fi

# ---- 7. brain itself (best-effort) -----------------------------------------
sect "7/7 brain's own crates/npu path (best-effort, needs cargo)"
if ! command -v cargo >/dev/null 2>&1; then
  skip "no cargo on PATH -- this layer only runs inside the devcontainer"
else
  export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
  brain_out="$(cargo test -p brain-npu --release --test npu_live -- --nocapture 2>&1)"
  brain_rc=$?
  verbose "$brain_out"
  if [ "$brain_rc" -ne 0 ]; then
    bad "cargo test -p brain-npu --test npu_live exited ${brain_rc} (likely a crash -- see --verbose output)"
  elif echo "$brain_out" | grep -q "SKIP: NPU unavailable"; then
    reason="$(echo "$brain_out" | grep -o 'SKIP: NPU unavailable.*' | head -1)"
    warn "brain's own test reports: ${reason}"
  elif echo "$brain_out" | grep -q "test result: ok"; then
    ok "cargo test -p brain-npu --test npu_live passed (brain --device npu should work)"
  else
    warn "unexpected output -- rerun with --verbose"
  fi
fi

# ---- verdict ----------------------------------------------------------------
sect "summary"
echo "  OK=${PASS} WARN=${WARN} FAIL=${FAIL} SKIP=${SKIP}"
echo
if [ "$npu_seen" -eq 1 ] && [ "$FAIL" -eq 0 ]; then
  echo "NPU_DIAGNOSE_VERDICT: accessible"
  exit 0
elif [ "$FAIL" -gt 0 ]; then
  echo "NPU_DIAGNOSE_VERDICT: not-accessible"
  exit 1
else
  echo "NPU_DIAGNOSE_VERDICT: inconclusive (rerun with sudo and/or directly on the host for the layers this run skipped)"
  exit 2
fi
