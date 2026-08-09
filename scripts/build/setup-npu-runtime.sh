#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# setup-npu-runtime.sh — detect the Intel NPU (if any) and make sure the
# OpenVINO runtime `brain --device npu` / `brain npu run` dlopens at run time
# (crates/npu/src/openvino/real.rs) is actually installed and can see it.
#
# Run by `make environment`. Safe on a box with no NPU: it detects that and
# exits 0 having done nothing, so `make environment` never fails on a machine
# without this hardware (most dev boxes).
#
# What "compatible version" means here, checked against Intel's own docs
# (openvinotoolkit/openvino's intel_npu plugin README, docs.openvino.ai's NPU
# device page, as of the 2026.x release line): there is no published strict
# lockstep table pinning one exact linux-npu-driver/intel_vpu version to one
# exact OpenVINO version. Instead:
#   - the in-tree `intel_vpu` kernel driver (exposed via /dev/accel/accel*)
#     needs a Linux kernel >= 6.6 for NPU inference at all;
#   - the OpenVINO NPU plugin negotiates its own compiler path at RUNTIME
#     (Compiler-In-Plugin vs the older Compiler-in-Driver fallback) based on
#     the driver version it actually finds, not a version this script can
#     usefully pre-compute -- e.g. Meteor Lake specifically falls back to
#     Compiler-in-Driver below driver v2565, transparently.
# So this script's job is the part that IS well-defined: confirm the
# hardware+kernel prerequisites, then install/upgrade to the latest OpenVINO
# (>= requirements.txt's floor) and let its own plugin handle the rest --
# and PROVE it worked by asking OpenVINO itself what devices it now sees,
# rather than asserting success from the install step alone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIP="${PIP:-python3 -m pip}"
MIN_KERNEL="6.6"

log() { echo "setup-npu-runtime: $*"; }

# ---- 1. hardware presence -------------------------------------------------
# The in-tree `intel_vpu` driver exposes each NPU as /dev/accel/accelN --
# the same dependency-free check crates/gpu-core/src/devices.rs's
# `npu_count()` uses, so this script and brain's own detection never disagree.
shopt -s nullglob
accel_nodes=(/dev/accel/accel*)
shopt -u nullglob
if [ "${#accel_nodes[@]}" -eq 0 ]; then
    log "no /dev/accel/accel* device found -- no NPU on this box, nothing to do"
    exit 0
fi
log "found ${#accel_nodes[@]} NPU device node(s): ${accel_nodes[*]}"

# ---- 2. driver binding + kernel version -----------------------------------
driver_bound="unknown"
for dev in "${accel_nodes[@]}"; do
    sys_driver="/sys/class/accel/$(basename "$dev")/device/driver"
    if [ -e "$sys_driver" ]; then
        driver_bound="$(basename "$(readlink -f "$sys_driver")")"
        break
    fi
done
log "kernel driver bound: ${driver_bound}"
if [ "$driver_bound" != "intel_vpu" ]; then
    log "WARNING: expected the 'intel_vpu' driver, found '${driver_bound}' -- continuing anyway, but the OpenVINO NPU plugin may not find a usable device"
fi

kernel_version="$(uname -r | grep -oE '^[0-9]+\.[0-9]+' || true)"
if [ -n "$kernel_version" ] && ! printf '%s\n%s\n' "$MIN_KERNEL" "$kernel_version" | sort -C -V; then
    log "WARNING: kernel ${kernel_version} is below OpenVINO's documented NPU minimum (${MIN_KERNEL}) -- NPU inference may not work even after this script installs the runtime"
else
    log "kernel $(uname -r) meets the >= ${MIN_KERNEL} NPU minimum"
fi

# ---- 3. install/upgrade the OpenVINO runtime ------------------------------
# The same `openvino` pip wheel requirements.txt already lists (this is not a
# second, competing install path -- `make requirements` covers the rest of
# the Python tooling; this step exists to (a) make sure openvino specifically
# is present/current when an NPU was just detected, even if `make
# requirements` was run before the NPU was attached/passed through, and
# (b) verify it, which `make requirements` does not.
floor="$(grep -E '^openvino' "$REPO_ROOT/requirements.txt" | head -1 || true)"
floor="${floor%%#*}"
floor="$(echo "$floor" | xargs)"
: "${floor:=openvino>=2024.0}"
log "installing/upgrading: ${floor}"
$PIP install --upgrade "$floor"

# ---- 4. verify: ask OpenVINO itself, not just the installer's exit code ---
# Mirrors crates/npu/src/openvino/real.rs's `ensure_openvino_on_path()`: the
# pip wheel's libs usually aren't on the default loader path, so a plain
# `import openvino` succeeding does not by itself prove `Core()` can see a
# device. Recreate that in Python (LD_LIBRARY_PATH set for this process only)
# so a failure here is caught NOW, in an obvious place, not on the first real
# `brain --device npu` invocation.
verify_out="$(python3 - <<'PYEOF' 2>&1 || true
import os, sys, glob
try:
    import openvino
except Exception as e:
    print(f"IMPORT_FAILED: {e}")
    sys.exit(0)
libs = os.path.join(os.path.dirname(openvino.__file__), "libs")
if os.path.isdir(libs):
    os.environ["LD_LIBRARY_PATH"] = libs + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    # openvino-sys/the C API dlopens the UNVERSIONED libopenvino_c.so; the pip
    # wheel ships only versioned files -- same gap crates/npu/src/openvino/
    # real.rs's ensure_unversioned_solinks() papers over for the Rust loader.
    for base in ("libopenvino_c", "libopenvino", "libopenvino_onnx_frontend", "libopenvino_ir_frontend"):
        target = os.path.join(libs, base + ".so")
        if not os.path.exists(target):
            versioned = sorted(glob.glob(os.path.join(libs, base + ".so.*")))
            if versioned:
                try:
                    os.symlink(os.path.basename(versioned[0]), target)
                except OSError:
                    pass
try:
    from openvino import Core
    devices = Core().available_devices
except Exception as e:
    print(f"CORE_FAILED: {e}")
    sys.exit(0)
print(f"VERSION: {openvino.__version__}")
print(f"DEVICES: {devices}")
print(f"NPU_VISIBLE: {'NPU' in devices}")
PYEOF
)"
echo "$verify_out" | sed 's/^/setup-npu-runtime: /'

if echo "$verify_out" | grep -q "NPU_VISIBLE: True"; then
    log "OK -- OpenVINO sees the NPU. brain --device npu / brain npu run should work."
    log "(no INTEL_OPENVINO_DIR/OPENVINO_INSTALL_DIR needed: crates/npu/src/openvino/real.rs auto-discovers this same pip install at runtime.)"
elif echo "$verify_out" | grep -q "^VERSION:"; then
    log "OpenVINO installed and loadable, but did NOT report NPU in its device list."
    log "The device node and driver are present (checked above), so this is not a missing-package problem. Checking the next most likely cause:"
    if [ ! -d /lib/firmware/intel/vpu ] && [ ! -d /usr/lib/firmware/intel/vpu ]; then
        log "  /lib/firmware/intel/vpu is ABSENT in THIS filesystem view."
        # request_firmware() is served by the kernel via
        # kernel_read_file_from_path_initns() -- by design it ALWAYS reads
        # from the host's initial mount namespace, never the namespace of
        # whatever container/process triggered the device probe (deliberate
        # hardening: a container must not be able to feed the host kernel
        # arbitrary "firmware" bytes). So inside a container, this directory
        # check can only ever see what THIS container's filesystem exposes at
        # that path -- if /lib/firmware isn't bind-mounted in from the host
        # (the common case; it usually isn't), "ABSENT here" says nothing
        # about whether the host itself has it. Don't let an "absent" result
        # from inside a container be read as proof the host is missing it.
        if [ -f /.dockerenv ] || grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
            log "  This looks like a container. This check is inconclusive here -- verify"
            log "  directly ON THE HOST instead (outside any container):"
            log "    ls -la /lib/firmware/intel/vpu/"
            log "    sudo modprobe -r intel_vpu && sudo modprobe intel_vpu && sudo dmesg | tail -40 | grep -iE 'vpu|firmware'"
            log "  If the host copy is present and correctly named, the kernel finds it regardless"
            log "  of any container -- a container cannot make already-correct host firmware invisible."
        fi
        log "  The intel_vpu kernel driver can bind and create /dev/accel/accelN without its firmware blob,"
        log "  but the NPU has no usable compute until the HOST kernel loads it via request_firmware()."
        log "  Firmware loading happens in the HOST kernel's namespace, not this container's -- installing"
        log "  linux-firmware INSIDE a container that doesn't own /lib/firmware will not fix this from here."
        log "  This needs to be resolved on the HOST (or wherever this container's kernel firmware search"
        log "  path actually resolves): install linux-firmware (or the Intel NPU firmware package for this"
        log "  kernel) there, then re-run this script."
    else
        log "  Firmware IS present, so this is likely a permissions issue -- see:"
        log "  https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html"
    fi
else
    log "OpenVINO did not load correctly even after the install step -- see the output above."
    exit 1
fi
