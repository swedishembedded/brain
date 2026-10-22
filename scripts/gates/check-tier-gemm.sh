#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements fast GEMV and tensor-core GEMM kernels for every quantized
# weight format for its clients. If your team needs expertise in bandwidth-bound decode and
# batched prefill over stored low-precision weights then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# Tier coverage (`make check/scripts`): every weight storage format has a decode GEMV and a
# batched tensor-core path, or a recorded gap. scripts/gates/tier-gemm.map declares, per
# kernels::template::DecodeFormat (plus the dense F32 tier), the GEMV kernel and how a batched
# product reaches tensor cores; this gate prints that table and fails when
#   - a DecodeFormat has no row, or a row names no DecodeFormat;
#   - a row names a kernel file that does not exist;
#   - a gap (no gemv, no tensor core, or a scratch-tile library path for a non-dense format) is
#     missing from scripts/gates/tier-gemm.allow, or an allowed gap has been closed (shrink-only).
# Behaviour is verified by crates/model/tests/qops_tier_coverage.rs.
#
# Usage: scripts/gates/check-tier-gemm.sh [--root DIR] [--map FILE] [--allow FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
map="$here/tier-gemm.map"
allow="$here/tier-gemm.allow"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --map) map="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  *) echo "check-tier-gemm: unknown argument $1" >&2; exit 2 ;;
  esac
done
exec python3 - "$root" "$map" "$allow" <<'PY'
import os, re, sys

root, map_path, allow_path = sys.argv[1:4]
DENSE = {"F32", "Bf16"}

def rows(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.strip() and not line.startswith("#"):
            out.append(line.split("\t"))
    return out

src = open(os.path.join(root, "crates/kernels/src/template.rs"), encoding="utf-8").read()
enum = re.search(r"pub enum DecodeFormat \{(.*?)\n\}", src, re.S)
formats = {"F32"} | set(re.findall(r"^\s*([A-Z][A-Za-z0-9]*)\s*=\s*\d+,", enum.group(1), re.M)) if enum else None
errors = []
if formats is None:
    errors.append("crates/kernels/src/template.rs: enum DecodeFormat not found")
    formats = set()

table = {}
for r in rows(map_path):
    if len(r) != 3:
        errors.append(f"{map_path}: malformed row {r!r} (want format<TAB>gemv<TAB>tensor_core)")
        continue
    table[r[0]] = (r[1], r[2])
for f in sorted(formats - table.keys()):
    errors.append(f"{f}: weight format has no row in tier-gemm.map")
for f in sorted(table.keys() - formats):
    errors.append(f"{f}: tier-gemm.map row for a format that DecodeFormat does not define")

def kernel_exists(name):
    return any(os.path.exists(os.path.join(root, d, name + ext)) for d, ext in (("crates/kernels/wgsl", ".wgsl"), ("crates/kernels-cuda/cu", ".cu")))

gaps = {}
for f, (gemv, tc) in table.items():
    mine = []
    if gemv == "-":
        mine.append("gemv")
    elif not kernel_exists(gemv):
        errors.append(f"{f}: gemv kernel '{gemv}' does not exist")
    if tc == "none":
        mine.append("tensor_core")
    elif tc.startswith("native:"):
        if not kernel_exists(tc[7:]):
            errors.append(f"{f}: tensor-core kernel '{tc[7:]}' does not exist")
    elif tc == "library":
        if f not in DENSE:
            mine.append("fused")
    else:
        errors.append(f"{f}: tensor_core '{tc}' is not native:<kernel>, library or none")
    gaps[f] = mine

allowed = {}
for r in rows(allow_path):
    if len(r) != 3 or not r[2].strip():
        errors.append(f"{allow_path}: malformed row {r!r} (want format<TAB>gap<TAB>reason)")
        continue
    allowed[(r[0], r[1])] = r[2]

present = {(f, g) for f, gs in gaps.items() for g in gs}
for f, g in sorted(present - allowed.keys()):
    errors.append(f"NEW GAP {f} {g}: add the kernel, or (never for a new format) record it in tier-gemm.allow")
for f, g in sorted(allowed.keys() - present):
    errors.append(f"STALE {f} {g}: the gap is closed, delete it from tier-gemm.allow")

print(f"{'format':<11} {'gemv':<20} tensor_core")
for f in sorted(table):
    gemv, tc = table[f]
    print(f"{f:<11} {gemv:<20} {tc}")
print(f"{len(present)} allowed gaps across {len(table)} formats")
if errors:
    print("check-tier-gemm: FAIL", file=sys.stderr)
    for e in errors:
        print("  " + e, file=sys.stderr)
    sys.exit(1)
print("check-tier-gemm: ok")
PY
