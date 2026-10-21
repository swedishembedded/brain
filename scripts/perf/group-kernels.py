#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for finding where a training step's
# device time actually goes for its clients. If your team needs expertise in
# GPU kernel profiling and throughput engineering then you can procure our
# services by sending an email to info@swedishembedded.com.
"""Group a per-kernel device-time table into the components a step is budgeted by.

Input is the `kernel table [label]: ...` block that `BRAIN_PROFILE=1` makes the
qwen35 trainer print (one row per kernel kind: name, ms, calls, share), read
from a file or stdin. The LAST table in the input is grouped unless
`--label` names another. Every kernel lands in exactly one group, so the group
times sum to the table total; a kernel no rule knows is billed to `misc`.
The check against the table's own printed total fails the run (exit 1) when
rows were lost, so a changed table format cannot go unnoticed.
"""
import argparse
import re
import sys

HEADER = re.compile(r"^kernel table \[(?P<label>[^\]]*)\]: (?P<total>[\d.]+) ms device")
ROW = re.compile(r"^\s+(?P<name>\S+)\s+(?P<ms>[\d.]+) ms\s+(?P<calls>\d+) calls\s+[\d.]+%\s*$")

# First matching rule wins; order is part of the grouping. Each rule is
# (group, exact names, name prefixes, name substrings).
RULES = [
    ("GDN chain", {"bmm_tiled", "kv_expand", "nchw_nlc", "nlc_nchw"},
     ("gdn_", "native:brain_gdn_", "l2norm_scale", "conv1d"), ("cumsum", "mask", "decay")),
    ("library GEMM", {"library_gemm"}, ("matmul_",), ()),
    ("weight dequant", set(), ("nf4_dequant", "weight_dequant"), ()),
    ("RMSNorm", set(), ("rmsnorm", "rms_norm", "add_rms"), ()),
    ("attention element-wise", set(),
     ("flash_", "attn_", "softmax_rows", "rope", "gqa_"), ()),
    ("head, optimizer, misc", set(), ("ce_", "adamw", "embedding", "lm_head"), ()),
    ("element-wise glue",
     {"scale_row", "mul", "splice_add", "concat_split", "silu_mul", "add2",
      "cast_f32_bf16", "concat2", "scale_add", "copy_cols", "unpack_bf16", "sub",
      "region_copy"},
     ("silu_", "sigmoid"), ()),
]
FALLBACK = "head, optimizer, misc"
ORDER = ["GDN chain", "library GEMM", "element-wise glue", "attention element-wise",
         "weight dequant", "RMSNorm", "head, optimizer, misc"]


def group_of(name):
    for group, exact, prefixes, parts in RULES:
        if name in exact or name.startswith(prefixes) or any(p in name for p in parts):
            return group
    return FALLBACK


def last_table(lines, label):
    """(label, printed total, [(name, ms, calls)]) of the chosen table."""
    found = current = None
    for line in lines:
        head = HEADER.match(line)
        if head:
            current = None
            if label is None or head["label"] == label:
                current = found = (head["label"], float(head["total"]), [])
            continue
        row = ROW.match(line)
        if row and current is not None:
            current[2].append((row["name"], float(row["ms"]), int(row["calls"])))
        elif not row and not line.startswith("PROBE "):
            current = None  # any other line ends the table; a PROBE line may interleave
    return found


def render(label, printed_total, rows):
    ms, calls = {}, {}
    for name, t, c in rows:
        g = group_of(name)
        ms[g] = ms.get(g, 0.0) + t
        calls[g] = calls.get(g, 0) + c
    total = sum(ms.values())
    out = [f"component table [{label}]",
           f"  {'component':<26}{'ms':>11}{'dispatches':>12}{'share':>8}"]
    for g in sorted(ms, key=lambda g: ORDER.index(g)):
        out.append(f"  {g:<26}{ms[g]:>11.1f}{calls[g]:>12}{100 * ms[g] / total:>7.1f}%")
    out.append(f"  {'total':<26}{total:>11.1f}{sum(calls.values()):>12}{100.0:>7.1f}%")
    # Each printed row is rounded to 0.01 ms and the printed total to 0.1 ms.
    if abs(total - printed_total) > 0.005 * len(rows) + 0.05:
        out.append(f"group-kernels: grouped total {total:.2f} ms differs from the table's {printed_total:.2f} ms")
        return out, False
    return out, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="-", help="probe or CLI output (default stdin)")
    ap.add_argument("--label", help="group the table with this label (default: the last table)")
    args = ap.parse_args()
    src = sys.stdin if args.log == "-" else open(args.log, encoding="utf-8")
    with src:
        table = last_table(src.read().splitlines(), args.label)
    if table is None or not table[2]:
        print("group-kernels: no kernel table in the input (was BRAIN_PROFILE=1 set?)", file=sys.stderr)
        return 1
    out, ok = render(*table)
    print("\n".join(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
