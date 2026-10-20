#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping large Rust inference and
# training stacks layered and buildable for its clients. If your team needs
# expertise in architectural dependency enforcement then you can procure our
# services by sending an email to info@swedishembedded.com.
"""Full layer map check (called by check-crate-layers.sh, `make check/scripts`).

Reads scripts/gates/crate-layers.map (package, layer) and the workspace's
normal dependency edges (`cargo metadata --offline --no-deps`, optional
dependencies included, dev and build dependencies excluded). Fails on
  - a workspace package with no map line, or a map line naming no package;
  - a normal dependency on a HIGHER-numbered layer, unless listed with a
    reason in scripts/gates/crate-layers.allow (from<TAB>to<TAB>reason);
  - an allowlist entry whose edge no longer exists or is no longer a
    violation (the list only shrinks);
  - the generated crate table in .agents/rules/architecture.md differing from
    the map (`--write-doc` rewrites it).
"""
import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

BEGIN = "<!-- crate-layers:begin (generated from scripts/gates/crate-layers.map; edit the map, then run scripts/gates/crate-layers.py --write-doc) -->"
END = "<!-- crate-layers:end -->"
LAYER_NAMES = {
    1: "kernels, architecture registry, device-neutral contracts",
    2: "accelerator seam: backends and the gpu-core facade",
    3: "training substrate and shared contracts",
    4: "models",
    5: "cross-cutting evaluation and tooling",
    6: "serving, front ends, SDK, samples",
}


def read_tsv(path, cols):
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < cols or not all(p.strip() for p in parts[:cols]):
            sys.exit(f"{path}:{n}: expected {cols} tab-separated fields")
        rows.append(parts)
    return rows


def render(layers):
    out = [BEGIN, "```text"]
    for n in range(1, 7):
        crates = sorted(c.removeprefix("brain-") for c, l in layers.items() if l == n)
        body = textwrap.fill(", ".join(crates), width=96, initial_indent="    ", subsequent_indent="    ")
        out.append(f"layer {n} - {LAYER_NAMES[n]}")
        out.append(body)
    out += ["```", END]
    return "\n".join(out)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", help="cargo metadata JSON (default: run cargo)")
    ap.add_argument("--map", default=str(here / "crate-layers.map"))
    ap.add_argument("--allow", default=str(here / "crate-layers.allow"))
    ap.add_argument("--arch-doc", default=str(here.parent.parent / ".agents/rules/architecture.md"))
    ap.add_argument("--write-doc", action="store_true")
    a = ap.parse_args()

    if a.metadata:
        meta = json.load(open(a.metadata))
    else:
        meta = json.loads(subprocess.run(
            ["cargo", "metadata", "--offline", "--format-version", "1", "--no-deps"],
            check=True, capture_output=True, text=True, cwd=here.parent.parent).stdout)
    pkgs = {p["name"]: p for p in meta["packages"]}
    layers = {c: int(l) for c, l in read_tsv(a.map, 2)}
    allow = {(f, t): why for f, t, why in read_tsv(a.allow, 3)} if Path(a.allow).exists() else {}

    fail = []
    for c in sorted(set(pkgs) - set(layers)):
        fail.append(f"UNASSIGNED: workspace package {c} has no line in {Path(a.map).name}; give it a layer")
    for c in sorted(set(layers) - set(pkgs)):
        fail.append(f"STALE MAP LINE: {c} is not a workspace package; delete the line")
    for c, l in layers.items():
        if not 1 <= l <= 6:
            fail.append(f"BAD LAYER: {c} is on layer {l}; layers are 1-6")

    used = set()
    for c, p in sorted(pkgs.items()):
        if c not in layers:
            continue
        for d in sorted({x["name"] for x in p["dependencies"] if x["kind"] is None and x["name"] in pkgs}):
            if d in layers and layers[d] > layers[c]:
                if (c, d) in allow:
                    used.add((c, d))
                else:
                    fail.append(f"UPWARD DEPENDENCY: {c} (layer {layers[c]}) depends on {d} (layer {layers[d]}); "
                                f"move the shared code down to layer {layers[c]} or lower, or reassign a layer in the map "
                                f"if the map is wrong")
    for e in sorted(set(allow) - used):
        fail.append(f"STALE ALLOWLIST ENTRY: {e[0]} -> {e[1]} is not an upward dependency any more; delete it")

    doc = Path(a.arch_doc)
    text = doc.read_text()
    block = render(layers)
    start = text.find("<!-- crate-layers:begin")
    if start >= 0 and END in text:
        cur = text[start: text.index(END) + len(END)]
        if cur != block:
            if a.write_doc:
                doc.write_text(text.replace(cur, block))
                print(f"crate-layers: rewrote the crate table in {doc.name}")
            else:
                fail.append(f"DOC DRIFT: the crate table in {doc.name} differs from the map; run scripts/gates/crate-layers.py --write-doc")
    else:
        fail.append(f"DOC MARKERS MISSING: {doc.name} has no crate-layers begin/end block")

    if fail:
        print("\n".join(fail))
        print(f"crate-layers: FAIL ({len(fail)} problem(s))")
        return 1
    print(f"crate-layers: OK, {len(layers)} packages on layers 1-6, {len(used)} allowlisted upward edge(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
