#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Kernel-selection gate (`make check/scripts`, folded into `make test/full`).
#
# `crates/qwen3/tests/no_kernel_names.rs` (still in place, unchanged -- see
# below) polices ONE crate's ONE migrated fork: qwen3's fp32-vs-int8 linear
# dispatch. The kernel catalogue (`docs/reference/kernels.md`) is
# workspace-wide, and so is the class of bug it was written to catch -- a
# model crate that hand-dispatches a kernel by name when a structurally
# faster sibling already exists in the catalogue, bypassing whatever
# `backend_api::select::Op` policy that family already has. `gn_stats` (a
# 159x regression carried for a long time before anyone noticed) is this
# repo's own worked example of what that costs.
#
# WHAT COUNTS AS "A FASTER SIBLING" -- cross-referenced against
# docs/reference/kernels.md, not guessed:
#
#   Two kernels are siblings if their names share a STEM once trailing
#   "structural variant" words are stripped from each -- the SAME six suffix
#   families this campaign's plan names (`_rows`/`_wg`/`_reg*`/`_tiled`/
#   `_part`/`_dyn`), plus two more the catalogue's OWN naming convention
#   already uses for the identical purpose: `_batched` (a batched-dispatch
#   variant of the same op -- `paged_decode_scores_batched`,
#   `decode_softmax_batched`) and `_final` (the second half of a `_part`/
#   `_final` split-reduction pair). Stripping only THESE words (never an
#   arbitrary token) is what keeps `conv2d_dw`/`conv2d_dx` (backward passes --
#   a different computation) from being mistaken for `conv2d`'s siblings just
#   because `conv2d_tiled` exists; an early, more permissive version of this
#   script made exactly that mistake and was caught by hand-checking its own
#   output against the kernel sources before this version shipped, per this
#   campaign's own "a finding is a hypothesis until checked against source"
#   rule.
#
#   Within a stem family, any member whose `@opt` is BELOW the family's max
#   is "slow" and the max-`@opt` member(s) are its faster sibling(s).
#
# WHAT COUNTS AS "OUTSIDE A SELECTION SEAM":
#
#   A slow kernel's constant name appearing as the first argument of a real
#   dispatch call (`.step(`/`.step_buf(`/`.step_sliced(`/`.dispatch(`) is a
#   SELECTION, and every such call is in scope. It is EXEMPT only if:
#
#     - the call lives in `crates/backend-api/src/select.rs` itself (the
#       policy definition, which may name every kernel it governs), or
#     - a `KernelVariant::`/`.select(Op::`/`selector.select(`/`candidates(Op::`
#       token appears within the ten lines above it (the shape every existing
#       seam consumer already has -- `model::ops::Ops::bind`, `optim::Optim::
#       coop_gradnorm`, `qwen3::serve::Engine::rms` -- a `match` arm or an
#       explicit `candidates()`/`.select()` call feeding the chosen name), or
#     - it is a row in ALLOWLIST below, each carrying its own reason.
#
#   This is a heuristic, not a parser -- it cannot see through an indirection
#   (a helper function that itself calls `.step(` with the name baked in one
#   level down is invisible to it). That is the same class of limitation
#   `check-arch-names.sh`'s own header names for its own greps; a gate this
#   cheap to run on every `test/full` is worth having even so, and widening it
#   to a real AST walk is future work, not a blocker for landing it.
#
# THE ALLOWLIST is the inventory this gate itself produced on first run,
# checked against source (not rubber-stamped) and filed as one of:
#
#   - a Phase 1 (M1.4) / Phase 5 backlog item: a real, pre-existing dispatch
#     that predates this gate and this campaign, for a family that already
#     HAS a `select::Op` (`MatMul`/`RmsNorm`/`LayerNorm`) that the crate has
#     simply never been migrated onto. Fixing 40+ call sites across a dozen
#     model crates is exactly M1.4's/Phase 5's job, not this gate's -- see
#     `.agents/roadmap/kernel-performance.md`'s M1.3 entry.
#   - a genuinely out-of-scope regime, same category as `vae::blocks`'
#     already-documented `vision::blocks::Conv` exemption (`Op::Conv2d`'s own
#     doc comment): a deliberate, narrower `Op` that was never meant to cover
#     this call site.
#   - test-harness code that constructs raw GPU steps by design and never
#     goes through a model's own op facade.
#
# A row that no longer matches any live violation makes the gate FAIL (a
# stale allow-list entry is exactly as much rot as a missing one) -- so this
# list can only ever track reality, never merely grow.
#
# Usage: scripts/gates/check-kernel-selection.sh   (exits non-zero listing
# every unallowed violation, file:line, kernel and its best sibling)
set -uo pipefail
cd "$(dirname "$0")/../.."

# kernel<TAB>file<TAB>reason -- see the three categories above. Keep sorted by
# kernel then file so a diff shows exactly what changed.
ALLOWLIST=$(cat <<'EOF'
conv2d	crates/minimaxmusic3/src/discriminator.rs	Outside Op::Conv2d's deliberately narrow scope (that Op covers only vae::blocks::Builder::conv_s, per its own doc comment) - same category as vision::blocks::Conv's already-documented exemption, not a regression.
decode_softmax	crates/glmdsa/src/model.rs	No select::Op covers this model's own paged/incremental-decode softmax yet (Op::Softmax is a different kernel family - see its doc); part of the campaign's already-identified paged-attention triad, tracked for Phase 2/Phase 5.
decode_softmax	crates/gpt2/src/model.rs	No select::Op covers this model's own paged/incremental-decode softmax yet (Op::Softmax is a different kernel family - see its doc); part of the campaign's already-identified paged-attention triad, tracked for Phase 2/Phase 5.
layernorm	crates/glmdsa/src/model.rs	Not yet migrated onto the existing Op::LayerNorm seam - Phase 1 M1.4 / Phase 5 backlog item (see this gate's M1.3 inventory in kernel-performance.md), not fixed here.
matmul	crates/deepseek2/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/fincast/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/gpt2/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/kronos/src/train.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/mimi/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/model/tests/tensor_parallel.rs	Test harness for dp/shard-parity checks constructs raw GPU steps directly by design, never through model::ops::Ops - not a served/trained model path.
matmul	crates/qwen3/src/model.rs	B7's Ops migration (no_kernel_names.rs) scoped only forward_steps/decode_steps/run_batched_steps/head_steps; lora_fwd/proj_bwd (LoRA delta + backward) were explicitly out of that scope and still hand-dispatch - backlog, not a new regression.
matmul	crates/qwen35/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/qwen35moe/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/qwen3omnimoe/src/talker.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/qwen3omnimoe/src/thinker.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/qwen3tts/src/gen.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/toyautoencoder/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/toymoe/src/train.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/toypid/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul	crates/toyseq2seq/src/model.rs	Not yet migrated onto the existing Op::MatMul/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
matmul_dw	crates/deepseek2/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/glmdsa/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/kronos/src/train.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/model/tests/tensor_parallel.rs	Test harness for dp/shard-parity checks constructs raw GPU steps directly by design, never through model::ops::Ops - not a served/trained model path.
matmul_dw	crates/qwen35/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/qwen35moe/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/toyautoencoder/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/toymoe/src/train.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/toypid/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dw	crates/toyseq2seq/src/model.rs	No select::Op covers the backward GEMMs (matmul_dw vs matmul_dw_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/deepseek2/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/glmdsa/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/kronos/src/train.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/model/tests/tensor_parallel.rs	Test harness for dp/shard-parity checks constructs raw GPU steps directly by design, never through model::ops::Ops - not a served/trained model path.
matmul_dx	crates/qwen35/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/qwen35moe/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/toyautoencoder/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/toymoe/src/train.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/toypid/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
matmul_dx	crates/toyseq2seq/src/model.rs	No select::Op covers the backward GEMMs (matmul_dx vs matmul_dx_reg) at all yet - a gap Phase 5's family table does not currently itemise; tracked via this gate's M1.3 inventory, not fixed here.
rmsnorm	crates/chronos2/src/model.rs	Not yet migrated onto the existing Op::RmsNorm/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
rmsnorm	crates/fincast/src/model.rs	Not yet migrated onto the existing Op::RmsNorm/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
rmsnorm	crates/kronos/src/nn.rs	Not yet migrated onto the existing Op::RmsNorm/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
rmsnorm	crates/toymoe/src/train.rs	Not yet migrated onto the existing Op::RmsNorm/model::ops::Ops seam - Phase 1 M1.4 / Phase 5 backlog item, not fixed here.
EOF
)

export ALLOWLIST
out=$(python3 - <<'PY'
import collections
import os
import pathlib
import re
import sys

root = pathlib.Path(".")

# ---- 1. the kernel catalogue: name -> @opt, cross-referenced not guessed ---
md = (root / "docs/reference/kernels.md").read_text()
rows = re.findall(r"^\| \[`([a-z0-9_]+)`\]\([^)]*\) \| .*? \| .*? \| (\d)/5 \|", md, re.M)
if len(rows) < 400:
    sys.exit(f"check-kernel-selection: parsed only {len(rows)} kernel rows from kernels.md - table format changed?")
opt = {name: int(o) for name, o in rows}
names = set(opt)

# Structural "variant" words this catalogue's OWN naming convention uses for
# "same op, different implementation strategy" - the six the campaign's plan
# names, plus the two the catalogue itself already needs (see the .sh header).
VARIANT_WORDS = {"rows", "wg", "reg", "reg2", "reg3", "reg4", "tiled", "tiled2", "part", "dyn", "batched", "final"}


def stem(name):
    toks = name.split("_")
    while len(toks) > 1 and toks[-1] in VARIANT_WORDS:
        toks = toks[:-1]
    return "_".join(toks)


families = collections.defaultdict(list)
for n in names:
    families[stem(n)].append(n)

# slow kernel name -> (best sibling names, its own opt, the family max opt)
slow_to_best = {}
for fam, members in families.items():
    if len(members) < 2:
        continue
    best_opt = max(opt[m] for m in members)
    best = sorted(m for m in members if opt[m] == best_opt)
    for m in members:
        if opt[m] < best_opt:
            slow_to_best[m] = (best, opt[m], best_opt)

# ---- 2. the allowlist -------------------------------------------------------
allow = collections.defaultdict(list)  # (kernel, file) -> [reasons]
for line in os.environ["ALLOWLIST"].splitlines():
    line = line.strip("\n")
    if not line.strip():
        continue
    kernel, file, reason = line.split("\t", 2)
    allow[(kernel, file)].append(reason)
allow_seen = set()

# ---- 3. scan every dispatch call site for a slow kernel's identifier -------
SEAM_MARKERS = re.compile(r"KernelVariant::|\.select\(Op::|selector\.select\(|candidates\(Op::")
# The kernel-index identifier is ALWAYS SCREAMING_SNAKE_CASE by convention
# (`const MATMUL_GEMV: usize = ...`, mirroring `kernels::MATMUL_GEMV`'s own
# name) - restricting the capture to that shape is what keeps a lowercase
# local binding of the same word (a test fixture's `matmul: usize` field,
# `self.matmul`) from being mistaken for a kernel dispatch.
DISPATCH = re.compile(r"\.(?:step|step_buf|step_sliced|dispatch)\(\s*(?:self\.)?([A-Z][A-Z0-9_]*)\b")

violations = []
for f in sorted(root.glob("crates/**/*.rs")):
    posix = f.as_posix()
    if posix == "crates/kernels/src/lib.rs" or posix == "crates/backend-api/src/select.rs":
        continue
    text = f.read_text(errors="replace")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        for m in DISPATCH.finditer(line):
            slow = m.group(1).lower()
            if slow not in slow_to_best:
                continue
            window = "\n".join(lines[max(0, idx - 10) : idx + 2])
            if SEAM_MARKERS.search(window):
                continue
            key = (slow, posix)
            if key in allow:
                allow_seen.add(key)
                continue
            violations.append((slow, opt[slow], slow_to_best[slow][0], posix, idx + 1, line.strip()))

stale = sorted(set(allow) - allow_seen)

print(f"check-kernel-selection: {len(slow_to_best)} kernel(s) have a faster sibling in the catalogue")
print(f"check-kernel-selection: {len(allow)} allow-list row(s), {len(allow_seen)} matched a real violation")

fail = False
if violations:
    fail = True
    print(f"\ncheck-kernel-selection: {len(violations)} unallowed dispatch(es) of a kernel with a faster sibling:\n")
    for slow, o, best, f, i, line in violations:
        print(f"  {f}:{i}: dispatches `{slow}` (@opt {o}/5) outside a selection seam - faster: {', '.join(best)}")
        print(f"      {line}")
    print(
        "\n  Fix by routing the choice through backend_api::select::candidates()/an "
        "existing Op, or add an allow-list row in scripts/gates/check-kernel-selection.sh "
        "with a real reason if this is a deliberate, out-of-scope exception."
    )

if stale:
    fail = True
    print(f"\ncheck-kernel-selection: {len(stale)} allow-list row(s) no longer match any real dispatch (stale - remove them):\n")
    for slow, f in stale:
        print(f"  {slow}\t{f}")

sys.exit(1 if fail else 0)
PY
)
rc=$?
echo "$out"
if [ "$rc" -ne 0 ]; then
  exit 1
fi
exit 0
