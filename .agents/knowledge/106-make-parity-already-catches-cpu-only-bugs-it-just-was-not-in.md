<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 106. `make parity` already catches CPU-only bugs - it just was not in `make test/full`, so nobody ran it

Two real, unrelated CPU-backend bugs surfaced in one sitting: #105 above (a
model pairing two kernels that do not share a Params contract), and
`controlnet::train::TRAIN_KERNELS` registering `"scale_chan"` twice in one
kernel table - once for `conditioning_scale`, once inherited verbatim from
`vae::blocks::BWD_KERNELS`, which `Op::Gn`'s adjoint dispatches at a fixed
offset no caller may ever skip or reorder. The GPU backend does not care -
each entry gets its own pipeline regardless of name. The CPU backend's
Cranelift JIT declares one function per registered NAME in a single module,
so the second `"scale_chan"` failed `DuplicateDefinition`, and
`check_controlnet` (150 tensors) could not so much as START on
`BRAIN_DEVICE=cpu` - not a tolerance miss, a hard panic before the first FD
probe. Fixed by registering controlnet's own slot as `"scale_chan_cond"`
instead: same WGSL body, dispatched by its numeric index exactly as before
(dispatch is never by name), so nothing about what runs changes - only the
label freeing the string for `BWD_KERNELS`'s copy.

Neither bug is why this entry exists. `scripts/gates/parity-gate.sh`
(`make parity`) already runs the full `brain-gradcheck` suite - `check_
controlnet` included - under `BRAIN_DEVICE=cpu`, which is exactly the run
that hits both failures immediately: an epsilon silently read on one backend
and discarded on the other, and a kernel table that cannot even compile on
CPU. That gate existed and was correct. It was simply never part of
`make test/full`, which depended on `parity/strict` (the golden-fixture
numeric suite) and stopped there - so "run everything" never actually ran
the one check most likely to catch a backend-only defect, and both bugs sat
merged until someone happened to invoke `make parity` by hand.

**A correctness gate that exists but is not wired into the thing people
actually run before calling something done is equivalent to a gate that does
not exist.** The fix was not writing a new check - one already existed,
named the right thing, checked the right property - it was one line in
`Makefile`'s `test/full` prerequisite list. Before adding a NEW test for a
newly-found bug, check whether a gate already covers it and is merely
disconnected; wiring an existing gate in is cheaper than writing a redundant
one, and unlike a narrowly-scoped regression test it also catches whatever
the NEXT model gets wrong the same way.
