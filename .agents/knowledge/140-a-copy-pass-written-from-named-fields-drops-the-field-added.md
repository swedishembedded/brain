<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 140. A copy pass written from named fields drops the field added after it

`Splats` is struct-of-arrays, and `sh_rest` - the higher-order spherical
harmonics - is an `Option` beside the arrays rather than one of them. Every
pass that rebuilds the scene gaussian by gaussian was written against the
arrays it knew about, so density control emitted a scene with `sh_rest: None`.
A fit with `sh_degree` above 0 and `densify_every` set therefore threw away
every harmonic at the first density-control round and carried on from flat
colour, having discarded the only thing in the model that can represent a
surface which looks different from different sides.

Nothing failed. The fit went on converging, the PLY wrote, the loss went down,
and the scene was simply worse from every angle it was not fitted at - the
exact failure mode view-dependent colour exists to prevent.

THE RULE. Adding a field to a struct that is rebuilt element-wise is not done
when the field round-trips through IO. It is done when every pass that
reconstructs the struct carries it, and when something CHECKS that - the check
that works is to give each element a value only it could have and assert that
the value is still on the right element afterwards, which catches carrying the
field but pairing it with the wrong element as well as dropping it. A drop
that is deliberate, like `splat::prune` merging DC only, says so where it
happens; the dangerous one is the pass that never mentions the field at all.
