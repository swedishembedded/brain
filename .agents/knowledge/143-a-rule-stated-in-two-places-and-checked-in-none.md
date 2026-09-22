<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 143. A rule stated in two places and checked in none

`crates/kernels/src/lib.rs` and `AGENTS.md` both say it: single bind group,
**<=8 storage buffers per kernel**. It is the WebGPU minimum limit, it is what
keeps the engine portable to old cards and to a browser tab, and the splat
backward kernels sit exactly on it.

Nothing checked it. `kernels-table/check` cross-checks `@cpu` against the
barrier count and `@gpu` against `@workgroup_size`, and has no opinion about
bindings at all - so a kernel could grow a ninth one and pass every gate in
the tree.

`lif_step` did, while gaining per-neuron physiology and short-term depression.
The failure is not a warning or a slow path: wgpu refuses to create the
pipeline outright -

    In Device::create_compute_pipeline, label = 'lif_step'
      Unable to derive an implicit layout
        Too many bindings of type StorageBuffers, limit is 8, count was 9

- so every consumer loses its GPU path, and the panic names a wgpu internal
rather than the kernel's own budget. It surfaced as a sample failing to start,
several layers from the cause.

HOW IT WAS FOUND, which is the part worth copying: not by reading the kernel,
but by trying to RUN something that had never been run in this tree before.
The sample exercising it needed an optional native dependency and some
assets, so nobody had started it since the kernel changed. A rule with no
gate is only as good as the last time someone happened to exercise it.

`crates/kernels/tests/binding_budget.rs` now counts, over `kernels::ALL`, and
names the kernel and its count instead of letting wgpu name itself. Across 489
kernels it flags exactly one, so the invariant genuinely held everywhere else -
the gate is precise, not a net that needs exceptions.

Two assertions, not one: a COUNT past the budget, and an INDEX past it. The
second catches a kernel that declares `@binding(9)` while leaving a lower
index unused, which a count alone cannot see.

THE RULE. When you write an invariant into a doc comment, write the test in
the same change. "Documented in two places, enforced in none" is the state
every one of these lessons starts from.
