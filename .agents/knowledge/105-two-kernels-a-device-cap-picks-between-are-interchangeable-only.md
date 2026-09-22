<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 105. Two kernels a DEVICE CAP picks between are interchangeable only if they read the same Params - a constant hardcoded in one of them is a cross-backend correctness bug, not a rounding difference

`block::rms_variant(g, reference, coop, rows, d)` returns the cooperative
`rmsnorm_rows` where `DeviceCaps::workgroup_reductions` is set and the
per-element `reference` where it is not. The model supplies both indices, and
there are two candidates for the reference slot:

* `rmsnorm.wgsl` - `struct Params { d_model, seq_len }`, epsilon **hardcoded
  to 1e-6**.
* `rmsnorm_eps.wgsl` - identical math, `struct Params { d_model, seq_len, eps }`.

`rmsnorm_rows.wgsl` is the eps-taking shape. Pair it with `rmsnorm` and the
call site still compiles, still dispatches, still produces finite plausible
numbers, and every unit test that checks one backend against itself still
passes - but the third Params word the caller wrote is read on one device and
discarded on the other. The normalization a model computes then depends on the
hardware it is running on.

TimesFM-3 hit this with an `f32::EPSILON` (1.19e-7) epsilon, about an order of
magnitude under the hardcoded 1e-6, on QK-norm rows only `head_dim` wide.
Pure inference `core_forward` disagreed between the Vulkan and CPU backends by
`1.76e-2` against a `1.76e-1` output scale on a 3-layer tiny model, and the
error compounded per norm, so a downstream finite-difference gradcheck's
failure GREW with layer count - which reads like a backward-pass bug and sent
the search into the backward, the AVX2 GEMM and the WGSL-to-native JIT before
anyone re-read the forward's pipeline registration.

Three things are worth carrying out of it:

**"The selection logic is correct" and "the two things it selects between are
equivalent" are separate claims.** `rms_variant`'s policy was right the whole
time: it read `DeviceCaps`, never a backend name, and picked exactly what it
was supposed to. The defect was one entry in the model's own `PIPELINES`
table, in a different crate from the selector that consumed it.

**A per-kernel cross-backend diff cannot find it.** Every kernel in that
model's pipeline list agreed between backends to floating-point rounding when
dispatched in isolation with identical inputs, because each kernel WAS
correct. What differed
was which kernel ran. The isolating experiment is to force one backend to make
the other's selection and see the divergence vanish - a kernel-by-kernel diff
answers a question that was never the question.

**The gate belongs on both indices, at the model's own epsilon.**
`block::assert_rmsnorm_variant_agrees` checks whichever variant the CURRENT
device selects, at `block::RMSNORM_EPS`, which is precisely blind to this. A
model whose epsilon is not 1e-6 owes a test that dispatches BOTH registered
indices and compares each to a host reference at the epsilon it actually
passes (`timesfm3/tests/kernels.rs`'s
`both_registered_rmsnorm_kernels_honour_the_configured_epsilon`).
