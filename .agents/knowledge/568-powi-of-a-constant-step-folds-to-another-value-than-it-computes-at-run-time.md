<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 568. `powi` of a constant step folds to another value than it computes at run time

**Lesson:** in an optimised build LLVM constant-folds `f32::powi` with a compile-time exponent to the correctly
rounded power, while a run-time exponent multiplies in f32 (`__powisf2`), rounding at every product. The two
differ in the last bit for some inputs: 0.99f32 cubed is 0.97029907 folded and 0.97029901 multiplied. AdamW's bias
corrections were `1 - beta.powi(t)`, so one binary computed a different update for the same step depending on
whether `t` was a constant where the call was inlined. `optim`'s
`parallel_update::a_large_tile_updates_on_several_threads_with_the_serial_result` (reference with literal steps
1..=3) failed in every optimised build since it was written (adab58541, also under `--release`) and passed only
under the debug profile, where neither side folds; 212k of 300k elements were one rounding apart. The parallel
update was right; the reference's bias correction moved. Fixed at the source: `optim::bias_correction` raises
beta to t by explicit square-and-multiply in f64 (explicit products fold exactly as they run) and
`Adam::bias_corrections` rounds it once, which every AdamW and every device AdamW kernel (they receive the
corrections from the host) use.

**How to apply:** no `powi`/`powf` with a step-dependent exponent in arithmetic that must repeat bit for bit;
call `optim::Adam::bias_corrections` (or `optim::bias_correction` for an f64 optimiser). A test that passes only
under one profile is a finding, not a flake: run the optimised build (`make test/one` builds `target/fast`).
