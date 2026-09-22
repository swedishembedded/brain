<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 31. Timing a kernel by host wall-clock around a drained slice is not a measurement

The per-kernel-kind profiler (`gpu_core::profile`, and the benches that
preceded it) times a group by submitting that slice alone and bracketing it with
`poll_wait`. That is honest about *drain* - it prints the group-sum vs
whole-pass inflation, lesson #21 - but it is not a measurement of the kernel. It
measures **launch + execute + fence**, and the floor of that is roughly constant,
so the error is inversely proportional to how small the kernel is.

Measured both ways on the same workload, host-timed slices against the
backend's **GPU timestamp queries** (`BRAIN_PROFILE=1`, which brackets each
dispatch with `beginning_of_pass_write_index`/`end_of_pass_write_index`):

| kernel | host-timed ms/call | timestamped ms/call | inflation |
|---|---:|---:|---:|
| `matmul_reg3` | 0.605 | 0.400 | 1.5× |
| `rmsnorm_rows` | 0.240 | 0.0122 | **19.7×** |
| `paged_decode_scores_batched` | 0.349 | 0.0205 | **17×** |
| `paged_decode_apply_batched` | 0.443 | 0.0152 | **29×** |

**The ranking the host method produces is wrong, not merely imprecise.** By
device time `matmul_reg3` is **94.8%** of the pass; the host-timed table put it at
53.8% and promoted `rmsnorm_rows` (1.7% of real GPU time) and `add2` to "DEFECT"
rows at 12.3% and 6.2%. Optimising either would have been work against an
artifact - the same failure §E already records twice (`bias_grad`,
`matmul_gemv`), except those were caught by the whole-pass number afterwards
rather than by the profile being right in the first place.

It also explains an "impossible rate" that three rounds of fixing the byte model
never closed. `paged_decode_scores_batched` was flagged at 250-292% of the roof;
with the true (shorter) device time the same byte model gives **6537 GB/s**, i.e.
the byte model is wrong by more than 20×, not by the 2-3× the successive
corrections were chasing. Two independent errors were being tuned against each
other, and the impossible-rate guard was the only thing that stopped a plausible
wrong answer from being published.

**The rule:** rank with **device** time. A host-bracketed slice is a legitimate
measurement of *a submit*, and the whole-pass number it produces is still the
thing a change is judged by (#21 stands) - but it must not be used to attribute
time *between* kernels.

**RESOLVED** - `gpu_core::profile` now uses device time wherever the backend can
give it. `Backend::set_kernel_timing`/`kernel_times` expose per-kernel device
totals, and `backend-wgpu` implements them with `TIMESTAMP_QUERY_INSIDE_PASSES`:
`n+1` timestamps written *between dispatches inside the production single
compute pass*, so the pass structure being measured is the one that ships. The
validation is that the two numbers now account for each other - **kernel device
time 80.28 ms against a whole pass of 80.85 ms** (0.7% apart), where the
per-dispatch-pass mode reported 330.7 ms against 114.9 ms.

It immediately inverted the ranking it was built to fix: `rmsnorm_rows` and
`add2`, previously "DEFECT" rows at 12.3% and 6.2% of the pass, are 0.8% and
0.3% at **31% and 54% of the bandwidth roof** - not defects at all.

And it exposed a much larger harness bug underneath, recorded as #32.

**Where this left the tooling before the fix.** Neither source was clean:

* the host method has the floor above;
* `BRAIN_PROFILE`'s timestamps measure real device execution, but only in a mode
  that puts **one compute pass per dispatch**, which changes the execution the
  production single-pass flush actually performs - so its *absolute* times are
  not production times either.

The fix is timestamp queries **inside the production single-pass flush** - one
query pair per dispatch, resolved once at the end - so per-kernel attribution and
the whole-pass number come from the same execution. Treat every per-kernel
*share* recorded before this fix as suspect for small kernels, and re-derive any
ranking from `BRAIN_PROFILE`'s distribution. Whole-pass numbers, and every
speedup in this document that was judged by one, are unaffected.
