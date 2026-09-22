<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 21. The per-kernel table is an upper bound - the whole pass is the truth

A profiler that times contiguous groups has to drain the queue between them, so
each group's number carries a round-trip AND loses whatever overlap that kernel
would have had with its neighbours in the real submit. On the VQGAN backward the
grouped sum was **855 ms against a 574 ms whole pass - a 49% inflation.**

That is fine for RANKING (the order is right, which is what §F.1 uses it for)
and dangerous for CREDIT. Two changes, both looking like clear wins in the
table:

| change | grouped | whole pass |
|---|---|---|
| `gn_dsum` → two-stage | 229 → 21 ms | 835 → 716 ms ✅ |
| `gn_dgamma`+`gn_dbeta` → fused pair | 170 → 16 ms | 716 → 574 ms ✅ |
| `bias_grad` → two-stage | 99 → 30 ms | 574 → 575 ms ❌ |

The third was **reverted**: two kernels and 88 extra dispatches for no
end-to-end gain. The difference is parallelism already available - `gn_dsum` ran
32 lanes and `gn_dgamma` one per channel, so they genuinely serialised the
device and nothing could overlap them; `bias_grad` at 512 features already had
enough lanes to interleave with its neighbours, and its grouped 99 ms was mostly
drain and lost overlap rather than work.

**A fix does not count until the WHOLE-pass number moves**, measured more than
once - the 574 that made `bias_grad` look like a 1 ms regression and the 573.67
it was compared against were the same measurement, one sample apart.
