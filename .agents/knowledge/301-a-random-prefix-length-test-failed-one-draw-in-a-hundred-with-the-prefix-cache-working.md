<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 301. A random-prefix-length test failed one draw in a hundred with the prefix cache working

`serve::tests::random_shared_prefixes_stay_exact` failed once in a full CUDA lib
run and passed six times alone, which reads as a prefix-cache race. It was not.

Reproduced by looping the one test on CUDA with the drawn seed printed (125 runs,
one failure, no warm-vs-cold divergence in any of them). The failing run was
`kv_int8=true: at least one trial must have actually reused a prefix (0/48)`:
every one of the six trials had drawn `keep < 4` tokens of the shared base, and
the KV block is 4 tokens, so no whole block was ever shareable. With
`keep` uniform in `0..14` the chance that fewer than two of six trials reach a
block is about 1% per dtype. The assertion that a prefix was reused is a
property of the draw, not of the cache.

The fix keeps the prefix random but at least one block long, and the failing draw
is pinned (`shared_prefix_trials_reuse_a_block_for_the_draw_that_once_kept_none`,
seed 12198852813623603017, red before the fix with the same 0/48).

Lesson: when a randomised test asserts that an effect happened (a hit, a reuse, a
collision), bound the draw so the effect is guaranteed to be reachable, and keep the
exactness assertion separate; otherwise a legitimate draw reads as a defect.
