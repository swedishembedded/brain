<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 252. The GDN chunk backward recomputes four of the five per-chunk working buffers instead of saving them

What changed. `gdn_chunk_fwd_train` used to commit every chunk's `q_scaled`,
`decayed_k`, `v_prime`, `v_new` (each `dk` or `dv` wide, 6144 floats per token
at 48 heads of width 128) and `decay_scale` into `[bhc, c, d]` history buffers.
`v_prime`'s history was never read. `gdn_chunk_bwd` now recomputes the four wide
ones at the top of each reverse-sweep iteration with the forward's own kernels
and params from `state_history[ci]`, `w`, `u`, `query`, `key`, `exp_g_cs` and the
one kept history, `decay_scale_hist` (one float per token per head; the
`gdn_decay_scale_bwd*` kernels index it by chunk offset, so it stays).

Effect, one layer at 8192 tokens, same harness as finding 250: 1346 to 1250
KiB per token of peak (about 7 percent), forward launches 2021 to 1509, backward
launches up by the same amount (the total is unchanged: four recompute
dispatches per chunk against four fewer commits per chunk). Gradients are the
same to the last printed digit of a sum-of-squares fingerprint of `gated` and
the four input gradients on a multi-chunk tiny shape, and the f64 finite
difference gradcheck `gdn_chunk_bwd_gradcheck` is unchanged green.

What was not done and why. The remaining saved set is whole-tensor chunk
internals the backward reads again in phase 2 (`t_mat`, `k_beta`, `k_beta_decay`,
`v_beta`, `u`, `w`, `decay_mask`, `raw_attn0`, `raw_intra`, `intra_scores`) and
the backward's own scratch. Recomputing them means redoing the UT transform, a
`c`-step serial solve per chunk, for roughly 100 KiB per token; bf16 storage
would touch every consumer kernel. With segments (finding 250) the per-layer
peak no longer scales with the sequence, so these were left. `state_history`
(48 KiB per token at chunk 64) could be checkpointed every k chunks; it is 4
percent of the layer and was not worth a second recompute path.
