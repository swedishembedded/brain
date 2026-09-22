<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 132. The slow kernel is usually one nobody selected, not one nobody wrote

A decision training step cost 1187 ms. Profiled per stage (§F.2c: wall minus
device, before ranking any kernel), **97.3% of it was AdamW** - 1163 ms against
30 ms for the encoder's forward and backward passes combined.

Nothing in AdamW was slow. `optim::Optim` picks its gradient-norm reduction by
probing the device for `gradnorm_part`/`clip_coef_wg` BY NAME, and silently
runs `gradnorm_sq` when they are absent. That reference kernel returns from
every invocation except `gid == 0`, which then loops serially over the tensor.
The model's largest tensor is a `30522 x 384` embedding table, so one thread
walked 11.7M floats every step. Registering the two kernels - two lines in the
model's PIPELINES list - took the step to 37 ms.

The same pass then gave up three more, all of the same kind:

* `attn_bwd_dscores_cross` -> `attn_bwd_dscores_cross_rows`. The cooperative
  twin had been written, benchmarked and left unadopted by every caller in the
  tree. 4.24 ms -> 0.67 ms.
* `emb_bwd` spends an invocation per (vocab row, channel) whichever rows were
  looked up. A call cannot touch more than `n_rows` of them. 5.65 ms -> off
  the table.
* `Gpu::enable_step_cache` was armed on a handle that was then MOVED into one
  half of the model, so `Gpu::share`'s independent memo left the other half -
  the one with 96% of the dispatches - uncached. ~10 ms of re-recording.

24 ms/step end to end, **49x**, and not one line of arithmetic changed.

THE RULE. Before optimizing a kernel, ask whether the fast one already exists
and this caller simply is not reaching it. Every by-name seam in this repo
(`LayerNormIds::resolve`, `CrossBwdIds::resolve`, `optim::coop_gradnorm`,
`EmbBwdIds::resolve`) fails OPEN: a model that never registers the fast kernel
keeps the slow one, correctly, forever, and nothing anywhere says so. That is
the right default for correctness and the worst possible one for discovery -
so a new model's kernel list is a place to look, not a place to trust.

And when a seam's whole purpose is speed, gate it. `decide/tests/
optimizer_kernels.rs` asserts the pair is registered, because the failure
produces identical numbers 30x slower and no other test can tell.
