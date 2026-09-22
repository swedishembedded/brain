<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 55. `push_step` past a reverse-mode tape is a silent ZERO gradient, and the SDXL UNet had 200 of them

`vae::blocks::Builder::push_step` appends a dispatch to the forward step list
and records **nothing** on the `Op` tape. Its own doc has always said what that
means: `grad::Trace::backward` walks the tape and skips any producer whose
output no consumer claimed, so a pushed step in the middle of a differentiated
chain breaks the chain and *every parameter upstream of it gets a zero gradient
with no error*.

`sdxlunet::model::Rec` emitted its entire transformer half that way - every
LayerNorm, every `nn.Linear`, the GEGLU pair, both attentions and the resnets'
timestep broadcast. Nothing was wrong with the forward, which is why 165/165
forward-parity comparisons passed and nobody noticed; the graph simply was not
differentiable, and the roadmap recorded that as "backward not done yet" rather
than as a hazard.

**The thing to carry**: a builder that offers both a recording API and an escape
hatch has a failure mode where the escape hatch is *quietly* not covered by the
gate. Grep for `push_step` (or its equivalent) before believing a model's
backward is merely missing rather than partially wrong - and prefer to give the
shared builder a real recorder for the stage instead of pushing past it, which
is what closing `check_unet` actually consisted of. No new kernel was needed:
`matmul_dx`/`matmul_dw`, `bias_grad`, `gelu_erf_bwd`, `layernorm_dx`/`_dgamma`/
`_dbeta`, `add_chan_bcast_dv`, `concat_split` and the `attn_bwd_*_cross` quartet
all already existed for the decoder LMs.

Two things train mode has to change in a forward that a reverse walk will read,
both of which run fine and are wrong:

* **Flash attention has no softmax to bind.** `flash_attn_bidir` never
  materialises `probs`, and the adjoint quartet reads exactly that. A recording
  builder must take the materialised path however cooperative the device is.
* **One softmax slab per attention SITE.** The eval graph reused a single
  `(scores, probs)` pair across all sites, which is correct when nothing reads
  them again. Under a reverse walk two sites then differentiate against each
  other's softmax - a plausible number, not a shape error.
