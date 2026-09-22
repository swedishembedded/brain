<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 63. A shared "diffusers VAE" builder is diffusers-NAMED, not diffusers-SHAPED - reusing it for a CompVis-named checkpoint of the identical topology still needs a rename pass

`vae::VaeEncoder`/`VaeDecoder` (`crates/vae/src/decoder.rs`) look like the
generic autoencoder graph builder every latent model in the tree should be
able to point at any byte-identical-topology checkpoint through - and
`vae::blocks::Builder` genuinely is that generic, parameterized by
`BlockNames` for exactly this reason (`diffusers()`, `vqgan()`, `diamond()`
already cover three unrelated naming conventions over the SAME block shapes).
But `VaeEncoder::build`/`VaeDecoder::build` themselves do not expose that
parameter: their function BODIES hard-code the diffusers path strings
(`"encoder.conv_in"`, `"encoder.down_blocks.{i}.resnets.{j}"`,
`"encoder.mid_block.attentions.0"`, ...) even though they call into the
generic `Builder` underneath. So a checkpoint whose topology is byte-for-byte
the same VAE encoder but whose KEYS are CompVis-named
(`down.{i}.block.{j}`, `mid.attn_1.{q,k,v,proj_out}`, `nin_shortcut`) - SUPIR's
`denoise_encoder`, byte-identical to the frozen SDXL VAE encoder, only the
weights differ - cannot be handed to `VaeEncoder::from_diffusers` as-is, even
though "same topology, different weights" is exactly the situation that type
exists to cover.

The fix is a plain host-side key rename (`crate::import::denoise_encoder_diffusers_names`
in `crates/supir`): walk the CompVis leaf names into their diffusers
equivalents (`nin_shortcut` -> `conv_shortcut`, `mid.block_1`/`mid.block_2` ->
`mid_block.resnets.0`/`.1`, `mid.attn_1.{q,k,v,proj_out}` ->
`mid_block.attentions.0.{to_q,to_k,to_v,to_out.0}`), plus ONE shape metadata
change: CompVis stores attention q/k/v/proj as `[C,C,1,1]` 1x1-conv weights,
diffusers stores the identical row-major bytes as a `[C,C]` linear - a
`Vec<usize>` edit, not a data transform, easy to miss as "just a reshape,
nothing to verify" and correspondingly easy to get backwards (transposed
instead of flattened) without a test pinning the output rank.

The general shape: before reusing a "generic, parameterized" builder for a
new checkpoint naming convention, check whether the parameterization the type
ADVERTISES (here, `BlockNames`) actually reaches the call site you need, or
whether a higher-level wrapper over that builder hard-coded one specific
naming convention's path strings when it was written for its first (and so
far only) caller. The generic layer being genuinely generic does not mean
every layer built on top of it inherited that property.
