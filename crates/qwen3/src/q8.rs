// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-layer linears a reduced-precision Qwen3 build re-tiers.
//!
//! The fp32 Qwen3-4B encoder is ~16 GB of weights; storing its 7 per-layer linears at a narrower tier (int8
//! group-wise symmetric, packed 4-per-`u32`, drops them to a quarter) is what lets the encoder sit on one card
//! beside a DiT. Those linears are `model::ops::Weight`s and every product over them goes through
//! `model::ops::Ops::matmul`, which picks the kernel per device; this module only names which linears they are.

/// The namespace of the re-tiered linears: [`Q8::LINEARS`] and [`Q8::is_i8_linear`].
pub struct Q8;

impl Q8 {
    /// The 7 leaf names that become int8 (everything else stays fp32).
    pub const LINEARS: [&'static str; 7] = [
        "attn.wq.weight",
        "attn.wk.weight",
        "attn.wv.weight",
        "attn.wo.weight",
        "mlp.gate.weight",
        "mlp.up.weight",
        "mlp.down.weight",
    ];

    /// Is `name` (e.g. `blocks.5.attn.wq.weight`) one of the int8 linears?
    ///
    /// The set is tier-NEUTRAL despite the int8-flavoured name: it is "the
    /// per-layer projections a reduced-precision build re-tiers", and
    /// `Qwen::new_shard_dt` uses it for **every** non-fp32 tier (int8, f16,
    /// bf16, q4). What keeps the set the same across tiers is structural, not
    /// numeric - these 7 are exactly the linears that dispatch through
    /// `model::ops::Ops::matmul`, the one dispatch seam that knows how to read
    /// a packed weight. The token embedding and the LM head are excluded
    /// because this crate gathers/GEMMs them with `embed_tile`/`linear_kernel`
    /// straight off the fp32 `ParamStore` buffer, and the norm gains because
    /// they are `[d]` vectors consumed by norm kernels, not GEMMs.
    pub fn is_i8_linear(name: &str) -> bool {
        name.strip_prefix("blocks.").is_some_and(|rest| {
            rest.split_once('.').is_some_and(|(_, leaf)| Self::LINEARS.contains(&leaf))
        })
    }
}
