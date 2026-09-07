// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which physical card each range of the streamed DiT's 50 blocks runs on.
//!
//! Swedish Embedded AB implements multi-accelerator placement for oversized
//! diffusion transformers - spreading a model no single card can hold across
//! every card a box really has, without giving up bit-exact reproducibility.
//! If your team needs expertise in multi-GPU inference placement, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # The idle card this exists to remove
//!
//! `crate::model::H3Transformer::forward_streaming_with_taps` streams one
//! block's weights onto ONE device, runs it, overwrites them with the next
//! block's - because the resident form of this model is ~132 GB at fp32 and
//! no single card holds it. That streaming loop opened exactly one
//! `crate::block::Ctx`, so on a two-card box every block of every denoise
//! step ran on card 0 while card 1 sat at 1 MiB used for the whole
//! generation, and card 0 was the one that ran out of memory.
//!
//! Streaming and sharding are not alternatives - they compose. Each card
//! streams its OWN contiguous block range one block at a time, so the
//! block-at-a-time discipline is untouched (this is not "there is room now,
//! load everything"), and each card carries only its own share of the
//! per-step allocation churn, its own block slot pool
//! (`crate::block::load_block_streaming`), and its own endpoint weights: the
//! token refiner and input projections live on the first stage, `norm_out`
//! and the output heads on the last.
//!
//! # Placement is `model::shard`'s job, not this crate's
//!
//! The cut placement is [`model::StreamPlan`] - the same
//! `model::plan_balanced` exact DP `crates/gpt2`, `crates/ltxv`,
//! `crates/qwen35` and `crates/minimaxmusic3` already cut their resident
//! pipelines with, and the same `--device`/`BRAIN_DEVICE` schedulable-set
//! resolution every other placement decision in this workspace goes through.
//! This module supplies only the two things that are genuinely H3's: the cost
//! model below, and the open devices a stage runs on.
//!
//! # Why a split run is bit-identical, not merely close
//!
//! The only value crossing a cut is the residual stream, carried host-staged
//! (`gpu.read` on one stage, `Ctx::upload` on the next) - an f32 round trip
//! through host memory is lossless, and no sum is reassociated and no
//! accumulator partitioned by the split. Every block computes exactly the
//! same dispatches over exactly the same bytes it would have computed on one
//! card. `crate::model`'s own
//! `a_two_stage_split_is_numerically_identical_to_the_single_stage_path`
//! gate checks that rather than assuming it.

use model::{ShardCost, StreamPlan};

use crate::block::Ctx;
use crate::config::{H3TransformerConfig, MODALITY_NUM};

/// Parameters in one `MiniMaxH3TransformerBlock` - `crate::block::
/// BlockWeights`' own tensor list, counted rather than measured (the shapes
/// are `cfg`'s, and a cost model that had to open the checkpoint could not be
/// consulted before deciding which devices to open).
fn block_params(cfg: &H3TransformerConfig) -> f64 {
    let hidden = cfg.hidden_size as f64;
    let inner = cfg.inner_dim() as f64;
    let ffn = cfg.ffn_dim as f64;
    let attn = 3.0 * inner * hidden + hidden * inner + 2.0 * cfg.attention_head_dim as f64;
    let ff = 2.0 * ffn * hidden + hidden * ffn;
    // `adaln_proj.linear`: `[MODALITY_NUM*6*hidden, time_embed_dim]` + bias.
    // 260M params per block at the real config - the single largest tensor in
    // a block, and the reason `crate::precompute_adaln` exists.
    let adaln = MODALITY_NUM as f64 * 6.0 * hidden * (cfg.time_embed_dim as f64 + 1.0);
    attn + ff + 2.0 * hidden + adaln
}

/// One token-refiner block: [`block_params`]'s attention+FFN without AdaLN.
fn refiner_block_params(cfg: &H3TransformerConfig) -> f64 {
    let hidden = cfg.hidden_size as f64;
    let inner = cfg.inner_dim() as f64;
    let ffn = cfg.ffn_dim as f64;
    3.0 * inner * hidden + hidden * inner + 2.0 * cfg.attention_head_dim as f64 + 3.0 * ffn * hidden + 2.0 * hidden
}

/// The DiT's per-stage cost model, in PARAMETERS - [`ShardCost`]'s own
/// documented unit, and the right one here for the same reason it is right
/// there: what a stage costs its card is bytes, and for a streamed stage
/// those bytes are what crosses the bus every step as well as what sits on it.
///
/// The endpoint terms are not decoration. The token refiner is
/// `num_refiner_layers` full-width attention+FFN blocks (two at the real
/// config, ~700M params) that only the first stage runs, and they are a
/// significant fraction of a main block's own cost - so the first stage gives
/// up main blocks to pay for them, which is exactly the balance
/// `model::plan_balanced` is asked to find and not something a hardcoded
/// "halve the stack" rule could express.
pub fn shard_cost(cfg: &H3TransformerConfig) -> ShardCost {
    let hidden = cfg.hidden_size as f64;
    let te = cfg.time_embed_dim as f64;
    // Stage 0: the three input projections, the token refiner and its final
    // norm. (`time_embedder` is host-side and replicated - every stage
    // recomputes the timestep table from the same scalars, exactly as
    // `minimaxmusic3::dit_shard` replicates its own timestep weights.)
    let embed = hidden * (cfg.video_patch_dim() as f64 + 1.0)
        + hidden * (cfg.audio_in_channels as f64 + 1.0)
        + hidden * (cfg.text_dim as f64 + 1.0)
        + cfg.num_refiner_layers as f64 * refiner_block_params(cfg)
        + hidden;
    // The last stage: `norm_out` (norm + a `[2*hidden, time_embed_dim]`
    // shift/scale projection) and the two output heads.
    let head = hidden + 2.0 * hidden * (te + 1.0) + cfg.video_patch_dim() as f64 * hidden + cfg.video_patch_dim() as f64 + cfg.audio_in_channels as f64 * (hidden + 1.0);
    ShardCost { n_layers: cfg.num_layers as usize, per_layer: block_params(cfg), embed, head, boundary_words: 0 }
}

/// The streamed DiT's open devices - one per stage of a [`StreamPlan`].
///
/// Opened ONCE, outside the denoise loop, and reused for every step. Opening
/// a device per step (which an earlier version of this loop did) OOM'd a 24 GB
/// P40 partway through a real 16-step generation: wgpu gives no guarantee one
/// device's resources are reclaimed before the next opens, so the devices
/// accumulate instead of one device's steady state repeating. That constraint
/// is why this type exists at all rather than a plan being resolved inside
/// the forward.
pub struct StreamingDit {
    plan: StreamPlan,
    stages: Vec<Ctx>,
}

impl StreamingDit {
    /// Open one device per stage of the plan this machine's schedulable set
    /// supports - both P40s on this box, a single ambient device on a
    /// one-card or CPU-backend run.
    ///
    /// `device` is the caller's own `--device`-shaped token, resolved by
    /// [`StreamPlan::auto`] through the same grammar `--device` uses: `gpu1`
    /// names one card and is honoured as one stage, `gpu`/`vulkan` name every
    /// card behind a backend and are spread across them.
    pub fn open(cfg: &H3TransformerConfig, device: Option<&str>) -> StreamingDit {
        StreamingDit::open_with_plan(StreamPlan::auto(&shard_cost(cfg), device), device)
    }

    /// [`Self::open`] with the placement decided by the caller - what a test
    /// forcing a genuine two-stage split (on one device or two) uses.
    pub fn open_with_plan(plan: StreamPlan, device: Option<&str>) -> StreamingDit {
        let stages = (0..plan.n_stages())
            .map(|s| plan.open_stage(s, || Ctx::new(device)).unwrap_or_else(|e| panic!("minimaxh3 DiT stage {s} placement: {e}")))
            .collect();
        tracing::info!(plan = %plan.describe(), "minimaxh3 DiT block placement");
        StreamingDit { plan, stages }
    }

    /// One stage on the ambient device - the un-sharded path, byte for byte.
    pub fn open_single(cfg: &H3TransformerConfig, device: Option<&str>) -> StreamingDit {
        StreamingDit::open_with_plan(StreamPlan::single(cfg.num_layers as usize), device)
    }

    pub fn plan(&self) -> &StreamPlan {
        &self.plan
    }

    pub fn stage(&self, s: usize) -> &Ctx {
        &self.stages[s]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real config's 50 blocks really do land on both cards of a two-card
    /// box, balanced by the cost model rather than halved by fiat - and the
    /// token refiner's weight on the first stage really does move the cut.
    /// Pure host-side arithmetic on `H3TransformerConfig::real()`'s numbers:
    /// no GPU, no checkpoint, so it is meaningful on any machine.
    #[test]
    fn the_real_fifty_block_stack_splits_across_two_cards() {
        let cfg = H3TransformerConfig::real();
        let cost = shard_cost(&cfg);
        assert_eq!(cost.n_layers, 50);
        let plan = StreamPlan::balanced(&cost, &[0, 1]);
        assert!(plan.is_split(), "two cards must produce two stages");
        let counts: Vec<usize> = plan.stages().iter().map(|s| s.end - s.start).collect();
        assert_eq!(counts.iter().sum::<usize>(), 50, "every block runs exactly once");
        assert!(counts.iter().all(|&c| c >= 20), "neither card may be left idle: {counts:?}");
        assert!(counts[0] < counts[1], "the token refiner rides on stage 0, which must give up blocks for it: {counts:?}");
        for l in 0..50 {
            assert!(plan.stages()[plan.stage_of(l)].owns(l));
        }
    }

    /// The cost model's own shape: one main block must dominate either
    /// endpoint's extra weights (otherwise a two-card plan would degenerate
    /// into "the endpoints, and everything else"), while the refiner-carrying
    /// embed stage must still cost meaningfully more than the head stage.
    #[test]
    fn the_cost_model_ranks_a_block_above_the_endpoint_weights() {
        let cfg = H3TransformerConfig::real();
        let cost = shard_cost(&cfg);
        assert!(cost.embed > cost.head * 10.0, "the token refiner makes the embed stage far heavier than the head stage: {} vs {}", cost.embed, cost.head);
        assert!(cost.embed < cost.per_layer * 5.0, "the endpoint weights must not dwarf the block stack: {} vs {}", cost.embed, cost.per_layer);
        // adaln_proj is the largest tensor in a block - the roadmap's
        // "260M params/block" line, recomputed here from `cfg` alone.
        let adaln = MODALITY_NUM as f64 * 6.0 * cfg.hidden_size as f64 * (cfg.time_embed_dim as f64 + 1.0);
        assert!(adaln > 0.35 * cost.per_layer, "adaln_proj should be a large share of a block: {adaln} of {}", cost.per_layer);
    }

    /// More cards than blocks (the tiny config has two) must not open a
    /// device for an empty stage.
    #[test]
    fn the_tiny_config_never_plans_an_empty_stage() {
        let cfg = H3TransformerConfig::tiny();
        let plan = StreamPlan::balanced(&shard_cost(&cfg), &[0, 1, 2, 3]);
        assert_eq!(plan.n_stages(), 2);
        for s in plan.stages() {
            assert!(s.end > s.start, "no stage may be empty: {:?}", plan.stages());
        }
    }
}
