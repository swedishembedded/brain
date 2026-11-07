// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Placement: what a Ripple deployment costs on a device, layer by layer, and how many cards hold it.
//!
//! Swedish Embedded AB implements solutions for sizing very large language model deployments to the accelerators
//! a customer owns, down to the byte, before anything is loaded, for its clients. If your team needs expertise
//! in memory planning and sharding of long-context inference then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The generic machinery is `model::shard`: a byte-exact [`LayerBytes`] per layer, [`plan_fewest_devices`] ("size a
//! replica to the fewest cards that hold the model") and [`StreamPlan`], which the engine opens its stages from
//! ([`crate::RicursiveEngine::from_source_staged`]). This module is the Ripple adapter: it states what each layer
//! costs at the weight tiers and KV tier it is served at, from the same functions the engine allocates with
//! (`Weight::bytes_for`, `KvTier::plane_bytes`, [`crate::serve::ring_rows`], [`crate::lsa_device_bytes`]), so a plan and a loaded engine cannot
//! disagree on bytes.
//!
//! # Why this is not `impl Shardable`
//!
//! `model::Shardable` is the training pipeline's seam: it requires `model::Model` (a loss, a backward, a
//! gradient per parameter) and a stage constructor over a weight map. Ripple has no backward until step 6 of the
//! port, so claiming the trait would be a fiction; the serving engine needs only what the resident engines of
//! the other families use, the byte-exact [`LayerBytes`] and a [`StreamPlan`]. [`shard_cost`] gives the
//! parameter-count [`ShardCost`] the trait's planner balances by, for equal cards.
//!
//! # What is charged where
//!
//! * a layer: its linears at their tier, its norms and gains (f32), and its KV: a sliding layer's rings for
//!   every live and snapshot slot, a global layer's share of the block pool; and, on a global layer, its share of the
//!   latent-sparse-attention index when one is attached;
//! * the last stage: `lm_head` at its tier, the final norm and the logits of the widest forward;
//! * the first stage: nothing for the embedding, which is gathered on the host a row at a time;
//! * every stage, whatever it holds: the score scratch of the dense paged triad, an estimate of the activations
//!   of the widest forward, and (for tiers that decode a tile before multiplying) the tile budget.

use gpu_core::select::Dtype;
use model::ops::{TierPolicy, Weight, DEFAULT_TILE_BUDGET_BYTES};
use model::shard::{plan_fewest_devices, LayerBytes, Placement, ShardCost, StreamPlan};

use crate::config::{LayerType, RicursiveConfig};
use crate::import::layer_tensor;
use crate::serve::{ring_rows, ServeConfig};
use crate::weights::layer_linears;

/// The tiers of the official `Ripple-65B-Preview-NVFP4` release: every MLP linear NVFP4, everything else bf16
/// (what [`crate::resolve_tiers`] finds in the real files).
pub fn nvfp4_release_tiers(cfg: &RicursiveConfig) -> TierPolicy {
    (0..cfg.num_hidden_layers).flat_map(|l| ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"].map(|s| layer_tensor(l, s))).fold(TierPolicy::uniform(Dtype::BF16), |p, name| p.with(&[&name], Dtype::F4E2M1))
}

/// Device bytes of one layer's gains and norms (f32): four layer norms, the per-head q and k norms and the
/// value-residual scalar.
fn small_tensor_bytes(cfg: &RicursiveConfig) -> u64 {
    4 * (4 * cfg.hidden_size as u64 + 2 * cfg.head_dim as u64 + 1)
}

/// What a deployment costs, split so a report can show each part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Per-layer bytes (weights, KV, index share) and the head's, for the planner.
    pub bytes: LayerBytes,
    /// Bytes every stage carries whatever layers it holds.
    pub per_stage: u64,
    pub weights: u64,
    pub kv_global: u64,
    pub kv_rings: u64,
    pub lsa: u64,
}

impl Footprint {
    /// Everything on one device (a single stage): layers, head and one stage's fixed bytes.
    pub fn single_card_bytes(&self) -> u64 {
        self.bytes.total() + self.per_stage
    }
}

/// The byte-exact footprint of serving `cfg` sized by `serve` ([`ServeConfig::kv_cache`] is the KV tier) with
/// weights at `tiers` and, when latent sparse attention is attached, `lsa_bytes` of index
/// ([`crate::lsa_device_bytes`]).
pub fn footprint(cfg: &RicursiveConfig, serve: &ServeConfig, tiers: &TierPolicy, lsa_bytes: u64) -> Footprint {
    let (kv_dim, hd) = (cfg.kv_dim() as u64, cfg.head_dim as u64);
    let kv = serve.kv_cache;
    let slots = u64::from(serve.max_batch + serve.prefix_slots);
    let ring_bytes = 2 * kv.plane_bytes(slots * u64::from(ring_rows(cfg, serve.prefill_chunk)), kv_dim, hd);
    let pool_bytes = 2 * kv.plane_bytes(u64::from(serve.num_blocks) * u64::from(serve.block_size), kv_dim, hd);
    let globals = cfg.layer_types.iter().filter(|t| **t == LayerType::Full).count() as u64;

    let mut weights = 0u64;
    let per_layer: Vec<u64> = (0..cfg.num_hidden_layers)
        .map(|l| {
            let w: u64 = layer_linears(cfg).into_iter().map(|(suffix, n, k)| Weight::bytes_for(tiers.want(&layer_tensor(l, suffix)), n, k)).sum::<u64>() + small_tensor_bytes(cfg);
            weights += w;
            w + match cfg.layer_type(l) {
                LayerType::Sliding => ring_bytes,
                LayerType::Full => pool_bytes + lsa_bytes / globals.max(1),
            }
        })
        .collect();

    let rows = u64::from(serve.max_batch.max(serve.prefill_chunk));
    let head_weight = Weight::bytes_for(tiers.want("lm_head.weight"), cfg.vocab_size, cfg.hidden_size);
    weights += head_weight;
    let head = head_weight + 4 * cfg.hidden_size as u64 + 4 * rows * cfg.vocab_size as u64;

    // The dense triad's two score slabs, sized by the keys a dense read may span.
    let keys = u64::from(serve.dense_keys.unwrap_or(serve.max_seq_len).max(cfg.sliding_window as u32));
    let scores = 2 * 4 * rows * cfg.num_attention_heads as u64 * keys;
    // The widest forward's live activations: gate, up and their product, the attention streams, the residuals.
    let activations = 4 * rows * (3 * cfg.intermediate_size as u64 + 5 * cfg.q_dim() as u64 + 4 * cfg.kv_dim() as u64 + 10 * cfg.hidden_size as u64);
    let decodes_tiles = tiers.dtypes().any(|t| matches!(t, Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::NF4 | Dtype::F4E2M1));
    let tiles = if decodes_tiles { DEFAULT_TILE_BUDGET_BYTES } else { 0 };

    let n_slide = cfg.layer_types.len() as u64 - globals;
    Footprint {
        bytes: LayerBytes { per_layer, embed: 0, head },
        per_stage: scores + activations + tiles,
        weights,
        kv_global: pool_bytes * globals,
        kv_rings: ring_bytes * n_slide,
        lsa: lsa_bytes,
    }
}

/// The usable bytes of a card of `total` bytes: what `gpu_core::capacity` leaves after the driver context and
/// the allocator's overhead, the budget every other planner in brain places by.
pub fn card_budget(total: u64) -> u64 {
    total.saturating_sub(gpu_core::capacity::headroom(total))
}

/// `cards` (`(device index, usable bytes)`) with each stage's fixed bytes taken off: what the planner may fill with layers.
fn net_of_stage_overhead(fp: &Footprint, cards: &[(usize, u64)]) -> Vec<(usize, u64)> {
    cards.iter().map(|&(i, usable)| (i, usable.saturating_sub(fp.per_stage))).collect()
}

/// The placement over the fewest leading `cards` that hold the deployment, or `None` when all of them together do not.
/// `Placement::bytes` is the layers' and endpoints' bytes; a stage also carries [`Footprint::per_stage`].
pub fn place(fp: &Footprint, cards: &[(usize, u64)]) -> Option<Vec<Placement>> {
    plan_fewest_devices(&fp.bytes, &net_of_stage_overhead(fp, cards))
}

/// [`place`] as the [`StreamPlan`] the engine opens its stages from.
pub fn stream_plan(fp: &Footprint, cards: &[(usize, u64)]) -> Option<StreamPlan> {
    StreamPlan::fit(&fp.bytes, &net_of_stage_overhead(fp, cards))
}

/// The parameter-count cost model `model::shard::plan_balanced` cuts by for interchangeable cards.
pub fn shard_cost(cfg: &RicursiveConfig, rows: usize) -> ShardCost {
    let per_layer: f64 = layer_linears(cfg).into_iter().map(|(_, n, k)| (n * k) as f64).sum();
    ShardCost { n_layers: cfg.num_hidden_layers, per_layer, embed: 0.0, head: (cfg.vocab_size * cfg.hidden_size) as f64, boundary_words: rows * cfg.hidden_size }
}

/// One line per stage for a log or a runbook: `gpu0 layers 0..40 12.3 GiB + 1.0 GiB fixed`.
pub fn describe(fp: &Footprint, placed: &[Placement]) -> String {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    placed
        .iter()
        .map(|p| format!("gpu{} layers {}..{}: {:.1} GiB + {:.1} GiB fixed", p.shard.gpu_index, p.shard.start, p.shard.end, gib(p.bytes), gib(fp.per_stage)))
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use model::kv_tier::KvTier;

    use super::*;

    const GIB: u64 = 1 << 30;

    /// The published Ripple-65B-Preview configuration (`~/resources/models/ripple-65b-preview/config.json`), the
    /// fields placement reads.
    fn real() -> RicursiveConfig {
        let mut types = Vec::new();
        for i in 0..80 {
            types.push(if i % 5 == 4 { "full_attention" } else { "sliding_attention" });
        }
        let layer_types = serde_json::to_string(&types).unwrap();
        RicursiveConfig::from_json(&format!(
            r#"{{"model_type":"ricursive","vocab_size":100352,"hidden_size":5120,"intermediate_size":40960,"num_hidden_layers":80,"num_attention_heads":80,
                "num_key_value_heads":8,"head_dim":128,"sliding_window":1024,"use_valres":true,"use_gated_attention":true,"ffn_clamp_threshold":20.0,
                "embed_scale":71.55417527999327,"layer_types":{layer_types}}}"#
        ))
        .unwrap()
    }

    fn serve(tokens: u32, kv: KvTier, sequences: u32) -> ServeConfig {
        let block = 256;
        ServeConfig {
            sparse_attention: true,
            kv_cache: kv,
            dense_keys: Some(8192),
            ..ServeConfig::new(block, (tokens * sequences).div_ceil(block), tokens, sequences, 0, 512)
        }
    }

    /// Per-layer and total parameter counts of the published checkpoint, from its shapes alone.
    #[test]
    fn the_bf16_weights_are_the_published_parameter_count_at_two_bytes() {
        let cfg = real();
        let fp = footprint(&cfg, &serve(4096, KvTier::F32, 1), &TierPolicy::uniform(Dtype::BF16), 0);
        let linears_per_layer = (10240 * 5120) * 3 + (1024 * 5120) * 2 + 3 * (40960 * 5120);
        let small = 4 * (4 * 5120 + 2 * 128 + 1);
        let expected = 80 * (2 * linears_per_layer as u64 + small) + 2 * (100352u64 * 5120);
        assert_eq!(fp.weights, expected);
        // 64.8 billion weights, the embedding (gathered on the host, never resident) included.
        let params = fp.weights as f64 / 2.0 + (100352u64 * 5120) as f64;
        assert!((params / 1e9 - 64.8).abs() < 0.2, "{params}");
    }

    /// The release's own `model.safetensors.index.json` says `total_size` 57,213,636,640 bytes, embedding included.
    #[test]
    fn the_nvfp4_release_tiers_cost_what_the_published_files_weigh() {
        let cfg = real();
        let fp = footprint(&cfg, &serve(4096, KvTier::F32, 1), &nvfp4_release_tiers(&cfg), 0);
        let embedding = 2 * 100352u64 * 5120;
        // Four bytes a tensor for the per-tensor `weight_scale_2` and `input_scale` (3 MLP linears a layer, two each).
        let stored = fp.weights + embedding + 80 * 3 * 4 * 2;
        let published = 57_213_636_640u64;
        let err = (stored as f64 - published as f64).abs() / published as f64;
        eprintln!("NVFP4 release: costed {stored} bytes, published total_size {published} ({err:.2e})");
        assert!(err < 2e-3, "stored {stored} vs published {published} ({err:.2e})");
    }

    #[test]
    fn nvfp4_with_an_int8_cache_at_128k_fits_one_96_gib_card_and_bf16_needs_two() {
        let cfg = real();
        let one = [(0, card_budget(96 * GIB))];
        let two = [(0, card_budget(96 * GIB)), (1, card_budget(96 * GIB))];

        let nv = footprint(&cfg, &serve(131_072, KvTier::INT8, 1), &nvfp4_release_tiers(&cfg), 0);
        let placed = place(&nv, &one).expect("NVFP4 + int8 KV at 128k fits one card");
        assert_eq!(placed.len(), 1);
        assert!(placed[0].bytes + nv.per_stage <= one[0].1);

        let bf = footprint(&cfg, &serve(131_072, KvTier::INT8, 1), &TierPolicy::uniform(Dtype::BF16), 0);
        assert!(place(&bf, &one).is_none(), "129.6 GB of bf16 weights cannot fit one card");
        let placed = place(&bf, &two).expect("bf16 fits two cards");
        assert_eq!(placed.len(), 2);
        assert_eq!((placed[0].shard.start, placed[0].shard.end, placed[1].shard.end), (0, placed[1].shard.start, 80));
        for (p, card) in placed.iter().zip(&two) {
            assert!(p.bytes + bf.per_stage <= card.1, "stage on gpu{} needs {} of {}", p.shard.gpu_index, p.bytes + bf.per_stage, card.1);
        }
    }

    #[test]
    fn a_million_tokens_of_int8_cache_is_half_the_bf16_cache_and_needs_the_sparse_index() {
        let cfg = real();
        let bf16 = footprint(&cfg, &serve(1 << 20, KvTier::BF16, 1), &nvfp4_release_tiers(&cfg), 0);
        let int8 = footprint(&cfg, &serve(1 << 20, KvTier::INT8, 1), &nvfp4_release_tiers(&cfg), 0);
        // 16 global layers x 2 planes x 1024 elements x 2 bytes = 64 KiB a token: 64 GiB at 1M.
        assert_eq!(bf16.kv_global, 64 * GIB);
        // One byte an element and a 4-byte scale per 128: 33 GiB.
        assert_eq!(int8.kv_global, 16 * 2 * (1 << 20) * (1024 + 8 * 4));
        assert!(int8.kv_global < bf16.kv_global * 52 / 100);
        // The dense triad over a million keys would be 2 x 4 x 512 x 80 x 1M bytes of score scratch; the sparse engine caps it.
        assert!(int8.per_stage < 8 * GIB, "{}", int8.per_stage);
    }

    #[test]
    fn the_rings_cost_the_same_at_any_context() {
        let cfg = real();
        let short = footprint(&cfg, &serve(4096, KvTier::BF16, 1), &nvfp4_release_tiers(&cfg), 0);
        let long = footprint(&cfg, &serve(1 << 20, KvTier::BF16, 1), &nvfp4_release_tiers(&cfg), 0);
        assert_eq!(short.kv_rings, long.kv_rings);
        // 64 sliding layers x 2 planes x (1024 + 512 slack) rows x 1024 elements x 2 bytes.
        assert_eq!(short.kv_rings, 64 * 2 * (1024 + 512) * 1024 * 2);
    }

    #[test]
    fn an_infeasible_deployment_is_reported_not_papered_over() {
        let cfg = real();
        let fp = footprint(&cfg, &serve(1 << 20, KvTier::BF16, 4), &TierPolicy::uniform(Dtype::BF16), 0);
        assert!(place(&fp, &[(0, card_budget(96 * GIB)), (1, card_budget(96 * GIB))]).is_none());
        assert!(stream_plan(&fp, &[(0, card_budget(96 * GIB))]).is_none());
    }

    #[test]
    fn the_stream_plan_tiles_the_layers_in_card_order() {
        let cfg = real();
        let fp = footprint(&cfg, &serve(131_072, KvTier::INT8, 1), &TierPolicy::uniform(Dtype::BF16), 0);
        let plan = stream_plan(&fp, &[(0, card_budget(96 * GIB)), (1, card_budget(96 * GIB))]).expect("fits two");
        assert_eq!(plan.n_layers(), 80);
        assert_eq!(plan.n_stages(), 2);
        assert!(plan.describe().starts_with("gpu0:0.."), "{}", plan.describe());
    }

    #[test]
    fn the_parameter_cost_proxy_balances_equal_cards() {
        let cfg = real();
        let cost = shard_cost(&cfg, 1);
        assert_eq!(cost.n_layers, 80);
        let plan = StreamPlan::balanced(&cost, &[0, 1]);
        // The head stage is heavier by the lm_head, so it gets fewer layers.
        assert!(plan.stages()[1].end - plan.stages()[1].start <= plan.stages()[0].end - plan.stages()[0].start);
    }
}
