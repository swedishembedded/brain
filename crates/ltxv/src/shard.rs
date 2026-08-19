// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-parallel sharding for the VIDEO-ONLY DiT ([`crate::dit::LtxDit`]),
//! wiring it onto the generic `model::Shardable` seam `model::shard`'s module
//! doc describes (cut placement via `plan_balanced`, orchestration via
//! `model::Pipeline`). The audio+video `LtxAvDit` is explicitly OUT of scope
//! here (its bidirectional cross-attention couples the two streams every
//! block, a materially bigger seam than one stack of blocks - a later pass).
//!
//! # Who computes what, and why nothing but the residual crosses the wire
//!
//! `LtxDit::forward`'s op sequence has a part that runs ONCE (patchify, the
//! per-token adaLN-single table, the RoPE tables) and a part that runs PER
//! BLOCK (self-/cross-attention, the FFN). A naive per-layer cut would also
//! need to ship the adaLN table, the RoPE tables, and the raw text context
//! across every stage boundary - extra wire traffic on top of the residual
//! `model::shard`'s own doc says should be the ONLY thing that crosses a cut.
//!
//! Instead, every stage loads its OWN copy of the (small) `adaln_single.*`
//! weights and independently recomputes the adaLN table and RoPE tables from
//! the batch - the same "every stage gets the whole batch" shape
//! `model::Pipeline::forward` already uses for its own `Batch` clone, just
//! applied to a value (`adaln_single.*`) that is a REPLICATED PARAMETER
//! here instead of activation state. Only `patchify_proj.*`/
//! `keyframes_abs_pos_embedding` (embed-stage-only: they produce the first
//! stage's initial `x`) and `scale_shift_table`/`proj_out.*` (head-stage-
//! only: the final projection) are NOT replicated - see
//! `crate::dit::shard_owns_weight`'s doc for the exact partition.
//!
//! # `model::Batch` does not fit a diffusion batch - `DitBatch` does
//!
//! `LtxDit` cannot implement `Model::set_batch` meaningfully: `model::Batch`'s
//! variants carry LM tokens, seq2seq pairs, or an image splice, never a
//! per-token RoPE-bounds array or a raw cross-attention context. Per this
//! crate's own `crate::dit::DitBatch` doc, `Model::set_batch` is therefore a
//! documented no-op and the real seam is `LtxDit::load_shard_batch` (owned
//! data, survives past one call) - the exact shortcut `s3dit::train::
//! ZTrainModel` already takes for the same reason (a diffusion transformer's
//! batch does not fit the LM-shaped `Batch` enum either).
//!
//! # What is real here, and what is an explicit, tracked gap
//!
//! Real and tested (`crates/ltxv/tests/shard_parity.rs`):
//! * [`shard_cost`](Shardable::shard_cost) - a FLOP-shaped per-block cost
//!   (self-attention QKVO + its O(T²) score/weighted-sum term, text
//!   cross-attention's own QKVO, the 4x-width FFN), fed to `plan_balanced`
//!   for both the tiny test config and the real 22B config's shape (48
//!   layers, `inner_dim` 4096) - a plan can be COMPUTED for the real model
//!   even though it cannot be built or run on hardware this port has access
//!   to (see this port's own validation-policy doc).
//! * [`Shardable::new_shard`] genuinely loads only its block range's weight
//!   subset (`crate::dit::shard_weights`), not the full stack.
//! * The single-shard degenerate case (`num_shards = 1`, one stage owns
//!   every block) runs for real and is checked bit-for-bit-close against
//!   `LtxDit::forward` on identical weights/inputs - proving the block-range
//!   slicing and the embed/head branch selection are not silently wrong.
//! * A genuine two-stage split (real block-range partition, NOT both stages
//!   secretly owning everything) with the boundary handed off through
//!   [`LtxDit::write_in_res`]/[`LtxDit::read_out_res`], run sequentially on
//!   a single device, composed and checked against the same
//!   non-sharded reference.
//!
//! Explicit gaps, not silently glossed over:
//! * **No real multi-device execution proven by this test file's own
//!   coverage.** A single-GPU run cannot exercise two stages resident on two
//!   different physical devices. The two-stage test above proves the
//!   BOUNDARY HANDOFF is correct, not that two real cards agree.
//! * **No backward pass.** [`LtxDit`] has no gradient/training machinery at
//!   all (unlike `qwen3`/`gpt2`, whose `Model::backward` runs a real GPU
//!   backward graph) - `crate::grad`/`crate::modelgrad` are a SEPARATE,
//!   non-GPU, non-sharded host-math training path this pipeline seam does not
//!   build on. `Shardable::run_backward_stage`/`read_in_dres`/
//!   `write_out_dres` and `Model::backward`/`read_grad` all `unimplemented!()`
//!   loudly rather than returning a silently-wrong zero gradient.
//! * **`model::Pipeline<LtxDit>` type-checks but is not a usable end-to-end
//!   entry point.** `Pipeline::forward`/`pipelined_fwd_bwd` drive a stage
//!   purely through `Model::set_batch`, which - per this module's doc - is a
//!   no-op for `LtxDit`; there is no way to reach `load_shard_batch` through
//!   `Pipeline`'s own API. This mirrors `s3dit::train::ZTrainModel`'s own
//!   documented limitation (its `adamw_step` is a no-op too; real
//!   optimization there goes through a `Collective`, not `Pipeline`). The
//!   tests in this pass therefore hand-drive `Shardable::new_shard` /
//!   `run_forward_stage` / `read_out_res` / `write_in_res` directly instead
//!   of going through `Pipeline`.
//! * `residency::multi::MultiDeviceResidentModel` (inference-time residency
//!   placement/eviction) is untouched - a separate, later lift.

use std::collections::HashMap;

use model::{Model, ModelConfig, Shard, ShardCost, Shardable};

use crate::config::{LtxAudioDitConfig, LtxAvDitConfig, LtxDitConfig};
use crate::dit::{LtxAvDit, LtxDit};

impl ModelConfig for LtxDitConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        crate::dit::dit_tensor_manifest(self).into_iter().map(|(n, shape)| (n, shape.iter().product())).collect()
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "inner_dim": self.inner_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "cross_attention_dim": self.cross_attention_dim,
            "ff_bias": self.ff_bias,
            "cross_attention_adaln": self.cross_attention_adaln,
            "use_prompt_adaln_single": self.use_prompt_adaln_single,
            "use_keyframes_abs_pos_embedding": self.use_keyframes_abs_pos_embedding,
            "norm_eps": self.norm_eps,
            "positional_embedding_theta": self.positional_embedding_theta,
            "positional_embedding_max_pos": self.positional_embedding_max_pos,
            "timestep_scale_multiplier": self.timestep_scale_multiplier,
            "use_middle_indices_grid": self.use_middle_indices_grid,
            "apply_gated_attention": self.apply_gated_attention,
            "connector_apply_gated_attention": self.connector_apply_gated_attention,
            "connector_num_layers": self.connector_num_layers,
            "connector_num_attention_heads": self.connector_num_attention_heads,
            "connector_attention_head_dim": self.connector_attention_head_dim,
            "connector_num_learnable_registers": self.connector_num_learnable_registers,
            "connector_positional_embedding_max_pos": self.connector_positional_embedding_max_pos,
            "connector_norm_output": self.connector_norm_output,
            "caption_proj_before_connector": self.caption_proj_before_connector,
            "use_embeddings_connector": self.use_embeddings_connector,
        })
    }
    fn from_json(v: &serde_json::Value) -> LtxDitConfig {
        let u = |k: &str| v[k].as_u64().unwrap_or_else(|| panic!("LtxDitConfig::from_json: missing {k}")) as u32;
        let b = |k: &str| v[k].as_bool().unwrap_or_else(|| panic!("LtxDitConfig::from_json: missing {k}"));
        let max_pos = v["positional_embedding_max_pos"].as_array().unwrap_or_else(|| panic!("LtxDitConfig::from_json: missing positional_embedding_max_pos"));
        let connector_max_pos = v["connector_positional_embedding_max_pos"]
            .as_array()
            .unwrap_or_else(|| panic!("LtxDitConfig::from_json: missing connector_positional_embedding_max_pos"));
        LtxDitConfig {
            inner_dim: u("inner_dim"),
            num_heads: u("num_heads"),
            num_layers: u("num_layers"),
            in_channels: u("in_channels"),
            out_channels: u("out_channels"),
            cross_attention_dim: u("cross_attention_dim"),
            ff_bias: b("ff_bias"),
            cross_attention_adaln: b("cross_attention_adaln"),
            use_prompt_adaln_single: b("use_prompt_adaln_single"),
            use_keyframes_abs_pos_embedding: b("use_keyframes_abs_pos_embedding"),
            norm_eps: v["norm_eps"].as_f64().unwrap_or_else(|| panic!("LtxDitConfig::from_json: missing norm_eps")) as f32,
            positional_embedding_theta: v["positional_embedding_theta"].as_f64().unwrap_or_else(|| panic!("LtxDitConfig::from_json: missing positional_embedding_theta")),
            positional_embedding_max_pos: [max_pos[0].as_u64().unwrap() as u32, max_pos[1].as_u64().unwrap() as u32, max_pos[2].as_u64().unwrap() as u32],
            timestep_scale_multiplier: u("timestep_scale_multiplier"),
            use_middle_indices_grid: b("use_middle_indices_grid"),
            apply_gated_attention: b("apply_gated_attention"),
            connector_apply_gated_attention: b("connector_apply_gated_attention"),
            connector_num_layers: u("connector_num_layers"),
            connector_num_attention_heads: u("connector_num_attention_heads"),
            connector_attention_head_dim: u("connector_attention_head_dim"),
            connector_num_learnable_registers: u("connector_num_learnable_registers"),
            connector_positional_embedding_max_pos: [connector_max_pos[0].as_u64().unwrap() as u32],
            connector_norm_output: b("connector_norm_output"),
            caption_proj_before_connector: b("caption_proj_before_connector"),
            use_embeddings_connector: b("use_embeddings_connector"),
        }
    }
    fn vocab(&self) -> u32 {
        0 // no token vocabulary - a diffusion transformer over continuous latents
    }
    fn block_size(&self) -> u32 {
        0 // no fixed context window - sequence length is the (variable) token count
    }
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> LtxDitConfig {
        self // nothing dataset-derived to apply (unlike GPT's 4x-d_model FF default)
    }
}

impl Model for LtxDit {
    type Config = LtxDitConfig;

    fn new(cfg: LtxDitConfig, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> LtxDit {
        LtxDit::from_flat_weights(cfg, init, Shard::whole(cfg.num_layers as usize))
    }

    fn init_weights(cfg: &LtxDitConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::dit::random_tiny_weights(cfg, seed).into_iter().map(|(name, (_, data))| (name, data)).collect()
    }

    fn config(&self) -> &LtxDitConfig {
        LtxDit::config(self)
    }

    /// A documented no-op - see this module's doc ("`model::Batch` does not
    /// fit a diffusion batch"). Use [`LtxDit::load_shard_batch`].
    fn set_batch(&self, _batch: model::Batch) {}

    fn forward(&self) -> f32 {
        self.run_stage_forward().unwrap_or(0.0)
    }

    fn backward(&self) {
        unimplemented!("ltxv::LtxDit: no backward pass exists yet (forward-only pipeline-sharding slice) - see crate::shard's module doc for the tracked gap");
    }
    fn zero_grads(&self) {
        // Nothing to zero - there is no gradient accumulator (see `backward`'s doc).
    }
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {
        // No-op: with no working `backward`, there is never a real gradient to
        // apply here (mirrors `s3dit::train::ZTrainModel::adamw_step`, whose
        // own real optimizer path is a `Collective`, not this trait method).
    }
    fn poll_wait(&self) {
        // Nothing to wait for: `LtxBlock::forward` reads its result back to
        // the host synchronously before returning (see `crate::block`'s doc),
        // unlike `qwen3`'s async submit/poll dispatch.
    }

    fn param_names(&self) -> Vec<String> {
        self.weight_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.weight(name)
    }
    fn write_weight(&self, _name: &str, _data: &[f32]) {
        unimplemented!("ltxv::LtxDit: weights are immutable post-construction in this slice (no backward/optimizer path exists - see crate::shard's module doc)");
    }
    fn read_grad(&self, _name: &str) -> Vec<f32> {
        unimplemented!("ltxv::LtxDit: no gradients exist yet - see crate::shard's module doc for the tracked gap");
    }

    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None // no token-classification head - a continuous-latent diffusion transformer
    }

    fn save(&self, _path: &str) {
        // No-op: `crate::pipeline`/`crate::dfr` already have their own weight
        // construction (`load_tiny_weights`/`random_tiny_weights`); nothing in
        // this crate persists an `LtxDit` through `Model::save`.
    }
    fn config_json(&self) -> serde_json::Value {
        LtxDit::config(self).to_json()
    }
}

impl Shardable for LtxDit {
    fn shard_cost(cfg: &LtxDitConfig, b: u32, t: u32) -> ShardCost {
        let dim = cfg.inner_dim as f64;
        let tt = (b.max(1) as f64) * (t as f64);
        // Self-attention: QKVO projections (4·dim²  per token) plus the
        // O(T²) score/weighted-sum term (this DiT's attention is always the
        // full bidirectional T-token attention, unlike a KV-cache-amortized
        // causal-LM decode, so - unlike `qwen3::shard_cost`/`gpt2::
        // shard_cost`, which cost purely by parameter count - the T² term is
        // included here).
        let self_attn = 4.0 * dim * dim * tt + 2.0 * tt * tt * dim;
        // Text cross-attention's own Q/K/V/O projections (context length is
        // not part of this signature, approximated at the same per-token
        // cost as self-attention's projections - an abstract-unit proxy, not
        // a byte-exact FLOP count, matching `model::shard::ShardCost`'s own
        // "parameter count works well" guidance).
        let cross_attn = 4.0 * dim * dim * tt;
        // FFN: `ff.net.0` (dim -> 4·dim) + `ff.net.2` (4·dim -> dim).
        let ffn = 8.0 * dim * dim * tt;
        let per_layer = self_attn + cross_attn + ffn;
        // Embed stage: `patchify_proj` + the adaLN-single timestep MLP + its linear.
        let embed = dim * cfg.in_channels as f64 + 256.0 * dim + dim * dim + cfg.adaln_rows() as f64 * dim * dim;
        // Head stage: `proj_out` (`scale_shift_table` is negligible next to it).
        let head = dim * cfg.out_channels as f64;
        ShardCost { n_layers: cfg.num_layers as usize, per_layer, embed, head, boundary_words: (b.max(1) * t) as usize * cfg.inner_dim as usize }
    }

    fn new_shard(cfg: LtxDitConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> LtxDit {
        let _ = (b, t); // see `LtxDit::from_flat_weights`'s doc: construction needs only the weight subset
        LtxDit::from_flat_weights(cfg, init, shard)
    }

    fn replicated_params(&self) -> Vec<String> {
        // Every stage loads its own copy of `adaln_single.*` (this module's
        // doc: "who computes what"), so these names appear in every stage's
        // `param_names()` and would need their gradient SUMMED across
        // holders if a backward pass existed.
        Model::param_names(self).into_iter().filter(|n| n.starts_with("adaln_single.")).collect()
    }

    fn run_forward_stage(&self) -> Option<f32> {
        self.run_stage_forward()
    }
    fn run_backward_stage(&self) {
        unimplemented!("ltxv::LtxDit: no backward pass exists yet - see crate::shard's module doc for the tracked gap");
    }
    fn read_out_res(&self) -> Vec<f32> {
        LtxDit::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        LtxDit::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        unimplemented!("ltxv::LtxDit: no backward pass exists yet - see crate::shard's module doc for the tracked gap");
    }
    fn write_out_dres(&self, _data: &[f32]) {
        unimplemented!("ltxv::LtxDit: no backward pass exists yet - see crate::shard's module doc for the tracked gap");
    }
}

// ---------------------------------------------------------------------------
// Audio+video DiT sharding (`LtxAvDit`) - extends the video-only `Shardable`
// impl above to the coupled AV model. See `crate::dit::AvDitBatch`'s doc for
// why the stage boundary carries TWO residuals (video's and audio's, packed
// into one `Vec<f32>` per `LtxAvDit::read_out_res`'s doc) and `crate::dit::
// av_shard_owns_weight`'s doc for the weight-replication policy (both
// embeddings connectors are replicated here, unlike the video-only path,
// because `LtxAvDit::run_stage_forward` - unlike `LtxDit::run_stage_forward`
// - actually routes `context` through them on every stage).
// ---------------------------------------------------------------------------

/// [`LtxAudioDitConfig`]'s own JSON round trip - not a [`ModelConfig`] on its
/// own (only the bundled [`LtxAvDitConfig`] needs to satisfy that trait), just
/// the nested-object encode/decode `LtxAvDitConfig::to_json`/`from_json` uses
/// for its `audio` field.
fn audio_cfg_to_json(c: &LtxAudioDitConfig) -> serde_json::Value {
    serde_json::json!({
        "inner_dim": c.inner_dim,
        "num_heads": c.num_heads,
        "in_channels": c.in_channels,
        "out_channels": c.out_channels,
        "cross_attention_dim": c.cross_attention_dim,
        "ff_bias": c.ff_bias,
        "positional_embedding_max_pos": c.positional_embedding_max_pos,
        "connector_num_attention_heads": c.connector_num_attention_heads,
        "connector_attention_head_dim": c.connector_attention_head_dim,
    })
}

fn audio_cfg_from_json(v: &serde_json::Value) -> LtxAudioDitConfig {
    let u = |k: &str| v[k].as_u64().unwrap_or_else(|| panic!("LtxAudioDitConfig::from_json: missing {k}")) as u32;
    let max_pos = v["positional_embedding_max_pos"].as_array().unwrap_or_else(|| panic!("LtxAudioDitConfig::from_json: missing positional_embedding_max_pos"));
    LtxAudioDitConfig {
        inner_dim: u("inner_dim"),
        num_heads: u("num_heads"),
        in_channels: u("in_channels"),
        out_channels: u("out_channels"),
        cross_attention_dim: u("cross_attention_dim"),
        ff_bias: v["ff_bias"].as_bool().unwrap_or_else(|| panic!("LtxAudioDitConfig::from_json: missing ff_bias")),
        positional_embedding_max_pos: [max_pos[0].as_u64().unwrap() as u32],
        connector_num_attention_heads: u("connector_num_attention_heads"),
        connector_attention_head_dim: u("connector_attention_head_dim"),
    }
}

impl ModelConfig for LtxAvDitConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        crate::dit::av_dit_tensor_manifest(self).into_iter().map(|(n, shape)| (n, shape.iter().product())).collect()
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "video": self.video.to_json(),
            "audio": audio_cfg_to_json(&self.audio),
            "av_ca_timestep_scale_multiplier": self.av_ca_timestep_scale_multiplier,
        })
    }
    fn from_json(v: &serde_json::Value) -> LtxAvDitConfig {
        LtxAvDitConfig {
            video: LtxDitConfig::from_json(&v["video"]),
            audio: audio_cfg_from_json(&v["audio"]),
            av_ca_timestep_scale_multiplier: v["av_ca_timestep_scale_multiplier"].as_f64().unwrap_or_else(|| panic!("LtxAvDitConfig::from_json: missing av_ca_timestep_scale_multiplier")) as f32,
        }
    }
    fn vocab(&self) -> u32 {
        0
    }
    fn block_size(&self) -> u32 {
        0
    }
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> LtxAvDitConfig {
        self
    }
}

impl Model for LtxAvDit {
    type Config = LtxAvDitConfig;

    fn new(cfg: LtxAvDitConfig, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> LtxAvDit {
        LtxAvDit::from_flat_weights(cfg, init, Shard::whole(cfg.video.num_layers as usize))
    }

    fn init_weights(cfg: &LtxAvDitConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::dit::random_av_tiny_weights(cfg, seed).into_iter().map(|(name, (_, data))| (name, data)).collect()
    }

    fn config(&self) -> &LtxAvDitConfig {
        LtxAvDit::config(self)
    }

    /// A documented no-op - same reason as [`LtxDit::set_batch`]. Use
    /// [`LtxAvDit::load_shard_batch`].
    fn set_batch(&self, _batch: model::Batch) {}

    fn forward(&self) -> f32 {
        self.run_stage_forward().unwrap_or(0.0)
    }

    fn backward(&self) {
        unimplemented!("ltxv::LtxAvDit: no backward pass exists yet (forward-only pipeline-sharding slice) - see crate::shard's module doc for the tracked gap");
    }
    fn zero_grads(&self) {}
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _clip: Option<f32>, _extra_scale: f32) {}
    fn poll_wait(&self) {}

    fn param_names(&self) -> Vec<String> {
        self.weight_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.weight(name)
    }
    fn write_weight(&self, _name: &str, _data: &[f32]) {
        unimplemented!("ltxv::LtxAvDit: weights are immutable post-construction in this slice (no backward/optimizer path exists - see crate::shard's module doc)");
    }
    fn read_grad(&self, _name: &str) -> Vec<f32> {
        unimplemented!("ltxv::LtxAvDit: no gradients exist yet - see crate::shard's module doc for the tracked gap");
    }

    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }

    fn save(&self, _path: &str) {}
    fn config_json(&self) -> serde_json::Value {
        LtxAvDit::config(self).to_json()
    }
}

impl Shardable for LtxAvDit {
    fn shard_cost(cfg: &LtxAvDitConfig, b: u32, t: u32) -> ShardCost {
        // Both streams' per-layer cost, PLUS the bidirectional AV
        // cross-attention (two `Attention` modules per block, each running
        // at the audio stream's - narrower - geometry, see `crate::block::
        // LtxAvBlock`'s doc) - a materially bigger per-layer cost than the
        // video-only path's own `shard_cost`, matching the plan's own
        // framing of this as "the single most architecturally interesting
        // piece" of this extension, not a copy-paste of the video-only
        // formula.
        let vdim = cfg.video.inner_dim as f64;
        let adim = cfg.audio.inner_dim as f64;
        let tt = (b.max(1) as f64) * (t as f64);
        let stream_cost = |dim: f64| -> f64 {
            let self_attn = 4.0 * dim * dim * tt + 2.0 * tt * tt * dim;
            let cross_attn = 4.0 * dim * dim * tt;
            let ffn = 8.0 * dim * dim * tt;
            self_attn + cross_attn + ffn
        };
        // AV cross-attention: TWO directions (A2V, V2A), each 4 projections
        // (Q/K/V/O) at the audio stream's `adim` geometry regardless of which
        // stream is query, plus its own O(Tv·Ta) score/weighted-sum term.
        let av_cross = 2.0 * (4.0 * adim * adim * tt) + 4.0 * tt * tt * adim;
        let per_layer = stream_cost(vdim) + stream_cost(adim) + av_cross;
        // Embed stage: both patchify projections + both adaLN-single timestep
        // MLPs/linears.
        let embed = vdim * cfg.video.in_channels as f64
            + adim * cfg.audio.in_channels as f64
            + 256.0 * vdim
            + vdim * vdim
            + cfg.video.adaln_rows() as f64 * vdim * vdim
            + 256.0 * adim
            + adim * adim
            + cfg.video.adaln_rows() as f64 * adim * adim;
        // Head stage: both `proj_out`s.
        let head = vdim * cfg.video.out_channels as f64 + adim * cfg.audio.out_channels as f64;
        // Boundary traffic: both streams' residuals, concatenated (`LtxAvDit::
        // read_out_res`'s doc).
        let boundary_words = (b.max(1) * t) as usize * (cfg.video.inner_dim as usize + cfg.audio.inner_dim as usize);
        ShardCost { n_layers: cfg.video.num_layers as usize, per_layer, embed, head, boundary_words }
    }

    fn new_shard(cfg: LtxAvDitConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, shard: Shard) -> LtxAvDit {
        let _ = (b, t);
        LtxAvDit::from_flat_weights(cfg, init, shard)
    }

    fn replicated_params(&self) -> Vec<String> {
        // A pseudo-shard that owns no layers and is neither embed nor head:
        // `av_shard_owns_weight` returns `true` for it ONLY on the genuinely
        // replicated names (the REPLICATED_PREFIXES set - its own doc), since
        // every block name needs a non-empty owned range and every embed-/
        // head-only name needs `embed`/`head` set, both false here.
        let none = Shard { start: 0, end: 0, embed: false, head: false, gpu_index: Shard::ANY_GPU };
        Model::param_names(self).into_iter().filter(|n| crate::dit::av_shard_owns_weight(&none, n)).collect()
    }

    fn run_forward_stage(&self) -> Option<f32> {
        self.run_stage_forward()
    }
    fn run_backward_stage(&self) {
        unimplemented!("ltxv::LtxAvDit: no backward pass exists yet - see crate::shard's module doc for the tracked gap");
    }
    fn read_out_res(&self) -> Vec<f32> {
        LtxAvDit::read_out_res(self)
    }
    fn write_in_res(&self, data: &[f32]) {
        LtxAvDit::write_in_res(self, data)
    }
    fn read_in_dres(&self) -> Vec<f32> {
        unimplemented!("ltxv::LtxAvDit: no backward pass exists yet - see crate::shard's module doc for the tracked gap");
    }
    fn write_out_dres(&self, _data: &[f32]) {
        unimplemented!("ltxv::LtxAvDit: no backward pass exists yet - see crate::shard's module doc for the tracked gap");
    }
}

#[cfg(test)]
mod av_tests {
    use super::*;
    use crate::config::LtxAvDitConfig;

    #[test]
    fn av_config_json_round_trips() {
        let cfg = LtxAvDitConfig::tiny_gated();
        let v = cfg.to_json();
        let back = LtxAvDitConfig::from_json(&v);
        assert_eq!(cfg, back);
    }

    /// [`crate::dit::av_shard_owns_weight`]'s own analogue of `shard_owns_
    /// weight`'s coverage test: a whole shard must claim every real manifest
    /// name EXACTLY once (no gap, no double-claim), and a genuine partial
    /// shard must load strictly fewer floats than the whole model.
    #[test]
    fn new_av_shard_loads_only_its_own_weight_subset() {
        let cfg = LtxAvDitConfig::tiny_gated();
        let init = <LtxAvDit as Model>::init_weights(&cfg, 1);

        let whole = Shard::whole(cfg.video.num_layers as usize);
        let full = <LtxAvDit as Shardable>::new_shard(cfg, 1, 4, &init, whole);
        let full_names: std::collections::HashSet<String> = Model::param_names(&full).into_iter().collect();
        let manifest_names: std::collections::HashSet<String> = crate::dit::av_dit_tensor_manifest(&cfg).into_iter().map(|(n, _)| n).collect();
        assert_eq!(full_names, manifest_names, "a whole shard must load exactly the full manifest, no more, no less");

        assert!(cfg.video.num_layers >= 2, "test assumes >=2 layers so a partial shard is meaningful");
        let partial_shard = Shard { start: 0, end: 1, embed: true, head: false, gpu_index: Shard::ANY_GPU };
        let partial = <LtxAvDit as Shardable>::new_shard(cfg, 1, 4, &init, partial_shard);
        let partial_total: usize = Model::param_names(&partial).iter().map(|n| Model::read_weight(&partial, n).len()).sum();
        let full_total: usize = Model::param_names(&full).iter().map(|n| Model::read_weight(&full, n).len()).sum();
        assert!(partial_total < full_total, "a 1-of-{}-layer shard ({partial_total} floats) must be smaller than the whole model ({full_total} floats)", cfg.video.num_layers);

        for name in Model::param_names(&partial) {
            if let Some(rest) = name.strip_prefix("transformer_blocks.") {
                let l: usize = rest.split('.').next().unwrap().parse().unwrap();
                assert_eq!(l, 0, "partial shard [0,1) must not own block {l}'s weights ({name})");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LtxDitConfig;

    /// Every name in the tensor manifest is claimed by exactly the stage(s)
    /// `crate::dit::shard_owns_weight` says it should be - a whole shard
    /// (`Shard::whole`) must cover EVERY name exactly once (proving the
    /// filter is exhaustive, no tensor silently dropped), and a genuine
    /// partial shard must load STRICTLY FEWER floats than the whole model -
    /// proving `new_shard` really does skip the rest of the stack rather
    /// than building it and discarding it.
    #[test]
    fn new_shard_loads_only_its_own_weight_subset() {
        let cfg = LtxDitConfig::tiny();
        let init = <LtxDit as Model>::init_weights(&cfg, 1);

        let whole = Shard::whole(cfg.num_layers as usize);
        let full = <LtxDit as Shardable>::new_shard(cfg, 1, 4, &init, whole);
        let full_names: std::collections::HashSet<String> = Model::param_names(&full).into_iter().collect();
        let manifest_names: std::collections::HashSet<String> = crate::dit::dit_tensor_manifest(&cfg).into_iter().map(|(n, _)| n).collect();
        assert_eq!(full_names, manifest_names, "a whole shard must load exactly the full manifest, no more, no less");

        assert!(cfg.num_layers >= 2, "test assumes >=2 layers so a partial shard is meaningful");
        let partial_shard = Shard { start: 0, end: 1, embed: true, head: false, gpu_index: Shard::ANY_GPU };
        let partial = <LtxDit as Shardable>::new_shard(cfg, 1, 4, &init, partial_shard);
        let partial_total: usize = Model::param_names(&partial).iter().map(|n| Model::read_weight(&partial, n).len()).sum();
        let full_total: usize = Model::param_names(&full).iter().map(|n| Model::read_weight(&full, n).len()).sum();
        assert!(partial_total < full_total, "a 1-of-{}-layer shard ({partial_total} floats) must be smaller than the whole model ({full_total} floats)", cfg.num_layers);

        // Every layer-owning name in the partial shard really is inside [0, 1).
        for name in Model::param_names(&partial) {
            if let Some(rest) = name.strip_prefix("transformer_blocks.") {
                let l: usize = rest.split('.').next().unwrap().parse().unwrap();
                assert_eq!(l, 0, "partial shard [0,1) must not own block {l}'s weights ({name})");
            }
        }
    }

    /// [`plan_balanced`] over [`Shardable::shard_cost`] produces a sane,
    /// well-formed partition - no forward pass required, this is a pure
    /// cost-model/partition-logic check. Run at two scales: the tiny test
    /// config (small enough to actually instantiate) and the REAL
    /// LTX-2.5 22B config's shape (48 layers, `inner_dim` 4096 = 32 heads x
    /// 128) - the cost model only needs the layer count and per-layer dims,
    /// which does not require building or running the 22B checkpoint itself
    /// (a hardware ceiling this port's own validation policy already tracks
    /// separately, not something this test works around).
    fn assert_well_formed(shards: &[Shard], n_layers: usize) {
        assert_eq!(shards[0].start, 0, "first stage must start at layer 0");
        assert_eq!(shards.last().unwrap().end, n_layers, "last stage must end at n_layers");
        for w in shards.windows(2) {
            assert_eq!(w[0].end, w[1].start, "stage layer ranges must be contiguous with no gap or overlap");
            assert!(w[0].end > w[0].start, "no stage may be empty");
        }
        assert!(shards.last().unwrap().end > shards.last().unwrap().start, "no stage may be empty");
        assert_eq!(shards.iter().filter(|s| s.embed).count(), 1, "exactly one embed stage");
        assert_eq!(shards.iter().filter(|s| s.head).count(), 1, "exactly one head stage");
    }

    #[test]
    fn plan_balanced_is_well_formed_for_the_tiny_config() {
        let cfg = LtxDitConfig::tiny();
        let cost = <LtxDit as Shardable>::shard_cost(&cfg, 1, 32);
        assert_eq!(cost.n_layers, cfg.num_layers as usize);
        let shards = model::plan_balanced(&cost, &[0, 1]);
        assert_eq!(shards.len(), 2);
        assert_well_formed(&shards, cfg.num_layers as usize);
    }

    #[test]
    fn plan_balanced_is_well_formed_for_the_real_22b_config_shape() {
        // The real LTX-2.5 video stream: 48 layers, 32 heads x 128 = 4096
        // `inner_dim` - too large to fully instantiate on modest hardware
        // (see this crate's own module doc), but `shard_cost` only reads `cfg`'s
        // plain numeric fields, so the PLAN can be computed without ever
        // allocating the 22B checkpoint's weights.
        let cfg = LtxDitConfig {
            inner_dim: 4096,
            num_heads: 32,
            num_layers: 48,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 4096,
            ..LtxDitConfig::tiny()
        };
        let cost = <LtxDit as Shardable>::shard_cost(&cfg, 1, 4096);
        assert_eq!(cost.n_layers, 48);
        for k in [2usize, 4, 8] {
            let gpus: Vec<usize> = (0..k).collect();
            let shards = model::plan_balanced(&cost, &gpus);
            assert_eq!(shards.len(), k);
            assert_well_formed(&shards, 48);
            // The embed/head stages carry real extra weight (patchify_proj /
            // proj_out at 4096-wide dims) - they must not end up with MORE
            // layers than an interior stage, the same "endpoints give up
            // layers for their extra weight" property `model::shard`'s own
            // tests check for `plan_balanced`.
            if k >= 3 {
                let interior_max = shards[1..k - 1].iter().map(|s| s.end - s.start).max().unwrap();
                assert!(shards[0].end - shards[0].start <= interior_max, "embed stage should not carry MORE layers than an interior stage");
                assert!(shards[k - 1].end - shards[k - 1].start <= interior_max, "head stage should not carry MORE layers than an interior stage");
            }
        }
    }

    #[test]
    fn config_json_round_trips() {
        let cfg = LtxDitConfig::tiny();
        let v = cfg.to_json();
        let back = LtxDitConfig::from_json(&v);
        assert_eq!(cfg, back);
    }
}
