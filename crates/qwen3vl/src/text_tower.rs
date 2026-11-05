// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL's language tower alone, for text-only conditioning, placed across
//! as many cards as it needs.
//!
//! Swedish Embedded AB implements solutions for running text encoders larger
//! than any one accelerator for its clients. If your team needs expertise in
//! multi-device model placement then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! A diffusion model that conditions on an intermediate hidden state of a
//! large Qwen3-VL (MiniMax-H3 reads the 32B-class tower at layer 50) needs
//! neither the vision tower nor the LM head, and the layers it does need can
//! outgrow one card even at int8: 50 layers of the 32B tower are ~27 GB of
//! packed weights. The tower is therefore cut into consecutive layer stages
//! by [`model::shard::plan_fewest_devices`] over the schedulable cards' real
//! capacity, each stage a [`qwen3::Qwen`] shard on its own card, and an encode
//! hands the prompt's rows from one stage to the next
//! ([`qwen3::Qwen::encode_stage`]). With no card schedulable it is one stage on
//! the ambient device.
//!
//! Every stage is built at the prompt's own length: the tower serves one
//! encode and is dropped, so its activations are sized for the prompt rather
//! than for a maximum the caller would otherwise have to guess.

use gpu_core::select::Dtype;
use model::shard::{LayerBytes, Shard};
use qwen3::{Qwen, QwenConfig};

use crate::mrope::mrope_tables;

/// Per card, what placement keeps free for the driver and the batched
/// forward's transient buffers beyond the per-layer activations it prices.
const CARD_RESERVE_BYTES: u64 = 1 << 30;

/// The vision blocks of one multimodal prompt, already through the vision
/// tower ([`crate::encoder::VisionTower`]): `(row0, n_rows)` per block in
/// prompt order, the blocks' visual rows concatenated in that order, and per
/// DeepStack level the blocks' rows concatenated the same way.
pub struct VisionSplice<'a> {
    pub runs: &'a [(u32, u32)],
    pub visual: &'a [f32],
    pub deepstack: &'a [Vec<f32>],
}

/// The language tower's first `depth` layers, one [`Qwen`] stage per card.
pub struct TextTower {
    stages: Vec<Qwen>,
    cfg: QwenConfig,
    mrope_section: [u32; 3],
}

/// Device bytes each of the first `depth` layers holds in a batched encode
/// of `rows` tokens at linear tier `dt` - the packed linears, the fp32 norms
/// and the per-layer activations - with the fp32 embedding on the first
/// stage and no head.
pub fn tower_cost(cfg: &QwenConfig, depth: usize, dt: Dtype, rows: u32) -> LayerBytes {
    let (t, d, ff) = (rows as u64, cfg.d_model as u64, cfg.d_ff as u64);
    let (q, kv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
    let norms = (2 * d + 2 * cfg.head_dim as u64) * 4;
    let activations = (t * (4 * d + q + 2 * kv + 3 * ff) + cfg.n_heads as u64 * t * t) * 4;
    LayerBytes { per_layer: vec![crate::footprint::decoder_layer_linear_bytes(cfg, dt) + norms + activations; depth], embed: crate::footprint::embed_bytes(cfg), head: 0 }
}

impl TextTower {
    /// Build the first `depth` layers of the HF Qwen3-VL checkpoint at `dir`
    /// for prompts of up to `rows` tokens, its linears at `dt`, placed over
    /// the schedulable cards. Refuses, naming the bytes, when the tower fits
    /// no combination of them.
    pub fn from_hf_dir(dir: &str, cfg: QwenConfig, mrope_section: [u32; 3], dt: Dtype, depth: usize, rows: u32) -> Result<TextTower, String> {
        assert!(depth <= cfg.n_layers as usize, "text tower: depth {depth} past the checkpoint's {} layers", cfg.n_layers);
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(dir)).map_err(|e| format!("qwen3vl text tower: open {dir}: {e}"))?;
        let source = crate::import::decoder_source(&reader);
        let cards: Vec<(usize, u64)> = gpu_core::devices::schedulable_gpus().iter().map(|d| (d.index as usize, d.identity.vram_bytes.saturating_sub(CARD_RESERVE_BYTES))).collect();
        let shards: Vec<Shard> = if cards.is_empty() {
            vec![Shard { start: 0, end: depth, embed: true, head: false, gpu_index: Shard::ANY_GPU }]
        } else {
            let cost = tower_cost(&cfg, depth, dt, rows);
            let placed = model::shard::plan_fewest_devices(&cost, &cards).ok_or_else(|| {
                format!("qwen3vl text tower: {depth} layers at {dt:?} need {} bytes, which fit no combination of the schedulable cards {cards:?} (usable bytes)", cost.total())
            })?;
            placed.into_iter().map(|p| Shard { head: false, ..p.shard }).collect()
        };
        Ok(TextTower::with_shards(&source, cfg, mrope_section, dt, shards, rows))
    }

    /// The tower over `source` (decoder tensors in `qwen3`'s own names) at
    /// the caller's own layer stages, for prompts of up to `rows` tokens.
    pub fn with_shards(source: &dyn checkpoint::TensorSource, cfg: QwenConfig, mrope_section: [u32; 3], dt: Dtype, shards: Vec<Shard>, rows: u32) -> TextTower {
        let stages = shards
            .into_iter()
            .map(|shard| {
                let mut stage = Qwen::new_shard_dt(cfg.clone(), 1, rows, source, shard, dt);
                stage.enable_mrope();
                stage
            })
            .collect();
        TextTower { stages, cfg, mrope_section }
    }

    /// How many stages (cards) the tower spans.
    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// The hidden state after the tower's `depth` layers for a text-only
    /// prompt, `[tokens.len(), d_model]` - what `Qwen3Vl::encode_hidden(tokens,
    /// depth)` returns, read across however many stages the tower spans.
    pub fn encode(&self, tokens: &[u32]) -> Vec<f32> {
        // A pure-text stream's M-RoPE position is `[i, i, i]` - see
        // `Qwen3Vl::write_text_mrope_tables` for why the table must be written.
        let positions: Vec<[u32; 3]> = (0..tokens.len() as u32).map(|i| [i; 3]).collect();
        self.encode_at(tokens, &positions)
    }

    /// [`Self::encode`] for a prompt that carries vision blocks: each block's
    /// placeholder rows are overwritten by its visual tokens, its DeepStack
    /// levels are added after the first layers, and every row rotates by
    /// its 3-axis `positions` (`get_rope_index_multi` over the blocks'
    /// merged grids). The DeepStack layers must sit on the first stage,
    /// which a placement of any real tower satisfies, and the tower must
    /// reach past them: transformers records `hidden_states[l]` as layer
    /// `l - 1`'s output before that layer's DeepStack add, so only past the
    /// last add is the stream the same quantity.
    pub fn encode_multimodal(&mut self, tokens: &[u32], positions: &[[u32; 3]], vision: &VisionSplice) -> Vec<f32> {
        assert_eq!(positions.len(), tokens.len(), "text tower: one 3-axis position per token");
        let depth = self.stages.last().expect("a tower has at least one stage").shard.end;
        assert!(depth > vision.deepstack.len(), "text tower: a tap at layer {depth} does not reach past the {} DeepStack adds", vision.deepstack.len());
        let d = self.cfg.d_model as usize;
        let n_rows: usize = vision.runs.iter().map(|&(_, n)| n as usize).sum();
        assert_eq!(vision.visual.len(), n_rows * d, "text tower: the visual rows must cover every run");
        let first = &mut self.stages[0];
        assert!(first.shard.end >= vision.deepstack.len(), "text tower: DeepStack adds after layers 0..{} but the first stage ends at layer {}", vision.deepstack.len(), first.shard.end);
        first.enable_mm_splices(vision.runs);
        first.write_img_embeds(vision.visual);
        if !vision.deepstack.is_empty() {
            first.enable_deepstack_runs(vision.runs, vision.deepstack.len() as u32);
            for (level, rows) in vision.deepstack.iter().enumerate() {
                first.write_deepstack(level, rows);
            }
        }
        self.encode_at(tokens, positions)
    }

    fn encode_at(&self, tokens: &[u32], positions: &[[u32; 3]]) -> Vec<f32> {
        let (cos, sin) = mrope_tables(positions, self.mrope_section, self.cfg.head_dim, self.cfg.rope_theta);
        let mut stream: Option<Vec<f32>> = None;
        for stage in &self.stages {
            stage.write_mrope_tables(&cos, &sin);
            stream = Some(stage.encode_stage(tokens, stream.as_deref()));
        }
        stream.expect("a tower has at least one stage")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cut anywhere, the tower encodes what it encodes whole - which also
    /// means every stage, not only the first, rotates by the prompt's own
    /// M-RoPE table (a stage left on an unwritten table zeroes its q/k).
    #[test]
    fn a_tower_cut_into_stages_encodes_what_it_encodes_whole() {
        let cfg = QwenConfig::tiny();
        let weights = qwen3::init::init_weights(&cfg, data::rng::random_seed());
        let depth = cfg.n_layers as usize;
        // Three axes splitting the half head, as every released section does.
        let half = cfg.head_dim / 2;
        let section = [half - 2 * (half / 3), half / 3, half / 3];
        let shard = |start: usize, end: usize| Shard { start, end, embed: start == 0, head: false, gpu_index: Shard::ANY_GPU };
        let tokens: Vec<u32> = (0..9u32).map(|i| (i * 11 + 5) % cfg.vocab).collect();

        let whole = TextTower::with_shards(&weights, cfg.clone(), section, Dtype::F32, vec![shard(0, depth)], tokens.len() as u32);
        let cut = TextTower::with_shards(&weights, cfg.clone(), section, Dtype::F32, vec![shard(0, depth / 2), shard(depth / 2, depth)], tokens.len() as u32);
        assert_eq!(cut.encode(&tokens), whole.encode(&tokens));
    }
}
