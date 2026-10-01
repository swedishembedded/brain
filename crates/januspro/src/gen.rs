// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro's generation heads: what turns the decoder into an image-token
//! model.
//!
//! * `gen_head` maps a decoder hidden state to logits over the VQ codebook
//!   (Linear, erf GELU, Linear into the image vocabulary);
//! * `gen_embed` is the codebook-sized embedding table of the tokens fed
//!   back (its own 8-wide table, not the VQ codebook);
//! * `gen_aligner` lifts those rows to the decoder width (the same
//!   `mlp_gelu` shape as the understanding aligner).
//!
//! Both MLPs are `model::projector` stacks on one device, sized for a fixed
//! number of rows: one per sequence of a batch (conditional and
//! unconditional rows for classifier-free guidance).

use std::collections::HashMap;

use checkpoint::weightio::WeightReader;
use gpu_core::{DeviceBuffer, Gpu};
use model::projector::{MlpProjector, ProjectorConfig, PROJECTOR_PIPELINES};

use crate::config::JanusProConfig;

/// Where the checkpoint keeps the generation parts.
pub const GEN_HEAD_PREFIX: &str = "gen_head.";
pub const GEN_ALIGNER_PREFIX: &str = "gen_aligner.";
pub const GEN_EMBED: &str = "gen_embed.weight";

pub struct GenHeads {
    gpu: Gpu,
    head: MlpProjector,
    aligner: MlpProjector,
    /// `[image_vocab, code_dim]`.
    embed: Vec<f32>,
    code_dim: usize,
    rows: u32,
    hidden_in: DeviceBuffer,
    codes_in: DeviceBuffer,
}

/// `gen_head`'s two linears by their brain projector names.
pub(crate) fn head_weights(rd: &WeightReader, cfg: &ProjectorConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut out = HashMap::new();
    for (name, n) in cfg.param_list() {
        let (module, param) = name.rsplit_once('.').expect("projector params are module.param");
        let upstream = match module {
            "in" => "output_mlp_projector",
            "layers.1" => "vision_head",
            other => return Err(format!("gen_head has no linear for {other}")),
        };
        let full = format!("{GEN_HEAD_PREFIX}{upstream}.{param}");
        let data = rd.tensor(&full).ok_or_else(|| format!("{full} is not in the checkpoint"))?;
        if data.len() != n {
            return Err(format!("{full} holds {} values, {name} takes {n}", data.len()));
        }
        out.insert(name, data);
    }
    let expected: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
    if let Some(stray) = rd.names().filter(|n| n.starts_with(GEN_HEAD_PREFIX)).find(|n| {
        let leaf = &n[GEN_HEAD_PREFIX.len()..];
        !["output_mlp_projector.weight", "output_mlp_projector.bias", "vision_head.weight", "vision_head.bias"].contains(&leaf)
    }) {
        return Err(format!("{stray} is not one of gen_head's {} parameters", expected.len()));
    }
    Ok(out)
}

impl GenHeads {
    /// The heads from a Janus-Pro checkpoint, for batches of `rows`
    /// sequences.
    pub fn load(rd: &WeightReader, cfg: &JanusProConfig, rows: u32) -> Result<GenHeads, String> {
        let h = &cfg.gen_head;
        let head_cfg = ProjectorConfig::from_type("mlp_gelu", 2, h.n_embed, h.image_token_embed)?.with_out_dim(h.image_token_size)?;
        let a = &cfg.gen_aligner;
        let aligner_cfg = ProjectorConfig::from_type(&a.projector_type, a.depth, a.input_dim, a.n_embed)?;
        let code_dim = cfg.gen_vision.n_embed as usize;
        if aligner_cfg.input_dim as usize != code_dim || aligner_cfg.n_embed != h.n_embed {
            return Err(format!("gen_aligner maps {} to {}, the generation rows are {code_dim} wide and the decoder {}", aligner_cfg.input_dim, aligner_cfg.n_embed, h.n_embed));
        }
        let embed = rd.tensor(GEN_EMBED).ok_or_else(|| format!("{GEN_EMBED} is not in the checkpoint"))?;
        if embed.len() != h.image_token_size as usize * code_dim {
            return Err(format!("{GEN_EMBED} holds {} values, not {} x {code_dim}", embed.len(), h.image_token_size));
        }
        let gpu = Gpu::new(PROJECTOR_PIPELINES);
        let head = MlpProjector::new_frozen(&gpu, head_cfg, rows, &head_weights(rd, &head_cfg)?)?;
        let aligner = MlpProjector::new_frozen(&gpu, aligner_cfg, rows, &deepseekvl::import::aligner_weights(rd, GEN_ALIGNER_PREFIX, &aligner_cfg)?)?;
        let hidden_in = gpu.storage((rows * h.n_embed) as u64);
        let codes_in = gpu.storage(rows as u64 * code_dim as u64);
        Ok(GenHeads { gpu, head, aligner, embed, code_dim, rows, hidden_in, codes_in })
    }

    /// Replace the heads with a fine-tune's: `tensors` as
    /// [`crate::train::GenOutcome::save`] names them (`gen_head.*`,
    /// `gen_aligner.*`, `gen_embed.weight`).
    pub fn apply_tuned(&mut self, tensors: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        for (prefix, projector) in [(GEN_HEAD_PREFIX, &self.head), (GEN_ALIGNER_PREFIX, &self.aligner)] {
            for (name, n) in projector.cfg.param_list() {
                let key = format!("{prefix}{name}");
                let w = tensors.get(&key).ok_or_else(|| format!("the fine-tune has no {key}"))?;
                if w.len() != n {
                    return Err(format!("{key} holds {} values, expected {n}", w.len()));
                }
                self.gpu.write_f32(projector.param(&name), w);
            }
        }
        let embed = tensors.get(GEN_EMBED).ok_or_else(|| format!("the fine-tune has no {GEN_EMBED}"))?;
        if embed.len() != self.embed.len() {
            return Err(format!("{GEN_EMBED} holds {} values, expected {}", embed.len(), self.embed.len()));
        }
        self.embed.copy_from_slice(embed);
        Ok(())
    }

    /// The batch size the heads were built for.
    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    /// The image vocabulary (the VQ codebook size).
    pub fn vocab(&self) -> usize {
        self.head.cfg.out_dim as usize
    }

    /// Image-token logits `[rows, vocab]` for the decoder's final-norm
    /// hidden states `[rows, n_embed]`.
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        assert_eq!(hidden.len(), self.rows as usize * self.head.cfg.input_dim as usize, "one hidden row per sequence");
        self.gpu.write_f32(&self.hidden_in, hidden);
        self.gpu.submit(&[], &self.head.forward(&self.gpu, &[&self.hidden_in]));
        self.gpu.read(self.head.out(), self.rows as usize * self.vocab())
    }

    /// The decoder-width rows `[rows, n_embed]` that feed image tokens `ids`
    /// (one per sequence) back into the decoder.
    pub fn token_embeds(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.len() != self.rows as usize {
            return Err(format!("{} image tokens for {} sequences", ids.len(), self.rows));
        }
        let mut codes = Vec::with_capacity(ids.len() * self.code_dim);
        for &t in ids {
            let row = self.embed.get(t as usize * self.code_dim..(t as usize + 1) * self.code_dim).ok_or_else(|| format!("image token {t} is outside the {}-entry vocabulary", self.vocab()))?;
            codes.extend_from_slice(row);
        }
        self.gpu.write_f32(&self.codes_in, &codes);
        self.gpu.submit(&[], &self.aligner.forward(&self.gpu, &[&self.codes_in]));
        Ok(self.gpu.read(self.aligner.out(), self.rows as usize * self.aligner.cfg.n_embed as usize))
    }
}
