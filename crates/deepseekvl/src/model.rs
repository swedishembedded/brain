// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `MultiModalityCausalLM` understanding composite: tokenizer,
//! preprocessing, a vision tower with its aligner, and the decoder, loaded
//! from one checkpoint directory. [`load`] assembles DeepSeek-VL's;
//! `brain-januspro` assembles Janus-Pro's from the same parts.
//!
//! The decoder is `brain-qwen3`'s, reading the checkpoint's partial
//! `language_config` with `LlamaConfig`'s defaults (a 30-layer, 4096-wide
//! Llama). Its weight tier is the caller's `qwen3::Dtype`; the checkpoint's
//! own (fp16 for DeepSeek-VL, bf16 for Janus-Pro) keeps its values exactly.

use std::path::Path;

use checkpoint::weightio::WeightReader;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use imaging::pixels::Rgb8;
use model::shard::Shard;
use qwen3::model::PrefillInput;
use qwen3::{Qwen, QwenConfig};

use crate::config::DeepseekVlConfig;
use crate::preprocess::ImageProcessor;
use crate::prompt::{self, ImageSplice, Style, Turn};
use crate::tower::{HybridTower, VisionTower};

/// Everything of the composite but the decoder's weights: what turns an
/// image and a conversation into the decoder's inputs.
pub struct Frontend {
    pub decoder_cfg: QwenConfig,
    pub processor: ImageProcessor,
    pub tower: Box<dyn VisionTower>,
    pub tokenizer: QwenBpe,
    pub style: Style,
    pub splice: ImageSplice,
    /// The begin- and end-of-sentence texts, as `tokenizer_config.json`
    /// names them.
    pub bos: String,
    pub eos: String,
    pub eos_id: u32,
}

/// The composite: its [`Frontend`] (reachable as the composite's own fields)
/// and the decoder.
pub struct Vlm {
    frontend: Frontend,
    pub decoder: Qwen,
}

impl std::ops::Deref for Vlm {
    type Target = Frontend;
    fn deref(&self) -> &Frontend {
        &self.frontend
    }
}

fn special(tok_cfg: &serde_json::Value, key: &str) -> Result<String, String> {
    let v = &tok_cfg[key];
    v.as_str().or_else(|| v["content"].as_str()).map(str::to_string).ok_or_else(|| format!("tokenizer_config.json: no '{key}'"))
}

/// The parts of a composite that differ between the models built from it.
pub struct Parts {
    pub tower: Box<dyn VisionTower>,
    pub style: Style,
    /// The begin- and end-of-image tags around each image, when the model
    /// wraps its images.
    pub wrap: Option<(&'static str, &'static str)>,
    /// `language_config`, as the checkpoint writes it.
    pub language: serde_json::Value,
}

impl Frontend {
    /// Open the tokenizer, processor and splice of the composite in `dir`
    /// around `parts.tower`, reading the decoder's shape from
    /// `parts.language`.
    pub fn open(dir: &Path, parts: Parts) -> Result<Frontend, String> {
        let decoder_cfg = qwen3::hf::decoder_config_as(&parts.language.to_string(), "llama")?;
        let processor = ImageProcessor::from_dir(dir)?;
        if processor.image_size as usize != parts.tower.image_size() {
            return Err(format!("the processor makes {} px squares for a {} px tower", processor.image_size, parts.tower.image_size()));
        }
        let dir_str = dir.to_str().ok_or("checkpoint path is not UTF-8")?;
        let tokenizer = QwenBpe::from_dir(dir_str)?;
        let tok_cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).map_err(|e| format!("tokenizer_config.json: {e}"))?)
            .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let (bos, eos) = (special(&tok_cfg, "bos_token")?, special(&tok_cfg, "eos_token")?);
        let id = |s: &str| tokenizer.special_id(s).ok_or_else(|| format!("the tokenizer has no {s:?} token"));
        id(&bos)?;
        let eos_id = id(&eos)?;
        let wrap = match parts.wrap {
            Some((open, close)) => Some((id(open)?, id(close)?)),
            None => None,
        };
        let splice = ImageSplice { image_id: id(prompt::IMAGE_TAG)?, rows: parts.tower.rows(), wrap };
        Ok(Frontend { decoder_cfg, processor, tower: parts.tower, tokenizer, style: parts.style, splice, bos, eos, eos_id })
    }

    /// The prompt's token ids, BOS first, with one placeholder id per image,
    /// under the processor's own system prompt.
    pub fn prompt_ids(&self, turns: &[Turn]) -> Result<Vec<u32>, String> {
        self.prompt_ids_with(None, turns)
    }

    /// [`Self::prompt_ids`] under `system` instead, when given.
    pub fn prompt_ids_with(&self, system: Option<&str>, turns: &[Turn]) -> Result<Vec<u32>, String> {
        let text = prompt::render(&self.style, system.unwrap_or(prompt::SYSTEM_PROMPT), turns, &self.eos)?;
        Ok(self.tokenizer.encode(&format!("{}{text}", self.bos)))
    }

    /// Every image's aligner rows, concatenated in order.
    pub fn image_embeds(&self, images: &[Rgb8]) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(images.len() * self.tower.rows() * self.decoder_cfg.d_model as usize);
        for img in images {
            out.extend(self.tower.encode(&self.processor.pixel_values(img)?).embeds);
        }
        Ok(out)
    }

    /// The prefill inputs for `ids` with `embeds` spliced at the placeholders.
    pub fn inputs<'a>(&self, ids: &[u32], embeds: &'a [f32]) -> Result<Vec<PrefillInput<'a>>, String> {
        self.splice.inputs(ids, embeds, self.decoder_cfg.d_model as usize)
    }

}

impl Vlm {
    /// Assemble a composite from `parts` and the tokenizer, processor config
    /// and decoder in `dir` (read through `rd`), the decoder at `dtype` with
    /// a KV cache for `ctx` tokens.
    pub fn assemble(dir: &Path, rd: &WeightReader, parts: Parts, dtype: qwen3::Dtype, ctx: u32) -> Result<Vlm, String> {
        let frontend = Frontend::open(dir, parts)?;
        let decoder_cfg = &frontend.decoder_cfg;
        let src = qwen3::import::nested_source(rd, crate::import::DECODER, decoder_cfg)?;
        let decoder = Qwen::new_shard_dt_decode(decoder_cfg.clone(), ctx, &src, Shard::whole(decoder_cfg.n_layers as usize), dtype);
        drop(src);
        Ok(Vlm { frontend, decoder })
    }

    /// Prefill `ids` with `embeds` spliced at the placeholders, then decode
    /// greedily until the end-of-sentence id, `max_new` tokens, or `on_token`
    /// (which sees each id as it is produced) returns `false`.
    pub fn generate_greedy(&self, ids: &[u32], embeds: &[f32], max_new: usize, on_token: &mut dyn FnMut(u32) -> bool) -> Result<Vec<u32>, String> {
        let inputs = self.inputs(ids, embeds)?;
        let ctx = self.decoder.ctx_len();
        if inputs.len() + max_new > ctx {
            return Err(format!("{} prompt rows and {max_new} new tokens exceed the {ctx}-token context this model was loaded with", inputs.len()));
        }
        self.decoder.reset_cache();
        self.decoder.prefill(&inputs);
        let mut out = Vec::new();
        for _ in 0..max_new {
            let next = argmax(&self.decoder.decode_logits());
            if next == self.eos_id {
                break;
            }
            out.push(next);
            if !on_token(next) {
                break;
            }
            self.decoder.step(next);
        }
        Ok(out)
    }
}

/// DeepSeek-VL's hybrid tower on its device: SAM-B's and SigLIP-L's fp32
/// weights with the activations of one 1024-pixel image (SAM's global
/// attention over 4096 patches dominates), measured on the released
/// checkpoint.
pub const HYBRID_TOWER_BYTES: u64 = 8 << 30;

/// DeepSeek-VL's footprint, from the checkpoint's own config.
pub fn footprint(dir: &Path) -> Result<Footprint, String> {
    let cfg = DeepseekVlConfig::from_dir(dir)?;
    Ok(Footprint::of(&qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?, HYBRID_TOWER_BYTES))
}

/// Load DeepSeek-VL from `dir` with the decoder at `dtype`, its KV cache
/// sized for `ctx` tokens (prompt, image rows and reply together), on the
/// ambient device.
pub fn load(dir: &Path, dtype: qwen3::Dtype, ctx: u32) -> Result<Vlm, String> {
    load_with(dir, dtype, ctx, &|f| Ok(f()), &|f| Ok(f()))
}

/// [`load`] as `placement` says: the tower on its card, the decoder on its.
pub fn load_placed(dir: &Path, dtype: qwen3::Dtype, placement: Placement) -> Result<Vlm, String> {
    let on = |card: u32| move |f: &mut dyn FnMut()| gpu_core::devices::with_gpu(card, f);
    load_with(dir, dtype, placement.context, &on(placement.tower), &on(placement.decoder))
}

/// Build the tower inside `on_tower` and the decoder inside `on_decoder` (each
/// scopes the device every `Gpu::new` in it lands on).
type OnDevice<'a> = &'a dyn Fn(&mut dyn FnMut()) -> Result<(), String>;

fn load_with(dir: &Path, dtype: qwen3::Dtype, ctx: u32, on_tower: OnDevice, on_decoder: OnDevice) -> Result<Vlm, String> {
    let cfg = DeepseekVlConfig::from_dir(dir)?;
    let width = qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?.d_model;
    if width != cfg.aligner.n_embed {
        return Err(format!("the aligner produces {}-wide rows for a {width}-wide decoder", cfg.aligner.n_embed));
    }
    let rd = WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    crate::import::check_coverage(&rd, &[crate::import::SAM_PREFIX, clip::import::siglip::DEEPSEEK_VL_LOW_PREFIX, crate::import::ALIGNER_PREFIX, "language_model."])?;
    let mut tower = None;
    on_tower(&mut || tower = Some(HybridTower::load(&rd, &cfg)))?;
    let tower = Box::new(tower.expect("built in scope")?);
    let mut vlm = None;
    let parts = Parts { tower, style: prompt::DEEPSEEK_VL, wrap: None, language: cfg.language };
    let mut parts = Some(parts);
    on_decoder(&mut || vlm = Some(Vlm::assemble(dir, &rd, parts.take().expect("built once"), dtype, ctx)))?;
    vlm.expect("built in scope")
}

/// What a composite occupies on its devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// The vision tower and aligner with their activations.
    pub tower: u64,
    /// The decoder's weights and scratch, before its KV cache.
    pub decoder: u64,
    /// The decoder's fp32 KV cache per context token.
    pub kv_per_token: u64,
    /// The longest context the checkpoint's position table covers.
    pub max_context: u32,
}

/// Device bytes a fresh device and its activations need beyond the weights.
const DECODER_OVERHEAD: u64 = 1 << 30;

impl Footprint {
    /// The footprint of a decoder shaped `cfg` with half-precision linears and
    /// head and an fp32 embedding table, beside a tower of `tower` bytes.
    pub fn of(cfg: &QwenConfig, tower: u64) -> Footprint {
        let (d, ff, v, l) = (cfg.d_model as u64, cfg.d_ff as u64, cfg.vocab as u64, cfg.n_layers as u64);
        let (q, kv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        let linears = l * (d * (q + 2 * kv) + q * d + 3 * d * ff);
        Footprint {
            tower,
            decoder: linears * 2 + v * d * 4 + v * d * 2 + DECODER_OVERHEAD,
            kv_per_token: l * 2 * kv * 4,
            max_context: cfg.max_position_embeddings,
        }
    }

    /// The decoder's bytes at `context` tokens.
    pub fn decoder_at(&self, context: u32) -> u64 {
        self.decoder + context as u64 * self.kv_per_token
    }
}

/// Where a composite's two parts go and the context its decoder is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub tower: u32,
    pub decoder: u32,
    pub context: u32,
}

/// The shortest context worth serving: one image's rows and a conversation.
pub const MIN_CONTEXT: u32 = 1024;

/// Place a composite over `cards` (`(index, free bytes)`): the decoder on the
/// roomiest card, the tower on the next one when there is one (else beside the
/// decoder), and the context whatever KV cache fits the decoder's card, up to
/// the checkpoint's own maximum, in whole 256-token steps.
pub fn place(fp: &Footprint, cards: &[(u32, u64)]) -> Result<Placement, String> {
    let mut by_room: Vec<(u32, u64)> = cards.to_vec();
    by_room.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let &(decoder, room) = by_room.first().ok_or("no GPU to place the model on")?;
    let (tower, room) = match by_room.get(1) {
        Some(&(t, free)) if free >= fp.tower => (t, room),
        _ => (decoder, room.saturating_sub(fp.tower)),
    };
    let tokens = room.saturating_sub(fp.decoder) / fp.kv_per_token.max(1);
    let context = (tokens.min(fp.max_context as u64) as u32) / 256 * 256;
    if context < MIN_CONTEXT {
        return Err(format!(
            "the model needs {} GiB for its decoder and {} GiB for its tower plus {} MiB per context token; the roomiest card has {} GiB free",
            fp.decoder >> 30,
            fp.tower >> 30,
            fp.kv_per_token >> 20,
            by_room[0].1 >> 30
        ));
    }
    Ok(Placement { tower, decoder, context })
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn deepseek_vl() -> Footprint {
        let cfg = qwen3::hf::decoder_config_as(r#"{"max_position_embeddings":16384,"model_type":"llama","num_hidden_layers":30,"vocab_size":102400}"#, "llama").unwrap();
        Footprint::of(&cfg, HYBRID_TOWER_BYTES)
    }

    #[test]
    fn two_cards_split_the_tower_from_the_decoder_and_the_rest_is_context() {
        let fp = deepseek_vl();
        let p = place(&fp, &[(0, 20 * GIB), (1, 22 * GIB)]).unwrap();
        assert_eq!((p.tower, p.decoder), (0, 1), "the decoder takes the roomiest card");
        assert!(fp.decoder_at(p.context) <= 22 * GIB && fp.decoder_at(p.context + 256) > 22 * GIB, "{p:?}");
        assert_eq!(p.context % 256, 0);
    }

    #[test]
    fn one_card_holds_both_only_when_a_useful_context_remains() {
        let fp = deepseek_vl();
        assert!(place(&fp, &[(0, 22 * GIB)]).unwrap_err().contains("per context token"));
        let p = place(&fp, &[(0, 40 * GIB)]).unwrap();
        assert_eq!((p.tower, p.decoder), (0, 0));
        assert!(fp.tower + fp.decoder_at(p.context) <= 40 * GIB);
    }

    #[test]
    fn the_context_stops_at_the_checkpoints_position_table() {
        let fp = deepseek_vl();
        assert_eq!(place(&fp, &[(0, 80 * GIB), (1, 80 * GIB)]).unwrap().context, 16384);
    }
}
