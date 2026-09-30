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

pub struct Vlm {
    pub decoder_cfg: QwenConfig,
    pub processor: ImageProcessor,
    pub tower: Box<dyn VisionTower>,
    pub decoder: Qwen,
    pub tokenizer: QwenBpe,
    pub style: Style,
    pub splice: ImageSplice,
    /// The begin- and end-of-sentence texts, as `tokenizer_config.json`
    /// names them.
    pub bos: String,
    pub eos: String,
    pub eos_id: u32,
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

impl Vlm {
    /// Assemble a composite from `parts` and the tokenizer, processor config
    /// and decoder in `dir` (read through `rd`), the decoder at `dtype` with
    /// a KV cache for `ctx` tokens.
    pub fn assemble(dir: &Path, rd: &WeightReader, parts: Parts, dtype: qwen3::Dtype, ctx: u32) -> Result<Vlm, String> {
        let decoder_cfg = qwen3::hf::decoder_config_as(&parts.language.to_string(), "llama")?;
        let processor = ImageProcessor::from_dir(dir)?;
        if processor.image_size as usize != parts.tower.image_size() {
            return Err(format!("the processor makes {} px squares for a {} px tower", processor.image_size, parts.tower.image_size()));
        }
        let src = qwen3::import::nested_source(rd, crate::import::DECODER, &decoder_cfg)?;
        let decoder = Qwen::new_shard_dt_decode(decoder_cfg.clone(), ctx, &src, Shard::whole(decoder_cfg.n_layers as usize), dtype);
        drop(src);

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
        Ok(Vlm { decoder_cfg, processor, tower: parts.tower, decoder, tokenizer, style: parts.style, splice, bos, eos, eos_id })
    }

    /// The prompt's token ids, BOS first, with one placeholder id per image.
    pub fn prompt_ids(&self, turns: &[Turn]) -> Result<Vec<u32>, String> {
        let text = prompt::render(&self.style, prompt::SYSTEM_PROMPT, turns, &self.eos)?;
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

    /// Prefill `ids` with `embeds` spliced at the placeholders, then decode
    /// greedily until the end-of-sentence id or `max_new` tokens. `on_token`
    /// sees each id as it is produced.
    pub fn generate_greedy(&self, ids: &[u32], embeds: &[f32], max_new: usize, on_token: &mut dyn FnMut(u32)) -> Result<Vec<u32>, String> {
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
            on_token(next);
            self.decoder.step(next);
        }
        Ok(out)
    }
}

/// Load DeepSeek-VL from `dir` with the decoder at `dtype`, its KV cache
/// sized for `ctx` tokens (prompt, image rows and reply together).
pub fn load(dir: &Path, dtype: qwen3::Dtype, ctx: u32) -> Result<Vlm, String> {
    let cfg = DeepseekVlConfig::from_dir(dir)?;
    let width = qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?.d_model;
    if width != cfg.aligner.n_embed {
        return Err(format!("the aligner produces {}-wide rows for a {width}-wide decoder", cfg.aligner.n_embed));
    }
    let rd = WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    crate::import::check_coverage(&rd, &[crate::import::SAM_PREFIX, clip::import::siglip::DEEPSEEK_VL_LOW_PREFIX, crate::import::ALIGNER_PREFIX, "language_model."])?;
    let tower = Box::new(HybridTower::load(&rd, &cfg)?);
    Vlm::assemble(dir, &rd, Parts { tower, style: prompt::DEEPSEEK_VL, wrap: None, language: cfg.language }, dtype, ctx)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}
