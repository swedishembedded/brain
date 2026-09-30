// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DeepSeek-VL composite: tokenizer, preprocessing, hybrid tower and
//! decoder, loaded from one checkpoint directory.
//!
//! The decoder is `brain-qwen3`'s, reading the checkpoint's partial
//! `language_config` with `LlamaConfig`'s defaults (a 30-layer, 4096-wide
//! Llama). Its weight tier is the caller's: [`qwen3::Dtype::F16`] keeps the
//! checkpoint's own values, [`qwen3::Dtype::I8`] quantizes them to fit a
//! smaller card.

use std::path::Path;

use checkpoint::weightio::WeightReader;
use data::qwen_tokenizer::QwenBpe;
use imaging::pixels::Rgb8;
use model::shard::Shard;
use qwen3::model::PrefillInput;
use qwen3::{Qwen, QwenConfig};

use crate::config::DeepseekVlConfig;
use crate::preprocess::ImageProcessor;
use crate::prompt::{self, Turn};
use crate::tower::HybridTower;

pub struct DeepseekVl {
    pub cfg: DeepseekVlConfig,
    pub decoder_cfg: QwenConfig,
    pub processor: ImageProcessor,
    pub tower: HybridTower,
    pub decoder: Qwen,
    pub tokenizer: QwenBpe,
    /// `<image_placeholder>`'s id.
    pub image_id: u32,
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

impl DeepseekVl {
    /// Load the checkpoint in `dir` with the decoder at `dtype`, its KV cache
    /// sized for `ctx` tokens (prompt, image rows and reply together).
    pub fn load(dir: &Path, dtype: qwen3::Dtype, ctx: u32) -> Result<DeepseekVl, String> {
        let cfg = DeepseekVlConfig::from_dir(dir)?;
        let decoder_cfg = qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?;
        if decoder_cfg.d_model != cfg.aligner.n_embed {
            return Err(format!("the aligner produces {}-wide rows for a {}-wide decoder", cfg.aligner.n_embed, decoder_cfg.d_model));
        }
        let rd = WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        crate::import::check_coverage(&rd, &[crate::import::SAM_PREFIX, clip::import::siglip::DEEPSEEK_VL_LOW_PREFIX, crate::import::ALIGNER_PREFIX, "language_model."])?;
        let tower = HybridTower::load(&rd, &cfg)?;
        let src = qwen3::import::nested_source(&rd, crate::import::DECODER, &decoder_cfg)?;
        let decoder = Qwen::new_shard_dt_decode(decoder_cfg.clone(), ctx, &src, Shard::whole(decoder_cfg.n_layers as usize), dtype);
        drop(src);

        let dir_str = dir.to_str().ok_or("checkpoint path is not UTF-8")?;
        let tokenizer = QwenBpe::from_dir(dir_str)?;
        let tok_cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).map_err(|e| format!("tokenizer_config.json: {e}"))?)
            .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let (bos, eos) = (special(&tok_cfg, "bos_token")?, special(&tok_cfg, "eos_token")?);
        let id = |s: &str| tokenizer.special_id(s).ok_or_else(|| format!("the tokenizer has no {s:?} token"));
        let (image_id, eos_id) = (id(prompt::IMAGE_TAG)?, id(&eos)?);
        id(&bos)?;
        Ok(DeepseekVl { processor: ImageProcessor::from_dir(dir)?, cfg, decoder_cfg, tower, decoder, tokenizer, image_id, bos, eos, eos_id })
    }

    /// The prompt's token ids, BOS first, with one placeholder id per image.
    pub fn prompt_ids(&self, turns: &[Turn]) -> Result<Vec<u32>, String> {
        let text = prompt::render(prompt::SYSTEM_PROMPT, turns, &self.eos)?;
        Ok(data::tokenizer::Tokenizer::encode(&self.tokenizer, &format!("{}{text}", self.bos)))
    }

    /// Every image's aligner rows, concatenated in order.
    pub fn image_embeds(&self, images: &[Rgb8]) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(images.len() * self.tower.rows() * self.decoder_cfg.d_model as usize);
        for img in images {
            out.extend(self.tower.encode(&self.processor.pixel_values(img)?).embeds);
        }
        Ok(out)
    }

    /// Prefill `ids` with `embeds` spliced at the placeholders, then decode
    /// greedily until the end-of-sentence id or `max_new` tokens. `on_token`
    /// sees each id as it is produced.
    pub fn generate_greedy(&self, ids: &[u32], embeds: &[f32], max_new: usize, on_token: &mut dyn FnMut(u32)) -> Result<Vec<u32>, String> {
        let inputs: Vec<PrefillInput> = prompt::splice(ids, self.image_id, embeds, self.tower.rows(), self.decoder_cfg.d_model as usize)?;
        let ctx = self.decoder.ctx_len();
        if inputs.len() + max_new > ctx {
            return Err(format!("{} prompt rows and {max_new} new tokens exceed the {ctx}-token context this model was loaded with", inputs.len()));
        }
        self.decoder.reset_cache();
        self.decoder.prefill(&inputs);
        let mut out = Vec::new();
        for _ in 0..max_new {
            let logits = self.decoder.decode_logits();
            let next = argmax(&logits);
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

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}
