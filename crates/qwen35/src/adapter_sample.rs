// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for showing that a trained adapter
// changes what a very large model says, without a second copy of the model,
// for its clients. If your team needs expertise in evaluating fine-tuned
// language models you can procure our services by sending an email to
// info@swedishembedded.com.

//! Greedy decoding of a quantised, streamed base with a LoRA adapter overlaid.
//!
//! [`AdapterSampler`] builds the same training-mode model `finetune_streamed`
//! does (frozen base on the device at its tier, adapter factors beside it), so
//! a trained adapter can be used on a base that never exists on the host as a
//! whole - folding into a fp32 copy of the 27B needs more host memory than the
//! job has. It decodes by repeated full forwards at one fixed length, with
//! no KV cache: slow, but it is the exact forward the adapter was trained
//! through, which is what a content check of the adapter needs.

use std::path::Path;

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;

use crate::base_source::{lora_base, BaseDtype};
use crate::model::{pipelines, Qwen35, TrainOpts};

/// Rank and alpha of the zero-delta overlay used when no adapter is named;
/// any shape works, the factors are inert.
const INERT_RANK: u32 = 1;
const INERT_ALPHA: f32 = 2.0;

/// A streamed-base model with an adapter that can be switched off.
pub struct AdapterSampler {
    model: Qwen35,
    /// Tokens every forward runs over (prompt, generated tokens and padding).
    len: u32,
    /// The adapter's `lora_b` factors, kept host side so the adapter can be
    /// switched off (all `lora_b` zero is the base, exactly) and back on.
    lora_b: Vec<(String, Vec<f32>)>,
    enabled: bool,
}

impl AdapterSampler {
    /// Stream `base` onto the device at `dtype` with `adapter` (a file written
    /// by [`crate::lora::save_adapter`]) overlaid, or with an inert zero-delta
    /// overlay when `adapter` is `None`. `len` is the longest prompt plus
    /// generation [`Self::greedy`] will be asked for.
    pub fn open(base: &Path, adapter: Option<&Path>, dtype: BaseDtype, len: u32) -> Result<AdapterSampler, String> {
        let (rank, alpha) = match adapter {
            Some(file) => adapter_shape(file)?,
            None => (INERT_RANK, INERT_ALPHA),
        };
        // seed: only draws lora_a of the inert overlay; an adapter's own factors replace it
        let seed = data::rng::Rng::random().next_u64();
        let (cfg, src) = lora_base(base, rank, alpha, seed, adapter)?;
        let tier = match dtype {
            BaseDtype::F32 => Dtype::F32,
            BaseDtype::Bf16 => Dtype::BF16,
            BaseDtype::Nf4 => Dtype::NF4,
        };
        let opts = TrainOpts { base: TierPolicy::uniform(tier), recompute: dtype.is_frozen_only(), ..TrainOpts::default() };
        let model = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, 1, len, &src, &opts);
        let lora_b = model.param_names().into_iter().filter(|n| n.ends_with(".lora_b")).map(|n| { let w = model.read_weight(&n); (n, w) }).collect();
        Ok(AdapterSampler { model, len, lora_b, enabled: true })
    }

    /// Switch the adapter on or off. Off is the base model exactly: every
    /// `lora_b` factor is zero, so the low-rank correction adds nothing.
    pub fn set_adapter(&mut self, on: bool) {
        if on == self.enabled {
            return;
        }
        for (name, original) in &self.lora_b {
            if on {
                self.model.write_weight(name, original);
            } else {
                self.model.write_weight(name, &vec![0.0; original.len()]);
            }
        }
        self.enabled = on;
    }

    /// Greedy continuation of `prompt` by up to `max_new` tokens, stopping
    /// after the first token in `eos` (which is included). Each token is one
    /// full forward over the built length: tokens past the current position
    /// are padding, which a causal model never reads.
    pub fn greedy(&self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err("greedy: the prompt is empty".into());
        }
        if prompt.len() + max_new > self.len as usize {
            return Err(format!("greedy: prompt {} + max_new {max_new} exceeds the built length {}", prompt.len(), self.len));
        }
        let vocab = self.model.cfg.vocab as usize;
        let mut tokens = prompt.to_vec();
        let mut out = Vec::new();
        for _ in 0..max_new {
            let mut padded = tokens.clone();
            padded.resize(self.len as usize, 0);
            let logits = self.model.logits_all(&padded);
            let at = (tokens.len() - 1) * vocab;
            let next = argmax(&logits[at..at + vocab]);
            out.push(next);
            tokens.push(next);
            if eos.contains(&next) {
                break;
            }
        }
        Ok(out)
    }
}

/// Lowest index of the largest value; a NaN never wins.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] || row[best].is_nan() {
            best = i;
        }
    }
    best as u32
}

/// Rank and alpha an adapter file was trained at, from its card.
fn adapter_shape(file: &Path) -> Result<(u32, f32), String> {
    let name = file.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", file.display()))?;
    let st = checkpoint::st::load_safetensors(name).map_err(|e| format!("{name}: {e}"))?;
    let card = st.card().and_then(|c| c.adapter).ok_or_else(|| format!("{name}: not an adapter file (no adapter card)"))?;
    let rank = card.rank.ok_or_else(|| format!("{name}: the adapter card names no rank"))?;
    let alpha = card.alpha.ok_or_else(|| format!("{name}: the adapter card names no alpha"))?;
    Ok((rank, alpha))
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_takes_the_first_maximum_and_ignores_nan() {
        assert_eq!(argmax(&[0.5, 2.0, 2.0, -1.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 1.0, 0.0]), 1);
    }
}
