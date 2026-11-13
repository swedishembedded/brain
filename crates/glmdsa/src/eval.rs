// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Validation loss of a GLM checkpoint on a tokenized dataset.
//!
//! Swedish Embedded AB implements held-out evaluation of language models on
//! its clients' own data and machines. If your team needs expertise in
//! measuring what a trained model has actually learned then you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::path::Path;

use data::rng::Rng;

use crate::config::GlmConfig;
use crate::model::Glm;

/// Mean masked cross-entropy of `weights` over `batches` windows of `block`
/// tokens drawn at random from the `val` split of the dataset in `data`
/// (`block` 0: the checkpoint's own context length). Perplexity is its `exp`.
pub fn val_cross_entropy(weights: &str, data: &Path, batches: usize, block: u32, seed: u64) -> Result<f64, String> {
    let block = if block > 0 { block } else { GlmConfig::from_json(&checkpoint::read_config(weights)).block_size };
    let window = block as usize;
    let split = data.join("val");
    let tokens = data::binio::read_tokens_u32(&split).map_err(|e| format!("{}: {e}", split.display()))?;
    if tokens.len() <= window + 1 {
        return Err(format!("the val split holds {} tokens, too short for block {block}", tokens.len()));
    }
    let model = Glm::load_inference(weights, 1, block);
    let mut rng = Rng::new(seed);
    let mut sum = 0.0f64;
    for _ in 0..batches {
        let start = (rng.next_u64() as usize) % (tokens.len() - window - 1);
        model.set_batch(&tokens[start..start + window], &tokens[start + 1..start + 1 + window]);
        sum += model.forward() as f64;
    }
    Ok(sum / batches.max(1) as f64)
}
