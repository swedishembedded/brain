// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MLM (encoder) evaluation — pseudo-perplexity + masked-token accuracy on a
//! val split, mirroring the [`crate::gpt_val_perplexity`] discipline: fixed
//! input distribution, deterministic corruption, metric separated from task.

use std::path::Path;

use data::mlm::MlmConfig;

/// Evaluate an LFM checkpoint on a val token split: returns
/// `(pseudo_perplexity, masked_accuracy, n_masked)`.
pub fn lfm_mlm_eval(
    weights: &str,
    data_dir: &Path,
    batches: u32,
    b: u32,
    t: u32,
    mlm: &MlmConfig,
    seed: u64,
) -> std::io::Result<(f32, f32, usize)> {
    let val = data::binio::read_tokens_u32(&data_dir.join("val"))?;
    let model = lfm2::Lfm::load_inference(weights, b, t);
    let loss = lfm2::train::mlm_val_loss(&model, &val, mlm, batches, b, t, seed);
    let (acc, n) = lfm2::train::mlm_masked_accuracy(&model, &val, mlm, b, t, seed ^ 0xacc);
    Ok((loss.exp(), acc, n))
}
