// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One function pair per registered architecture - see `registry()`'s doc for
//! why this list is short and how it grows.
//!
//! Every `price_*` builds a WHOLE, single-shard, batch-1, inference-fp32
//! model at zero-init weights (never real weight bytes - `Model::cost_fwd`'s
//! cost is a function of shape alone, the same fact that lets `brain perf flops`
//! price a 4B model without holding it) and reports one forward pass. This is
//! deliberately narrower than `brain perf flops`'s own CLI flags (no `--train`,
//! `--i8`, `--stages`, `--weights`) - `models list`/`models profile` want ONE
//! representative number per model, not the full sweep a human driving `brain
//! flops` by hand might want.

use std::time::Instant;

use gpu_core::cost::{CostReport, Recording};
use model::{Shard, Shardable};
use serde_json::Value;

use crate::{CostSummary, Measurement};

fn synthetic_tokens(n: usize, vocab: u32) -> Vec<u32> {
    (0..n).map(|i| i as u32 % vocab.max(1)).collect()
}

/// Device bytes live on `gpu` right now; `None` where the backend does not
/// account for its allocations (never a fabricated zero).
///
/// Every model builds its own `Gpu` (no constructor takes a device), so that
/// device's tally holds this model and nothing else: reading it right after
/// the build IS "bytes held by the model", with no other allocation to
/// subtract.
fn live_bytes(gpu: &gpu_core::Gpu) -> Option<u64> {
    gpu.memory_usage().map(|u| u.live)
}

/// The peak of live device bytes since `gpu.reset_memory_peak()`; `None` as
/// [`live_bytes`].
fn peak_bytes(gpu: &gpu_core::Gpu) -> Option<u64> {
    gpu.memory_usage().map(|u| u.peak_live)
}

/// Cost of exactly ONE repeated unit (a transformer layer) in a HOMOGENEOUS
/// stack, derived from dry probes at 0 and 1 units and verified affine at 2 -
/// the single-axis case of the same block-depth derivation
/// `crates/cli/src/flops_cli.rs::affine_block_cost` uses for flux2/ltxv's
/// multi-axis (double/single block) graphs. `probe(n)` must build and cost a
/// graph of exactly `n` units; `0` must be a buildable depth (embed+head with
/// no layers).
fn per_unit_cost(mut probe: impl FnMut(usize) -> CostReport) -> Result<CostReport, String> {
    let base = probe(0);
    let one = probe(1);
    let delta = one.checked_sub(&base).ok_or_else(|| "the +1-unit graph does not contain the 0-unit graph; per-layer cost is not derivable".to_string())?;
    let mut predicted = base.clone();
    predicted.merge(&delta.scaled(2));
    let two = probe(2);
    if predicted.total != two.total || predicted.steps != two.steps {
        return Err(format!(
            "block-depth linearity check failed at 2 units: predicted {:?} ({} steps), measured {:?} ({} steps) - this architecture's forward pass is not affine in layer count",
            predicted.total, predicted.steps, two.total, two.steps
        ));
    }
    Ok(delta)
}

pub(crate) fn price_qwen3(config: &Value) -> Result<CostReport, String> {
    let cfg = qwen3::QwenConfig::from_json_checked(config)?;
    let init = qwen3::init_weights(&cfg, 0);
    let shard = Shard::whole(cfg.n_layers as usize);
    let m = qwen3::Qwen::new_shard(cfg.clone(), 1, cfg.block_size, &init, false, shard);
    Ok(m.cost_fwd())
}

pub(crate) fn manifest_qwen3(config: &Value) -> Result<Vec<(String, usize)>, String> {
    Ok(qwen3::QwenConfig::from_json_checked(config)?.param_list())
}

pub(crate) fn measure_qwen3(config: &Value, reps: usize) -> Result<Measurement, String> {
    let cfg = qwen3::QwenConfig::from_json_checked(config)?;
    let init = qwen3::init_weights(&cfg, 0);

    let per_layer = per_unit_cost(|n| {
        let shard = Shard { start: 0, end: n, embed: true, head: true, gpu_index: Shard::ANY_GPU };
        qwen3::Qwen::new_shard(cfg.clone(), 1, cfg.block_size, &init, false, shard).cost_fwd()
    })?;

    let load_start = Instant::now();
    let m = qwen3::Qwen::new_shard(cfg.clone(), 1, cfg.block_size, &init, false, Shard::whole(cfg.n_layers as usize));
    let x = synthetic_tokens(cfg.block_size as usize, cfg.vocab);
    m.set_batch(&x, &x);
    let load_seconds = load_start.elapsed().as_secs_f64();
    let held_bytes = live_bytes(m.gpu());
    m.gpu().reset_memory_peak();

    let t0 = Instant::now();
    let _ = m.run_forward_stage();
    m.poll_wait();
    let cold_seconds = t0.elapsed().as_secs_f64();

    let mut hot_seconds = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        let _ = m.run_forward_stage();
        m.poll_wait();
        hot_seconds = hot_seconds.min(t.elapsed().as_secs_f64());
    }

    let total = CostSummary::from(&m.cost_fwd());
    Ok(Measurement { load_seconds, cold_seconds, hot_seconds, total, per_layer: CostSummary::from(&per_layer), held_bytes, peak_bytes: peak_bytes(m.gpu()) })
}

pub(crate) fn price_gpt2(config: &Value) -> Result<CostReport, String> {
    let cfg = gpt2::GptConfig::from_json_checked(config)?;
    let init = gpt2::init_weights(&cfg, 0);
    let shard = Shard::whole(cfg.n_layers as usize);
    let m = gpt2::Gpt::new_shard(cfg.clone(), 1, cfg.block_size, &init, shard);
    Ok(m.cost_fwd())
}

pub(crate) fn manifest_gpt2(config: &Value) -> Result<Vec<(String, usize)>, String> {
    Ok(gpt2::GptConfig::from_json_checked(config)?.param_list())
}

pub(crate) fn measure_gpt2(config: &Value, reps: usize) -> Result<Measurement, String> {
    let cfg = gpt2::GptConfig::from_json_checked(config)?;
    let init = gpt2::init_weights(&cfg, 0);

    let per_layer = per_unit_cost(|n| {
        let shard = Shard { start: 0, end: n, embed: true, head: true, gpu_index: Shard::ANY_GPU };
        gpt2::Gpt::new_shard(cfg.clone(), 1, cfg.block_size, &init, shard).cost_fwd()
    })?;

    let load_start = Instant::now();
    let m = gpt2::Gpt::new_shard(cfg.clone(), 1, cfg.block_size, &init, Shard::whole(cfg.n_layers as usize));
    let x = synthetic_tokens(cfg.block_size as usize, cfg.vocab);
    m.set_batch(&x, &x);
    let load_seconds = load_start.elapsed().as_secs_f64();
    let held_bytes = live_bytes(&m.gpu);
    m.gpu.reset_memory_peak();

    let t0 = Instant::now();
    let _ = m.run_forward_stage();
    m.poll_wait();
    let cold_seconds = t0.elapsed().as_secs_f64();

    let mut hot_seconds = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        let _ = m.run_forward_stage();
        m.poll_wait();
        hot_seconds = hot_seconds.min(t.elapsed().as_secs_f64());
    }

    let total = CostSummary::from(&m.cost_fwd());
    Ok(Measurement { load_seconds, cold_seconds, hot_seconds, total, per_layer: CostSummary::from(&per_layer), held_bytes, peak_bytes: peak_bytes(&m.gpu) })
}

pub(crate) fn price_lfm2(config: &Value) -> Result<CostReport, String> {
    let cfg = lfm2::config::LfmConfig::from_json_checked(config)?;
    let init = lfm2::init::init_weights(&cfg, 0);
    let t = cfg.block_size;
    let m = lfm2::model::Lfm::new(cfg, 1, t, &init);
    Ok(m.cost_fwd())
}

pub(crate) fn manifest_lfm2(config: &Value) -> Result<Vec<(String, usize)>, String> {
    Ok(lfm2::config::LfmConfig::from_json_checked(config)?.param_list())
}

pub(crate) fn measure_lfm2(config: &Value, reps: usize) -> Result<Measurement, String> {
    let cfg = lfm2::config::LfmConfig::from_json_checked(config)?;
    let n_layers = cfg.layer_types.len().max(1);
    let init = lfm2::init::init_weights(&cfg, 0);
    let t = cfg.block_size;

    let load_start = Instant::now();
    let m = lfm2::model::Lfm::new(cfg.clone(), 1, t, &init);
    let x = synthetic_tokens(t as usize, cfg.vocab);
    m.set_batch(&x, &x);
    let load_seconds = load_start.elapsed().as_secs_f64();
    let held_bytes = live_bytes(&m.gpu);
    m.gpu.reset_memory_peak();

    let t0 = Instant::now();
    m.forward();
    m.poll_wait();
    let cold_seconds = t0.elapsed().as_secs_f64();

    let mut hot_seconds = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let ti = Instant::now();
        m.forward();
        m.poll_wait();
        hot_seconds = hot_seconds.min(ti.elapsed().as_secs_f64());
    }

    let total = CostSummary::from(&m.cost_fwd());
    // LFM2.5 is a HYBRID stack (per-layer choice of gated short-conv vs
    // bidirectional GQA attention, from the checkpoint's own `layer_types` -
    // see that field's doc), not a uniform one like qwen3/gpt2 - a conv
    // layer and an attention layer cost genuinely different amounts, so
    // there is no single "the" per-layer cost to DERIVE the way
    // `per_unit_cost`'s 0/1/2-probe affine check does for a homogeneous
    // stack (probing "N layers" for a hybrid model would silently mix which
    // TYPE of layer got added, which `per_unit_cost`'s own linearity check
    // would - correctly - refuse to trust). This is the average per layer
    // (`total / n_layers`), not a derived exact single-layer-type cost -
    // named as an average, not claimed as more precise than it is.
    let per_layer = CostSummary {
        flops: total.flops / n_layers as u64,
        int_ops: total.int_ops / n_layers as u64,
        bytes: total.bytes / n_layers as u64,
        coverage: total.coverage,
    };
    Ok(Measurement { load_seconds, cold_seconds, hot_seconds, total, per_layer, held_bytes, peak_bytes: peak_bytes(&m.gpu) })
}

/// `qwen35` is hybrid (gated delta-net and full-attention layers), so like
/// lfm2 it has no single per-layer cost to derive; it is priced whole.
///
/// Its forward pass has no recorded step list to walk (`cost_fwd` is a
/// property of the qwen3-style graph builders), so the pass is run under a dry
/// `Recording`: every dispatch is tallied and none reaches the device.
pub(crate) fn price_qwen35(config: &Value) -> Result<CostReport, String> {
    let cfg = qwen35::config::Qwen35Config::from_json_checked(config)?;
    let init = qwen35::init::init_weights(&cfg, 0);
    let t = cfg.block_size;
    let m = qwen35::Qwen35::new_on(gpu_core::Gpu::new(qwen35::model::pipelines()), cfg.clone(), 1, t, &init);
    let x = synthetic_tokens(t as usize, cfg.vocab);
    m.set_batch(&x, &x);
    let rec = Recording::dry();
    let _ = m.forward();
    Ok(rec.take())
}

pub(crate) fn manifest_qwen35(config: &Value) -> Result<Vec<(String, usize)>, String> {
    Ok(qwen35::config::Qwen35Config::from_json_checked(config)?.param_list())
}

/// [`price_qwen35`] for the mixture-of-experts sibling. Which experts a token
/// visits is data a dry pass does not have; the dispatch count and shapes are
/// fixed by the config regardless. `gpu_core::cost` has no formula yet for the
/// grouped-expert matmul and routing kernels, so they show as uncovered in the
/// report and the total understates the expert work (coverage says by how much).
pub(crate) fn price_qwen35moe(config: &Value) -> Result<CostReport, String> {
    let cfg = qwen35moe::config::Qwen35Config::from_json_checked(config)?;
    let init = qwen35moe::init::init_weights(&cfg, 0);
    let t = cfg.block_size;
    let m = qwen35moe::Qwen35::new(cfg.clone(), 1, t, &init);
    let x = synthetic_tokens(t as usize, cfg.vocab);
    m.set_batch(&x, &x);
    let rec = Recording::dry();
    let _ = m.forward();
    Ok(rec.take())
}

pub(crate) fn manifest_qwen35moe(config: &Value) -> Result<Vec<(String, usize)>, String> {
    Ok(qwen35moe::config::Qwen35Config::from_json_checked(config)?.param_list())
}

/// Tokens in the one forward pass [`price_gemma4`] prices. A Gemma-4 config
/// carries no context length (the text tower runs at whatever length the
/// encoder is handed), so the pass is a fixed 16 tokens: comparable between
/// configs, to be scaled by the real prompt length for a particular request.
pub(crate) const GEMMA4_PRICED_TOKENS: usize = 16;

pub(crate) fn price_gemma4(config: &Value) -> Result<CostReport, String> {
    let cfg = gemma4::Gemma4Config::from_json_checked(config)?;
    // The text tower reads only the embedding, the final norm and the layers;
    // the aggregate-embed projections are LTX's own host-side head.
    let weights: gemma4::block::Tensors = gemma4::gemma4_tensor_manifest(&cfg)
        .into_iter()
        .filter(|(name, _)| !name.starts_with("text_embedding_projection."))
        .map(|(name, shape)| {
            let n = shape.iter().product();
            (name, (shape, vec![0.0f32; n]))
        })
        .collect();
    let m = gemma4::Gemma4Model::new(cfg, weights, Some("cpu"));
    let tokens = synthetic_tokens(GEMMA4_PRICED_TOKENS, cfg.vocab_size);
    let rec = Recording::dry();
    let _ = m.forward(&tokens);
    Ok(rec.take())
}

pub(crate) fn manifest_gemma4(config: &Value) -> Result<Vec<(String, usize)>, String> {
    let cfg = gemma4::Gemma4Config::from_json_checked(config)?;
    Ok(gemma4::gemma4_tensor_manifest(&cfg).into_iter().map(|(name, shape)| (name, shape.iter().product())).collect())
}

pub(crate) fn price_minimaxh3(config: &Value) -> Result<CostReport, String> {
    minimaxh3::cost::forward_cost(&minimaxh3::config::H3TransformerConfig::from_json_checked(config)?)
}

pub(crate) fn manifest_minimaxh3(config: &Value) -> Result<Vec<(String, usize)>, String> {
    let cfg = minimaxh3::config::H3TransformerConfig::from_json_checked(config)?;
    Ok(minimaxh3::model::dit_tensor_manifest(&cfg).into_iter().map(|(name, shape)| (name, shape.iter().product())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(flops: u64) -> CostReport {
        let mut r = CostReport { total: gpu_core::cost::Cost { flops, int_ops: 0, bytes: 0 }, steps: 1, covered: 1, ..CostReport::default() };
        r.by_kernel.insert("probe".to_string(), gpu_core::cost::KernelCost { calls: 1, cost: r.total });
        r
    }

    #[test]
    fn per_unit_cost_recovers_a_real_linear_series() {
        // 100, 130, 160 - base 100, +30 per unit, genuinely affine.
        let delta = per_unit_cost(|n| flat(100 + 30 * n as u64)).expect("a real linear series must be accepted");
        assert_eq!(delta.total.flops, 30);
    }

    #[test]
    fn per_unit_cost_refuses_a_non_affine_series() {
        // 100, 130, 300 - the jump from 1->2 units is not double the 0->1
        // delta, so this is NOT a uniform repeated-layer stack and must be
        // refused rather than silently reporting a wrong per-layer number.
        let err = per_unit_cost(|n| flat(if n == 2 { 300 } else { 100 + 30 * n as u64 })).expect_err("a non-affine series must be refused, not silently trusted");
        assert!(err.contains("linearity"));
    }
}
