// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-GPU fine-tuning of large language
// models for its clients. If your team needs expertise in training models
// that do not fit one card then you can procure our services by sending an
// email to info@swedishembedded.com.

//! A LoRA model split across cards trains through the same fit loop as one
//! card, and learns the same adapters: two stages (the fp32 build, and the
//! bf16 frozen base) end a short run at the single-card model's loss, with
//! the adapters it trained close to the single card's.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use model::{Model, Pipeline, PipelineModel};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn stage_gpus() -> Vec<usize> {
    std::env::var("SHARD_TEST_GPUS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0, 1])
}

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    let need = stage_gpus().iter().copied().max().unwrap_or(0) + 1;
    let have = gpu_core::discrete_gpu_count();
    if have < need {
        brain_testutil::skip_unavailable(&format!("needs {need} discrete GPU(s), found {have}"));
        return true;
    }
    false
}

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-pipeline-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_dataset(dir: &Path) {
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for _ in 0..200 {
        tokens.extend_from_slice(&PROMPT);
        tokens.push(TARGET);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();
}

fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn opts() -> model::FitOpts {
    model::FitOpts {
        steps: 40,
        batch_size: 1,
        block_size: 16,
        lr: 2e-2,
        min_lr: 2e-3,
        warmup: 4,
        decay_iters: 40,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 1234,
        ..Default::default()
    }
}

fn data<M: Model>(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<M> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<M>(train, val, bcfg, itos)
}

fn adapters(m: &impl Model) -> Vec<(String, Vec<f32>)> {
    let mut names: Vec<String> = m.param_names().into_iter().filter(|n| n.contains(".lora_")).collect();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b)).max(1e-12)
}

#[test]
fn a_pipeline_trains_the_adapters_the_single_card_does() {
    if gpu_disabled() {
        return;
    }
    let scratch = tmp("run");
    write_dataset(&scratch);
    let cfg = QwenConfig {
        vocab: VOCAB,
        block_size: 16,
        n_layers: 2,
        d_model: 16,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_ff: 32,
        max_position_embeddings: 16,
        lora: Some(LoraCfg::attn(3, 6.0)),
        ..QwenConfig::tiny()
    };
    let mut init = qwen3::init_weights(&cfg, 7);
    for v in init.iter_mut().filter(|(n, _)| !n.contains(".lora_")).map(|(_, v)| v) {
        v.iter_mut().for_each(|x| *x = round_bf16(*x));
    }
    for dt in [Dtype::F32, Dtype::BF16] {
        let (opts, (b, t)) = (opts(), (1u32, 16u32));
        let single = Qwen::new_lora_dt(cfg.clone(), b, t, &init, dt);
        if single.linear_dtype() != Some(dt) {
            continue; // no such storage tier on this device
        }
        let (want_report, want) = model::fit_controlled(single, data::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
        let pipe = PipelineModel::new(Pipeline::<Qwen>::new_dt(cfg.clone(), b, t, &init, &stage_gpus(), dt), cfg.clone());
        let (got_report, got) = model::fit_controlled(pipe, data::<PipelineModel<Qwen>>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();

        let (w, g) = (want_report.final_loss.unwrap(), got_report.final_loss.unwrap());
        assert!((w - g).abs() < 2e-2 * w.abs().max(1.0), "{dt:?}: final loss {g} vs single-card {w}");
        assert!(got_report.final_loss.unwrap() < got_report.initial_loss, "{dt:?}: the pipeline learned");
        let (want_a, got_a) = (adapters(&want), adapters(&got));
        assert_eq!(want_a.len(), got_a.len());
        assert!(!want_a.is_empty());
        for ((name, a), (_, b)) in want_a.iter().zip(&got_a) {
            if a.iter().all(|x| *x == 0.0) && b.iter().all(|x| *x == 0.0) {
                continue;
            }
            let c = cosine(a, b);
            assert!(c > 0.99, "{dt:?} {name}: trained adapter cosine {c}");
        }
    }
}

/// The hooks DPO, GRPO and distillation drive - per-token log-probabilities
/// after the forward, then per-position loss weights before the backward -
/// reach the head stage of a pipeline, so a weighted step on two stages is
/// the single card's: the same log-probabilities, loss and adapter gradients.
#[test]
fn a_weighted_loss_step_on_a_pipeline_is_the_single_cards() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig { block_size: 12, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    let mut init = qwen3::init_weights(&cfg, 7);
    for (n, v) in init.iter_mut().filter(|(n, _)| n.ends_with(".lora_b")) {
        v.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 3 + n.len()) % 7) as f32 * 0.04 - 0.1);
    }
    let tokens: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let targets: Vec<u32> = (0..12).map(|i| (i * 5 + 6) % 23).collect();
    let weights: Vec<f32> = (0..12).map(|i| if i % 3 == 0 { 0.0 } else { 0.5 + i as f32 * 0.1 }).collect();

    let mut single = Qwen::new(cfg.clone(), 1, 12, &init);
    Model::enable_weighted_loss(&mut single);
    Model::set_batch(&single, model::Batch::Lm { tokens: &tokens, targets: &targets });
    Model::zero_grads(&single);
    let want_loss = Model::forward(&single);
    let want_lp = Model::batch_token_logprobs(&single).unwrap();
    Model::set_loss_weights(&single, &weights);
    Model::backward(&single);

    let mut pipe = PipelineModel::new(Pipeline::<Qwen>::new_dt(cfg.clone(), 1, 12, &init, &stage_gpus(), Dtype::F32), cfg.clone());
    Model::enable_weighted_loss(&mut pipe);
    Model::set_batch(&pipe, model::Batch::Lm { tokens: &tokens, targets: &targets });
    Model::zero_grads(&pipe);
    let loss = Model::forward(&pipe);
    let lp = Model::batch_token_logprobs(&pipe).expect("the head stage's log-probabilities");
    Model::set_loss_weights(&pipe, &weights);
    Model::backward(&pipe);

    assert!((loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0), "loss {loss} vs {want_loss}");
    for (a, b) in lp.iter().zip(&want_lp) {
        assert!((a - b).abs() < 1e-4, "log-prob {a} vs {b}");
    }
    let mut compared = 0;
    for name in adapters(&single).into_iter().map(|(n, _)| n) {
        let (a, b) = (Model::read_grad(&single, &name), Model::read_grad(&pipe, &name));
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let worst = a.iter().zip(&b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / scale));
        assert!(worst < 1e-3, "{name}: weighted gradient differs by {worst}");
        compared += 1;
    }
    assert!(compared >= 4);
}
