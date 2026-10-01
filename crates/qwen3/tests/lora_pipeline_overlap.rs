// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-GPU fine-tuning of large language
// models for its clients. If your team needs expertise in keeping several
// cards busy while a model that does not fit one trains then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A step that accumulates several micro-batches runs them through the
//! pipeline's stages concurrently (the stages work on neighbouring
//! micro-batches at once) instead of one micro-batch after another, and ends
//! where the sequential schedule would: the same loss and the same adapters.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use model::{Model, Pipeline, PipelineModel};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    let have = gpu_core::discrete_gpu_count();
    if have < 2 {
        brain_testutil::skip_unavailable(&format!("needs 2 discrete GPU(s), found {have}"));
        return true;
    }
    false
}

const VOCAB: u32 = 24;

fn dataset(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-overlap-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (mut tokens, mut mask) = (Vec::new(), Vec::new());
    for i in 0..300u32 {
        tokens.extend_from_slice(&[2 + i % 5, 5, 8, 15 - i % 3]);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();
    dir
}

fn opts() -> model::FitOpts {
    model::FitOpts { steps: 12, batch_size: 1, block_size: 16, grad_accum: 3, lr: 2e-2, min_lr: 2e-3, warmup: 2, decay_iters: 12, weight_decay: 0.0, grad_clip: 1.0, eval_interval: 0, eval_batches: 0, seed: 99, ..Default::default() }
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

#[test]
fn accumulated_micro_batches_overlap_across_stages_and_train_the_same_adapters() {
    if gpu_disabled() {
        return;
    }
    let scratch = dataset("run");
    let cfg = QwenConfig {
        vocab: VOCAB,
        block_size: 16,
        n_layers: 4,
        d_model: 16,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_ff: 32,
        max_position_embeddings: 16,
        lora: Some(LoraCfg::attn(3, 6.0)),
        ..QwenConfig::tiny()
    };
    let init = qwen3::init_weights(&cfg, 7);
    let opts = opts();
    let single = Qwen::new_lora_dt(cfg.clone(), 1, 16, &init, Dtype::F32);
    let (want_report, want) = model::fit_controlled(single, data::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();

    let pipe = PipelineModel::new(Pipeline::<Qwen>::new_dt(cfg.clone(), 1, 16, &init, &[0, 1], Dtype::F32), cfg.clone());
    let (got_report, got) = model::fit_controlled(pipe, data::<PipelineModel<Qwen>>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();

    assert_eq!(got.overlapped_steps(), opts.steps as u64, "every step ran its three micro-batches through the stages concurrently");
    let (a, b) = (got_report.final_loss.unwrap(), want_report.final_loss.unwrap());
    assert!((a - b).abs() < 2e-3 * b.abs().max(1.0), "final loss {a} vs {b}");
    let (got, want) = (adapters(&got), adapters(&want));
    assert_eq!(got.len(), want.len());
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        let scale = w.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let worst = g.iter().zip(w).fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / scale));
        assert!(worst < 5e-2, "{name}: the adapter differs by {worst}");
    }
}
