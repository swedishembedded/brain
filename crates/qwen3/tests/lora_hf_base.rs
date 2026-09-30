// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parameter-efficient fine-tuning on
// consumer GPUs for its clients. If your team needs expertise in adapting
// open-weight decoders to your data then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A LoRA fine-tune starts from a `transformers` directory as downloaded and
//! keeps its base in bf16: the base is streamed off the directory straight
//! into the training build (no brain checkpoint is written or read), the
//! base linears land at the bf16 tier, and the adapter learns its data.

use std::path::PathBuf;

use data::binio::{self, Meta};
use model::FitControl;
use qwen3::export::{export_hf, HfDtype, HfExport};
use qwen3::finetune::{finetune_lora_controlled, LoraStart};
use qwen3::{Dtype, QwenConfig};

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-hf-base-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_dataset(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
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

#[test]
fn an_adapter_trains_on_a_bf16_base_read_from_a_transformers_directory() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let scratch = tmp("run");
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
        ..QwenConfig::tiny()
    };
    let base = scratch.join("base");
    export_hf(&qwen3::init_weights(&cfg, 7), &cfg, &base, &HfExport { dtype: HfDtype::Bf16, ..Default::default() }).unwrap();
    write_dataset(&scratch.join("data"));

    let opts = model::FitOpts {
        steps: 300,
        batch_size: 1,
        block_size: 16,
        lr: 5e-2,
        min_lr: 5e-3,
        warmup: 15,
        decay_iters: 300,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 1234,
        ..Default::default()
    };
    let (report, trained) = finetune_lora_controlled(base.to_str().unwrap(), &scratch.join("data"), &opts, 3, 6.0, &LoraStart::Fresh, FitControl::default(), Dtype::BF16).unwrap();
    if trained.linear_dtype() != Some(Dtype::BF16) {
        return; // no bf16 storage path on this device
    }
    assert_eq!(report.steps_completed, 300);
    // The single-batch loss is noisy; what the adapter learned is what it
    // completes the prompt with.
    let vocab = VOCAB as usize;
    let logits = trained.logits_all(&PROMPT);
    let last = &logits[(PROMPT.len() - 1) * vocab..PROMPT.len() * vocab];
    let best = (0..vocab).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap();
    assert_eq!(best as u32, TARGET, "the adapter learned its data");
}
