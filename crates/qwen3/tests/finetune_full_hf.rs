// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of open-weight language models
// for its clients. If your team needs expertise in adapting a pretrained
// decoder on your own hardware then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A full-parameter fine-tune starts from a `transformers` directory as
//! downloaded: the base is streamed into the training build (no brain
//! checkpoint is written first), every weight trains, and what comes out is
//! one brain checkpoint that reloads as the trained model.

use std::path::PathBuf;

use data::binio::{self, Meta};
use qwen3::export::{export_hf, HfDtype, HfExport};
use qwen3::finetune::{finetune, Mode};
use qwen3::{Qwen, QwenConfig};

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-finetune-full-hf-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_full_fine_tune_of_a_transformers_directory_trains_every_weight_and_saves_one_checkpoint() {
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
    let init = qwen3::init_weights(&cfg, 7);
    export_hf(&init, &cfg, &base, &HfExport { dtype: HfDtype::F32, ..Default::default() }).unwrap();

    let data = scratch.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for _ in 0..200 {
        tokens.extend_from_slice(&PROMPT);
        tokens.push(TARGET);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&data.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&data.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&data.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&data.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(data.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();

    let opts = model::FitOpts {
        steps: 120,
        batch_size: 1,
        block_size: 16,
        lr: 3e-3,
        min_lr: 3e-4,
        warmup: 6,
        decay_iters: 120,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: 1234,
        ..Default::default()
    };
    let out = scratch.join("tuned.safetensors");
    let (first, last) = finetune(base.to_str().unwrap(), &data, &opts, &Mode::FullOffload, out.to_str().unwrap()).unwrap();
    assert!(last < first, "the loss fell: {first} -> {last}");

    // One brain checkpoint, which reloads as the trained model and now
    // completes the prompt with the target.
    assert!(out.is_file(), "the trained checkpoint is written");
    let tuned = Qwen::load_inference(out.to_str().unwrap(), 1, 8);
    let logits = tuned.logits_all(&PROMPT);
    let last_row = &logits[(PROMPT.len() - 1) * VOCAB as usize..PROMPT.len() * VOCAB as usize];
    let best = (0..VOCAB as usize).max_by(|&a, &b| last_row[a].total_cmp(&last_row[b])).unwrap();
    assert_eq!(best as u32, TARGET, "every weight trained toward the data");
    assert!(std::fs::read_dir(&base).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().contains("brain")), "nothing was written into the base directory");
}
