// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training smokes for the Qwen3-TTS Talker: from-scratch overfit and a
//! single-speaker LoRA fine-tune on the synthetic `text->codes` dataset
//! (`data::gen_tts`). Both run on the CPU backend (no real checkpoint needed):
//!   BRAIN_DEVICE=cpu cargo test -p brain-tts --test sft_train -- --nocapture
//!
//! Gated by `MOE_SKIP_GPU_TESTS` (they JIT the WGSL kernels and take a few
//! seconds) so the shared machine can skip them.

use std::path::PathBuf;

use data::gen_tts::TtsGenConfig;
use qwen3tts::{FinetuneOpts, TalkerConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("tts_sft_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Tiny dataset: small vocab so a tiny decoder can overfit the text->codes rule.
fn gen_cfg() -> TtsGenConfig {
    TtsGenConfig { n_text: 8, n_code: 16, frames: 8, examples: 1200 }
}

/// From-scratch: a tiny Talker decoder trained on `text->codes` must drop its
/// loss substantially (it can essentially memorise the deterministic rule).
#[test]
fn talker_from_scratch_overfits() {
    if skip() {
        eprintln!("skip: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let dir = tmp("scratch_data");
    let cfg = gen_cfg();
    data::gen_tts::write(&dir, cfg, 7).unwrap();

    let block = 16u32;
    let qcfg = TalkerConfig::tiny().to_qwen(block);
    let out = tmp("scratch_out").join("talker.safetensors");
    let opts = model::train::FitOpts {
        steps: 400,
        batch_size: 32,
        block_size: block,
        lr: 3e-3,
        min_lr: 3e-4,
        warmup: 40,
        decay_iters: 400,
        eval_interval: 0,
        seed: 1234,
        ..Default::default()
    };
    let (initial, final_loss) =
        model::train::fit::<qwen3::Qwen>(&dir, qcfg, &opts, Some(&out)).expect("fit");
    eprintln!("from-scratch: initial {initial:.4} -> final {final_loss:.4}");
    assert!(final_loss < initial * 0.6, "loss did not drop enough: {initial:.3} -> {final_loss:.3}");
    assert!(out.exists(), "checkpoint not written");
}

/// LoRA fine-tune: train a base from scratch, then adapt it with attention LoRA
/// on the same task. Only the adapters train; the loss must still decrease.
#[test]
fn talker_lora_finetune_decreases_loss() {
    if skip() {
        eprintln!("skip: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let dir = tmp("lora_data");
    let cfg = gen_cfg();
    data::gen_tts::write(&dir, cfg, 11).unwrap();

    let block = 16u32;
    // A barely-trained base (high loss), leaving the adapters plenty of headroom.
    let base = tmp("lora_base").join("talker.safetensors");
    let base_opts = model::train::FitOpts {
        steps: 25,
        batch_size: 32,
        block_size: block,
        lr: 1e-3,
        eval_interval: 0,
        seed: 5,
        ..Default::default()
    };
    model::train::fit::<qwen3::Qwen>(&dir, TalkerConfig::tiny().to_qwen(block), &base_opts, Some(&base))
        .expect("base fit");

    let out = tmp("lora_out").join("talker_lora.safetensors");
    // rank = d_model (16): a full-capacity attention adapter so the decrease is
    // unambiguous even with the embeddings/head frozen.
    let fopts = FinetuneOpts { steps: 500, batch: 32, block, lr: 5e-3, rank: 16, alpha: 16.0, seed: 9 };
    let (initial, final_loss) =
        qwen3tts::sft::finetune_lora(base.to_str().unwrap(), &dir, out.to_str().unwrap(), &fopts).expect("finetune");
    eprintln!("lora finetune: initial {initial:.4} -> final {final_loss:.4}");
    assert!(final_loss < initial - 0.1, "LoRA finetune did not reduce loss: {initial:.3} -> {final_loss:.3}");
    assert!(out.exists(), "adapter checkpoint not written");
}
