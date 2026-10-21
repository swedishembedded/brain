// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in finding why a fine-tune diverges then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The CUDA trainer computes what the CPU trainer computes at the published
//! model's own attention and Gated-DeltaNet geometry (128-wide key and value
//! heads, 16 key and 48 value heads, 256-wide attention heads) over several
//! 64-token chunks, with and without the long-sequence settings. The tiny
//! config's 16- and 20-wide heads over 12 tokens (chunks of 4) exercise none
//! of the kernels those shapes select.

mod common;

use common::*;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::{lora_cfg, Qwen35Config};
use model::saved::{SavedFormat, SavedPolicy};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const ROWS: usize = 320;

/// The 27B's mixer geometry at a width and depth a test can afford.
fn real_geometry() -> Qwen35Config {
    Qwen35Config {
        vocab: 64,
        block_size: ROWS as u32,
        n_layers: 4,
        d_model: 256,
        n_heads: 6,
        n_kv_heads: 1,
        head_dim: 256,
        mrope_section: [11, 11, 10],
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        intermediate_size: 512,
        max_position_embeddings: ROWS as u32,
        lora: Some(lora_cfg(16, 32.0)),
        ..Qwen35Config::tiny()
    }
}

#[test]
fn the_cuda_trainer_matches_the_cpu_at_the_published_mixer_geometry() {
    let Ok(cuda) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip("no CUDA device on this box");
        return;
    };
    let cfg = real_geometry();
    let init = weights(&cfg, 0.05);
    let (x, y) = batch_of(ROWS);
    let long = [
        ("whole sequence", TrainOpts::default()),
        ("recomputed", TrainOpts { saved: SavedPolicy::Recompute, ..TrainOpts::default() }),
        ("recomputed, segmented", TrainOpts { saved: SavedPolicy::Recompute, gdn_segment_tokens: Some(128), ..TrainOpts::default() }),
        ("recomputed, chunked", TrainOpts { saved: SavedPolicy::Recompute, mlp_rows: Some(128), ..TrainOpts::default() }),
        ("segmented, recomputed, chunked", TrainOpts { saved: SavedPolicy::Recompute, gdn_segment_tokens: Some(128), mlp_rows: Some(128), ..TrainOpts::default() }),
        // The settings a 27B run at 15K tokens gets by itself; the bf16 stream rounds identically on both devices
        // up to the order of the fp32 sums that feed the rounding.
        ("long-sequence settings with a bf16 residual stream", TrainOpts { saved: SavedPolicy::Quantized(SavedFormat::Bf16), gdn_segment_tokens: Some(128), mlp_rows: Some(128), ..TrainOpts::default() }),
    ];
    for (name, extra) in long {
        let min_cos = if extra.saved.boundary_format() == SavedFormat::Bf16 { 0.999 } else { 0.99999 };
        let opts = TrainOpts { base: TierPolicy::uniform(Dtype::F32), gemm_precision: GemmPrecision::Fp32Reference, ..extra };
        let on_cpu = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, ROWS as u32, &init, &opts);
        let on_cuda = Qwen35::new_train_opts_on(cuda.share(), cfg.clone(), 1, ROWS as u32, &init, &opts);
        println!("{name}");
        let (got, want) = (loss_and_grads_on(&on_cuda, &x, &y), loss_and_grads_on(&on_cpu, &x, &y));
        println!("loss cuda {} cpu {}", got.0, want.0);
        let mut worst: Vec<(f64, String)> = got.1.iter().zip(&want.1).filter(|(_, (_, w))| w.iter().any(|x| *x != 0.0)).map(|((n, g), (_, w))| (cosine(g, w), n.clone())).collect();
        worst.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (c, n) in worst.iter().take(8) {
            println!("  cos {c:.6} {n}");
        }
        assert_same_training_signal(got, want, 1e-3, min_cos);
    }
}
