// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prefill from caller-supplied input embeddings (`Qwen35::prefill_embeds`),
//! the inference seam a vision-language wrapper uses to put image rows into
//! the KV cache and the Gated-DeltaNet state:
//!
//! 1. embeddings equal to the token gather ([`Qwen35::embed_tokens`])
//!    reproduce the token prefill and the decode that follows it;
//! 2. embeddings with some rows replaced reproduce the training forward with
//!    those rows spliced in (`enable_mm_splices`) - the same model, reached
//!    through the two paths a VLM uses for inference and for training.
//!
//! Rounds are smaller than the prompt and the replaced runs straddle round
//! boundaries, so the state threaded between rounds is exercised.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn head(head_w: &[f32], hidden: &[f32]) -> Vec<f32> {
    head_w.chunks_exact(hidden.len()).map(|r| r.iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
}

fn run(gpu: Gpu) {
    let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny() };
    let t = cfg.block_size;
    let d = cfg.d_model as usize;
    let init = qwen35::init::init_weights(&cfg, 11);
    let tokens: Vec<u32> = (0..t).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let prompt = &tokens[..t as usize - 4];
    let tail = &tokens[t as usize - 4..];

    let m = Qwen35::new_on_dt(gpu.share(), cfg.clone(), 1, t, &init, &TierPolicy::uniform(Dtype::F32));
    let head_w = m.read_weight(cfg.head_weight());

    // 1. Embeddings equal to the gather: the same prefill and decode.
    m.reset_decode_cache();
    let mut want = vec![m.prefill_chunked(prompt, 5)];
    want.extend(tail.iter().map(|&tok| m.step(tok)));
    let embeds = m.embed_tokens(prompt);
    assert_eq!(embeds.len(), prompt.len() * d);
    m.reset_decode_cache();
    let mut got = vec![m.prefill_embeds(&embeds, 5)];
    got.extend(tail.iter().map(|&tok| m.step(tok)));
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let err = maxabs(g, w);
        assert!(err < 1e-5, "row {i}: embedding prefill differs from token prefill by {err}");
    }

    // 2. Rows replaced: the training forward with the same rows spliced in.
    let runs = [(3u32, 4u32), (9, 2)];
    let mut replaced = embeds.clone();
    let mut img = Vec::new();
    for &(row0, n) in &runs {
        for r in row0..row0 + n {
            let row: Vec<f32> = (0..d).map(|k| ((r as usize * 31 + k * 7) % 17) as f32 / 17.0 - 0.5).collect();
            replaced[r as usize * d..(r as usize + 1) * d].copy_from_slice(&row);
            img.extend(row);
        }
    }
    m.reset_decode_cache();
    let last = m.prefill_embeds(&replaced, 5);
    let got_logits = head(&head_w, &last);

    let mut tm = Qwen35::new_train_on(gpu, cfg.clone(), 1, prompt.len() as u32, &init);
    tm.enable_mm_splices(&runs);
    tm.write_img_embeds(&img);
    let all = tm.logits_all(prompt);
    let v = cfg.vocab as usize;
    let want_logits = &all[(prompt.len() - 1) * v..];
    let err = maxabs(&got_logits, want_logits);
    assert!(err < 2e-3, "spliced embedding prefill differs from the spliced training forward by {err}");
}

#[test]
fn embedding_prefill_matches_token_prefill_and_the_spliced_forward_cpu() {
    run(Gpu::new_cpu(pipelines()));
}

#[test]
fn embedding_prefill_matches_token_prefill_and_the_spliced_forward_default_backend() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    run(gpu_core::testgpu::dev(pipelines()));
}
