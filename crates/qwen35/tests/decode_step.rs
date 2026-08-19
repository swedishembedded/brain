// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen35::step`'s single-sequence incremental decode must reproduce
//! `Qwen35::logits_all`'s whole-sequence prefill for every position: same
//! engine, same weights, same tokens - any divergence between the two paths
//! can only be a decode-path bug (wrong GDN recurrence/conv history, wrong
//! GQA KV-cache append/attend, wrong single-position M-RoPE, ...), not a
//! numerical-noise artifact, since both paths are the exact same fp32
//! kernels evaluated in a different order. Mirrors
//! `qwen35moe/tests/decode_step.rs` exactly.
//!
//! One `Qwen35` instance serves BOTH roles: `logits_all` and `step` touch
//! entirely disjoint device buffers (`res`/`tokens`/`logits` vs. the
//! `dec_*`/`gqa_kcache`/`gdn_state`/`gdn_hist` decode state), so there is no
//! aliasing risk in calling `logits_all` first (the reference) and then
//! replaying the SAME token sequence through `reset_decode_cache`/`step`
//! (the thing under test) on that same model.
//!
//! `Qwen35Config::tiny()` exercises both layer types (Gated DeltaNet at
//! layers 0-2, GQA at layer 3) and a real multi-chunk GDN prefill
//! (`gdn_chunk_size(24)` is smaller than 24, multiple chunks) - so a
//! decode-path bug that only shows up across a chunk boundary has a chance
//! to surface here. Runs on both the CPU JIT and the default GPU backend.

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn run(gpu: Gpu) {
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    let d = cfg.d_model as usize;
    let v = cfg.vocab as usize;
    let init = qwen35::init::init_weights(&cfg, 7);

    let m = Qwen35::new_on(gpu, cfg.clone(), 1, t, &init);

    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();

    // Reference: one whole-sequence prefill forward.
    let full_logits = m.logits_all(&tokens);
    assert_eq!(full_logits.len(), t as usize * v);
    assert!(full_logits.iter().all(|x| x.is_finite()), "logits_all produced a non-finite value");

    // Incremental: replay the SAME tokens one at a time through the decode
    // path, applying the (untied, `tie_embeddings=false` in `tiny()`) head on
    // the host to each returned final-norm hidden state.
    let head_w = m.read_weight(cfg.head_weight()); // [v, d]
    m.reset_decode_cache();
    assert_eq!(m.decode_pos(), 0);

    let mut worst = 0.0f32;
    for (i, &tok) in tokens.iter().enumerate() {
        let hidden = m.step(tok);
        assert_eq!(hidden.len(), d, "step() hidden state must be [d_model]");
        assert!(hidden.iter().all(|x| x.is_finite()), "position {i}: step() produced a non-finite hidden state");
        assert_eq!(m.decode_pos(), i as u32 + 1);

        let logits_i: Vec<f32> =
            (0..v).map(|row| { let wr = &head_w[row * d..(row + 1) * d]; wr.iter().zip(&hidden).map(|(a, b)| a * b).sum::<f32>() }).collect();
        let want = &full_logits[i * v..(i + 1) * v];
        let err = maxabs(&logits_i, want);
        worst = worst.max(err);
        assert!(err < 2e-3, "position {i}: incremental decode vs full prefill maxabs={err}");
    }
    println!("decode_step_matches_full_prefill: worst maxabs over {t} positions = {worst:e}");
}

#[test]
fn decode_step_matches_full_prefill_cpu() {
    run(Gpu::new_cpu(pipelines()));
}

/// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
/// backend otherwise - run this under both, since a barrier-crossing kernel
/// can silently misbehave on exactly one backend.
#[test]
fn decode_step_matches_full_prefill_default_backend() {
    run(Gpu::new(pipelines()));
}
