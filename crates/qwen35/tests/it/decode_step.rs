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

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
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

/// The SAME prefill-vs-decode equivalence at the **INT8** tier, which the two
/// tests above cannot see: they build with `Qwen35::new_on`, i.e. fp32, so
/// every quantized code path is unreachable from them.
///
/// This is the crossing a prior fix named and only half closed: that fix
/// addressed a *panic* - the int8 decode tape looked its
/// projections up in the fp32 `ParamStore`, which an int8 build deliberately
/// does not populate - and the regression test it left behind
/// (`model.rs`'s `two_shard_int8_decode_matches_the_whole_shard_model`)
/// compares int8-decode against int8-decode. A systematic numeric error in
/// the int8 decode tape (a wrong activation-quantization scale, a mis-shaped
/// `Act`, the wrong `Weight` for a leaf) is identical on both sides of that
/// comparison and passes it. Only a cross-TIER or cross-PATH reference can
/// catch it, and this is the cross-path one: the int8 PREFILL tape
/// (`ops_linear` through `Ops::matmul`, validated by `model_i8_smoke.rs` and
/// `int8_real_weight_sanity.rs`) against the int8 DECODE tape at every
/// position of the same sequence.
///
/// Tolerance: the two tapes run the identical quantized kernels over the
/// identical weights and differ only in row count per dispatch (`t` vs 1),
/// so the residual difference is dynamic activation quantization on
/// different row groupings plus fp32 reduction order - not a different
/// computation. `2e-2` on logits of order 1 is loose enough for that and
/// nowhere near loose enough to hide a wrong tape (the failure mode this
/// exists for produces logits that disagree in ARGMAX, not in the third
/// decimal).
fn run_i8(gpu: Gpu) {
    let cfg = Qwen35Config::tiny_i8();
    let t = cfg.block_size;
    let d = cfg.d_model as usize;
    let v = cfg.vocab as usize;
    let init = qwen35::init::init_weights(&cfg, 7);

    let m = Qwen35::new_on_i8(gpu, cfg.clone(), 1, t, &init);
    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();

    let full_logits = m.logits_all(&tokens);
    assert_eq!(full_logits.len(), t as usize * v);
    assert!(full_logits.iter().all(|x| x.is_finite()), "int8 logits_all produced a non-finite value");

    // `lm_head` stays fp32 on an int8 build (`is_i8_linear` never names it),
    // so the host head below is the same weight both tapes use.
    let head_w = m.read_weight(cfg.head_weight()); // [v, d]
    m.reset_decode_cache();

    let mut worst = 0.0f32;
    let mut argmax_mismatches = 0;
    for (i, &tok) in tokens.iter().enumerate() {
        let hidden = m.step(tok);
        assert!(hidden.iter().all(|x| x.is_finite()), "position {i}: int8 step() produced a non-finite hidden state");
        let logits_i: Vec<f32> =
            (0..v).map(|row| { let wr = &head_w[row * d..(row + 1) * d]; wr.iter().zip(&hidden).map(|(a, b)| a * b).sum::<f32>() }).collect();
        let want = &full_logits[i * v..(i + 1) * v];
        worst = worst.max(maxabs(&logits_i, want));
        let am = |s: &[f32]| s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(j, _)| j).unwrap();
        if am(&logits_i) != am(want) {
            argmax_mismatches += 1;
        }
    }
    println!("int8 decode vs int8 prefill over {t} positions: worst maxabs = {worst:e}, argmax mismatches = {argmax_mismatches}");
    assert_eq!(argmax_mismatches, 0, "int8 decode picks a different token than int8 prefill at {argmax_mismatches} position(s)");
    assert!(worst < 2e-2, "int8 decode vs int8 prefill maxabs={worst}");
}

#[test]
fn int8_decode_step_matches_int8_full_prefill_cpu() {
    run_i8(Gpu::new_cpu(pipelines()));
}

#[test]
fn int8_decode_step_matches_int8_full_prefill_default_backend() {
    run_i8(Gpu::new(pipelines()));
}

/// The Q4 (W4A8) twin of [`run_i8`], for the exact same reason: without it,
/// the Q4 DECODE tape (M24) is ungated - `model_q4_smoke.rs` only exercises
/// Q4 PREFILL. Same tolerance rationale as `run_i8`'s own doc: identical
/// quantized kernels on both tapes, differing only in row count per
/// dispatch, so a systematic tape bug shows up as a wrong ARGMAX, not a
/// third-decimal wobble.
fn run_q4(gpu: Gpu) {
    let cfg = Qwen35Config::tiny_i8();
    let t = cfg.block_size;
    let d = cfg.d_model as usize;
    let v = cfg.vocab as usize;
    let init = qwen35::init::init_weights(&cfg, 7);

    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, t, &init, &TierPolicy::uniform(Dtype::Q4));
    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();

    let full_logits = m.logits_all(&tokens);
    assert_eq!(full_logits.len(), t as usize * v);
    assert!(full_logits.iter().all(|x| x.is_finite()), "q4 logits_all produced a non-finite value");

    let head_w = m.read_weight(cfg.head_weight()); // [v, d] -- lm_head stays fp32 on a q4 build too
    m.reset_decode_cache();

    let mut worst = 0.0f32;
    let mut argmax_mismatches = 0;
    for (i, &tok) in tokens.iter().enumerate() {
        let hidden = m.step(tok);
        assert!(hidden.iter().all(|x| x.is_finite()), "position {i}: q4 step() produced a non-finite hidden state");
        let logits_i: Vec<f32> =
            (0..v).map(|row| { let wr = &head_w[row * d..(row + 1) * d]; wr.iter().zip(&hidden).map(|(a, b)| a * b).sum::<f32>() }).collect();
        let want = &full_logits[i * v..(i + 1) * v];
        worst = worst.max(maxabs(&logits_i, want));
        let am = |s: &[f32]| s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(j, _)| j).unwrap();
        if am(&logits_i) != am(want) {
            argmax_mismatches += 1;
        }
    }
    println!("q4 decode vs q4 prefill over {t} positions: worst maxabs = {worst:e}, argmax mismatches = {argmax_mismatches}");
    assert_eq!(argmax_mismatches, 0, "q4 decode picks a different token than q4 prefill at {argmax_mismatches} position(s)");
    assert!(worst < 2e-2, "q4 decode vs q4 prefill maxabs={worst}");
}

#[test]
fn q4_decode_step_matches_q4_full_prefill_cpu() {
    run_q4(Gpu::new_cpu(pipelines()));
}

#[test]
fn q4_decode_step_matches_q4_full_prefill_default_backend() {
    run_q4(Gpu::new(pipelines()));
}
