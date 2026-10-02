// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A decode step with the native fused kernels computes EXACTLY what the same
//! step computes as the chain of WGSL dispatches they replace, in fewer
//! launches.
//!
//! Swedish Embedded AB implements bit-exact fused kernels for quantised LLM
//! decode. If your team needs expertise in cutting the launch count of a token
//! loop without moving a single output bit, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The per-kernel gates (`gpu-core/tests/add_rms_quant_native.rs`,
//! `quant_epilogue_native.rs`) prove each fused kernel against its chain on
//! random data. What they cannot show is that the DECODE TAPE wires them
//! correctly: the right residual into the right norm, the right activation to
//! the right linear, the sigmoid gate on the attention output and not the
//! MLP's. Those are shape-compatible mistakes - a swapped buffer still
//! produces finite numbers - so the claim here is the strong one: the hidden
//! state after every position is BIT-identical with the fusion on and off.
//!
//! CUDA backend only; skips elsewhere. The second test checks the fusion
//! actually engaged - without it the identity above would hold trivially on a
//! device that was never offered the kernels.

use backend_cuda::call_totals;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use std::sync::{Mutex, MutexGuard};

/// The call totals are process-global.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The tiny int8 shape, and the same shape with the real model's 128-wide
/// linear-attention heads - the only width the native GDN decode kernel serves,
/// so the only one on which its wiring in the decode tape is exercised.
fn tiny() -> Qwen35Config {
    Qwen35Config::tiny_i8()
}

fn wide_gdn() -> Qwen35Config {
    Qwen35Config { linear_key_head_dim: 128, linear_value_head_dim: 128, ..Qwen35Config::tiny_i8() }
}

fn model(gpu: Gpu, cfg: Qwen35Config) -> (Qwen35, Vec<u32>, usize) {
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, t, &init, &TierPolicy::uniform(Dtype::I8));
    let tokens = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    (m, tokens, cfg.n_layers as usize)
}

fn decode(m: &Qwen35, tokens: &[u32]) -> Vec<Vec<f32>> {
    m.reset_decode_cache();
    tokens.iter().map(|&t| m.step(t)).collect()
}

fn bit_identical(cfg: Qwen35Config) {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let (m, tokens, _) = model(gpu, cfg);
    let fused = decode(&m, &tokens);
    m.set_decode_fusion(false);
    let unfused = decode(&m, &tokens);
    for (i, (a, b)) in fused.iter().zip(&unfused).enumerate() {
        assert!(a.iter().all(|x| x.is_finite()), "position {i}: non-finite hidden state");
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "position {i}: the fused decode step differs from the unfused chain"
        );
    }
}

#[test]
fn the_fused_decode_tape_is_bit_identical_to_the_unfused_one() {
    bit_identical(tiny());
}

#[test]
fn the_fused_decode_tape_is_bit_identical_with_128_wide_linear_attention_heads() {
    bit_identical(wide_gdn());
}

/// A fresh model's first two decode steps are issued and recorded launch by
/// launch (the eager step, then the capturing one), so the launch count over
/// them is twice the dispatches in a token. The fused tape has strictly fewer.
fn launches(gpu: &Gpu, cfg: &Qwen35Config, fusion: bool) -> u64 {
    let (m, tokens, _) = model(gpu.share(), cfg.clone());
    m.set_decode_fusion(fusion);
    m.reset_decode_cache();
    let before = call_totals().host_launches;
    for &t in &tokens[..2] {
        m.step(t);
    }
    call_totals().host_launches - before
}

#[test]
fn the_fused_decode_tape_launches_fewer_kernels() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let cfg = tiny();
    let (fused, unfused) = (launches(&gpu, &cfg, true), launches(&gpu, &cfg, false));
    // Per token, each layer saves at least the three launches its second norm
    // front folds away.
    assert!(
        unfused >= fused + 2 * 3 * cfg.n_layers as u64,
        "the fused tape launched {fused} kernels over two decode steps, the unfused one {unfused}: the fusion did not engage"
    );
}

/// With 128-wide linear-attention heads the GDN layers take the one-launch
/// step too: each of the three GDN layers drops its nineteen-kernel chain.
#[test]
fn the_gdn_step_is_one_launch_with_128_wide_heads() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let (narrow, wide) = (tiny(), wide_gdn());
    let saved = |cfg: &Qwen35Config| launches(&gpu, cfg, false) - launches(&gpu, cfg, true);
    let gdn_layers = 3u64;
    assert!(
        saved(&wide) >= saved(&narrow) + 2 * gdn_layers * 17,
        "128-wide heads saved {} launches over two steps against {} for the narrow ones: the GDN kernel did not engage",
        saved(&wide),
        saved(&narrow)
    );
}
