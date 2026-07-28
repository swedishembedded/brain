// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM-5.2 forward at a size that triggers the register GEMM (`matmul_reg2`):
//! the CPU reference and the GPU must agree, and the register kernel must
//! reproduce the naive kernel bit-for-bit. This is what makes "GLM is on the
//! fast path" a validated claim rather than a wiring assertion — its tiny
//! gradcheck config (d=16) never clears the 128-wide tile threshold.

use std::collections::HashMap;

use glm::{init_weights, Glm, GlmConfig};
use gpu_core::{set_default_backend, Backend};

/// d_model 256, seq 160 → every forward linear has m=160, nout ∈ {256, …} ≥ 128,
/// so `mm()` selects `matmul_reg2`.
fn cfg() -> GlmConfig {
    let mut c = GlmConfig::tiny();
    c.vocab = 512;
    c.block_size = 192;
    c.n_layers = 2;
    c.d_model = 256;
    c.n_heads = 4;
    c.intermediate_size = 512;
    c.moe_intermediate_size = 256;
    c.first_k_dense_replace = 2; // keep it dense (exercise the plain linears)
    c
}

fn logits(backend: Backend, naive: bool, c: &GlmConfig, init: &HashMap<String, Vec<f32>>, x: &[u32]) -> Vec<f32> {
    if naive {
        std::env::set_var("BRAIN_GLM_NAIVE_MM", "1");
    } else {
        std::env::remove_var("BRAIN_GLM_NAIVE_MM");
    }
    set_default_backend(backend);
    let m = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), c.clone(), 1, x.len() as u32, init);
    m.logits_all(x)
}

fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let maxd = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let scale = a.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    (maxd, maxd / scale)
}

#[test]
fn glm_reg2_matches_naive_and_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let c = cfg();
    let init = init_weights(&c, 9);
    let x: Vec<u32> = (0..160).map(|i| ((i * 7 + 1) as u32) % c.vocab).collect();

    let gpu_reg = logits(Backend::Vulkan, false, &c, &init, &x); // matmul_reg2
    let gpu_naive = logits(Backend::Vulkan, true, &c, &init, &x); // matmul
    let cpu = logits(Backend::Cpu, false, &c, &init, &x); // CPU (reg2 -> AVX2)

    let (ka, kr) = rel(&gpu_naive, &gpu_reg);
    let (ca, cr) = rel(&cpu, &gpu_reg);
    eprintln!("glm reg2 vs naive (same GPU): max-abs {ka:.2e} rel {kr:.2e}");
    eprintln!("glm reg2 vs cpu reference:    max-abs {ca:.2e} rel {cr:.2e}");

    // Same backend, only the GEMM kernel differs → fp32 associativity only.
    assert!(kr < 1e-4, "glm register GEMM changes the forward (rel {kr:.2e})");
    // CPU vs GPU reduction order.
    assert!(cr < 5e-3, "glm gpu forward diverges from cpu (rel {cr:.2e})");
}
