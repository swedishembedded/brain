// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The register GEMM (`matmul_reg`) routes to the CPU AVX2 GEMM on the CPU
//! backend; this asserts the CPU forward is identical whether `linear_kernel`
//! picks the register or the naive kernel — i.e. the one-graph routing is real,
//! not just a GPU story.
use gpt2::{Gpt, GptConfig};
use gpu_core::{set_default_backend, Backend};

#[test]
fn cpu_register_equals_cpu_naive() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; } // uses gpu_core (CPU here), gated like siblings
    set_default_backend(Backend::Cpu);
    let c = GptConfig { vocab: 512, block_size: 128, n_layers: 2, d_model: 256, n_heads: 4, d_ff: 1024 }.with_ff_default();
    let init = gpt2::init_weights(&c, 7);
    let x: Vec<u32> = (0..512).map(|i| (i as u32 * 5 + 1) % c.vocab).collect();
    let y = vec![gpt2::model::IGNORE; 512];

    std::env::set_var("BRAIN_GPT_NAIVE_MM", "1");
    let m1 = Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), c.clone(), 4, 128, &init);
    m1.set_batch(&x, &y); m1.forward_submit();
    let naive = m1.logits_host();

    std::env::remove_var("BRAIN_GPT_NAIVE_MM");
    let m2 = Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), c.clone(), 4, 128, &init); // M=512>=128, N picks reg where >=128
    m2.set_batch(&x, &y); m2.forward_submit();
    let reg = m2.logits_host();

    let maxd = naive.iter().zip(&reg).fold(0f32, |m,(a,b)| m.max((a-b).abs()));
    eprintln!("cpu naive-vs-register max-abs {maxd:.2e}");
    assert!(maxd < 1e-4, "cpu register-gemm routing differs from naive (max-abs {maxd})");
}
