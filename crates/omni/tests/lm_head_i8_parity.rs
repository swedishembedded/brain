// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `omni::thinker::lm_head_fwd_i8` vs. the fp32 `lm_head_fwd` -- same
//! discipline as `crates/model/tests/moe_shared_expert_i8.rs` (fp32 oracle,
//! deterministic `Lcg` seed, measured relative-L2 recorded with headroom).
//! `lm_head.weight` is the largest non-expert weight in a real checkpoint
//! (`[vocab, hidden]` = 152064 x 2048 at Thinker's real shape, ~1.2 GiB fp32
//! / ~300 MiB int8) and, unlike the router, an approximation error here only
//! perturbs logit MAGNITUDES feeding a softmax/argmax, not a hard routing
//! decision -- the reasoning `docs/models/omni/status.md`'s Gap 4 entry
//! records for quantizing this component before the router.

use data::rng::Lcg;
use model::int8::quantize_weight;
use model::moe::Lin8;
use omni::thinker::{lm_head_fwd, lm_head_fwd_i8, thinker_pipelines, LmHeadIds8};

fn idx(g: &gpu_core::Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (b as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

#[test]
fn lm_head_i8_matches_fp32_within_quant_tolerance() {
    let g = gpu_core::testgpu::dev(thinker_pipelines());
    let ids8 = LmHeadIds8 { matmul_i8: idx(&g, "matmul_i8_dyn"), quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")] };

    // n (rows) small, d/vocab multiples of 4 (int8 packing needs k % 4 == 0
    // on the contracted dim; vocab, the output dim, has no such constraint).
    let (n, d, vocab) = (5u32, 16u32, 37u32);
    let mut rng = Lcg::new(0xC0FFEE);
    let h_host = rng.vec_scaled((n * d) as usize, 1.0);
    let w_host = rng.vec_scaled((vocab * d) as usize, 0.4);

    let hidden = g.storage_init("h", &h_host);
    let w_fp32 = g.storage_init("w", &w_host);
    let fp32_out = lm_head_fwd(&g, &w_fp32, &hidden, n, d, vocab);
    let fp32 = g.read(&fp32_out, (n * vocab) as usize).to_vec();
    assert!(fp32.iter().any(|&v| v.abs() > 1e-9), "fp32 reference is all-zero -- the test shape is degenerate");

    let (packed, scale) = quantize_weight(&w_host, vocab as usize, d as usize);
    let wq = g.storage(packed.len() as u64);
    g.write(&wq, &packed);
    let sw = g.storage_init("sw", &scale);
    let i8_out = lm_head_fwd_i8(&g, &ids8, Lin8 { wq: &wq, sw: &sw }, &hidden, n, d, vocab);
    let i8 = g.read(&i8_out, (n * vocab) as usize).to_vec();

    let err = rel_l2(&i8, &fp32);
    eprintln!("lm_head_fwd_i8 rel_l2: {err:.6}");
    // Measured 0.0050 on this shape/seed -- same order of magnitude as
    // moe_shared_expert_i8.rs's chained-SwiGLU numbers (0.0014-0.0038)
    // despite this being a single GEMM, since the tiny `d=16` contraction
    // dim here gives quantization noise less to average out over than a
    // real 2048-wide checkpoint would. 0.02 (the same bound
    // moe_sparse_i8_parity.rs/moe_shared_expert_i8.rs use) leaves real
    // headroom without being a rubber stamp on this shape.
    assert!(err < 0.02, "lm_head_fwd_i8 vs fp32 rel_l2 {err} exceeds tolerance");
}
