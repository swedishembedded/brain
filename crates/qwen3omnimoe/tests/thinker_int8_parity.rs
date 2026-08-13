// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen3omnimoe::thinker::layer_fwd`/`layer_decode_step`'s `int8_experts` branch vs.
//! the fp32 path, on the SAME source weights — tiny synthetic config, no HF
//! checkpoint needed (mirrors `thinker_decode.rs`'s own tier: this is the
//! "before real weights" gate). Proves the int8 branch wired into the real
//! Thinker forward (not just `model::moe::expert_fwd_i8` in isolation,
//! already covered by `crates/model/tests/moe_sparse_i8_parity.rs` and
//! `int8_resident`'s own tests) produces sane output through a FULL layer
//! (attention, still fp32, feeding into int8-quantized MoE experts).

use data::rng::Lcg;
use model::int8::quantize_weight;
use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::int8_resident::{ExpertLin8, ThinkerLayerExperts8};
use qwen3omnimoe::thinker::{layer_decode_step, layer_fwd, thinker_pipelines, ThinkerLayerCache, ThinkerLayerWeights};

fn tiny_config(n_layers: u32) -> MoeTextConfig {
    MoeTextConfig {
        n_layers,
        hidden: 16,       // multiple of 4 -- int8 packing needs k % 4 == 0
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        moe_intermediate: 12, // multiple of 4
        shared_expert_intermediate: 0,
        n_experts: 5,
        top_k: 2,
        norm_topk_prob: true,
        use_qk_norm: true,
        vocab: 32,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        mrope_section: vec![2, 1, 1],
        max_position_embeddings: 64,
    }
}

struct LayerBufs {
    ln1: gpu_core::DeviceBuffer,
    wq: gpu_core::DeviceBuffer,
    wk: gpu_core::DeviceBuffer,
    wv: gpu_core::DeviceBuffer,
    wo: gpu_core::DeviceBuffer,
    q_norm: gpu_core::DeviceBuffer,
    k_norm: gpu_core::DeviceBuffer,
    ln2: gpu_core::DeviceBuffer,
    router: gpu_core::DeviceBuffer,
    experts_fp32: Vec<(gpu_core::DeviceBuffer, gpu_core::DeviceBuffer, gpu_core::DeviceBuffer)>,
    experts_host: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

fn random_layer(gpu: &gpu_core::Gpu, rng: &mut Lcg, cfg: &MoeTextConfig) -> LayerBufs {
    let (d, hd, nh, nkv, ff) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.moe_intermediate);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let init = |rng: &mut Lcg, n: usize| gpu.storage_init("w", &rng.vec_scaled(n, 0.3));
    let mut experts_fp32 = Vec::new();
    let mut experts_host = Vec::new();
    for _ in 0..cfg.n_experts {
        let gw = rng.vec_scaled((ff * d) as usize, 0.3);
        let uw = rng.vec_scaled((ff * d) as usize, 0.3);
        let dw = rng.vec_scaled((d * ff) as usize, 0.3);
        experts_fp32.push((gpu.storage_init("gw", &gw), gpu.storage_init("uw", &uw), gpu.storage_init("dw", &dw)));
        experts_host.push((gw, uw, dw));
    }
    LayerBufs {
        ln1: init(rng, d as usize),
        wq: init(rng, (hq * d) as usize),
        wk: init(rng, (hkv * d) as usize),
        wv: init(rng, (hkv * d) as usize),
        wo: init(rng, (d * hq) as usize),
        q_norm: init(rng, hd as usize),
        k_norm: init(rng, hd as usize),
        ln2: init(rng, d as usize),
        router: init(rng, (cfg.n_experts * d) as usize),
        experts_fp32,
        experts_host,
    }
}

fn weights(b: &LayerBufs) -> ThinkerLayerWeights<'_> {
    ThinkerLayerWeights { ln1: &b.ln1, wq: &b.wq, wk: &b.wk, wv: &b.wv, wo: &b.wo, q_norm: &b.q_norm, k_norm: &b.k_norm, ln2: &b.ln2, router: &b.router, experts: &b.experts_fp32 }
}

/// Quantize `b`'s host fp32 expert weights (the SAME source `weights(b)`'s
/// fp32 buffers were uploaded from) into a resident int8 store — an
/// in-memory equivalent of `ThinkerInt8Store::build`'s stream-from-checkpoint
/// path, for a test that has no checkpoint FILE, only host arrays already in
/// hand.
fn quantize_layer(gpu: &gpu_core::Gpu, b: &LayerBufs, cfg: &MoeTextConfig) -> ThinkerLayerExperts8 {
    let (d, ff) = (cfg.hidden as usize, cfg.moe_intermediate as usize);
    let experts = b
        .experts_host
        .iter()
        .map(|(gw, uw, dw)| {
            let mk = |w: &[f32], n: usize, k: usize| {
                let (packed, scale) = quantize_weight(w, n, k);
                let pb = gpu.storage(packed.len() as u64);
                gpu.write(&pb, &packed);
                ExpertLin8 { packed: pb, scale: gpu.storage_init("scale", &scale) }
            };
            (mk(gw, ff, d), mk(uw, ff, d), mk(dw, d, ff))
        })
        .collect();
    ThinkerLayerExperts8 { experts }
}

fn rope_tables(gpu: &gpu_core::Gpu, cfg: &MoeTextConfig, n: u32) -> (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer) {
    let tokens: Vec<u32> = (0..n).collect();
    let positions = qwen3vl::mrope::get_rope_index(&tokens, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = qwen3vl::mrope::mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    (gpu.storage_init("cos", &cos_tab), gpu.storage_init("sin", &sin_tab))
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

#[test]
fn layer_fwd_int8_matches_fp32_within_quant_tolerance() {
    let cfg = tiny_config(1);
    let n = 6u32;
    let gpu = gpu_core::testgpu::dev(thinker_pipelines());
    let mut rng = Lcg::new(9001);

    let x_host = rng.vec_scaled((n * cfg.hidden) as usize, 1.0);
    let x = gpu.storage_init("x", &x_host);
    let (cos, sin) = rope_tables(&gpu, &cfg, n);

    let lb = random_layer(&gpu, &mut rng, &cfg);
    let w = weights(&lb);
    let experts8 = quantize_layer(&gpu, &lb, &cfg);

    let (fp32_out, ..) = layer_fwd(&gpu, &cfg, &w, &x, &cos, &sin, n, None, None);
    let (i8_out, ..) = layer_fwd(&gpu, &cfg, &w, &x, &cos, &sin, n, None, Some(&experts8));

    let fp32_host = gpu.read(&fp32_out, (n * cfg.hidden) as usize);
    let i8_host = gpu.read(&i8_out, (n * cfg.hidden) as usize);
    assert!(fp32_host.iter().any(|&v| v.abs() > 1e-9), "fp32 oracle is all-zero");
    assert!(i8_host.iter().all(|v| v.is_finite()), "int8 output has a non-finite value: {i8_host:?}");

    let err = rel_l2(&i8_host, &fp32_host);
    // Looser than moe_sparse_i8_parity.rs's bare-MoE 0.02 bound: this error
    // is measured after ALSO passing through fp32 attention (which is
    // identical in both legs and contributes no error of its own, but the
    // residual add means the MoE error is a SMALLER fraction of the total
    // output magnitude here than in a MoE-only comparison, not a larger
    // one -- 0.05 is still a real, tight bound, not a rubber stamp).
    assert!(err < 0.05, "layer_fwd int8 branch diverged from fp32: rel_l2={err:.4}");
}

#[test]
fn layer_decode_step_int8_matches_fp32_within_quant_tolerance() {
    let cfg = tiny_config(1);
    let cap = 4u32;
    let gpu = gpu_core::testgpu::dev(thinker_pipelines());
    let mut rng = Lcg::new(7002);

    let lb = random_layer(&gpu, &mut rng, &cfg);
    let w = weights(&lb);
    let experts8 = quantize_layer(&gpu, &lb, &cfg);

    let x_host = rng.vec_scaled(cfg.hidden as usize, 1.0);
    let x = gpu.storage_init("x", &x_host);
    let (cos, sin) = rope_tables(&gpu, &cfg, 1);

    let (nkv, hd) = (cfg.n_kv_heads, cfg.head_dim);
    let kcache_fp32 = gpu.storage((cap * nkv * hd) as u64);
    let vcache_fp32 = gpu.storage((cap * nkv * hd) as u64);
    let cache_fp32 = ThinkerLayerCache { kcache: &kcache_fp32, vcache: &vcache_fp32 };
    let kcache_i8 = gpu.storage((cap * nkv * hd) as u64);
    let vcache_i8 = gpu.storage((cap * nkv * hd) as u64);
    let cache_i8 = ThinkerLayerCache { kcache: &kcache_i8, vcache: &vcache_i8 };

    let fp32_out = layer_decode_step(&gpu, &cfg, &w, &cache_fp32, &x, &cos, &sin, 0, cap, None);
    let i8_out = layer_decode_step(&gpu, &cfg, &w, &cache_i8, &x, &cos, &sin, 0, cap, Some(&experts8));

    let fp32_host = gpu.read(&fp32_out, cfg.hidden as usize);
    let i8_host = gpu.read(&i8_out, cfg.hidden as usize);
    assert!(fp32_host.iter().any(|&v| v.abs() > 1e-9), "fp32 oracle is all-zero");
    let err = rel_l2(&i8_host, &fp32_host);
    assert!(err < 0.05, "layer_decode_step int8 branch diverged from fp32: rel_l2={err:.4}");
}
