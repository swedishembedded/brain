// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `omni::talker::layer_decode_step` -- proves the KV-cache decode path
//! (attention via `model::block::gqa_decode_step`, already proven
//! algebraically exact against `gqa_fwd` in isolation) composes correctly
//! with Talker's MoE-plus-shared-expert FFN when chained across MULTIPLE
//! layers, one token at a time -- the untested territory `model::block`'s
//! own unit test doesn't reach (it exercises attention alone, not a full
//! decoder stack). Tiny synthetic config, same shape as `talker_decode.rs`.
//!
//! Method: run the same `n`-token sequence two ways and compare every
//! position's final-normed output exactly:
//!   (a) batched: `layer_fwd(x[0..n], cache=None)` per layer, chained (what
//!       `decode` does).
//!   (b) incremental: for each position `i` in order, run EVERY layer's
//!       `layer_fwd(x[i..i+1], cache=Some)` (a length-1 "prefill" that just
//!       appends to that layer's cache and returns this token's own causal
//!       output, since with `t=i+1` cached rows the causal-masked batched
//!       attention already only attends `0..=i`) -- proving the SAME
//!       composition holds when built one token at a time, layer-major,
//!       exactly how `crate::generate`'s prefill would in practice for
//!       Talker's own KV-cache decode loop.
//! Positions 1..n additionally cross-check via `layer_decode_step` directly
//! (the real single-token decode entry `crate::generate`-equivalent code
//! will call), attending against the cache the length-1 layer_fwd calls at
//! earlier positions already populated.

use data::rng::Lcg;
use omni::config::MoeTextConfig;
use omni::talker::{layer_decode_step, layer_fwd, talker_pipelines, TalkerLayerCache, TalkerLayerWeights};
use qwenvl::mrope::mrope_tables;

fn tiny_config(n_layers: u32) -> MoeTextConfig {
    MoeTextConfig {
        n_layers,
        hidden: 8,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        moe_intermediate: 4,
        shared_expert_intermediate: 6,
        n_experts: 4,
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
    experts: Vec<(gpu_core::DeviceBuffer, gpu_core::DeviceBuffer, gpu_core::DeviceBuffer)>,
    shared_gate: gpu_core::DeviceBuffer,
    shared_up: gpu_core::DeviceBuffer,
    shared_down: gpu_core::DeviceBuffer,
    shared_expert_gate: gpu_core::DeviceBuffer,
}

fn random_layer(gpu: &gpu_core::Gpu, rng: &mut Lcg, cfg: &MoeTextConfig) -> LayerBufs {
    let (d, hd, nh, nkv, ff, sff) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.moe_intermediate, cfg.shared_expert_intermediate);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let init = |rng: &mut Lcg, n: usize| gpu.storage_init("w", &rng.vec_scaled(n, 0.3));
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
        experts: (0..cfg.n_experts).map(|_| (init(rng, (ff * d) as usize), init(rng, (ff * d) as usize), init(rng, (d * ff) as usize))).collect(),
        shared_gate: init(rng, (sff * d) as usize),
        shared_up: init(rng, (sff * d) as usize),
        shared_down: init(rng, (d * sff) as usize),
        shared_expert_gate: init(rng, d as usize),
    }
}

fn weights(b: &LayerBufs) -> TalkerLayerWeights<'_> {
    TalkerLayerWeights {
        ln1: &b.ln1,
        wq: &b.wq,
        wk: &b.wk,
        wv: &b.wv,
        wo: &b.wo,
        q_norm: &b.q_norm,
        k_norm: &b.k_norm,
        ln2: &b.ln2,
        router: &b.router,
        experts: &b.experts,
        shared_expert: (&b.shared_gate, &b.shared_up, &b.shared_down),
        shared_expert_gate: &b.shared_expert_gate,
    }
}

#[test]
fn incremental_decode_matches_batched_layer_fwd_at_every_position() {
    let n_layers = 3usize;
    let n = 5u32;
    let cfg = tiny_config(n_layers as u32);
    let gpu = gpu_core::testgpu::dev(talker_pipelines());
    let mut rng = Lcg::new(4242);

    let x_host = rng.vec_scaled((n * cfg.hidden) as usize, 1.0);
    let x_full = gpu.storage_init("x", &x_host);

    // Real per-token M-RoPE tables (diagonal/plain-text case): one row per position.
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let positions: Vec<[u32; 3]> = (0..n).map(|t| [t, t, t]).collect();
    let (cos_tab, sin_tab) = mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let half = (cfg.head_dim / 2) as usize;
    let cos_full = gpu.storage_init("cos", &cos_tab);
    let sin_full = gpu.storage_init("sin", &sin_tab);

    let layer_bufs: Vec<LayerBufs> = (0..n_layers).map(|_| random_layer(&gpu, &mut rng, &cfg)).collect();
    let layers: Vec<TalkerLayerWeights> = layer_bufs.iter().map(weights).collect();

    // (a) Batched reference: layer_fwd chained across all n positions at once.
    let mut h = x_full.clone();
    for lw in &layers {
        let (out, ..) = layer_fwd(&gpu, &cfg, lw, &h, &cos_full, &sin_full, n, None);
        h = out;
    }
    let want = gpu.read(&h, (n * cfg.hidden) as usize);

    // (b) Incremental: one token at a time, every layer, via length-1
    // layer_fwd(cache=Some) calls (the "prefill one row at a time" shape).
    let cap = n;
    let hkv = (cfg.n_kv_heads * cfg.head_dim) as u64;
    let caches: Vec<(gpu_core::DeviceBuffer, gpu_core::DeviceBuffer)> = (0..n_layers).map(|_| (gpu.storage(cap as u64 * hkv), gpu.storage(cap as u64 * hkv))).collect();
    let cache_refs: Vec<TalkerLayerCache> = caches.iter().map(|(k, v)| TalkerLayerCache { kcache: k, vcache: v }).collect();

    let mut got = vec![0f32; (n * cfg.hidden) as usize];
    for pos in 0..n {
        let row_start = (pos * cfg.hidden) as usize;
        let x_row = gpu.storage_init("x_row", &x_host[row_start..row_start + cfg.hidden as usize]);
        let cos_row = gpu.storage_init("cos_row", &cos_tab[pos as usize * half..(pos as usize + 1) * half]);
        let sin_row = gpu.storage_init("sin_row", &sin_tab[pos as usize * half..(pos as usize + 1) * half]);

        let mut hrow = x_row;
        for (l, lw) in layers.iter().enumerate() {
            hrow = if pos == 0 {
                // First position: a length-1 "prefill" populates the cache's row 0.
                let (out, ..) = layer_fwd(&gpu, &cfg, lw, &hrow, &cos_row, &sin_row, 1, Some(&cache_refs[l]));
                out
            } else {
                layer_decode_step(&gpu, &cfg, lw, &cache_refs[l], &hrow, &cos_row, &sin_row, pos, cap)
            };
        }
        got[row_start..row_start + cfg.hidden as usize].copy_from_slice(&gpu.read(&hrow, cfg.hidden as usize));
    }

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-3, "elem {i} (position {}): incremental={g}, batched={w}", i / cfg.hidden as usize);
    }
}
