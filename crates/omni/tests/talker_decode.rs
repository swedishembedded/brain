// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `omni::talker::decode` -- tiny-config smoke test, same shape as
//! `thinker_decode.rs`: proves the full-stack composition (N `layer_fwd`
//! calls chained residual-to-residual, then the top-level final RMSNorm)
//! matches a hand-chained oracle. Synthetic random weights via
//! `data::rng::Lcg`, no real checkpoint, no `#[ignore]` -- the porting
//! playbook's tiny-config-before-real-weights tier.

use data::rng::Lcg;
use model::block::{rmsnorm_fwd, KernelIds};
use omni::config::MoeTextConfig;
use omni::talker::{decode, layer_fwd, talker_pipelines, TalkerLayerWeights, TalkerWeights};
use qwenvl::mrope::{get_rope_index, mrope_tables};

fn tiny_config(n_layers: u32) -> MoeTextConfig {
    MoeTextConfig {
        n_layers,
        hidden: 8,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        moe_intermediate: 4,
        shared_expert_intermediate: 6, // deliberately different from moe_intermediate
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
        shared_expert_gate: init(rng, d as usize), // [1, d]
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
fn decode_matches_hand_chained_layer_fwd_plus_final_norm() {
    let n_layers = 3u32;
    let n = 5u32;
    let cfg = tiny_config(n_layers);
    let gpu = gpu_core::testgpu::dev(talker_pipelines());
    let mut rng = Lcg::new(1337);

    let x_host = rng.vec_scaled((n * cfg.hidden) as usize, 1.0);
    let x = gpu.storage_init("x", &x_host);

    let tokens: Vec<u32> = (0..n).collect();
    let positions = get_rope_index(&tokens, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);

    let layer_bufs: Vec<LayerBufs> = (0..n_layers).map(|_| random_layer(&gpu, &mut rng, &cfg)).collect();
    let layers: Vec<TalkerLayerWeights> = layer_bufs.iter().map(weights).collect();
    let final_norm = gpu.storage_init("final_norm", &rng.vec_scaled(cfg.hidden as usize, 0.3));

    let w = TalkerWeights { layers: &layers, final_norm: &final_norm };
    let got = decode(&gpu, &cfg, &w, &x, &cos, &sin, n);
    let got_host = gpu.read(&got, (n * cfg.hidden) as usize);
    assert!(got_host.iter().all(|v| v.is_finite()), "decode produced a non-finite value: {got_host:?}");

    let mut h = x;
    for lw in &layers {
        let (out, ..) = layer_fwd(&gpu, &cfg, lw, &h, &cos, &sin, n, None, None);
        h = out;
    }
    let ids = KernelIds { rmsnorm: 0, rms_inv: 0, rmsnorm_dx: 0, rmsnorm_dw: 0, rope: 0, rope_bwd: 0, gqa_scores: 0, gqa_apply: 0, attn_softmax: 0, gqa_dscores: 0, gqa_dv: 0, gqa_dq: 0, gqa_dk: 0, silu_mul: 0, silu_da: 0, silu_db: 0 };
    let normed = gpu.storage((n * cfg.hidden) as u64);
    gpu.submit(&[], &[rmsnorm_fwd(&gpu, &ids, &h, &final_norm, &normed, cfg.hidden, n)]);
    let want_host = gpu.read(&normed, (n * cfg.hidden) as usize);

    assert_eq!(got_host, want_host, "decode diverged from hand-chained layer_fwd + final rmsnorm");
}
