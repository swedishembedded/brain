// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Talker decoder — one MoE layer, with its shared expert — vs.
//! the real transformers reference, on real weights. Validates
//! `qwen3omnimoe::talker::layer_fwd`: RMSNorm + GQA (QK-norm, table-driven M-RoPE) +
//! sparse MoE FFN + the always-active shared expert
//! (`model::moe::shared_expert_fwd`) — the one architectural difference from
//! `qwen3omnimoe::thinker::layer_fwd`. See `crates/omni/src/talker.rs`'s module doc.
//!
//! Same `layer_out`-not-`hidden` comparison-target subtlety as
//! `thinker_layer_parity.rs` (`Qwen3OmniMoeTalkerModel.forward` also applies
//! a top-level final RMSNorm after the layer loop, even truncated to 1
//! layer) — `tools/goldens/omni_dump_reference.py`'s `dump_talker_layer0`
//! hooks `model.norm`'s input for the same reason.
//!
//! Real-weight-adjacent: skips cleanly when the checkpoint shard holding
//! `talker.model.layers.0.*` is absent.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test talker_layer_parity -- --ignored --nocapture`

use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::talker::{layer_fwd, talker_pipelines, TalkerLayerWeights};
use qwen3vl::mrope::{get_rope_index, mrope_tables};

fn shard_for(tensor: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("BRAIN_OMNI_HF_DIR").ok()?);
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let shard = idx["weight_map"].as_object()?.get(tensor)?.as_str()?;
    let p = dir.join(shard);
    p.exists().then_some(p)
}

fn cosine_max_abs(got: &[f32], want: &[f32]) -> (f64, f32) {
    assert_eq!(got.len(), want.len(), "shape mismatch: got {} elems, want {}", got.len(), want.len());
    let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb).max(1e-12), got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
}

#[test]
#[ignore]
fn matches_the_real_talker_layer0() {
    let Some(shard) = shard_for("talker.model.layers.0.mlp.gate.weight") else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset, or its index doesn't (yet) have the shard holding talker.model.layers.0");
        return;
    };
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_talker_layer0.safetensors");
    if !golden_path.exists() {
        eprintln!("skip: {golden_path:?} missing (run `make fetch/testdata`)");
        return;
    }

    let mmap = MmapSafetensors::open(&shard).expect("open shard");
    // codec_embedding lives in a different shard than layer 0's weights.
    let embed_shard = shard_for("talker.model.codec_embedding.weight").expect("shard holding codec_embedding");
    let embed_mmap = MmapSafetensors::open(&embed_shard).expect("open codec_embedding shard");
    let cfg = MoeTextConfig::talker_defaults();

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let ids_f = golden.tensor_f32("codec_ids").expect("golden codec_ids");
    let ids: Vec<usize> = ids_f.iter().map(|&t| t as usize).collect();
    let n = ids.len() as u32;

    // Embedding lookup: Talker has no self.embed_tokens (real usage always
    // assembles inputs_embeds itself); codec_embedding is what the golden
    // dumper used, so this test matches it exactly.
    let embed = embed_mmap.tensor_f32("talker.model.codec_embedding.weight").expect("codec_embedding");
    let d = cfg.hidden as usize;
    let mut x_host = Vec::with_capacity(ids.len() * d);
    for &t in &ids {
        x_host.extend_from_slice(&embed[t * d..(t + 1) * d]);
    }

    let gpu = gpu_core::testgpu::dev(talker_pipelines());
    let x = gpu.storage_init("x", &x_host);

    // Diagonal M-RoPE table, same construction as thinker_layer_parity.rs
    // (this golden is a pure codec-id stream, no image/video span).
    let ids_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let positions = get_rope_index(&ids_u32, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("mrope_cos", &cos_tab);
    let sin = gpu.storage_init("mrope_sin", &sin_tab);

    let p = |leaf: &str| format!("talker.model.layers.0.{leaf}");
    let get = |name: &str| gpu.storage_init(name, &mmap.tensor_f32(name).unwrap_or_else(|| panic!("missing tensor {name}")));
    let ln1 = get(&p("input_layernorm.weight"));
    let wq = get(&p("self_attn.q_proj.weight"));
    let wk = get(&p("self_attn.k_proj.weight"));
    let wv = get(&p("self_attn.v_proj.weight"));
    let wo = get(&p("self_attn.o_proj.weight"));
    let q_norm = get(&p("self_attn.q_norm.weight"));
    let k_norm = get(&p("self_attn.k_norm.weight"));
    let ln2 = get(&p("post_attention_layernorm.weight"));
    let router = get(&p("mlp.gate.weight"));
    let experts: Vec<_> = (0..cfg.n_experts)
        .map(|e| {
            (
                get(&p(&format!("mlp.experts.{e}.gate_proj.weight"))),
                get(&p(&format!("mlp.experts.{e}.up_proj.weight"))),
                get(&p(&format!("mlp.experts.{e}.down_proj.weight"))),
            )
        })
        .collect();
    let shared_gate = get(&p("mlp.shared_expert.gate_proj.weight"));
    let shared_up = get(&p("mlp.shared_expert.up_proj.weight"));
    let shared_down = get(&p("mlp.shared_expert.down_proj.weight"));
    let shared_expert_gate = get(&p("mlp.shared_expert_gate.weight"));

    let w = TalkerLayerWeights {
        ln1: &ln1,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wo: &wo,
        q_norm: &q_norm,
        k_norm: &k_norm,
        ln2: &ln2,
        router: &router,
        experts: &experts,
        shared_expert: (&shared_gate, &shared_up, &shared_down),
        shared_expert_gate: &shared_expert_gate,
    };
    let (out, router_logits, xmid, gate) = layer_fwd(&gpu, &cfg, &w, &x, &cos, &sin, n, None, None);

    let got_xmid = gpu.read(&xmid, (n * cfg.hidden) as usize);
    let want_xmid = golden.tensor_f32("xmid").expect("golden xmid");
    let (cos_x, max_abs_x) = cosine_max_abs(&got_xmid, &want_xmid);
    println!("talker layer0 xmid: cosine={cos_x:.6} max_abs={max_abs_x:.6}");
    assert!(cos_x > 0.999, "xmid cosine {cos_x} <= 0.999 (attention stage diverges)");

    let got_router = gpu.read(&router_logits, (n * cfg.n_experts) as usize);
    let want_router = golden.tensor_f32("router_logits").expect("golden router_logits");
    let (cos_r, max_abs_r) = cosine_max_abs(&got_router, &want_router);
    println!("talker layer0 router_logits: cosine={cos_r:.6} max_abs={max_abs_r:.6}");
    assert!(cos_r > 0.999, "router_logits cosine {cos_r} <= 0.999");

    let got_gate = gpu.read(&gate, (n * cfg.n_experts) as usize);
    let want_ids_f = golden.tensor_f32("router_topk_ids").expect("golden router_topk_ids");
    let want_weights = golden.tensor_f32("router_topk_weights").expect("golden router_topk_weights");
    let mut want_gate = vec![0f32; (n * cfg.n_experts) as usize];
    for row in 0..n as usize {
        for kk in 0..cfg.top_k as usize {
            let eid = want_ids_f[row * cfg.top_k as usize + kk] as usize;
            want_gate[row * cfg.n_experts as usize + eid] = want_weights[row * cfg.top_k as usize + kk];
        }
    }
    let (cos_g, max_abs_g) = cosine_max_abs(&got_gate, &want_gate);
    println!("talker layer0 gate: cosine={cos_g:.6} max_abs={max_abs_g:.6}");
    assert!(cos_g > 0.999, "gate cosine {cos_g} <= 0.999 (router top-k/renorm diverges)");

    for row in 0..n as usize {
        let mut got_top: Vec<usize> = (0..cfg.n_experts as usize).collect();
        got_top.sort_by(|&a, &b| got_router[row * cfg.n_experts as usize + b].total_cmp(&got_router[row * cfg.n_experts as usize + a]));
        let mut got_set: Vec<usize> = got_top[..cfg.top_k as usize].to_vec();
        got_set.sort_unstable();
        let mut want_set: Vec<usize> = want_ids_f[row * cfg.top_k as usize..(row + 1) * cfg.top_k as usize].iter().map(|&v| v as usize).collect();
        want_set.sort_unstable();
        assert_eq!(got_set, want_set, "row {row}: routed expert SET differs from the reference");
    }

    let got_out = gpu.read(&out, (n * cfg.hidden) as usize);
    let want_out = golden.tensor_f32("layer_out").expect("golden layer_out");
    let (cos_o, max_abs_o) = cosine_max_abs(&got_out, &want_out);
    println!("talker layer0 out: cosine={cos_o:.6} max_abs={max_abs_o:.6}");
    assert!(cos_o > 0.999, "layer output cosine {cos_o} <= 0.999");
}
