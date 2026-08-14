// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Thinker decoder — one MoE layer — vs. the real transformers
//! reference, on real weights. Validates `qwen3omnimoe::thinker::layer_fwd`: RMSNorm,
//! GQA (QK-norm, half-split RoPE), and a sparse MoE FFN (`model::moe`, 128
//! experts / top-8, no shared expert), composed fresh from
//! `model::block`/`model::moe` primitives rather than a modified
//! `qwen3::Qwen` — see `crates/omni/src/thinker.rs`'s module doc for why.
//!
//! The golden (`tools/goldens/qwen3omnimoe_dump_reference.py`'s `layer0`) is a pure
//! 9-token TEXT prompt with no image/audio, so its M-RoPE table (built here
//! via the real `qwen3vl::mrope::{get_rope_index, mrope_tables}` path, same as
//! a mixed-modality prompt would use) is the degenerate diagonal case where
//! all three axes carry the same position — proven to collapse exactly to
//! plain half-split RoPE by `qwen3vl::mrope`'s own test, not a simplification
//! this test takes a shortcut on.
//!
//! `layer_out` (not `hidden`) is the right golden tensor to compare `out`
//! against: `Qwen3OmniMoeThinkerTextModel.forward` always applies its
//! top-level `self.norm` (the final decoder-stack RMSNorm) after the layer
//! loop, even truncated to 1 layer — `hidden`/`last_hidden_state` is
//! `model.norm(layer0_output)`, not layer 0's own raw output that a single
//! `layer_fwd` call actually produces. `layer_out` is hooked straight off
//! `model.norm`'s input in the dumper for an apples-to-apples comparison.
//!
//! Real-weight-adjacent: skips cleanly when the checkpoint shard holding
//! `thinker.model.layers.0.*` (shard 1, same shard M4/M5 use) is absent.
//!
//! usage: `BRAIN_QWEN3OMNIMOE_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test thinker_layer_parity -- --ignored --nocapture`

use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::thinker::{layer_fwd, thinker_pipelines, ThinkerLayerWeights};
use qwen3vl::mrope::{get_rope_index, mrope_tables};

fn shard_with_layer0() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("BRAIN_QWEN3OMNIMOE_HF_DIR").ok()?);
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let shard = idx["weight_map"].as_object()?.get("thinker.model.layers.0.mlp.gate.weight")?.as_str()?;
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
fn matches_the_real_thinker_layer0() {
    let Some(shard) = shard_with_layer0() else {
        eprintln!("skip: BRAIN_QWEN3OMNIMOE_HF_DIR unset, or its index doesn't (yet) have the shard holding thinker.model.layers.0");
        return;
    };
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_layer0.safetensors");
    if !golden_path.exists() {
        eprintln!("skip: {golden_path:?} missing (run `make fetch/testdata`)");
        return;
    }

    let mmap = MmapSafetensors::open(&shard).expect("open shard");
    let cfg = MoeTextConfig::thinker_defaults();

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let tokens_f = golden.tensor_f32("tokens").expect("golden tokens");
    let tokens: Vec<usize> = tokens_f.iter().map(|&t| t as usize).collect();
    let n = tokens.len() as u32;

    // Embedding lookup: gather this prompt's 9 rows from the (huge) embed
    // table, entirely host-side -- no GPU buffer for the other 152055 rows.
    let embed = mmap.tensor_f32("thinker.model.embed_tokens.weight").expect("embed_tokens");
    let d = cfg.hidden as usize;
    let mut x_host = Vec::with_capacity(tokens.len() * d);
    for &t in &tokens {
        x_host.extend_from_slice(&embed[t * d..(t + 1) * d]);
    }

    let gpu = gpu_core::testgpu::dev(thinker_pipelines());
    let x = gpu.storage_init("x", &x_host);

    // Pure text: get_rope_index's diagonal case (no image token, empty grids)
    // -- every axis carries the same position, which mrope_tables/rope2d_fwd
    // must reduce to plain half-split RoPE (proven by qwen3vl::mrope's own
    // diagonal_positions_collapse_to_plain_rope test).
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let positions = get_rope_index(&tokens_u32, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("mrope_cos", &cos_tab);
    let sin = gpu.storage_init("mrope_sin", &sin_tab);

    let p = |leaf: &str| format!("thinker.model.layers.0.{leaf}");
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

    let w = ThinkerLayerWeights { ln1: &ln1, wq: &wq, wk: &wk, wv: &wv, wo: &wo, q_norm: &q_norm, k_norm: &k_norm, ln2: &ln2, router: &router, experts: &experts };
    let (out, router_logits, xmid, gate) = layer_fwd(&gpu, &cfg, &w, &x, &cos, &sin, n, None, None);

    // Attention stage: the post-attention, pre-MoE residual state.
    let got_xmid = gpu.read(&xmid, (n * cfg.hidden) as usize);
    let want_xmid = golden.tensor_f32("xmid").expect("golden xmid");
    let (cos_x, max_abs_x) = cosine_max_abs(&got_xmid, &want_xmid);
    println!("thinker layer0 xmid: cosine={cos_x:.6} max_abs={max_abs_x:.6}");
    assert!(cos_x > 0.999, "xmid cosine {cos_x} <= 0.999 (attention stage diverges)");

    // Router: raw logits, then the dense post-topk-renorm gate.
    let got_router = gpu.read(&router_logits, (n * cfg.n_experts) as usize);
    let want_router = golden.tensor_f32("router_logits").expect("golden router_logits");
    let (cos_r, max_abs_r) = cosine_max_abs(&got_router, &want_router);
    println!("thinker layer0 router_logits: cosine={cos_r:.6} max_abs={max_abs_r:.6}");
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
    println!("thinker layer0 gate: cosine={cos_g:.6} max_abs={max_abs_g:.6}");
    assert!(cos_g > 0.999, "gate cosine {cos_g} <= 0.999 (router top-k/renorm diverges)");

    // The router's own top-k SET should exactly match too (same logits ->
    // same argmax set), independent of the renormalised weights above.
    for row in 0..n as usize {
        let mut got_top: Vec<usize> = (0..cfg.n_experts as usize).collect();
        got_top.sort_by(|&a, &b| got_router[row * cfg.n_experts as usize + b].total_cmp(&got_router[row * cfg.n_experts as usize + a]));
        let mut got_set: Vec<usize> = got_top[..cfg.top_k as usize].to_vec();
        got_set.sort_unstable();
        let mut want_set: Vec<usize> = want_ids_f[row * cfg.top_k as usize..(row + 1) * cfg.top_k as usize].iter().map(|&v| v as usize).collect();
        want_set.sort_unstable();
        assert_eq!(got_set, want_set, "row {row}: routed expert SET differs from the reference");
    }

    // Full layer output: attention + sparse MoE FFN combine.
    let got_out = gpu.read(&out, (n * cfg.hidden) as usize);
    let want_out = golden.tensor_f32("layer_out").expect("golden layer_out");
    let (cos_o, max_abs_o) = cosine_max_abs(&got_out, &want_out);
    println!("thinker layer0 out: cosine={cos_o:.6} max_abs={max_abs_o:.6}");
    assert!(cos_o > 0.999, "layer output cosine {cos_o} <= 0.999");
}
