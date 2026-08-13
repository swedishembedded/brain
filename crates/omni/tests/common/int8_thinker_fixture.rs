// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scaffolding for the int8 dual-GPU Thinker tests: a tiny config and a
//! synthetic on-disk checkpoint with real `omni::import` naming/dtypes for
//! everything `Int8ThinkerInstance` needs. Included (via `#[path]`) into
//! `int8_thinker_multi_gpu.rs` (which drives `ResidencyManager::claim_multi`
//! directly) and `int8_thinker_executor.rs` (which drives the same model
//! through the full `residency::Executor` dispatcher/lane machinery) — both
//! must load from the SAME kind of checkpoint via the SAME real loaders for
//! their comparisons to mean anything.

#![allow(dead_code)]

use checkpoint::weightio::{Dtype, StWriter};
use data::rng::Lcg;
use model::int8::quantize_weight;
use omni::config::MoeTextConfig;
use std::collections::HashMap;

pub fn tiny_cfg(n_layers: u32) -> MoeTextConfig {
    MoeTextConfig {
        n_layers,
        hidden: 16,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 8,
        moe_intermediate: 12,
        shared_expert_intermediate: 0,
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

pub fn expert_name(l: usize, e: usize, leaf: &str) -> String {
    format!("thinker.blocks.{l}.mlp.experts.{e}.{leaf}.weight")
}

/// Per-device USABLE capacity that forces the checkpoint at `path` to be
/// placed across exactly `n` of `devices`.
///
/// Placement is capacity-driven (`model::shard::plan_fewest_devices`), so a
/// test that wants a 2-way split has to make one card genuinely too small -
/// handing out "generous" budgets would now correctly put everything on card
/// 0.
///
/// The capacity is found by ASKING the real planner rather than by an
/// arithmetic guess: stage count falls monotonically as capacity rises, so
/// the smallest capacity on a fine grid that yields exactly `n` stages is the
/// one wanted. A closed-form `total/n + slack` looks right and quietly is not:
/// it has to clear a whole layer's granularity plus the head weight, which
/// depends on the fixture's shape, and when it fails to it produces a plan
/// with the WRONG number of stages rather than an error.
pub fn caps_for_split(path: &str, cfg: &MoeTextConfig, devices: &[residency::Device], n: usize) -> Vec<(residency::Device, u64)> {
    use residency::MultiDeviceResidentModel;
    let full_cfg = omni::config::ThinkerConfig::defaults().with_text(cfg.clone());
    let probe = omni::int8_thinker_resident::Int8ThinkerResident::new(path.to_string(), full_cfg.clone(), Vec::new());
    let total = probe.total_device_bytes().expect("synthetic checkpoint must be measurable");
    let key = residency::InstanceKey::new(omni::int8_thinker_resident::MODEL, "default");
    const STEPS: u64 = 128;
    for i in 1..=STEPS {
        let cap = (total * i).div_ceil(STEPS);
        let caps: Vec<(residency::Device, u64)> = devices.iter().map(|&d| (d, cap)).collect();
        let r = omni::int8_thinker_resident::Int8ThinkerResident::new(path.to_string(), full_cfg.clone(), caps.clone());
        if r.estimate_multi(&key).devices().count() == n {
            return caps;
        }
    }
    panic!("no per-device capacity over {} device(s) produces an {n}-way split of a {total}-byte checkpoint", devices.len());
}

/// A tiny synthetic int8 checkpoint, real `omni::import` naming/dtypes for
/// BOTH the routed-expert tensors `ThinkerInt8Store` reads AND the
/// attention/norm/router tensors `int8_thinker_resident::load_layer_bufs`
/// reads — everything `Int8ThinkerInstance::forward` actually needs, so the
/// sharded and unsharded paths in a comparison test both load from the same
/// checkpoint via the same real loaders, not a mix of real and synthetic.
pub fn write_synthetic_checkpoint(path: &str, cfg: &MoeTextConfig, seed: u64) {
    let mut rng = Lcg::new(seed);
    let (d, ff) = (cfg.hidden as usize, cfg.moe_intermediate as usize);
    let (hd, nh, nkv, ne) = (cfg.head_dim as usize, cfg.n_heads as usize, cfg.n_kv_heads as usize, cfg.n_experts as usize);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let mut plan: Vec<(String, Vec<u64>, Dtype)> = Vec::new();
    let mut f32_by_name: HashMap<String, Vec<f32>> = HashMap::new();
    let mut packed_by_name: HashMap<String, (Vec<u32>, Vec<f32>)> = HashMap::new();

    // omni::import::should_quantize's own rule: rank-2, last dim a multiple
    // of 4 -- true for every 2-D tensor at this test's shapes, so every one
    // of these ends up quantized, exactly like a real import would (this is
    // what makes the reference path exercise `load_layer_bufs`'s dequant
    // branch, not just its plain-f32 branch).
    fn plan_mat(
        rng: &mut Lcg,
        plan: &mut Vec<(String, Vec<u64>, Dtype)>,
        packed_by_name: &mut HashMap<String, (Vec<u32>, Vec<f32>)>,
        name: String,
        n: usize,
        k: usize,
    ) {
        let w = rng.vec_scaled(n * k, 0.5);
        let (packed, scale) = quantize_weight(&w, n, k);
        plan.push((name.clone(), vec![n as u64, (k / 4) as u64], Dtype::U32));
        plan.push((format!("{name}.scale"), vec![n as u64], Dtype::F32));
        packed_by_name.insert(name, (packed, scale));
    }
    fn plan_vec(rng: &mut Lcg, plan: &mut Vec<(String, Vec<u64>, Dtype)>, f32_by_name: &mut HashMap<String, Vec<f32>>, name: String, n: usize) {
        plan.push((name.clone(), vec![n as u64], Dtype::F32));
        f32_by_name.insert(name, rng.vec_scaled(n, 0.5));
    }

    for l in 0..cfg.n_layers as usize {
        for e in 0..cfg.n_experts as usize {
            for (leaf, n, k) in [("gate", ff, d), ("up", ff, d), ("down", d, ff)] {
                plan_mat(&mut rng, &mut plan, &mut packed_by_name, expert_name(l, e, leaf), n, k);
            }
        }
        let p = |leaf: &str| format!("thinker.blocks.{l}.{leaf}");
        plan_vec(&mut rng, &mut plan, &mut f32_by_name, p("ln1.weight"), d);
        plan_mat(&mut rng, &mut plan, &mut packed_by_name, p("attn.wq.weight"), hq, d);
        plan_mat(&mut rng, &mut plan, &mut packed_by_name, p("attn.wk.weight"), hkv, d);
        plan_mat(&mut rng, &mut plan, &mut packed_by_name, p("attn.wv.weight"), hkv, d);
        plan_mat(&mut rng, &mut plan, &mut packed_by_name, p("attn.wo.weight"), d, hq);
        plan_vec(&mut rng, &mut plan, &mut f32_by_name, p("attn.q_norm.weight"), hd);
        plan_vec(&mut rng, &mut plan, &mut f32_by_name, p("attn.k_norm.weight"), hd);
        plan_vec(&mut rng, &mut plan, &mut f32_by_name, p("ln2.weight"), d);
        plan_mat(&mut rng, &mut plan, &mut packed_by_name, p("mlp.router.weight"), ne, d);
    }
    let vocab = cfg.vocab as usize;
    plan_mat(&mut rng, &mut plan, &mut packed_by_name, "thinker.embed_tokens.weight".to_string(), vocab, d);
    plan_vec(&mut rng, &mut plan, &mut f32_by_name, "thinker.norm.weight".to_string(), d);
    plan_mat(&mut rng, &mut plan, &mut packed_by_name, "thinker.lm_head.weight".to_string(), vocab, d);

    let mut writer = StWriter::create_mixed(path, &plan, &serde_json::Value::Null, None).expect("create synthetic checkpoint");
    for (name, _, dtype) in &plan {
        if let Some(base) = name.strip_suffix(".scale") {
            writer.write(name, &packed_by_name[base].1).expect("write scale");
        } else if *dtype == Dtype::U32 {
            writer.write_u32(name, &packed_by_name[name].0).expect("write packed");
        } else {
            writer.write(name, &f32_by_name[name]).expect("write f32");
        }
    }
    writer.finish().expect("finish synthetic checkpoint");
}
