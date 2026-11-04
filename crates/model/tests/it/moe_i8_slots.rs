// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sparse mixture-of-experts inference that
// stays on the memory roofline at both decode and prefill. If your team needs
// expertise in int8 expert banks, device-side routing and tensor-core grouped
// GEMMs then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `model::moe_i8::expert_fwd_i8_slots` (int8 expert banks, device-side routing, no host
//! synchronisation) against the fp32 grouped sublayer (`model::moe::expert_fwd_grouped`) run on
//! the very weights the int8 banks hold, dequantised.
//!
//! The two compute the same routed experts with the same router weights. They are not bitwise
//! equal: the slot path also quantises the ACTIVATIONS to int8 (per 32-element group), twice per
//! expert, which the fp32 path does not. So the spec is that the output agrees to the size of
//! that quantisation noise (relative L2 error and cosine), at every row count: the gather GEMV a
//! decode step takes, the grouped GEMM from [`GROUPED_MIN_ROWS`] up, and either router setting
//! (raw DeepSeek-V2 weights, renormalised weights).

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::int8::GROUP;
use model::moe::{expert_fwd_grouped, router_fwd_kind, GroupedExpertFwdIds, GroupedExpertScratch, MoeIds, MoeShape, RouterKind};
use model::moe_i8::{expert_fwd_i8_slots, I8Bank, I8Experts, SlotKernels, SlotRouter, SlotScratch, SlotShape, GROUPED_MIN_ROWS};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = vec![
            ("matmul", kernels::MATMUL),
            ("matmul_reg3", kernels::MATMUL_REG3),
            ("matmul_reg3_grouped", kernels::MATMUL_REG3_GROUPED),
            ("silu_mul", kernels::SILU_MUL),
            ("scale_add", kernels::SCALE_ADD),
            ("router_gate", kernels::ROUTER_GATE),
            ("router_topk_compact", kernels::ROUTER_TOPK_COMPACT),
            ("moe_group_counts", kernels::MOE_GROUP_COUNTS),
            ("moe_group_perm_emit", kernels::MOE_GROUP_PERM_EMIT),
            ("moe_group_combine", kernels::MOE_GROUP_COMBINE),
            ("scan_block", kernels::SCAN_BLOCK),
            ("scan_add", kernels::SCAN_ADD),
            ("embed", kernels::EMBED),
        ];
        v.extend(SlotKernels::pipelines());
        v
    })
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Host-side dequantisation of the bank the int8 path reads: `q * group scale`.
fn dequantised(source: &std::collections::HashMap<String, Vec<f32>>, name: &str, rows: usize, k: usize) -> Vec<f32> {
    let (words, scales) = model::int8::quantize_from(source, name, rows, k).expect("tensor");
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for c in 0..k {
            let word = words[r * (k / 4) + c / 4];
            let q = (word >> (8 * (c % 4))) as u8 as i8;
            out[r * k + c] = q as f32 * scales[r * (k / GROUP) + c / GROUP];
        }
    }
    out
}

fn relative_l2(got: &[f32], want: &[f32]) -> (f32, f32) {
    let (mut num, mut den, mut dot, mut gg) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
        dot += (*a as f64) * (*b as f64);
        gg += (*a as f64).powi(2);
    }
    ((num / den.max(1e-30)).sqrt() as f32, (dot / (gg.sqrt() * den.sqrt()).max(1e-30)) as f32)
}

fn run(rows: u32, norm: bool, scale: f32) {
    let g = gpu_core::testgpu::dev(pipelines());
    if !g.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the slot kernels need workgroup reductions");
    }
    let (d, ff, e, top_k) = (64u32, 64u32, 16u32, 3u32);
    let mut rng = Lcg::new(data::rng::random_seed());
    let mut src = std::collections::HashMap::new();
    src.insert("gate".to_string(), rng.vec_scaled((e * ff * d) as usize, 0.3));
    src.insert("up".to_string(), rng.vec_scaled((e * ff * d) as usize, 0.3));
    src.insert("down".to_string(), rng.vec_scaled((e * d * ff) as usize, 0.3));
    let logits_host = rng.vec_scaled((rows * e) as usize, 2.0);
    let x_host = rng.vec_scaled((rows * d) as usize, 1.0);
    let (logits, x) = (g.storage_init("logits", &logits_host), g.storage_init("x", &x_host));

    // The int8 path.
    let bank = |name: &str, n: u32, k: u32| I8Bank::from_source(&g, &src, name, e as usize, n as usize, k as usize).expect("bank");
    let experts = I8Experts { gate: bank("gate", ff, d), up: bank("up", ff, d), down: bank("down", d, ff) };
    let kernels = SlotKernels::resolve(&g).expect("kernels");
    let scratch = SlotScratch::new(&g, rows, d, ff, e, top_k, 0);
    let out_i8 = g.storage((rows * d) as u64);
    let shape = SlotShape { rows, d_model: d, moe_ff: ff, n_experts: e, top_k, shared: 0 };
    let steps = expert_fwd_i8_slots(&g, &kernels, &shape, SlotRouter { norm, scale }, &x, &logits, &experts, &scratch, GROUPED_MIN_ROWS, &out_i8);
    g.submit(&[], &steps);

    // The fp32 reference on the dequantised weights.
    let upload = |name: &str, n: usize, k: usize| -> DeviceBuffer { g.storage_init(name, &dequantised(&src, name, e as usize * n, k)) };
    let (gate_w, up_w, down_w) = (upload("gate", ff as usize, d as usize), upload("up", ff as usize, d as usize), upload("down", d as usize, ff as usize));
    let ids = GroupedExpertFwdIds {
        router_topk_compact: idx(&g, "router_topk_compact"),
        group_counts: idx(&g, "moe_group_counts"),
        scan_block: idx(&g, "scan_block"),
        scan_add: idx(&g, "scan_add"),
        perm_emit: idx(&g, "moe_group_perm_emit"),
        gather: idx(&g, "embed"),
        gemm_grouped: idx(&g, "matmul_reg3_grouped"),
        silu_mul: idx(&g, "silu_mul"),
        combine: idx(&g, "moe_group_combine"),
    };
    let moe_ids = MoeIds { router_gate: idx(&g, "router_gate"), linear_gated: idx(&g, "matmul"), silu_mul: idx(&g, "silu_mul"), scale_add: idx(&g, "scale_add") };
    let mshape = MoeShape { rows, d_model: d, moe_ff: ff, n_experts: e, top_k };
    let gate = g.storage((rows * e) as u64);
    let sc = GroupedExpertScratch::new(&g, &mshape);
    let out_f32 = g.storage((rows * d) as u64);
    let kind = RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: norm, routed_scaling: scale };
    let mut steps = vec![router_fwd_kind(&g, &moe_ids, kind, &mshape, &logits, None, &gate, None)];
    steps.extend(expert_fwd_grouped(&g, &ids, &mshape, &x, &gate, &gate_w, &up_w, &down_w, &sc, &out_f32));
    g.submit(&[], &steps);

    let (got, want) = (g.read(&out_i8, (rows * d) as usize), g.read(&out_f32, (rows * d) as usize));
    let (rel, cos) = relative_l2(&got, &want);
    println!("moe_i8_slots rows={rows} norm={norm} scale={scale}: relative L2 {rel:.4}, cosine {cos:.5}");
    assert!(got.iter().all(|v| v.is_finite()), "rows={rows} norm={norm} scale={scale}: non-finite output");
    assert!(rel < 0.03 && cos > 0.9995, "rows={rows} norm={norm} scale={scale}: relative L2 {rel}, cosine {cos} - more than activation quantisation noise");
}

/// A decode step (the gather GEMV, one slot per weight pass).
#[test]
fn a_decode_step_matches_the_fp32_sublayer_to_activation_quantisation_noise() {
    run(1, false, 1.0);
    run(4, false, 1.0);
}

/// A prefill chunk (slots ordered by expert, the grouped GEMM).
#[test]
fn a_prefill_chunk_matches_the_fp32_sublayer_to_activation_quantisation_noise() {
    run(GROUPED_MIN_ROWS, false, 1.0);
    run(40, false, 1.0);
}

/// DeepSeek-V2's raw weights, and the renormalised and scaled settings other checkpoints use.
#[test]
fn every_router_setting_matches() {
    run(5, true, 1.0);
    run(5, false, 2.5);
    run(30, true, 2.5);
}

/// The shared expert as one more slot of every row: the output is the routed experts' output plus
/// `sigmoid(gate . x)` times the shared expert alone. Both parts are computed by the same kernels on separate banks,
/// so the combined slot form must equal their host sum to float rounding, in the gather and the grouped regime.
fn run_shared(rows: u32) {
    let g = gpu_core::testgpu::dev(pipelines());
    if !g.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the slot kernels need workgroup reductions");
    }
    let (d, ff, e, top_k) = (64u32, 64u32, 16u32, 3u32);
    let mut rng = Lcg::new(data::rng::random_seed());
    let (mut routed_src, mut shared_src, mut all_src) = (std::collections::HashMap::new(), std::collections::HashMap::new(), std::collections::HashMap::new());
    for (name, n, k) in [("gate", ff, d), ("up", ff, d), ("down", d, ff)] {
        let routed = rng.vec_scaled((e * n * k) as usize, 0.3);
        let shared = rng.vec_scaled((n * k) as usize, 0.3);
        all_src.insert(name.to_string(), [routed.clone(), shared.clone()].concat());
        routed_src.insert(name.to_string(), routed);
        shared_src.insert(name.to_string(), shared);
    }
    let logits_host = rng.vec_scaled((rows * (e + 1)) as usize, 2.0);
    let routed_logits: Vec<f32> = logits_host.chunks(e as usize + 1).flat_map(|r| r[..e as usize].to_vec()).collect();
    let shared_gate: Vec<f32> = logits_host.chunks(e as usize + 1).map(|r| r[e as usize]).collect();
    let x_host = rng.vec_scaled((rows * d) as usize, 1.0);
    let x = g.storage_init("x", &x_host);
    let kernels = SlotKernels::resolve(&g).expect("kernels");
    let experts = |src: &std::collections::HashMap<String, Vec<f32>>, blocks: u32| {
        let bank = |name: &str, n: u32, k: u32| I8Bank::from_source(&g, src, name, blocks as usize, n as usize, k as usize).expect("bank");
        I8Experts { gate: bank("gate", ff, d), up: bank("up", ff, d), down: bank("down", d, ff) }
    };
    let call = |experts: &I8Experts, logits: &[f32], experts_n: u32, k: u32, shared: u32, router: SlotRouter| {
        let scratch = SlotScratch::new(&g, rows, d, ff, experts_n, k, shared);
        let (logits, out) = (g.storage_init("logits", logits), g.storage((rows * d) as u64));
        let shape = SlotShape { rows, d_model: d, moe_ff: ff, n_experts: experts_n, top_k: k, shared };
        g.submit(&[], &expert_fwd_i8_slots(&g, &kernels, &shape, router, &x, &logits, experts, &scratch, GROUPED_MIN_ROWS, &out));
        g.read(&out, (rows * d) as usize)
    };
    let router = SlotRouter { norm: true, scale: 1.0 };
    let combined = call(&experts(&all_src, e + 1), &logits_host, e, top_k, 1, router);
    let routed = call(&experts(&routed_src, e), &routed_logits, e, top_k, 0, router);
    // One expert, one slot, router weight exactly 1: the shared expert's own output.
    let shared_out = call(&experts(&shared_src, 1), &vec![0.0; rows as usize], 1, 1, 0, router);
    let want: Vec<f32> = (0..(rows * d) as usize)
        .map(|i| {
            let gate = 1.0 / (1.0 + (-shared_gate[i / d as usize]).exp());
            routed[i] + gate * shared_out[i]
        })
        .collect();
    let (rel, _) = relative_l2(&combined, &want);
    println!("moe_i8_slots shared rows={rows}: relative L2 {rel:.2e}");
    assert!(combined.iter().all(|v| v.is_finite()) && rel < 1e-4, "rows={rows}: the shared slot differs from routed + gate * shared by {rel}");
}

#[test]
fn a_shared_expert_slot_adds_its_gated_output_to_the_routed_experts() {
    run_shared(1);
    run_shared(12);
}
