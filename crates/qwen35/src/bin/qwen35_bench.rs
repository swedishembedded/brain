// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Performance pass: where one GDN layer, one GQA layer, and the dense
//! SwiGLU MLP actually spend their time, per kernel kind, at
//! `Qwen35Config::qwen38_27b()`'s real dimensions - profile-first, per the
//! task's own rule ("a native device-side FP8 GEMM belongs here and only if
//! the profile says arithmetic is the limiter - a precision change is not a
//! speed change").
//!
//! Random weights throughout: cost depends on shape, not values (same
//! discipline `qwen3::qwen_bench` and this crate's own real-dims
//! vision-tower parity test (`vision_parity.rs`) already use for a shape too
//! large to hold real weights for comfortably). This is valid for cost,
//! meaningless for output quality.
//!
//! This box has no discrete GPU (Intel iGPU only) - every number here is
//! iGPU-and-CPU-JIT-relative, not a datacenter-GPU projection. `[gflops]`/
//! `[gbs]` columns grade against THIS device's own MEASURED roofline
//! (`gpu_core::roof`), so the numbers stay honest about what device produced
//! them.
//!
//! Usage:
//!   qwen35_bench gdn   [T] [reps]   # one Gated-DeltaNet layer
//!   qwen35_bench gqa   [T] [reps]   # one gated-GQA layer
//!   qwen35_bench mlp   [T] [reps]   # the dense SwiGLU MLP
//!   qwen35_bench all   [T] [reps]   # all three (default)

use std::collections::HashMap;
use std::time::Instant;

use audio::conv::ConvKernels;
use data::rng::Lcg;
use gpu_core::roof::Roofs;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{swiglu_fwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_fwd, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_fwd, GqaMixerIds, GqaMixerShape, GqaMixerWeights};
use qwen35::config::Qwen35Config;

const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("bmm", kernels::BMM),
    ("bmm_acc", kernels::BMM_ACC),
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP),
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK),
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER),
    ("gdn_ut_step", kernels::GDN_UT_STEP),
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY),
    ("scale_row", kernels::SCALE_ROW),
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF),
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE),
    ("gdn_state_decay", kernels::GDN_STATE_DECAY),
    ("exp", kernels::EXP),
    ("sub", kernels::SUB),
    ("mul", kernels::MUL),
    ("region_copy", kernels::REGION_COPY),
    ("splice_add", kernels::SPLICE_ADD),
    ("row_dot", kernels::ROW_DOT),
    ("scale_add", kernels::SCALE_ADD),
    ("gdn_chunk_reverse_cumsum_step", kernels::GDN_CHUNK_REVERSE_CUMSUM_STEP),
    ("gdn_ut_bwd_dattn0", kernels::GDN_UT_BWD_DATTN0),
    ("gdn_ut_bwd_dtmat", kernels::GDN_UT_BWD_DTMAT),
    ("gdn_mask_strict_lower_bwd", kernels::GDN_MASK_STRICT_LOWER_BWD),
    ("gdn_decay_mask_bwd", kernels::GDN_DECAY_MASK_BWD),
    ("gdn_decay_scale_bwd", kernels::GDN_DECAY_SCALE_BWD),
    ("gdn_decay_scale_bwd_last", kernels::GDN_DECAY_SCALE_BWD_LAST),
    ("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("concat2", kernels::CONCAT2),
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("gdn_decay_gate", kernels::GDN_DECAY_GATE),
    ("gdn_decay_gate_bwd", kernels::GDN_DECAY_GATE_BWD),
    ("kv_expand", kernels::KV_EXPAND),
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD),
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE),
    ("bias_grad", kernels::BIAS_GRAD),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("matmul", kernels::MATMUL),
    ("add2", kernels::ADD2),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn kernel_ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: idx(g, "rmsnorm"),
        rms_inv: idx(g, "rms_inv"),
        rmsnorm_dx: idx(g, "rmsnorm_dx"),
        rmsnorm_dw: idx(g, "rmsnorm_dw"),
        // M-RoPE (`rope2d`) is what rotates here, never `block::rope_fwd` -
        // so UNREGISTERED, not a stand-in `rmsnorm` index (see
        // `model::block::UNREGISTERED`).
        rope: model::block::UNREGISTERED,
        rope_bwd: model::block::UNREGISTERED,
        gqa_scores: idx(g, "gqa_scores"),
        gqa_apply: idx(g, "gqa_apply"),
        attn_softmax: idx(g, "attn_softmax"),
        gqa_dscores: idx(g, "gqa_bwd_dscores"),
        gqa_dv: idx(g, "gqa_bwd_dv"),
        gqa_dq: idx(g, "gqa_bwd_dq"),
        gqa_dk: idx(g, "gqa_bwd_dk"),
        silu_mul: idx(g, "silu_mul"),
        silu_da: idx(g, "silu_bwd_da"),
        silu_db: idx(g, "silu_bwd_db"),
    }
}

fn gdn_mixer_ids(g: &Gpu) -> GdnMixerIds {
    GdnMixerIds {
        kernels: kernel_ids(g),
        conv: ConvKernels { fwd: idx(g, "conv1d"), dx: idx(g, "conv1d_dx"), dw: idx(g, "conv1d_dw") },
        chunk: GdnIds {
            bmm: idx(g, "bmm"),
            bmm_acc: idx(g, "bmm_acc"),
            cumsum_step: idx(g, "gdn_chunk_cumsum_step"),
            decay_mask: idx(g, "gdn_decay_mask"),
            mask_strict_lower: idx(g, "gdn_mask_strict_lower"),
            ut_step: idx(g, "gdn_ut_step"),
            add_identity: idx(g, "gdn_add_identity"),
            row_scale: idx(g, "scale_row"),
            row_scale_off: idx(g, "gdn_row_scale_off"),
            decay_scale: idx(g, "gdn_decay_scale"),
            state_decay: idx(g, "gdn_state_decay"),
            exp: idx(g, "exp"),
            sub: idx(g, "sub"),
            mul: idx(g, "mul"),
            region_copy: idx(g, "region_copy"),
        },
        chunk_bwd: GdnBwdIds {
            splice_add: idx(g, "splice_add"),
            row_dot: idx(g, "row_dot"),
            scale_add: idx(g, "scale_add"),
            reverse_cumsum_step: idx(g, "gdn_chunk_reverse_cumsum_step"),
            ut_bwd_dattn0: idx(g, "gdn_ut_bwd_dattn0"),
            ut_bwd_dtmat: idx(g, "gdn_ut_bwd_dtmat"),
            mask_strict_lower_bwd: idx(g, "gdn_mask_strict_lower_bwd"),
            decay_mask_bwd: idx(g, "gdn_decay_mask_bwd"),
            decay_scale_bwd: idx(g, "gdn_decay_scale_bwd"),
            decay_scale_bwd_last: idx(g, "gdn_decay_scale_bwd_last"),
            state_decay_bwd_dscale: idx(g, "gdn_state_decay_bwd_dscale"),
        },
        nlc_nchw: idx(g, "nlc_nchw"),
        nchw_nlc: idx(g, "nchw_nlc"),
        silu: idx(g, "silu"),
        silu_bwd: idx(g, "silu_bwd"),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        l2norm_scale: idx(g, "l2norm_scale"),
        l2norm_scale_dx: idx(g, "l2norm_scale_dx"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        gdn_decay_gate: idx(g, "gdn_decay_gate"),
        gdn_decay_gate_bwd: idx(g, "gdn_decay_gate_bwd"),
        kv_expand: idx(g, "kv_expand"),
        kv_expand_bwd: idx(g, "kv_expand_bwd"),
        gdn_layout_permute: idx(g, "gdn_layout_permute"),
        mul: idx(g, "mul"),
        bias_grad: idx(g, "bias_grad"),
    }
}

fn gqa_mixer_ids(g: &Gpu) -> GqaMixerIds {
    GqaMixerIds {
        kernels: kernel_ids(g),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        mul: idx(g, "mul"),
        rope2d_partial: idx(g, "rope2d_partial"),
    }
}

fn rand_buf(g: &Gpu, rng: &mut Lcg, n: usize) -> DeviceBuffer {
    g.storage_init("bench", &rng.vec_scaled(n, 0.02))
}

/// Run `f` `reps` times under device kernel timing, print a table sorted by
/// time descending: `kernel | ms | calls | % of pass`. Falls back to a
/// one-line notice when this backend cannot time kernels at all (the CPU
/// JIT backend: `lessons.md` #31, host-bracketed timing inflates small
/// kernels up to 29x, so it is not printed as if it were the same kind of
/// number).
fn report(gpu: &Gpu, label: &str, reps: usize, mut f: impl FnMut()) {
    let stats0 = gpu.stats();
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let wall = t0.elapsed().as_secs_f64() / reps as f64;
    let stats1 = gpu.stats();
    let dispatches_per_rep = match (stats0, stats1) {
        (Some(a), Some(b)) => Some((b.dispatches - a.dispatches) / reps as u64),
        _ => None,
    };
    let dispatch_note = match dispatches_per_rep {
        Some(n) => format!(", {n} dispatches/rep"),
        None => String::new(),
    };

    if !gpu.set_kernel_timing(true) {
        println!("\n=== {label} === wall {:.3} ms/rep{dispatch_note} (device cannot time individual kernels on this backend)", wall * 1e3);
        return;
    }
    gpu.reset_kernel_times();
    f();
    gpu.poll_wait();
    let times = gpu.kernel_times().unwrap_or_default();
    gpu.set_kernel_timing(false);

    let total: f64 = times.iter().map(|(_, ms, _)| ms).sum();
    println!("\n=== {label} === wall {:.3} ms/rep{dispatch_note}, device-timed pass {:.3} ms", wall * 1e3, total);
    if total <= 0.0 {
        // The backend CLAIMED support but recorded nothing (a real, observed
        // possibility `gpu_core::profile::profile`'s own doc warns about) -
        // printing an all-zero table would misreport "every kernel is free"
        // instead of "this number is not available here". Wall-clock +
        // dispatch count above are the honest evidence on this backend.
        println!("(per-kernel device timing unavailable on this backend/run - see wall-clock above)");
        return;
    }
    let mut rows = times;
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("{:<28} {:>10} {:>8} {:>7}", "kernel", "ms", "calls", "%");
    for (name, ms, calls) in &rows {
        println!("{name:<28} {ms:>10.4} {calls:>8} {:>6.1}%", 100.0 * ms / total.max(1e-9));
    }
}

fn bench_gdn(gpu: &Gpu, cfg: &Qwen35Config, t: u32, reps: usize) {
    let mut rng = Lcg::new(7);
    let d = cfg.d_model;
    let conv_dim = cfg.linear_conv_dim();
    let value_dim = cfg.linear_value_dim();
    let nvh = cfg.linear_num_value_heads;
    let khd = cfg.linear_key_head_dim;
    let vhd = cfg.linear_value_head_dim;

    let xn1 = rand_buf(gpu, &mut rng, (t * d) as usize);
    let qkvw = rand_buf(gpu, &mut rng, (conv_dim * d) as usize);
    let bw = rand_buf(gpu, &mut rng, (nvh * d) as usize);
    let aw = rand_buf(gpu, &mut rng, (nvh * d) as usize);
    let zw = rand_buf(gpu, &mut rng, (value_dim * d) as usize);
    let outw = rand_buf(gpu, &mut rng, (d * value_dim) as usize);
    let conv1d_weight = rand_buf(gpu, &mut rng, (conv_dim * cfg.linear_conv_kernel_dim) as usize);
    let a_log = rand_buf(gpu, &mut rng, nvh as usize);
    let dt_bias = rand_buf(gpu, &mut rng, nvh as usize);
    let norm_weight = rand_buf(gpu, &mut rng, vhd as usize);
    let ones_khd = gpu.storage_init("ones_khd", &vec![1.0f32; khd as usize]);

    let ids = gdn_mixer_ids(gpu);
    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t, dk: khd, dv: vhd, chunk: model::gdn::gdn_chunk_size(t) }, nkh: cfg.linear_num_key_heads, conv_kernel: cfg.linear_conv_kernel_dim };
    let weights = GdnMixerWeights { conv1d_weight: &conv1d_weight, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_weight, ones_khd: &ones_khd };

    report(gpu, &format!("GDN layer (T={t})"), reps, || {
        let mixed_qkv = gpu.storage((t * conv_dim) as u64);
        let bproj = gpu.storage((t * nvh) as u64);
        let aproj = gpu.storage((t * nvh) as u64);
        let z = gpu.storage((t * value_dim) as u64);
        gpu.submit(
            &[],
            &[
                gpu.step(idx(gpu, "matmul"), &[&xn1, &qkvw, &mixed_qkv], &[t, d, conv_dim], t * conv_dim),
                gpu.step(idx(gpu, "matmul"), &[&xn1, &bw, &bproj], &[t, d, nvh], t * nvh),
                gpu.step(idx(gpu, "matmul"), &[&xn1, &aw, &aproj], &[t, d, nvh], t * nvh),
                gpu.step(idx(gpu, "matmul"), &[&xn1, &zw, &z], &[t, d, value_dim], t * value_dim),
            ],
        );
        let (gated, _) = gdn_mixer_fwd(gpu, &ids, &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, t, false);
        let out = gpu.storage((t * d) as u64);
        gpu.submit(&[], &[gpu.step(idx(gpu, "matmul"), &[&gated, &outw, &out], &[t, value_dim, d], t * d)]);
    });
}

fn bench_gqa(gpu: &Gpu, cfg: &Qwen35Config, t: u32, reps: usize) {
    let mut rng = Lcg::new(11);
    let d = cfg.d_model;
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (qpd, kvd) = (cfg.q_proj_dim(), cfg.kv_dim());

    let xn1 = rand_buf(gpu, &mut rng, (t * d) as usize);
    let qw = rand_buf(gpu, &mut rng, (qpd * d) as usize);
    let kw = rand_buf(gpu, &mut rng, (kvd * d) as usize);
    let vw = rand_buf(gpu, &mut rng, (kvd * d) as usize);
    let ow = rand_buf(gpu, &mut rng, (d * cfg.q_dim()) as usize);
    let q_norm = rand_buf(gpu, &mut rng, hd as usize);
    let k_norm = rand_buf(gpu, &mut rng, hd as usize);
    let (cos_h, sin_h) = qwen3vl::mrope::mrope_tables(&(0..t).map(|ti| [ti, ti, ti]).collect::<Vec<_>>(), cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_h);
    let sin = gpu.storage_init("sin", &sin_h);

    let ids = gqa_mixer_ids(gpu);
    let shape = GqaMixerShape { b: 1, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2 };
    let weights = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };

    report(gpu, &format!("GQA layer (T={t})"), reps, || {
        let q_full = gpu.storage((t * qpd) as u64);
        let k = gpu.storage((t * kvd) as u64);
        let v = gpu.storage((t * kvd) as u64);
        gpu.submit(
            &[],
            &[
                gpu.step(idx(gpu, "matmul"), &[&xn1, &qw, &q_full], &[t, d, qpd], t * qpd),
                gpu.step(idx(gpu, "matmul"), &[&xn1, &kw, &k], &[t, d, kvd], t * kvd),
                gpu.step(idx(gpu, "matmul"), &[&xn1, &vw, &v], &[t, d, kvd], t * kvd),
            ],
        );
        let (ctx_gated, _) = gqa_mixer_fwd(gpu, &ids, &shape, &weights, &q_full, &k, &v, t, false);
        let out = gpu.storage((t * d) as u64);
        gpu.submit(&[], &[gpu.step(idx(gpu, "matmul"), &[&ctx_gated, &ow, &out], &[t, shape.qd(), d], t * d)]);
    });
}

fn bench_mlp(gpu: &Gpu, cfg: &Qwen35Config, t: u32, reps: usize) {
    let mut rng = Lcg::new(13);
    let d = cfg.d_model;
    let ff = cfg.intermediate_size;
    let xn2 = rand_buf(gpu, &mut rng, (t * d) as usize);
    let gatew = rand_buf(gpu, &mut rng, (ff * d) as usize);
    let upw = rand_buf(gpu, &mut rng, (ff * d) as usize);
    let downw = rand_buf(gpu, &mut rng, (d * ff) as usize);
    let kids = kernel_ids(gpu);

    report(gpu, &format!("dense SwiGLU MLP (T={t})"), reps, || {
        let gate_pre = gpu.storage((t * ff) as u64);
        let up = gpu.storage((t * ff) as u64);
        gpu.submit(
            &[],
            &[
                gpu.step(idx(gpu, "matmul"), &[&xn2, &gatew, &gate_pre], &[t, d, ff], t * ff),
                gpu.step(idx(gpu, "matmul"), &[&xn2, &upw, &up], &[t, d, ff], t * ff),
            ],
        );
        let h_act = gpu.storage((t * ff) as u64);
        gpu.submit(&[], &[swiglu_fwd(gpu, &kids, &gate_pre, &up, &h_act, t * ff)]);
        let down = gpu.storage((t * d) as u64);
        gpu.submit(&[], &[gpu.step(idx(gpu, "matmul"), &[&h_act, &downw, &down], &[t, ff, d], t * d)]);
    });
}

/// Per-layer parameter/FLOP accounting, offline (no device) - the "what
/// SHOULD this cost" baseline `report`'s measured numbers are judged
/// against, mirroring `qwen3::qwen_bench`'s own `cost` subcommand.
fn print_cost_table(cfg: &Qwen35Config, t: u64) {
    let d = cfg.d_model as u64;
    let ff = cfg.intermediate_size as u64;
    let conv_dim = cfg.linear_conv_dim() as u64;
    let value_dim = cfg.linear_value_dim() as u64;
    let nvh = cfg.linear_num_value_heads as u64;
    let (qpd, qd, kvd) = (cfg.q_proj_dim() as u64, cfg.q_dim() as u64, cfg.kv_dim() as u64);

    let gdn_gemm_flops = 2 * t * d * conv_dim + 2 * t * d * nvh * 2 + 2 * t * d * value_dim + 2 * t * value_dim * d;
    let gqa_gemm_flops = 2 * t * d * qpd + 2 * t * d * kvd * 2 + 2 * t * qd * d;
    let mlp_gemm_flops = 2 * t * d * ff * 2 + 2 * t * ff * d;

    println!("\noffline GEMM FLOP accounting (T={t}, forward only, matmul-family only):");
    let mut rows: HashMap<&str, u64> = HashMap::new();
    rows.insert("GDN layer projections", gdn_gemm_flops);
    rows.insert("GQA layer projections", gqa_gemm_flops);
    rows.insert("dense MLP", mlp_gemm_flops);
    for (name, flops) in &rows {
        println!("  {name:<26} {:>10.3} GFLOP", *flops as f64 / 1e9);
    }
}

fn banner(gpu: &Gpu) -> Option<Roofs> {
    match gpu_core::roof::ensure(gpu) {
        Some(r) => {
            println!("measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte", r.gflops, r.gbs, r.cache_gbs, r.ridge());
            Some(r)
        }
        None => {
            println!("roofline unmeasured on this backend");
            None
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("all");
    let t: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let cfg = Qwen35Config::qwen38_27b();

    eprintln!("qwen35_bench: Qwen3.8-27B real dims, T={t}, {reps} reps, random weights (cost is valid, output is not)");
    print_cost_table(&cfg, t as u64);

    if mode == "cost" {
        return;
    }

    let gpu = Gpu::new(KERNELS);
    eprintln!("backend: {}", gpu.kind());
    banner(&gpu);

    match mode {
        "gdn" => bench_gdn(&gpu, &cfg, t, reps),
        "gqa" => bench_gqa(&gpu, &cfg, t, reps),
        "mlp" => bench_mlp(&gpu, &cfg, t, reps),
        _ => {
            bench_gdn(&gpu, &cfg, t, reps);
            bench_gqa(&gpu, &cfg, t, reps);
            bench_mlp(&gpu, &cfg, t, reps);
        }
    }
}
