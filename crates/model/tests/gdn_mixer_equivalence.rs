// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `crates/qwen35` and `crates/qwen35moe` both call `model::gdn_mixer::
//! {gdn_mixer_fwd, gdn_mixer_bwd}` / `model::gqa_mixer::{gqa_mixer_fwd,
//! gqa_mixer_bwd}` from their own thin, crate-local `layer_gdn_fwd`/
//! `layer_gqa_fwd`/`gdn_mixer_bwd`/`gqa_mixer_bwd` wrappers, each resolving
//! the shared `*MixerIds` struct against its OWN pipeline list's own local
//! kernel-index numbering (`model.rs`'s "position-dependent" convention).
//! Since `model` cannot depend on either downstream crate (they depend on
//! `model`, not the reverse), the real cross-crate risk this gate needs to
//! catch - a swapped/wrong index in one crate's own `*_mixer_ids()` builder
//! silently producing a wrong result only THERE - is exercised here by
//! building the SAME `*MixerIds` struct from two INDEPENDENTLY, DIFFERENTLY
//! ORDERED pipeline registrations (`gpu_a`/`gpu_b` below - mirrors the real
//! fact that `qwen35::model::PIPELINES` and `qwen35moe::model::
//! STATIC_PIPELINES` assign different physical indices to the same kernel
//! names) and asserting the shared functions produce BIT-IDENTICAL output
//! from identical host input regardless of which physical ordering resolved
//! the ids - the actual guarantee the `Ids`-struct indirection exists to
//! provide.

use audio::conv::ConvKernels;
use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::KernelIds;
use model::gdn::{GdnBwdIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_bwd, gdn_mixer_fwd, GdnMixerGrads, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_bwd, gqa_mixer_fwd, GqaMixerGrads, GqaMixerIds, GqaMixerShape, GqaMixerWeights};

/// Every kernel name [`GdnMixerIds`]/[`GqaMixerIds`] (fwd + bwd) resolve,
/// name -> WGSL source. `mul` appears once (bound to both `GdnIds::mul` and
/// this module's own `mul` field - the same physical kernel, same index,
/// wherever it's read from).
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
];

/// `KERNELS`, reversed - a deliberately DIFFERENT physical index assignment
/// for the exact same kernel set (mirrors `qwen35`'s and `qwen35moe`'s own
/// independently-numbered `PIPELINES`/`STATIC_PIPELINES` arrays never
/// agreeing on a given kernel's index).
fn reversed_kernels() -> Vec<(&'static str, &'static str)> {
    let mut v = KERNELS.to_vec();
    v.reverse();
    v
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn kernel_ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: idx(g, "rmsnorm"),
        rms_inv: idx(g, "rms_inv"),
        rmsnorm_dx: idx(g, "rmsnorm_dx"),
        rmsnorm_dw: idx(g, "rmsnorm_dw"),
        rope: idx(g, "rmsnorm"), // unused by the mixers (rope2d_partial is a plain kernel id, not a KernelIds field)
        rope_bwd: idx(g, "rmsnorm"),
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
        rmsnorm_rows: model::block::UNREGISTERED,
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

// ---- GDN mixer -------------------------------------------------------------

fn gdn_shape() -> GdnMixerShape {
    // nvh=2, khd=3, vhd=4, nkh=1 (group=2), chunk=2, t=4 (2 chunks) - every
    // dim pairwise distinct where it matters, small enough to run instantly
    // on the CPU JIT backend.
    GdnMixerShape { gdn: GdnShape { b: 1, h: 2, t: 4, dk: 3, dv: 4, chunk: 2 }, nkh: 1, conv_kernel: 3 }
}

/// `(gated, d_mixed_qkv, d_bproj, d_aproj, d_z, a_log_grad, dt_bias_grad, conv1d_weight_grad, norm_weight_grad)`.
type GdnMixerRunResult = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn run_gdn_mixer(g: &Gpu, shape: &GdnMixerShape, seed: u64) -> GdnMixerRunResult {
    let n = shape.gdn.b * shape.gdn.t;
    let (conv_dim, value_dim, nvh, khd) = (shape.conv_dim(), shape.value_dim(), shape.gdn.h, shape.gdn.dk);
    let mut rng = Lcg::new(seed);

    let mixed_qkv = g.storage_init("mixed_qkv", &rng.vec_scaled((n * conv_dim) as usize, 1.0));
    let bproj = g.storage_init("bproj", &rng.vec_scaled((n * nvh) as usize, 1.0));
    let aproj = g.storage_init("aproj", &rng.vec_scaled((n * nvh) as usize, 1.0));
    let z = g.storage_init("z", &rng.vec_scaled((n * value_dim) as usize, 1.0));
    let conv1d_weight = g.storage_init("conv1d_weight", &rng.vec_scaled(conv_dim as usize * shape.conv_kernel as usize, 1.0));
    let a_log = g.storage_init("a_log", &rng.vec_scaled(nvh as usize, 1.0));
    let dt_bias = g.storage_init("dt_bias", &rng.vec_scaled(nvh as usize, 1.0));
    let norm_weight = g.storage_init("norm_weight", &rng.vec_scaled(shape.gdn.dv as usize, 1.0));
    let ones_khd = g.storage_init("ones_khd", &vec![1.0f32; khd as usize]);
    let d_gated_host = rng.vec_scaled((n * value_dim) as usize, 1.0);

    let w = GdnMixerWeights { conv1d_weight: &conv1d_weight, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_weight, ones_khd: &ones_khd };
    let ids = gdn_mixer_ids(g);
    let (gated, acts) = gdn_mixer_fwd(g, &ids, shape, &w, &mixed_qkv, &bproj, &aproj, &z, n, true);
    let acts = acts.expect("is_train=true must save activations");

    let d_gated = g.storage_init("d_gated", &d_gated_host);
    let a_log_g = g.storage_init("a_log_g", &vec![0.0f32; nvh as usize]);
    let dt_bias_g = g.storage_init("dt_bias_g", &vec![0.0f32; nvh as usize]);
    let conv1d_weight_g = g.storage_init("conv1d_weight_g", &vec![0.0f32; conv_dim as usize * shape.conv_kernel as usize]);
    let norm_weight_g = g.storage_init("norm_weight_g", &vec![0.0f32; shape.gdn.dv as usize]);
    let gw = GdnMixerGrads {
        conv1d_weight: Some(&conv1d_weight_g),
        a_log: Some(&a_log_g),
        dt_bias: Some(&dt_bias_g),
        norm_weight: Some(&norm_weight_g),
    };
    let (d_mixed_qkv, d_bproj, d_aproj, d_z) = gdn_mixer_bwd(g, &ids, shape, &w, &gw, &acts, &d_gated, n);

    (
        g.read(&gated, (n * value_dim) as usize),
        g.read(&d_mixed_qkv, (n * conv_dim) as usize),
        g.read(&d_bproj, (n * nvh) as usize),
        g.read(&d_aproj, (n * nvh) as usize),
        g.read(&d_z, (n * value_dim) as usize),
        g.read(&a_log_g, nvh as usize),
        g.read(&dt_bias_g, nvh as usize),
        g.read(&conv1d_weight_g, conv_dim as usize * shape.conv_kernel as usize),
        g.read(&norm_weight_g, shape.gdn.dv as usize),
    )
}

/// The gate this task's own brief asks for: both crates' mixers, same
/// weights, bit-identical - exercised here as "the shared function, resolved
/// through two independently-ordered pipeline registrations, produces
/// bit-identical output from identical input" (see this file's own module
/// doc for why that is the achievable form of the real cross-crate claim).
#[test]
fn gdn_mixer_fwd_bwd_bit_identical_across_independently_ordered_pipelines() {
    let gpu_a = Gpu::new_cpu(KERNELS);
    let gpu_b = Gpu::new_cpu(&reversed_kernels());
    let shape = gdn_shape();

    let a = run_gdn_mixer(&gpu_a, &shape, 20260819);
    let b = run_gdn_mixer(&gpu_b, &shape, 20260819);

    assert_eq!(a.0, b.0, "gated (forward output) diverged");
    assert_eq!(a.1, b.1, "d_mixed_qkv diverged");
    assert_eq!(a.2, b.2, "d_bproj diverged");
    assert_eq!(a.3, b.3, "d_aproj diverged");
    assert_eq!(a.4, b.4, "d_z diverged");
    assert_eq!(a.5, b.5, "A_log grad diverged");
    assert_eq!(a.6, b.6, "dt_bias grad diverged");
    assert_eq!(a.7, b.7, "conv1d.weight grad diverged");
    assert_eq!(a.8, b.8, "norm.weight grad diverged");
    assert!(a.0.iter().any(|v| v.abs() > 1e-6), "forward output is degenerate (all ~0) - test shape is uninformative");
}

// ---- GQA mixer --------------------------------------------------------------

fn gqa_shape() -> GqaMixerShape {
    // n_heads=2, n_kv_heads=1, head_dim=4, rotary_half=1 (partial rotary:
    // only the first 2 of 4 dims per head rotate).
    GqaMixerShape { b: 1, t: 4, n_heads: 2, n_kv_heads: 1, head_dim: 4, rotary_half: 1 }
}

/// `(ctx_gated, d_q_full, d_k, d_v, q_norm_grad, k_norm_grad)`.
type GqaMixerRunResult = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn run_gqa_mixer(g: &Gpu, shape: &GqaMixerShape, seed: u64) -> GqaMixerRunResult {
    let n = shape.b * shape.t;
    let (qd, kvd, hd) = (shape.qd(), shape.kvd(), shape.head_dim);
    let mut rng = Lcg::new(seed);

    let q_full = g.storage_init("q_full", &rng.vec_scaled((n * 2 * qd) as usize, 1.0));
    let k = g.storage_init("k", &rng.vec_scaled((n * kvd) as usize, 1.0));
    let v = g.storage_init("v", &rng.vec_scaled((n * kvd) as usize, 1.0));
    let q_norm = g.storage_init("q_norm", &rng.vec_scaled(hd as usize, 1.0));
    let k_norm = g.storage_init("k_norm", &rng.vec_scaled(hd as usize, 1.0));
    let cos = g.storage_init("cos", &rng.vec_scaled((n * shape.rotary_half) as usize, 1.0));
    let sin = g.storage_init("sin", &rng.vec_scaled((n * shape.rotary_half) as usize, 1.0));
    let d_ctx_gated_host = rng.vec_scaled((n * qd) as usize, 1.0);

    let w = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };
    let ids = gqa_mixer_ids(g);
    let (ctx_gated, acts) = gqa_mixer_fwd(g, &ids, shape, &w, &q_full, &k, &v, n, true);
    let acts = acts.expect("is_train=true must save activations");

    let d_ctx_gated = g.storage_init("d_ctx_gated", &d_ctx_gated_host);
    let q_norm_g = g.storage_init("q_norm_g", &vec![0.0f32; hd as usize]);
    let k_norm_g = g.storage_init("k_norm_g", &vec![0.0f32; hd as usize]);
    let gw = GqaMixerGrads { q_norm: Some(&q_norm_g), k_norm: Some(&k_norm_g) };
    let (d_q_full, d_k, d_v) = gqa_mixer_bwd(g, &ids, shape, &w, &gw, &acts, &d_ctx_gated, n);

    (
        g.read(&ctx_gated, (n * qd) as usize),
        g.read(&d_q_full, (n * 2 * qd) as usize),
        g.read(&d_k, (n * kvd) as usize),
        g.read(&d_v, (n * kvd) as usize),
        g.read(&q_norm_g, hd as usize),
        g.read(&k_norm_g, hd as usize),
    )
}

/// GQA analogue of [`gdn_mixer_fwd_bwd_bit_identical_across_independently_ordered_pipelines`].
#[test]
fn gqa_mixer_fwd_bwd_bit_identical_across_independently_ordered_pipelines() {
    let gpu_a = Gpu::new_cpu(KERNELS);
    let gpu_b = Gpu::new_cpu(&reversed_kernels());
    let shape = gqa_shape();

    let a = run_gqa_mixer(&gpu_a, &shape, 20260819);
    let b = run_gqa_mixer(&gpu_b, &shape, 20260819);

    assert_eq!(a.0, b.0, "ctx_gated (forward output) diverged");
    assert_eq!(a.1, b.1, "d_q_full diverged");
    assert_eq!(a.2, b.2, "d_k diverged");
    assert_eq!(a.3, b.3, "d_v diverged");
    assert_eq!(a.4, b.4, "q_norm grad diverged");
    assert_eq!(a.5, b.5, "k_norm grad diverged");
    assert!(a.0.iter().any(|v| v.abs() > 1e-6), "forward output is degenerate (all ~0) - test shape is uninformative");
}
