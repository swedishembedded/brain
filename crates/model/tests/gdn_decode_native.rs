// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native `gdn_decode` kernel (`kernels_cuda`, `cu/gdn_decode.cu`) - one
//! Gated DeltaNet decode step in a single launch - against the nineteen-kernel
//! WGSL chain `gdn_mixer_decode_fwd` dispatches, on the same device from the
//! same inputs.
//!
//! Swedish Embedded AB implements bit-exact fused kernels for linear-attention
//! decode. If your team needs expertise in replacing a recurrent layer's chain
//! of tiny kernels with one launch without moving a single output bit, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! The claim is BYTE-identity of everything the step produces: the layer's
//! output, the recurrent state, and the conv window it rewrites in place. The
//! kernel keeps every reduction in the chain's own order (the conv taps, the L2
//! and RMS sums, the two state contractions over the key index), so identity is
//! achievable and a tolerance would only hide a defect.
//!
//! Inputs start from RANDOM non-zero state and window - a zero state makes the
//! delta-rule update and most layout mix-ups invisible - and cover every key
//! head count and group size the block shape serves.

use audio::conv::ConvKernels;
use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{KernelIds, UNREGISTERED};
use model::gdn::{GdnBwdIds, GdnConvIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_decode_fused, gdn_mixer_decode_fwd, gdn_mixer_decode_state_fwd, gdn_mixer_rows_fused, GdnDecodeState, GdnMixerDecodeIds, GdnMixerIds, GdnMixerShape, GdnMixerWeights, GdnPoolRows, GdnStream};

const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("conv1d", kernels::CONV1D),
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
    ("nlc_nchw", kernels::NLC_NCHW),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("silu", kernels::SILU),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("concat2", kernels::CONCAT2),
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("sigmoid", kernels::SIGMOID),
    ("gdn_decay_gate", kernels::GDN_DECAY_GATE),
    ("kv_expand", kernels::KV_EXPAND),
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE),
    ("causal_conv1d_step", kernels::CAUSAL_CONV1D_STEP),
    ("splice", kernels::SPLICE),
    ("pool_rows_gather2", kernels::POOL_ROWS_GATHER2),
    ("pool_rows_scatter2", kernels::POOL_ROWS_SCATTER2),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn ids(g: &Gpu) -> GdnMixerIds {
    let bwd = UNREGISTERED;
    GdnMixerIds {
        kernels: KernelIds {
            rmsnorm: idx(g, "rmsnorm"),
            rms_inv: bwd,
            rmsnorm_dx: bwd,
            rmsnorm_dx_rows: bwd,
            rmsnorm_dw: bwd,
            rope: bwd,
            rope_bwd: bwd,
            gqa_scores: bwd,
            gqa_apply: bwd,
            attn_softmax: bwd,
            gqa_dscores: bwd,
            gqa_dv: bwd,
            gqa_dq: bwd,
            gqa_dk: bwd,
            silu_mul: bwd,
            silu_da: bwd,
            silu_db: bwd,
            rmsnorm_rows: bwd,
        },
        conv: ConvKernels { fwd: idx(g, "conv1d"), dx: bwd, dw: bwd },
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
            fast: None,
        },
        chunk_bwd: GdnBwdIds {
            splice_add: bwd,
            row_dot: bwd,
            scale_add: bwd,
            reverse_cumsum_step: bwd,
            ut_bwd_dattn0: bwd,
            ut_bwd_dtmat: bwd,
            mask_strict_lower_bwd: bwd,
            decay_mask_bwd: bwd,
            decay_scale_bwd: bwd,
            decay_scale_bwd_last: bwd,
            state_decay_bwd_dscale: bwd,
        },
        nlc_nchw: idx(g, "nlc_nchw"),
        nchw_nlc: idx(g, "nchw_nlc"),
        silu: idx(g, "silu"),
        silu_bwd: bwd,
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        l2norm_scale: idx(g, "l2norm_scale"),
        l2norm_scale_dx: bwd,
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: bwd,
        gdn_decay_gate: idx(g, "gdn_decay_gate"),
        gdn_decay_gate_bwd: bwd,
        kv_expand: idx(g, "kv_expand"),
        kv_expand_bwd: bwd,
        gdn_layout_permute: idx(g, "gdn_layout_permute"),
        mul: idx(g, "mul"),
        bias_grad: bwd,
    }
}

fn dec_ids(g: &Gpu) -> GdnMixerDecodeIds {
    GdnMixerDecodeIds { conv: GdnConvIds { causal_conv1d_step: idx(g, "causal_conv1d_step") }, splice: idx(g, "splice") }
}


fn rnd(r: &mut Lcg, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| r.scaled(s)).collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Whether this device is offered the native kernels: a CUDA device, with
/// `BRAIN_NO_NATIVE_KERNELS` (the A/B switch that pins the WGSL tier) unset.
/// Under it these gates skip, and the "offered only where it can run" test
/// asserts the withholding.
fn is_cuda(g: &Gpu) -> bool {
    g.kind() == "cuda" && g.caps().arch.compute_capability.is_some() && !std::env::var("BRAIN_NO_NATIVE_KERNELS").is_ok_and(|v| v != "0")
}

/// Run the chain and the fused kernel from identical inputs and compare every
/// output bit.
fn check(g: &Gpu, nkh: u32, group: u32, seed: u64) {
    let (dk, dv, kw) = (128u32, 128u32, 4u32);
    let nvh = nkh * group;
    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
    let (value_dim, conv_dim) = (shape.value_dim(), shape.conv_dim());
    let state_len = (nvh * dk * dv) as usize;
    let hist_len = (conv_dim * (kw - 1)) as usize;

    let mut r = Lcg::new(seed);
    let conv_w = g.storage_init("conv_w", &rnd(&mut r, (conv_dim * kw) as usize, 0.5));
    let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
    let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
    let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
    let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
    let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };

    let mixed = g.storage_init("mixed", &rnd(&mut r, conv_dim as usize, 1.0));
    let bp = g.storage_init("bp", &rnd(&mut r, nvh as usize, 1.0));
    let ap = g.storage_init("ap", &rnd(&mut r, nvh as usize, 1.0));
    let z = g.storage_init("z", &rnd(&mut r, value_dim as usize, 1.0));
    let state0 = rnd(&mut r, state_len, 0.3);
    let hist0 = rnd(&mut r, hist_len, 0.3);

    // The chain.
    let (s_ref, h_ref) = (g.storage_init("state", &state0), g.storage_init("hist", &hist0));
    let gated_ref = gdn_mixer_decode_fwd(g, &ids(g), &dec_ids(g), &shape, &w, &mixed, &bp, &ap, &z, &[GdnStream { state: &s_ref, hist: &h_ref }]);
    // The fused kernel.
    let (s_nat, h_nat) = (g.storage_init("state", &state0), g.storage_init("hist", &hist0));
    let gated_nat = gdn_mixer_decode_fused(g, &shape, &w, &mixed, &bp, &ap, &z, &GdnStream { state: &s_nat, hist: &h_nat })
        .expect("the fused GDN decode kernel was declined on a CUDA device");
    g.poll_wait();

    let ctx = format!("nkh={nkh} group={group} seed={seed:#x}");
    assert_eq!(bits(g.read(&gated_nat, value_dim as usize)), bits(g.read(&gated_ref, value_dim as usize)), "layer output differs: {ctx}");
    assert_eq!(bits(g.read(&s_nat, state_len)), bits(g.read(&s_ref, state_len)), "recurrent state differs: {ctx}");
    assert_eq!(bits(g.read(&h_nat, hist_len)), bits(g.read(&h_ref, hist_len)), "conv window differs: {ctx}");
}

#[test]
fn the_fused_step_is_offered_only_where_it_can_run() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (nkh, group) = (2u32, 3u32);
    let shape = |dk: u32, dv: u32, kw: u32| GdnMixerShape { gdn: GdnShape { b: 1, h: nkh * group, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
    let dummy = g.storage(4);
    let w = GdnMixerWeights { conv1d_weight: &dummy, a_log: &dummy, dt_bias: &dummy, norm_weight: &dummy, ones_khd: &dummy };
    let stream = GdnStream { state: &dummy, hist: &dummy };
    let try_shape = |s: &GdnMixerShape| gdn_mixer_decode_fused(&g, s, &w, &dummy, &dummy, &dummy, &dummy, &stream).is_some();
    if !is_cuda(&g) {
        assert!(!try_shape(&shape(128, 128, 4)), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // Shapes outside the kernel's block shape keep the chain.
    assert!(!try_shape(&shape(64, 128, 4)), "key head dim other than 128");
    assert!(!try_shape(&shape(128, 64, 4)), "value head dim other than 128");
    assert!(!try_shape(&shape(128, 128, 3)), "a conv kernel other than 4 taps");
    let batched = GdnMixerShape { gdn: GdnShape { b: 2, ..shape(128, 128, 4).gdn }, ..shape(128, 128, 4) };
    assert!(!try_shape(&batched), "a batch of more than one sequence");
}

#[test]
fn the_fused_step_is_byte_identical_to_the_wgsl_chain() {
    let g = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    // Group 1, 2 and 3 (the block is group x 128 threads), a single key head,
    // and the real Qwen3.8-27B head counts (16 key heads x 3).
    for (nkh, group) in [(1u32, 1u32), (1, 2), (1, 3), (2, 1), (2, 2), (3, 3), (4, 3), (16, 3)] {
        for seed in [1u64, 0xdead_beef] {
            check(&g, nkh, group, seed ^ u64::from(nkh * 7 + group));
        }
    }
}

/// Several consecutive steps from one starting state: the window and the state
/// feed the next token, so an error that is invisible in step one compounds.
#[test]
fn a_run_of_steps_stays_byte_identical() {
    let g = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let (nkh, group, dk, dv, kw) = (4u32, 3u32, 128u32, 128u32, 4u32);
    let nvh = nkh * group;
    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
    let (value_dim, conv_dim) = (shape.value_dim(), shape.conv_dim());
    let state_len = (nvh * dk * dv) as usize;
    let hist_len = (conv_dim * (kw - 1)) as usize;
    let mut r = Lcg::new(0xfeed_f00d);
    let conv_w = g.storage_init("conv_w", &rnd(&mut r, (conv_dim * kw) as usize, 0.5));
    let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
    let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
    let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
    let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
    let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };
    let (s_ref, h_ref) = (g.storage(state_len as u64), g.storage(hist_len as u64));
    let (s_nat, h_nat) = (g.storage(state_len as u64), g.storage(hist_len as u64));
    for step in 0..12 {
        let mixed = g.storage_init("mixed", &rnd(&mut r, conv_dim as usize, 1.0));
        let bp = g.storage_init("bp", &rnd(&mut r, nvh as usize, 1.0));
        let ap = g.storage_init("ap", &rnd(&mut r, nvh as usize, 1.0));
        let z = g.storage_init("z", &rnd(&mut r, value_dim as usize, 1.0));
        let a = gdn_mixer_decode_fwd(&g, &ids(&g), &dec_ids(&g), &shape, &w, &mixed, &bp, &ap, &z, &[GdnStream { state: &s_ref, hist: &h_ref }]);
        let b = gdn_mixer_decode_fused(&g, &shape, &w, &mixed, &bp, &ap, &z, &GdnStream { state: &s_nat, hist: &h_nat }).expect("offered");
        assert_eq!(bits(g.read(&b, value_dim as usize)), bits(g.read(&a, value_dim as usize)), "step {step}: output differs");
        assert_eq!(bits(g.read(&s_nat, state_len)), bits(g.read(&s_ref, state_len)), "step {step}: state differs");
        assert_eq!(bits(g.read(&h_nat, hist_len)), bits(g.read(&h_ref, hist_len)), "step {step}: window differs");
    }
}

/// A batch over rows of a pool, as one launch that updates the rows in place, is
/// the batch staged in, run through the chain and staged out - to the bit, for the
/// layer's outputs and for EVERY row of both pools (a row that is not in the batch
/// must come through untouched).
#[test]
fn a_pooled_batch_is_byte_identical_to_the_staged_chain() {
    let g = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let (dk, dv, kw) = (128u32, 128u32, 4u32);
    // (key heads, group, batch, pool rows, picked rows): one row, a ragged
    // order, the real 35B-A3B head counts (16 key heads x 2) at a decode batch.
    let cases: [(u32, u32, &[u32], u32); 4] = [(2, 3, &[1], 3), (2, 2, &[3, 0, 2], 5), (16, 2, &[5, 1, 7, 0, 3, 6], 8), (1, 1, &[2, 4, 0, 1, 3], 5)];
    for (case, &(nkh, group, picked, pool_rows)) in cases.iter().enumerate() {
        let (nvh, b) = (nkh * group, picked.len() as u32);
        let shape = GdnMixerShape { gdn: GdnShape { b, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
        let (value_dim, conv_dim) = (shape.value_dim(), shape.conv_dim());
        let state_len = (nvh * dk * dv) as usize;
        let hist_len = (conv_dim * (kw - 1)) as usize;
        let mut r = Lcg::new(0x900d_0000 + case as u64);
        let conv_w = g.storage_init("conv_w", &rnd(&mut r, (conv_dim * kw) as usize, 0.5));
        let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
        let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
        let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
        let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
        let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };
        let mixed = g.storage_init("mixed", &rnd(&mut r, (b * conv_dim) as usize, 1.0));
        let bp = g.storage_init("bp", &rnd(&mut r, (b * nvh) as usize, 1.0));
        let ap = g.storage_init("ap", &rnd(&mut r, (b * nvh) as usize, 1.0));
        let z = g.storage_init("z", &rnd(&mut r, (b * value_dim) as usize, 1.0));
        let state0 = rnd(&mut r, pool_rows as usize * state_len, 0.3);
        let hist0 = rnd(&mut r, pool_rows as usize * hist_len, 0.3);
        let rows = g.storage_init("rows", &picked.iter().map(|&i| f32::from_bits(i)).collect::<Vec<_>>());

        let run = |fuse: bool| {
            let (st, hi) = (g.storage_init("state", &state0), g.storage_init("hist", &hist0));
            let pool = GdnPoolRows { state: &st, hist: &hi, rows: &rows, gather: idx(&g, "pool_rows_gather2"), scatter: idx(&g, "pool_rows_scatter2"), fuse };
            let gated = gdn_mixer_decode_state_fwd(&g, &ids(&g), &dec_ids(&g), &shape, &w, &mixed, &bp, &ap, &z, &GdnDecodeState::Pool(pool));
            g.poll_wait();
            (bits(g.read(&gated, (b * value_dim) as usize)), bits(g.read(&st, pool_rows as usize * state_len)), bits(g.read(&hi, pool_rows as usize * hist_len)))
        };
        let (chain, fused) = (run(false), run(true));
        let ctx = format!("case {case}: nkh={nkh} group={group} batch={b}");
        assert_eq!(fused.0, chain.0, "layer output differs: {ctx}");
        assert_eq!(fused.1, chain.1, "state pool differs: {ctx}");
        assert_eq!(fused.2, chain.2, "window pool differs: {ctx}");
    }
}

/// The multi-row form: `rows` consecutive tokens of ONE sequence through one
/// launch, which must leave exactly what `rows` single steps leave - the same
/// outputs row by row, and the state and conv window after the last. This is the
/// kernel a speculative verify round runs its recurrent layers on, where the
/// rows depend on one another and have to be taken in order.
#[test]
fn a_multi_row_launch_is_byte_identical_to_the_same_rows_one_step_at_a_time() {
    let g = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let (dk, dv, kw) = (128u32, 128u32, 4u32);
    for (nkh, group, rows) in [(1u32, 1u32, 2u32), (2, 3, 1), (2, 3, 3), (4, 3, 5), (16, 3, 8)] {
        let nvh = nkh * group;
        let step_shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
        let rows_shape = GdnMixerShape { gdn: GdnShape { t: rows, ..step_shape.gdn }, ..step_shape };
        let (value_dim, conv_dim) = (step_shape.value_dim(), step_shape.conv_dim());
        let state_len = (nvh * dk * dv) as usize;
        let hist_len = (conv_dim * (kw - 1)) as usize;
        let ctx = format!("nkh={nkh} group={group} rows={rows}");

        let mut r = Lcg::new(0x5eed ^ u64::from(nkh * 31 + group * 7 + rows));
        let conv_w = g.storage_init("conv_w", &rnd(&mut r, (conv_dim * kw) as usize, 0.5));
        let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
        let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
        let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
        let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
        let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };

        let (rn, cd, vd) = (rows as usize, conv_dim as usize, value_dim as usize);
        let mixed = rnd(&mut r, rn * cd, 1.0);
        let bp = rnd(&mut r, rn * nvh as usize, 1.0);
        let ap = rnd(&mut r, rn * nvh as usize, 1.0);
        let z = rnd(&mut r, rn * vd, 1.0);
        let state0 = rnd(&mut r, state_len, 0.3);
        let hist0 = rnd(&mut r, hist_len, 0.3);

        // One step at a time through the chain, rows in order.
        let (s_ref, h_ref) = (g.storage_init("state", &state0), g.storage_init("hist", &hist0));
        let mut ref_out = Vec::new();
        for i in 0..rn {
            let row = |v: &[f32], width: usize, name: &str| g.storage_init(name, &v[i * width..(i + 1) * width]);
            let gated = gdn_mixer_decode_fwd(
                &g,
                &ids(&g),
                &dec_ids(&g),
                &step_shape,
                &w,
                &row(&mixed, cd, "mixed"),
                &row(&bp, nvh as usize, "bp"),
                &row(&ap, nvh as usize, "ap"),
                &row(&z, vd, "z"),
                &[GdnStream { state: &s_ref, hist: &h_ref }],
            );
            ref_out.extend(g.read(&gated, vd));
        }

        // All the rows in one launch.
        let (s_nat, h_nat) = (g.storage_init("state", &state0), g.storage_init("hist", &hist0));
        let gated = gdn_mixer_rows_fused(
            &g,
            &rows_shape,
            &w,
            &g.storage_init("mixed", &mixed),
            &g.storage_init("bp", &bp),
            &g.storage_init("ap", &ap),
            &g.storage_init("z", &z),
            &GdnStream { state: &s_nat, hist: &h_nat },
            rows,
        )
        .expect("the multi-row GDN kernel was declined on a CUDA device");
        g.poll_wait();

        assert_eq!(bits(g.read(&gated, rn * vd)), bits(ref_out), "layer output differs: {ctx}");
        assert_eq!(bits(g.read(&s_nat, state_len)), bits(g.read(&s_ref, state_len)), "recurrent state differs: {ctx}");
        assert_eq!(bits(g.read(&h_nat, hist_len)), bits(g.read(&h_ref, hist_len)), "conv window differs: {ctx}");
    }
}
