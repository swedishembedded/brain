// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::gdn_mixer::gdn_mixer_decode_fwd` at a batch of `B` sequences must
//! reproduce `B` independent single-sequence calls, in the layer's OUTPUT and
//! in both pieces of persistent state it leaves behind.
//!
//! This is the spec of the batched hybrid decode path. A Gated-DeltaNet layer
//! carries per-sequence state that a full-attention layer does not - a
//! recurrent `[h, dk, dv]` matrix and a `[conv_dim, K-1]` causal-conv window -
//! and the batched kernels address both through a single `b*h` axis that is
//! indistinguishable, to the kernel, from a wider HEAD count. So the failure
//! mode this gate exists for is a layout one: a batch row reading or writing
//! another row's slice of the recurrent state. That corrupts only the
//! sequences that share a dispatch, which is exactly the situation a
//! single-sequence test can never enter.
//!
//! The sequences are given DIFFERENT starting states (not a shared or zero
//! one), because a zero initial state makes almost every cross-row mix-up
//! invisible - the thing being confused is then the same value either way.
//!
//! Swedish Embedded AB implements batched recurrent-attention decode paths for
//! inference engines for its clients. If your team needs expertise in
//! linear-attention state management under continuous batching then you can
//! procure our services by sending an email to info@swedishembedded.com.

use audio::conv::ConvKernels;
use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{KernelIds, UNREGISTERED};
use model::gdn::{GdnBwdIds, GdnConvIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_decode_fwd, GdnMixerDecodeIds, GdnMixerIds, GdnMixerShape, GdnMixerWeights, GdnStream};

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

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn batched_decode_matches_per_sequence_decode() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let (nkh, nvh, dk, dv, kw) = (2u32, 4u32, 8u32, 6u32, 4u32);
    let batch = 3u32;
    let shape_of = |b: u32| GdnMixerShape { gdn: GdnShape { b, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw };
    let one = shape_of(1);
    let (value_dim, conv_dim) = (one.value_dim(), one.conv_dim());
    let state_len = (nvh * dk * dv) as usize;
    let hist_len = (conv_dim * (kw - 1)) as usize;

    let mut r = Lcg::new(0x5eed_1234);
    // Shared (non-projection) weights - identical for both runs.
    let conv_w = g.storage_init("conv_w", &rnd(&mut r, (conv_dim * kw) as usize, 0.5));
    let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
    let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
    let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
    let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
    let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };

    // Per-sequence projected inputs and per-sequence STARTING state.
    let mixed: Vec<Vec<f32>> = (0..batch).map(|_| rnd(&mut r, conv_dim as usize, 1.0)).collect();
    let bp: Vec<Vec<f32>> = (0..batch).map(|_| rnd(&mut r, nvh as usize, 1.0)).collect();
    let ap: Vec<Vec<f32>> = (0..batch).map(|_| rnd(&mut r, nvh as usize, 1.0)).collect();
    let zz: Vec<Vec<f32>> = (0..batch).map(|_| rnd(&mut r, value_dim as usize, 1.0)).collect();
    let st0: Vec<Vec<f32>> = (0..batch).map(|_| rnd(&mut r, state_len, 0.3)).collect();
    let hi0: Vec<Vec<f32>> = (0..batch).map(|_| rnd(&mut r, hist_len, 0.3)).collect();

    let flat = |v: &[Vec<f32>]| -> Vec<f32> { v.iter().flatten().copied().collect() };

    // ---- batched: one dispatch set over all `batch` sequences -------------
    let mixed_b = g.storage_init("mixed", &flat(&mixed));
    let bp_b = g.storage_init("bp", &flat(&bp));
    let ap_b = g.storage_init("ap", &flat(&ap));
    let z_b = g.storage_init("z", &flat(&zz));
    let states: Vec<DeviceBuffer> = st0.iter().map(|s| g.storage_init("state", s)).collect();
    let hists: Vec<DeviceBuffer> = hi0.iter().map(|h| g.storage_init("hist", h)).collect();
    let streams: Vec<GdnStream> = states.iter().zip(&hists).map(|(s, h)| GdnStream { state: s, hist: h }).collect();
    let gated = gdn_mixer_decode_fwd(&g, &ids(&g), &dec_ids(&g), &shape_of(batch), &w, &mixed_b, &bp_b, &ap_b, &z_b, &streams);
    let got = g.read(&gated, (batch * value_dim) as usize);
    let got_state: Vec<Vec<f32>> = states.iter().map(|s| g.read(s, state_len)).collect();
    let got_hist: Vec<Vec<f32>> = hists.iter().map(|h| g.read(h, hist_len)).collect();

    // ---- reference: the same call, one sequence at a time ------------------
    let (mut wo, mut ws, mut wh) = (0f32, 0f32, 0f32);
    for b in 0..batch as usize {
        let m1 = g.storage_init("mixed1", &mixed[b]);
        let b1 = g.storage_init("bp1", &bp[b]);
        let a1 = g.storage_init("ap1", &ap[b]);
        let z1 = g.storage_init("z1", &zz[b]);
        let s1 = g.storage_init("state1", &st0[b]);
        let h1 = g.storage_init("hist1", &hi0[b]);
        let one_stream = [GdnStream { state: &s1, hist: &h1 }];
        let out = gdn_mixer_decode_fwd(&g, &ids(&g), &dec_ids(&g), &one, &w, &m1, &b1, &a1, &z1, &one_stream);
        wo = wo.max(worst(&g.read(&out, value_dim as usize), &got[b * value_dim as usize..(b + 1) * value_dim as usize]));
        ws = ws.max(worst(&g.read(&s1, state_len), &got_state[b]));
        wh = wh.max(worst(&g.read(&h1, hist_len), &got_hist[b]));
    }
    println!("gdn_mixer_decode_fwd batch={batch} vs per-sequence: gated={wo:e} state={ws:e} hist={wh:e}");
    assert!(wo < 1e-6, "batched GDN decode output maxabs={wo}");
    assert!(ws < 1e-6, "batched GDN decode recurrent state maxabs={ws}");
    assert!(wh < 1e-6, "batched GDN decode conv history maxabs={wh}");
}
