// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splitting a sequence into ROUNDS and threading `model::gdn_mixer::
//! GdnStream` through them must reproduce the one-shot whole-sequence
//! `gdn_mixer_fwd` over the same rows. This is the spec of the chunked-prefill
//! seam: a Gated-DeltaNet layer carries TWO pieces of state across a round
//! boundary - `gdn_chunk_fwd`'s recurrent `final_state` and the causal conv's
//! `K-1`-row input tail - and getting either wrong produces an output that is
//! wrong only in the rows near the boundary, which a whole-model end-to-end
//! comparison dilutes into fp32 noise (measured: a completely DROPPED
//! recurrent state moved `qwen35`'s final hidden state by 5e-7 on random
//! weights). Here the mixer's own output is compared directly, where the same
//! error is four orders of magnitude larger and impossible to miss.
//!
//! The rounds deliberately use different lengths AND different internal chunk
//! sizes from each other and from the whole-sequence reference, since a real
//! prefill's last round is ragged and `gdn::gdn_chunk_size` picks a different
//! chunk for it.

use audio::conv::ConvKernels;
use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{KernelIds, UNREGISTERED};
use model::gdn::{GdnBwdIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_fwd, gdn_mixer_stream_fwd, GdnMixerIds, GdnMixerShape, GdnMixerWeights, GdnStream};

/// Forward-only kernel set for the GDN mixer (no backward tier - this gate
/// never builds a training forward).
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
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn ids(g: &Gpu) -> GdnMixerIds {
    // Every slot the forward never dispatches is UNREGISTERED, per
    // `model::block::UNREGISTERED`'s own contract (a 0 there would silently
    // run whatever kernel sits at index 0).
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

/// nvh=2, khd=3, vhd=4, nkh=1 (group 2), conv kernel 3 - the same tiny,
/// pairwise-distinct shape `gdn_mixer_equivalence.rs` uses, with `t`/`chunk`
/// supplied per call.
fn shape(t: u32, chunk: u32) -> GdnMixerShape {
    GdnMixerShape { gdn: GdnShape { b: 1, h: 2, t, dk: 3, dv: 4, chunk }, nkh: 1, conv_kernel: 3 }
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn threading_the_stream_state_across_rounds_matches_the_whole_sequence_forward() {
    let g = Gpu::new_cpu(KERNELS);
    let whole = shape(8, 2);
    let (conv_dim, value_dim, nvh, khd, vhd) = (whole.conv_dim(), whole.value_dim(), whole.gdn.h, whole.gdn.dk, whole.gdn.dv);
    let t = whole.gdn.t;

    let mut rng = Lcg::new(20260902);
    let mixed_qkv_h = rng.vec_scaled((t * conv_dim) as usize, 1.0);
    let bproj_h = rng.vec_scaled((t * nvh) as usize, 1.0);
    let aproj_h = rng.vec_scaled((t * nvh) as usize, 1.0);
    let z_h = rng.vec_scaled((t * value_dim) as usize, 1.0);
    let conv1d_weight = g.storage_init("conv1d_weight", &rng.vec_scaled(conv_dim as usize * whole.conv_kernel as usize, 1.0));
    let a_log = g.storage_init("a_log", &rng.vec_scaled(nvh as usize, 1.0));
    let dt_bias = g.storage_init("dt_bias", &rng.vec_scaled(nvh as usize, 1.0));
    let norm_weight = g.storage_init("norm_weight", &rng.vec_scaled(vhd as usize, 1.0));
    let ones_khd = g.storage_init("ones_khd", &vec![1.0f32; khd as usize]);
    let w = GdnMixerWeights { conv1d_weight: &conv1d_weight, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_weight, ones_khd: &ones_khd };
    let ids = ids(&g);

    // Reference: the whole 8 rows in one call, fresh state.
    let want = {
        let mixed_qkv = g.storage_init("mixed_qkv", &mixed_qkv_h);
        let bproj = g.storage_init("bproj", &bproj_h);
        let aproj = g.storage_init("aproj", &aproj_h);
        let z = g.storage_init("z", &z_h);
        let (gated, _) = gdn_mixer_fwd(&g, &ids, &whole, &w, &mixed_qkv, &bproj, &aproj, &z, t, false);
        g.read(&gated, (t * value_dim) as usize)
    };

    // Under test: the same rows in two rounds of different lengths and
    // different internal chunk sizes, threading one GdnStream. Both buffers
    // start zeroed, which is what a fresh sequence means.
    let state = g.storage(whole.gdn.bh() as u64 * khd as u64 * vhd as u64);
    let hist = g.storage(conv_dim as u64 * (whole.conv_kernel - 1) as u64);
    g.submit(&[&state, &hist], &[]);

    let mut got: Vec<f32> = Vec::new();
    let mut row = 0u32;
    for (rows, chunk) in [(3u32, 1u32), (5u32, 5u32)] {
        let sl = |v: &[f32], width: u32| v[(row * width) as usize..((row + rows) * width) as usize].to_vec();
        let mixed_qkv = g.storage_init("mixed_qkv", &sl(&mixed_qkv_h, conv_dim));
        let bproj = g.storage_init("bproj", &sl(&bproj_h, nvh));
        let aproj = g.storage_init("aproj", &sl(&aproj_h, nvh));
        let z = g.storage_init("z", &sl(&z_h, value_dim));
        let cont = GdnStream { state: &state, hist: &hist };
        let (gated, _) = gdn_mixer_stream_fwd(&g, &ids, &shape(rows, chunk), &w, &mixed_qkv, &bproj, &aproj, &z, rows, false, Some(cont));
        got.extend(g.read(&gated, (rows * value_dim) as usize));
        row += rows;
    }

    let err = maxabs(&got, &want);
    println!("gdn_mixer_stream: 3+5 rounds vs one 8-row forward, maxabs = {err:e}");
    assert!(err < 1e-5, "streamed rounds diverged from the whole-sequence forward: maxabs={err}");
}
