// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared fixture for the Gated DeltaNet mixer's training-path tests: the
//! complete forward+backward kernel registration, the [`GdnMixerIds`]
//! resolved against it, and a seeded random weight/input set for a given
//! [`GdnMixerShape`].
//!
//! Swedish Embedded AB implements solutions for training long-context
//! linear-attention models on a single accelerator for its clients. If your
//! team needs expertise in memory-bounded sequence-model training you can
//! procure our services by sending an email to info@swedishembedded.com.

#![allow(dead_code)]

use audio::conv::ConvKernels;
use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::KernelIds;
use model::gdn::{GdnBwdIds, GdnIds};
use model::gdn_mixer::{GdnMixerIds, GdnMixerShape};

/// Every kernel the GDN mixer's forward and backward dispatch.
pub const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
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
];

pub fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

pub fn mixer_ids(g: &Gpu) -> GdnMixerIds {
    let unused = model::block::UNREGISTERED;
    GdnMixerIds {
        kernels: KernelIds {
            rmsnorm: idx(g, "rmsnorm"),
            rms_inv: idx(g, "rms_inv"),
            rmsnorm_dx: idx(g, "rmsnorm_dx"),
            rmsnorm_dx_rows: unused,
            rmsnorm_dw: idx(g, "rmsnorm_dw"),
            rope: unused,
            rope_bwd: unused,
            gqa_scores: unused,
            gqa_apply: unused,
            attn_softmax: unused,
            gqa_dscores: unused,
            gqa_dv: unused,
            gqa_dq: unused,
            gqa_dk: unused,
            silu_mul: unused,
            silu_da: unused,
            silu_db: unused,
            rmsnorm_rows: unused,
        },
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
            fast: None,
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

/// Host copies of one mixer call's inputs and weights, so the same values can
/// be uploaded as many times as a comparison needs.
#[derive(Clone)]
pub struct Host {
    pub mixed_qkv: Vec<f32>,
    pub bproj: Vec<f32>,
    pub aproj: Vec<f32>,
    pub z: Vec<f32>,
    pub d_gated: Vec<f32>,
    pub conv1d_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
}

impl Host {
    /// `seed` is the caller's: tests pass a literal with a `// seed:` reason.
    pub fn random(shape: &GdnMixerShape, seed: u64) -> Host {
        let n = (shape.gdn.b * shape.gdn.t) as usize;
        let (conv_dim, value_dim, nvh) = (shape.conv_dim() as usize, shape.value_dim() as usize, shape.gdn.h as usize);
        let mut rng = Lcg::new(seed);
        Host {
            mixed_qkv: rng.vec_scaled(n * conv_dim, 1.0),
            bproj: rng.vec_scaled(n * nvh, 1.0),
            aproj: rng.vec_scaled(n * nvh, 1.0),
            z: rng.vec_scaled(n * value_dim, 1.0),
            d_gated: rng.vec_scaled(n * value_dim, 1.0),
            conv1d_weight: rng.vec_scaled(conv_dim * shape.conv_kernel as usize, 1.0),
            a_log: rng.vec_scaled(nvh, 1.0),
            dt_bias: rng.vec_scaled(nvh, 1.0),
            norm_weight: rng.vec_scaled(shape.gdn.dv as usize, 1.0),
        }
    }
}

/// Device copies of the four non-projection weights plus zeroed gradient
/// accumulators for them.
pub struct DeviceWeights {
    pub conv1d_weight: DeviceBuffer,
    pub a_log: DeviceBuffer,
    pub dt_bias: DeviceBuffer,
    pub norm_weight: DeviceBuffer,
    pub ones_khd: DeviceBuffer,
    pub conv1d_weight_g: DeviceBuffer,
    pub a_log_g: DeviceBuffer,
    pub dt_bias_g: DeviceBuffer,
    pub norm_weight_g: DeviceBuffer,
}

impl DeviceWeights {
    pub fn upload(g: &Gpu, shape: &GdnMixerShape, h: &Host) -> DeviceWeights {
        DeviceWeights {
            conv1d_weight: g.storage_init("conv1d_weight", &h.conv1d_weight),
            a_log: g.storage_init("a_log", &h.a_log),
            dt_bias: g.storage_init("dt_bias", &h.dt_bias),
            norm_weight: g.storage_init("norm_weight", &h.norm_weight),
            ones_khd: g.storage_init("ones_khd", &vec![1.0f32; shape.gdn.dk as usize]),
            conv1d_weight_g: g.storage(h.conv1d_weight.len() as u64),
            a_log_g: g.storage(h.a_log.len() as u64),
            dt_bias_g: g.storage(h.dt_bias.len() as u64),
            norm_weight_g: g.storage(h.norm_weight.len() as u64),
        }
    }

    pub fn weights(&self) -> model::gdn_mixer::GdnMixerWeights<'_> {
        model::gdn_mixer::GdnMixerWeights {
            conv1d_weight: &self.conv1d_weight,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm_weight: &self.norm_weight,
            ones_khd: &self.ones_khd,
        }
    }

    pub fn grads(&self) -> model::gdn_mixer::GdnMixerGrads<'_> {
        model::gdn_mixer::GdnMixerGrads {
            conv1d_weight: Some(&self.conv1d_weight_g),
            a_log: Some(&self.a_log_g),
            dt_bias: Some(&self.dt_bias_g),
            norm_weight: Some(&self.norm_weight_g),
        }
    }

    /// The four weight gradients read back, in a fixed order.
    pub fn read_grads(&self, g: &Gpu, h: &Host) -> [Vec<f32>; 4] {
        [
            g.read(&self.conv1d_weight_g, h.conv1d_weight.len()),
            g.read(&self.a_log_g, h.a_log.len()),
            g.read(&self.dt_bias_g, h.dt_bias.len()),
            g.read(&self.norm_weight_g, h.norm_weight.len()),
        ]
    }
}

pub fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}
