// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What Ripple honours of the model-agnostic serving policy.
//!
//! Swedish Embedded AB implements solutions for serving many model families behind one precision and capacity
//! contract for its clients. If your team needs expertise in quantized serving configuration across
//! architectures then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The declaration is data; `model::policy::PolicySupport::check` is the one place a choice outside it is
//! refused, by model name and supported set. Leave `--weights` unset to hold every linear as the checkpoint
//! stores it (bf16 stays bf16, the official NVFP4 release keeps its MLP in NVFP4); NVFP4 cannot be asked for by
//! name because nothing quantizes into it.

use gpu_core::select::Dtype;
use model::policy::PolicySupport;

/// Ripple's serving options.
pub const POLICY: PolicySupport = PolicySupport {
    model: "ricursive",
    // The weight formats a bf16 or f32 checkpoint can be held in. int8 executes where the device has a packed-int8
    // dot (CUDA, and wgpu with DP4A) and is refused by name elsewhere; the CPU JIT holds the others.
    weights: &[Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::F8E4M3],
    weight_rules: true,
    kv_cache: &["f32", "bf16", "int8", "fp8"],
    gemm: false,
    fuse_norm: false,
    kv_calibration: false,
    exact_decode: false,
    act_precision: false,
    gdn_state: false,
    linear_activations: false,
    sparse_attention: true,
    draft_model: true,
};
