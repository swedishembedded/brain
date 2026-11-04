// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements decode throughput for FP8 hybrid-attention models at the memory roof for its clients. If
// your team needs expertise in cutting a decode step's launches and weight traffic on a single GPU then you can procure our
// services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! A Gated DeltaNet layer of an FP8 build holds `in_proj_qkv` and `in_proj_z` as ONE weight. Spec:
//!
//! * the two projections are still two linears to everything that asks about them (their tier, their values, their
//!   precision-report rows), and the weight is held once: the report's bytes are the two projections' bytes, no more;
//! * a decode step multiplies the whole weight in one product and the streaming mixer reads the qkv and z columns of its
//!   rows in place; with one candidate kernel pinned for both forms, the logits are bit for bit those of the two window
//!   products a prefill round multiplies, and the one product is one launch fewer a layer;
//! * with the tuner free to pick the candidate for the wider product, the logits agree to the activation's storage
//!   resolution.

use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use gpu_core::Gpu;
use model::ops::{TierPolicy, Weight};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

use crate::common;
use common::fused_decode::{assert_logits_agree, decode, serial};
use common::hf_fixture::fp8_resident_cfg;

/// The recurrent geometry the streaming kernels serve (128-wide key and value heads, a 4-tap conv) at FP8-sized widths:
/// `in_proj_qkv` 512 rows, `in_proj_z` 256, both whole 128-row scale blocks. The model width is one the pinned streaming
/// candidate serves (a multiple of 256), so every product of the step runs it and the comparison is exact.
fn cfg() -> Qwen35Config {
    Qwen35Config { d_model: 512, linear_num_key_heads: 1, linear_num_value_heads: 2, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, ..fp8_resident_cfg() }
}

#[test]
fn the_two_projections_are_one_fp8_weight_that_answers_for_each() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let cfg = cfg();
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, cfg.block_size, &init, &TierPolicy::uniform(Dtype::F8E4M3));
    let (qkv, z) = ("blocks.0.linear_attn.in_proj_qkv.weight", "blocks.0.linear_attn.in_proj_z.weight");
    assert!(m.holds_fused_qkvz(), "an FP8 build on a device with the native FP8 products and the streaming GDN kernels holds the two projections as one weight");
    assert_eq!(m.linear_dtype(qkv), Some(Dtype::F8E4M3));
    assert_eq!(m.linear_dtype(z), Some(Dtype::F8E4M3));

    // Each projection reads back the values its own rows store: block-wise scales make the fused weight's rows the same
    // bytes as the projection quantized alone.
    let ops = model::ops::Ops::new(Gpu::new(model::ops::training_kernel_list())).expect("Ops::new");
    for (name, rows) in [(qkv, cfg.linear_conv_dim()), (z, cfg.linear_value_dim())] {
        let alone = Weight::upload(&ops, &init[name], rows as usize, cfg.d_model as usize, Dtype::F8E4M3);
        assert_eq!(m.read_weight(name), ops.read_host(&alone), "{name}: the rows of the fused weight are not the projection's own");
    }

    // Held once: the report charges the two projections their own bytes.
    let report = m.precision_report();
    let held: u64 = [qkv, z].iter().map(|n| report.rows().iter().find(|r| r.name == *n).unwrap_or_else(|| panic!("{n} is missing from the precision report")).bytes).sum();
    let (n_qkv, n_z, k) = (cfg.linear_conv_dim() as usize, cfg.linear_value_dim() as usize, cfg.d_model as usize);
    assert_eq!(held, Weight::bytes_for(Dtype::F8E4M3, n_qkv, k) + Weight::bytes_for(Dtype::F8E4M3, n_z, k));
}

#[test]
fn a_decode_step_is_one_product_over_the_fused_weight_and_computes_what_the_two_windows_do() {
    let _s = serial();
    if !gpu_core::native_kernels_enabled() {
        return brain_testutil::skip_unavailable("native kernels are withheld in this process");
    }
    let Ok(_gpu) = Gpu::try_new_cuda(pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    // One streaming candidate for every width: a dot product is summed in the same order whatever the width of the weight.
    gpu_core::force_candidate("matmul_fp8_native", "matmul_fp8_dq_stream_r4");
    let (one, one_launches) = decode(&cfg(), true, "qkvz_one", 2, KvTier::F32);
    let (two, two_launches) = decode(&cfg(), false, "qkvz_two", 2, KvTier::F32);
    gpu_core::clear_forced_candidates();
    let gdn_layers = cfg().layer_types().iter().filter(|t| matches!(t, qwen35::config::LayerType::Linear)).count() as u64;
    assert!(one_launches + gdn_layers <= two_launches, "the first step issued {one_launches} launches with the one product and {two_launches} with the two: the one product did not engage");
    assert_logits_agree(&one, &two, None);
}

#[test]
fn the_tuned_choice_for_the_wider_product_keeps_the_logits_within_the_activation_resolution() {
    let _s = serial();
    if !gpu_core::native_kernels_enabled() {
        return brain_testutil::skip_unavailable("native kernels are withheld in this process");
    }
    let Ok(_gpu) = Gpu::try_new_cuda(pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let (one, _) = decode(&cfg(), true, "qkvz_tuned_one", 2, KvTier::F32);
    let (two, _) = decode(&cfg(), false, "qkvz_tuned_two", 2, KvTier::F32);
    assert_logits_agree(&one, &two, Some(0.05));
}
