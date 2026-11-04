// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements decode throughput for FP8 hybrid-attention models at the memory roof for its clients. If
// your team needs expertise in cutting a decode step's launches and weight traffic on a single GPU then you can procure our
// services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! A layer of an FP8 build holds the SwiGLU's `gate` and `up` as ONE weight (gate rows first). Spec:
//!
//! * the two projections are still two linears to everything that asks about them (their tier, their values, their
//!   precision-report rows), and the weight is held once: the report's bytes are the two projections' bytes, no more;
//! * a decode step of at most 128 rows multiplies the whole weight in one product and the SwiGLU pass reads the gate and up
//!   columns of its rows in place; with one candidate kernel pinned for both forms, the logits are bit for bit those of the
//!   two window products a prefill round multiplies, and the one product is one launch fewer a layer, at the 2-row
//!   (epilogue) and the 20-row (E4M3 SwiGLU pass) widths;
//! * with the tuner free to pick the candidate for the wider product, the logits agree to the activation's storage
//!   resolution.

use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use gpu_core::Gpu;
use model::ops::{TierPolicy, Weight};
use qwen35::model::{pipelines, Qwen35};

use crate::common;
use common::fused_decode::{assert_logits_agree, decode, serial};
use common::hf_fixture::fp8_resident_cfg;
use qwen35::config::Qwen35Config;

/// A width the streaming candidates serve (their K is a multiple of 256) with whole 128-row scale blocks of gate and up.
fn cfg() -> Qwen35Config {
    Qwen35Config { d_model: 512, intermediate_size: 768, ..fp8_resident_cfg() }
}

#[test]
fn gate_and_up_are_one_fp8_weight_that_answers_for_each() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let cfg = cfg();
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, cfg.block_size, &init, &TierPolicy::uniform(Dtype::F8E4M3));
    let (gate, up) = ("blocks.0.mlp.gate.weight", "blocks.0.mlp.up.weight");
    assert!(m.holds_fused_gate_up(), "an FP8 build on a device with the native FP8 products holds gate and up as one weight");
    assert_eq!(m.linear_dtype(gate), Some(Dtype::F8E4M3));
    assert_eq!(m.linear_dtype(up), Some(Dtype::F8E4M3));

    // Each projection reads back the values its own rows store: block-wise scales make the fused weight's rows the same
    // bytes as the projection quantized alone.
    let ops = model::ops::Ops::new(Gpu::new(model::ops::training_kernel_list())).expect("Ops::new");
    for name in [gate, up] {
        let alone = Weight::upload(&ops, &init[name], cfg.intermediate_size as usize, cfg.d_model as usize, Dtype::F8E4M3);
        assert_eq!(m.read_weight(name), ops.read_host(&alone), "{name}: the rows of the fused weight are not the projection's own");
    }

    // Held once: the report charges the two projections their own bytes.
    let report = m.precision_report();
    let held: u64 = [gate, up].iter().map(|n| report.rows().iter().find(|r| r.name == *n).unwrap_or_else(|| panic!("{n} is missing from the precision report")).bytes).sum();
    assert_eq!(held, 2 * Weight::bytes_for(Dtype::F8E4M3, cfg.intermediate_size as usize, cfg.d_model as usize));
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
    for rows in [2u32, 20] {
        let (one, one_launches) = decode(&cfg(), true, &format!("gate_up_one_{rows}"), rows, KvTier::F32);
        let (two, two_launches) = decode(&cfg(), false, &format!("gate_up_two_{rows}"), rows, KvTier::F32);
        let layers = u64::from(cfg().n_layers);
        assert!(one_launches + layers <= two_launches, "{rows} rows: the first step issued {one_launches} launches with the one product and {two_launches} with the two: the one product did not engage");
        assert_logits_agree(&one, &two, None);
    }
    gpu_core::clear_forced_candidates();
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
    for rows in [2u32, 20, 64] {
        let (one, _) = decode(&cfg(), true, &format!("gate_up_tuned_one_{rows}"), rows, KvTier::F32);
        let (two, _) = decode(&cfg(), false, &format!("gate_up_tuned_two_{rows}"), rows, KvTier::F32);
        assert_logits_agree(&one, &two, Some(0.05));
    }
}
