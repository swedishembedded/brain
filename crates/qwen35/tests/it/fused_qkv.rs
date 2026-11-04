// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements decode throughput for FP8 hybrid-attention models at the memory roof for its clients. If
// your team needs expertise in cutting a decode step's launches and weight traffic on a single GPU then you can procure our
// services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! An attention layer of an FP8 build holds `q_proj`, `k_proj` and `v_proj` as ONE weight (rows in that order). Spec:
//!
//! * the three projections are still three linears to everything that asks about them (their tier, their values, their
//!   precision-report rows), and the weight is held once: the report's bytes are the three projections' bytes, no more;
//! * a decode step of at most 128 rows multiplies the whole weight in one product and one launch takes its rows apart for the
//!   attention front half: with one candidate kernel pinned for both forms, the logits are bit for bit those of the three
//!   window products a prefill round multiplies, over the f32 KV tier (the native front half reads the rows interleaved as the
//!   projection writes them) and the bf16 tier (the chain reads the query's values and gates apart), and the step issues
//!   launches fewer for it;
//! * with the tuner free to pick the candidate for the wider product, the logits agree to the activation's storage
//!   resolution.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::kv_tier::KvTier;
use model::ops::{TierPolicy, Weight};
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

use crate::common;
use common::fused_decode::{assert_logits_agree, decode, serial};
use common::hf_fixture::fp8_resident_cfg;

/// Attention heads of 64 with whole 128-row scale blocks in all three projections (`q_proj` 512 rows with its gate, `k_proj` and
/// `v_proj` 128), and a model width the pinned streaming candidate serves.
fn cfg() -> Qwen35Config {
    Qwen35Config { d_model: 512, n_heads: 4, n_kv_heads: 2, head_dim: 64, ..fp8_resident_cfg() }
}

fn attention_layers() -> u64 {
    cfg().layer_types().iter().filter(|t| matches!(t, LayerType::Full)).count() as u64
}

#[test]
fn q_k_and_v_are_one_fp8_weight_that_answers_for_each() {
    let _s = serial();
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    let cfg = cfg();
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, cfg.block_size, &init, &TierPolicy::uniform(Dtype::F8E4M3));
    assert!(m.holds_fused_qkv(), "an FP8 build on a device with the native FP8 products holds q, k and v as one weight");
    let layer = cfg.layer_types().iter().position(|t| matches!(t, LayerType::Full)).expect("an attention layer");
    let parts = [("q_proj", cfg.q_proj_dim()), ("k_proj", cfg.kv_dim()), ("v_proj", cfg.kv_dim())].map(|(leaf, rows)| (format!("blocks.{layer}.self_attn.{leaf}.weight"), rows));

    // Each projection reads back the values its own rows store: block-wise scales make the fused weight's rows the same
    // bytes as the projection quantized alone.
    let ops = model::ops::Ops::new(Gpu::new(model::ops::training_kernel_list())).expect("Ops::new");
    for (name, rows) in &parts {
        assert_eq!(m.linear_dtype(name), Some(Dtype::F8E4M3));
        let alone = Weight::upload(&ops, &init[name], *rows as usize, cfg.d_model as usize, Dtype::F8E4M3);
        assert_eq!(m.read_weight(name), ops.read_host(&alone), "{name}: the rows of the fused weight are not the projection's own");
    }

    // Held once: the report charges the three projections their own bytes.
    let report = m.precision_report();
    let held: u64 = parts.iter().map(|(n, _)| report.rows().iter().find(|r| r.name == *n).unwrap_or_else(|| panic!("{n} is missing from the precision report")).bytes).sum();
    let want: u64 = parts.iter().map(|(_, rows)| Weight::bytes_for(Dtype::F8E4M3, *rows as usize, cfg.d_model as usize)).sum();
    assert_eq!(held, want);
}

#[test]
fn a_decode_step_is_one_product_over_the_fused_weight_and_computes_what_the_three_windows_do() {
    let _s = serial();
    if !gpu_core::native_kernels_enabled() {
        return brain_testutil::skip_unavailable("native kernels are withheld in this process");
    }
    let Ok(_gpu) = Gpu::try_new_cuda(pipelines()) else {
        return brain_testutil::skip_unavailable("no usable CUDA backend");
    };
    // One streaming candidate for every width: a dot product is summed in the same order whatever the width of the weight.
    gpu_core::force_candidate("matmul_fp8_native", "matmul_fp8_dq_stream_r4");
    // Launches saved a attention layer: f32 KV is the native front half (two products out, one split in), bf16 the chain (two
    // products and two `concat_split` out, one split in).
    for (kv, tag, saved) in [(KvTier::F32, "f32", 1u64), (KvTier::BF16, "bf16", 3)] {
        for rows in [2u32, 20] {
            let (one, one_launches) = decode(&cfg(), true, &format!("qkv_one_{tag}_{rows}"), rows, kv);
            let (two, two_launches) = decode(&cfg(), false, &format!("qkv_two_{tag}_{rows}"), rows, kv);
            assert!(one_launches + saved * attention_layers() <= two_launches, "{tag} KV, {rows} rows: the first step issued {one_launches} launches with the one product and {two_launches} with the three: the one product did not engage");
            assert_logits_agree(&one, &two, None);
        }
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
    for kv in [KvTier::F32, KvTier::BF16] {
        let (one, _) = decode(&cfg(), true, "qkv_tuned_one", 2, kv);
        let (two, _) = decode(&cfg(), false, "qkv_tuned_two", 2, kv);
        assert_logits_agree(&one, &two, Some(0.05));
    }
}
