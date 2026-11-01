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

use backend_cuda::call_totals;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::kv_tier::KvTier;
use model::ops::{TierPolicy, Weight};
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufInstance, Qwen35GgufResident};
use qwen35::model::{pipelines, Qwen35};
use std::sync::{Mutex, MutexGuard};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

/// The pinned candidate and the call totals are process-global.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The recurrent geometry the streaming kernels serve (128-wide key and value heads, a 4-tap conv) at FP8-sized widths:
/// `in_proj_qkv` 512 rows, `in_proj_z` 256, both whole 128-row scale blocks.
fn cfg() -> Qwen35Config {
    Qwen35Config { linear_num_key_heads: 1, linear_num_value_heads: 2, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, ..fp8_resident_cfg() }
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let den: f64 = want.iter().map(|b| f64::from(*b).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
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

fn instance(tag: &str) -> (Qwen35GgufInstance, std::path::PathBuf) {
    let cfg = cfg();
    let dir = write_checkpoint_conditioned(&cfg, tag, 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    assert_eq!(tier.default_dtype(), Dtype::F8E4M3, "the linears are held as FP8");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the test needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, 128, tier).with_max_batch(4).with_kv_block_rows(16).with_kv_tier(KvTier::F32).with_gdn_state(model::policy::GdnState::Bf16);
    (resident.activate_owned(&[device]).expect("the tiny FP8 directory loads"), dir)
}

/// Prefill one prompt, then three batched decode steps (two sequences), with the decode's one product on or off; the logits
/// of every step and the kernel launches the first step issued.
fn decode(fused: bool, tag: &str) -> (Vec<Vec<Vec<f32>>>, u64) {
    let (inst, dir) = instance(tag);
    inst.set_fused_qkvz_decode(fused);
    inst.prefill_timed(&[3, 9, 4, 1, 7, 7, 2, 5, 8, 6, 1, 3]).expect("prefill");
    let before = call_totals().host_launches;
    let mut logits = Vec::new();
    let mut launches = 0;
    for step in 0..3u32 {
        logits.push(inst.decode_batch_at(&[11 + step, 21 + step], &[12 + step, 4 + step]).expect("batched decode"));
        if step == 0 {
            launches = call_totals().host_launches - before;
        }
    }
    drop(inst);
    std::fs::remove_dir_all(dir).ok();
    (logits, launches)
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
    let (one, one_launches) = decode(true, "qkvz_one");
    let (two, two_launches) = decode(false, "qkvz_two");
    gpu_core::clear_forced_candidates();
    let gdn_layers = cfg().layer_types().iter().filter(|t| matches!(t, qwen35::config::LayerType::Linear)).count() as u64;
    assert!(one_launches + gdn_layers <= two_launches, "the first step issued {one_launches} launches with the one product and {two_launches} with the two: the one product did not engage");
    for (step, (a, b)) in one.iter().zip(&two).enumerate() {
        for (row, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(x.iter().all(|v| v.is_finite()), "step {step} row {row}: non-finite logits");
            assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "step {step} row {row}: the one product differs from the two window products");
        }
    }
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
    let (one, _) = decode(true, "qkvz_tuned_one");
    let (two, _) = decode(false, "qkvz_tuned_two");
    for (step, (a, b)) in one.iter().zip(&two).enumerate() {
        for (row, (x, y)) in a.iter().zip(b).enumerate() {
            let err = rel_l2(x, y);
            assert!(err < 0.05, "step {step} row {row}: the logits are {err} (relative L2) from the two-product logits");
        }
    }
}
