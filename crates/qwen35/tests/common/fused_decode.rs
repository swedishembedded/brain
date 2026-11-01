// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements decode throughput for FP8 hybrid-attention models at the memory roof for its clients. If
// your team needs expertise in cutting a decode step's launches and weight traffic on a single GPU then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The harness of the tests that hold several projections of a layer as one FP8 weight: a tiny FP8 checkpoint served by the
//! resident, a prefill and three batched decode steps with the decode's one product on or off, and the comparisons the specs
//! make between the two runs.

#![allow(dead_code)]

use backend_cuda::call_totals;
use gpu_core::select::Dtype;
use model::kv_tier::KvTier;
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufInstance, Qwen35GgufResident};
use std::sync::{Mutex, MutexGuard};

use super::hf_fixture::{write_checkpoint_conditioned, write_tokenizer};

/// The pinned candidate and the call totals are process-global.
static SERIAL: Mutex<()> = Mutex::new(());

pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let den: f64 = want.iter().map(|b| f64::from(*b).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

pub fn instance(cfg: &Qwen35Config, tag: &str, max_batch: u32, kv: KvTier) -> (Qwen35GgufInstance, std::path::PathBuf) {
    let dir = write_checkpoint_conditioned(cfg, tag, 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    assert_eq!(tier.default_dtype(), Dtype::F8E4M3, "the linears are held as FP8");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the test needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, 128, tier).with_max_batch(max_batch).with_kv_block_rows(16).with_kv_tier(kv).with_gdn_state(model::policy::GdnState::Bf16);
    (resident.activate_owned(&[device]).expect("the tiny FP8 directory loads"), dir)
}

/// Prefill one prompt, then three batched decode steps (`rows` sequences), with the decode's one product on or off; the logits
/// of every step and the kernel launches the first step issued.
pub fn decode(cfg: &Qwen35Config, fused: bool, tag: &str, rows: u32, kv: KvTier) -> (Vec<Vec<Vec<f32>>>, u64) {
    let (inst, dir) = instance(cfg, tag, rows.max(4), kv);
    inst.set_fused_projections_decode(fused);
    inst.prefill_timed(&[3, 9, 4, 1, 7, 7, 2, 5, 8, 6, 1, 3]).expect("prefill");
    let before = call_totals().host_launches;
    let mut logits = Vec::new();
    let mut launches = 0;
    for step in 0..3u32 {
        logits.push(inst.decode_batch_at(&(0..rows).map(|i| (11 + step + 10 * i) % 60).collect::<Vec<_>>(), &(0..rows).map(|i| 12 + step - 2 * (i % 4) + i).collect::<Vec<_>>()).expect("batched decode"));
        if step == 0 {
            launches = call_totals().host_launches - before;
        }
    }
    drop(inst);
    std::fs::remove_dir_all(dir).ok();
    (logits, launches)
}

/// Every logit of every step is finite and, with `exact`, bit for bit equal across the two runs; otherwise the rows agree to
/// `bound` (relative L2).
pub fn assert_logits_agree(one: &[Vec<Vec<f32>>], two: &[Vec<Vec<f32>>], bound: Option<f64>) {
    for (step, (a, b)) in one.iter().zip(two).enumerate() {
        for (row, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(x.iter().all(|v| v.is_finite()), "step {step} row {row}: non-finite logits");
            match bound {
                None => {
                    let differing = x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    assert!(differing == 0, "step {step} row {row}: {differing} of {} logits differ between the one product and the window products (relative L2 {})", x.len(), rel_l2(x, y));
                }
                Some(bound) => {
                    let err = rel_l2(x, y);
                    assert!(err < bound, "step {step} row {row}: the logits are {err} (relative L2) from the window-product logits");
                }
            }
        }
    }
}
