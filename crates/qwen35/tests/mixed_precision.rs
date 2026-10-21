// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mixed-precision loading and fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in running one model whose tensors are held in different numeric
// formats then you can procure our services by sending an email to
// info@swedishembedded.com.

//! End-to-end acceptance of a MIXED-precision Qwen3.5 checkpoint: attention FP8, MLP NF4, embeddings bf16,
//! norms f32, all in one packed file. Spec:
//!   * the precision report lists every tensor of the file with the format it is stored in;
//!   * the model loads each tensor in that format (no tensor is widened to fp32 on the way to the device) and
//!     its own report of what is resident agrees with the file's, row by row;
//!   * its forward and LoRA gradients equal those of an fp32 build over the decoded values (the packed kernels
//!     only reorder fp32 sums, so the bound is tight);
//!   * LoRA over it trains, and the adapter saves and reloads: a model rebuilt from the saved adapter over the
//!     same packed base reproduces the trained logits, and folding the adapter into the decoded base agrees.

mod common;

use std::collections::HashMap;

use common::hf_fixture::{write_dense_checkpoint, BLOCK};
use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{encoding_for, stored_tier_policy, TierPolicy, Weight, FP8_BLOCK};
use qwen35::base_source::{lora_overlay_tuned, AdapterOverlay, BaseCheckpoint};
use qwen35::config::Qwen35Config;
use qwen35::fp8_source::{open_dir, write_converted_base};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const RANK: u32 = 3;
const ALPHA: f32 = 6.0;

fn cfg() -> Qwen35Config {
    // `tiny_i8` keeps every contraction dimension a multiple of the 32-wide quantization groups.
    Qwen35Config { block_size: T as u32, ..Qwen35Config::tiny_i8() }
}

/// Attention FP8, MLP NF4, embeddings bf16, norms f32; every other tensor as stored.
fn mixed_policy() -> TierPolicy {
    TierPolicy::uniform(Dtype::F8E4M3).with(&["mlp."], Dtype::NF4).with(&["tok.weight", "lm_head.weight"], Dtype::BF16).with(&["norm.weight", "ln1.weight", "ln2.weight"], Dtype::F32)
}

/// The tier a tensor of the mixed file is expected to be stored in, `None` for one the policy leaves as stored.
fn expected_tier(cfg: &Qwen35Config, name: &str) -> Option<Dtype> {
    let leaves: Vec<String> = cfg.layer_types().iter().enumerate().flat_map(|(l, ty)| cfg.layer_leaves(*ty).into_iter().map(move |(s, _, _)| format!("blocks.{l}.{s}"))).collect();
    if leaves.iter().any(|n| n == name) {
        Some(if name.contains("mlp.") { Dtype::NF4 } else { Dtype::F8E4M3 })
    } else if name == "tok.weight" || name == "lm_head.weight" {
        Some(Dtype::BF16)
    } else if name.ends_with("norm.weight") || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") {
        Some(Dtype::F32)
    } else {
        None
    }
}

struct Fixture {
    cfg: Qwen35Config,
    dir: std::path::PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

/// A dense bf16 checkpoint of random weights, converted ONCE into the mixed packed file.
fn fixture(tag: &str) -> Fixture {
    let cfg = cfg();
    let dense = write_dense_checkpoint(&cfg, tag, &weights(&cfg, 0.0));
    let out = dense.join("mixed");
    write_converted_base(&dense, &cfg, BLOCK, &out, &mixed_policy()).expect("convert to the mixed file");
    Fixture { cfg, dir: dense }
}

impl Fixture {
    fn mixed(&self) -> std::path::PathBuf {
        self.dir.join("mixed")
    }

    fn overlay(&self, resume: Option<&std::path::Path>) -> (Qwen35Config, AdapterOverlay) {
        let src = open_dir(&self.mixed(), &self.cfg, BLOCK).expect("the mixed file opens as a base");
        // seed: the adapters' A factors only; both sides of every comparison are built from this same one
        lora_overlay_tuned(BaseCheckpoint { cfg: self.cfg.clone(), src: Box::new(src) }, RANK, ALPHA, Default::default(), 17, resume).expect("adapter overlay")
    }

    fn leaves(&self) -> Vec<String> {
        self.cfg.layer_types().iter().enumerate().flat_map(|(l, ty)| self.cfg.layer_leaves(*ty).into_iter().map(move |(s, _, _)| format!("blocks.{l}.{s}"))).collect()
    }

    /// The mixed model: every frozen linear in the tier its file stores it in.
    fn model(&self, overlay: &(Qwen35Config, AdapterOverlay)) -> Qwen35 {
        let names = self.leaves();
        let base = stored_tier_policy(&overlay.1, names.iter().map(String::as_str));
        Qwen35::new_train_opts_on(Gpu::new(pipelines()), overlay.0.clone(), 1, T as u32, &overlay.1, &TrainOpts { base, ..TrainOpts::default() })
    }

    /// The decoded fp32 values of the mixed file, with the adapters of `overlay`.
    fn decoded(&self, overlay: &(Qwen35Config, AdapterOverlay)) -> HashMap<String, Vec<f32>> {
        use checkpoint::TensorSource;
        let mut map = HashMap::new();
        for (name, _) in overlay.0.param_list() {
            assert!(overlay.1.with_tensor(&name, &mut |d| drop(map.insert(name.clone(), d.to_vec()))), "{name} is served");
        }
        map
    }
}

#[test]
fn the_report_lists_every_tensor_of_the_file_in_the_format_it_is_stored_in() {
    let fx = fixture("report");
    let src = open_dir(&fx.mixed(), &fx.cfg, BLOCK).unwrap();
    let report = src.precision_report();
    let want: Vec<String> = { let mut v: Vec<String> = fx.cfg.param_list().into_iter().map(|(n, _)| n).collect(); v.sort(); v };
    assert_eq!(report.rows().iter().map(|r| r.name.clone()).collect::<Vec<_>>(), want, "a row for every tensor of the model, and no other");
    for row in report.rows() {
        match expected_tier(&fx.cfg, &row.name) {
            Some(tier) => assert_eq!(row.format, encoding_for(tier).unwrap().to_string(), "{}", row.name),
            None => assert!(!row.format.is_empty(), "{}: a tensor the policy leaves as stored still has a stated format", row.name),
        }
    }
    let formats = report.by_format();
    assert!(formats.len() >= 4, "FP8, NF4, bf16 and f32 all appear: {formats:?}");
    eprintln!("{report}");
}

#[test]
fn every_tensor_reaches_the_device_in_its_stored_format() {
    let fx = fixture("resident");
    let overlay = fx.overlay(None);
    let m = fx.model(&overlay);
    let stored = open_dir(&fx.mixed(), &fx.cfg, BLOCK).unwrap().precision_report();
    let resident = m.precision_report();
    for row in stored.rows() {
        let got = resident.get(&row.name).unwrap_or_else(|| panic!("{}: not resident", row.name));
        let Some(tier) = expected_tier(&fx.cfg, &row.name) else { continue };
        assert_eq!(got.format, row.format, "{}: resident format", row.name);
        let widened = (row.shape.iter().product::<u64>() * 4, got.bytes);
        assert!(got.bytes < widened.0 || tier == Dtype::F32, "{}: {} B resident is not below the {} B of an fp32 expansion", row.name, widened.1, widened.0);
        // Resident bytes are the file's bytes, except that the loader widens an FP8 file's bf16 block scales to
        // the f32 the kernels read (two bytes more per scale), and that is the whole difference.
        let scales = if tier == Dtype::F8E4M3 { (row.shape[0].div_ceil(FP8_BLOCK as u64) * row.shape[1].div_ceil(FP8_BLOCK as u64)) * 2 } else { 0 };
        assert_eq!(got.bytes, row.bytes + scales, "{}: resident bytes", row.name);
        if tier != Dtype::F32 && row.shape.len() == 2 && got.shape.len() == 2 {
            assert_eq!(got.bytes, Weight::bytes_for(tier, row.shape[0] as usize, row.shape[1] as usize), "{}: the footprint formula", row.name);
        }
    }
    for name in fx.leaves() {
        assert_eq!(m.linear_dtype(&name), expected_tier(&fx.cfg, &name), "{name}: held as the file stores it");
    }
    eprintln!("{resident}");
}

#[test]
fn the_mixed_model_computes_what_an_fp32_build_over_its_decoded_values_does() {
    let fx = fixture("parity");
    let overlay = fx.overlay(None);
    let mixed = fx.model(&overlay);
    let reference = Qwen35::new_train_opts_on(Gpu::new(pipelines()), overlay.0.clone(), 1, T as u32, &fx.decoded(&overlay), &opts(Dtype::F32));
    // Same values, summed in a different order by different kernels: loss within 1e-4 relative, adapter
    // gradients within cosine 0.9999 (the fp8/nf4/bf16 decodes are exact, only the accumulation order differs).
    assert_same_training_signal(loss_and_grads(&mixed), loss_and_grads(&reference), 1e-4, 0.9999);
}

#[test]
fn lora_over_the_mixed_base_trains_and_its_adapter_saves_and_reloads() {
    let fx = fixture("train");
    let overlay = fx.overlay(None);
    let m = fx.model(&overlay);
    m.set_batch(&TOKENS, &TARGETS);
    let before = m.forward();
    for step in 1..=60 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let after = m.forward();
    assert!(after < before * 0.8, "LoRA over the mixed base did not learn: {before} -> {after}");
    let live = m.logits_all(&TOKENS);

    let adapter = fx.dir.join("adapter.safetensors");
    qwen35::lora::save_adapter(adapter.to_str().unwrap(), &m, "test/adapter", "test/base", None).expect("save the adapter");

    // Reload: the saved adapter over the same packed base reproduces the trained model.
    let reloaded_overlay = fx.overlay(Some(&adapter));
    let reloaded = fx.model(&reloaded_overlay);
    let reloaded_logits = reloaded.logits_all(&TOKENS);
    let mean_abs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    assert!(mean_abs(&live, &reloaded_logits) < 1e-5, "the reloaded adapter does not reproduce the trained logits: {}", mean_abs(&live, &reloaded_logits));

    // Folding the adapter into the decoded base gives the same function.
    let mut folded = fx.decoded(&overlay);
    folded.retain(|n, _| !n.contains(".lora_"));
    qwen35::lora::fold_adapter_into(&mut folded, adapter.to_str().unwrap()).expect("fold");
    let plain = Qwen35::new_on(Gpu::new(pipelines()), fx.cfg.clone(), 1, T as u32, &folded);
    let folded_logits = plain.logits_all(&TOKENS);
    assert!(mean_abs(&live, &folded_logits) < 1e-3, "folded base differs from the live adapter forward: {}", mean_abs(&live, &folded_logits));
    // And the adapter really moved the function away from the untouched base.
    let untouched = fx.model(&fx.overlay(None)).logits_all(&TOKENS);
    assert!(mean_abs(&live, &untouched) > 1e-3, "training left the logits where they were");
}
