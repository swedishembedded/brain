// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models that do not fit on one
// accelerator for its clients. If your team needs expertise in training every weight of a hybrid linear-attention
// model larger than the device memory, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A resident and a LAYER-STREAMED full fine-tune of the tiny hybrid Qwen3.5 (Gated-DeltaNet and GQA layers) through
//! the generic fit loop. The per-layer linears live in bf16 or FP8 with an off-device master; the layer-streamed
//! build keeps only a window of layers on the device and updates each layer between the backward segments.
//!
//! * the layer-streamed run ends bit for bit where the resident run ends, with and without the global clip, with the
//!   forward's activations kept and recomputed;
//! * the device holds the window's layers and not the model;
//! * a run stopped half way and resumed ends exactly where the uninterrupted one does;
//! * a plan that does not fit is refused before anything is allocated.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use gpu_core::Gpu;
use model::memplan::Capacity;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentPlan, ResidentValue};
use model::saved::SavedPolicy;
use model::Model;
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const T: u32 = 16;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn cfg() -> Qwen35Config {
    Qwen35Config { block_size: T, max_position_embeddings: T, ..Qwen35Config::tiny() }
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-streamed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_dataset(dir: &Path) {
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for _ in 0..200 {
        tokens.extend_from_slice(&PROMPT);
        tokens.push(TARGET);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(cfg().vocab as usize)).unwrap();
}

fn opts(steps: u32, grad_clip: f32) -> model::FitOpts {
    model::FitOpts { steps, batch_size: 1, block_size: T, lr: 1e-2, min_lr: 1e-3, warmup: 2, decay_iters: steps, weight_decay: 0.0, grad_clip, eval_interval: 0, eval_batches: 0, seed: 3, ..Default::default() }
}

fn objective<M: Model>(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<M> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<M>(train, val, bcfg, itos)
}

fn plan(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass) -> ResidentPlan {
    let cfg = ResidentFtConfig {
        value,
        master: MasterPolicy::F32,
        state: StateFormat::Int8Block { block: 64 },
        rule: Rule::AdamW { adam: model::Adam::default(), wd: 0.0 },
        grad: GradDtype::F32,
        stochastic_state: false,
        seed: Some(5),
        tile_elems: 256,
        state_dir: None,
        defer_publication: false,
        window: window.map(|layers| LayerWindow { layers }),
        clip_pass,
    };
    ResidentPlan { cfg, capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: Vec::new() }
}

fn try_build(plan: ResidentPlan, saved: SavedPolicy) -> Result<Qwen35, String> {
    let cfg = cfg();
    let init = qwen35::init::init_weights(&cfg, 11);
    let opts = TrainOpts { saved, gemm_precision: gpu_core::provider::library::GemmPrecision::Fp32Reference, resident: Some(plan), ..TrainOpts::default() };
    Qwen35::new_resident(Gpu::new(pipelines()), cfg, 1, T, &init, &opts)
}

fn build(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass, saved: SavedPolicy) -> Qwen35 {
    try_build(plan(value, window, clip_pass), saved).expect("builds")
}

fn weights(m: &Qwen35) -> Vec<(String, Vec<f32>)> {
    let mut names = m.param_names();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

fn fit(m: Qwen35, dir: &Path, opts: &model::FitOpts) -> (model::FitReport, Qwen35) {
    model::fit_controlled(m, objective::<Qwen35>(dir, opts), opts, None, model::FitControl::default()).unwrap()
}

/// Spec: layer streaming changes where the weights live, not the arithmetic: after four steps the layer-streamed run's
/// final loss and every weight (the linears' masters and the fp32 embedding, head and norms) equal the resident
/// run's, bit for bit, with the clip off and with an active global clip, and with the forward's activations kept or
/// recomputed.
#[test]
fn a_layer_streamed_run_ends_exactly_where_the_resident_run_does() {
    let scratch = tmp("equal");
    write_dataset(&scratch);
    let cases = [(ResidentValue::Bf16, 0.05f32, SavedPolicy::Keep, 1u32), (ResidentValue::Fp8E4M3 { block: 128 }, 0.0, SavedPolicy::Recompute, 2)];
    for (value, clip, saved, window) in cases {
        let opts = opts(4, clip);
        let (want_report, want) = fit(build(value, None, ClipPass::TwoPass, saved), &scratch, &opts);
        let (report, got) = fit(build(value, Some(window), ClipPass::TwoPass, saved), &scratch, &opts);
        let label = format!("{value:?} clip {clip} {saved:?} window {window}");
        if clip > 0.0 {
            assert!(want.last_grad_norm().unwrap() > clip, "{label}: the clip is active");
            assert_eq!(want.last_grad_norm().unwrap().to_bits(), got.last_grad_norm().unwrap().to_bits(), "{label}: the measured global norm");
        }
        assert_eq!(report.final_loss.unwrap().to_bits(), want_report.final_loss.unwrap().to_bits(), "{label}: final loss");
        for ((name, a), (_, b)) in weights(&want).iter().zip(weights(&got)) {
            assert_eq!(a, &b, "{label}: {name}");
        }
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: the device holds the window's layers (per layer shape), not the model.
#[test]
fn the_device_holds_the_window_not_the_model() {
    let bytes = |window| build(ResidentValue::Bf16, window, ClipPass::TwoPass, SavedPolicy::Keep).resident_device_weight_bytes().unwrap();
    let (all, one, two) = (bytes(None), bytes(Some(1)), bytes(Some(2)));
    assert!(one < two && two < all, "window of one ({one}) < window of two ({two}) < every layer ({all})");
    // Three Gated-DeltaNet layers and one GQA layer: a window of one holds one of each kind, so the whole model is
    // four layers' worth of leaves and a window of one holds the larger two kinds once.
    assert!(all > 2 * one, "the model ({all}) is more than twice a one-layer window ({one})");
}

/// Spec: a run stopped at step 3 and resumed from its state file in a fresh layer-streamed model ends exactly where the
/// uninterrupted run ends.
#[test]
fn a_resumed_layer_streamed_run_ends_exactly_where_the_uninterrupted_run_does() {
    let scratch = tmp("resume");
    write_dataset(&scratch);
    let opts = opts(6, 0.05);
    let identity = serde_json::json!({"test": "qwen35-layer-streamed-resume"});
    let value = ResidentValue::Bf16;
    let fresh = || build(value, Some(1), ClipPass::TwoPass, SavedPolicy::Keep);
    let (straight_report, straight) = fit(fresh(), &scratch, &opts);
    let state = scratch.join("run.state");
    let mut stop_at_3 = |s: &model::StepReport| s.step < 3;
    let control = model::FitControl { on_step: Some(&mut stop_at_3), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = model::fit_controlled(fresh(), objective::<Qwen35>(&scratch, &opts), &opts, None, control).unwrap();
    assert!(first.interrupted && state.is_file());
    assert!(state.with_extension("offdevice").is_dir(), "the master copies are committed beside the state file");
    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = model::fit_controlled(fresh(), objective::<Qwen35>(&scratch, &opts), &opts, None, control).unwrap();
    assert_eq!(second.resumed_at, Some(3));
    assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits());
    for ((name, a), (_, b)) in weights(&straight).iter().zip(weights(&resumed)) {
        assert_eq!(a, &b, "{name}");
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: a plan the machine cannot hold is refused before anything is allocated, naming the category; one the
/// window makes fit is accepted.
#[test]
fn a_plan_that_does_not_fit_is_refused_and_the_window_makes_it_fit() {
    let tight = |p: &mut ResidentPlan, device_bytes| p.capacity = Capacity { device_bytes, host_bytes: 1 << 30 };
    let mut resident = plan(ResidentValue::Bf16, None, ClipPass::TwoPass);
    tight(&mut resident, 100_000);
    let e = try_build(resident, SavedPolicy::Keep).err().expect("refused");
    assert!(e.contains("does not fit") && e.contains("Weights"), "{e}");
    let mut windowed = plan(ResidentValue::Bf16, Some(1), ClipPass::TwoPass);
    tight(&mut windowed, 100_000);
    // The window's weights and one layer of f32 gradients are what the planner prices.
    let demands = model::resident_ft::demands(&windowed.cfg, &qwen35::model::resident_specs(&cfg(), &model::Shard::whole(4))).unwrap();
    let device = model::memplan::device_total(&demands);
    tight(&mut windowed, device);
    try_build(windowed, SavedPolicy::Keep).expect("a window fits where the whole model does not");
}

/// Spec: a resident full fine-tune holds the vocabulary tables (the embedding table and the untied LM head) in its own
/// bf16 state with the master off the device, not as fp32 parameters with fp32 gradients and moments: both are bf16
/// device weights, neither is a parameter-store tensor, and both train, resident or layer-streamed.
#[test]
fn the_vocabulary_tables_train_in_bf16_outside_the_parameter_store() {
    let scratch = tmp("tables");
    write_dataset(&scratch);
    let opts = opts(3, 0.0);
    for window in [None, Some(1)] {
        let m = build(ResidentValue::Fp8E4M3 { block: 128 }, window, ClipPass::TwoPass, SavedPolicy::Keep);
        let tables = ["tok.weight", "lm_head.weight"];
        for name in tables {
            assert_eq!(m.linear_dtype(name), Some(gpu_core::select::Dtype::BF16), "window {window:?}: {name} is a bf16 device weight");
        }
        let report = m.precision_report();
        for name in tables {
            let row = report.get(name).unwrap_or_else(|| panic!("window {window:?}: {name} is not in the precision report"));
            assert_eq!(row.format, checkpoint::packed::QuantEncoding::bf16().to_string(), "window {window:?}: {name} is held as {}", row.format);
        }
        let before: Vec<Vec<f32>> = tables.iter().map(|n| m.read_weight(n)).collect();
        let (_, m) = fit(m, &scratch, &opts);
        for (name, before) in tables.iter().zip(before) {
            assert_ne!(m.read_weight(name), before, "window {window:?}: {name} trained");
        }
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}
