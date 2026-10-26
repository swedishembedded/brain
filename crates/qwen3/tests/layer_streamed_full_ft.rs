// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models that do not fit on one
// accelerator for its clients. If your team needs expertise in training every weight of a model larger than the
// device memory, then you can procure our services by sending an email to info@swedishembedded.com.

//! A LAYER-STREAMED full fine-tune of a tiny random Qwen3 through the generic fit loop: only a window of layers is on
//! the device, every layer is loaded from its host copy when the forward or the backward reaches it, and its weights
//! are updated between the backward segments.
//!
//! * the run ends bit for bit where the resident run (every layer on the device) ends, with and without the global
//!   clip, in the exact two-pass form;
//! * the device holds the window's layers and not the model;
//! * a run stopped half way and resumed ends exactly where the uninterrupted one does;
//! * the single-pass running clip learns the task.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use model::memplan::Capacity;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentValue};
use model::Model;
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen3::{Qwen, QwenConfig, ResidentPlan};

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen3-streamed-{name}-{}", std::process::id()));
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
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();
}

/// Three layers, so a window of one or two really rotates.
fn cfg() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: 16, n_layers: 3, d_model: 32, n_heads: 4, n_kv_heads: 2, head_dim: 8, d_ff: 64, max_position_embeddings: 16, ..QwenConfig::tiny() }
}

fn opts(steps: u32, grad_clip: f32) -> model::FitOpts {
    model::FitOpts { steps, batch_size: 1, block_size: 16, lr: 1e-2, min_lr: 1e-3, warmup: 2, decay_iters: steps, weight_decay: 0.0, grad_clip, eval_interval: 0, eval_batches: 0, seed: 3, ..Default::default() }
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

fn build(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass) -> Qwen {
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    Qwen::new_resident(cfg, 1, 16, &init, &plan(value, window, clip_pass), gpu_core::provider::library::GemmPrecision::Fp32Reference).unwrap()
}

fn linears(m: &Qwen) -> Vec<(String, Vec<f32>)> {
    let mut names: Vec<String> = m.param_names().into_iter().filter(|n| n.starts_with("blocks.") && n.ends_with(".weight") && !n.contains("norm") && !n.contains("ln")).collect();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

/// Spec: layer streaming changes where the weights live, not the arithmetic: after eight steps the layer-streamed
/// run's losses and master weights equal the resident run's, bit for bit, for windows of one and two layers, with
/// the clip off and with an active global clip.
#[test]
fn a_layer_streamed_run_ends_exactly_where_the_resident_run_does() {
    let scratch = tmp("equal");
    write_dataset(&scratch);
    // Every window and clip form once, FP8 included, without the product of all of them.
    for (value, clip, windows) in [(ResidentValue::Bf16, 0.0f32, vec![1u32, 2]), (ResidentValue::Bf16, 0.05, vec![1]), (ResidentValue::Fp8E4M3 { block: 128 }, 0.05, vec![2])] {
        let opts = opts(8, clip);
        let (want_report, want) = model::fit_controlled(build(value, None, ClipPass::TwoPass), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
        for window in windows {
            let (report, got) = model::fit_controlled(build(value, Some(window), ClipPass::TwoPass), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
            assert_eq!(report.final_loss.unwrap().to_bits(), want_report.final_loss.unwrap().to_bits(), "{value:?} clip {clip} window {window}: final loss");
            for ((name, a), (_, b)) in linears(&want).iter().zip(linears(&got)) {
                assert_eq!(a, &b, "{value:?} clip {clip} window {window}: master of {name}");
            }
        }
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: the device holds the window's layers, not the model, beside the bf16 embedding table (tied to the head),
/// which stays resident whatever the window.
#[test]
fn the_device_holds_the_window_not_the_model() {
    let table = ResidentValue::Bf16.device_bytes(u64::from(cfg().vocab), u64::from(cfg().d_model));
    let layers = |window| build(ResidentValue::Bf16, window, ClipPass::TwoPass).resident_device_weight_bytes().unwrap() - table;
    let (resident, two, one) = (layers(None), layers(Some(2)), layers(Some(1)));
    assert_eq!((resident, two, one), (3 * one, 2 * one, one), "three layers resident, a window of two, a window of one");
}

/// Spec: a run stopped at step 4 and resumed from its state file in a fresh layer-streamed model ends exactly where
/// the uninterrupted run ends.
#[test]
fn a_resumed_layer_streamed_run_ends_exactly_where_the_uninterrupted_run_does() {
    let scratch = tmp("resume");
    write_dataset(&scratch);
    let opts = opts(8, 0.05);
    let identity = serde_json::json!({"test": "layer-streamed-resume"});
    let value = ResidentValue::Fp8E4M3 { block: 128 };
    let (straight_report, straight) = model::fit_controlled(build(value, Some(1), ClipPass::TwoPass), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
    let state = scratch.join("run.state");
    let mut stop_at_4 = |s: &model::StepReport| s.step < 4;
    let control = model::FitControl { on_step: Some(&mut stop_at_4), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = model::fit_controlled(build(value, Some(1), ClipPass::TwoPass), objective::<Qwen>(&scratch, &opts), &opts, None, control).unwrap();
    assert!(first.interrupted && state.is_file());
    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = model::fit_controlled(build(value, Some(1), ClipPass::TwoPass), objective::<Qwen>(&scratch, &opts), &opts, None, control).unwrap();
    assert_eq!(second.resumed_at, Some(4));
    assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits());
    for ((name, a), (_, b)) in linears(&straight).iter().zip(linears(&resumed)) {
        assert_eq!(a, &b, "master of {name}");
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: the single-pass running clip trains the task (it clips each layer by the norm seen so far, never harder
/// than the global clip).
#[test]
fn the_running_clip_learns_the_task() {
    let scratch = tmp("running");
    write_dataset(&scratch);
    let opts = opts(40, 0.5);
    let (report, _) = model::fit_controlled(build(ResidentValue::Bf16, Some(1), ClipPass::Running), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
    assert!(report.final_loss.unwrap() < 0.5 * report.initial_loss, "{report:?}");
    std::fs::remove_dir_all(&scratch).unwrap();
}
