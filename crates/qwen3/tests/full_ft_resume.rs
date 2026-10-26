// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models for its clients. If your team
// needs expertise in resuming and verifying long training runs exactly, with the optimizer state anywhere from
// device memory to disk, you can procure our services by sending an email to info@swedishembedded.com.

//! A full fine-tune stopped half way and resumed from its state file ends bit for bit where the uninterrupted run
//! ends, wherever its AdamW state lives (on the device in f32, bf16 or int8 blocks, or off it in host memory); a
//! corrupted generation is refused rather than replaced by an older one; the precision of a run is one typed value
//! and an unsupported choice is an error naming the parameter; and the planner narrows the optimizer state only
//! when the run authorized it, reporting the decision.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use model::memplan::Capacity;
use model::{Model, TrainPrecision};
use qwen3::{Dtype, Qwen, QwenConfig};

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen3-fullft-{name}-{}", std::process::id()));
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

fn cfg() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: 16, n_layers: 2, d_model: 32, n_heads: 4, n_kv_heads: 2, head_dim: 8, d_ff: 64, max_position_embeddings: 16, ..QwenConfig::tiny() }
}

fn opts(steps: u32, precision: &str) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: 16,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup: 2,
        decay_iters: steps,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed: data::rng::random_seed(),
        precision: TrainPrecision::parse(precision).unwrap(),
        ..Default::default()
    }
}

fn objective<M: Model>(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<M> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<M>(train, val, bcfg, itos)
}

fn build(opts: &model::FitOpts, offload: bool) -> Qwen {
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    qwen3::finetune::build_for_training_with(cfg, opts, &init, Dtype::F32, offload).unwrap()
}

fn weights(m: &Qwen) -> Vec<(String, Vec<f32>)> {
    let mut names = m.param_names();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

/// Spec: a run stopped at step 4 and resumed in a fresh model ends exactly where the uninterrupted run ends, with
/// the AdamW state on the device in f32, bf16 or int8 blocks and off it in host memory in f32 or int8 blocks
/// (stochastic rounding on: the draws are keyed on the run's seed and the step, so the resume replays them).
#[test]
fn a_resumed_full_finetune_ends_exactly_where_the_uninterrupted_run_does() {
    let scratch = tmp("resume");
    write_dataset(&scratch);
    let identity = serde_json::json!({"test": "full-ft-resume"});
    for (label, offload, precision) in [
        ("device f32", false, "state=f32"),
        ("device bf16", false, "state=bf16,sr=on"),
        ("device int8", false, "state=int8:64,sr=on"),
        ("host f32", true, "state=f32"),
        ("host int8", true, "state=int8:64,sr=on"),
    ] {
        let run = opts(8, precision);
        let (straight_report, straight) = model::fit_controlled(build(&run, offload), objective::<Qwen>(&scratch, &run), &run, None, model::FitControl::default()).unwrap();

        let state = scratch.join(format!("{}.state", label.replace(' ', "-")));
        let mut stop_at_4 = |s: &model::StepReport| s.step < 4;
        let control = model::FitControl { on_step: Some(&mut stop_at_4), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
        let (first, _) = model::fit_controlled(build(&run, offload), objective::<Qwen>(&scratch, &run), &run, None, control).unwrap();
        assert!(first.interrupted && state.is_file(), "{label}: stopped at step 4 with a state file");

        let control = model::FitControl { state: Some(&state), identity: identity.clone(), ..Default::default() };
        let (second, resumed) = model::fit_controlled(build(&run, offload), objective::<Qwen>(&scratch, &run), &run, None, control).unwrap();
        assert_eq!(second.resumed_at, Some(4), "{label}");
        assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits(), "{label}: the last loss is identical");
        for ((name, a), (_, b)) in weights(&straight).iter().zip(weights(&resumed)) {
            assert_eq!(a, &b, "{label}: {name}");
        }
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: a generation whose files no longer verify is refused by name; the resume does not quietly continue from an
/// older state than the one its resume file names.
#[test]
fn a_corrupt_off_device_generation_is_refused_not_replaced_by_an_older_one() {
    let scratch = tmp("corrupt");
    write_dataset(&scratch);
    let run = opts(8, "state=f32");
    let identity = serde_json::json!({"test": "full-ft-corrupt"});
    let state = scratch.join("run.state");
    let mut stop_at_4 = |s: &model::StepReport| s.step < 4;
    let control = model::FitControl { on_step: Some(&mut stop_at_4), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    model::fit_controlled(build(&run, true), objective::<Qwen>(&scratch, &run), &run, None, control).unwrap();
    let generation = std::fs::read_dir(state.with_extension("offdevice")).unwrap().filter_map(Result::ok).map(|e| e.path()).find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("gen-"))).expect("a committed generation");
    let victim = generation.join("state.bin");
    let bytes = std::fs::read(&victim).unwrap();
    std::fs::write(&victim, &bytes[..bytes.len() / 2]).unwrap();

    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let err = model::fit_controlled(build(&run, true), objective::<Qwen>(&scratch, &run), &run, None, control).err().expect("refused");
    assert!(err.to_string().contains("gen-") || err.to_string().contains("committed"), "{err}");
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: a choice the trainer does not run is an error naming the parameter, never a panic.
#[test]
fn an_unsupported_precision_is_an_error_naming_the_parameter() {
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    for (precision, offload, needle) in [("master=f32", true, "master"), ("state=log8", false, "host-offloaded"), ("value=bf16", false, "f32 device weights")] {
        let run = opts(2, precision);
        let err = qwen3::finetune::build_for_training_with(cfg.clone(), &run, &init, Dtype::F32, offload).err().expect("refused");
        assert!(err.to_string().contains(needle) && err.to_string().contains("tok.weight"), "{precision}: {err}");
    }
}

/// Spec: when the requested optimizer state does not fit the device, the run is refused with the planner's
/// arithmetic, or - when it authorized narrowing - gets the narrower state with the decision reported.
#[test]
fn the_planner_narrows_the_optimizer_state_only_when_authorized() {
    let cfg = cfg();
    let strict = opts(2, "state=f32");
    let may_narrow = opts(2, "state=f32,narrow=on");
    let full = qwen3::finetune::full_ft_footprint_bytes(&cfg, &strict, false);
    // Room for the model with bf16 moments but not with f32 ones.
    let capacity = Capacity { device_bytes: full - 1, host_bytes: u64::MAX };
    let err = qwen3::finetune::plan_full_precision(&cfg, &strict, false, capacity).expect_err("refused");
    assert!(err.contains("does not fit") && err.contains("OptimizerState"), "{err}");
    let planned = qwen3::finetune::plan_full_precision(&cfg, &may_narrow, false, capacity).unwrap();
    assert!(matches!(planned.precision.policy.default.state, paramstore::precision::StateFormat::Bf16 | paramstore::precision::StateFormat::Int8Block { .. }));
    assert!(planned.report.decisions.iter().any(|d| d.lever == model::memplan::Lever::OptimizerState), "{}", planned.report);
}
