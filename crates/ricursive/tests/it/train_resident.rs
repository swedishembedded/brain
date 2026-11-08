// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident low-precision full fine-tuning of Ripple: the linears and the two tables on the device in bf16 or FP8 with
//! their master copy and optimizer state off it, optionally streamed through a window of layers, through the generic fit
//! loop on a tiny model.
//!
//! * the bf16 run follows the f32 run (loss and gradients within bf16's tolerance) and learns;
//! * a layer-streamed run ends bit for bit where the resident run ends, with and without the global clip;
//! * the device holds the window's layers and not the model;
//! * a run stopped half way and resumed ends exactly where the uninterrupted one does;
//! * a request the build cannot honour is refused by name.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, with the master copy and the optimizer state where the memory is. If your team needs expertise in
//! training every weight of a model that does not fit an accelerator then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use model::memplan::Capacity;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentPlan, ResidentValue};
use model::{Batch, Model, Shard};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use ricursive::train::{plan_resident, resident_specs};
use ricursive::{Ripple, TrainConfig, TrainOpts};

const VOCAB: usize = 32;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;
const SEQ: u32 = 8;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-ricursive-resident-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A corpus of one repeated sentence whose last token is the only supervised one.
fn write_dataset(dir: &Path) {
    let (mut tokens, mut mask) = (Vec::new(), Vec::new());
    for _ in 0..200 {
        tokens.extend_from_slice(&PROMPT);
        tokens.push(TARGET);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB)).unwrap();
}

fn opts(steps: u32, grad_clip: f32) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: SEQ,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup: 2,
        decay_iters: steps,
        weight_decay: 0.0,
        grad_clip,
        eval_interval: 0,
        eval_batches: 0,
        seed: data::rng::random_seed(),
        precision: model::TrainPrecision::parse("state=f32").unwrap(),
        ..Default::default()
    }
}

fn objective(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<Ripple> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<Ripple>(train, val, bcfg, itos)
}

fn resident_cfg(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass) -> ResidentFtConfig {
    ResidentFtConfig {
        value,
        master: MasterPolicy::F32,
        state: StateFormat::F32,
        rule: Rule::AdamW { adam: model::Adam::default(), wd: 0.0 },
        grad: GradDtype::F32,
        stochastic_state: false,
        seed: None,
        tile_elems: 256,
        state_dir: None,
        defer_publication: false,
        window: window.map(|layers| LayerWindow { layers }),
        clip_pass,
    }
}

fn plan(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass) -> ResidentPlan {
    let cfg = TrainConfig::tiny();
    plan_resident(&cfg, &Shard::whole(cfg.model.num_hidden_layers), 1, SEQ, &resident_cfg(value, window, clip_pass), Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 })
}

fn build_plain() -> Ripple {
    let cfg = TrainConfig::tiny();
    Ripple::build(cfg.clone(), 1, SEQ, &cfg.tiny_weights(7), &TrainOpts::default(), true).unwrap()
}

fn build(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass) -> Ripple {
    let cfg = TrainConfig::tiny();
    let opts = TrainOpts { resident: Some(plan(value, window, clip_pass)), ..TrainOpts::default() };
    let shard = Shard::whole(cfg.model.num_hidden_layers);
    Ripple::build_shard(cfg.clone(), 1, SEQ, &cfg.tiny_weights(7), &opts, true, shard).unwrap_or_else(|e| panic!("{e}"))
}

fn tensors(m: &Ripple) -> Vec<(String, Vec<f32>)> {
    let mut names = m.param_names();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

fn fit(m: Ripple, dir: &Path, o: &model::FitOpts) -> (model::FitReport, Ripple) {
    model::fit_controlled(m, objective(dir, o), o, None, model::FitControl::default()).unwrap()
}

/// Spec: the resident run's tensors are the model's: the same names as the f32 build's, every linear and the two tables
/// among them, none left in the parameter store.
#[test]
fn a_resident_build_trains_every_tensor_of_the_model() {
    let plain = build_plain();
    let resident = build(ResidentValue::Bf16, None, ClipPass::TwoPass);
    let (mut a, mut b) = (plain.param_names(), resident.param_names());
    a.sort();
    b.sort();
    assert_eq!(a, b);
    let specs = resident_specs(&TrainConfig::tiny(), &Shard::whole(3));
    assert_eq!(specs.len(), 3 * 8 + 2, "eight linears of each of three layers, the embedding and the head");
    assert!(specs.iter().all(|s| resident.has_param(&s.name)) && resident.has_param("model.layers.0.input_layernorm.weight"), "the norms are the store's");
}

/// Spec: a bf16 resident gradient is the f32 build's within bf16's tolerance, tensor by tensor (the stored weights are
/// the master rounded to bf16, so the loss moves in the third digit and the gradients by a few percent).
#[test]
fn the_resident_loss_and_gradients_are_the_f32_builds_within_bf16() {
    let (plain, resident) = (build_plain(), build(ResidentValue::Bf16, None, ClipPass::TwoPass));
    let (x, y): (Vec<u32>, Vec<u32>) = ((0..SEQ).map(|i| (i * 5 + 1) % VOCAB as u32).collect(), (0..SEQ).map(|i| (i * 5 + 6) % VOCAB as u32).collect());
    let run = |m: &Ripple| {
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        m.zero_grads();
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        (loss, m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect::<Vec<_>>())
    };
    let ((want_loss, want), (loss, got)) = (run(&plain), run(&resident));
    eprintln!("loss {want_loss} (f32) vs {loss} (bf16 resident)");
    assert!((loss - want_loss).abs() <= 3e-2 * want_loss, "{loss} vs {want_loss}");
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let mut worst = (0f64, String::new());
    for (name, w) in &want {
        let g = &got.iter().find(|(n, _)| n == name).expect("the same tensors").1;
        let diff = norm(&g.iter().zip(w).map(|(a, b)| a - b).collect::<Vec<_>>()) / norm(w).max(1e-9);
        if diff > worst.0 {
            worst = (diff, name.clone());
        }
    }
    eprintln!("worst relative gradient difference {:.3e} ({})", worst.0, worst.1);
    assert!(worst.0 < 0.15, "{} differs by {:.3e}", worst.1, worst.0);
}

/// Spec: through the fit loop the bf16 resident run learns the corpus as the f32 run does: the loss falls to a fraction
/// of its start and ends near the f32 run's.
#[test]
fn a_bf16_resident_run_learns_as_the_f32_run_does() {
    let dir = scratch("learn");
    write_dataset(&dir);
    let o = opts(40, 1.0);
    let (want, _) = fit(build_plain(), &dir, &o);
    let (got, _) = fit(build(ResidentValue::Bf16, None, ClipPass::TwoPass), &dir, &o);
    let (w, g) = (want.final_loss.unwrap(), got.final_loss.unwrap());
    eprintln!("loss {:.4} -> {w:.4} (f32), {:.4} -> {g:.4} (bf16 resident)", want.initial_loss, got.initial_loss);
    assert!(g < 0.25 * got.initial_loss, "the resident run learns: {} -> {g}", got.initial_loss);
    assert!((g - w).abs() < 0.1 * want.initial_loss, "the resident run ends near the f32 run: {g} vs {w}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Spec: layer streaming changes where the weights live, not the arithmetic: after eight steps the layer-streamed run's
/// losses and master weights equal the resident run's, bit for bit, for windows of one and two layers, with the clip off
/// and with an active global clip, in bf16 and FP8.
#[test]
fn a_layer_streamed_run_ends_exactly_where_the_resident_run_does() {
    let dir = scratch("equal");
    write_dataset(&dir);
    for (value, clip, windows) in [(ResidentValue::Bf16, 0.0f32, vec![1u32, 2]), (ResidentValue::Bf16, 0.05, vec![1]), (ResidentValue::Fp8E4M3 { block: 128 }, 0.05, vec![2])] {
        let o = opts(8, clip);
        let (want_report, want) = fit(build(value, None, ClipPass::TwoPass), &dir, &o);
        for window in windows {
            let (report, got) = fit(build(value, Some(window), ClipPass::TwoPass), &dir, &o);
            assert_eq!(report.final_loss.unwrap().to_bits(), want_report.final_loss.unwrap().to_bits(), "{value:?} clip {clip} window {window}: final loss");
            for ((name, a), (_, b)) in tensors(&want).iter().zip(tensors(&got)) {
                assert_eq!(a, &b, "{value:?} clip {clip} window {window}: master of {name}");
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Spec: the device holds the window's layers, not the model, beside the two bf16 tables, which stay resident.
#[test]
fn the_device_holds_the_window_not_the_model() {
    let table = 2 * ResidentValue::Bf16.device_bytes(VOCAB as u64, TrainConfig::tiny().model.hidden_size as u64);
    let layers = |window| build(ResidentValue::Bf16, window, ClipPass::TwoPass).resident_device_weight_bytes().unwrap() - table;
    let (resident, two, one) = (layers(None), layers(Some(2)), layers(Some(1)));
    assert_eq!((resident, two, one), (3 * one, 2 * one, one), "three layers resident, a window of two, a window of one");
}

/// Spec: a run stopped at step 4 and resumed from its state file in a fresh model ends exactly where the uninterrupted
/// run ends: resident, and layer-streamed.
#[test]
fn a_resumed_run_ends_exactly_where_the_uninterrupted_run_does() {
    let dir = scratch("resume");
    write_dataset(&dir);
    let o = opts(8, 0.05);
    let value = ResidentValue::Fp8E4M3 { block: 128 };
    for window in [None, Some(1)] {
        let identity = serde_json::json!({"test": "ricursive-resident-resume", "window": window});
        let (straight_report, straight) = fit(build(value, window, ClipPass::TwoPass), &dir, &o);
        let state = dir.join(format!("run-{window:?}.state"));
        let mut stop_at_4 = |s: &model::StepReport| s.step < 4;
        let control = model::FitControl { on_step: Some(&mut stop_at_4), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
        let (first, _) = model::fit_controlled(build(value, window, ClipPass::TwoPass), objective(&dir, &o), &o, None, control).unwrap();
        assert!(first.interrupted && state.is_file(), "stopped at step 4 with a state file");
        let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
        let (second, resumed) = model::fit_controlled(build(value, window, ClipPass::TwoPass), objective(&dir, &o), &o, None, control).unwrap();
        assert_eq!(second.resumed_at, Some(4));
        assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits(), "window {window:?}: the last loss");
        for ((name, a), (_, b)) in tensors(&straight).iter().zip(tensors(&resumed)) {
            assert_eq!(a, &b, "window {window:?}: {name}");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Spec: the single-pass running clip trains the task.
#[test]
fn the_running_clip_learns_the_task() {
    let dir = scratch("running");
    write_dataset(&dir);
    let o = opts(40, 0.5);
    let (report, _) = fit(build(ResidentValue::Bf16, Some(1), ClipPass::Running), &dir, &o);
    assert!(report.final_loss.unwrap() < 0.5 * report.initial_loss, "{report:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Spec: a request the resident build cannot honour is refused by name: an adapter beside the resident state, a
/// packed tier given beside it, a stage count, and a precision the plain build does not run.
#[test]
fn what_the_resident_build_cannot_honour_is_refused_by_name() {
    let cfg = TrainConfig::tiny();
    let with_lora = cfg.clone().with_lora(model::adapter::device::LoraCfg::new(2, 4.0, ricursive::train::default_targets()));
    let opts = TrainOpts { resident: Some(plan(ResidentValue::Bf16, None, ClipPass::TwoPass)), ..TrainOpts::default() };
    let err = Ripple::build(with_lora.clone(), 1, SEQ, &with_lora.tiny_weights(7), &opts, true).err().expect("refused");
    assert!(err.contains("full fine-tune"), "{err}");
    let packed = TrainOpts { tiers: model::ops::TierPolicy::uniform(gpu_core::select::Dtype::BF16), ..opts };
    let err = Ripple::build(cfg.clone(), 1, SEQ, &cfg.tiny_weights(7), &packed, true).err().expect("refused");
    assert!(err.contains("bf16") && err.contains("full fine-tune"), "{err}");

    let nowhere = Path::new("/nonexistent/ricursive-base");
    let resident = model::TrainPrecision::parse("weights=bf16").unwrap();
    let staged = model::FitOpts { precision: resident, stages: 2, ..model::FitOpts::default() };
    let err = model::finetune::fit_full::<ricursive::finetune::RippleTrainer>(nowhere, nowhere, &staged, nowhere, model::finetune::OptimizerState::HostOffload, model::FitControl::default()).expect_err("refused");
    assert!(err.to_string().contains("one stage"), "{err}");
    for spec in ["state=int8", "weights=f32,window=2"] {
        let o = model::FitOpts { precision: model::TrainPrecision::parse(spec).unwrap(), ..model::FitOpts::default() };
        let err = model::finetune::fit_full::<ricursive::finetune::RippleTrainer>(nowhere, nowhere, &o, nowhere, model::finetune::OptimizerState::HostOffload, model::FitControl::default()).expect_err(spec);
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{spec}: {err}");
    }
}

/// The published Ripple-65B configuration (hidden 5120, 80 layers of four sliding and one global, 100,352 tokens).
fn ripple_65b() -> TrainConfig {
    let layer_types: Vec<&str> = (0..80).map(|l| if l % 5 == 4 { "full_attention" } else { "sliding_attention" }).collect();
    let model = ricursive::RicursiveConfig::from_json(
        &serde_json::json!({
            "model_type": "ricursive", "vocab_size": 100352, "hidden_size": 5120, "intermediate_size": 40960, "num_hidden_layers": 80,
            "num_attention_heads": 80, "num_key_value_heads": 8, "head_dim": 128, "sliding_window": 1024, "use_valres": true,
            "use_gated_attention": true, "ffn_clamp_threshold": 20.0, "embed_scale": 71.554, "layer_types": layer_types,
        })
        .to_string(),
    )
    .expect("the published configuration is valid");
    TrainConfig::new(model, 4096)
}

const GIB: f64 = (1u64 << 30) as f64;

/// Where a 65B full fine-tune lands on the cards of a GH200 node, from the planner's own arithmetic. Prints the table
/// `ripple.md` records, and holds the relationships it rests on: a layer window bounds the device bytes by the window
/// and not by the model, the stored bf16 weights are the same bytes on the host whatever the window, and the
/// activations are the build's own count.
#[test]
fn the_65b_full_fine_tune_placement_on_two_and_four_gh200() {
    use model::Shardable;
    let cfg = ripple_65b();
    let (b, t) = (1u32, 4096u32);
    let card = 97_871.0 * 1024.0 * 1024.0 / GIB; // GiB of HBM3 per card, as the driver reports it
    let layout = |stages: usize| model::plan_balanced(&<Ripple as Shardable>::shard_cost(&cfg, b, t), &vec![Shard::ANY_GPU; stages]);
    let mut single_card_window = None;
    for (label, value, window, state, master) in [
        ("bf16 resident, f32 master, f32 state", ResidentValue::Bf16, None, StateFormat::F32, MasterPolicy::F32),
        ("bf16 resident, f32 master, int8 state", ResidentValue::Bf16, None, StateFormat::Int8Block { block: 64 }, MasterPolicy::F32),
        ("fp8 resident, f32 master, int8 state", ResidentValue::Fp8E4M3 { block: 128 }, None, StateFormat::Int8Block { block: 64 }, MasterPolicy::F32),
        ("bf16 window of 2 layers, f32 master, int8 state", ResidentValue::Bf16, Some(2u32), StateFormat::Int8Block { block: 64 }, MasterPolicy::F32),
        ("bf16 window of 2 layers, bf16 master, int8 state", ResidentValue::Bf16, Some(2u32), StateFormat::Int8Block { block: 64 }, MasterPolicy::Bf16),
    ] {
        let resident = ResidentFtConfig { master, state, grad: GradDtype::Bf16, ..resident_cfg(value, window, ClipPass::TwoPass) };
        for stages in [1usize, 2, 4] {
            let costs = stage_costs_of(&cfg, &layout(stages), b, t, &resident);
            let worst = costs.iter().map(|c| c.device()).max().unwrap() as f64 / GIB;
            let host: u64 = costs.iter().map(|c| c.host).sum();
            let c0 = &costs[0];
            eprintln!(
                "{label:52} | {stages} stage(s) | stage 0 layers {:?}: weights {:6.1} grads {:6.1} norms {:4.2} act {:6.1} GiB | worst stage {worst:6.1} of {card:.1} GiB: {} | host {:7.1} GiB",
                c0.layers,
                c0.weights as f64 / GIB,
                c0.grads as f64 / GIB,
                c0.norms as f64 / GIB,
                c0.activations as f64 / GIB,
                if worst <= card { "fits" } else { "does not fit" },
                host as f64 / GIB
            );
            if window.is_some() && stages == 1 && master == MasterPolicy::F32 {
                single_card_window = Some((worst, host as f64 / GIB));
            }
            // Without a window the weights and their gradients are the model: it takes four cards, and only barely.
            if window.is_none() && matches!(value, ResidentValue::Bf16) {
                assert_eq!(worst <= card, stages == 4, "bf16 resident, {stages} stage(s): {worst:.1} of {card:.1} GiB");
            }
        }
    }
    let (device, _) = single_card_window.expect("the window arm ran");
    assert!(device < card, "a window of two layers on one card needs {device:.1} GiB of {card:.1}");
}

fn stage_costs_of(cfg: &TrainConfig, shards: &[Shard], b: u32, t: u32, resident: &ResidentFtConfig) -> Vec<ricursive::train::StageCost> {
    ricursive::train::stage_costs(cfg, shards, b, t, resident).unwrap()
}
