// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models in bf16 and FP8 for its
// clients. If your team needs expertise in resuming and exporting low-precision training runs exactly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A full fine-tune of a tiny random Qwen3 whose linears are RESIDENT in bf16 or FP8 on the device, trained
//! against a host-side master copy through the generic fit loop:
//!
//! * it learns what the fp32 full fine-tune learns;
//! * a run stopped half way and resumed from its state file (the master copies, moments and step are committed
//!   transactionally beside it) ends bit for bit where the uninterrupted run ends;
//! * the model exported in the resident format reloads and computes the loss the trained model computes;
//! * a plan that does not fit is refused before the model is built.

use std::path::{Path, PathBuf};

use checkpoint::packed::QuantEncoding;
use data::binio::{self, Meta};
use model::memplan::Capacity;
use model::resident_ft::{ResidentFtConfig, ResidentValue};
use model::{Model, Shard};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen3::{Qwen, QwenConfig, ResidentPlan};

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen3-resident-{name}-{}", std::process::id()));
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

fn opts(steps: u32) -> model::FitOpts {
    // seed: the 2e-3 fp8 export-reload tolerance and the 0.5x loss-fall margin were tuned on this draw
    model::FitOpts { steps, batch_size: 1, block_size: 16, lr: 1e-2, min_lr: 1e-3, warmup: 2, decay_iters: steps, weight_decay: 0.0, grad_clip: 1.0, eval_interval: 0, eval_batches: 0, seed: 3, ..Default::default() }
}

fn objective<M: Model>(dir: &Path, opts: &model::FitOpts) -> impl model::Objective<M> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    model::causal_lm::<M>(train, val, bcfg, itos)
}

fn resident(value: ResidentValue, state: StateFormat) -> ResidentFtConfig {
    ResidentFtConfig {
        value,
        master: MasterPolicy::F32,
        state,
        rule: Rule::AdamW { adam: model::Adam::default(), wd: 0.0 },
        grad: GradDtype::F32,
        stochastic_state: false,
        // seed: the rounding stream the fp8 export-reload tolerance was tuned on
        seed: Some(5),
        tile_elems: 256,
        state_dir: None,
        defer_publication: false,
        window: None,
        clip_pass: model::resident_ft::ClipPass::TwoPass,
    }
}

fn plan(value: ResidentValue, state: StateFormat) -> ResidentPlan {
    ResidentPlan { cfg: resident(value, state), capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: Vec::new() }
}

fn build(value: ResidentValue, state: StateFormat) -> Qwen {
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    Qwen::new_resident(cfg, 1, 16, &init, &plan(value, state), gpu_core::provider::library::GemmPrecision::Fp32Reference).unwrap()
}

fn linears(m: &Qwen) -> Vec<(String, Vec<f32>)> {
    let mut names: Vec<String> = m.param_names().into_iter().filter(|n| n.starts_with("blocks.") && n.ends_with(".weight") && !n.contains("norm") && !n.contains("ln")).collect();
    names.sort();
    names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
}

/// Spec: the resident low-precision full fine-tune learns what the fp32 one learns.
#[test]
fn resident_full_finetunes_learn_what_the_fp32_one_does() {
    let scratch = tmp("learn");
    write_dataset(&scratch);
    let opts = opts(40);
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    let fp32 = Qwen::new_shard(cfg.clone(), 1, 16, &init, true, Shard::whole(cfg.n_layers as usize));
    let (base, _) = model::fit_controlled(fp32, objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
    assert!(base.final_loss.unwrap() < 0.5 * base.initial_loss, "the fp32 reference learns: {base:?}");
    for (label, value, state) in [
        ("bf16, f32 master, f32 moments", ResidentValue::Bf16, StateFormat::F32),
        ("bf16, f32 master, int8 moments", ResidentValue::Bf16, StateFormat::Int8Block { block: 64 }),
        ("fp8 e4m3, f32 master, f32 moments", ResidentValue::Fp8E4M3 { block: 128 }, StateFormat::F32),
        ("QAT int8, f32 master, f32 moments", ResidentValue::Int8, StateFormat::F32),
        ("QAT nf4, f32 master, f32 moments", ResidentValue::Nf4, StateFormat::F32),
    ] {
        let (report, _) = model::fit_controlled(build(value, state), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
        eprintln!("{label}: loss {:?} -> {:?} (fp32 {:?} -> {:?})", report.initial_loss, report.final_loss.unwrap(), base.initial_loss, base.final_loss.unwrap());
        assert!(report.final_loss.unwrap() < 0.5 * report.initial_loss, "{label}: {report:?}");
        assert!(report.final_loss.unwrap() < base.final_loss.unwrap() + 0.25 * base.initial_loss, "{label}: {report:?} vs fp32 {base:?}");
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: a run stopped at step 4 and resumed from its state file in a fresh model ends exactly where the
/// uninterrupted run ends: master copies, moments and step come back from the transactional commit.
#[test]
fn a_resumed_resident_run_ends_exactly_where_the_uninterrupted_run_does() {
    let scratch = tmp("resume");
    write_dataset(&scratch);
    let opts = opts(8);
    let identity = serde_json::json!({"test": "resident-resume"});
    for value in [ResidentValue::Bf16, ResidentValue::Fp8E4M3 { block: 128 }, ResidentValue::Int8] {
        let state_format = StateFormat::Int8Block { block: 64 };
        let (straight_report, straight) = model::fit_controlled(build(value, state_format), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();

        let state = scratch.join(format!("{value:?}.state"));
        let mut stop_at_4 = |s: &model::StepReport| s.step < 4;
        let control = model::FitControl { on_step: Some(&mut stop_at_4), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
        let (first, _) = model::fit_controlled(build(value, state_format), objective::<Qwen>(&scratch, &opts), &opts, None, control).unwrap();
        assert!(first.interrupted && state.is_file(), "stopped at step 4 with a state file");
        assert!(state.with_extension("offdevice").is_dir(), "the master copies are committed beside it");

        let control = model::FitControl { state: Some(&state), identity: identity.clone(), ..Default::default() };
        let (second, resumed) = model::fit_controlled(build(value, state_format), objective::<Qwen>(&scratch, &opts), &opts, None, control).unwrap();
        assert_eq!(second.resumed_at, Some(4));
        assert_eq!(second.final_loss.unwrap().to_bits(), straight_report.final_loss.unwrap().to_bits(), "{value:?}: the last step's loss is identical");
        for ((name, a), (_, b)) in linears(&straight).iter().zip(linears(&resumed)) {
            assert_eq!(a, &b, "{value:?}: master of {name}");
        }
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: the export is a checkpoint the loaders open. Exported in the format the model trains in, the reloaded
/// model computes the loss the trained one computes (to the arithmetic of the two forward paths); exported to a
/// coarser format than it trained in (bf16 to FP8) it computes the loss of that quantization, close to the
/// trained one.
#[test]
fn the_exported_model_reloads_and_computes_the_trained_loss() {
    let scratch = tmp("export");
    write_dataset(&scratch);
    let opts = opts(12);
    for (label, value, enc, tol) in [
        ("bf16 to bf16", ResidentValue::Bf16, QuantEncoding::bf16(), 2e-3f32),
        ("fp8 to fp8", ResidentValue::Fp8E4M3 { block: 128 }, QuantEncoding::fp8_e4m3_tiled(128), 2e-3),
        ("bf16 to fp8", ResidentValue::Bf16, QuantEncoding::fp8_e4m3_tiled(128), 0.1),
    ] {
        let (_, trained) = model::fit_controlled(build(value, StateFormat::F32), objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
        let x: Vec<u32> = (0..4).flat_map(|_| PROMPT.into_iter().chain([TARGET])).collect();
        let y: Vec<u32> = x.iter().map(|t| (t + 1) % VOCAB).collect();
        trained.set_batch(&x, &y);
        let want = Model::forward(&trained);
        let path = scratch.join(format!("{}.safetensors", label.replace(' ', "-")));
        let report = trained.export_resident(&path, &enc).unwrap();
        assert!(report.linears.staging_bytes < 1 << 20, "{label}: export staging is a tile: {report:?}");
        let (cfg, src) = qwen3::open_checkpoint(path.to_str().unwrap()).unwrap();
        let reloaded = Qwen::new_shard(cfg.clone(), 1, 16, &*src, false, Shard::whole(cfg.n_layers as usize));
        reloaded.set_batch(&x, &y);
        let got = Model::forward(&reloaded);
        eprintln!("{label}: trained loss {want}, reloaded loss {got}");
        assert!((got - want).abs() <= tol * want.abs().max(1.0), "{label}: reloaded {got} vs trained {want}");
    }
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: updating the tensors in backward order into a `DeferredPublish` store and publishing them together gives the
/// run exactly the weights the direct update gives (the resident state is the consumer of both).
#[test]
fn a_run_with_deferred_publication_ends_exactly_where_the_direct_run_does() {
    let scratch = tmp("defer");
    write_dataset(&scratch);
    let opts = opts(8);
    let run = |defer: bool| {
        let cfg = cfg();
        let init = qwen3::init_weights(&cfg, 7);
        let mut plan = plan(ResidentValue::Fp8E4M3 { block: 128 }, StateFormat::Int8Block { block: 64 });
        plan.cfg.defer_publication = defer;
        let m = Qwen::new_resident(cfg, 1, 16, &init, &plan, gpu_core::provider::library::GemmPrecision::Fp32Reference).unwrap();
        let (_, m) = model::fit_controlled(m, objective::<Qwen>(&scratch, &opts), &opts, None, model::FitControl::default()).unwrap();
        linears(&m)
    };
    assert_eq!(run(false), run(true));
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Spec: a plan the machine cannot hold is refused before the model is built, naming the category.
#[test]
fn a_resident_plan_that_does_not_fit_is_refused_before_building() {
    let cfg = cfg();
    let init = qwen3::init_weights(&cfg, 7);
    let mut p = plan(ResidentValue::Bf16, StateFormat::F32);
    p.capacity = Capacity { device_bytes: 1000, host_bytes: 1 << 30 };
    let e = Qwen::new_resident(cfg.clone(), 1, 16, &init, &p, gpu_core::provider::library::GemmPrecision::Fp32Reference).err().expect("refused");
    assert!(e.contains("does not fit") && e.contains("Weights"), "{e}");
}
