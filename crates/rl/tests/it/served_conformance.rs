// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring a compressed model
// exactly as it is served, for every model family, for its clients. If your
// team needs expertise in model compression and serving then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The generic serving measurement (`model::finetune::served_loss`), for
//! every family that offers a serving engine (`ChatTrainer::serving_engine`):
//! scored through the engine at fp32, a held-out set measures what the
//! family's trainer build computes on it. This holds the generic
//! teacher-forced scoring (`model::serve::teacher_forced_loss` with the
//! default `PagedDecoder::hidden_at`, and an engine's own override) to the
//! model, for each family the same way. Every format a family declares it
//! serves per linear (`ChatTrainer::SERVED_WEIGHTS`) serves a mixed policy
//! (its MLP linears at that format): a finite loss near the fp32 one, and not
//! the fp32 one, so the format was really applied. And structured pruning
//! (`modelopt::PruneFit` over the family's trainer build) writes a narrower
//! checkpoint that the family's trainer and serving engine both measure as the
//! zeroed plan predicted, with every declared linear in the search's manifest.

use std::path::Path;

use gpu_core::select::Dtype;
use model::finetune::{ChatTrainer, ModelFit, OptimizerState};
use model::ops::TierPolicy;
use model::{Batch, FitOpts, FitReport, Model};

const SEQ: usize = 12;

/// A prepared dataset of token records over `vocab`: train and validation
/// splits, each record's second half supervised. `val_is_train` writes the
/// training records as the validation split too.
fn write_prepared(dir: &Path, vocab: u32, seed: u64, val_is_train: bool) {
    std::fs::create_dir_all(dir).expect("the prepared directory");
    let mut rng = data::rng::Rng::new(seed);
    let mut draw = || {
        let mut encoded = data::chat::EncodedSplit::default();
        for _ in 0..4 {
            let ids: Vec<u32> = (0..SEQ).map(|_| rng.gen_range_inclusive(0, vocab as i64 - 1) as u32).collect();
            let mask: Vec<bool> = (0..SEQ).map(|p| p >= SEQ / 2).collect();
            encoded.push(&ids, &mask);
        }
        encoded
    };
    let train = draw();
    let val = if val_is_train { train.clone() } else { draw() };
    data::chat::write_split(dir, "train", &train).expect("write a split");
    data::chat::write_split(dir, "val", &val).expect("write a split");
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(vocab as usize)).expect("write meta");
}

/// Every tensor the model declares prunable is one the precision search can
/// place a tier on: the search's manifest covers the model's linears.
struct ManifestCovers;

impl ModelFit for ManifestCovers {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let sites = modelopt::sites(&model).map_err(std::io::Error::other)?;
        let searched: Vec<&str> = sites.iter().flat_map(|s| &s.tensors).map(|t| t.name.as_str()).collect();
        for (name, _) in model.prunable_dims().iter().flat_map(|d| &d.slices) {
            assert!(searched.contains(&name.as_str()), "{name} is prunable but not in the search's manifest");
        }
        let report = FitReport { initial_loss: 0.0, final_loss: None, steps_completed: 0, resumed_at: None, interrupted: false, evaluations: Vec::new(), kept_best: None, stopped_early: false, applied_steps: 0 };
        Ok((report, model))
    }
}

/// The held-out loss of whichever model the trainer builds, one record a row:
/// the trainer build's own number for the validation split.
struct HeldOut<'a> {
    dir: &'a Path,
    opts: &'a FitOpts,
    loss: Option<f64>,
}

impl ModelFit for HeldOut<'_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (_, val, cfg, _) = model::load_dataset(self.dir, self.opts)?;
        let rows = model::example_rows(&val, &cfg);
        let (mut sum, mut count) = (0f64, 0f64);
        for (tokens, targets) in &rows {
            let n = targets.iter().filter(|&&t| t != model::IGNORE).count() as f64;
            model.set_batch(Batch::Lm { tokens, targets });
            sum += model.forward() as f64 * n;
            count += n;
        }
        self.loss = Some(sum / count);
        let report = FitReport { initial_loss: 0.0, final_loss: None, steps_completed: 0, resumed_at: None, interrupted: false, evaluations: Vec::new(), kept_best: None, stopped_early: false, applied_steps: 0 };
        Ok((report, model))
    }
}

fn served_matches_the_trainer<T: ChatTrainer>(vocab: u32, write_base: impl Fn(&Path)) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    write_base(&base);
    let prepared = dir.path().join("prepared");
    write_prepared(&prepared, vocab, data::rng::random_seed(), false);
    let opts = FitOpts { steps: 1, batch_size: 1, block_size: SEQ as u32, seed: data::rng::random_seed(), ..Default::default() };
    let served = model::finetune::served_loss::<T>(&base, &TierPolicy::uniform(Dtype::F32), None, &prepared, &opts)
        .unwrap_or_else(|| panic!("{} offers a serving engine", T::LORA_BASES.model))
        .unwrap_or_else(|e| panic!("{}: {e}", T::LORA_BASES.model));
    let mut fit = HeldOut { dir: &prepared, opts: &opts, loss: None };
    T::train_full(&base, &opts, OptimizerState::HostOffload, &mut fit).expect("the trainer build");
    let trained = fit.loss.expect("measured");
    assert!((served - trained).abs() <= 1e-3 * trained.abs(), "{}: the serving engine measures {served}, the trainer build {trained}", T::LORA_BASES.model);

    T::train_full(&base, &opts, OptimizerState::HostOffload, &mut ManifestCovers).expect("the manifest");

    // Pruning: half of every prunable dimension, exported and measured back.
    let narrow = dir.path().join("pruned.safetensors");
    let mut prune = modelopt::PruneFit { dir: &prepared, opts: &opts, width: modelopt::PruneWidth::Fraction(0.5), out: &narrow, on_progress: &mut |_| {}, report: None };
    T::train_full(&base, &opts, OptimizerState::HostOffload, &mut prune).unwrap_or_else(|e| panic!("{}: pruning: {e}", T::LORA_BASES.model));
    let report = prune.report.expect("pruned");
    assert!(report.params_after < report.params_before && report.widths.iter().all(|&(_, before, after)| after == (before as f64 * 0.5).round() as usize), "{}: {report:?}", T::LORA_BASES.model);
    let mut reopened = HeldOut { dir: &prepared, opts: &opts, loss: None };
    T::train_full(&narrow, &opts, OptimizerState::HostOffload, &mut reopened).expect("the pruned checkpoint opens");
    let pruned_trained = reopened.loss.expect("measured");
    assert!((pruned_trained - report.pruned_loss).abs() <= 1e-3 * report.pruned_loss.abs(), "{}: the pruned checkpoint measures {pruned_trained}, the zeroed plan {}", T::LORA_BASES.model, report.pruned_loss);
    let pruned_served = model::finetune::served_loss::<T>(&narrow, &TierPolicy::uniform(Dtype::F32), None, &prepared, &opts).expect("a serving engine").expect("served");
    assert!((pruned_served - report.pruned_loss).abs() <= 1e-3 * report.pruned_loss.abs(), "{}: the engine serves the pruned checkpoint at {pruned_served}, the zeroed plan {}", T::LORA_BASES.model, report.pruned_loss);

    let numeric = gpu_core::testgpu::dev(&[]).caps().numeric;
    for &tier in T::SERVED_WEIGHTS.iter().filter(|&&t| t != Dtype::F32) {
        if tier.promote(&numeric) != tier {
            brain_testutil::skip_unavailable(&format!("{}: this device holds {} as {}", T::LORA_BASES.model, tier.name(), tier.promote(&numeric).name()));
            continue;
        }
        let mixed = TierPolicy::uniform(Dtype::F32).with(&["mlp."], tier);
        let loss = model::finetune::served_loss::<T>(&base, &mixed, None, &prepared, &opts).expect("a serving engine").unwrap_or_else(|e| panic!("{} at {}: {e}", T::LORA_BASES.model, mixed.describe()));
        assert!(loss.is_finite() && (loss - served).abs() <= 0.1 * served.abs(), "{} serves {} near fp32: {loss} against {served}", T::LORA_BASES.model, mixed.describe());
        assert_ne!(loss, served, "{}: {} was applied", T::LORA_BASES.model, mixed.describe());
    }
}

/// The served artifact carries the trained behaviour: a LoRA fine-tune of the base, its adapter written to disk by
/// the family's own writer, then read back from that file into the family's serving engine. Through the engine the
/// held-out set (here the training records, so training must move it) measures what the trainer measured on the
/// model the run handed back, and the base alone measures what the trainer measured before the first step; the two
/// are apart, so an adapter that served as a no-op, or one read back wrongly, fails. The same holds for a full
/// fine-tune of the base: its saved checkpoint, opened as the engine's base, serves what its trainer measured.
fn served_artifacts_have_the_trained_loss<T: ChatTrainer>(vocab: u32, write_base: impl Fn(&Path)) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    write_base(&base);
    let prepared = dir.path().join("prepared");
    // The held-out split is the training split: the loss the adapter was trained to lower.
    write_prepared(&prepared, vocab, data::rng::random_seed(), true);
    let opts = FitOpts { steps: 40, batch_size: 1, block_size: SEQ as u32, lr: 1e-2, min_lr: 1e-2, warmup: 0, eval_interval: 0, eval_batches: 0, seed: data::rng::random_seed(), ..Default::default() };
    let f32_base = TierPolicy::uniform(Dtype::F32);
    let lora = model::finetune::LoraRequest { rank: 4, alpha: 8.0, continue_from: None, base: &f32_base, modules_to_save: &[] };
    let mut train = model::finetune::CausalLmFit { dir: &prepared, opts: &opts, out: None, control: Some(model::FitControl::default()) };
    let held_out = [prepared.as_path()];
    let mut scored = model::finetune::ScoredFit::new(&mut train, &held_out, &opts, false);
    let (_, trained) = T::train_lora(&base, &lora, &opts, &mut scored).unwrap_or_else(|e| panic!("{}: {e}", T::LORA_BASES.model));
    let set = scored.into_scored().pop().expect("one held-out set");
    let (base_loss, tuned_loss) = (f64::from(set.base_loss.expect("scored before")), f64::from(set.tuned_loss.expect("scored after")));
    let adapter = dir.path().join("adapter.safetensors");
    let lineage = model::finetune::AdapterLineage { card_id: "served-conformance", base_id: "base", dataset_id: None, provenance: None };
    T::save_adapter(&trained, &adapter, &lineage).unwrap_or_else(|e| panic!("{}: saving the adapter: {e}", T::LORA_BASES.model));
    drop(trained);

    let served = |adapter: Option<&Path>| model::finetune::served_loss::<T>(&base, &f32_base, adapter, &prepared, &opts).expect("a serving engine").unwrap_or_else(|e| panic!("{}: {e}", T::LORA_BASES.model));
    let (served_base, served_tuned) = (served(None), served(Some(&adapter)));
    let model = T::LORA_BASES.model;
    // Twenty times the comparison tolerance below: a served no-op cannot pass for the trained adapter.
    assert!(base_loss - tuned_loss > 0.02 * base_loss, "{model}: training must lower the loss it is served on: {base_loss} -> {tuned_loss}");
    assert!((served_base - base_loss).abs() <= 1e-3 * base_loss.abs(), "{model}: the engine serves the base at {served_base}, the trainer measured {base_loss}");
    assert!((served_tuned - tuned_loss).abs() <= 1e-3 * tuned_loss.abs(), "{model}: the engine serves the saved adapter at {served_tuned}, the trainer measured {tuned_loss} on the model it trained");

    // A full fine-tune's artifact is a checkpoint: opened as the engine's base, it serves what its trainer measured.
    let trained_base = dir.path().join("full.safetensors");
    let mut train = model::finetune::CausalLmFit { dir: &prepared, opts: &opts, out: Some(&trained_base), control: Some(model::FitControl::default()) };
    let mut scored = model::finetune::ScoredFit::new(&mut train, &held_out, &opts, false);
    T::train_full(&base, &opts, OptimizerState::HostOffload, &mut scored).unwrap_or_else(|e| panic!("{model}: full fine-tune: {e}"));
    let full_loss = f64::from(scored.into_scored().pop().expect("one held-out set").tuned_loss.expect("scored after"));
    let served_full = model::finetune::served_loss::<T>(&trained_base, &f32_base, None, &prepared, &opts).expect("a serving engine").unwrap_or_else(|e| panic!("{model}: {e}"));
    assert!(base_loss - full_loss > 0.02 * base_loss, "{model}: the full fine-tune must lower the loss it is served on: {base_loss} -> {full_loss}");
    assert!((served_full - full_loss).abs() <= 1e-3 * full_loss.abs(), "{model}: the engine serves the saved checkpoint at {served_full}, the trainer measured {full_loss} on the model it trained");
}

#[test]
fn qwen3_serves_the_artifacts_it_trains() {
    let cfg = qwen3::QwenConfig { block_size: SEQ as u32, ..qwen3::QwenConfig::tiny_i8() };
    served_artifacts_have_the_trained_loss::<qwen3::finetune::Qwen3Trainer>(cfg.vocab, |path| {
        qwen3::Qwen::new(cfg.clone(), 1, SEQ as u32, &qwen3::init_weights(&cfg, data::rng::random_seed())).save(path.to_str().unwrap());
    });
}

#[test]
fn qwen35_serves_the_artifacts_it_trains() {
    let cfg = qwen35::config::Qwen35Config { block_size: SEQ as u32, ..qwen35::config::Qwen35Config::tiny_i8() };
    served_artifacts_have_the_trained_loss::<qwen35::finetune::Qwen35Trainer>(cfg.vocab, |path| {
        <qwen35::model::Qwen35 as Model>::new(cfg.clone(), 1, SEQ as u32, &qwen35::init::init_weights(&cfg, data::rng::random_seed())).save(path.to_str().unwrap());
    });
}

#[test]
fn qwen35moe_serves_the_artifacts_it_trains() {
    let cfg = qwen35moe::config::Qwen35Config { n_layers: 4, block_size: SEQ as u32, d_model: 32, moe_intermediate_size: 32, shared_expert_intermediate_size: 32, ..qwen35moe::config::Qwen35Config::tiny() };
    served_artifacts_have_the_trained_loss::<qwen35moe::finetune::Qwen35MoeTrainer>(cfg.vocab, |path| {
        let init = <qwen35moe::model::Qwen35 as Model>::init_weights(&cfg, data::rng::random_seed());
        <qwen35moe::model::Qwen35 as Model>::new(cfg.clone(), 1, SEQ as u32, &init).save(path.to_str().unwrap());
    });
}

/// `init` with its MLP weights scaled up: at the initializers' small scale the
/// MLP adds too little for a weight format to move the loss, and the check
/// that each declared format is applied needs it to.
fn loud_mlp(mut init: std::collections::HashMap<String, Vec<f32>>) -> std::collections::HashMap<String, Vec<f32>> {
    for (_, w) in init.iter_mut().filter(|(n, _)| n.contains("mlp.")) {
        w.iter_mut().for_each(|v| *v *= 25.0);
    }
    init
}

#[test]
fn qwen3_serves_what_it_trains() {
    let cfg = qwen3::QwenConfig { block_size: SEQ as u32, ..qwen3::QwenConfig::tiny_i8() };
    served_matches_the_trainer::<qwen3::finetune::Qwen3Trainer>(cfg.vocab, |path| {
        qwen3::Qwen::new(cfg.clone(), 1, SEQ as u32, &loud_mlp(qwen3::init_weights(&cfg, data::rng::random_seed()))).save(path.to_str().unwrap());
    });
}

/// The Qwen2/Llama config variant the registry serves with this crate:
/// attention biases and no QK-norm. The biases are drawn large enough that a
/// path dropping them would move the loss past the comparison's tolerance.
#[test]
fn qwen2_variant_serves_what_it_trains() {
    let cfg = qwen3::QwenConfig { block_size: SEQ as u32, qk_norm: false, attn_bias: true, ..qwen3::QwenConfig::tiny_i8() };
    served_matches_the_trainer::<qwen3::finetune::Qwen3Trainer>(cfg.vocab, |path| {
        let mut init = loud_mlp(qwen3::init_weights(&cfg, data::rng::random_seed()));
        let mut biases = 0;
        for (_, b) in init.iter_mut().filter(|(n, _)| n.ends_with(".bias")) {
            b.iter_mut().for_each(|v| *v *= 50.0);
            biases += 1;
        }
        assert!(biases > 0, "the variant carries attention biases");
        qwen3::Qwen::new(cfg.clone(), 1, SEQ as u32, &init).save(path.to_str().unwrap());
    });
}

#[test]
fn qwen35_serves_what_it_trains() {
    let cfg = qwen35::config::Qwen35Config { block_size: SEQ as u32, ..qwen35::config::Qwen35Config::tiny_i8() };
    served_matches_the_trainer::<qwen35::finetune::Qwen35Trainer>(cfg.vocab, |path| {
        <qwen35::model::Qwen35 as Model>::new(cfg.clone(), 1, SEQ as u32, &loud_mlp(qwen35::init::init_weights(&cfg, data::rng::random_seed()))).save(path.to_str().unwrap());
    });
}

#[test]
fn qwen35moe_serves_what_it_trains() {
    // Widths a whole number of int8 groups (32), so the int8 the engine serves can hold every expert.
    let cfg = qwen35moe::config::Qwen35Config { n_layers: 4, block_size: SEQ as u32, d_model: 32, moe_intermediate_size: 32, shared_expert_intermediate_size: 32, ..qwen35moe::config::Qwen35Config::tiny() };
    served_matches_the_trainer::<qwen35moe::finetune::Qwen35MoeTrainer>(cfg.vocab, |path| {
        let init = loud_mlp(<qwen35moe::model::Qwen35 as Model>::init_weights(&cfg, data::rng::random_seed()));
        <qwen35moe::model::Qwen35 as Model>::new(cfg.clone(), 1, SEQ as u32, &init).save(path.to_str().unwrap());
    });
}

/// Below what scoring one row of the tiny model through quantized linears
/// would drop if every product allocated its own decoded tile (72 KiB), above
/// what it drops with the tiles reused.
const RECLAIM_CEILING: u64 = 1 << 16;

/// The measurements `brain optimize` makes in turn, the original and then
/// quantized formats, each scoring every held-out row through an engine. Run
/// only in a child process (the gate below), so the lowered ceiling reaches no
/// other test.
#[test]
#[ignore = "child-process helper, driven by the test below under a lowered reclaim ceiling"]
fn repeated_measurements_under_a_lowered_ceiling_helper() {
    gpu_core::hardware::set_reclaim_ceiling(Some(RECLAIM_CEILING));
    let cfg = qwen3::QwenConfig { block_size: SEQ as u32, ..qwen3::QwenConfig::tiny_i8() };
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    qwen3::Qwen::new(cfg.clone(), 1, SEQ as u32, &qwen3::init_weights(&cfg, 7)).save(base.to_str().unwrap());
    let prepared = dir.path().join("prepared");
    write_prepared(&prepared, cfg.vocab, 7, false);
    let opts = FitOpts { steps: 1, batch_size: 1, block_size: SEQ as u32, seed: 7, ..Default::default() };
    // The order `brain optimize` measures in: the original, then each baseline format, then the original again.
    let policies = [Dtype::F32, Dtype::NF4, Dtype::I8, Dtype::F32].map(|t| TierPolicy::uniform(Dtype::F32).with(&["blocks."], t));
    let losses: Vec<f64> = policies.iter().map(|p| model::finetune::served_loss::<qwen3::finetune::Qwen3Trainer>(&base, p, None, &prepared, &opts).expect("an engine").unwrap_or_else(|e| panic!("{}: {e}", p.describe()))).collect();
    assert_eq!(losses[0], losses[3], "the same measurement repeated after others: {losses:?}");
}

/// Scoring through quantized linears reuses its decoded tiles: a per-product
/// tile made one recorded forward of a 1.3B model at nf4 hold its f32 size in
/// temporaries, dropped past the device's 4 GB reclaim ceiling.
#[test]
fn scoring_quantized_linears_reuses_its_tiles() {
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("served_conformance: needs a real device whose reclaim is deferred");
        return;
    }
    let helper = "repeated_measurements_under_a_lowered_ceiling_helper";
    let out = std::process::Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"])
        .output()
        .expect("spawn the helper");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{helper} must complete under a {RECLAIM_CEILING}-byte reclaim ceiling; output:\n{text}");
}
