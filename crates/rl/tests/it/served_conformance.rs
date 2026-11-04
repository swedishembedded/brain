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
/// splits, each record's second half supervised.
fn write_prepared(dir: &Path, vocab: u32, seed: u64) {
    std::fs::create_dir_all(dir).expect("the prepared directory");
    let mut rng = data::rng::Rng::new(seed);
    for split in ["train", "val"] {
        let mut encoded = data::chat::EncodedSplit::default();
        for _ in 0..4 {
            let ids: Vec<u32> = (0..SEQ).map(|_| rng.gen_range_inclusive(0, vocab as i64 - 1) as u32).collect();
            let mask: Vec<bool> = (0..SEQ).map(|p| p >= SEQ / 2).collect();
            encoded.push(&ids, &mask);
        }
        data::chat::write_split(dir, split, &encoded).expect("write a split");
    }
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
        let report = FitReport { initial_loss: 0.0, final_loss: None, steps_completed: 0, resumed_at: None, interrupted: false, evaluations: Vec::new(), kept_best: None, stopped_early: false };
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
        let report = FitReport { initial_loss: 0.0, final_loss: None, steps_completed: 0, resumed_at: None, interrupted: false, evaluations: Vec::new(), kept_best: None, stopped_early: false };
        Ok((report, model))
    }
}

fn served_matches_the_trainer<T: ChatTrainer>(vocab: u32, write_base: impl Fn(&Path)) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    write_base(&base);
    let prepared = dir.path().join("prepared");
    write_prepared(&prepared, vocab, data::rng::random_seed());
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
