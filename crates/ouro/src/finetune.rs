// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training recurrent-depth
// language models from scratch and fine-tuning them on chat data for its
// clients. If your team needs expertise in looped decoders trained through one
// generic training stack then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Ouro as a [`model::finetune::ChatTrainer`]: the generic full-parameter fine-tuner, the raw-text pretraining
//! objective and the from-scratch initialization run on it, all through the one loop ([`model::fit_controlled`]),
//! so a run stopped at a step boundary resumes exactly from its saved state.
//!
//! The base is a brain checkpoint (what [`crate::model::Ouro`] saves and [`ChatTrainer::init_checkpoint`] writes:
//! the tensors under brain's names, the configuration in the header, a card of family `ouro`) or a Hugging Face
//! directory. Every weight trains in f32 with f32 moments; there is no LoRA trainer for the looped stack yet, so an
//! adapter run is refused by name.

use std::path::{Path, PathBuf};

use checkpoint::TensorSource;
use model::finetune::{AdapterLineage, ChatTrainer, LoraRequest, ModelFit, OptimizerState};
use model::{FitOpts, FitReport};

use crate::model::Ouro;
use crate::OuroConfig;

/// The card family a brain Ouro checkpoint carries.
pub const FAMILY: &str = "ouro";

/// The exit threshold of a from-scratch model whose configuration names none: the pass at which half the exit mass
/// is reached (see [`ChatTrainer::init_checkpoint`] below). On a small model pretrained from scratch and chat
/// fine-tuned, the fine-tune moved nearly all the exit mass to the first pass, which it trained, and the final pass
/// it barely weighed got worse (knowledge 496 has the measurement).
pub const FROM_SCRATCH_EXIT_THRESHOLD: f32 = 0.5;

/// The generic trainers' entry for Ouro.
pub struct OuroTrainer;

fn invalid(path: &Path, why: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{}: {why}", path.display()))
}

fn unsupported(why: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, format!("ouro: {why}"))
}

/// The configuration and tensors of the checkpoint at `path`: a Hugging Face directory or a brain checkpoint file.
pub fn open(path: &Path) -> Result<(OuroConfig, Box<dyn TensorSource>), String> {
    if path.is_dir() {
        let (cfg, src) = crate::import::open_dir(path)?;
        return Ok((cfg, Box::new(src)));
    }
    let name = path.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))?;
    let reader = checkpoint::weightio::WeightReader::open(name).map_err(|e| format!("{}: {e}", path.display()))?;
    let cfg = OuroConfig::from_json(&reader.config()).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((cfg, Box::new(reader)))
}

impl ChatTrainer for OuroTrainer {
    type Trained = Ouro;
    /// No LoRA trainer: [`Self::train_lora`] refuses, so no frozen-base format is offered.
    const LORA_BASES: model::policy::TrainBases = model::policy::TrainBases { model: "ouro", bases: &[] };
    /// f32 weights and f32 moments on the device; a resident or streamed state is not wired through this entry.
    const TRAIN_PRECISION: model::TrainPrecisionSupport = model::TrainPrecisionSupport { model: "ouro", state: model::StateSupport::Fp32, resident: false, layer_streamed: false, f32_streamed: false };

    fn checkpoint_path(weights: &Path, model_dir: &Path) -> PathBuf {
        if weights.file_name().is_some_and(|n| n == "config.json") {
            model_dir.to_path_buf()
        } else {
            weights.to_path_buf()
        }
    }

    fn vocab(open: &Path) -> Result<usize, String> {
        self::open(open).map(|(cfg, _)| cfg.vocab as usize)
    }

    fn train_full(base: &Path, opts: &FitOpts, state: OptimizerState, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Ouro)> {
        Self::TRAIN_PRECISION.check(&opts.precision).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))?;
        let (cfg, src) = open(base).map_err(|e| invalid(base, e))?;
        if opts.block_size > cfg.block_size {
            return Err(invalid(base, format!("training rows of {} tokens are longer than the model's {} positions", opts.block_size, cfg.block_size)));
        }
        let m = state.scope(|| Ouro::new_train_with(cfg, opts.batch_size, opts.block_size, &*src, opts.gemm_precision));
        fit.fit(m)
    }

    fn train_lora(_base: &Path, _lora: &LoraRequest<'_>, _opts: &FitOpts, _fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Ouro)> {
        Err(unsupported("there is no LoRA trainer for the looped stack; train every weight (a full fine-tune)"))
    }

    fn save_adapter(_trained: &Ouro, _path: &Path, _lineage: &AdapterLineage<'_>) -> std::io::Result<()> {
        Err(unsupported("there is no LoRA adapter to save"))
    }

    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>, keep_going: &mut dyn FnMut() -> bool) -> Result<(), String> {
        model::finetune::export_brain_checkpoint(FAMILY, trained, tokenizer_dir, out, adapter, keep_going)
    }

    const FROM_SCRATCH: bool = true;

    /// The exit gates choose the depth; the trainers weigh their entropy into the objective
    /// ([`model::Model::set_exit_entropy`]).
    const EXIT_GATES: bool = true;

    fn init_params(config: &serde_json::Value) -> Result<u64, String> {
        let cfg = OuroConfig::from_json(config)?;
        Ok(cfg.param_list().iter().map(|(_, n)| *n as u64).sum())
    }

    /// The tensors of [`crate::init::init_weights`] at `seed`, written as the brain checkpoint `model.safetensors`
    /// the model itself saves (same names, shapes, header and card), so training opens it like any trained one.
    ///
    /// A configuration that names no `early_exit_threshold` is written with [`FROM_SCRATCH_EXIT_THRESHOLD`]: the
    /// training objective weighs each pass's loss by its exit probability, and a model trained from scratch moves
    /// that mass to the pass it learns first, so the final pass (the released checkpoints' 1.0) is the one it trains
    /// least; serving the median exit serves the pass it trains.
    fn init_checkpoint(config: &serde_json::Value, seed: u64, out_dir: &Path) -> Result<(), String> {
        let mut cfg = OuroConfig::from_json(config)?;
        if config.get("early_exit_threshold").is_none() {
            cfg.early_exit_threshold = FROM_SCRATCH_EXIT_THRESHOLD;
        }
        let mut weights = crate::init::init_weights(&cfg, seed);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, numel)| {
                let w = weights.remove(&name).ok_or_else(|| format!("ouro init: no tensor {name}"))?;
                Ok((name, vec![numel as u64], w))
            })
            .collect::<Result<_, String>>()?;
        let path = out_dir.join(model::finetune::BRAIN_WEIGHTS_FILE);
        let name = path.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))?;
        let card = checkpoint::st::ModelCard::new("brain/ouro", FAMILY);
        checkpoint::st::save_safetensors(name, &tensors, &cfg.to_json(), Some(&card)).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// There is no LoRA trainer for the looped stack: an adapter run is refused by name, before any base is read.
    #[test]
    fn an_adapter_run_is_refused_by_name() {
        let policy = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
        let lora = LoraRequest { rank: 2, alpha: 4.0, continue_from: None, base: &policy, modules_to_save: &[] };
        struct Never;
        impl ModelFit for Never {
            fn fit<M: model::Model>(&mut self, _: M) -> std::io::Result<(FitReport, M)> {
                unreachable!("nothing is trained")
            }
        }
        let e = OuroTrainer::train_lora(Path::new("/nonexistent"), &lora, &FitOpts::default(), &mut Never).err().expect("refused");
        assert_eq!(e.kind(), std::io::ErrorKind::Unsupported);
        assert!(e.to_string().contains("ouro") && e.to_string().contains("LoRA"), "{e}");
    }

    /// A from-scratch model is served at the pass its training objective weighs: the exit rule reads the median of
    /// the exit distribution unless the configuration names a threshold, so a model whose gates put most of the exit
    /// mass on an early pass serves that pass, not a final one the objective barely trained.
    #[test]
    fn a_from_scratch_model_serves_the_pass_its_objective_weighs() {
        let dir = std::env::temp_dir().join(format!("ouro-init-exit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = OuroConfig::tiny().to_json();
        cfg.as_object_mut().unwrap().remove("early_exit_threshold");
        OuroTrainer::init_checkpoint(&cfg, 5, &dir).unwrap();
        let (opened, _) = open(&dir.join(model::finetune::BRAIN_WEIGHTS_FILE)).unwrap();
        // Gates putting 0.98 of the exit mass on the first of two passes.
        let gates = [(0.98f64 / 0.02).ln(), 0.0];
        assert_eq!(crate::model::exit_pass(&gates, opened.early_exit_threshold as f64), 0, "threshold {}", opened.early_exit_threshold);
        // A threshold the configuration names is kept.
        cfg["early_exit_threshold"] = serde_json::json!(1.0);
        OuroTrainer::init_checkpoint(&cfg, 5, &dir).unwrap();
        assert_eq!(open(&dir.join(model::finetune::BRAIN_WEIGHTS_FILE)).unwrap().0.early_exit_threshold, 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A from-scratch checkpoint opens as the configuration it was asked for, and the same seed writes the same bytes.
    #[test]
    fn an_initialized_checkpoint_opens_as_its_config_and_is_deterministic_in_its_seed() {
        let dir = std::env::temp_dir().join(format!("ouro-init-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        for d in [&a, &b] {
            std::fs::create_dir_all(d).unwrap();
        }
        let cfg = OuroConfig::tiny().to_json();
        OuroTrainer::init_checkpoint(&cfg, 5, &a).unwrap();
        OuroTrainer::init_checkpoint(&cfg, 5, &b).unwrap();
        let read = |d: &Path| std::fs::read(d.join(model::finetune::BRAIN_WEIGHTS_FILE)).unwrap();
        assert_eq!(read(&a), read(&b), "the same seed writes the same checkpoint");
        let (opened, _) = open(&a.join(model::finetune::BRAIN_WEIGHTS_FILE)).unwrap();
        assert_eq!(opened, OuroConfig::tiny());
        assert_eq!(OuroTrainer::init_params(&cfg).unwrap(), OuroConfig::tiny().param_list().iter().map(|(_, n)| *n as u64).sum::<u64>());
        assert!(OuroTrainer::init_params(&serde_json::json!({"vocab_size": 3})).unwrap_err().contains("missing keys"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
