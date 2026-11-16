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
        let m = state.scope(|| Ouro::new_train(cfg, opts.batch_size, opts.block_size, &*src));
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
}
