// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ripple as a [`model::finetune::ChatTrainer`]: the generic full-parameter, LoRA and preference fine-tuners run on it.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, from full-weight training to adapters over a 4-bit base. If your team needs expertise in
//! training hybrid-attention language models then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The base is either a brain-format file (what a full fine-tune writes: the tensors under the release's own
//! names and a header carrying the [`TrainConfig`]) or the release itself, a directory (or single file beside a
//! `config.json`) in the published layout, read in whatever encoding it stores (bf16, FP8, the ModelOpt NVFP4
//! release). Tensors are read one at a time into the device; the model never exists as f32 on the host.
//! A full fine-tune trains f32 weights, so it needs the base in a float encoding; an adapter trains over the frozen
//! base held at any tier [`ChatTrainer::LORA_BASES`] lists (NVFP4 only as the ModelOpt release stores it).

use std::path::{Path, PathBuf};

use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use model::adapter::device::LoraCfg;
use model::finetune::{AdapterLineage, ChatTrainer, LoraRequest, ModelFit, OptimizerState};
use model::{FitOpts, FitReport};

use crate::config::RicursiveConfig;
use crate::import::open_checkpoint;
use crate::train::{default_targets, AdapterInit};
use crate::{Ripple, TrainConfig, TrainOpts};

/// The generic trainers' entry for Ripple.
pub struct RippleTrainer;

/// A base opened for training: the model's configuration and its tensors in canonical names.
struct Base {
    model: RicursiveConfig,
    source: Box<dyn TensorSource>,
}

fn invalid(path: &Path, why: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{}: {why}", path.display()))
}

fn path_str(path: &Path) -> std::io::Result<&str> {
    path.to_str().ok_or_else(|| invalid(path, "not a UTF-8 path"))
}

impl Base {
    fn open(path: &Path) -> std::io::Result<Base> {
        let name = path_str(path)?;
        if !path.is_dir() {
            if let Ok(reader) = checkpoint::weightio::WeightReader::open(name) {
                if reader.config().get("model").is_some() {
                    let cfg = TrainConfig::from_json(&reader.config()).map_err(|e| invalid(path, e))?;
                    return Ok(Base { model: cfg.model, source: Box::new(reader) });
                }
            }
        }
        // The release layout: `config.json` in the directory, or beside the single file.
        let dir = if path.is_dir() { path } else { path.parent().ok_or_else(|| invalid(path, "has no directory"))? };
        let config = dir.join("config.json");
        let model = RicursiveConfig::from_file(path_str(&config)?).map_err(|e| invalid(&config, e))?;
        let layers: Vec<usize> = (0..model.num_hidden_layers).collect();
        let source = open_checkpoint(name, &model, &layers).map_err(|e| invalid(path, e))?;
        Ok(Base { model, source: Box::new(source) })
    }

    /// The training configuration for rows of `block` tokens.
    fn train_config(&self, block: u32) -> TrainConfig {
        TrainConfig::new(self.model.clone(), block)
    }
}

/// `(parameters, layers, model width)` of the base at `path`, what a tier choice sizes a base by.
pub fn base_shape(path: &Path) -> Result<(u64, u64, u64), String> {
    let base = Base::open(path).map_err(|e| e.to_string())?;
    let cfg = base.train_config(1);
    Ok((cfg.param_list().iter().map(|(_, n)| *n as u64).sum(), cfg.model.num_hidden_layers as u64, cfg.model.hidden_size as u64))
}

fn built(result: Result<Ripple, String>, path: &Path) -> std::io::Result<Ripple> {
    result.map_err(|e| invalid(path, e))
}

impl ChatTrainer for RippleTrainer {
    type Trained = Ripple;
    const LORA_BASES: model::policy::TrainBases = model::policy::TrainBases { model: "ricursive", bases: &[Dtype::F32, Dtype::BF16, Dtype::I8, Dtype::NF4, Dtype::F8E4M3, Dtype::F4E2M1] };
    /// The optimizer is the device AdamW over the parameter store with f32 moments; every other choice is refused by
    /// name.
    const TRAIN_PRECISION: model::TrainPrecisionSupport = model::TrainPrecisionSupport { model: "ricursive", state: model::StateSupport::Fp32, resident: false, layer_streamed: false, f32_streamed: false };

    fn checkpoint_path(weights: &Path, model_dir: &Path) -> PathBuf {
        if weights.file_name().is_some_and(|n| n == "config.json") {
            model_dir.to_path_buf()
        } else {
            weights.to_path_buf()
        }
    }

    fn vocab(open: &Path) -> Result<usize, String> {
        Base::open(open).map(|b| b.model.vocab_size).map_err(|e| e.to_string())
    }

    fn train_full(base: &Path, opts: &FitOpts, state: OptimizerState, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Ripple)> {
        Self::TRAIN_PRECISION.check(&opts.precision).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))?;
        let opened = Base::open(base)?;
        let cfg = opened.train_config(opts.block_size);
        let train_opts = TrainOpts { gemm: opts.gemm_precision, ..TrainOpts::default() };
        let model = state.scope(|| built(Ripple::build(cfg, opts.batch_size, opts.block_size, &*opened.source, &train_opts, true), base))?;
        fit.fit(model)
    }

    fn train_lora(base: &Path, lora: &LoraRequest<'_>, opts: &FitOpts, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Ripple)> {
        let asked = lora.checked::<Self>()?.clone();
        let opened = Base::open(base)?;
        // NVFP4 is held as the release published it (nothing quantizes into it): the base must store it.
        let tiers = if asked.dtypes().any(|t| t == Dtype::F4E2M1) {
            let stored = crate::weights::resolve_tiers(&opened.model, &*opened.source, None);
            if !stored.dtypes().any(|t| t == Dtype::F4E2M1) {
                return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, format!("{}: a base format of nvfp4 holds the NVFP4 release as published, and this checkpoint stores {}; nothing quantizes into NVFP4", base.display(), stored.describe())));
            }
            stored
        } else {
            asked
        };
        let cfg = opened.train_config(opts.block_size).with_lora(LoraCfg::new(lora.rank, lora.alpha, default_targets()));
        let init = match lora.continue_from {
            Some(path) => {
                let adapter = model::lora::device_adapter::read_adapter(path_str(path)?)?;
                AdapterInit::continued(&cfg, &*opened.source, &adapter).map_err(std::io::Error::other)?
            }
            None => AdapterInit::fresh(&cfg, &*opened.source, opts.seed),
        };
        let train_opts = TrainOpts { tiers, gemm: opts.gemm_precision, ..TrainOpts::default() };
        let model = OptimizerState::Device.scope(|| built(Ripple::build(cfg, opts.batch_size, opts.block_size, &init, &train_opts, true), base))?;
        fit.fit(model)
    }

    fn save_adapter(trained: &Ripple, path: &Path, lineage: &AdapterLineage<'_>) -> std::io::Result<()> {
        trained.save_adapter(path_str(path)?, lineage.card_id, lineage.base_id, lineage.dataset_id, lineage.provenance.clone())
    }

    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>) -> Result<(), String> {
        model::finetune::export_brain_checkpoint("ricursive", trained, tokenizer_dir, out, adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: a train precision outside the f32 run is refused by name through the generic entry point, before the
    /// base is opened.
    #[test]
    fn a_train_precision_it_does_not_run_is_refused_by_name_before_anything_is_opened() {
        let nowhere = Path::new("/nonexistent/ricursive-base");
        for spec in ["weights=fp8", "weights=bf16,window=2"] {
            let opts = FitOpts { precision: model::TrainPrecision::parse(spec).unwrap(), ..FitOpts::default() };
            let e = model::finetune::fit_full::<RippleTrainer>(nowhere, nowhere, &opts, nowhere, OptimizerState::HostOffload, model::FitControl::default()).expect_err(spec);
            assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{spec}: {e}");
            assert!(e.to_string().contains("ricursive"), "{spec}: {e}");
        }
    }
}
