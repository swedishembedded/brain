// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of looped mixture-of-experts models on their published quantized weights
// for its clients. If your team needs expertise in LoRA, preference or reinforcement fine-tuning of such a model then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Inkling 2 as a [`ChatTrainer`]: the generic supervised, preference (DPO) and policy-gradient (GRPO) fine-tuners run
//! on [`Inkling2Train`] through one loop.
//!
//! - LoRA ([`ChatTrainer::train_lora`]) over the release as published: the expert banks stay in their NVFP4 (layers
//!   0-19) and MXFP8 (20-23) encoding, read in place and never dequantized (QLoRA); the dense linears are held at the
//!   run's base tier, bf16 as published or f32. The adapter covers every site of [`crate::train::sites`] (attention,
//!   relative states, sinks, depth and message queries, message gates, stacked experts).
//! - Full fine-tuning ([`ChatTrainer::train_full`]) for a checkpoint whose experts are stored as floats (a small
//!   configuration, or one trained from scratch): every tensor in f32 with f32 moments on the device. The released
//!   26B is refused by name: its full fine-tune needs a streamed window, ZeRO sharding or an expert-parallel layout.
//!
//! A checkpoint is a Hugging Face directory in the published names (`config.json`, the safetensors and, for a
//! quantized release, `hf_quant_config.json`); a full fine-tune saves that same layout ([`ChatTrainer::export_checkpoint`]
//! writes `config.json` beside it), so a trained model is a base again.

use std::path::{Path, PathBuf};

use gpu_core::select::Dtype;
use model::finetune::{AdapterLineage, ChatTrainer, LoraRequest, ModelFit, OptimizerState};
use model::{FitOpts, FitReport};

use crate::train::{BuildOptions, Inkling2Train, TrainConfig, FAMILY};

/// The generic trainers' entry for Inkling 2.
pub struct Inkling2Trainer;

fn invalid(path: &Path, why: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{}: {why}", path.display()))
}

/// Write the adapter of a LoRA build to `path` (the frozen base is not written), carded as an Inkling 2 adapter.
pub fn save_adapter_file(trained: &Inkling2Train, path: &Path) -> std::io::Result<()> {
    let lineage = AdapterLineage { card_id: "brain/inkling2-adapter", base_id: "thinkingmachines/Inkling-2-Mini", dataset_id: None, provenance: None };
    <Inkling2Trainer as ChatTrainer>::save_adapter(trained, path, &lineage)
}

/// The weights file of a full fine-tune's export, beside `config.json`.
pub const WEIGHTS_FILE: &str = "model.safetensors";

impl ChatTrainer for Inkling2Trainer {
    type Trained = Inkling2Train;
    /// The tier the frozen dense linears are held at: `bf16` keeps the release's bf16 tensors as published, `f32`
    /// widens them. The expert banks are always held as published - NVFP4 and MXFP8 banks read in place, never
    /// dequantized - whatever the tier (the tier set has no name for a mixed NVFP4/MXFP8 bank).
    const LORA_BASES: model::policy::TrainBases = model::policy::TrainBases { model: "inkling2", bases: &[Dtype::BF16, Dtype::F32] };
    /// f32 weights and f32 moments on the device; no resident or streamed state.
    const TRAIN_PRECISION: model::TrainPrecisionSupport = model::TrainPrecisionSupport { model: "inkling2", state: model::StateSupport::Fp32, resident: false, layer_streamed: false, f32_streamed: false };
    /// The model carries [`model::training::TokenObjectives`] (with routing replay).
    const POLICY_GRADIENT: bool = true;

    fn checkpoint_path(weights: &Path, model_dir: &Path) -> PathBuf {
        if weights.is_dir() {
            weights.to_path_buf()
        } else {
            model_dir.to_path_buf()
        }
    }

    fn vocab(open: &Path) -> Result<usize, String> {
        TrainConfig::from_dir(open).map(|tc| tc.cfg.valid_vocab as usize)
    }

    fn train_full(base: &Path, opts: &FitOpts, state: OptimizerState, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Inkling2Train)> {
        Self::TRAIN_PRECISION.check(&opts.precision).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))?;
        if state != OptimizerState::Device {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "inkling2: the optimizer state lives on the device (no host offload)"));
        }
        let m = Inkling2Train::from_dir(base, opts.batch_size, opts.block_size).map_err(|e| {
            if e.contains("microscaled") {
                std::io::Error::new(std::io::ErrorKind::Unsupported, e)
            } else {
                invalid(base, e)
            }
        })?;
        fit.fit(m)
    }

    fn train_lora(base: &Path, lora: &LoraRequest<'_>, opts: &FitOpts, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Inkling2Train)> {
        let tiers = lora.checked::<Self>()?;
        if !lora.modules_to_save.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "inkling2: modules_to_save is not offered; adapt the tensor through its LoRA site"));
        }
        TrainConfig::from_dir(base).map_err(|e| invalid(base, e))?;
        // The float expert banks of a small checkpoint have no stacked-expert adapter: its targets leave them out.
        let micro = crate::model::open_checkpoint(base).map_err(|e| invalid(base, e))?.2;
        let targets = crate::train::targets(!micro.is_empty());
        let build = BuildOptions { batch: opts.batch_size, seq: opts.block_size, base: tiers, seed: opts.seed, continue_from: lora.continue_from };
        let m = Inkling2Train::from_dir_lora(base, model::adapter::device::LoraCfg::new(lora.rank, lora.alpha, targets), &build).map_err(|e| invalid(base, e))?;
        fit.fit(m)
    }

    /// The served engine over the release at `base` (as published, whatever `tiers` names: the engine holds the
    /// linears as stored and the expert banks in their encoding), f32 K/V, sized for one sequence of `max_seq_len`
    /// tokens. An adapter is refused by name: the engine has no adapter bank yet and the expert pairs cannot be folded
    /// into quantized banks.
    fn serving_engine(base: &Path, _tiers: &model::ops::TierPolicy, adapter: Option<&Path>, max_seq_len: u32) -> Option<std::io::Result<Box<dyn model::serve::PagedDecoder>>> {
        Some((|| {
            if let Some(a) = adapter {
                return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, format!("inkling2: the served engine applies no adapter yet ({})", a.display())));
            }
            const BLOCK: u32 = 16;
            let tokens = max_seq_len.max(1);
            let sizing = model::serve::visit_state::VisitPoolSizing { block_size: BLOCK, num_blocks: tokens.div_ceil(BLOCK) + 1, live_rows: 1, snapshot_rows: 0, undo_rows: false };
            let opts = crate::serve::EngineOptions { kv_tier: model::kv_tier::KvTier::F32, sizing, max_seq_len: tokens as usize, prefill_chunk: tokens.min(256) };
            let ops = crate::serve::device_ops().map_err(std::io::Error::other)?;
            let engine = crate::serve::Engine::open(ops, base, opts).map_err(|e| invalid(base, e))?;
            Ok(Box::new(engine) as Box<dyn model::serve::PagedDecoder>)
        })())
    }

    fn save_adapter(trained: &Inkling2Train, path: &Path, lineage: &AdapterLineage<'_>) -> std::io::Result<()> {
        let lora = trained.train_config().lora.as_ref().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "inkling2: a full fine-tune has no adapter to save"))?;
        let name = path.to_str().ok_or_else(|| invalid(path, "not a UTF-8 path"))?;
        model::lora::device_adapter::save_lora_with_lineage(name, trained, lora, lineage.card_id, lineage.base_id, FAMILY, lineage.dataset_id, lineage.provenance.clone())
    }

    /// The trained file (a full fine-tune's save: the published names with the checkpoint's config in its header) as
    /// a checkpoint directory: `config.json`, [`WEIGHTS_FILE`] and the tokenizer files. An adapter is not folded in:
    /// the banks it corrects are quantized, so it is served beside the base.
    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>, keep_going: &mut dyn FnMut() -> bool) -> Result<(), String> {
        if adapter.is_some() {
            return Err("inkling2: an adapter is not folded into a checkpoint (its expert pairs correct quantized banks); serve it beside the base".into());
        }
        let name = trained.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", trained.display()))?;
        let config = checkpoint::weightio::WeightReader::open(name).map_err(|e| format!("{name}: {e}"))?.config();
        TrainConfig::from_value(config.clone()).map_err(|e| format!("{name}: {e}"))?;
        if !keep_going() {
            return Err(capability::failure::CANCELLED.to_string());
        }
        std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
        let text = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
        std::fs::write(out.join("config.json"), text).map_err(|e| format!("{}: {e}", out.display()))?;
        if !keep_going() {
            let _ = std::fs::remove_file(out.join("config.json"));
            return Err(capability::failure::CANCELLED.to_string());
        }
        std::fs::copy(trained, out.join(WEIGHTS_FILE)).map_err(|e| format!("{}: {e}", trained.display()))?;
        for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja"] {
            let from = tokenizer_dir.join(f);
            if from.is_file() {
                std::fs::copy(&from, out.join(f)).map_err(|e| format!("{}: {e}", from.display()))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: a train precision the trainer does not run is refused by name through the generic entry point, before the
    /// base is opened.
    #[test]
    fn a_train_precision_it_does_not_run_is_refused_by_name_before_anything_is_opened() {
        let nowhere = Path::new("/nonexistent/inkling2-base");
        for spec in ["weights=bf16", "state=int8"] {
            let opts = FitOpts { precision: model::TrainPrecision::parse(spec).unwrap(), ..FitOpts::default() };
            let e = model::finetune::fit_full::<Inkling2Trainer>(nowhere, nowhere, &opts, nowhere, OptimizerState::Device, model::FitControl::default()).expect_err(spec);
            assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{spec}: {e}");
            assert!(e.to_string().contains("inkling2"), "{spec}: {e}");
        }
    }
}
