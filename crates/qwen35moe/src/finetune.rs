// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning sparse
// mixture-of-experts language models for its clients. If your team needs
// expertise in training routed-expert models then you can procure our services
// by sending an email to info@swedishembedded.com.

//! Qwen3.5-35B-A3B as a [`model::finetune::ChatTrainer`]: the generic
//! full-parameter, LoRA and preference fine-tuners run on it.
//!
//! Which parameters train follows the method, not this file: a full fine-tune
//! trains every tensor (the routed experts and the router included), a LoRA
//! adapter covers only the mixer projections ([`crate::config::lora_targets`])
//! and the experts stay frozen. The base is a brain-format checkpoint read as
//! fp32 on the host and held at fp32 on the card; there is no streamed or
//! narrow-tier constructor for this architecture yet.

use std::path::{Path, PathBuf};

use model::finetune::{AdapterLineage, LoraRequest, ModelFit, OptimizerState};
use model::memplan::{plan, Authorized, Capacity, Category, Demand, Lever, Variant};
use model::{FitOpts, FitReport};

use crate::config::{lora_cfg, Qwen35Config};
use crate::model::Qwen35;

pub struct Qwen35MoeTrainer;

/// The most host RAM this trainer may use to read a base (the shared machine's
/// rule for anything that may exceed ~20 GB): the checkpoint is held as fp32 on
/// the host while it is uploaded, twice over at the peak (the container and the
/// map built from it). Far below a real 35B checkpoint (140 GB of fp32 weights),
/// which needs a streamed constructor this architecture does not have.
pub const HOST_LOAD_BUDGET_BYTES: u64 = 16 << 30;

fn path_str(path: &Path) -> std::io::Result<&str> {
    path.to_str().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{}: not a UTF-8 path", path.display())))
}

/// Host RAM the machine reports as available, `None` when it does not say.
fn available_host_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = meminfo.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb * 1024)
}

/// The planner's refusal, before anything is read: the training categories of this run (frozen base,
/// trainable values, gradients, AdamW moments) plus the transient fp32 host copy of the base, against
/// `host_bytes` of host RAM. The device is not asked here; the build refuses what a card cannot hold.
fn plan_before_load(params: &[(String, usize)], adapter_names: &[String], state: OptimizerState, host_bytes: u64) -> Result<(), String> {
    use model::memplan::{training_demands, TrainingFootprint};
    let total: u64 = params.iter().map(|(_, n)| *n as u64).sum();
    let trainable: Vec<(String, usize)> = if adapter_names.is_empty() { params.to_vec() } else { params.iter().filter(|(n, _)| adapter_names.contains(n)).cloned().collect() };
    let trainable_elems: u64 = trainable.iter().map(|(_, n)| *n as u64).sum();
    let footprint = TrainingFootprint {
        frozen_weight_bytes: 4 * (total - if adapter_names.is_empty() { total } else { trainable_elems }),
        trainable,
        precision: model::memplan::ParamPrecision::FP32,
        moment_slots: 2,
        int8_block: 256,
        optimizer_state_on_host: state == OptimizerState::HostOffload,
        activation_bytes: 0,
        recompute_activation_bytes: None,
        saved_boundaries: None,
    };
    let mut demands = training_demands(&footprint).map_err(|e| e.to_string())?;
    demands.retain(|d| d.category != Category::Activations);
    demands.push(Demand::with_variants(Category::Staging, vec![Variant::new(Lever::Base, "fp32 host copy of the base while it loads", 0, 2 * 4 * total)]).map_err(|e| e.to_string())?);
    let capacity = Capacity { device_bytes: u64::MAX, host_bytes };
    plan(&demands, capacity, &Authorized::none()).map(drop).map_err(|e| format!("qwen35moe reads its base as fp32 on the host and has no streamed loader: {e}"))
}

/// [`plan_before_load`] for the checkpoint at `path`, from its header alone.
fn check_before_load(path: &Path, adapter: Option<&LoraRequest<'_>>, state: OptimizerState) -> std::io::Result<()> {
    let name = path_str(path)?;
    let reader = checkpoint::weightio::WeightReader::open(name).map_err(|e| std::io::Error::other(format!("open {name}: {e}")))?;
    let mut cfg = Qwen35Config::from_json_checked(&reader.config()).map_err(|e| std::io::Error::other(format!("{name}: {e}")))?;
    let mut adapters = Vec::new();
    if let Some(lora) = adapter {
        cfg.lora = Some(lora_cfg(lora.rank, lora.alpha));
        adapters = cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| model::adapter::device::is_adapter_param(n)).collect();
    }
    let host = HOST_LOAD_BUDGET_BYTES.min(available_host_bytes().unwrap_or(u64::MAX));
    plan_before_load(&cfg.param_list(), &adapters, state, host).map_err(std::io::Error::other)
}

impl model::finetune::ChatTrainer for Qwen35MoeTrainer {
    type Trained = Qwen35;
    /// The experts have no narrow training tier: the base is fp32.
    const LORA_BASES: model::policy::TrainBases = model::policy::TrainBases { model: "qwen35moe", bases: &[gpu_core::select::Dtype::F32] };
    const TRAIN_PRECISION: model::TrainPrecisionSupport = model::TrainPrecisionSupport { model: "qwen35moe", state: model::StateSupport::Any, resident: false, layer_streamed: false, f32_streamed: true };

    fn checkpoint_path(weights: &Path, model_dir: &Path) -> PathBuf {
        qwen3::store_checkpoint_path(weights, model_dir)
    }

    fn vocab(open: &Path) -> Result<usize, String> {
        let name = open.to_str().ok_or_else(|| format!("{}: not a UTF-8 path", open.display()))?;
        let reader = checkpoint::weightio::WeightReader::open(name).map_err(|e| format!("open {name}: {e}"))?;
        Ok(Qwen35Config::from_json_checked(&reader.config()).map_err(|e| format!("{name}: {e}"))?.vocab as usize)
    }

    fn train_full(base: &Path, opts: &FitOpts, state: OptimizerState, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Qwen35)> {
        Self::TRAIN_PRECISION.check(&opts.precision).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))?;
        check_before_load(base, None, state)?;
        let m = state.scope(|| Ok::<_, std::io::Error>(Qwen35::load_with_adapter(path_str(base)?, opts.batch_size, opts.block_size, None)))?;
        fit.fit(m)
    }

    fn train_lora(base: &Path, lora: &LoraRequest<'_>, opts: &FitOpts, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Qwen35)> {
        lora.checked::<Self>()?;
        if lora.continue_from.is_some() {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "qwen35moe does not continue an adapter yet"));
        }
        check_before_load(base, Some(lora), OptimizerState::Device)?;
        let m = OptimizerState::Device.scope(|| Ok::<_, std::io::Error>(Qwen35::load_with_adapter(path_str(base)?, opts.batch_size, opts.block_size, Some((lora_cfg(lora.rank, lora.alpha), opts.seed)))))?;
        fit.fit(m)
    }

    /// The adapter file carries no training provenance: this architecture's
    /// writer has none.
    fn save_adapter(trained: &Qwen35, path: &Path, lineage: &AdapterLineage<'_>) -> std::io::Result<()> {
        crate::lora::save_adapter(path_str(path)?, trained, lineage.card_id, lineage.base_id, lineage.dataset_id)
    }

    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>) -> Result<(), String> {
        model::finetune::export_brain_checkpoint("qwen35moe", trained, tokenizer_dir, out, adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_base_whose_fp32_host_copy_exceeds_the_budget_is_refused_before_it_is_read() {
        let toy = vec![("w".to_string(), 10_000_000usize)];
        assert!(plan_before_load(&toy, &[], OptimizerState::HostOffload, HOST_LOAD_BUDGET_BYTES).is_ok(), "a toy base fits");
        let big = vec![("w".to_string(), 35_000_000_000usize)];
        let refused = plan_before_load(&big, &[], OptimizerState::HostOffload, HOST_LOAD_BUDGET_BYTES).expect_err("a 35B base does not");
        assert!(refused.contains("does not fit") && refused.contains("no streamed loader"), "{refused}");
    }

    /// Spec: this family's full fine-tune keeps fp32 parameters and AdamW state, so a train precision it does not run is
    /// refused by name, through the generic entry point, before the base is opened.
    #[test]
    fn a_train_precision_it_does_not_run_is_refused_by_name_before_anything_is_opened() {
        let nowhere = Path::new("/nonexistent/qwen35moe-base");
        for spec in ["weights=fp8", "weights=bf16,window=2"] {
            let opts = FitOpts { precision: model::TrainPrecision::parse(spec).unwrap(), ..FitOpts::default() };
            let e = model::finetune::fit_full::<Qwen35MoeTrainer>(nowhere, nowhere, &opts, nowhere, OptimizerState::HostOffload, model::FitControl::default()).expect_err(spec);
            assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{spec}: {e}");
            assert!(e.to_string().contains("qwen35moe"), "{spec}: {e}");
        }
    }
}
