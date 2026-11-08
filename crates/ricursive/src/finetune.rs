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
use model::{FitOpts, FitReport, Model, Pipeline, PipelineModel, Shard};

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

/// A trained Ripple: one stage on one device, or the pipeline of stages the layout cut it into. Both arms are boxed:
/// each is a whole on-device model.
pub enum RippleRun {
    Single(Box<Ripple>),
    Pipeline(Box<PipelineModel<Ripple>>),
}

impl RippleRun {
    /// Write the trained adapter (never the frozen base) to `path`, as [`Ripple::save_adapter`].
    pub fn save_adapter(&self, path: &str, card_id: &str, base_id: &str, dataset_id: Option<&str>, training: Option<checkpoint::st::TrainingProvenance>) -> std::io::Result<()> {
        match self {
            RippleRun::Single(m) => m.save_adapter(path, card_id, base_id, dataset_id, training),
            RippleRun::Pipeline(m) => {
                let lora = Model::config(&**m).lora.clone().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "ricursive: this model was not built with an adapter"))?;
                model::lora::device_adapter::save_lora_with_lineage(path, &**m, &lora, card_id, base_id, crate::train::adapter::FAMILY, dataset_id, training)
            }
        }
    }

    /// The stages of the run (1 for a single device).
    pub fn stages(&self) -> usize {
        match self {
            RippleRun::Single(_) => 1,
            RippleRun::Pipeline(m) => m.pipeline().n_stages(),
        }
    }
}

/// `cfg` built over `src` and laid out over the machine as the generic planner ([`model::plan_stage_layout`]) says:
/// one stage when a device holds the model, the fewest stages that fit otherwise, or exactly `opts.stages`. The
/// layout is printed, so an automatic split is never a silent one.
fn build_run(cfg: TrainConfig, opts: &FitOpts, src: &dyn TensorSource, train_opts: &TrainOpts, resident: Option<&model::resident_ft::ResidentFtConfig>, base: &Path) -> std::io::Result<RippleRun> {
    let (b, t) = (opts.batch_size, opts.block_size);
    let place = |needs: &[gpu_core::devices::Need]| gpu_core::devices::place(needs).map(|homes| homes.parts().iter().map(|(_, home)| *home).collect());
    // A resident stage costs what its resident state and the planner say; any other, what the f32 or packed build holds.
    let stage_bytes = |sh: &Shard| match resident {
        Some(r) => crate::train::stage_costs(&cfg, std::slice::from_ref(sh), b, t, r).map_or(u64::MAX, |c| c[0].device()),
        None => crate::train::estimate_stage_bytes(&cfg, sh, &train_opts.tiers, b, t),
    };
    let shards = model::plan_stage_layout::<Ripple>("ricursive finetune", &cfg, b, t, gpu_core::devices::gpus().len(), opts.stages, stage_bytes, place).map_err(|e| invalid(base, e))?;
    if resident.is_some_and(|r| r.window.is_some()) && shards.len() > 1 {
        return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "ricursive: a layer window and several pipeline stages do not compose: a streamed stage runs its backward inside its update, while the pipeline hands the gradient of the boundary from stage to stage; give weights= without window= for a pipeline"));
    }
    let make = |cfg: TrainConfig, shard: Shard| {
        let stage_opts = TrainOpts { resident: resident.map(|r| crate::train::plan_resident(&cfg, &shard, b, t, r, model::memplan::Capacity::measured())), ..train_opts.clone() };
        Ripple::build_shard(cfg, b, t, src, &stage_opts, true, shard).unwrap_or_else(|e| panic!("{}: {e}", base.display()))
    };
    if let [whole] = shards.as_slice() {
        let cfg_whole = TrainConfig { ..cfg };
        let whole = Shard { start: 0, end: cfg_whole.model.num_hidden_layers, embed: true, head: true, gpu_index: whole.gpu_index };
        let built = if whole.gpu_index == Shard::ANY_GPU { Ok(make(cfg_whole, whole)) } else { gpu_core::devices::with_gpu(whole.gpu_index as u32, || make(cfg_whole, whole)) };
        return built.map(|m| RippleRun::Single(Box::new(m))).map_err(std::io::Error::other);
    }
    let layout: Vec<String> = shards.iter().map(|sh| if sh.gpu_index == Shard::ANY_GPU { format!("host layers {}..{}", sh.start, sh.end) } else { format!("gpu{} layers {}..{}", sh.gpu_index, sh.start, sh.end) }).collect();
    println!("ricursive finetune: pipeline of {} stages: {}", shards.len(), layout.join(", "));
    Ok(RippleRun::Pipeline(Box::new(PipelineModel::new(Pipeline::<Ripple>::from_stages(cfg.clone(), shards, make), cfg))))
}

/// Train `run` through `fit`, whichever layout it has.
fn fit_run(run: RippleRun, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, RippleRun)> {
    match run {
        RippleRun::Single(m) => fit.fit(*m).map(|(report, m)| (report, RippleRun::Single(Box::new(m)))),
        RippleRun::Pipeline(m) => fit.fit(*m).map(|(report, m)| (report, RippleRun::Pipeline(Box::new(m)))),
    }
}

impl ChatTrainer for RippleTrainer {
    type Trained = RippleRun;
    const LORA_BASES: model::policy::TrainBases = model::policy::TrainBases { model: "ricursive", bases: &[Dtype::F32, Dtype::BF16, Dtype::I8, Dtype::NF4, Dtype::F8E4M3, Dtype::F4E2M1] };
    /// A plain full fine-tune trains f32 weights with f32 moments ([`Self::train_full`] refuses any other precision by
    /// name); a resident one (`weights=bf16|fp8`, with or without a layer `window`) holds the optimizer state of its
    /// off-device master in any format the resident state runs, the f32 layers' streaming through a parameter-store
    /// window is not wired.
    const TRAIN_PRECISION: model::TrainPrecisionSupport = model::TrainPrecisionSupport { model: "ricursive", state: model::StateSupport::Any, resident: true, layer_streamed: true, f32_streamed: false };

    const PIPELINED: bool = true;
    const NATIVE_SPARSE: bool = true;
    const POLICY_GRADIENT: bool = true;
    const MODULES_TO_SAVE: bool = true;
    const SCORES_HELD_OUT: bool = true;

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

    fn train_full(base: &Path, opts: &FitOpts, state: OptimizerState, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, RippleRun)> {
        Self::TRAIN_PRECISION.check(&opts.precision).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))?;
        let default = paramstore::precision::PrecisionPolicy::uniform(paramstore::precision::ParamPrecision::FP32);
        if opts.precision.policy.default != default.default || !opts.precision.policy.rules.is_empty() || opts.precision.narrow_state || opts.precision.host_moments {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "ricursive: a plain full fine-tune trains f32 weights with f32 moments on the device; give weights=bf16 or weights=fp8 for a low-precision optimizer state"));
        }
        let opened = Base::open(base)?;
        let cfg = opened.train_config(opts.block_size);
        let train_opts = TrainOpts { gemm: opts.gemm_precision, ..TrainOpts::default() };
        let run = state.scope(|| build_run(cfg, opts, &*opened.source, &train_opts, None, base))?;
        fit_run(run, fit)
    }

    /// The linears and the two tables resident in `resident`'s encoding with a master copy and the optimizer state off
    /// the device, optionally streamed through a window of layers; the norms and gates train in f32 beside them. Over
    /// several stages each stage owns its resident state and optimizer (the pipeline exchanges the gradient norm and the
    /// replicated embedding's gradient); a layer window is one stage's. Refused when the planner finds it does not fit.
    fn train_full_resident(base: &Path, opts: &FitOpts, resident: &model::resident_ft::ResidentFtConfig, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, RippleRun)> {
        if !matches!(resident.value, model::resident_ft::ResidentValue::Bf16 | model::resident_ft::ResidentValue::Fp8E4M3 { .. }) {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "ricursive: a resident full fine-tune holds its linears in bf16 or fp8; int8 and nf4 (quantization-aware training) are not tested for it"));
        }
        let opened = Base::open(base)?;
        let cfg = opened.train_config(opts.block_size);
        let train_opts = TrainOpts { gemm: opts.gemm_precision, ..TrainOpts::default() };
        let run = build_run(cfg, opts, &*opened.source, &train_opts, Some(resident), base)?;
        fit_run(run, fit)
    }

    fn train_lora(base: &Path, lora: &LoraRequest<'_>, opts: &FitOpts, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, RippleRun)> {
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
        // The adapter being continued brings its own full tensors along: they keep training unless the run names others too.
        let continued = lora.continue_from.map(|path| model::lora::device_adapter::read_adapter(path_str(path)?)).transpose()?;
        let mut saved = crate::train::expand_modules_to_save(lora.modules_to_save);
        saved.extend(continued.iter().flat_map(|a| a.saved.iter().map(|t| t.base.clone())));
        let cfg = opened.train_config(opts.block_size).with_lora(LoraCfg::new(lora.rank, lora.alpha, default_targets()).with_modules_to_save(saved));
        cfg.validate().map_err(|e| invalid(base, e))?;
        let init = match &continued {
            Some(adapter) => AdapterInit::continued(&cfg, &*opened.source, adapter).map_err(std::io::Error::other)?,
            None => AdapterInit::fresh(&cfg, &*opened.source, opts.seed),
        };
        let train_opts = TrainOpts { tiers, gemm: opts.gemm_precision, ..TrainOpts::default() };
        let run = OptimizerState::Device.scope(|| build_run(cfg, opts, &init, &train_opts, None, base))?;
        fit_run(run, fit)
    }

    fn save_adapter(trained: &RippleRun, path: &Path, lineage: &AdapterLineage<'_>) -> std::io::Result<()> {
        trained.save_adapter(path_str(path)?, lineage.card_id, lineage.base_id, lineage.dataset_id, lineage.provenance.clone())
    }

    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>) -> Result<(), String> {
        model::finetune::export_brain_checkpoint("ricursive", trained, tokenizer_dir, out, adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: a train precision the trainer does not run is refused by name through the generic entry point, before
    /// the base is opened.
    #[test]
    fn a_train_precision_it_does_not_run_is_refused_by_name_before_anything_is_opened() {
        let nowhere = Path::new("/nonexistent/ricursive-base");
        for spec in ["weights=f32,window=2", "state=int8"] {
            let opts = FitOpts { precision: model::TrainPrecision::parse(spec).unwrap(), ..FitOpts::default() };
            let e = model::finetune::fit_full::<RippleTrainer>(nowhere, nowhere, &opts, nowhere, OptimizerState::HostOffload, model::FitControl::default()).expect_err(spec);
            assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{spec}: {e}");
            assert!(e.to_string().contains("ricursive"), "{spec}: {e}");
        }
    }
}
