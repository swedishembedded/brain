// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fine-tuning MiniCPM-V: the decoder, the connector (merger) and the vision
//! tower, each trained or frozen on its own.
//!
//! | part | modes |
//! |---|---|
//! | decoder | LoRA on its 12 projections, full, or frozen |
//! | connector | the merger trained, or frozen |
//! | vision | LoRA on the tower's linears, full, or frozen |
//!
//! One step on a [`Sample`]: the images and video frames go through the
//! tower (keeping what its backward needs only when the vision side
//! trains), the visual rows are spliced over the placeholder rows of the
//! decoder's input, the decoder runs the next-token loss over the supervised
//! answer, and its backward hands the gradient of the spliced rows
//! (`read_d_img_embeds`) to the tower's backward. Gradients accumulate over
//! a batch; [`Trainer::step`] applies AdamW to every trained part and clears
//! them.
//!
//! A frozen decoder is a rank-1 LoRA build whose adapters are never stepped:
//! `B` starts at zero, so it computes exactly the base while still giving the
//! input gradient the vision side needs.

use std::collections::HashMap;
use std::path::Path;

use data::rng::Rng;
use gpu_core::Gpu;
use model::saved::SavedPolicy;
use optim::Optim;
use qwen35::base_source::BaseCheckpoint;
use qwen35::model::{Qwen35, TrainOpts};

use crate::config::{Downsample, MiniCpmVConfig};
use crate::dataset::Sample;
use crate::preprocess::ProcessedImage;
use crate::vision::{Tower, View};
use crate::vision_train::{train_pipelines, TowerLora, TrainPass, LORA_LEAVES};

/// How a part trains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartMode {
    Frozen,
    Lora { rank: u32, alpha: f32 },
    Full,
}

impl PartMode {
    pub fn name(self) -> String {
        match self {
            PartMode::Frozen => "frozen".into(),
            PartMode::Lora { rank, alpha } => format!("lora:{rank}:{alpha}"),
            PartMode::Full => "full".into(),
        }
    }

    /// `frozen`, `full`, or `lora:<rank>:<alpha>`.
    pub fn parse(s: &str) -> Result<PartMode, String> {
        match s {
            "frozen" => Ok(PartMode::Frozen),
            "full" => Ok(PartMode::Full),
            other => {
                let mut it = other.strip_prefix("lora:").ok_or_else(|| format!("training mode {other:?}: expected frozen, full or lora:<rank>:<alpha>"))?.split(':');
                let rank: u32 = it.next().and_then(|r| r.parse().ok()).filter(|r| *r > 0).ok_or_else(|| format!("{other:?}: a LoRA rank must be a positive integer"))?;
                let alpha: f32 = it.next().map(|a| a.parse().map_err(|_| format!("{other:?}: alpha is not a number"))).transpose()?.unwrap_or(2.0 * rank as f32);
                Ok(PartMode::Lora { rank, alpha })
            }
        }
    }
}

/// Peak learning rates per part; the schedule scales all three alike. Parts
/// trained in full need far smaller rates than LoRA factors: Adam moves
/// every weight by about the rate each step, and the connector's and the
/// tower's weights are small enough that a LoRA-sized rate scrambles them
/// within a few dozen steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    pub decoder: f32,
    pub connector: f32,
    pub vision: f32,
}

impl Rates {
    /// The same rate everywhere.
    pub fn uniform(lr: f32) -> Rates {
        Rates { decoder: lr, connector: lr, vision: lr }
    }

    /// The default rates around a LoRA rate `lr`: the decoder and a LoRA
    /// tower at `lr`, the connector at a tenth, a tower in full at a fiftieth
    /// (the usual ratios for fine-tuning a vision-language model's projector
    /// and vision encoder).
    pub fn around(lr: f32, vision: PartMode) -> Rates {
        let vision = if vision == PartMode::Full { lr / 50.0 } else { lr };
        Rates { decoder: lr, connector: lr / 10.0, vision }
    }
}

/// What trains and how.
#[derive(Clone, Debug)]
pub struct TrainSpec {
    pub decoder: PartMode,
    pub connector: bool,
    pub vision: PartMode,
    /// Tokens per sample the decoder is built for; longer samples are refused.
    pub block: u32,
    pub downsample: Downsample,
    pub lr: Rates,
    pub weight_decay: f32,
    pub grad_clip: Option<f32>,
    /// Seeds the adapters' `A` initialisation.
    pub seed: u64,
}

pub struct Trainer {
    pub cfg: MiniCpmVConfig,
    pub spec: TrainSpec,
    pub tower: Tower,
    pub lora: Option<TowerLora>,
    pub decoder: Qwen35,
    tower_opt: Optim,
    lora_opt: Optim,
    /// AdamW steps taken.
    pub steps: u32,
    folded: bool,
}

/// A tensor's AdamW first and second moments.
type Moments = (Vec<f32>, Vec<f32>);

/// One forward's state for the vision backward.
struct VisionPass {
    image: Option<TrainPass>,
    video: Option<TrainPass>,
}

fn optim_on(g: &Gpu) -> Optim {
    let k = |n: &str| g.kernel_index(n).unwrap_or_else(|| panic!("minicpmv: optimiser kernel {n} missing"));
    Optim::new(k("adamw"), k("gradnorm_sq"), k("grad_scale"), k("clip_coef"), k("grad_scale_buf"))
}

impl Trainer {
    /// Build the trainer on `gpu`'s device: the tower from `vision` weights
    /// (canonical names, [`crate::vision::tensor_manifest`]) and the decoder
    /// from `decoder` (its base checkpoint), each as `spec` says.
    pub fn new(gpu: &Gpu, cfg: MiniCpmVConfig, vision: &dyn checkpoint::TensorSource, decoder: BaseCheckpoint, spec: TrainSpec) -> Result<Trainer, String> {
        if spec.decoder == PartMode::Frozen && !spec.connector && spec.vision == PartMode::Frozen {
            return Err("minicpmv: nothing to train - every part is frozen".into());
        }
        let tower_trains = spec.vision != PartMode::Frozen;
        let trainable = |n: &str| if n.starts_with("merger.") { spec.connector } else { tower_trains };
        let tgpu = gpu.new_like(train_pipelines());
        let mut tower = Tower::new_train_on(tgpu, cfg.clone(), vision, &trainable);
        // One optimiser steps the connector and a tower trained in full, each
        // at its own rate: the step's rate is the schedule factor, the
        // per-tensor multiplier the part's peak.
        let names: Vec<String> = tower.ps.opt_params().iter().map(|(n, _)| n.clone()).collect();
        for n in names {
            let rate = if n.starts_with("merger.") { spec.lr.connector } else { spec.lr.vision };
            tower.ps.set_lr_mult(&n, rate);
        }
        let lora = match spec.vision {
            PartMode::Lora { rank, alpha } => {
                // The base keeps its gradient buffers (the factors' gradients
                // are projected from them) but leaves the optimiser.
                tower.ps.freeze_where(|n| !n.starts_with("merger."));
                let mut rng = Rng::new(spec.seed ^ 0x7015);
                Some(TowerLora::new(&tower, LORA_LEAVES, rank, alpha, &mut |n| {
                    let bound = 1.0 / (n as f32 / rank as f32).sqrt();
                    (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * bound).collect()
                })?)
            }
            _ => None,
        };
        let dgpu = gpu.new_like(qwen35::model::pipelines());
        let opts = TrainOpts::default();
        let decoder = match spec.decoder {
            PartMode::Full => {
                let dcfg = qwen35::config::Qwen35Config { block_size: spec.block, ..decoder.cfg.clone() };
                Qwen35::new_train_opts_on(dgpu, dcfg, 1, spec.block, &*decoder.src, &opts)
            }
            PartMode::Lora { rank, alpha } => Self::lora_decoder(dgpu, decoder, rank, alpha, spec.seed, spec.block, &opts)?,
            PartMode::Frozen => Self::lora_decoder(dgpu, decoder, 1, 1.0, spec.seed, spec.block, &opts)?,
        };
        let (tower_opt, lora_opt) = (optim_on(&tower.gpu), optim_on(&tower.gpu));
        Ok(Trainer { cfg, spec, tower, lora, decoder, tower_opt, lora_opt, steps: 0, folded: false })
    }

    fn lora_decoder(gpu: Gpu, base: BaseCheckpoint, rank: u32, alpha: f32, seed: u64, block: u32, opts: &TrainOpts) -> Result<Qwen35, String> {
        let (cfg, overlay) = qwen35::base_source::lora_overlay(base, rank, alpha, seed, None)?;
        let cfg = qwen35::config::Qwen35Config { block_size: block, ..cfg };
        Ok(Qwen35::new_train_opts_on(gpu, cfg, 1, block, &overlay, opts))
    }

    /// Whether any vision-side parameter has a gradient (so the decoder's
    /// input gradient has somewhere to go).
    fn vision_trains(&self) -> bool {
        self.spec.connector || self.spec.vision != PartMode::Frozen
    }

    /// The tower's rows for `items`, keeping its backward state when the
    /// vision side trains.
    fn encode(&self, items: &[&ProcessedImage]) -> (Vec<f32>, Option<TrainPass>) {
        if items.is_empty() {
            return (Vec::new(), None);
        }
        let views: Vec<View> = items.iter().flat_map(|p| p.views.iter().copied()).collect();
        let patches: Vec<f32> = items.iter().flat_map(|p| p.patches.iter().copied()).collect();
        if self.vision_trains() {
            let (rows, pass) = self.tower.encode_train(&patches, &views, self.spec.downsample);
            (rows, Some(pass))
        } else {
            (self.tower.encode(&patches, &views, self.spec.downsample), None)
        }
    }

    /// Load `s` into the decoder: tokens and targets padded to the block,
    /// visual rows spliced over the placeholder rows. Returns the vision
    /// state its backward needs.
    fn load(&mut self, s: &Sample) -> Result<VisionPass, String> {
        let block = self.spec.block as usize;
        if s.ids.len() > block {
            return Err(format!("minicpmv: a sample of {} tokens does not fit the {block}-token block", s.ids.len()));
        }
        if let Some(l) = &self.lora {
            if !self.folded {
                l.fold(&self.tower);
                self.folded = true;
            }
        }
        let (image_rows, image) = self.encode(&s.images.iter().collect::<Vec<_>>());
        let (video_rows, video) = self.encode(&s.videos.iter().flatten().collect::<Vec<_>>());
        let runs: Vec<(u32, u32)> = s.image_runs.iter().chain(&s.video_runs).copied().collect();
        let d = self.cfg.text.d_model as usize;
        let want: usize = runs.iter().map(|r| r.1 as usize).sum::<usize>() * d;
        if want != image_rows.len() + video_rows.len() {
            return Err(format!("minicpmv: {} placeholder rows for {} visual rows", want / d, (image_rows.len() + video_rows.len()) / d));
        }
        if runs.is_empty() {
            self.decoder.disable_mm_splice();
        } else {
            self.decoder.enable_mm_splices(&runs);
            let mut rows = image_rows;
            rows.extend(video_rows);
            self.decoder.write_img_embeds(&rows);
        }
        let mut ids = s.ids.clone();
        let mut targets = s.targets.clone();
        ids.resize(block, 0);
        targets.resize(block, model::IGNORE);
        self.decoder.set_batch(&ids, &targets);
        Ok(VisionPass { image, video })
    }

    /// What the backward keeps of the forward ([`SavedPolicy`]) in the parts that train. The decoder takes it as
    /// `qwen35` does. A vision tower that trains below the connector always rebuilds each encoder block from its
    /// input ([`Tower::set_saved_policy`]), so it accepts only recomputing and keeping is refused by name; with the
    /// tower frozen the decoder alone decides.
    pub fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        if self.spec.vision != PartMode::Frozen {
            self.tower.set_saved_policy(policy)?;
        }
        model::Model::set_saved_policy(&mut self.decoder, policy)
    }

    /// Re-fold the vision LoRA before the next forward (after its factors
    /// were written from outside, as a finite-difference check does).
    pub fn refold(&mut self) {
        self.folded = false;
    }

    /// The mean next-token loss over `s`'s supervised answer.
    pub fn loss(&mut self, s: &Sample) -> Result<f32, String> {
        self.load(s)?;
        Ok(self.decoder.forward())
    }

    /// [`Self::loss`], accumulating every trained part's gradient.
    pub fn accumulate(&mut self, s: &Sample) -> Result<f32, String> {
        let vp = self.load(s)?;
        let loss = self.decoder.forward();
        self.decoder.backward();
        if self.vision_trains() && !(s.image_runs.is_empty() && s.video_runs.is_empty()) {
            let d_rows = self.decoder.read_d_img_embeds();
            let d = self.cfg.text.d_model as usize;
            let n_img: usize = s.image_runs.iter().map(|r| r.1 as usize).sum::<usize>() * d;
            if let Some(p) = &vp.image {
                self.tower.backward(p, &d_rows[..n_img]);
            }
            if let Some(p) = &vp.video {
                self.tower.backward(p, &d_rows[n_img..]);
            }
            if let Some(l) = &self.lora {
                l.project(&self.tower);
            }
        }
        Ok(loss)
    }

    /// One AdamW step on every trained part at `factor` (the schedule's
    /// value in `[0, 1]`) times its peak rate, gradients averaged over
    /// `batch` accumulated samples; then clear the gradients.
    pub fn step(&mut self, factor: f32, batch: usize) {
        self.steps += 1;
        let (t, wd, clip, scale) = (self.steps, self.spec.weight_decay, self.spec.grad_clip, 1.0 / batch.max(1) as f32);
        let adam = model::Adam::default();
        if self.spec.decoder != PartMode::Frozen {
            self.decoder.adamw_step(t, factor * self.spec.lr.decoder, wd, adam, clip, scale);
        }
        if !self.tower.ps.opt_params().is_empty() {
            self.tower_opt.step(&self.tower.gpu, &self.tower.ps, t, factor, wd, adam, clip, scale);
        }
        if let Some(l) = &self.lora {
            self.lora_opt.step(&self.tower.gpu, &l.ps, t, factor * self.spec.lr.vision, wd, adam, clip, scale);
            l.ps.zero_grads(&self.tower.gpu);
            self.folded = false;
        }
        self.decoder.zero_grads();
        self.tower.zero_grads();
    }

    /// Every trained tensor, by part: `decoder.<name>`, `vision.<name>`,
    /// `vision_lora.<name>`. The base's frozen tensors are not included.
    pub fn trained_tensors(&self) -> Vec<(String, Vec<f32>)> {
        let mut out = Vec::new();
        if self.spec.decoder != PartMode::Frozen {
            for n in self.decoder.param_names() {
                out.push((format!("decoder.{n}"), self.decoder.read_weight(&n)));
            }
        }
        for (n, _) in self.tower.ps.opt_params() {
            out.push((format!("vision.{n}"), self.tower.read_weight(n)));
        }
        if let Some(l) = &self.lora {
            for (n, v) in l.tensors(&self.tower.gpu) {
                out.push((format!("vision_lora.{n}"), v));
            }
        }
        out
    }

    /// The AdamW moments of every trained tensor, by the same names.
    fn moments(&self) -> Vec<(String, Moments)> {
        let mut out = Vec::new();
        if self.spec.decoder != PartMode::Frozen {
            for n in self.decoder.param_names() {
                if let Some(m) = self.decoder.read_moments(&n) {
                    out.push((format!("decoder.{n}"), m));
                }
            }
        }
        for (n, _) in self.tower.ps.opt_params() {
            if let Some(m) = self.tower.ps.read_moments(&self.tower.gpu, n) {
                out.push((format!("vision.{n}"), m));
            }
        }
        if let Some(l) = &self.lora {
            for (n, _) in l.ps.opt_params() {
                if let Some(m) = l.ps.read_moments(&self.tower.gpu, n) {
                    out.push((format!("vision_lora.{n}"), m));
                }
            }
        }
        out
    }

    /// Write the whole training state - trained tensors, AdamW moments, the
    /// step count and the caller's data position - to `path`, so
    /// [`Self::restore`] continues exactly where this run stands.
    pub fn save_state(&self, path: &Path, position: u64) -> Result<(), String> {
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self.trained_tensors().into_iter().map(|(n, v)| (n, vec![v.len() as u64], v)).collect();
        for (n, (m, v)) in self.moments() {
            tensors.push((format!("adam_m.{n}"), vec![m.len() as u64], m));
            tensors.push((format!("adam_v.{n}"), vec![v.len() as u64], v));
        }
        let config = serde_json::json!({"steps": self.steps, "position": position, "spec": format!("{:?}", self.spec)});
        let p = path.to_str().ok_or("non-UTF-8 state path")?;
        checkpoint::st::save_safetensors(p, &tensors, &config, None).map_err(|e| format!("{p}: {e}"))
    }

    /// Restore a [`Self::save_state`] file into a trainer built with the same
    /// spec; returns the saved data position.
    pub fn restore(&mut self, path: &Path) -> Result<u64, String> {
        let p = path.to_str().ok_or("non-UTF-8 state path")?;
        let st = checkpoint::st::load_safetensors(p).map_err(|e| format!("{p}: {e}"))?;
        let cfg = st.config();
        if cfg["spec"].as_str() != Some(format!("{:?}", self.spec).as_str()) {
            return Err(format!("{p}: saved by a trainer with a different spec ({})", cfg["spec"]));
        }
        let tensors: HashMap<String, Vec<f32>> = st.tensors.into_iter().collect();
        let get = |n: &str| tensors.get(n).ok_or_else(|| format!("{p}: no {n}"));
        for (name, _) in self.trained_tensors() {
            let v = get(&name)?;
            if let Some(n) = name.strip_prefix("decoder.") {
                self.decoder.write_weight(n, v);
            } else if let Some(n) = name.strip_prefix("vision.") {
                self.tower.write_weight(n, v);
            } else if let (Some(n), Some(l)) = (name.strip_prefix("vision_lora."), &self.lora) {
                self.tower.gpu.write_f32(l.ps.w(n), v);
            }
        }
        for (name, _) in self.moments() {
            let (m, v) = (get(&format!("adam_m.{name}"))?, get(&format!("adam_v.{name}"))?);
            if let Some(n) = name.strip_prefix("decoder.") {
                self.decoder.write_moments(n, m, v)?;
            } else if let Some(n) = name.strip_prefix("vision.") {
                self.tower.ps.write_moments(&self.tower.gpu, n, m, v)?;
            } else if let (Some(n), Some(l)) = (name.strip_prefix("vision_lora."), &self.lora) {
                l.ps.write_moments(&self.tower.gpu, n, m, v)?;
            }
        }
        self.steps = cfg["steps"].as_u64().ok_or_else(|| format!("{p}: no step count"))? as u32;
        self.folded = false;
        cfg["position"].as_u64().ok_or_else(|| format!("{p}: no data position"))
    }

    /// Write what serving needs to run this fine-tune into `dir`:
    /// `tune.json` (which parts trained, how, over which base), the decoder's
    /// LoRA adapter (`decoder.adapter.safetensors`, qwen35's adapter format)
    /// or its full weights (`decoder.safetensors`), and the vision side's
    /// trained tensors and LoRA factors (`vision.safetensors`).
    pub fn save_tune(&self, dir: &Path, base_id: &str, label: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = |f: &str| dir.join(f).to_str().map(str::to_string).ok_or_else(|| "non-UTF-8 tune path".to_string());
        match self.spec.decoder {
            PartMode::Lora { .. } => qwen35::lora::save_adapter(&path(TUNE_DECODER_ADAPTER)?, &self.decoder, label, base_id, None).map_err(|e| e.to_string())?,
            PartMode::Full => {
                let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self.decoder.param_names().into_iter().map(|n| {
                    let v = self.decoder.read_weight(&n);
                    (n, vec![v.len() as u64], v)
                }).collect();
                checkpoint::save(&path(TUNE_DECODER_FULL)?, self.decoder.cfg.to_json(), &tensors);
            }
            PartMode::Frozen => {}
        }
        let mut vision: Vec<(String, Vec<u64>, Vec<f32>)> = self.tower.ps.opt_params().iter().map(|(n, _)| {
            let v = self.tower.read_weight(n);
            (n.clone(), vec![v.len() as u64], v)
        }).collect();
        let mut vcfg = serde_json::json!({});
        if let Some(l) = &self.lora {
            vision.extend(l.tensors(&self.tower.gpu).into_iter().map(|(n, v)| (n, vec![v.len() as u64], v)));
            vcfg = serde_json::json!({"lora_rank": l.rank, "lora_alpha": l.alpha});
        }
        if !vision.is_empty() {
            checkpoint::save(&path(TUNE_VISION)?, vcfg, &vision);
        }
        let manifest = serde_json::json!({
            "format": TUNE_FORMAT,
            "base": base_id,
            "label": label,
            "decoder": self.spec.decoder.name(),
            "connector": self.spec.connector,
            "vision": self.spec.vision.name(),
            "downsample": self.spec.downsample.name(),
            "steps": self.steps,
        });
        std::fs::write(dir.join(TUNE_MANIFEST), serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", dir.display()))
    }
}

/// The tune directory's manifest and its format tag.
pub const TUNE_MANIFEST: &str = "tune.json";
pub const TUNE_FORMAT: &str = "minicpmv-tune-1";
pub const TUNE_DECODER_ADAPTER: &str = "decoder.adapter.safetensors";
pub const TUNE_DECODER_FULL: &str = "decoder.safetensors";
pub const TUNE_VISION: &str = "vision.safetensors";

/// Apply a tune directory's vision side to `weights` (canonical names, in
/// place): its trained tensors replace the base's, its LoRA factors fold in
/// as `W += (alpha/rank) B A`.
pub fn apply_vision_tune(weights: &mut HashMap<String, Vec<f32>>, tune: &Path) -> Result<(), String> {
    let path = tune.join(TUNE_VISION);
    if !path.is_file() {
        return Ok(());
    }
    let p = path.to_str().ok_or("non-UTF-8 tune path")?;
    let st = checkpoint::st::load_safetensors(p).map_err(|e| format!("{p}: {e}"))?;
    let cfg = st.config();
    let tensors: HashMap<String, Vec<f32>> = st.tensors.into_iter().collect();
    for (name, v) in &tensors {
        if name.ends_with(".lora_a") || name.ends_with(".lora_b") {
            continue;
        }
        let slot = weights.get_mut(name).ok_or_else(|| format!("{p}: {name} is not a vision tensor of this base"))?;
        if slot.len() != v.len() {
            return Err(format!("{p}: {name} holds {} values, the base's {}", v.len(), slot.len()));
        }
        slot.copy_from_slice(v);
    }
    if let (Some(rank), Some(alpha)) = (cfg["lora_rank"].as_u64(), cfg["lora_alpha"].as_f64()) {
        let scale = alpha as f32 / rank as f32;
        for (name, a) in tensors.iter().filter(|(n, _)| n.ends_with(".lora_a")) {
            let base = name.trim_end_matches(".lora_a");
            let b = tensors.get(&format!("{base}.lora_b")).ok_or_else(|| format!("{p}: {name} has no lora_b"))?;
            let w = weights.get_mut(base).ok_or_else(|| format!("{p}: {base} is not a vision tensor of this base"))?;
            model::lora::device_adapter::fold_delta(w, a, b, rank as usize, scale);
        }
    }
    Ok(())
}
