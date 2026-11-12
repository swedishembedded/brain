// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resident low-precision full fine-tune of Ripple: its linears and its two tables live on the device in bf16 or
//! FP8 with their master copy and optimizer state off it, optionally streamed through a window of layers
//! ([`model::resident_ft`]); the norms and the value-residual gates stay in the parameter store under the device
//! optimizer.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, with the master copy and the optimizer state where the memory is. If your team needs expertise in
//! training every weight of a model that does not fit an accelerator then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! What a model adds to the generic state is small: which tensors are resident ([`resident_specs`]), what the rest
//! of the step needs the device optimizer for ([`ResidentBackward`]), the layer boundaries of its recorded passes
//! (the segments of [`model::segmented::SegmentedPass`]) and the commit and restore of the off-device state
//! ([`OffDeviceState`]).

use model::memplan::{Capacity, Category, Demand};
use model::resident_ft::{LayerEvent, ResidentBackward, ResidentFtConfig, ResidentPlan, ResidentSpec, StepArgs};
use model::segmented::SegmentEdge;
use model::Shard;

use super::{Ripple, TrainConfig};

/// The tensors of `shard` a resident full fine-tune holds in its own state: every linear of its layers (with `g_proj`)
/// in the run's encoding, then the embedding table on the stages that read it and the untied head on the head stage,
/// in bf16. The norms and the gates stay in the parameter store.
pub fn resident_specs(cfg: &TrainConfig, shard: &Shard) -> Vec<ResidentSpec> {
    let c = &cfg.model;
    let mut specs: Vec<ResidentSpec> = (shard.start..shard.end).flat_map(|l| cfg.layer_linears(l).into_iter().map(move |(_, name, n, k)| ResidentSpec::linear(name, n as u32, k as u32, l as u32))).collect();
    if cfg.shard_owns("model.embed_tokens.weight", shard) {
        specs.push(ResidentSpec::table("model.embed_tokens.weight", c.vocab_size as u32, c.hidden_size as u32));
    }
    if shard.head {
        specs.push(ResidentSpec::table("lm_head.weight", c.vocab_size as u32, c.hidden_size as u32));
    }
    specs
}

/// Refuse, before anything is allocated, a resident full fine-tune of `shard` the device cannot hold: the planner's
/// arithmetic over the resident weights, gradients, host-side master and state, and the demands of the norms and the
/// activations.
pub(super) fn check_resident_fits(cfg: &TrainConfig, shard: &Shard, plan: &ResidentPlan) -> Result<model::memplan::PlanReport, String> {
    model::resident_ft::check_fits(&plan.cfg, &resident_specs(cfg, shard), &plan.budget())
}

/// The plan of a resident full fine-tune of `shard` at `b` rows of `t` tokens against `capacity`: the resident
/// state under `resident`, the f32 norms and gates with their gradients and both moments, and the activations of
/// the stage's layers (the build's own count, the boundaries held in f32).
pub fn plan_resident(cfg: &TrainConfig, shard: &Shard, b: u32, t: u32, resident: &ResidentFtConfig, capacity: Capacity) -> ResidentPlan {
    let resident_names: std::collections::HashSet<String> = resident_specs(cfg, shard).into_iter().map(|s| s.name).collect();
    let remainder: u64 = cfg.shard_param_list(shard).into_iter().filter(|(name, _)| !resident_names.contains(name)).map(|(_, numel)| numel as u64).sum();
    let mut stage = cfg.clone();
    stage.model.num_hidden_layers = shard.end - shard.start;
    let activations = super::graph::activation_bytes(&stage, b, t, model::saved::SavedFormat::F32, true);
    ResidentPlan {
        cfg: resident.clone(),
        capacity,
        others: vec![Demand::fixed(Category::Weights, "f32 norms and gates (with gradients and moments)", remainder * 16), Demand::fixed(Category::Activations, format!("activations of {b} x {t} tokens"), activations)],
    }
}

impl Ripple {
    /// Whether this build streams its layers through a bounded window of device slots.
    pub(super) fn layer_streamed(&self) -> bool {
        self.resident.as_ref().is_some_and(|ft| ft.borrow().is_layer_streamed())
    }

    /// Load `layer` into its device slots.
    pub(super) fn stream_load(&self, layer: usize) {
        let mut ft = self.resident.as_ref().expect("layer-streamed builds are resident").borrow_mut();
        self.gpu.flush();
        ft.load_layer(&self.ops, layer).unwrap_or_else(|e| panic!("ricursive layer-streamed full fine-tune: loading layer {layer}: {e}"));
    }

    /// The index of `name` in the resident state, `None` for a tensor of the store.
    pub(super) fn resident_index(&self, name: &str) -> Option<usize> {
        self.resident.as_ref().and_then(|ft| ft.borrow().index_of(name))
    }

    /// The gradient of resident tensor `name` as f32, off the accumulating buffer (bf16 gradients unpacked).
    pub(super) fn read_resident_grad(&self, name: &str) -> Option<Vec<f32>> {
        use model::ops::GradDtype;
        let ft = self.resident.as_ref()?.borrow();
        let i = ft.index_of(name)?;
        let spec = &ft.specs()[i];
        let numel = spec.n as usize * spec.k as usize;
        self.gpu.flush();
        Some(match ft.grad_dtype() {
            GradDtype::F32 => self.gpu.read(ft.grad_buffer(i), numel),
            GradDtype::Bf16 => self.gpu.read(ft.grad_buffer(i), numel / 2).into_iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xffff_0000)]).collect(),
        })
    }

    /// Device bytes of the resident linears' weights: every layer's, or the window's slots when layer-streamed.
    pub fn resident_device_weight_bytes(&self) -> Option<u64> {
        self.resident.as_ref().map(|ft| ft.borrow().device_weight_bytes())
    }
}

impl ResidentBackward for Ripple {
    fn flush_backward(&self) {
        self.gpu.flush();
    }

    fn zero_remainder_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    fn remainder_grad_sum_sq(&self, norm: &optim::gradnorm::DeviceGradNorm) -> f64 {
        let items: Vec<optim::gradnorm::NormItem<'_>> = self.ps.opt_params().iter().map(|(name, numel)| optim::gradnorm::NormItem { buf: self.ps.g(name), numel: *numel, dtype: model::ops::GradDtype::F32 }).collect();
        norm.sum_sq(&self.gpu, &items)
    }

    /// The recorded backward: segment 0 the head and final norm, then one segment per layer from the last down, then the
    /// embedding.
    fn backward_pass(&self, on: &mut dyn FnMut(LayerEvent) -> std::io::Result<()>) -> std::io::Result<()> {
        // The tape's hook cannot fail: the first error is kept and the remaining segments report nothing.
        let mut failed: Option<std::io::Error> = None;
        self.bwd.submit_hooked(&self.gpu, &mut |edge| {
            let event = match edge {
                SegmentEdge::Before { layer: Some(l), .. } => LayerEvent::BeforeBackward(l),
                SegmentEdge::After { layer: Some(l), .. } => LayerEvent::AfterBackward(l),
                SegmentEdge::After { index: 0, layer: None } => LayerEvent::AfterHead,
                _ => return,
            };
            if failed.is_none() {
                failed = on(event).err();
            }
        });
        failed.map_or(Ok(()), Err)
    }

    fn update_remainder(&self, args: &StepArgs, grad_multiplier: f32) {
        let StepArgs { step, lr, wd, adam, .. } = *args;
        let opt = self.opt.as_ref().expect("a resident build trains");
        opt.step(&self.gpu, &self.ps, step, lr, wd, adam, None, grad_multiplier);
    }
}

impl model::OffDeviceState for Ripple {
    fn commit(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let ckpt = optim::checkpoint::CheckpointDir::new(dir)?;
        let ft = self.resident.as_ref().ok_or_else(|| std::io::Error::other("ricursive: no resident state to commit"))?;
        self.gpu.flush();
        let mut ft = ft.borrow_mut();
        let run = ft.run_state(0, Vec::new());
        if run.step != step {
            return Err(std::io::Error::other(format!("the run has absorbed {step} updates, the resident state {}", run.step)));
        }
        ft.commit(&self.ops, &ckpt, &run).map(drop)
    }

    fn restore(&self, dir: &std::path::Path, step: u32) -> std::io::Result<()> {
        let ckpt = optim::checkpoint::CheckpointDir::new(dir)?;
        let ft = self.resident.as_ref().ok_or_else(|| std::io::Error::other("ricursive: no resident state to restore"))?;
        self.gpu.flush();
        if ft.borrow_mut().restore_at(&self.ops, &ckpt, step)?.is_some() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("{}: no training state was committed at update {step}", dir.display())))
        }
    }
}

/// What one stage of a resident full fine-tune asks of the machine, in bytes: the planner's own categories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageCost {
    pub layers: std::ops::Range<usize>,
    /// Device: the resident weights (every layer's, or the window's slots with the tables).
    pub weights: u64,
    /// Device: the accumulating gradients (every tensor's, or one layer's lanes with the tables').
    pub grads: u64,
    /// Device: the f32 norms and gates with their gradients and both moments.
    pub norms: u64,
    /// Device: the activations of the stage's layers at the planned batch (checkpointed: the boundary residuals and one
    /// layer's working set).
    pub activations: u64,
    /// Host: the master copy, the optimizer state and, for a layer-streamed stage, the stored values.
    pub host: u64,
}

impl StageCost {
    pub fn device(&self) -> u64 {
        self.weights + self.grads + self.norms + self.activations
    }
}

/// The cost of each of `shards` (a layout of `cfg`) for a resident full fine-tune under `resident` at `b` rows of `t`
/// tokens: the planner's demands, not an allocation. A planning figure with the bounds of one: the weights, gradients and
/// host state are exact for the tensors the state holds; the activations are the build's own count.
pub fn stage_costs(cfg: &TrainConfig, shards: &[Shard], b: u32, t: u32, resident: &ResidentFtConfig) -> Result<Vec<StageCost>, String> {
    shards
        .iter()
        .map(|shard| {
            let plan = plan_resident(cfg, shard, b, t, resident, Capacity { device_bytes: u64::MAX, host_bytes: u64::MAX });
            let demands = model::resident_ft::demands(&plan.cfg, &resident_specs(cfg, shard)).map_err(|e| e.to_string())?;
            let of = |category: Category| demands.iter().find(|d| d.category == category).map(|d| (d.variants[0].device_bytes, d.variants[0].host_bytes)).unwrap_or((0, 0));
            let other = |category: Category| plan.others.iter().find(|d| d.category == category).map_or(0, |d| d.variants[0].device_bytes);
            Ok(StageCost { layers: shard.start..shard.end, weights: of(Category::Weights).0, grads: of(Category::Grads).0, norms: other(Category::Weights), activations: other(Category::Activations), host: of(Category::OptimizerState).1 })
        })
        .collect()
}
