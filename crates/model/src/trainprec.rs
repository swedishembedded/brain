// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models for its clients. If your
// team needs expertise in choosing and verifying the optimizer-state precision of a training run against a fixed
// accelerator budget, you can procure our services by sending an email to info@swedishembedded.com.

//! The training precision of a full fine-tune as ONE typed value, carried by [`crate::FitOpts::precision`].
//!
//! It replaces the thread-scoped policy a build used to read (a value set on the calling thread and read several
//! calls below, which two concurrent builds could not hold apart): a trainer hands [`TrainPrecision`] to the
//! model's builder, which resolves it parameter by parameter with [`TrainPrecision::resolve`] and returns an
//! `Err` naming the parameter when a choice is not one the trainer runs. Nothing is picked silently: the optimizer
//! state is narrowed by the memory planner only when the run authorized it ([`TrainPrecision::narrow_state`]), and
//! the report of that decision comes back with the effective precision.
//!
//! The textual form is model agnostic and parsed once, `state=int8,master=f32,value=bf16,grad=bf16,sr=on,narrow=on`
//! ([`TrainPrecision::parse`]); `weights=fp8,window=2,clip=two-pass` additionally asks for the per-layer linears to
//! live in a low-precision encoding with an off-device master ([`ResidentRequest`]), optionally streamed through a
//! window of layers. Every model declares what it supports ([`TrainPrecisionSupport`]) and refuses the rest by name.

use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, PrecisionPolicy, StateFormat, ValueDtype};

use crate::memplan::{self, Authorized, Capacity, Category, Lever, PlanReport, TrainingFootprint};
use crate::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentValue};
use optim::rule::Rule;

/// What a full fine-tune stores each trainable parameter as, and whether the planner may narrow its optimizer state.
#[derive(Clone, Debug)]
pub struct TrainPrecision {
    pub policy: PrecisionPolicy,
    /// Let the memory planner pick a narrower optimizer-state format (bf16, then int8 blocks) when the requested one
    /// does not fit the device. The decision is reported; without this a run that does not fit is refused.
    pub narrow_state: bool,
    /// Hold the per-layer linears in a low-precision encoding with an off-device master ([`crate::resident_ft`]).
    pub resident: Option<ResidentRequest>,
    /// Keep the optimizer state of the parameter store in host memory (`moments=host`): the device holds each weight
    /// and its gradient only. A state format only the host optimizer runs (log8, int8 blocks of another size) implies
    /// it.
    pub host_moments: bool,
    /// What the backward keeps of the forward (`activations=keep|recompute|bf16|fp8`): every layer's intermediates,
    /// or only the residual between layers (in f32, bf16 or fp8) with each layer replayed before its backward.
    /// `None` leaves the build's own choice.
    pub activations: Option<crate::saved::SavedPolicy>,
}

/// How the per-layer linears of a full fine-tune are held: their encoding and, when the model is larger than the
/// device, the window of layers kept on it (the rest streams through, see [`crate::resident_ft::LayerWindow`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentRequest {
    pub weights: ResidentValue,
    /// Layers resident at once; `None` keeps every layer on the device.
    pub window: Option<u32>,
    /// How a layer-streamed step clips by the global gradient norm.
    pub clip_pass: ClipPass,
}

impl Default for TrainPrecision {
    /// The historical layout: everything fp32, no master, no narrowing.
    fn default() -> Self {
        TrainPrecision { policy: PrecisionPolicy::uniform(ParamPrecision::FP32), narrow_state: false, resident: None, host_moments: false, activations: None }
    }
}

impl TrainPrecision {
    pub fn uniform(p: ParamPrecision) -> TrainPrecision {
        TrainPrecision { policy: PrecisionPolicy::uniform(p), narrow_state: false, resident: None, host_moments: false, activations: None }
    }

    /// Whether the parameter store's optimizer state lives in host memory: asked for (`moments=host`), or a state
    /// format only the host optimizer runs (log8, int8 blocks of another size than the device kernels').
    pub fn wants_host_moments(&self) -> bool {
        let device_runs = |s: StateFormat| matches!(s, StateFormat::F32 | StateFormat::Bf16) || s == StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK };
        self.host_moments || !device_runs(self.policy.default.state) || self.policy.rules.iter().any(|(_, p)| !device_runs(p.state))
    }

    /// Whether this is the default (all fp32): a resume file only records a precision that departs from it.
    pub fn is_default(&self) -> bool {
        self.policy.rules.is_empty() && self.policy.default == ParamPrecision::FP32 && !self.narrow_state && self.resident.is_none() && !self.host_moments && self.activations.is_none()
    }

    /// The resident full fine-tune this precision asks for, `None` for a plain one: the encoding and window of the
    /// linears, the master copy (an f32 master unless one was named: a low-precision weight cannot be trained without
    /// one), the optimizer state and gradient dtype of the policy, AdamW with the run's hyper-parameters set at each
    /// step. `seed` keys the stochastic rounding.
    pub fn resident_config(&self, seed: Option<u64>) -> Result<Option<ResidentFtConfig>, String> {
        let Some(r) = self.resident else { return Ok(None) };
        if !self.policy.rules.is_empty() {
            return Err("a resident full fine-tune takes one uniform precision (no per-group rules)".into());
        }
        let p = self.policy.default;
        // An f32 weight is its own master; a narrower one keeps an f32 master unless the policy names one.
        let master = match (r.weights, p.master) {
            (ResidentValue::F32, master) => master,
            (_, MasterPolicy::None) => MasterPolicy::F32,
            (_, master) => master,
        };
        Ok(Some(ResidentFtConfig {
            value: r.weights,
            master,
            state: p.state,
            rule: Rule::AdamW { adam: crate::Adam::default(), wd: 0.0 },
            grad: p.grad,
            stochastic_state: p.stochastic_rounding,
            seed,
            tile_elems: 1 << 22,
            state_dir: None,
            defer_publication: false,
            window: r.window.map(|layers| LayerWindow { layers }),
            clip_pass: r.clip_pass,
        }))
    }

    /// The precision of the tensors a resident full fine-tune leaves in the parameter store (the norms): the run's
    /// precision governs the resident linears and tables through [`Self::resident_config`], while the store tensors
    /// train as the store runs them, f32 value and gradient with no master, their moments in the run's state format.
    /// A resident run's other knobs (`value`, `grad`, `master`, `sr`) therefore do not reach the store; a state format
    /// the device optimizer does not run is still refused by [`Self::resolve`], by parameter name.
    pub fn resident_store(&self) -> TrainPrecision {
        let store = ParamPrecision { state: self.policy.default.state, ..ParamPrecision::FP32 };
        TrainPrecision { policy: PrecisionPolicy::uniform(store), ..self.clone() }
    }

    /// Parse `key=value,...` with keys `state` (`f32`, `bf16`, `int8[:block]`, `log8[:block]`), `master`
    /// (`none`, `bf16`, `f32`), `value` and `grad` (`f32`, `bf16`), `sr` and `narrow` (`on`, `off`), `moments` (`device`,
    /// `host`: where the parameter store's optimizer state lives), `activations` (`keep`, `recompute`, `bf16`, `fp8`:
    /// what the backward keeps of the forward), and for a resident
    /// full fine-tune `weights` (`f32`, `bf16`, `fp8[:block]`, `int8`, `nf4`), `window` (layers resident at once) and `clip`
    /// (`two-pass`, `running`). Unnamed keys keep the fp32 default. The result is uniform over every parameter.
    pub fn parse(spec: &str) -> Result<TrainPrecision, String> {
        let mut p = ParamPrecision::FP32;
        let mut narrow_state = false;
        let mut host_moments = false;
        let mut activations = None;
        let (mut weights, mut window, mut clip_pass) = (None, None, ClipPass::TwoPass);
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (key, value) = item.split_once('=').ok_or_else(|| format!("train precision '{item}': expected key=value"))?;
            let flag = |v: &str| match v {
                "on" | "true" => Ok(true),
                "off" | "false" => Ok(false),
                other => Err(format!("train precision '{key}': '{other}' is not on or off")),
            };
            match key {
                "state" => p.state = parse_state(value)?,
                "master" => {
                    p.master = match value {
                        "none" => MasterPolicy::None,
                        "bf16" => MasterPolicy::Bf16,
                        "f32" => MasterPolicy::F32,
                        other => return Err(format!("train precision 'master': '{other}' is not none, bf16 or f32")),
                    }
                }
                "value" => {
                    p.value = match value {
                        "f32" => ValueDtype::F32,
                        "bf16" => ValueDtype::Bf16,
                        other => return Err(format!("train precision 'value': '{other}' is not f32 or bf16")),
                    }
                }
                "grad" => {
                    p.grad = match value {
                        "f32" => GradDtype::F32,
                        "bf16" => GradDtype::Bf16,
                        other => return Err(format!("train precision 'grad': '{other}' is not f32 or bf16")),
                    }
                }
                "sr" => p.stochastic_rounding = flag(value)?,
                "narrow" => narrow_state = flag(value)?,
                "activations" => {
                    use crate::saved::{SavedFormat, SavedPolicy};
                    activations = Some(match value {
                        "keep" => SavedPolicy::Keep,
                        "recompute" => SavedPolicy::Recompute,
                        "bf16" => SavedPolicy::Quantized(SavedFormat::Bf16),
                        "fp8" => SavedPolicy::Quantized(SavedFormat::Fp8),
                        other => return Err(format!("train precision 'activations': '{other}' is not keep, recompute, bf16 or fp8")),
                    })
                }
                "moments" => {
                    host_moments = match value {
                        "host" => true,
                        "device" => false,
                        other => return Err(format!("train precision 'moments': '{other}' is not host or device")),
                    }
                }
                "weights" => weights = Some(parse_resident_value(value)?),
                "window" => {
                    window = Some(value.parse::<u32>().ok().filter(|&n| n > 0).ok_or_else(|| format!("train precision 'window': '{value}' is not a layer count of at least 1"))?)
                }
                "clip" => {
                    clip_pass = match value {
                        "two-pass" => ClipPass::TwoPass,
                        "running" => ClipPass::Running,
                        other => return Err(format!("train precision 'clip': '{other}' is not two-pass or running")),
                    }
                }
                other => return Err(format!("train precision: unknown key '{other}' (state, master, value, grad, sr, narrow, moments, activations, weights, window, clip)")),
            }
        }
        if weights.is_none() && (window.is_some() || item_named(spec, "clip")) {
            return Err("train precision: 'window' and 'clip' describe a resident full fine-tune and need 'weights' (f32, bf16, fp8, int8 or nf4)".into());
        }
        let resident = weights.map(|weights| ResidentRequest { weights, window, clip_pass });
        Ok(TrainPrecision { policy: PrecisionPolicy::uniform(p), narrow_state, resident, host_moments, activations })
    }

    /// The precision of each of `params` under this policy, checked against what the trainer runs. The device
    /// training paths keep f32 weights and gradients with no master copy and run AdamW state in f32, bf16 or
    /// 64-element int8 blocks (the fused kernels); with the state offloaded to the host any state format is
    /// available. Anything else is an `Err` naming the parameter, never a silent fp32 fallback.
    pub fn resolve(&self, params: &[(String, usize)], host_offload: bool) -> Result<Vec<ParamPrecision>, String> {
        params
            .iter()
            .map(|(name, numel)| {
                let p = self.policy.resolve(name, *numel).precision;
                if p.value != ValueDtype::F32 || p.grad != GradDtype::F32 || p.master != MasterPolicy::None {
                    return Err(format!("{name}: this trainer keeps f32 device weights and gradients with no master copy; got {p:?}"));
                }
                let device_ok = matches!(p.state, StateFormat::F32 | StateFormat::Bf16) || p.state == (StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK });
                if !host_offload && !device_ok {
                    return Err(format!(
                        "{name}: the device optimiser keeps f32 or bf16 moments or int8 blocks of {}; {:?} moments need the host-offloaded optimizer state",
                        paramstore::DEVICE_INT8_BLOCK, p.state
                    ));
                }
                Ok(p)
            })
            .collect()
    }

    /// This precision with the optimizer state in `state` for every parameter (a uniform policy only).
    fn with_state(&self, state: StateFormat) -> TrainPrecision {
        TrainPrecision { policy: PrecisionPolicy::uniform(ParamPrecision { state, ..self.policy.default }), narrow_state: self.narrow_state, resident: self.resident, host_moments: self.host_moments, activations: self.activations }
    }

    /// Plan the device footprint of a full fine-tune before anything loads. The optimizer state is held in the
    /// requested format when it fits; when it does not, a run that authorized narrowing gets the first narrower
    /// format that does (reported in the returned [`PlanReport`]) and any other run gets the planner's refusal with
    /// its arithmetic. Only a uniform policy can be narrowed.
    pub fn plan(&self, footprint: &TrainingFootprint, capacity: Capacity) -> Result<PlannedPrecision, String> {
        let authorized = if self.narrow_state { Authorized::only(&[Lever::OptimizerState]) } else { Authorized::none() };
        if self.narrow_state && !self.policy.rules.is_empty() {
            return Err("narrowing the optimizer state needs a uniform precision policy (no per-group rules)".into());
        }
        let footprint = TrainingFootprint { precision: self.policy.default, ..footprint.clone() };
        let demands = memplan::training_demands(&footprint).map_err(|e| e.to_string())?;
        let report = memplan::plan(&demands, capacity, &authorized).map_err(|e| format!("the full fine-tune does not fit: {e}"))?;
        let ladder = paramstore::precision::optimizer_state_ladder(&footprint.trainable, footprint.moment_slots, footprint.int8_block);
        let chosen = report.variant(Category::OptimizerState).map(|v| v.label.as_str()).unwrap_or_default();
        let state = ladder.iter().map(|(f, _)| *f).find(|f| chosen == format!("{f:?}")).unwrap_or(self.policy.default.state);
        Ok(PlannedPrecision { precision: if state == self.policy.default.state { self.clone() } else { self.with_state(state) }, report })
    }
}

/// A precision the planner accepted, with the report of what it decided.
#[derive(Clone, Debug)]
pub struct PlannedPrecision {
    pub precision: TrainPrecision,
    pub report: PlanReport,
}

/// The optimizer-state formats a model's plain (fp32 weights) full fine-tune runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateSupport {
    /// Only f32.
    Fp32,
    /// f32, bf16 or int8 blocks of [`paramstore::DEVICE_INT8_BLOCK`]: what the fused device kernels run.
    DeviceFused,
    /// Also the host-streamed formats (any block size, log8): the state is offloaded to host memory.
    Any,
}

/// A model's parameter store and the device optimizer that updates it ([`crate::Model::optimizer_state`]).
pub struct OptimizerState<'a> {
    pub gpu: &'a gpu_core::Gpu,
    pub store: &'a mut paramstore::ParamStore,
    pub optim: &'a optim::Optim,
}

/// [`OptimizerState`] read-only ([`crate::Model::optimizer_parts`]).
pub struct OptimizerParts<'a> {
    pub gpu: &'a gpu_core::Gpu,
    pub store: &'a paramstore::ParamStore,
    pub optim: &'a optim::Optim,
}

/// Move `model`'s optimizer state to host memory, so the device holds only its weights and gradients: for a model
/// whose updates another optimizer runs (a data-parallel replica, a pipeline stage), whose device moments would
/// otherwise sit allocated and unread. A no-op for a model with no [`OptimizerState`].
pub fn release_device_moments<M: crate::Model>(model: &mut M) {
    if let Some(state) = model.optimizer_state() {
        state.store.offload_all();
        state.optim.invalidate();
    }
}

/// Bring `model`'s optimizer state to what `precision` asks, before the first optimizer step: every trainable
/// parameter of its store gets its moments in the requested format (f32, bf16 or int8 blocks on the device), the
/// update graph is recorded again, and a model whose build already holds that format is left as it is. A request
/// the model cannot honour is an error naming the model or the parameter, never a silent fp32 run: a non-default
/// state for a model with no [`OptimizerState`], a format the device optimizer does not run, a moment kernel the
/// model did not register ([`optim::MOMENT_KERNELS`]), or resident low-precision weights for a build that keeps no
/// off-device state.
pub fn apply_train_precision<M: crate::Model>(model: &mut M, precision: &TrainPrecision, name: &str) -> Result<(), String> {
    if precision.is_default() {
        return Ok(());
    }
    if let Some(policy) = precision.activations {
        model.set_saved_policy(policy).map_err(|e| format!("{name}: {e}"))?;
    }
    let resident_store;
    let precision = match precision.resident_config(None)? {
        Some(cfg) => {
            // A model built with its own resident state keeps it; any other streams its layers now.
            if model.off_device_state().is_none() {
                if precision.wants_host_moments() {
                    return Err(format!("{name}: a layer-streamed build keeps its streamed tensors' moments on the host already; the rest of its store keeps its moments on the device"));
                }
                model.set_resident(&cfg).map_err(|e| format!("{name}: {e}"))?;
            }
            // What is left in the store trains f32 under the run's state format.
            resident_store = precision.resident_store();
            &resident_store
        }
        None => precision,
    };
    let uniform_f32 = precision.policy.rules.is_empty() && precision.policy.default == ParamPrecision::FP32;
    let Some(state) = model.optimizer_state() else {
        return if uniform_f32 { Ok(()) } else { Err(format!("{name}: this model does not expose its optimizer state, so it trains only with f32 moments")) };
    };
    let wanted: Vec<(String, ParamPrecision)> = state.store.trainable.iter().chain(&state.store.offload).map(|(n, numel)| (n.clone(), precision.policy.resolve(n, *numel).precision)).collect();
    if precision.wants_host_moments() || !state.store.offload.is_empty() {
        // The host optimizer keeps the moments: the device holds the weight and its gradient only.
        if let Some((n, p)) = wanted.iter().find(|(_, p)| p.value != ValueDtype::F32 || p.master != MasterPolicy::None) {
            return Err(format!("{name}: {n}: the host optimizer updates the f32 device weight with no master copy; got {p:?}"));
        }
        state.store.offload_all();
        for (n, p) in wanted {
            state.store.set_precision(&n, p);
        }
        state.optim.invalidate();
        return Ok(());
    }
    let formats = state.store.trainable.iter().map(|(n, numel)| precision.policy.resolve(n, *numel).precision.state);
    let missing = optim::missing_moment_kernels(state.gpu, formats);
    if !missing.is_empty() {
        return Err(format!("{name}: the requested optimizer state needs the {} kernel(s), which this model does not register", missing.join(", ")));
    }
    let changed = state.store.apply_state_policy(state.gpu, &precision.policy).map_err(|e| format!("{name}: {e}"))?;
    if changed > 0 {
        state.optim.invalidate();
    }
    Ok(())
}

/// Every named parameter's values, in name order.
type NamedValues = Vec<(String, Vec<f32>)>;

/// The optimizer-state contract every model with an [`OptimizerState`] honours, for its own test to run: `build`
/// makes a fresh trainable model (the same initial weights every call) and `step` runs one optimizer step on a fixed
/// batch (`zero_grads`, the batch, `forward`, `backward`, `adamw_step` at learning rate `lr`).
///
/// * bf16 moments and int8 blocks are held on the device for every trainable tensor;
/// * `moments=host` and the host-only log8 format move every tensor's moments to host memory: no moment buffer is
///   left on the device;
/// * the first update in each layout equals the f32 run's within the format's quantization: a first AdamW step
///   moves every weight by about `lr * sign(g)` whatever holds the moments, so the two may differ by 5% of `lr`;
/// * a precision no optimizer runs (a bf16 value without a master) is refused by parameter name and leaves the store
///   as it was.
pub fn check_optimizer_state_formats<M: crate::Model>(name: &str, build: impl Fn() -> M, step: impl Fn(&M), lr: f32) -> Result<(), String> {
    let snapshot = |m: &M| -> NamedValues {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
    };
    let first_update = |precision: &TrainPrecision| -> Result<(NamedValues, M), String> {
        let mut m = build();
        apply_train_precision(&mut m, precision, name)?;
        let before = snapshot(&m);
        step(&m);
        let after = snapshot(&m);
        Ok((before.into_iter().zip(after).map(|((n, b), (_, a))| (n, a.iter().zip(&b).map(|(x, y)| x - y).collect())).collect(), m))
    };
    let (want, _) = first_update(&TrainPrecision::default())?;
    let int8 = StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK };
    let layouts = [("state=bf16", StateFormat::Bf16, false), ("state=int8", int8, false), ("moments=host", StateFormat::F32, true), ("state=log8", StateFormat::Log8Block { block: 256 }, true)];
    for (spec, state, host) in layouts {
        let (got, mut m) = first_update(&TrainPrecision::parse(spec)?)?;
        let st = m.optimizer_state().ok_or_else(|| format!("{name}: no optimizer state"))?;
        let (held, other) = if host { (&st.store.offload, &st.store.trainable) } else { (&st.store.trainable, &st.store.offload) };
        if held.is_empty() || !other.is_empty() {
            return Err(format!("{name} {spec}: {} tensors hold moments where asked, {} elsewhere", held.len(), other.len()));
        }
        if host && !st.store.adam_m.is_empty() {
            return Err(format!("{name} {spec}: {} moment buffers stayed on the device", st.store.adam_m.len()));
        }
        if let Some((n, _)) = held.iter().find(|(n, _)| std::mem::discriminant(&st.store.precision_of(n).state) != std::mem::discriminant(&state)) {
            return Err(format!("{name} {spec}: {n} keeps {:?} moments", st.store.precision_of(n).state));
        }
        // Where the moments live changes no parameter the model lists, and the comparison must cover them all.
        let names = |v: &NamedValues| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        if want.is_empty() || names(&want) != names(&got) {
            return Err(format!("{name} {spec}: lists {} trained parameters, the f32 run {}", got.len(), want.len()));
        }
        for ((n, w), (_, g)) in want.iter().zip(&got) {
            if let Some((i, (a, b))) = w.iter().zip(g).enumerate().find(|(_, (a, b))| (*a - *b).abs() > 0.05 * lr) {
                return Err(format!("{name} {spec}: {n}[{i}] moved by {b}, the f32 run by {a}"));
            }
        }
    }
    let mut m = build();
    let refusal = apply_train_precision(&mut m, &TrainPrecision::parse("value=bf16")?, name).err().ok_or_else(|| format!("{name}: a bf16 value without a master was accepted"))?;
    let st = m.optimizer_state().ok_or_else(|| format!("{name}: no optimizer state"))?;
    if !st.store.trainable.iter().any(|(n, _)| refusal.contains(n.as_str())) {
        return Err(format!("{name}: the refusal names no parameter: {refusal}"));
    }
    if let Some((n, _)) = st.store.trainable.iter().find(|(n, _)| st.store.precision_of(n) != ParamPrecision::FP32) {
        return Err(format!("{name}: the refused request still changed {n}"));
    }
    Ok(())
}

/// The exact-resume contract of a model whose optimizer state lives in host memory, for its own test to run
/// ([`check_exact_resume`] under `moments=host`).
pub fn check_resume_with_host_moments<M: crate::Model>(name: &str, build: impl Fn() -> M, block: u32, scratch: &std::path::Path) -> Result<(), String> {
    check_exact_resume(name, build, block, scratch, "moments=host")
}

/// The exact-resume contract of a run whose state lives partly off the device (`precision`: host moments, or layers
/// streamed through a window), for a model's own test to run: a run of six steps, stopped after three and resumed
/// from its state file in a fresh model, ends with the loss and every weight of the uninterrupted run, bit for bit.
/// `build` makes a fresh trainable model (the same initial weights every call) for a vocabulary of at least 16, a
/// batch of one and a block of `block` tokens; `scratch` is an empty directory the run may write.
pub fn check_exact_resume<M: crate::Model>(name: &str, build: impl Fn() -> M, block: u32, scratch: &std::path::Path, precision: &str) -> Result<(), String> {
    use data::binio::{self, Meta};
    let err = |e: std::io::Error| format!("{name}: {e}");
    // A masked prompt/answer corpus over tokens 2..=15.
    let (mut tokens, mut mask) = (Vec::new(), Vec::new());
    for i in 0..200u32 {
        tokens.extend_from_slice(&[2 + i % 5, 7, 9, 15]);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    binio::write_u32_bin(&scratch.join("train.u32.bin"), &tokens).map_err(err)?;
    binio::write_mask_bin(&scratch.join("train.mask.bin"), &mask).map_err(err)?;
    binio::write_u32_bin(&scratch.join("val.u32.bin"), &[]).map_err(err)?;
    binio::write_mask_bin(&scratch.join("val.mask.bin"), &[]).map_err(err)?;
    std::fs::write(scratch.join("meta.json"), Meta::vocab_only(16)).map_err(err)?;
    let opts = crate::FitOpts {
        steps: 6,
        batch_size: 1,
        block_size: block,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup: 2,
        decay_iters: 6,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        // One drawn seed for the three runs: the comparison is between them, not against a fixed trajectory.
        seed: data::rng::random_seed(),
        precision: TrainPrecision::parse(precision)?,
        ..Default::default()
    };
    let objective = || -> Result<_, String> {
        let (train, val, bcfg, _, itos) = crate::load_dataset_with_itos(scratch, &opts).map_err(err)?;
        Ok(crate::causal_lm::<M>(train, val, bcfg, itos))
    };
    let weights = |m: &M| -> Vec<(String, Vec<u32>)> {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), m.read_weight(&n).into_iter().map(f32::to_bits).collect())).collect()
    };
    let (straight, straight_model) = crate::fit_controlled(build(), objective()?, &opts, None, crate::FitControl::default()).map_err(err)?;
    let state = scratch.join("run.state");
    let identity = serde_json::json!({"test": name});
    let mut stop_at_3 = |r: &crate::StepReport| r.step < 3;
    let control = crate::FitControl { on_step: Some(&mut stop_at_3), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = crate::fit_controlled(build(), objective()?, &opts, None, control).map_err(err)?;
    if !first.interrupted || !state.with_extension("offdevice").is_dir() {
        return Err(format!("{name}: the stopped run left no resume state with its off-device generation"));
    }
    let control = crate::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = crate::fit_controlled(build(), objective()?, &opts, None, control).map_err(err)?;
    if second.resumed_at != Some(3) {
        return Err(format!("{name}: resumed at {:?}, not 3", second.resumed_at));
    }
    let (a, b) = (straight.final_loss.unwrap_or(f32::NAN), second.final_loss.unwrap_or(f32::NAN));
    if a.to_bits() != b.to_bits() {
        return Err(format!("{name}: the resumed run ends at loss {b}, the uninterrupted one at {a}"));
    }
    if let Some(((n, _), _)) = weights(&straight_model).into_iter().zip(weights(&resumed)).find(|((_, x), (_, y))| x != y) {
        return Err(format!("{name}: {n} differs between the resumed and the uninterrupted run"));
    }
    Ok(())
}

/// What a model family's full fine-tune supports of [`TrainPrecision`], declared once by the trainer
/// ([`crate::finetune::ChatTrainer::TRAIN_PRECISION`]) so every surface (the SDK, the command line) validates a
/// request the same way and refuses what the model does not run by naming the model and the key.
#[derive(Clone, Copy, Debug)]
pub struct TrainPrecisionSupport {
    pub model: &'static str,
    pub state: StateSupport,
    /// The per-layer linears may live in a low-precision encoding with an off-device master (`weights=`).
    pub resident: bool,
    /// ... and be streamed through a window of layers (`window=`).
    pub layer_streamed: bool,
    /// The per-layer tensors of its parameter store may stream in f32 through a device window (`weights=f32`, with
    /// or without `window=`), through [`crate::Model::set_resident`] ([`crate::streamed`]).
    pub f32_streamed: bool,
}

impl TrainPrecisionSupport {
    /// `Ok` when `p` asks for nothing this model does not run; otherwise the refusal, naming the model and the key.
    /// Whether a particular parameter can take a choice is the later business of [`TrainPrecision::resolve`].
    pub fn check(&self, p: &TrainPrecision) -> Result<(), String> {
        let state = p.policy.default.state;
        let state_ok = match self.state {
            StateSupport::Fp32 => state == StateFormat::F32,
            StateSupport::DeviceFused => matches!(state, StateFormat::F32 | StateFormat::Bf16) || state == (StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK }),
            StateSupport::Any => true,
        };
        if !state_ok {
            let runs = match self.state {
                StateSupport::Fp32 => "f32 optimizer state",
                StateSupport::DeviceFused => "f32, bf16 or int8:64 optimizer state",
                StateSupport::Any => "any optimizer state",
            };
            return Err(format!("{}: train precision 'state={}' is not supported: its full fine-tune runs {runs}", self.model, state_name(state)));
        }
        let Some(r) = p.resident else { return Ok(()) };
        if r.weights == ResidentValue::F32 {
            return if self.f32_streamed {
                Ok(())
            } else {
                Err(format!("{}: train precision 'weights=f32' is not supported: its layers do not stream through a parameter-store window", self.model))
            };
        }
        if !self.resident {
            return Err(format!("{}: train precision 'weights={}' is not supported: its full fine-tune keeps fp32 weights (no resident low-precision linears)", self.model, resident_name(r.weights)));
        }
        if r.window.is_some() && !self.layer_streamed {
            return Err(format!("{}: train precision 'window' is not supported: its resident full fine-tune keeps every layer on the device", self.model));
        }
        Ok(())
    }
}

fn state_name(s: StateFormat) -> String {
    match s {
        StateFormat::F32 => "f32".into(),
        StateFormat::Bf16 => "bf16".into(),
        StateFormat::Int8Block { block } => format!("int8:{block}"),
        StateFormat::Log8Block { block } => format!("log8:{block}"),
    }
}

fn resident_name(v: ResidentValue) -> String {
    match v {
        ResidentValue::F32 => "f32".into(),
        ResidentValue::Bf16 => "bf16".into(),
        ResidentValue::Fp8E4M3 { block } => format!("fp8:{block}"),
        ResidentValue::Int8 => "int8".into(),
        ResidentValue::Nf4 => "nf4".into(),
    }
}

/// Whether `spec` names `key` (a `key=value` item).
fn item_named(spec: &str, key: &str) -> bool {
    spec.split(',').any(|item| item.trim().split_once('=').is_some_and(|(k, _)| k == key))
}

fn parse_resident_value(v: &str) -> Result<ResidentValue, String> {
    let (kind, block) = match v.split_once(':') {
        Some((k, b)) => (k, Some(b.parse::<u32>().ok().filter(|&b| b > 0).ok_or_else(|| format!("train precision 'weights': '{b}' is not a block size"))?)),
        None => (v, None),
    };
    match (kind, block) {
        ("f32", None) => Ok(ResidentValue::F32),
        ("bf16", None) => Ok(ResidentValue::Bf16),
        ("fp8", b) => Ok(ResidentValue::Fp8E4M3 { block: b.unwrap_or(128) }),
        ("int8", None) => Ok(ResidentValue::Int8),
        ("nf4", None) => Ok(ResidentValue::Nf4),
        ("f32" | "bf16" | "int8" | "nf4", Some(_)) => Err(format!("train precision 'weights': '{kind}' has a fixed layout (only fp8 takes a block size)")),
        (other, _) => Err(format!("train precision 'weights': '{other}' is not f32, bf16, fp8, int8 or nf4")),
    }
}

fn parse_state(v: &str) -> Result<StateFormat, String> {
    let (kind, block) = match v.split_once(':') {
        Some((k, b)) => (k, b.parse::<u32>().map_err(|_| format!("train precision 'state': '{b}' is not a block size"))?),
        None => (v, paramstore::DEVICE_INT8_BLOCK),
    };
    if block == 0 {
        return Err("train precision 'state': block size 0".into());
    }
    match kind {
        "f32" => Ok(StateFormat::F32),
        "bf16" => Ok(StateFormat::Bf16),
        "int8" => Ok(StateFormat::Int8Block { block }),
        "log8" => Ok(StateFormat::Log8Block { block }),
        other => Err(format!("train precision 'state': '{other}' is not f32, bf16, int8 or log8")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: the textual form round-trips every knob and rejects what it cannot read, naming the key.
    #[test]
    fn parse_reads_every_knob_and_names_what_it_rejects() {
        let t = TrainPrecision::parse("state=int8:128, master=f32,value=bf16,grad=bf16,sr=on,narrow=on").unwrap();
        let p = t.policy.default;
        assert_eq!((p.state, p.master, p.value, p.grad, p.stochastic_rounding, t.narrow_state), (StateFormat::Int8Block { block: 128 }, MasterPolicy::F32, ValueDtype::Bf16, GradDtype::Bf16, true, true));
        assert!(TrainPrecision::parse("").unwrap().is_default());
        assert!(TrainPrecision::parse("state=fp4").unwrap_err().contains("state"));
        assert!(TrainPrecision::parse("moment=f32").unwrap_err().contains("moment"));
        assert!(TrainPrecision::parse("master").unwrap_err().contains("key=value"));
    }

    /// Spec: in a resident run the precision governs the resident tensors; the store tensors left over (the norms)
    /// run f32 with no master whatever `value`, `grad` and `master` say, and keep the run's state format.
    #[test]
    fn resident_store_runs_f32_with_the_runs_state() {
        let norms = vec![("final_norm.weight".to_string(), 64usize)];
        for spec in ["weights=bf16,grad=bf16,state=bf16,master=f32", "weights=bf16,state=int8", "weights=bf16,value=bf16,sr=on"] {
            let run = TrainPrecision::parse(spec).unwrap();
            assert!(run.resolve(&norms, false).is_err() || spec.ends_with("state=int8"), "{spec}: the uniform policy must not already fit the store");
            let store = run.resident_store().resolve(&norms, false).unwrap_or_else(|e| panic!("{spec}: {e}"));
            assert_eq!((store[0].value, store[0].grad, store[0].master), (ValueDtype::F32, GradDtype::F32, MasterPolicy::None));
            assert_eq!(store[0].state, run.policy.default.state);
        }
        let log8 = TrainPrecision::parse("weights=bf16,state=log8").unwrap();
        assert!(log8.resident_store().resolve(&norms, false).unwrap_err().contains("final_norm.weight"));
    }

    /// Spec: a choice the trainer does not run is an error naming the parameter, not a panic; a host-offloaded
    /// state accepts every state format, the device only the fused ones.
    #[test]
    fn resolve_returns_an_error_naming_the_parameter() {
        let params = vec![("tok.weight".to_string(), 64usize)];
        let master = TrainPrecision::parse("master=f32").unwrap();
        assert!(master.resolve(&params, true).unwrap_err().contains("tok.weight"));
        let log8 = TrainPrecision::parse("state=log8").unwrap();
        assert!(log8.resolve(&params, false).unwrap_err().contains("tok.weight"));
        assert!(log8.resolve(&params, true).is_ok());
        assert!(TrainPrecision::parse("state=bf16").unwrap().resolve(&params, false).is_ok());
    }

    /// Spec: the resident keys parse into one typed request, a resident full fine-tune takes an f32 master unless one
    /// is named (f32 weights are their own master), and `window`/`clip` without `weights` are refused by name.
    #[test]
    fn resident_keys_parse_into_one_request_with_a_master() {
        let t = TrainPrecision::parse("weights=fp8:64,window=2,clip=running,state=int8").unwrap();
        assert_eq!(t.resident, Some(ResidentRequest { weights: ResidentValue::Fp8E4M3 { block: 64 }, window: Some(2), clip_pass: ClipPass::Running }));
        let cfg = t.resident_config(Some(7)).unwrap().expect("resident");
        assert_eq!((cfg.master, cfg.window, cfg.seed), (MasterPolicy::F32, Some(LayerWindow { layers: 2 }), Some(7)));
        assert!(!t.is_default());
        assert_eq!(TrainPrecision::parse("weights=bf16").unwrap().resident.unwrap().clip_pass, ClipPass::TwoPass);
        let f32_cfg = TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(None).unwrap().expect("resident");
        assert_eq!((f32_cfg.value, f32_cfg.master), (ResidentValue::F32, MasterPolicy::None));
        assert!(TrainPrecision::default().resident_config(None).unwrap().is_none());
        assert!(TrainPrecision::parse("window=2").unwrap_err().contains("weights"));
        assert!(TrainPrecision::parse("weights=fp4").unwrap_err().contains("weights"));
        assert!(TrainPrecision::parse("weights=bf16,window=0").unwrap_err().contains("window"));
        assert!(TrainPrecision::parse("weights=bf16:8").unwrap_err().contains("fixed layout"));
    }

    /// Spec: a model's declared support decides a request, and a refusal names the model and the key.
    #[test]
    fn declared_support_refuses_what_a_model_does_not_run_by_name() {
        let fused = TrainPrecisionSupport { model: "tiny", state: StateSupport::DeviceFused, resident: true, layer_streamed: false, f32_streamed: false };
        assert!(fused.check(&TrainPrecision::default()).is_ok());
        assert!(fused.check(&TrainPrecision::parse("state=int8,weights=bf16").unwrap()).is_ok());
        let e = fused.check(&TrainPrecision::parse("state=log8").unwrap()).unwrap_err();
        assert!(e.contains("tiny") && e.contains("state=log8:64"), "{e}");
        let e = fused.check(&TrainPrecision::parse("weights=nf4,window=2").unwrap()).unwrap_err();
        assert!(e.contains("tiny") && e.contains("window"), "{e}");
        let fp32 = TrainPrecisionSupport { model: "plain", state: StateSupport::Fp32, resident: false, layer_streamed: false, f32_streamed: false };
        assert!(fp32.check(&TrainPrecision::parse("state=bf16").unwrap()).unwrap_err().contains("plain"));
        assert!(fp32.check(&TrainPrecision::parse("weights=fp8").unwrap()).unwrap_err().contains("weights=fp8:128"));
        let any = TrainPrecisionSupport { model: "wide", state: StateSupport::Any, resident: false, layer_streamed: false, f32_streamed: false };
        assert!(any.check(&TrainPrecision::parse("state=log8:32").unwrap()).is_ok());
        // f32 layers streaming through a store window are their own offer: a model with a narrow resident state of
        // its own does not take them by that, and a store model takes them with or without a window.
        let e = fused.check(&TrainPrecision::parse("weights=f32,window=1").unwrap()).unwrap_err();
        assert!(e.contains("tiny") && e.contains("weights=f32"), "{e}");
        let streamed = TrainPrecisionSupport { model: "store", state: StateSupport::Any, resident: false, layer_streamed: false, f32_streamed: true };
        assert!(streamed.check(&TrainPrecision::parse("weights=f32,window=1").unwrap()).is_ok());
        assert!(streamed.check(&TrainPrecision::parse("weights=f32").unwrap()).is_ok());
        assert!(streamed.check(&TrainPrecision::parse("weights=bf16").unwrap()).unwrap_err().contains("weights=bf16"));
    }

    fn footprint(numel: usize) -> TrainingFootprint {
        TrainingFootprint {
            frozen_weight_bytes: 0,
            trainable: vec![("w".to_string(), numel)],
            precision: ParamPrecision::FP32,
            moment_slots: 2,
            int8_block: paramstore::DEVICE_INT8_BLOCK,
            optimizer_state_on_host: false,
            activation_bytes: 0,
            recompute_activation_bytes: None,
            saved_boundaries: None,
        }
    }

    /// Spec: a run that does not fit with fp32 state is refused with the arithmetic unless it authorized
    /// narrowing, in which case the first format that fits is chosen and the decision is in the report.
    #[test]
    fn the_state_is_narrowed_only_when_authorized_and_the_report_says_so() {
        let n = 1 << 20;
        // value + gradient (8 B/elem) + fp32 moments (8 B/elem) = 16 B/elem; bf16 moments make it 12, int8 ~10.
        let room = |bytes_per_elem: u64| Capacity { device_bytes: bytes_per_elem * n as u64 + 1024, host_bytes: u64::MAX };
        let strict = TrainPrecision::default();
        assert_eq!(strict.plan(&footprint(n), room(16)).unwrap().precision.policy.default.state, StateFormat::F32);
        let err = strict.plan(&footprint(n), room(13)).unwrap_err();
        assert!(err.contains("does not fit") && err.contains("OptimizerState"), "{err}");

        let may_narrow = TrainPrecision { narrow_state: true, ..TrainPrecision::default() };
        let bf16 = may_narrow.plan(&footprint(n), room(13)).unwrap();
        assert_eq!(bf16.precision.policy.default.state, StateFormat::Bf16);
        assert!(bf16.report.decisions.iter().any(|d| d.lever == Lever::OptimizerState && d.to.contains("Bf16")), "{}", bf16.report);
        let int8 = may_narrow.plan(&footprint(n), room(11)).unwrap();
        assert_eq!(int8.precision.policy.default.state, StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK });
        assert!(may_narrow.plan(&footprint(n), room(9)).is_err(), "even int8 state does not fit");
    }
}
