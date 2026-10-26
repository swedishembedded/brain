// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight of a model through a fixed device window, you can
// procure our services by sending an email to info@swedishembedded.com.

//! **Layer-streamed full fine-tuning of a parameter-store model** ([`StreamedStore`]): the per-layer tensors of a
//! model whose own kernels read f32 weights from its [`ParamStore`] move into an f32 [`ResidentFullFt`] (the value its
//! own master, the optimizer state on the host), held on the device only for a window of layers. The store keeps its
//! names: each streamed entry of [`ParamStore::weight`] and [`ParamStore::grad`] becomes a handle to the window slot
//! and gradient lane the tensor's layer uses, so the model's recorded passes run unchanged.
//!
//! What a model adds is where its layers begin and end in those passes ([`LayerTape`]): the forward loads a layer's
//! weights into its slot before the layer runs ([`StreamedStore::forward`]), and the backward is driven by the shared
//! update ([`crate::resident_ft::adamw_step`]), which loads each layer again, takes its gradient and updates it before
//! the next layer reuses the slot ([`StreamedStore::backward_events`]). Everything else in the store (embeddings, the
//! head, final norms) stays on the device under the model's own optimizer, updated by the step's same multiplier.
//!
//! No gradient outlives its layer, so a step is one micro-batch: gradient accumulation across backward passes is
//! refused, never silently replaced.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;

use checkpoint::TensorSource;
use gpu_core::Gpu;
use optim::checkpoint::CheckpointDir;
use paramstore::ParamStore;

use crate::memplan::{Authorized, Capacity};
use crate::ops::{StoreDevice, Weight};
use crate::resident_ft::{self, LayerEvent, ResidentBackward, ResidentBudget, ResidentFtConfig, ResidentFullFt, ResidentSpec, ResidentValue, StepArgs};
use crate::tape::{LayerTape, TapeEdge};

/// A recorded pass a layer-streamed build acts between the layers of: a [`LayerTape`], or a model's own segmented
/// pass (one that interleaves host work, such as routed experts built per pass).
pub trait LayeredPass {
    /// Submit the pass in order, reporting [`TapeEdge::Before`] and [`TapeEdge::After`] around each layer's part
    /// (edges with no layer may be reported too and are ignored).
    fn submit_hooked(&self, gpu: &Gpu, edge: &mut dyn FnMut(TapeEdge));
}

impl LayeredPass for LayerTape {
    fn submit_hooked(&self, gpu: &Gpu, edge: &mut dyn FnMut(TapeEdge)) {
        LayerTape::submit_hooked(self, gpu, edge);
    }
}

/// The streamed per-layer tensors of a [`ParamStore`] model.
pub struct StreamedStore {
    ft: RefCell<ResidentFullFt>,
    index: HashMap<String, usize>,
}

/// The store's current values, read off the device one tensor at a time, as the resident state's initial values.
struct StoreValues<'a> {
    gpu: &'a Gpu,
    ps: &'a ParamStore,
}

impl TensorSource for StoreValues<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        if !self.ps.weight.contains_key(name) {
            return false;
        }
        f(&self.ps.read_weight(self.gpu, name));
        true
    }
}

impl StreamedStore {
    /// Move every trained tensor of `ps` that `layer_of` places in a layer into a resident state under `cfg` (f32
    /// values; a narrower value would need the model's kernels to read it), alias the store's entries to the state's
    /// slots and lanes, and withdraw those tensors from the store's optimizer: the caller must re-record its
    /// optimizer (`optim::Optim::invalidate`). `gpu` must have registered [`crate::ops::store_device_kernels`]. The
    /// planner refuses a window that does not fit the device's free memory before anything is allocated.
    pub fn convert(gpu: &Gpu, ps: &mut ParamStore, cfg: ResidentFtConfig, layer_of: &dyn Fn(&str) -> Option<u32>) -> Result<StreamedStore, String> {
        if cfg.value != ResidentValue::F32 {
            return Err(format!("this model's kernels read f32 weights; its layers stream as weights=f32, not {:?}", cfg.value));
        }
        let mut specs: Vec<ResidentSpec> = ps.trained_params().filter_map(|(name, numel)| Some((layer_of(name)?, name, *numel))).map(|(layer, name, numel)| {
            let k = u32::try_from(numel).map_err(|_| format!("{name}: {numel} elements exceed one resident tensor"))?;
            Ok(ResidentSpec::linear(name.clone(), 1, k, layer))
        }).collect::<Result<_, String>>()?;
        if specs.is_empty() {
            return Err("no trained tensor of this model belongs to a layer: nothing to stream".into());
        }
        specs.sort_by(|a, b| (a.layer(), &a.name).cmp(&(b.layer(), &b.name)));
        let dev = StoreDevice::new(gpu)?;
        let budget = ResidentBudget { capacity: Capacity::measured(), others: &[], authorized: Authorized::none() };
        let ft = ResidentFullFt::new(&dev, cfg, specs, &StoreValues { gpu, ps }, &budget)?;
        let mut index = HashMap::new();
        for (i, spec) in ft.specs().iter().enumerate() {
            let Weight::F32 { w, .. } = ft.weight(i) else { unreachable!("an f32 resident value is an f32 weight") };
            ps.weight.insert(spec.name.clone(), w.clone());
            ps.grad.insert(spec.name.clone(), ft.grad_buffer(i).clone());
            ps.adam_m.remove(&spec.name);
            ps.adam_v.remove(&spec.name);
            ps.moment_scales.remove(&spec.name);
            index.insert(spec.name.clone(), i);
        }
        ps.trainable.retain(|(n, _)| !index.contains_key(n));
        ps.offload.retain(|(n, _)| !index.contains_key(n));
        Ok(StreamedStore { ft: RefCell::new(ft), index })
    }

    /// Whether `name` is one of the streamed tensors.
    pub fn holds(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// The streamed tensors' names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// Whether the layers stream through a window (the backward then runs inside [`Self::adamw_step`]); without one,
    /// every layer is held on the device and the model's backward runs as recorded before the update.
    pub fn is_layer_streamed(&self) -> bool {
        self.ft.borrow().is_layer_streamed()
    }

    /// Device bytes of the streamed weights this build holds (the window's slots).
    pub fn device_weight_bytes(&self) -> u64 {
        self.ft.borrow().device_weight_bytes()
    }

    /// Submit the forward `pass`, loading each layer's weights into its slot before its part.
    pub fn forward(&self, gpu: &Gpu, pass: &dyn LayeredPass) -> io::Result<()> {
        let dev = device(gpu)?;
        let mut failed = None;
        pass.submit_hooked(gpu, &mut |edge| {
            if let (TapeEdge::Before { layer: Some(layer), .. }, None) = (edge, &failed) {
                failed = self.ft.borrow_mut().load_layer(&dev, layer).err();
            }
        });
        failed.map_or(Ok(()), Err)
    }

    /// Submit the backward `pass` for [`ResidentBackward::backward_pass`]: [`LayerEvent::AfterHead`] once the parts
    /// before the first layer's are submitted, [`LayerEvent::BeforeBackward`] and [`LayerEvent::AfterBackward`]
    /// around each layer's part. The first error stops the reports.
    pub fn backward_events(&self, gpu: &Gpu, pass: &dyn LayeredPass, on: &mut dyn FnMut(LayerEvent) -> io::Result<()>) -> io::Result<()> {
        let mut failed: Option<io::Error> = None;
        let mut head_done = false;
        let mut report = |event: LayerEvent, failed: &mut Option<io::Error>| {
            if failed.is_none() {
                *failed = on(event).err();
            }
        };
        pass.submit_hooked(gpu, &mut |edge| match edge {
            TapeEdge::Before { layer: Some(layer), .. } => {
                if !std::mem::replace(&mut head_done, true) {
                    report(LayerEvent::AfterHead, &mut failed);
                }
                report(LayerEvent::BeforeBackward(layer), &mut failed);
            }
            TapeEdge::After { layer: Some(layer), .. } => report(LayerEvent::AfterBackward(layer), &mut failed),
            _ => {}
        });
        if !head_done {
            report(LayerEvent::AfterHead, &mut failed);
        }
        failed.map_or(Ok(()), Err)
    }

    /// One optimizer step of the whole model ([`resident_ft::adamw_step`]): with a window, the backward `pass` runs
    /// here layer by layer under the step's clip (the model's `backward` only arms it), every streamed tensor updated
    /// as its layer completes; without one, the backward already ran. The rest of `ps` is updated by the model's own
    /// device optimizer `optim`, by the same multiplier. Returns the global gradient norm the clip measured.
    pub fn adamw_step(&self, gpu: &Gpu, ps: &ParamStore, optim: &optim::Optim, pass: &dyn LayeredPass, args: &StepArgs) -> io::Result<Option<f32>> {
        let host = StoreBackward { gpu, ps, optim, pass, stream: self };
        resident_ft::adamw_step(&host, &self.ft, &device(gpu)?, args)
    }

    /// The current value of streamed tensor `name`.
    pub fn read_weight(&self, gpu: &Gpu, name: &str) -> io::Result<Vec<f32>> {
        let i = self.slot(name)?;
        self.ft.borrow_mut().read_master(&device(gpu)?, i)
    }

    /// Replace the value of streamed tensor `name`.
    pub fn write_weight(&self, gpu: &Gpu, name: &str, data: &[f32]) -> io::Result<()> {
        let i = self.slot(name)?;
        self.ft.borrow_mut().write_master(&device(gpu)?, i, data)
    }

    /// Commit the streamed values and their optimizer state as of update `step` into `dir`.
    pub fn commit(&self, gpu: &Gpu, dir: &std::path::Path, step: u32) -> io::Result<()> {
        let dir = CheckpointDir::new(dir)?;
        let mut ft = self.ft.borrow_mut();
        let run = ft.run_state(0, Vec::new());
        if run.step != step {
            return Err(io::Error::other(format!("the run has absorbed {step} updates, the streamed state {}", run.step)));
        }
        ft.commit(&device(gpu)?, &dir, &run).map(drop)
    }

    /// Restore the generation committed at update `step` from `dir`.
    pub fn restore(&self, gpu: &Gpu, dir: &std::path::Path, step: u32) -> io::Result<()> {
        let ckpt = CheckpointDir::new(dir)?;
        match self.ft.borrow_mut().restore_at(&device(gpu)?, &ckpt, step)? {
            Some(_) => Ok(()),
            None => Err(io::Error::other(format!("{}: no streamed training state was committed at update {step}", dir.display()))),
        }
    }

    fn slot(&self, name: &str) -> io::Result<usize> {
        self.index.get(name).copied().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{name} is not a streamed tensor")))
    }
}

/// What the shared resident update drives of a streamed store model: the store's own parameters under the model's
/// device optimizer, and the model's backward pass one layer at a time.
struct StoreBackward<'a> {
    gpu: &'a Gpu,
    ps: &'a ParamStore,
    optim: &'a optim::Optim,
    pass: &'a dyn LayeredPass,
    stream: &'a StreamedStore,
}

impl ResidentBackward for StoreBackward<'_> {
    fn flush_backward(&self) {
        self.gpu.flush();
    }

    fn zero_remainder_grads(&self) {
        self.ps.zero_grads(self.gpu);
    }

    fn remainder_grad_sum_sq(&self, norm: &optim::gradnorm::DeviceGradNorm) -> f64 {
        let items: Vec<optim::gradnorm::NormItem<'_>> =
            self.ps.opt_params().iter().map(|(name, numel)| optim::gradnorm::NormItem { buf: self.ps.g(name), numel: *numel, dtype: crate::ops::GradDtype::F32 }).collect();
        norm.sum_sq(self.gpu, &items)
    }

    fn backward_pass(&self, on: &mut dyn FnMut(LayerEvent) -> io::Result<()>) -> io::Result<()> {
        self.stream.backward_events(self.gpu, self.pass, on)
    }

    fn update_remainder(&self, args: &StepArgs, grad_multiplier: f32) {
        // The clip is already in the multiplier: the device optimizer must not clip again.
        self.optim.step(self.gpu, self.ps, args.step, args.lr, args.wd, args.adam, None, grad_multiplier);
    }
}

fn device(gpu: &Gpu) -> io::Result<StoreDevice<'_>> {
    StoreDevice::new(gpu).map_err(io::Error::other)
}

/// The layer `blocks.{l}.*` names (the convention every decoder here uses for its per-layer tensors).
pub fn blocks_layer(name: &str) -> Option<u32> {
    name.strip_prefix("blocks.")?.split('.').next()?.parse().ok()
}

/// The layer-streaming contract every model that offers [`crate::Model::set_resident`] honours, for its own test to
/// run: `build` makes a fresh trainable model (the same initial weights every call) and `step(model, t, clip)` runs
/// optimizer step `t` on a fixed batch (`zero_grads`, the batch, `forward`, `backward`, `adamw_step` with `clip`),
/// returning the loss.
///
/// * a build streaming its layers through a one-layer window (`weights=f32,window=1`) gives the loss of every step
///   and the final values of the build holding every layer in the same state (`weights=f32`), bit for bit, with the
///   clip off and with a clip that is active;
/// * both train what the plain f32 build trains on the device: the loss of every step agrees to 1e-5 relative. The
///   values are not compared one by one: the host update and the device kernel may round the AdamW step differently,
///   and an element whose gradient is pure rounding noise (a key bias under a softmax) then takes a full normalized
///   step in whichever direction its noise points, in either build;
/// * where the backend accounts for its allocations, the streamed build holds fewer live device bytes than the plain
///   one.
pub fn check_layer_streaming<M: crate::Model>(name: &str, build: impl Fn() -> M, step: impl Fn(&M, u32, Option<f32>) -> f32) -> Result<(), String> {
    let live = |m: &M| m.optimizer_parts().and_then(|p| p.gpu.memory_usage()).map(|u| u.live);
    let values = |m: &M| -> Vec<(String, Vec<f32>)> {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
    };
    let with = |spec: &str| -> Result<M, String> {
        let mut m = build();
        crate::trainprec::apply_train_precision(&mut m, &crate::TrainPrecision::parse(spec)?, name)?;
        Ok(m)
    };
    let plain_bytes = live(&build());
    let streamed_bytes = live(&with("weights=f32,window=1")?);
    if let (Some(plain), Some(streamed)) = (plain_bytes, streamed_bytes) {
        if streamed >= plain {
            return Err(format!("{name}: the streamed build holds {streamed} live device bytes, the plain one {plain}"));
        }
    }
    for clip in [None, Some(1e-3f32)] {
        let (resident, streamed, plain) = (with("weights=f32")?, with("weights=f32,window=1")?, build());
        for t in 1..=3 {
            let (want, got, reference) = (step(&resident, t, clip), step(&streamed, t, clip), step(&plain, t, clip));
            if want.to_bits() != got.to_bits() {
                return Err(format!("{name}: clip {clip:?} step {t}: streamed loss {got}, every layer held {want}"));
            }
            if (got - reference).abs() > 1e-5 * reference.abs().max(1.0) {
                return Err(format!("{name}: clip {clip:?} step {t}: streamed loss {got}, plain f32 build {reference}"));
            }
        }
        let (held, streamed_values) = (values(&resident), values(&streamed));
        let names = |v: &[(String, Vec<f32>)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        let mut plain_names = plain.param_names();
        plain_names.sort();
        if names(&streamed_values) != plain_names || names(&held) != plain_names {
            return Err(format!("{name}: a streamed build must list the parameters the plain build does (streamed {}, plain {})", streamed_values.len(), plain_names.len()));
        }
        for ((n, a), (_, b)) in held.iter().zip(&streamed_values) {
            if let Some(i) = a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits()) {
                return Err(format!("{name}: clip {clip:?}: {n}[{i}] is {} streamed, {} with every layer held", b[i], a[i]));
            }
        }
    }
    Ok(())
}
