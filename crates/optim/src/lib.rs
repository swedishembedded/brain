// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! AdamW with optional global grad-norm clipping, matching
//! `torch.optim.AdamW` + `torch.nn.utils.clip_grad_norm_`.
//!
//! Generic over either model: it only needs the pipeline indices of the
//! `adamw`, `gradnorm_sq`, and `grad_scale` kernels in that model's `Gpu`.
//!
//! ## The multi-tensor dispatch graph (M6.4)
//!
//! A step used to cost `3P+1` dispatches (`P` = trainable tensor count:
//! `P` grad-norm + 1 clip-coefficient + `P` grad-scale + `P` AdamW) plus `P`
//! separate 9-word `gpu.write`s to `P` physically distinct AdamW uniforms
//! that differed between tensors ONLY in `numel` - every other field
//! (`lr`/`beta1`/`beta2`/`eps`/`wd`/`bc1`/`bc2`) is identical across every
//! tensor in one step call, so rewriting all `P` copies of it every step was
//! pure waste, and on wgpu each `write` after the first pays an empty
//! `queue.submit(None)` (`Gpu::write`'s own doc).
//!
//! `adamw.wgsl` now folds the grad pre-scale directly in (`g = grad[i] *
//! scale * coef[0]`), which removes the grad-scale STAGE entirely (`3P+1` ->
//! `2P+1`: `P` grad-norm + 1 clip-coefficient + `P` AdamW). `coef` is a
//! device-resident scalar - the SAME `ps.clip_coef` buffer the clip stage
//! already computes when clipping is active, or a build-time-constant `[1.0]`
//! buffer otherwise - so folding it in costs no extra host write either way.
//! `scale` (a host-known constant, e.g. a `1/n_accum` averaging factor) and
//! the rest of AdamW's hyperparameters now live in ONE uniform buffer
//! (`Graph::hparams`) that every tensor's AdamW dispatch shares and binds
//! identically, written ONCE per `step()` call regardless of `P`. What
//! differs per tensor - `numel`, needed only to guard the padded tail of a
//! dispatch rounded up to a workgroup multiple - moves to a tiny per-tensor
//! descriptor buffer (`desc`) written ONCE at graph build time and never
//! again, since it cannot change for the life of a `Graph`. This is the
//! "descriptor-table foreach" this module's callers get for free: one (or a
//! few) persistent kernels, walking a per-tensor descriptor instead of a
//! per-tensor uniform full of duplicated state.
//!
//! Physically flattening `weight`/`grad`/`m`/`v` themselves into one
//! contiguous slab per category (making the whole step `O(1)` dispatches, not
//! just `O(1)` writes) was investigated and set aside for this pass: `Weight`
//! buffers are read directly by every model's forward-pass matmuls as whole,
//! independently-sized buffers, so flattening would mean threading
//! `step_sliced`'s existing offset/length binding through every one of those
//! call sites (or teaching `step`/`step_buf` to bind a baked-in sub-range of
//! a shared buffer transparently, a new cross-backend primitive) across the
//! ~15 crates that build a `ParamStore`. That is real, tractable follow-on
//! work, not something this milestone's "Commits: two" budget forces or
//! silently drops - it is out of scope for this pass and recorded as such in
//! `kernel-performance.md`, not attempted here.
//!
//! The grad-norm reduction ITSELF (the `P` `gradnorm_sq`/`gradnorm_part`
//! dispatches) is deliberately NOT fused further here: this exact fusion
//! ("a batch of small grad-norm dispatches needs fusing over an offset
//! table") was already tried and measured as a KILLED hypothesis - once each
//! dispatch is internally parallel, the whole group measured at a couple of
//! percent of a training step, and fusing it bought "well under half a
//! percent more". Re-attempting that specific fusion without new evidence
//! would be building a hypothesis the profile has already killed.

use std::cell::{Cell, RefCell};

use gpu_core::{f, Gpu, Step};
use paramstore::ParamStore;
#[cfg(not(target_arch = "wasm32"))]
pub mod offload;
pub use offload::OffloadAdam;

pub mod codec;
#[cfg(not(target_arch = "wasm32"))]
pub mod deferred;
#[cfg(not(target_arch = "wasm32"))]
pub mod encoded;
pub mod factored;
pub mod rule;
#[cfg(not(target_arch = "wasm32"))]
pub mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
pub mod storage;
#[cfg(not(target_arch = "wasm32"))]
pub mod streamed;
#[cfg(not(target_arch = "wasm32"))]
pub mod trainstate;

/// AdamW's moment decay rates and denominator epsilon. The default is
/// `torch.optim.AdamW`'s; a run that wants others (β2 0.95 is common for
/// large language models) sets them, and every optimiser in the tree
/// honours them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for Adam {
    fn default() -> Adam {
        Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

/// What every gradient is multiplied by before an AdamW update, given the
/// raw gradient's sum of squares: `extra_scale` (a `1/K` accumulation
/// average) times the clip coefficient of the scaled gradient,
/// `min(1, clip / (‖g · extra_scale‖ + 1e-6))` - torch's `clip_grad_norm_` on
/// the averaged gradient. The device optimiser's clip kernels compute the same.
pub fn grad_multiplier(raw_sum_sq: f64, clip: Option<f32>, extra_scale: f32) -> f32 {
    match clip {
        None => extra_scale,
        Some(max_norm) => {
            let norm = raw_sum_sq.sqrt() as f32 * extra_scale;
            extra_scale * (max_norm / (norm + 1e-6)).min(1.0)
        }
    }
}

impl Adam {
    /// The two bias corrections at 1-based step `t`.
    pub fn bias_corrections(&self, t: u32) -> (f32, f32) {
        (1.0 - self.beta1.powi(t as i32), 1.0 - self.beta2.powi(t as i32))
    }

    /// One element of a decoupled-weight-decay AdamW step, as `adamw.wgsl`
    /// computes it: `g` already carries any grad scale and clip coefficient,
    /// `bc` is [`Self::bias_corrections`] at this step.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn update(&self, bc: (f32, f32), lr: f32, wd: f32, g: f32, w: &mut f32, m: &mut f32, v: &mut f32) {
        *m = self.beta1 * *m + (1.0 - self.beta1) * g;
        *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
        let (mhat, vhat) = (*m / bc.0, *v / bc.1);
        *w -= lr * wd * *w;
        *w -= lr * mhat / (vhat.sqrt() + self.eps);
    }

    /// [`Self::update`] over whole slices, every gradient multiplied by `scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_slice(&self, t: u32, lr: f32, wd: f32, scale: f32, w: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32]) {
        let bc = self.bias_corrections(t);
        for i in 0..w.len() {
            self.update(bc, lr, wd, g[i] * scale, &mut w[i], &mut m[i], &mut v[i]);
        }
    }
}

/// AdamW state for a trainer whose parameters live in HOST memory, keyed by
/// parameter name.
///
/// A host trainer that owns its weights as a `name -> values` map (SUPIR's
/// fine-tune, and any future host-side trainer) needs exactly two things from
/// the optimiser: the moments for a parameter, allocated on first use, and a
/// step that IS [`Adam::update_slice`]. Owning the map here is what stops a
/// trainer from keeping its own `HashMap<String, (m, v)>` and re-deciding the
/// first step's initialisation and the shape checks.
pub struct NamedMoments {
    adam: Adam,
    moments: std::collections::HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl NamedMoments {
    pub fn new(adam: Adam) -> NamedMoments {
        NamedMoments { adam, moments: std::collections::HashMap::new() }
    }

    /// Step `name`'s weights `w` in place from gradient `g`: allocate the
    /// moments on first use, then [`Adam::update_slice`] with `scale` on the
    /// gradient (an accumulation average and/or clip coefficient).
    ///
    /// Refuses a gradient whose length differs from `w`, or from the moments
    /// already stored for `name` - a trainer that mixed up two parameters
    /// must see that as an error, not as a silent partial update.
    pub fn step(&mut self, t: u32, lr: f32, wd: f32, scale: f32, name: &str, w: &mut [f32], g: &[f32]) -> Result<(), String> {
        let (m, v) = self.moments.entry(name.to_string()).or_insert_with(|| (vec![0.0; w.len()], vec![0.0; w.len()]));
        if g.len() != w.len() {
            return Err(format!("{name}: {} gradient values for {} weights", g.len(), w.len()));
        }
        if m.len() != w.len() || v.len() != w.len() {
            return Err(format!("{name}: moments hold {} and {} values for {} weights", m.len(), v.len(), w.len()));
        }
        self.adam.update_slice(t, lr, wd, scale, w, m, v, g);
        Ok(())
    }

    /// The moments stored for `name`, when it has been stepped.
    pub fn moments(&self, name: &str) -> Option<(&[f32], &[f32])> {
        self.moments.get(name).map(|(m, v)| (&m[..], &v[..]))
    }

    /// How many parameters have state.
    pub fn len(&self) -> usize {
        self.moments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moments.is_empty()
    }
}

/// The optimiser dispatch graph, built once and reused. The bind groups (and
/// the storage buffers they reference) are fixed; only the uniform *contents*
/// change between steps (lr, bias corrections, clip factor), so each
/// dispatch's uniform is a persistent writable buffer updated via `gpu.write`.
/// Rebuilding these per step is what exhausts the GPU memory aperture and
/// triggers a device reset after a few thousand iterations.
struct Graph {
    clipped: bool,
    /// How many tensors this graph was built for. `ParamStore::freeze_where` can
    /// shrink that set between steps, and every stage here (grad-norm, the
    /// clip-coefficient fold, the per-tensor AdamW dispatches) is baked against
    /// it - so a change must force a rebuild, exactly like `clipped` does.
    n_opt: usize,
    /// How many f32s the clip-coefficient stage folds: one per parameter tensor
    /// on the reference path, one per workgroup per tensor on the cooperative
    /// one. Written into the clip uniform every step.
    n_norm_inputs: u32,
    steps: Vec<Step>,
    clip_uni: Option<gpu_core::DeviceBuffer>, // clip-coef stage (clip path only)
    /// Shared AdamW hyperparams (`lr, beta1, beta2, eps, wd, bc1, bc2, scale`):
    /// ONE buffer, bound identically by every tensor's AdamW dispatch,
    /// rewritten once per `step()` call regardless of tensor count. See the
    /// module doc.
    hparams: gpu_core::DeviceBuffer,
    /// Hyperparameters of the int8-moment kernel (`adamw_q8`), present when the
    /// store has such tensors: the eight above plus the rounding seed halves
    /// and the step index.
    hparams_q: Option<gpu_core::DeviceBuffer>,
    // constant/write-once buffers kept alive for the lifetime of their bind
    // groups: gradnorm's per-tensor uniforms, the unit coef buffer (unclipped
    // mode), and every tensor's `numel` descriptor.
    _const_unis: Vec<gpu_core::DeviceBuffer>,
}

/// The kernels the fused update runs for moments narrower than f32: bf16 moments, and blockwise int8 moments (the
/// portable kernel and its cooperative twin). A model whose full fine-tune offers those formats registers these beside
/// its own `adamw`; [`missing_moment_kernels`] names what a device lacks for a set of formats.
pub const MOMENT_KERNELS: &[(&str, &str)] = &[("adamw_bf16", kernels::ADAMW_BF16), ("adamw_q8", kernels::ADAMW_Q8), ("adamw_q8_wg", kernels::ADAMW_Q8_WG)];

/// `pipelines` with every [`MOMENT_KERNELS`] entry it does not already name appended at the end, so the positions a
/// model's kernel constants index stay where they are: the list a model whose optimizer state can be narrowed builds
/// its `Gpu` from.
pub fn with_moment_kernels(pipelines: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    let mut all = pipelines.to_vec();
    all.extend(MOMENT_KERNELS.iter().filter(|(name, _)| !pipelines.iter().any(|(n, _)| n == name)));
    all
}

/// The moment kernels the state `formats` need that `gpu` has not registered (empty when every format is runnable).
pub fn missing_moment_kernels(gpu: &Gpu, formats: impl IntoIterator<Item = paramstore::precision::StateFormat>) -> Vec<&'static str> {
    use paramstore::precision::StateFormat;
    let (bf16, int8) = formats.into_iter().fold((false, false), |(b, i), f| (b || f == StateFormat::Bf16, i || matches!(f, StateFormat::Int8Block { .. })));
    let mut missing = Vec::new();
    if bf16 && gpu.kernel_index("adamw_bf16").is_none() {
        missing.push("adamw_bf16");
    }
    if int8 && gpu.kernel_index("adamw_q8").is_none() {
        missing.push("adamw_q8");
    }
    missing
}

pub struct Optim {
    pub adamw: usize,
    pub gradnorm_sq: usize,
    /// Unused since M6.4 folded grad-scaling into `adamw.wgsl` directly (see
    /// the module doc) - kept so `Optim::new`'s signature, and the ~15 call
    /// sites that pass these five indices positionally, do not have to churn
    /// for an internal dispatch-graph change. A model that registers this
    /// kernel for `Optim::new` still builds and runs; the kernel itself is
    /// simply never dispatched by this crate anymore.
    pub grad_scale: usize,
    pub clip_coef: usize, // compute clip coefficient on-device
    /// Unused for the same reason as `grad_scale` above.
    pub grad_scale_buf: usize,
    cache: RefCell<Option<Graph>>,
    /// Seed of the stochastic rounding of int8 moments. Random per `Optim`
    /// unless [`Optim::set_rounding_seed`] pins it (a resumed run restores it).
    rounding_seed: Cell<u64>,
    /// The `extra_scale` of the latest clipped [`Self::step`], which scaled the norm it measured.
    last_scale: Cell<f32>,
    /// The host optimizer of the store's offloaded parameters ([`paramstore::Role::Offload`]): their moments live in
    /// host memory. Built on the first step that has such parameters, seeded with [`Self::rounding_seed`].
    host: RefCell<Option<OffloadAdam>>,
}

impl Optim {
    pub fn new(
        adamw: usize,
        gradnorm_sq: usize,
        grad_scale: usize,
        clip_coef: usize,
        grad_scale_buf: usize,
    ) -> Optim {
        use std::hash::{BuildHasher, Hasher};
        let random = std::collections::hash_map::RandomState::new().build_hasher().finish();
        Optim { adamw, gradnorm_sq, grad_scale, clip_coef, grad_scale_buf, cache: RefCell::new(None), rounding_seed: Cell::new(random), last_scale: Cell::new(1.0), host: RefCell::new(None) }
    }

    /// Drop the recorded update graph: the next [`Self::step`] records it again from the store. Call after the store's
    /// moment buffers were replaced (`ParamStore::apply_state_policy`).
    pub fn invalidate(&self) {
        *self.cache.borrow_mut() = None;
        *self.host.borrow_mut() = None;
    }

    /// Run `f` on the host optimizer of `ps`'s offloaded parameters, building it on first use.
    fn with_host<R>(&self, gpu: &Gpu, ps: &ParamStore, f: impl FnOnce(&mut OffloadAdam) -> R) -> R {
        let mut slot = self.host.borrow_mut();
        let host = slot.get_or_insert_with(|| {
            let mut host = OffloadAdam::new(gpu, ps);
            host.streamed().set_position(0, self.rounding_seed.get());
            host
        });
        f(host)
    }

    /// Commit the host optimizer state of the offloaded parameters (with their values) as the generation of update
    /// `step` in `dir`. An error when the state has absorbed a different number of updates than the run.
    pub fn host_commit(&self, gpu: &Gpu, ps: &ParamStore, dir: &crate::checkpoint::CheckpointDir, step: u32) -> std::io::Result<()> {
        self.with_host(gpu, ps, |host| {
            let absorbed = host.streamed().step_count();
            if absorbed != step {
                return Err(std::io::Error::other(format!("the run has absorbed {step} updates, the host optimizer state {absorbed}")));
            }
            host.commit(gpu, ps, dir).map(drop)
        })
    }

    /// Restore the generation of update `step` from `dir` into the host optimizer and the offloaded values; `false`
    /// when no such generation was committed.
    pub fn host_restore_at(&self, gpu: &Gpu, ps: &ParamStore, dir: &crate::checkpoint::CheckpointDir, step: u32) -> std::io::Result<bool> {
        self.with_host(gpu, ps, |host| host.restore_at(gpu, ps, dir, step)).map(|run| run.is_some())
    }

    /// Host bytes of the offloaded parameters' optimizer state and its staging; `None` before their first step.
    pub fn host_bytes(&self) -> Option<(u64, usize)> {
        self.host.borrow().as_ref().map(OffloadAdam::host_bytes)
    }

    /// Pin the stochastic-rounding seed of int8 moments (checkpoint resume).
    pub fn set_rounding_seed(&self, seed: u64) {
        self.rounding_seed.set(seed);
    }

    /// The seed in use; record it with the run so it can be resumed.
    pub fn rounding_seed(&self) -> u64 {
        self.rounding_seed.get()
    }

    /// The global gradient norm the latest clipped [`Self::step`] measured (of the gradient after its
    /// `extra_scale`, the norm `clip` bounds); `None` before one, or when the step did not clip - an
    /// unclipped step measures nothing. Reads the device's partial sums, so it waits for the step.
    pub fn last_grad_norm(&self, gpu: &Gpu, ps: &ParamStore) -> Option<f32> {
        let cache = self.cache.borrow();
        let graph = cache.as_ref().filter(|g| g.clipped)?;
        let sum_sq: f64 = gpu.read(&ps.norms, graph.n_norm_inputs as usize).iter().map(|&v| f64::from(v)).sum();
        Some((sum_sq.sqrt() * f64::from(self.last_scale.get())) as f32)
    }

    /// Build the dispatch graph. `clipped` selects whether the grad-norm ->
    /// clip-coefficient stage precedes AdamW; the AdamW stage itself is
    /// otherwise identical either way (see the module doc).
    fn build(&self, gpu: &Gpu, ps: &ParamStore, clipped: bool) -> Graph {
        let mut steps = Vec::new();
        let mut const_unis = Vec::new();
        let mut n_norm_inputs = 0u32;

        // Shared per-step AdamW hyperparams - see the module doc.
        let hparams = gpu.uniform_dynamic(8);

        let clip_uni = if clipped {
            // grad-norm -> clip coefficient: each stage depends on the
            // previous via storage buffers, which wgpu barriers.
            let coop = self.coop_gradnorm(gpu);
            let n_reduced = if let Some((part, _)) = coop {
                // Cooperative tree reduction: `n_wg` workgroups per tensor, each
                // writing one partial. See `gradnorm_part.wgsl`.
                let (offs, total) = ps.gradnorm_layout();
                for (i, (name, numel)) in ps.opt_params().iter().enumerate() {
                    let nwg = paramstore::gradnorm_parts(*numel);
                    let ub = gpu.uniform_dynamic(3);
                    gpu.write(&ub, &[*numel as u32, offs[i], nwg]);
                    steps.push(gpu.step_buf(part, &ub, &[ps.g(name), &ps.norms], nwg * 64));
                    const_unis.push(ub);
                }
                total
            } else {
                // Reference: one single-threaded dispatch per tensor.
                for (i, (name, numel)) in ps.opt_params().iter().enumerate() {
                    let ub = gpu.uniform_dynamic(2);
                    gpu.write(&ub, &[*numel as u32, i as u32]);
                    steps.push(gpu.step_buf(self.gradnorm_sq, &ub, &[ps.g(name), &ps.norms], 1));
                    const_unis.push(ub);
                }
                ps.opt_params().len() as u32
            };
            let cu = gpu.uniform_dynamic(3);
            // Same uniform layout either way; only `n` differs (partials vs tensors).
            let (coef_k, coef_threads) = match coop {
                Some((_, wg)) => (wg, 64),
                None => (self.clip_coef, 1),
            };
            steps.push(gpu.step_buf(coef_k, &cu, &[&ps.norms, &ps.clip_coef], coef_threads));
            n_norm_inputs = n_reduced;
            Some(cu)
        } else {
            None
        };

        // AdamW always reads a device-resident `coef[0]` (`adamw.wgsl`):
        // the clip stage's own `ps.clip_coef` when clipping is active
        // (already device-computed - no extra host write), or a build-time
        // constant `[1.0]` otherwise.
        let coef_buf = if clipped {
            ps.clip_coef.clone()
        } else {
            let unit = gpu.storage(1);
            gpu.write(&unit, &[f(1.0)]);
            const_unis.push(unit.clone());
            unit
        };

        // One dispatch per tensor, ALL sharing `hparams` and `coef_buf`.
        // `desc` is this tensor's element count plus its LoRA+ learning-rate
        // multiplier (1.0 for every tensor `ParamStore::set_lr_mult` was
        // never called on, so an all-1.0 store is bit-identical to before
        // this field existed) - written once, here, never again (neither can
        // change for the life of this graph).
        let is_bf16 = |n: &str| ps.precision_of(n).state == paramstore::precision::StateFormat::Bf16;
        let q8 = ps.opt_params().iter().any(|(n, _)| ps.moment_scales.contains_key(n) || is_bf16(n));
        let hparams_q = q8.then(|| gpu.uniform_dynamic(12));
        for (ti, (name, numel)) in ps.opt_params().iter().enumerate() {
            if is_bf16(name) {
                let kernel = gpu.kernel_index("adamw_bf16").unwrap_or_else(|| panic!("{name} has bf16 moments but this device has no `adamw_bf16` kernel registered"));
                let sr = ps.precision_of(name).stochastic_rounding as u32;
                let desc = gpu.storage(4);
                gpu.write(&desc, &[*numel as u32, f(ps.lr_mult_of(name)), ti as u32, sr]);
                steps.push(gpu.step_buf(kernel, hparams_q.as_ref().unwrap(), &[ps.w(name), ps.g(name), &ps.adam_m[name], &ps.adam_v[name], &desc, &coef_buf], numel.div_ceil(2) as u32));
                const_unis.push(desc);
                continue;
            }
            if let Some(scales) = ps.moment_scales.get(name) {
                let kernel = self.q8_kernel(gpu).unwrap_or_else(|| panic!("{name} has int8 moments but this device has no `adamw_q8` kernel registered"));
                let sr = ps.precision_of(name).stochastic_rounding as u32;
                let desc = gpu.storage(4);
                gpu.write(&desc, &[*numel as u32, f(ps.lr_mult_of(name)), ti as u32, sr]);
                let blocks = numel.div_ceil(paramstore::DEVICE_INT8_BLOCK as usize) as u32;
                let threads = if kernel == gpu.kernel_index("adamw_q8_wg").unwrap_or(usize::MAX) {
                    // four blocks per 64-lane workgroup (see `adamw_q8_wg.wgsl`)
                    blocks.div_ceil(4) * paramstore::DEVICE_INT8_BLOCK
                } else {
                    blocks
                };
                steps.push(gpu.step_buf(
                    kernel,
                    hparams_q.as_ref().unwrap(),
                    &[ps.w(name), ps.g(name), &ps.adam_m[name], &ps.adam_v[name], scales, &desc, &coef_buf],
                    threads,
                ));
                const_unis.push(desc);
                continue;
            }
            let words = kernels::adamw_desc(*numel, ps.lr_mult_of(name));
            let desc = gpu.storage(words.len() as u64);
            gpu.write(&desc, &words);
            steps.push(gpu.step_buf(
                self.adamw,
                &hparams,
                &[ps.w(name), ps.g(name), &ps.adam_m[name], &ps.adam_v[name], &desc, &coef_buf],
                *numel as u32,
            ));
            const_unis.push(desc);
        }

        Graph { clipped, n_opt: ps.opt_params().len(), n_norm_inputs, steps, clip_uni, hparams, hparams_q, _const_unis: const_unis }
    }

    /// The int8-moment AdamW kernel for this device: the cooperative
    /// `adamw_q8_wg` where the selection policy offers it (and the model
    /// registered it), else the portable `adamw_q8`. Resolved by name like
    /// [`Self::coop_gradnorm`], so registering the kernel is the whole opt-in.
    fn q8_kernel(&self, gpu: &Gpu) -> Option<usize> {
        use gpu_core::select::{candidates, Dtype, KernelVariant, Op, OpShape};
        let shape = OpShape { m: 1, n: 1 << 20, k: 0, dtype: Dtype::F32 };
        if candidates(Op::AdamWQ8, shape, &gpu.caps())[0] == KernelVariant::WorkgroupPerOutput {
            if let Some(k) = gpu.kernel_index("adamw_q8_wg") {
                return Some(k);
            }
        }
        gpu.kernel_index("adamw_q8")
    }

    /// The cooperative grad-norm pair `(gradnorm_part, clip_coef_wg)` for this
    /// device, or `None` to run the reference `gradnorm_sq` + `clip_coef`.
    ///
    /// Resolved **by name** through `Gpu::kernel_index`, not by a per-model
    /// index constant: a model opts in by appending the two kernels to its
    /// PIPELINES list and nothing else changes — no `Optim::new` signature
    /// churn across the ~15 crates that construct one, and a model that has not
    /// adopted them still runs. Policy comes from `backend_api::select` so the
    /// capability gate (the CPU JIT cannot execute the barrier) and the
    /// `BRAIN_NO_COOP_GRADNORM` A/B switch live in one place.
    fn coop_gradnorm(&self, gpu: &Gpu) -> Option<(usize, usize)> {
        use gpu_core::select::{candidates, Dtype, Op, OpShape};
        let caps = gpu.caps();
        // `numel` is per tensor and does not gate the choice (see `select.rs`),
        // so probe the policy once with a representative shape.
        let shape = OpShape { m: 1, n: 1 << 20, k: 0, dtype: Dtype::F32 };
        if candidates(Op::GradNorm, shape, &caps)[0] != gpu_core::select::KernelVariant::SplitReduction {
            return None;
        }
        Some((gpu.kernel_index("gradnorm_part")?, gpu.kernel_index("clip_coef_wg")?))
    }

    /// One optimiser step at (1-based) step index `t`, run entirely on-device in
    /// a single submit (no host readback). Gradients are multiplied by
    /// [`grad_multiplier`]: `extra_scale`, and with `clip` set the clip
    /// coefficient of the scaled gradient.
    pub fn step(
        &self,
        gpu: &Gpu,
        ps: &ParamStore,
        t: u32,
        lr: f32,
        wd: f32,
        adam: Adam,
        clip: Option<f32>,
        extra_scale: f32,
    ) {
        if !ps.offload.is_empty() {
            // The global clip spans every parameter: a store split between device and host moments has no single norm.
            assert!(
                clip.is_none() || ps.opt_params().is_empty(),
                "Optim::step: a clipped step over a store whose moments are split between the device and the host"
            );
            self.with_host(gpu, ps, |host| host.step(gpu, ps, t, lr, wd, adam, clip, extra_scale));
            if ps.opt_params().is_empty() {
                return;
            }
        }
        let (bc1, bc2) = adam.bias_corrections(t);
        let clipped = clip.is_some();

        // (Re)build the cached graph only if absent, the clip mode changed, or
        // the optimised tensor set changed (the two things that change the
        // dispatch SHAPE - see `build`).
        let need_build = match &*self.cache.borrow() {
            Some(g) => g.clipped != clipped || g.n_opt != ps.opt_params().len(),
            None => true,
        };
        if need_build {
            let g = self.build(gpu, ps, clipped);
            *self.cache.borrow_mut() = Some(g);
        }

        let cache = self.cache.borrow();
        let g = cache.as_ref().unwrap();

        // Refresh the per-step uniform contents in place (no allocation, and
        // exactly ONE write for AdamW's hyperparams regardless of how many
        // tensors this optimiser covers - see the module doc).
        if clipped {
            self.last_scale.set(extra_scale);
        }
        if let Some(max_norm) = clip {
            gpu.write(g.clip_uni.as_ref().unwrap(), &[g.n_norm_inputs, f(max_norm), f(extra_scale)]);
        }
        // `scale` folds `extra_scale` in directly only when there is no
        // device-computed clip coefficient to fold it into instead (`build`
        // already folded it into `clip_coef`'s own `extra_scale` field).
        let scale = if clipped { 1.0 } else { extra_scale };
        gpu.write(&g.hparams, &[f(lr), f(adam.beta1), f(adam.beta2), f(adam.eps), f(wd), f(bc1), f(bc2), f(scale)]);
        if let Some(hq) = &g.hparams_q {
            let seed = self.rounding_seed.get();
            gpu.write(hq, &[f(lr), f(adam.beta1), f(adam.beta2), f(adam.eps), f(wd), f(bc1), f(bc2), f(scale), seed as u32, (seed >> 32) as u32, t, 0]);
        }

        gpu.submit(&[], &g.steps);
    }
}

/// `name`'s AdamW moments as f32, whatever format the device holds them in (f32, bf16, int8 blocks); `None` for a
/// parameter with no device moments (frozen, or offloaded to the host optimizer).
pub fn read_moments(gpu: &Gpu, ps: &ParamStore, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
    if ps.moment_scales.contains_key(name) {
        return read_moments_q8(gpu, ps, name);
    }
    ps.read_moments(gpu, name)
}

/// Restore [`read_moments`]'s values into `name`'s device moments, re-encoded to the format they are held in.
pub fn write_moments(gpu: &Gpu, ps: &ParamStore, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
    if ps.moment_scales.contains_key(name) {
        return write_moments_q8(gpu, ps, name, m, v);
    }
    ps.write_moments(gpu, name, m, v)
}

/// Decode the blockwise-int8 moments of `name` from the device into f32
/// (`(first, second)`), through the same codec the host optimiser uses. `None`
/// when the parameter's moments are not int8 (use `ParamStore::read_moments`).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_moments_q8(gpu: &Gpu, ps: &ParamStore, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
    use codec::{SlotKind, StateCodec};
    use paramstore::precision::StateFormat;
    let scales = ps.moment_scales.get(name)?;
    let n = ps.numel(name);
    let nb = n.div_ceil(paramstore::DEVICE_INT8_BLOCK as usize);
    let words = n.div_ceil(4);
    let sc = gpu.read(scales, 2 * nb);
    let bytes_of = |buf: &gpu_core::DeviceBuffer| -> Vec<u8> {
        let w = gpu.read(buf, words);
        bytemuck::cast_slice::<f32, u8>(&w)[..n].to_vec()
    };
    let format = StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK };
    let decode = |kind, codes: Vec<u8>, scales: &[f32]| {
        let mut out = vec![0f32; n];
        StateCodec { format, kind }.decode(&codes, bytemuck::cast_slice(scales), &mut out);
        out
    };
    Some((
        decode(SlotKind::Signed, bytes_of(&ps.adam_m[name]), &sc[..nb]),
        decode(SlotKind::NonNegative, bytes_of(&ps.adam_v[name]), &sc[nb..]),
    ))
}

/// Restore `name`'s blockwise-int8 moments from decoded values - the inverse of [`read_moments_q8`]. Values that
/// were decoded from the device re-encode to the same codes (each block's scale is its largest decoded magnitude and
/// nearest rounding returns every level to itself), so a read-back and restore is exact.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_moments_q8(gpu: &Gpu, ps: &ParamStore, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
    use codec::{SlotKind, StateCodec};
    use paramstore::precision::StateFormat;
    let (Some(scales), Some(mb), Some(vb)) = (ps.moment_scales.get(name), ps.adam_m.get(name), ps.adam_v.get(name)) else {
        return Err(format!("{name} has no int8 moments on the device"));
    };
    let n = ps.numel(name);
    if m.len() != n || v.len() != n {
        return Err(format!("{name}: moments of {}/{} values for a {n}-value parameter", m.len(), v.len()));
    }
    let nb = n.div_ceil(paramstore::DEVICE_INT8_BLOCK as usize);
    let words = n.div_ceil(4);
    let format = StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK };
    let mut sc = vec![0f32; 2 * nb];
    let (first, second) = sc.split_at_mut(nb);
    for (kind, vals, buf, scale) in [(SlotKind::Signed, m, mb, first), (SlotKind::NonNegative, v, vb, second)] {
        let codec = StateCodec { format, kind };
        let mut codes = vec![0u8; words * 4];
        let mut scale_bytes = vec![0u8; 4 * nb];
        codec.encode(vals, &mut codes[..n], &mut scale_bytes, codec::Rounding::NEAREST, 0);
        for (s, c) in scale.iter_mut().zip(scale_bytes.as_chunks::<4>().0) {
            *s = f32::from_le_bytes(*c);
        }
        gpu.write(buf, &codes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect::<Vec<u32>>());
    }
    gpu.write_f32(scales, &sc);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use paramstore::ParamStore;
    use std::collections::HashMap;

    #[test]
    fn adamw_with_clip_matches_hand_computation() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut init = HashMap::new();
        init.insert("p".to_string(), vec![1.0f32; 4]);
        let ps = ParamStore::new(&gpu, vec![("p".to_string(), 4)], &init);
        // grad = 2.0 everywhere => global L2 norm = sqrt(16) = 4 > 1 => clipped by 1/4.
        gpu.write(ps.g("p"), bytemuck::cast_slice(&[2.0f32; 4]));

        opt.step(&gpu, &ps, 1, 0.1, 0.0, Adam::default(), Some(1.0), 1.0);
        let w = ps.read_weight(&gpu, "p");

        // expected: clip to g=0.5; AdamW t=1: mhat=g, vhat=g^2 => step = lr*g/|g| = lr.
        for &v in &w {
            assert!((v - 0.9).abs() < 1e-4, "AdamW+clip got {v}, expected ~0.9");
        }
    }

    /// The cooperative grad-norm path (`gradnorm_part` + `clip_coef_wg`) must
    /// produce the SAME clipped step as the serial one, at sizes that exercise
    /// several workgroups per tensor and several tensors per step. The kernels
    /// are resolved by name, so registering them is the whole opt-in — that is
    /// what this asserts as much as the arithmetic.
    #[test]
    fn cooperative_gradnorm_matches_the_serial_path() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        static BASE: &[(&str, &str)] = &[
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("grad_scale", kernels::GRAD_SCALE),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ];
        static COOP: &[(&str, &str)] = &[
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("grad_scale", kernels::GRAD_SCALE),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
            ("gradnorm_part", kernels::GRADNORM_PART),
            ("clip_coef_wg", kernels::CLIP_COEF_WG),
        ];
        // A tiny bias, a mid tensor and one spanning several workgroups.
        let shapes = [("b".to_string(), 7usize), ("w".to_string(), 20_000), ("e".to_string(), 3_000)];
        let mut init = HashMap::new();
        let mut grads = HashMap::new();
        for (n, c) in &shapes {
            init.insert(n.clone(), (0..*c).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect::<Vec<f32>>());
            grads.insert(n.clone(), (0..*c).map(|i| ((i % 29) as f32 - 14.0) * 0.001).collect::<Vec<f32>>());
        }
        let run = |ks: &'static [(&'static str, &'static str)]| -> Vec<Vec<f32>> {
            let gpu = gpu_core::testgpu::dev(ks);
            let opt = Optim::new(0, 1, 2, 3, 4);
            let ps = ParamStore::new(&gpu, shapes.to_vec(), &init);
            for (n, _) in &shapes {
                gpu.write(ps.g(n), bytemuck::cast_slice(&grads[n]));
            }
            opt.step(&gpu, &ps, 1, 0.01, 0.01, Adam::default(), Some(0.5), 1.0);
            shapes.iter().map(|(n, _)| ps.read_weight(&gpu, n)).collect()
        };
        let (a, b) = (run(BASE), run(COOP));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let md = x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            assert!(md < 2e-6, "tensor {i}: cooperative grad-norm changed the step by {md}");
        }
    }

    static KERNELS: &[(&str, &str)] = &[
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ];

    /// The no-clip, `extra_scale != 1` path (grad-accumulation averaging) had
    /// NO test coverage before M6.4 folded it into `adamw.wgsl` directly (it
    /// used to be a separate `grad_scale` dispatch this crate never exercised
    /// in a test). Same hand-computed shape as
    /// `adamw_with_clip_matches_hand_computation`: `extra_scale` here plays
    /// exactly the role the clip coefficient played there.
    #[test]
    fn unclipped_scale_matches_hand_computation() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut init = HashMap::new();
        init.insert("p".to_string(), vec![1.0f32; 4]);
        let ps = ParamStore::new(&gpu, vec![("p".to_string(), 4)], &init);
        gpu.write(ps.g("p"), bytemuck::cast_slice(&[2.0f32; 4]));

        // No clip; extra_scale folds in directly => effective g = 2.0*0.25 = 0.5,
        // the same effective gradient `adamw_with_clip_matches_hand_computation`
        // reaches via a clip coefficient of 0.25 instead.
        opt.step(&gpu, &ps, 1, 0.1, 0.0, Adam::default(), None, 0.25);
        let w = ps.read_weight(&gpu, "p");
        for &v in &w {
            assert!((v - 0.9).abs() < 1e-4, "AdamW+scale got {v}, expected ~0.9");
        }
    }

    /// M6.4's actual contract (`kernel-performance.md`): a step's dispatch
    /// count drops from the old `3P+1` (grad-norm + clip-coef + grad-scale +
    /// AdamW) to `2P+1` (grad-scale folded into AdamW, see the module doc) in
    /// the clipped path, and per-step HOST WRITES no longer scale with `P` at
    /// all - one shared hyperparams buffer serves every tensor's AdamW
    /// dispatch regardless of how many tensors there are.
    #[test]
    fn clipped_step_dispatches_2p_plus_1_and_writes_are_flat_in_tensor_count() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let run = |n_tensors: usize| -> (u64, u64, u64) {
            let gpu = gpu_core::testgpu::dev(KERNELS);
            let opt = Optim::new(0, 1, 2, 3, 4);
            let shapes: Vec<(String, usize)> = (0..n_tensors).map(|i| (format!("p{i}"), 16)).collect();
            let mut init = HashMap::new();
            for (n, _) in &shapes {
                init.insert(n.clone(), vec![1.0f32; 16]);
            }
            let ps = ParamStore::new(&gpu, shapes.clone(), &init);
            for (n, _) in &shapes {
                gpu.write(ps.g(n), bytemuck::cast_slice(&[2.0f32; 16]));
            }
            let stats = || gpu.stats().expect("this backend must report DeviceStats");

            // First call builds the graph; its dispatch count already equals
            // steady state (`build` only records Steps, `submit` is what
            // actually dispatches them), but its write count also carries the
            // one-off build-time writes (gradnorm uniforms, descriptors) - not
            // part of the per-step contract this test pins, so only the
            // SECOND call's deltas are asserted against.
            opt.step(&gpu, &ps, 1, 0.01, 0.0, Adam::default(), Some(1.0), 1.0);
            let (d0, w0) = {
                let s = stats();
                (s.dispatches, s.writes)
            };
            opt.step(&gpu, &ps, 2, 0.01, 0.0, Adam::default(), Some(1.0), 1.0);
            let s = stats();
            (s.dispatches - d0, s.writes - w0, n_tensors as u64)
        };
        for n in [3usize, 9] {
            let (dispatches, writes, p) = run(n);
            assert_eq!(dispatches, 2 * p + 1, "P={p}: expected 2P+1 dispatches (grad-scale folded into AdamW), got {dispatches}");
            assert!(writes <= 2, "P={p}: expected O(1) (<=2: clip_uni + shared hyperparams) writes per step, got {writes}");
        }
    }

    /// LoRA+: `ParamStore::set_lr_mult` scales exactly THAT tensor's step,
    /// leaving every other tensor's untouched. Measured at `t=1`
    /// deliberately - AdamW's bias-corrected update there is `lr * sign(g)`
    /// regardless of `|g|` (`mhat=g, vhat=g^2` cancel under the sqrt), so the
    /// ratio between two tensors' movement at `t=1` is EXACTLY their
    /// `lr_mult` ratio, not something Adam's own nonlinearity could blur.
    #[test]
    fn lr_mult_scales_exactly_that_tensors_step() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut init = HashMap::new();
        init.insert("a".to_string(), vec![1.0f32; 4]);
        init.insert("b".to_string(), vec![1.0f32; 4]);
        let mut ps = ParamStore::new(&gpu, vec![("a".to_string(), 4), ("b".to_string(), 4)], &init);
        ps.set_lr_mult("b", 4.0);
        gpu.write(ps.g("a"), bytemuck::cast_slice(&[2.0f32; 4]));
        gpu.write(ps.g("b"), bytemuck::cast_slice(&[2.0f32; 4]));

        opt.step(&gpu, &ps, 1, 0.1, 0.0, Adam::default(), None, 1.0);
        let wa = ps.read_weight(&gpu, "a");
        let wb = ps.read_weight(&gpu, "b");

        // a: plain step, w = 1.0 - lr = 0.9. b: 4x lr_mult, w = 1.0 - 4*lr = 0.6.
        for &v in &wa {
            assert!((v - 0.9).abs() < 1e-5, "a (lr_mult 1.0) got {v}, expected ~0.9");
        }
        for &v in &wb {
            assert!((v - 0.6).abs() < 1e-5, "b (lr_mult 4.0) got {v}, expected ~0.6");
        }
        // The ratio the LoRA+ mechanism actually promises: b moved exactly
        // 4x as far from its init as a did.
        let ratio = (1.0 - wb[0]) / (1.0 - wa[0]);
        assert!((ratio - 4.0).abs() < 1e-5, "expected exactly 4x, got {ratio}");
    }

    /// The freeze contract: a tensor withdrawn via `ParamStore::freeze_where`
    /// must be BIT-FOR-BIT unchanged by a step that moves its neighbours, even
    /// though its gradient buffer is full. Asserted on bits, not a tolerance -
    /// "barely moved" is not frozen.
    #[test]
    fn a_frozen_tensor_is_bit_for_bit_unchanged_by_a_step() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut init = HashMap::new();
        init.insert("keep".to_string(), vec![1.0f32; 4]);
        init.insert("train".to_string(), vec![1.0f32; 4]);
        let mut ps =
            ParamStore::new(&gpu, vec![("keep".to_string(), 4), ("train".to_string(), 4)], &init);
        assert_eq!(ps.freeze_where(|n| n == "keep"), 1);
        assert_eq!(ps.opt_params().len(), 1, "the frozen tensor must leave the optimised set");
        // Both carry a real gradient; only the unfrozen one may be applied.
        for n in ["keep", "train"] {
            gpu.write(ps.g(n), bytemuck::cast_slice(&[2.0f32; 4]));
        }
        let before = ps.read_weight(&gpu, "keep");
        opt.step(&gpu, &ps, 1, 0.1, 0.01, Adam::default(), Some(1.0), 1.0);
        assert_eq!(
            ps.read_weight(&gpu, "keep").iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a frozen tensor moved"
        );
        assert!(ps.read_weight(&gpu, "train")[0] < 0.95, "the unfrozen tensor must still be stepped");
        // Weight decay is part of the AdamW step, so it must not reach a frozen
        // tensor either - the subtle way a "frozen" weight still decays to zero.
        for t in 2..6 {
            opt.step(&gpu, &ps, t, 0.1, 0.5, Adam::default(), Some(1.0), 1.0);
        }
        assert_eq!(ps.read_weight(&gpu, "keep"), before, "weight decay reached a frozen tensor");
        // And the frozen tensor's gradient is still cleared each batch, so it
        // cannot accumulate without bound across a run.
        ps.zero_grads(&gpu);
        assert_eq!(ps.read_grad(&gpu, "keep"), vec![0.0; 4]);
    }

    /// `lr_mult == 1.0` for every tensor (the default, unset case) must be
    /// bit-identical to the pre-LoRA+ single-shared-lr behavior - the
    /// regression guard for this whole change.
    #[test]
    fn lr_mult_unset_is_bit_identical_to_no_lr_mult_field() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut init = HashMap::new();
        init.insert("p".to_string(), vec![1.0f32; 4]);
        let ps = ParamStore::new(&gpu, vec![("p".to_string(), 4)], &init);
        gpu.write(ps.g("p"), bytemuck::cast_slice(&[2.0f32; 4]));
        opt.step(&gpu, &ps, 1, 0.1, 0.0, Adam::default(), Some(1.0), 1.0);
        let w = ps.read_weight(&gpu, "p");
        for &v in &w {
            assert!((v - 0.9).abs() < 1e-4, "expected the untouched AdamW+clip result ~0.9, got {v}");
        }
    }

    /// `lr_mult` must not change the dispatch contract M6.4 pinned: still
    /// `2P+1` dispatches, still O(1) writes - the multiplier rides in the
    /// existing per-tensor `desc` buffer, which was already written once at
    /// build time regardless of its word count.
    #[test]
    fn lr_mult_does_not_change_the_dispatch_contract() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let shapes: Vec<(String, usize)> = (0..3).map(|i| (format!("p{i}"), 16)).collect();
        let mut init = HashMap::new();
        for (n, _) in &shapes {
            init.insert(n.clone(), vec![1.0f32; 16]);
        }
        let mut ps = ParamStore::new(&gpu, shapes.clone(), &init);
        for (n, _) in &shapes {
            gpu.write(ps.g(n), bytemuck::cast_slice(&[2.0f32; 16]));
        }
        ps.set_lr_mult("p1", 4.0);
        let stats = || gpu.stats().expect("this backend must report DeviceStats");

        opt.step(&gpu, &ps, 1, 0.01, 0.0, Adam::default(), Some(1.0), 1.0);
        let (d0, w0) = {
            let s = stats();
            (s.dispatches, s.writes)
        };
        opt.step(&gpu, &ps, 2, 0.01, 0.0, Adam::default(), Some(1.0), 1.0);
        let s = stats();
        assert_eq!(s.dispatches - d0, 2 * 3 + 1, "lr_mult must not add dispatches");
        assert!(s.writes - w0 <= 2, "lr_mult must not add per-step writes");
    }
}
