// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parameter storage shared by both models: for each named parameter it holds
//! the weight, its gradient, and the AdamW moment buffers, plus a small scratch
//! buffer for grad-norm reduction. Model-agnostic; the optimizer (see `optim`)
//! drives it.

use std::collections::HashMap;

use gpu_core::Gpu;

pub mod dtype;
pub mod precision;
pub mod upload;

/// Whether a parameter is optimised. `Frozen` parameters allocate **only** the
/// weight buffer — no gradient or AdamW moment buffers — which both cuts memory
/// (critical for loading a multi-hundred-MB model for inference, or a LoRA
/// frozen base) and excludes them from the optimiser.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Trainable,
    Frozen,
    /// Optimised **off-device**: the weight and its gradient live on the GPU (the
    /// forward/backward need them), but the AdamW moments (`m`/`v`) do NOT — they
    /// are held in host memory or on disk by [`optim::OffloadAdam`], which reads
    /// the grad off the GPU, updates the host-resident `m`/`v` (in the
    /// parameter's own state format: fp32, bf16 or blockwise int8) on the CPU,
    /// and writes the new weight back. Cuts GPU optimiser state from 4×params to
    /// 2×params (weight+grad) - enabling much larger models than fit in VRAM.
    Offload,
}

/// Elements one workgroup of the cooperative grad-norm reduction
/// (`gradnorm_part`) covers. Small enough that a 1.8 M-element tensor still
/// gets hundreds of workgroups (a P40 wants ≥ ~2 k threads to be busy), large
/// enough that a 768-element bias gets exactly one.
pub const GRADNORM_ELEMS_PER_WG: usize = 8192;
/// Cap on workgroups per tensor. Past this the reduction is already at memory
/// bandwidth and every extra workgroup is one more f32 for the second pass to
/// fold; 512 workgroups = 32 768 threads, several times a P40's core count.
pub const GRADNORM_MAX_WG: usize = 512;

/// Block size of the int8 moments the device optimiser (`adamw_q8`) runs.
pub const DEVICE_INT8_BLOCK: u32 = 64;

/// Per-tensor upload chunk size, in elements (4 MiB as f32/u32 words). Bounds
/// the HOST-side scratch a weight upload ever materializes to this many
/// elements, never a whole tensor (relevant tensors run up to ~1.5 GB as
/// f32) — see [`ParamStore::new_with_roles_src`]. This is a distinct fix
/// from wgpu's own per-buffer device-side staging overhead (lesson #35,
/// `crates/gpu-core/tests/vram_overhead.rs`): that one is backend-level and
/// chunking write CALLS does not change it; this constant only bounds what
/// this crate allocates on the host to produce those calls.
pub const UPLOAD_CHUNK_WORDS: usize = 1 << 20;

/// `v` as bf16 bits, to nearest even (inf and NaN keep their class).
fn bf16_bits(v: f32) -> u32 {
    let b = v.to_bits();
    if v.is_finite() {
        (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) & 0xffff
    } else {
        (b >> 16) | if v.is_nan() { 0x40 } else { 0 }
    }
}

/// Workgroups — equivalently, partial sums — `gradnorm_part` uses for a tensor
/// of `numel` elements. The single place this policy is written down; the
/// buffer sizing (`ParamStore::norms`) and the dispatch (`optim::Optim`) both
/// read it, so they cannot disagree.
pub fn gradnorm_parts(numel: usize) -> u32 {
    numel.div_ceil(GRADNORM_ELEMS_PER_WG).clamp(1, GRADNORM_MAX_WG) as u32
}

pub struct ParamStore {
    /// Every parameter (name, numel), trainable or frozen — the full save set.
    pub params: Vec<(String, usize)>,
    /// The optimised subset (those with grad/Adam buffers). Equals `params` for
    /// an all-trainable store. The optimiser iterates this, not `params`.
    pub trainable: Vec<(String, usize)>,
    pub weight: HashMap<String, gpu_core::DeviceBuffer>,
    pub grad: HashMap<String, gpu_core::DeviceBuffer>,
    pub adam_m: HashMap<String, gpu_core::DeviceBuffer>,
    pub adam_v: HashMap<String, gpu_core::DeviceBuffer>,
    /// Sum-of-squares scratch for grad clipping. Sized for the LARGER of the
    /// two reductions that write it: `gradnorm_sq` needs one f32 per trainable
    /// tensor, the cooperative `gradnorm_part` needs one per workgroup per
    /// tensor (see [`gradnorm_parts`]).
    pub norms: gpu_core::DeviceBuffer,
    pub clip_coef: gpu_core::DeviceBuffer, // [1] device-resident clip coefficient
    /// Params optimised off-device (grad on GPU, moments in RAM). Disjoint from
    /// `trainable`; the GPU optimiser skips these, `OffloadAdam` handles them.
    pub offload: Vec<(String, usize)>,
    /// Parameters withdrawn from the optimiser AFTER construction - a fine-tune
    /// freeze (see [`Self::freeze_where`]). Distinct from [`Role::Frozen`],
    /// which is a construction-time decision that never allocates a gradient or
    /// moment buffer at all and therefore cannot be backpropagated THROUGH.
    /// These keep their gradient buffer, so the backward pass still runs and
    /// still routes gradient past them to whatever is downstream; the AdamW step
    /// simply never reads it, so the weight cannot move.
    pub frozen: Vec<(String, usize)>,
    /// Per-tensor AdamW learning-rate multiplier (LoRA+: `B`'s effective lr
    /// is `lr_ratio * lr`). Absent = `1.0`, so every existing caller is
    /// unaffected. Set via [`Self::set_lr_mult`] after construction, then
    /// read by `optim::Optim::build` when it writes each tensor's `desc`.
    pub lr_mult: HashMap<String, f32>,
    /// Block scales of the parameters whose moments are blockwise int8
    /// ([`precision::StateFormat::Int8Block`] with block 64): `adam_m`/`adam_v`
    /// then hold packed code bytes (four per word) and this buffer holds the
    /// `ceil(numel/64)` first-moment scales followed by as many second-moment
    /// scales. Absent for fp32-moment parameters.
    pub moment_scales: HashMap<String, gpu_core::DeviceBuffer>,
    /// Per-tensor training precision (value / grad / state / master) chosen by
    /// the caller's [`precision::PrecisionPolicy`]. Absent = all-fp32, which is
    /// what the device buffers above physically are today; the streamed
    /// optimiser in `optim` is the consumer of non-default entries.
    pub precision: HashMap<String, precision::ParamPrecision>,
}

/// The AdamW moment buffers of one trainable tensor in the format its precision names: f32 (one word an element),
/// bf16 (two elements a word) or blockwise int8 of [`DEVICE_INT8_BLOCK`] (four codes a word, plus `ceil(numel/64)`
/// first-moment scales followed by as many second-moment scales). Freshly allocated buffers read as zero, which is
/// every format's empty state.
///
/// `gpu.storage(numel)` and no write: a buffer that is allocated and never written costs exactly its own bytes on a
/// non-ReBAR card, while any upload into it costs double (the resident cost tracks bytes ever written), and wgpu
/// already guarantees a fresh buffer reads as zero.
struct DeviceMoments {
    m: gpu_core::DeviceBuffer,
    v: gpu_core::DeviceBuffer,
    scales: Option<gpu_core::DeviceBuffer>,
}

impl DeviceMoments {
    /// The device optimiser runs these with an f32 value and gradient and no master copy; any other precision is an
    /// error naming the parameter (those formats belong to the streamed host optimiser in `optim`).
    fn alloc(gpu: &Gpu, name: &str, numel: usize, pr: &precision::ParamPrecision) -> Result<DeviceMoments, String> {
        if pr.value != precision::ValueDtype::F32 || pr.grad != precision::GradDtype::F32 || pr.master != precision::MasterPolicy::None {
            return Err(format!("{name}: the device optimiser keeps an f32 value and gradient with no master copy; got {pr:?}"));
        }
        let pair = |words: u64| (gpu.storage(words), gpu.storage(words));
        match pr.state {
            precision::StateFormat::F32 => {
                let (m, v) = pair(numel as u64);
                Ok(DeviceMoments { m, v, scales: None })
            }
            precision::StateFormat::Bf16 => {
                let (m, v) = pair(numel.div_ceil(2) as u64);
                Ok(DeviceMoments { m, v, scales: None })
            }
            precision::StateFormat::Int8Block { block } if block == DEVICE_INT8_BLOCK => {
                let (m, v) = pair(numel.div_ceil(4) as u64);
                Ok(DeviceMoments { m, v, scales: Some(gpu.storage(2 * numel.div_ceil(DEVICE_INT8_BLOCK as usize) as u64)) })
            }
            other => Err(format!(
                "{name}: the device optimiser keeps f32 or bf16 moments or int8 blocks of {DEVICE_INT8_BLOCK}; {other:?} moments need the host-offloaded optimizer state"
            )),
        }
    }
}

impl ParamStore {
    /// All-trainable store (every parameter gets grad + AdamW moments).
    pub fn new(gpu: &Gpu, params: Vec<(String, usize)>, init: &HashMap<String, Vec<f32>>) -> ParamStore {
        Self::new_src(gpu, params, init)
    }

    /// [`Self::new`] over any [`checkpoint::TensorSource`] — the eager `HashMap`
    /// (coerces here) or a streaming `WeightReader` (uploads one tensor at a
    /// time, never a whole-model host copy).
    pub fn new_src(gpu: &Gpu, params: Vec<(String, usize)>, source: &dyn checkpoint::TensorSource) -> ParamStore {
        let roles: Vec<(String, usize, Role)> =
            params.into_iter().map(|(n, c)| (n, c, Role::Trainable)).collect();
        Self::new_with_roles_src(gpu, roles, source)
    }

    /// Role-aware store: `Frozen` parameters allocate weights only. Used for
    /// inference (all frozen) and LoRA (frozen base + trainable adapters).
    pub fn new_with_roles(
        gpu: &Gpu,
        params_roles: Vec<(String, usize, Role)>,
        init: &HashMap<String, Vec<f32>>,
    ) -> ParamStore {
        Self::new_with_roles_src(gpu, params_roles, init)
    }

    /// [`Self::new_with_roles`] over a streaming [`checkpoint::TensorSource`].
    /// Each weight is fetched by name, converted to bits, uploaded to the device,
    /// and the host f32/u32 buffers are dropped before the next — so peak host
    /// allocation is ONE tensor, whatever the source (a `WeightReader` never
    /// holds the whole model as f32; the `&HashMap` overload keeps the caller's
    /// map but adds no second copy).
    pub fn new_with_roles_src(
        gpu: &Gpu,
        params_roles: Vec<(String, usize, Role)>,
        source: &dyn checkpoint::TensorSource,
    ) -> ParamStore {
        Self::new_with_policy_src(gpu, params_roles, source, &precision::PrecisionPolicy::uniform(precision::ParamPrecision::FP32))
    }

    /// [`Self::new_with_roles_src`] where each trainable parameter's training
    /// precision comes from `policy`. The device buffers honour the optimiser
    /// state format: fp32 moments, or blockwise-int8 moments of block
    /// [`DEVICE_INT8_BLOCK`] (code bytes plus scales, see [`Self::moment_scales`]).
    /// Anything the device optimiser cannot run (bf16/other-block state, a
    /// master copy, non-fp32 value or gradient storage) panics naming the
    /// parameter: those formats are served by the streamed host optimiser in
    /// `optim`, and silently falling back to fp32 would break the byte budget
    /// the policy was chosen for.
    pub fn new_with_policy_src(
        gpu: &Gpu,
        params_roles: Vec<(String, usize, Role)>,
        source: &dyn checkpoint::TensorSource,
        policy: &precision::PrecisionPolicy,
    ) -> ParamStore {
        let mut moment_scales = HashMap::new();
        let mut precisions = HashMap::new();
        let mut weight = HashMap::new();
        let mut grad = HashMap::new();
        let mut adam_m = HashMap::new();
        let mut adam_v = HashMap::new();
        let mut params = Vec::with_capacity(params_roles.len());
        let mut trainable = Vec::new();
        let mut offload = Vec::new();
        // The bounded disk->VRAM mover, held across the WHOLE load: both the
        // host-side chunking and the device-side staging-reclaim discipline
        // (which is inherently cross-tensor - it drains roughly every GiB, not
        // every tensor) live in `upload::Uploader`, so a resident model that
        // streams its own weights gets the identical guarantees without
        // re-deriving them. See that module's doc for what each one prevents.
        //
        // Per-phase accumulators for the `BRAIN_PROFILE` summary below --
        // added while chasing DeepSeek-OCR's 20+ second model-construction
        // cost: this loop is the whole of that cost on the CPU backend, and it
        // was never known whether the source read + device upload (`Uploader::
        // tensor`, which allocates the device buffer AND streams the source's
        // bytes into it) or the periodic staging-reclaim drain (`Uploader::
        // maybe_drain`) dominated it. Kept cheap (a couple `Instant::now()`
        // calls per tensor) and silent unless `BRAIN_PROFILE` is set, same
        // gate `deepseekocr`/`wgsl-cpu` already use.
        let profile = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        let mut t_upload = std::time::Duration::ZERO;
        let mut t_flush = std::time::Duration::ZERO;
        let mut up = upload::Uploader::new(gpu);
        for (name, numel, role) in &params_roles {
            let t0 = std::time::Instant::now();
            let wbuf = up
                .tensor(source, name, *numel)
                .unwrap_or_else(|e| panic!("missing init weight {name}: {e}"));
            if profile {
                t_upload += t0.elapsed();
            }
            let t1 = std::time::Instant::now();
            up.maybe_drain(&wbuf);
            if profile {
                t_flush += t1.elapsed();
            }
            // This tensor is now fully uploaded (and drained) into `wbuf` -
            // an mmap-backed source can drop the page-cache pages behind it
            // now instead of waiting for reclaim to happen under memory
            // pressure. See `TensorSource::advise_drop`'s own doc for the
            // measured 63GB-checkpoint case this exists for.
            source.advise_drop(name);
            weight.insert(name.clone(), wbuf);
            match role {
                Role::Trainable => {
                    let pr = policy.resolve(name, *numel).precision;
                    if pr != precision::ParamPrecision::FP32 {
                        precisions.insert(name.clone(), pr);
                    }
                    let moments = DeviceMoments::alloc(gpu, name, *numel, &pr).unwrap_or_else(|e| panic!("{e}"));
                    grad.insert(name.clone(), gpu.storage(*numel as u64));
                    adam_m.insert(name.clone(), moments.m);
                    adam_v.insert(name.clone(), moments.v);
                    if let Some(scales) = moments.scales {
                        moment_scales.insert(name.clone(), scales);
                    }
                    trainable.push((name.clone(), *numel));
                }
                Role::Offload => {
                    // grad on GPU (backward writes it); moments live in RAM.
                    // Same zero-init, no-write `storage()` - see above.
                    grad.insert(name.clone(), gpu.storage(*numel as u64));
                    offload.push((name.clone(), *numel));
                }
                Role::Frozen => {}
            }
            params.push((name.clone(), *numel));
        }
        if profile {
            eprintln!(
                "paramstore: new_with_roles_src ({} tensors): upload {:.1} ms, flush/readback {:.1} ms",
                params.len(),
                t_upload.as_secs_f64() * 1e3,
                t_flush.as_secs_f64() * 1e3,
            );
        }
        let n_parts: u64 = trainable.iter().map(|(_, n)| gradnorm_parts(*n) as u64).sum();
        let norms = gpu.storage(n_parts.max(trainable.len() as u64).max(1));
        let clip_coef = gpu.storage(1);
        ParamStore { params, trainable, weight, grad, adam_m, adam_v, norms, clip_coef, offload, frozen: Vec::new(), lr_mult: HashMap::new(), moment_scales, precision: precisions }
    }

    /// Hold the optimizer state of every trainable parameter in the format `policy` names, before the first optimizer
    /// step: a tensor whose format changes gets fresh (zero) moment buffers in the new format, the others keep theirs.
    /// Returns how many tensors changed format. Host-offloaded parameters ([`Role::Offload`]) are not touched: their
    /// moments live in host memory. A format the device optimiser does not run is an error naming the parameter, and
    /// nothing is changed. The caller rebuilds whatever recorded the old buffers (`optim::Optim::invalidate`).
    pub fn apply_state_policy(&mut self, gpu: &Gpu, policy: &precision::PrecisionPolicy) -> Result<usize, String> {
        let mut changes = Vec::new();
        for (name, numel) in &self.trainable {
            let pr = policy.resolve(name, *numel).precision;
            if pr != self.precision_of(name) {
                changes.push((name.clone(), pr, DeviceMoments::alloc(gpu, name, *numel, &pr)?));
            }
        }
        let changed = changes.len();
        for (name, pr, moments) in changes {
            self.adam_m.insert(name.clone(), moments.m);
            self.adam_v.insert(name.clone(), moments.v);
            match moments.scales {
                Some(scales) => self.moment_scales.insert(name.clone(), scales),
                None => self.moment_scales.remove(&name),
            };
            if pr == precision::ParamPrecision::FP32 {
                self.precision.remove(&name);
            } else {
                self.precision.insert(name, pr);
            }
        }
        Ok(changed)
    }

    /// Move the optimizer state of every trainable parameter to host memory ([`Role::Offload`]): its device moment
    /// buffers are released, its gradient stays on the device for the backward to write. Returns how many parameters
    /// moved. The host optimizer (`optim::Optim` over [`Self::offload`]) keeps their moments from then on; call before
    /// the first optimizer step.
    pub fn offload_all(&mut self) -> usize {
        let moved = std::mem::take(&mut self.trainable);
        for (name, _) in &moved {
            self.adam_m.remove(name);
            self.adam_v.remove(name);
            self.moment_scales.remove(name);
        }
        let n = moved.len();
        self.offload.extend(moved);
        n
    }

    /// Withdraw every trainable parameter whose name satisfies `pred` from the
    /// optimiser, returning how many were withdrawn. THE freeze mechanism: the
    /// optimiser iterates [`Self::opt_params`], so a withdrawn tensor is never
    /// stepped and its weight is bit-for-bit preserved across training.
    ///
    /// Gradients keep flowing (the buffers stay allocated and
    /// [`Self::zero_grads`] keeps clearing them), which is what lets a caller
    /// freeze a backbone while still training a head that sits above it.
    ///
    /// Note for callers freezing normalization layers: this covers the
    /// GRADIENT path only. A BatchNorm's running mean/variance are updated by
    /// the forward pass, not by the optimiser, so freezing a BN's `gamma`/
    /// `beta` here does NOT stop its running stats from drifting - the module
    /// must additionally be put in eval mode / have its running-stat update
    /// disabled. Missing that leaves a "frozen" backbone that still moves.
    pub fn freeze_where(&mut self, pred: impl Fn(&str) -> bool) -> usize {
        let (frozen, keep): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.trainable).into_iter().partition(|(n, _)| pred(n));
        self.trainable = keep;
        let n = frozen.len();
        self.frozen.extend(frozen);
        n
    }

    /// Give `name`'s AdamW step a per-tensor learning-rate multiplier
    /// (LoRA+). `1.0` (the default for every tensor not set here) is
    /// bit-identical to today's single shared `lr`.
    pub fn set_lr_mult(&mut self, name: &str, mult: f32) {
        self.lr_mult.insert(name.to_string(), mult);
    }

    /// Record `name`'s training precision (see [`precision::PrecisionPolicy`]).
    pub fn set_precision(&mut self, name: &str, p: precision::ParamPrecision) {
        self.precision.insert(name.to_string(), p);
    }

    /// `name`'s training precision, or all-fp32 if never set.
    pub fn precision_of(&self, name: &str) -> precision::ParamPrecision {
        self.precision.get(name).copied().unwrap_or(precision::ParamPrecision::FP32)
    }

    /// Physical bytes of the training state of every optimised parameter
    /// (trainable + offloaded) under their recorded precision, for an
    /// optimiser rule with `moment_slots` state slots.
    pub fn training_state_bytes(&self, moment_slots: usize) -> precision::ParamBytes {
        let mut total = precision::ParamBytes::default();
        for (n, numel) in self.trainable.iter().chain(&self.offload) {
            total += self.precision_of(n).bytes(*numel, moment_slots);
        }
        total
    }

    /// `name`'s multiplier, or `1.0` if never set.
    pub fn lr_mult_of(&self, name: &str) -> f32 {
        self.lr_mult.get(name).copied().unwrap_or(1.0)
    }

    /// Partial-sum layout of [`Self::norms`] for the cooperative grad-norm:
    /// `(offset per trainable tensor, total partials)`. Tensor `i` owns
    /// `norms[off[i] .. off[i] + gradnorm_parts(numel_i)]`.
    pub fn gradnorm_layout(&self) -> (Vec<u32>, u32) {
        let mut off = Vec::with_capacity(self.trainable.len());
        let mut cur = 0u32;
        for (_, numel) in &self.trainable {
            off.push(cur);
            cur += gradnorm_parts(*numel);
        }
        (off, cur)
    }

    /// The optimised parameter list (grad/Adam present) — what the optimiser and
    /// gradient zeroing iterate.
    pub fn opt_params(&self) -> &[(String, usize)] {
        &self.trainable
    }

    pub fn w(&self, name: &str) -> &gpu_core::DeviceBuffer {
        self.weight.get(name).unwrap_or_else(|| panic!("no weight {name}"))
    }
    pub fn g(&self, name: &str) -> &gpu_core::DeviceBuffer {
        self.grad.get(name).unwrap()
    }
    pub fn numel(&self, name: &str) -> usize {
        self.params.iter().find(|(n, _)| n == name).unwrap().1
    }

    /// Zero every gradient buffer the optimiser or the backward pass touches
    /// (call once per effective batch, before the accumulating backward passes).
    /// Includes [`Self::frozen`] tensors: the backward pass still ACCUMULATES
    /// into their grad buffers, so skipping them would let a frozen tensor's
    /// gradient grow without bound across a training run. So do the
    /// [`Self::offload`] tensors: the host optimiser reads their gradient but
    /// never clears it, so leaving them out made every step train on the sum of
    /// its own gradient and all earlier ones.
    pub fn zero_grads(&self, gpu: &Gpu) {
        let clears: Vec<&gpu_core::DeviceBuffer> =
            self.trainable.iter().chain(&self.offload).chain(&self.frozen).map(|(n, _)| self.g(n)).collect();
        gpu.submit(&clears, &[]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_weight(&self, gpu: &Gpu, name: &str) -> Vec<f32> {
        gpu.read(self.w(name), self.numel(name))
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_grad(&self, gpu: &Gpu, name: &str) -> Vec<f32> {
        gpu.read(self.g(name), self.numel(name))
    }
    /// Replace a parameter's gradient, e.g. with a gradient another rank
    /// contributed to. The length must equal the parameter's.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_grad(&self, gpu: &Gpu, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.numel(name), "gradient length mismatch for {name}");
        gpu.write(self.g(name), bytemuck::cast_slice(data));
    }

    /// The AdamW first and second moments of the device-optimised parameter
    /// `name`, or `None` when it has none on the device (frozen, or offloaded
    /// to the offloading optimiser's host copy). With the weights, the step
    /// index and the batch stream, these are everything an AdamW run carries
    /// from one step to the next.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_moments(&self, gpu: &Gpu, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.moment_scales.contains_key(name) {
            return None; // int8 moments: decode with `optim::read_moments_q8`
        }
        let (m, v) = (self.adam_m.get(name)?, self.adam_v.get(name)?);
        let n = self.numel(name);
        if self.precision_of(name).state == precision::StateFormat::Bf16 {
            // Two bf16 to a word, low half first: widening to f32 is exact.
            let unpack = |buf: &gpu_core::DeviceBuffer| -> Vec<f32> {
                let words = gpu.read(buf, n.div_ceil(2));
                words.iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xffff_0000)]).take(n).collect()
            };
            return Some((unpack(m), unpack(v)));
        }
        Some((gpu.read(m, n), gpu.read(v, n)))
    }

    /// Overwrite `name`'s AdamW moments - the restore half of
    /// [`Self::read_moments`]. Refuses a parameter with no device moments or
    /// data of the wrong length rather than writing a partial state.
    pub fn write_moments(&self, gpu: &Gpu, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
        let (Some(mb), Some(vb)) = (self.adam_m.get(name), self.adam_v.get(name)) else {
            return Err(format!("{name} has no AdamW moments on the device"));
        };
        if self.moment_scales.contains_key(name) {
            return Err(format!("{name} has int8 moments; restore them with the optim crate's int8 codec"));
        }
        let n = self.numel(name);
        if m.len() != n || v.len() != n {
            return Err(format!("{name}: moments of {}/{} values for a {n}-value parameter", m.len(), v.len()));
        }
        if self.precision_of(name).state == precision::StateFormat::Bf16 {
            // The moments the kernel wrote are bf16, so rounding to nearest here restores them exactly.
            let pack = |x: &[f32]| -> Vec<u32> {
                x.chunks(2)
                    .map(|c| {
                        let h = |v: f32| bf16_bits(v);
                        h(c[0]) | c.get(1).map_or(0, |&v| h(v) << 16)
                    })
                    .collect()
            };
            gpu.write(mb, &pack(m));
            gpu.write(vb, &pack(v));
            return Ok(());
        }
        gpu.write_f32(mb, m);
        gpu.write_f32(vb, v);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_readback_and_zero_grads() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        static KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let mut init = HashMap::new();
        init.insert("w".to_string(), vec![1.5f32, -2.0, 3.0]);
        let ps = ParamStore::new(&gpu, vec![("w".to_string(), 3)], &init);
        assert_eq!(ps.read_weight(&gpu, "w"), vec![1.5, -2.0, 3.0]);
        assert_eq!(ps.numel("w"), 3);
        // grads start zero; write then zero again
        assert_eq!(ps.read_grad(&gpu, "w"), vec![0.0, 0.0, 0.0]);
        gpu.write(ps.g("w"), bytemuck::cast_slice(&[9.0f32, 9.0, 9.0]));
        assert_eq!(ps.read_grad(&gpu, "w"), vec![9.0, 9.0, 9.0]);
        ps.zero_grads(&gpu);
        assert_eq!(ps.read_grad(&gpu, "w"), vec![0.0, 0.0, 0.0]);
    }

    /// Spec: the store reports the physical training-state bytes of each
    /// parameter under its own recorded precision (default fp32).
    #[test]
    fn training_state_bytes_follow_each_parameters_precision() {
        static KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let init: HashMap<String, Vec<f32>> =
            [("a".to_string(), vec![0.0; 512]), ("b".to_string(), vec![0.0; 512])].into_iter().collect();
        let mut ps = ParamStore::new(&gpu, vec![("a".to_string(), 512), ("b".to_string(), 512)], &init);
        ps.set_precision(
            "a",
            precision::ParamPrecision { state: precision::StateFormat::Int8Block { block: 256 }, ..precision::ParamPrecision::FP32 },
        );
        let b = ps.training_state_bytes(2);
        assert_eq!(b.moments, 2 * (512 + 8) + 2 * 512 * 4);
        assert_eq!(b.value, 2 * 512 * 4);
    }

    /// A step starts from zero gradients for EVERY parameter the backward pass
    /// accumulates into, whoever updates it. An offloaded parameter's gradient
    /// lives on the device like any other; when it was left out of the clear,
    /// each step trained on the sum of its own gradient and every earlier
    /// step's.
    #[test]
    fn zero_grads_clears_offloaded_gradients_too() {
        static KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let mut init = HashMap::new();
        for n in ["trained", "offloaded"] {
            init.insert(n.to_string(), vec![1.0f32; 3]);
        }
        let ps = ParamStore::new_with_roles(&gpu, vec![("trained".to_string(), 3, Role::Trainable), ("offloaded".to_string(), 3, Role::Offload)], &init);
        for n in ["trained", "offloaded"] {
            gpu.write(ps.g(n), bytemuck::cast_slice(&[9.0f32; 3]));
        }
        ps.zero_grads(&gpu);
        for n in ["trained", "offloaded"] {
            assert_eq!(ps.read_grad(&gpu, n), vec![0.0; 3], "{n}");
        }
    }

    /// A `TensorSource` test double that records, for one upload, the largest
    /// slice ever handed to a callback and whether the unbounded `with_tensor`
    /// path was ever invoked for the tensor under test — the two facts that
    /// distinguish "materialized the whole tensor" from "streamed it in
    /// bounded chunks" without depending on any backend's allocator.
    struct Probe {
        data: HashMap<String, Vec<f32>>,
        raw: HashMap<String, Vec<u32>>,
        max_chunk_seen: std::cell::Cell<usize>,
        with_tensor_called_for: std::cell::RefCell<Vec<String>>,
        raw_words_called_for: std::cell::RefCell<Vec<String>>,
    }
    impl checkpoint::TensorSource for Probe {
        fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
            self.with_tensor_called_for.borrow_mut().push(name.to_string());
            match self.data.get(name) {
                Some(v) => {
                    self.max_chunk_seen.set(self.max_chunk_seen.get().max(v.len()));
                    f(v);
                    true
                }
                None => false,
            }
        }
        fn raw_words(&self, name: &str) -> Option<&[u32]> {
            self.raw_words_called_for.borrow_mut().push(name.to_string());
            self.raw.get(name).map(|v| v.as_slice())
        }
        fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
            match self.data.get(name) {
                Some(v) => {
                    let chunk = if max_elems == 0 { v.len().max(1) } else { max_elems };
                    for (i, part) in v.chunks(chunk).enumerate() {
                        self.max_chunk_seen.set(self.max_chunk_seen.get().max(part.len()));
                        f((i * chunk) as u64, part);
                    }
                    true
                }
                None => false,
            }
        }
    }

    /// The loader-fix claim, measured rather than read off the code (lesson
    /// #34): a tensor far larger than `UPLOAD_CHUNK_WORDS` must never be
    /// handed to the store as one unbounded slice. Red before the upload
    /// loop is changed to call `with_tensor_chunks` (today it calls
    /// `with_tensor` once per tensor, which this test catches directly via
    /// `with_tensor_called_for`).
    #[test]
    fn param_store_upload_peak_is_one_chunk_not_one_tensor() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        static KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let n = UPLOAD_CHUNK_WORDS * 5 + 37; // deliberately not a multiple of the chunk
        let vals: Vec<f32> = (0..n).map(|i| (i % 251) as f32 * 0.25 - 3.0).collect();
        let probe = Probe {
            data: HashMap::from([("big".to_string(), vals.clone())]),
            raw: HashMap::new(),
            max_chunk_seen: std::cell::Cell::new(0),
            with_tensor_called_for: std::cell::RefCell::new(Vec::new()),
            raw_words_called_for: std::cell::RefCell::new(Vec::new()),
        };
        let ps = ParamStore::new_with_roles_src(&gpu, vec![("big".to_string(), n, Role::Frozen)], &probe);
        assert!(
            probe.with_tensor_called_for.borrow().is_empty(),
            "the unbounded with_tensor path must not be used when with_tensor_chunks is available"
        );
        assert!(
            probe.max_chunk_seen.get() <= UPLOAD_CHUNK_WORDS,
            "a chunk of {} elements exceeds the {}-element bound",
            probe.max_chunk_seen.get(),
            UPLOAD_CHUNK_WORDS
        );
        assert_eq!(ps.read_weight(&gpu, "big"), vals, "chunked upload must be bit-identical to the source");
    }

    /// When a source can lend already-device-shaped bytes (`raw_words`), the
    /// store must take that path instead of `with_tensor_chunks` — no
    /// intermediate f32->u32 conversion buffer at all for the common
    /// already-f32 case.
    #[test]
    fn param_store_prefers_raw_words_when_available() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        static KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let vals = vec![1.0f32, -2.0, 3.5, 0.0];
        let words: Vec<u32> = vals.iter().map(|v| v.to_bits()).collect();
        let probe = Probe {
            data: HashMap::new(), // with_tensor/with_tensor_chunks deliberately have nothing to give
            raw: HashMap::from([("w".to_string(), words)]),
            max_chunk_seen: std::cell::Cell::new(0),
            with_tensor_called_for: std::cell::RefCell::new(Vec::new()),
            raw_words_called_for: std::cell::RefCell::new(Vec::new()),
        };
        let ps = ParamStore::new_with_roles_src(&gpu, vec![("w".to_string(), 4, Role::Frozen)], &probe);
        assert_eq!(probe.raw_words_called_for.borrow().as_slice(), ["w".to_string()]);
        assert_eq!(ps.read_weight(&gpu, "w"), vals);
    }
}
