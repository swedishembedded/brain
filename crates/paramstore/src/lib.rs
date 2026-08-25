// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parameter storage shared by both models: for each named parameter it holds
//! the weight, its gradient, and the AdamW moment buffers, plus a small scratch
//! buffer for grad-norm reduction. Model-agnostic; the optimizer (see `optim`)
//! drives it.

use std::collections::HashMap;

use gpu_core::Gpu;

pub mod dtype;
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
    /// are held in system RAM by [`optim::OffloadAdam`], which reads the grad off
    /// the GPU, updates host-resident `m`/`v`/master-weights on the CPU (AVX2),
    /// and writes the new weight back. Cuts GPU optimiser state from 4×params to
    /// 2×params (weight+grad) and puts the other 2× (m+v) in the box's 177 GB of
    /// RAM — enabling much larger models than fit in 24 GB of VRAM.
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

/// Per-tensor upload chunk size, in elements (4 MiB as f32/u32 words). Bounds
/// the HOST-side scratch a weight upload ever materializes to this many
/// elements, never a whole tensor (relevant tensors run up to ~1.5 GB as
/// f32) — see [`ParamStore::new_with_roles_src`]. This is a distinct fix
/// from wgpu's own per-buffer device-side staging overhead (lesson #35,
/// `crates/gpu-core/tests/vram_overhead.rs`): that one is backend-level and
/// chunking write CALLS does not change it; this constant only bounds what
/// this crate allocates on the host to produce those calls.
pub const UPLOAD_CHUNK_WORDS: usize = 1 << 20;

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
            weight.insert(name.clone(), wbuf);
            match role {
                Role::Trainable => {
                    // `gpu.storage(numel)` - NOT `storage_init`, and no write
                    // at all - for the zero-initialized grad/Adam moment
                    // buffers. Measured precisely on this backend: a buffer
                    // that is allocated and never written costs exactly its
                    // own bytes on this non-ReBAR card; ANY upload into it -
                    // `storage_init` or a plain `storage()` + `write_f32`,
                    // chunked or not - costs double that, because the resident
                    // cost tracks cumulative bytes ever WRITTEN, not the
                    // call shape. wgpu already guarantees a freshly-created
                    // buffer reads as zero, so the old `let z = vec![0.0;
                    // numel]; storage_init(name, &z)` was paying that penalty for
                    // a write whose only content was the zero the buffer
                    // already had - three such buffers per trainable tensor
                    // (grad, adam_m, adam_v) made this the dominant real
                    // VRAM cost of a full (non-LoRA) finetune, invisible
                    // from the nominal parameter count.
                    grad.insert(name.clone(), gpu.storage(*numel as u64));
                    adam_m.insert(name.clone(), gpu.storage(*numel as u64));
                    adam_v.insert(name.clone(), gpu.storage(*numel as u64));
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
        ParamStore { params, trainable, weight, grad, adam_m, adam_v, norms, clip_coef, offload }
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

    /// Zero every (trainable) gradient buffer (call once per effective batch,
    /// before the accumulating backward passes).
    pub fn zero_grads(&self, gpu: &Gpu) {
        let clears: Vec<&gpu_core::DeviceBuffer> = self.trainable.iter().map(|(n, _)| self.g(n)).collect();
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
