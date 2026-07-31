// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parameter storage shared by both models: for each named parameter it holds
//! the weight, its gradient, and the AdamW moment buffers, plus a small scratch
//! buffer for grad-norm reduction. Model-agnostic; the optimizer (see `optim`)
//! drives it.

use std::collections::HashMap;

use gpu_core::Gpu;

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
/// fold; 512 workgroups = 32 768 threads, ~8.5× a P40's core count.
pub const GRADNORM_MAX_WG: usize = 512;

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
        // Bytes written since the last FORCED flush (see below).
        let mut uploaded = 0u64;
        for (name, numel, role) in &params_roles {
            // storage()+write() (plain DEVICE_LOCAL + transient staging) instead of
            // storage_init(): create_buffer_init's mapped-at-creation path forces
            // weights into an inefficient memory type on a non-ReBAR GPU, ballooning
            // e.g. a 16.8 GB encoder to ~30 GB (OOM). DEVICE_LOCAL buffers pack
            // tightly — the difference between the Qwen encoder fitting a 24 GB card
            // or not. (Same fix as zimage's BlockWeights::upload.)
            let wbuf = gpu.storage(*numel as u64);
            // Pull exactly this tensor from the source, upload it, and let the host
            // f32 (a streaming WeightReader's per-tensor decode) drop on return —
            // never a whole-model host map. Numerics are byte-identical to before.
            let mut found = false;
            source.with_tensor(name, &mut |data| {
                assert_eq!(data.len(), *numel, "size mismatch for {name}");
                let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
                gpu.write(&wbuf, &bits);
                found = true;
            });
            if !found {
                panic!("missing init weight {name}");
            }
            // Reclaim the write_buffer staging NOW, before the next weight — else it
            // accrues (wgpu only frees it on poll_wait), so a 16.8 GB model uploads
            // ~16.8 GB of extra staging on top of the weights and OOMs a 24 GB card.
            // Peak staging is then just this one tensor. (The DiT does the same via
            // poll_wait per block.)
            gpu.poll_wait();
            // poll_wait alone is not always enough: with no submitted compute the
            // poll can be a no-op and wgpu keeps holding the write_buffer staging
            // (observed OOM on a non-ReBAR P40 at ~14 GiB uploaded of a ~12 GiB
            // shard). A 1-element readback forces a real submit + drain, which is
            // what reclaims the staging — the same pattern as flux2's
            // Flux2Model::new upload. ~Every GiB keeps the cost negligible.
            uploaded += 4 * *numel as u64;
            if uploaded > (1 << 30) {
                let _ = gpu.read(&wbuf, 1);
                uploaded = 0;
            }
            weight.insert(name.clone(), wbuf);
            match role {
                Role::Trainable => {
                    let z = vec![0.0f32; *numel];
                    grad.insert(name.clone(), gpu.storage_init(name, &z));
                    adam_m.insert(name.clone(), gpu.storage_init(name, &z));
                    adam_v.insert(name.clone(), gpu.storage_init(name, &z));
                    trainable.push((name.clone(), *numel));
                }
                Role::Offload => {
                    // grad on GPU (backward writes it); moments live in RAM.
                    let z = vec![0.0f32; *numel];
                    grad.insert(name.clone(), gpu.storage_init(name, &z));
                    offload.push((name.clone(), *numel));
                }
                Role::Frozen => {}
            }
            params.push((name.clone(), *numel));
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
}
