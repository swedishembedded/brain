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
    pub norms: gpu_core::DeviceBuffer,     // [n_trainable] sum-of-squares scratch for grad clipping
    pub clip_coef: gpu_core::DeviceBuffer, // [1] device-resident clip coefficient
    /// Params optimised off-device (grad on GPU, moments in RAM). Disjoint from
    /// `trainable`; the GPU optimiser skips these, `OffloadAdam` handles them.
    pub offload: Vec<(String, usize)>,
}

impl ParamStore {
    /// All-trainable store (every parameter gets grad + AdamW moments).
    pub fn new(gpu: &Gpu, params: Vec<(String, usize)>, init: &HashMap<String, Vec<f32>>) -> ParamStore {
        let roles: Vec<(String, usize, Role)> =
            params.into_iter().map(|(n, c)| (n, c, Role::Trainable)).collect();
        Self::new_with_roles(gpu, roles, init)
    }

    /// Role-aware store: `Frozen` parameters allocate weights only. Used for
    /// inference (all frozen) and LoRA (frozen base + trainable adapters).
    pub fn new_with_roles(
        gpu: &Gpu,
        params_roles: Vec<(String, usize, Role)>,
        init: &HashMap<String, Vec<f32>>,
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
            let data = init
                .get(name)
                .unwrap_or_else(|| panic!("missing init weight {name}"));
            assert_eq!(data.len(), *numel, "size mismatch for {name}");
            // storage()+write() (plain DEVICE_LOCAL + transient staging) instead of
            // storage_init(): create_buffer_init's mapped-at-creation path forces
            // weights into an inefficient memory type on a non-ReBAR GPU, ballooning
            // e.g. a 16.8 GB encoder to ~30 GB (OOM). DEVICE_LOCAL buffers pack
            // tightly — the difference between the Qwen encoder fitting a 24 GB card
            // or not. (Same fix as zimage's BlockWeights::upload.)
            let wbuf = gpu.storage(*numel as u64);
            let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
            gpu.write(&wbuf, &bits);
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
        let norms = gpu.storage(trainable.len().max(1) as u64);
        let clip_coef = gpu.storage(1);
        ParamStore { params, trainable, weight, grad, adam_m, adam_v, norms, clip_coef, offload }
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
