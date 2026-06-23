// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parameter storage shared by both models: for each named parameter it holds
//! the weight, its gradient, and the AdamW moment buffers, plus a small scratch
//! buffer for grad-norm reduction. Model-agnostic; the optimizer (see `optim`)
//! drives it.

use std::collections::HashMap;

use crate::gpu::Gpu;

pub struct ParamStore {
    pub params: Vec<(String, usize)>,
    pub weight: HashMap<String, wgpu::Buffer>,
    pub grad: HashMap<String, wgpu::Buffer>,
    pub adam_m: HashMap<String, wgpu::Buffer>,
    pub adam_v: HashMap<String, wgpu::Buffer>,
    pub norms: wgpu::Buffer,     // [n_params] sum-of-squares scratch for grad clipping
    pub clip_coef: wgpu::Buffer, // [1] device-resident clip coefficient
}

impl ParamStore {
    pub fn new(gpu: &Gpu, params: Vec<(String, usize)>, init: &HashMap<String, Vec<f32>>) -> ParamStore {
        let mut weight = HashMap::new();
        let mut grad = HashMap::new();
        let mut adam_m = HashMap::new();
        let mut adam_v = HashMap::new();
        for (name, numel) in &params {
            let data = init
                .get(name)
                .unwrap_or_else(|| panic!("missing init weight {name}"));
            assert_eq!(data.len(), *numel, "size mismatch for {name}");
            weight.insert(name.clone(), gpu.storage_init(name, data));
            let z = vec![0.0f32; *numel];
            grad.insert(name.clone(), gpu.storage_init(name, &z));
            adam_m.insert(name.clone(), gpu.storage_init(name, &z));
            adam_v.insert(name.clone(), gpu.storage_init(name, &z));
        }
        let norms = gpu.storage(params.len().max(1) as u64);
        let clip_coef = gpu.storage(1);
        ParamStore { params, weight, grad, adam_m, adam_v, norms, clip_coef }
    }

    pub fn w(&self, name: &str) -> &wgpu::Buffer {
        self.weight.get(name).unwrap_or_else(|| panic!("no weight {name}"))
    }
    pub fn g(&self, name: &str) -> &wgpu::Buffer {
        self.grad.get(name).unwrap()
    }
    pub fn numel(&self, name: &str) -> usize {
        self.params.iter().find(|(n, _)| n == name).unwrap().1
    }

    /// Zero every gradient buffer (call once per effective batch, before the
    /// accumulating backward passes).
    pub fn zero_grads(&self, gpu: &Gpu) {
        let clears: Vec<&wgpu::Buffer> = self.params.iter().map(|(n, _)| self.g(n)).collect();
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
        let gpu = Gpu::new(&[("add2", include_str!("shaders/add2.wgsl"))]);
        let mut init = HashMap::new();
        init.insert("w".to_string(), vec![1.5f32, -2.0, 3.0]);
        let ps = ParamStore::new(&gpu, vec![("w".to_string(), 3)], &init);
        assert_eq!(ps.read_weight(&gpu, "w"), vec![1.5, -2.0, 3.0]);
        assert_eq!(ps.numel("w"), 3);
        // grads start zero; write then zero again
        assert_eq!(ps.read_grad(&gpu, "w"), vec![0.0, 0.0, 0.0]);
        gpu.queue.write_buffer(ps.g("w"), 0, bytemuck::cast_slice(&[9.0f32, 9.0, 9.0]));
        assert_eq!(ps.read_grad(&gpu, "w"), vec![9.0, 9.0, 9.0]);
        ps.zero_grads(&gpu);
        assert_eq!(ps.read_grad(&gpu, "w"), vec![0.0, 0.0, 0.0]);
    }
}
