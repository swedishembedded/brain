// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! AdamW with optional global grad-norm clipping, matching
//! `torch.optim.AdamW` + `torch.nn.utils.clip_grad_norm_`.
//!
//! Generic over either model: it only needs the pipeline indices of the
//! `adamw`, `gradnorm_sq`, and `grad_scale` kernels in that model's `Gpu`.

use std::cell::RefCell;

use gpu_core::{f, Gpu, Step};
use paramstore::ParamStore;
#[cfg(not(target_arch = "wasm32"))]
pub mod offload;
pub use offload::OffloadAdam;

/// The optimiser dispatch graph, built once and reused. The bind groups (and
/// the storage buffers they reference) are fixed; only the uniform *contents*
/// change between steps (lr, bias corrections, clip/scale factors), so each
/// dispatch's uniform is a persistent writable buffer updated via `gpu.write`.
/// Rebuilding these per step is what exhausts the GPU memory aperture and
/// triggers a device reset after a few thousand iterations.
struct Graph {
    clipped: bool,
    has_scale: bool,
    steps: Vec<Step>,
    // writable uniforms, in dispatch order within each group:
    clip_uni: Option<gpu_core::DeviceBuffer>, // clip-coef stage (clip path)
    scale_unis: Vec<gpu_core::DeviceBuffer>,  // per-param grad_scale (no-clip path)
    adamw_unis: Vec<gpu_core::DeviceBuffer>,  // per-param AdamW
    // constant uniforms kept alive for the lifetime of their bind groups:
    _const_unis: Vec<gpu_core::DeviceBuffer>,
}

pub struct Optim {
    pub adamw: usize,
    pub gradnorm_sq: usize,
    pub grad_scale: usize,      // scale by a host-supplied constant (no-clip path)
    pub clip_coef: usize,       // compute clip coefficient on-device
    pub grad_scale_buf: usize,  // scale by a device-resident coefficient
    cache: RefCell<Option<Graph>>,
}

impl Optim {
    pub fn new(
        adamw: usize,
        gradnorm_sq: usize,
        grad_scale: usize,
        clip_coef: usize,
        grad_scale_buf: usize,
    ) -> Optim {
        Optim { adamw, gradnorm_sq, grad_scale, clip_coef, grad_scale_buf, cache: RefCell::new(None) }
    }

    /// Build the dispatch graph for the given mode. `clipped`/`has_scale` select
    /// which grad-scaling stages precede AdamW; the structure is otherwise fixed.
    fn build(&self, gpu: &Gpu, ps: &ParamStore, clipped: bool, has_scale: bool) -> Graph {
        let mut steps = Vec::new();
        let mut const_unis = Vec::new();
        let mut clip_uni = None;
        let mut scale_unis = Vec::new();
        let mut adamw_unis = Vec::new();

        if clipped {
            // grad-norm -> clip coefficient -> scale, all in this pass: each stage
            // depends on the previous via storage buffers, which wgpu barriers.
            for (i, (name, numel)) in ps.opt_params().iter().enumerate() {
                let ub = gpu.uniform_dynamic(2);
                gpu.write(&ub, &[*numel as u32, i as u32]);
                steps.push(gpu.step_buf(self.gradnorm_sq, &ub, &[ps.g(name), &ps.norms], 1));
                const_unis.push(ub);
            }
            let cu = gpu.uniform_dynamic(3);
            steps.push(gpu.step_buf(self.clip_coef, &cu, &[&ps.norms, &ps.clip_coef], 1));
            clip_uni = Some(cu);
            for (name, numel) in ps.opt_params() {
                let ub = gpu.uniform_dynamic(1);
                gpu.write(&ub, &[*numel as u32]);
                steps.push(gpu.step_buf(self.grad_scale_buf, &ub, &[ps.g(name), &ps.clip_coef], *numel as u32));
                const_unis.push(ub);
            }
        } else if has_scale {
            // no clip: scale by the accumulation factor (uniform updated per step).
            for (name, numel) in ps.opt_params() {
                let ub = gpu.uniform_dynamic(2);
                steps.push(gpu.step_buf(self.grad_scale, &ub, &[ps.g(name)], *numel as u32));
                scale_unis.push(ub);
            }
        }

        for (name, numel) in ps.opt_params() {
            let ub = gpu.uniform_dynamic(9);
            steps.push(gpu.step_buf(
                self.adamw,
                &ub,
                &[ps.w(name), ps.g(name), &ps.adam_m[name], &ps.adam_v[name]],
                *numel as u32,
            ));
            adamw_unis.push(ub);
        }

        Graph { clipped, has_scale, steps, clip_uni, scale_unis, adamw_unis, _const_unis: const_unis }
    }

    /// One optimiser step at (1-based) step index `t`, run entirely on-device in
    /// a single submit (no host readback). If `clip` is set, gradients are
    /// scaled by `min(1, clip/(global_norm+1e-6)) * extra_scale`
    /// (clip_grad_norm_ semantics, with the accumulation scale folded in);
    /// otherwise just by `extra_scale`.
    pub fn step(
        &self,
        gpu: &Gpu,
        ps: &ParamStore,
        t: u32,
        lr: f32,
        wd: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        clip: Option<f32>,
        extra_scale: f32,
    ) {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        let clipped = clip.is_some();
        let has_scale = clip.is_none() && (extra_scale - 1.0).abs() > 1e-12;

        // (Re)build the cached graph only if absent or the scaling mode changed.
        let need_build = match &*self.cache.borrow() {
            Some(g) => g.clipped != clipped || g.has_scale != has_scale,
            None => true,
        };
        if need_build {
            let g = self.build(gpu, ps, clipped, has_scale);
            *self.cache.borrow_mut() = Some(g);
        }

        let cache = self.cache.borrow();
        let g = cache.as_ref().unwrap();

        // Refresh the per-step uniform contents in place (no allocation).
        if let Some(max_norm) = clip {
            gpu.write(g.clip_uni.as_ref().unwrap(), &[ps.opt_params().len() as u32, f(max_norm), f(extra_scale)]);
        }
        if has_scale {
            for (i, (_, numel)) in ps.opt_params().iter().enumerate() {
                gpu.write(&g.scale_unis[i], &[*numel as u32, f(extra_scale)]);
            }
        }
        for (i, (_, numel)) in ps.opt_params().iter().enumerate() {
            gpu.write(
                &g.adamw_unis[i],
                &[*numel as u32, 0, f(lr), f(beta1), f(beta2), f(eps), f(wd), f(bc1), f(bc2)],
            );
        }
        gpu.submit(&[], &g.steps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paramstore::ParamStore;
    use std::collections::HashMap;

    #[test]
    fn adamw_with_clip_matches_hand_computation() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = Gpu::new(&[
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("grad_scale", kernels::GRAD_SCALE),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ]);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut init = HashMap::new();
        init.insert("p".to_string(), vec![1.0f32; 4]);
        let ps = ParamStore::new(&gpu, vec![("p".to_string(), 4)], &init);
        // grad = 2.0 everywhere => global L2 norm = sqrt(16) = 4 > 1 => clipped by 1/4.
        gpu.write(ps.g("p"), bytemuck::cast_slice(&[2.0f32; 4]));

        opt.step(&gpu, &ps, 1, 0.1, 0.0, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
        let w = ps.read_weight(&gpu, "p");

        // expected: clip to g=0.5; AdamW t=1: mhat=g, vhat=g^2 => step = lr*g/|g| = lr.
        for &v in &w {
            assert!((v - 0.9).abs() < 1e-4, "AdamW+clip got {v}, expected ~0.9");
        }
    }
}
