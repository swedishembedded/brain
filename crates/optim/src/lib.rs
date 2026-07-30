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
    /// How many f32s the clip-coefficient stage folds: one per parameter tensor
    /// on the reference path, one per workgroup per tensor on the cooperative
    /// one. Written into the clip uniform every step.
    n_norm_inputs: u32,
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
        let mut n_norm_inputs = 0u32;

        if clipped {
            // grad-norm -> clip coefficient -> scale, all in this pass: each stage
            // depends on the previous via storage buffers, which wgpu barriers.
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
            clip_uni = Some(cu);
            n_norm_inputs = n_reduced;
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

        Graph {
            clipped,
            has_scale,
            n_norm_inputs,
            steps,
            clip_uni,
            scale_unis,
            adamw_unis,
            _const_unis: const_unis,
        }
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
            gpu.write(g.clip_uni.as_ref().unwrap(), &[g.n_norm_inputs, f(max_norm), f(extra_scale)]);
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
        static KERNELS: &[(&str, &str)] = &[
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("grad_scale", kernels::GRAD_SCALE),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ];
        let gpu = gpu_core::testgpu::dev(KERNELS);
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

    /// The cooperative grad-norm path (`gradnorm_part` + `clip_coef_wg`) must
    /// produce the SAME clipped step as the serial one, at sizes that exercise
    /// several workgroups per tensor and several tensors per step. The kernels
    /// are resolved by name, so registering them is the whole opt-in — that is
    /// what this asserts as much as the arithmetic.
    #[test]
    fn cooperative_gradnorm_matches_the_serial_path() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
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
            opt.step(&gpu, &ps, 1, 0.01, 0.01, 0.9, 0.999, 1e-8, Some(0.5), 1.0);
            shapes.iter().map(|(n, _)| ps.read_weight(&gpu, n)).collect()
        };
        let (a, b) = (run(BASE), run(COOP));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let md = x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            assert!(md < 2e-6, "tensor {i}: cooperative grad-norm changed the step by {md}");
        }
    }
}
