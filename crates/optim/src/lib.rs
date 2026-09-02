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

use std::cell::RefCell;

use gpu_core::{f, Gpu, Step};
use paramstore::ParamStore;
#[cfg(not(target_arch = "wasm32"))]
pub mod offload;
pub use offload::OffloadAdam;

/// The optimiser dispatch graph, built once and reused. The bind groups (and
/// the storage buffers they reference) are fixed; only the uniform *contents*
/// change between steps (lr, bias corrections, clip factor), so each
/// dispatch's uniform is a persistent writable buffer updated via `gpu.write`.
/// Rebuilding these per step is what exhausts the GPU memory aperture and
/// triggers a device reset after a few thousand iterations.
struct Graph {
    clipped: bool,
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
    // constant/write-once buffers kept alive for the lifetime of their bind
    // groups: gradnorm's per-tensor uniforms, the unit coef buffer (unclipped
    // mode), and every tensor's `numel` descriptor.
    _const_unis: Vec<gpu_core::DeviceBuffer>,
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
        // `desc` is this tensor's element count - written once, here, never
        // again (it cannot change for the life of this graph).
        for (name, numel) in ps.opt_params() {
            let desc = gpu.storage(1);
            gpu.write(&desc, &[*numel as u32]);
            steps.push(gpu.step_buf(
                self.adamw,
                &hparams,
                &[ps.w(name), ps.g(name), &ps.adam_m[name], &ps.adam_v[name], &desc, &coef_buf],
                *numel as u32,
            ));
            const_unis.push(desc);
        }

        Graph { clipped, n_norm_inputs, steps, clip_uni, hparams, _const_unis: const_unis }
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

        // (Re)build the cached graph only if absent or the clip mode changed
        // (the only thing that changes the dispatch SHAPE - see `build`).
        let need_build = match &*self.cache.borrow() {
            Some(g) => g.clipped != clipped,
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
        if let Some(max_norm) = clip {
            gpu.write(g.clip_uni.as_ref().unwrap(), &[g.n_norm_inputs, f(max_norm), f(extra_scale)]);
        }
        // `scale` folds `extra_scale` in directly only when there is no
        // device-computed clip coefficient to fold it into instead (`build`
        // already folded it into `clip_coef`'s own `extra_scale` field).
        let scale = if clipped { 1.0 } else { extra_scale };
        gpu.write(&g.hparams, &[f(lr), f(beta1), f(beta2), f(eps), f(wd), f(bc1), f(bc2), f(scale)]);

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
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
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
        opt.step(&gpu, &ps, 1, 0.1, 0.0, 0.9, 0.999, 1e-8, None, 0.25);
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
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
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
            opt.step(&gpu, &ps, 1, 0.01, 0.0, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
            let (d0, w0) = {
                let s = stats();
                (s.dispatches, s.writes)
            };
            opt.step(&gpu, &ps, 2, 0.01, 0.0, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
            let s = stats();
            (s.dispatches - d0, s.writes - w0, n_tensors as u64)
        };
        for n in [3usize, 9] {
            let (dispatches, writes, p) = run(n);
            assert_eq!(dispatches, 2 * p + 1, "P={p}: expected 2P+1 dispatches (grad-scale folded into AdamW), got {dispatches}");
            assert!(writes <= 2, "P={p}: expected O(1) (<=2: clip_uni + shared hyperparams) writes per step, got {writes}");
        }
    }
}
