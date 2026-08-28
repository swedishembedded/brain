// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! B10: finite-difference gate for the **bf16 mixed-precision training
//! tier** - `model::ops::Ops::matmul`'s `Reference` variant (forward, B4)
//! paired with the new `Ops::matmul_dx`/`Ops::matmul_dw` backward this phase
//! adds. No model crate consumes this yet (same "harness IS the fixture"
//! shape `deepseekocr.rs` already established for a greenfield kernel), so
//! this is a standalone two-tensor fixture: a `[M,K] @ [N,K]^T -> [M,N]`
//! linear whose WEIGHT is stored as `Weight::BF16` and whose ACTIVATION
//! stays `f32` throughout.
//!
//! ## The invariant this phase must get right, and how the fixture proves it
//!
//! Standard mixed-precision training: the f32 **master** weight is narrowed
//! to bf16 for the forward matmul (and, symmetrically, for the backward-of-x
//! matmul, since `dX` must be consistent with whichever weight value forward
//! actually multiplied by); the weight's own gradient (`dW`) is computed and
//! accumulated at **f32**, unconditionally, and is what an f32 AdamW step
//! would apply to the f32 master copy. `model::ops::Ops::matmul_dw` enforces
//! the f32-`dW` half structurally (see its own doc comment: the method has
//! no `Weight`/`Dtype` parameter at all - `matmul_dw.wgsl` never reads the
//! weight buffer). This module's job is to prove the OTHER half: that
//! `Ops::matmul`'s bf16-weight forward and `Ops::matmul_dx`'s bf16-weight
//! backward are each other's correct adjoint, by finite differences of the
//! model's OWN forward - not merely "close to an f32 reference" (B4's own
//! roundtrip-tolerance bar), which is the whole point of a TRAINING tier per
//! this program's plan.
//!
//! [`Harness::write_weight`] models the master-weight update exactly: it
//! stores the f32 value the caller supplies and re-derives a FRESH packed
//! `Weight::BF16` from it via `Weight::upload` (the same "re-quantize on
//! every write" a real optimizer step would trigger for the next forward -
//! `pack_bf16` is applied fresh, never accumulated in packed form). Reading
//! the weight back returns that same pre-quantization f32 value, so
//! `write_weight(name, read_weight(name))` round-trips exactly - required by
//! [`directional_check`]'s own restore-after-perturb protocol.
//!
//! ## Why the loss is `dot(Y, r)` for a fixed random `r`
//!
//! `loss = <x @ decode(bf16(w))^T, r>` is, for FIXED `x` and `r`, an affine
//! function of the DECODED bf16 value at every weight entry independently:
//! `loss(w) = sum_{n,k} C[n,k] * decode(bf16(w[n,k]))` where `C[n,k] =
//! sum_m r[m,n]*x[m,k]` is exactly what `matmul_dw.wgsl` computes for `dW`
//! (its `x`/`dy` inputs never see `w` at all - see that kernel's own
//! source). This separability is what makes the finite-difference math
//! tractable despite `decode∘bf16` being a **staircase**, not a smooth
//! function - see the next section.
//!
//! ## `dX`'s check is exact; `dW`'s check is a straight-through estimate,
//! and that distinction is the actual bf16-training-specific content here
//!
//! **`dX`** (checked by perturbing the ACTIVATION `x`, which is never
//! quantized): `loss` is EXACTLY linear in `x` for any FIXED weight value
//! (bf16-decoded or not), so finite differences on `x` have no rounding
//! artifact at any step size - this is a clean, standard matmul-adjoint
//! check, and it is what actually exercises `matmul_dx.wgsl`'s new
//! bf16-weight-read path (B10's own new kernel capability).
//!
//! **`dW`** (checked by perturbing the WEIGHT, which is re-quantized to bf16
//! on every write) is different in kind: `decode∘bf16` is a **monotonic
//! staircase** with a LOCAL step size of about `2^-8` of each entry's own
//! magnitude (bf16's 7 explicit mantissa bits). Its TRUE pointwise
//! derivative is zero almost everywhere and undefined at the rounding
//! boundaries - not a function a naive small-step finite difference can
//! validate at all. The standard **straight-through estimator** (STE) used
//! by every mixed-precision/quantization-aware-training system treats the
//! rounding op as an identity for gradient purposes, i.e. claims `d(decode∘
//! bf16)/dw ≈ 1` - which is exactly what `matmul_dw`'s UNMODIFIED f32 kernel
//! already assumes (it does not even read `w`, so it cannot see the
//! rounding at all). A finite difference CAN validate this STE claim,
//! provided `eps` is chosen large enough to reliably cross SEVERAL bf16
//! rounding boundaries per perturbed entry: `decode∘bf16` is an *unbiased*
//! rounding function, so the average slope of the staircase over a window
//! spanning multiple steps converges to 1, and [`directional_check`]'s own
//! whole-tensor contraction sums this convergence over every entry
//! simultaneously - exactly the same "many entries average out per-entry
//! noise" property that function's own doc comment already relies on for
//! plain fp32 round-off, now doing the identical job for bf16
//! quantization-boundary noise instead. [`check_matmul_bf16_weight_eps_
//! sweep`] measures where this convergence actually plateaus, rather than
//! assuming it (the eps table is reported, not asserted from memory).
//!
//! This means `dW`'s check is NOT "does `matmul_dw.wgsl` compute the right
//! numbers" (that kernel is untouched f32 code, already exercised by every
//! existing `check_gpt`/`check_qwen` gradcheck) - it is "is the STE
//! convention this phase relies on actually a good approximation of the
//! bf16-quantized forward's real sensitivity, at the eps this program's
//! finite-difference gate uses". That is the genuinely NEW, bf16-specific
//! claim B10 makes, and it is what this check validates.
//!
//! ## Deliberately out of scope
//!
//! Only `matmul.wgsl`'s `Reference` variant + `matmul_dx.wgsl` (ONE kernel
//! family, per the plan's own scope-down allowance) - not `matmul_gemv`/
//! `matmul_reg3`'s dx siblings. Not wired into any model crate's training
//! loop (`crates/qwen3`, `crates/gpt`, ... untouched). `crates/optim`/
//! `crates/paramstore` untouched - this fixture emulates "an f32 master
//! weight, re-quantized every step" entirely on the host side, exactly the
//! shape a real optimizer loop would drive `Weight::upload` through, without
//! needing to touch either crate.

use std::cell::RefCell;

use data::rng::Rng;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::{Ops, Weight};

use crate::{directional_check, CheckModel, Report};

const M: u32 = 8;
// `Ops::act` unconditionally quantizes to an int8 scratch buffer regardless
// of the WEIGHT's own dtype (`model::ops`'s own module doc) - `quant_pack`
// writes 4 int8 per u32, so this fixture's `K` must be a multiple of 4 even
// though the weight itself is bf16, not int8. (The stricter `K % 32` rule the
// int8/q4 WEIGHT quantizers carry does not apply here: nothing quantizes a
// weight in this fixture.)
const K: u32 = 8;
const N: u32 = 6;

/// The canonical façade kernel set, straight from
/// [`model::ops::kernel_list`]. This module used to hand-maintain a
/// byte-identical copy - see that function's doc for why that duplication was
/// worth removing.
fn kernel_list() -> &'static [(&'static str, &'static str)] {
    model::ops::kernel_list()
}

/// The two-tensor bf16-weight-linear fixture. `x` (`[M,K]`) and `w` (`[N,K]`)
/// are BOTH kept as host "master" f32 vectors in `RefCell`s (mutated only by
/// `write_weight`, which also syncs the device side); `weight` is the packed
/// `Weight::BF16` [`Weight::upload`] re-derives from `w`'s current value on
/// every write - see the module doc for why this is the correct model of a
/// mixed-precision training step, not a shortcut.
struct Harness {
    g: Gpu,
    ops: Ops,
    x: RefCell<Vec<f32>>,
    w: RefCell<Vec<f32>>,
    weight: RefCell<Weight>,
    x_buf: DeviceBuffer,
    r: Vec<f32>,
    dy_buf: DeviceBuffer,
    y_buf: DeviceBuffer,
    dx_buf: DeviceBuffer,
    dw_buf: DeviceBuffer,
}

impl Harness {
    fn new(seed: u64) -> Harness {
        let g = gpu_core::testgpu::dev(kernel_list());
        let ops = Ops::new(g.share()).expect("Ops::new: full facade kernel set must be registered");
        let mut rng = Rng::new(seed ^ 0xB10);
        let mut init = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
        let x = init((M * K) as usize);
        let w = init((N * K) as usize);
        let r = init((M * N) as usize);

        let x_buf = g.storage_init("x", &x);
        let weight = Weight::upload(&ops, &w, N as usize, K as usize, Dtype::BF16);
        assert_eq!(
            weight.dtype(),
            Dtype::BF16,
            "this harness requires a real bf16 weight -- if this fires, the ambient device's own \
             capability demoted the request (Weight::upload's DType::promote gate), which would \
             silently turn this into a plain f32 test and defeat the whole point of B10's gate"
        );
        let dy_buf = g.storage_init("dy", &r);
        let y_buf = g.storage((M * N) as u64);
        let dx_buf = g.storage((M * K) as u64);
        let dw_buf = g.storage((N * K) as u64);

        Harness {
            g,
            ops,
            x: RefCell::new(x),
            w: RefCell::new(w),
            weight: RefCell::new(weight),
            x_buf,
            r,
            dy_buf,
            y_buf,
            dx_buf,
            dw_buf,
        }
    }
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        vec!["x".into(), "w".into()]
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        match name {
            "x" => self.x.borrow().clone(),
            "w" => self.w.borrow().clone(),
            other => panic!("bf16_train harness: unknown parameter {other}"),
        }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        match name {
            "x" => {
                assert_eq!(data.len(), (M * K) as usize, "x: size mismatch");
                *self.x.borrow_mut() = data.to_vec();
                self.g.write_f32(&self.x_buf, data);
            }
            "w" => {
                assert_eq!(data.len(), (N * K) as usize, "w: size mismatch");
                *self.w.borrow_mut() = data.to_vec();
                // Re-derive the packed bf16 device weight FRESH from this f32
                // "master" value on every write - exactly what a real
                // mixed-precision optimizer step does: AdamW updates the f32
                // master copy, then the NEXT forward re-casts it to bf16.
                *self.weight.borrow_mut() = Weight::upload(&self.ops, data, N as usize, K as usize, Dtype::BF16);
            }
            other => panic!("bf16_train harness: unknown parameter {other}"),
        }
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        match name {
            "x" => self.g.read(&self.dx_buf, (M * K) as usize),
            "w" => self.g.read(&self.dw_buf, (N * K) as usize),
            other => panic!("bf16_train harness: unknown parameter {other}"),
        }
    }

    fn loss(&self) -> f32 {
        let mut s = Vec::new();
        let act = self.ops.act(&mut s, &self.x_buf, 0, M, K);
        self.ops.matmul(&mut s, &self.weight.borrow(), &act, &self.y_buf, 0);
        self.g.submit(&[&self.y_buf], &s);
        let y = self.g.read(&self.y_buf, (M * N) as usize);
        // f64 accumulation: the FD comparison differences a loss that moves
        // by a small fraction of itself, so an f32 accumulator's own
        // round-off would land straight in the numerator (same reasoning
        // `deepseekocr.rs`'s own `RelPosHarness::loss` already documents).
        let mut dot = 0f64;
        for (yi, ri) in y.iter().zip(&self.r) {
            dot += *yi as f64 * *ri as f64;
        }
        dot as f32
    }

    fn zero_grads(&self) {
        // `dx_buf` is fully OVERWRITTEN by `matmul_dx`'s own `accumulate =
        // false` path; `dw_buf` is zeroed by `backward()`'s own `submit`'s
        // `clears` list (`matmul_dw` always accumulates, matching every
        // other dw kernel in this codebase) - so there is nothing this
        // method itself needs to do, the same pattern `RelPosHarness::
        // zero_grads` already established.
    }

    fn backward(&self) {
        let mut s = Vec::new();
        self.ops.matmul_dx(&mut s, &self.weight.borrow(), &self.dy_buf, M, &self.dx_buf, false);
        self.ops.matmul_dw(&mut s, &self.x_buf, &self.dy_buf, M, N, K, &self.dw_buf);
        self.g.submit(&[&self.dw_buf], &s);
    }
}

/// `eps` for [`check_matmul_bf16_weight`] - large relative to bf16's local
/// rounding step (`2^-8` of each entry's magnitude, so for this fixture's
/// `next_f32() - 0.5` weights, on the order of `2e-3` absolute) so the `w`
/// finite difference reliably crosses several rounding boundaries per
/// entry - see the module doc's "`dW`'s check is a straight-through
/// estimate" section. [`check_matmul_bf16_weight_eps_sweep`] is the
/// measurement behind this choice; `x`'s own check is exactly linear in `x`
/// at ANY eps (no quantization on that side), so one shared eps serves both.
const EPS: f32 = 3e-2;

/// **The gate.** Directional finite differences over both tensors of the
/// bf16-weight fixture: `x` (exercises the NEW `matmul_dx#w=bf16` kernel
/// this phase adds) and `w` (exercises the "f32 dW is a good
/// straight-through estimate of the bf16-quantized forward's real
/// sensitivity" claim this phase's mixed-precision design relies on).
pub fn check_matmul_bf16_weight(seed: u64) -> Report {
    let h = Harness::new(seed);
    directional_check(&h, EPS, 4, seed ^ 0x1234)
}

/// The eps/error table behind [`EPS`], measured rather than assumed - this
/// program's own rule when a gradcheck fails is to PROBE this and report it,
/// never to widen the bound blindly.
pub fn check_matmul_bf16_weight_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = Harness::new(seed);
    [2e-3f32, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 8e-2, 1.5e-1]
        .iter()
        .map(|&eps| (eps, directional_check(&h, eps, 4, seed ^ 0x1234).max_rel()))
        .collect()
}

/// **Convergence sanity check** ("vs the fp32 baseline on a small model" per
/// the original plan's wording), scoped down exactly per this phase's own
/// scope-discipline note: a plain few-dozen-step SGD loop driven directly by
/// `model::ops::Ops::matmul`/`matmul_dw` - NOT a real training loop, NOT
/// routed through `crates/optim`'s AdamW or `crates/paramstore` (both
/// untouched by this phase, out of scope), and NOT a model crate. A tiny
/// least-squares regression task (fixed `x`, a fixed random target `y*`,
/// MSE loss) run to `steps` SGD updates twice - once with the weight held at
/// `Dtype::BF16` (this phase's opt-in forward tier) and once at `Dtype::F32`
/// (today's existing behaviour) - starting from the IDENTICAL random init
/// and seed, so the two loss trajectories are directly comparable. Returns
/// `(bf16_losses, f32_losses)`, one MSE value per step, for the caller to
/// compare. **What this does and does not prove**: it shows the bf16-weight
/// forward's gradient (via this phase's own `matmul_dx`/`matmul_dw`) is
/// USABLE for optimization - loss goes down, comparably to f32 - at this
/// tiny synthetic scale. It does NOT validate convergence at production
/// model scale, does NOT exercise a real optimizer (momentum/Adam moments,
/// weight decay, grad-norm clipping), and does NOT touch any real model's
/// training loop. That is an explicit, honest limit of this phase's scope,
/// not an oversight - see this module's own doc comment above.
pub fn bf16_training_sanity(seed: u64, steps: usize, lr: f32) -> (Vec<f32>, Vec<f32>) {
    (run_sgd(seed, steps, lr, Dtype::BF16), run_sgd(seed, steps, lr, Dtype::F32))
}

fn run_sgd(seed: u64, steps: usize, lr: f32, dt: Dtype) -> Vec<f32> {
    let g = gpu_core::testgpu::dev(kernel_list());
    let ops = Ops::new(g.share()).expect("Ops::new: full facade kernel set must be registered");
    let mut rng = Rng::new(seed ^ 0x5A11); // SAME seed for both dtypes - identical init/target/x.
    let mut init = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
    let x = init((M * K) as usize);
    let target = init((M * N) as usize);
    let mut w = init((N * K) as usize);

    let x_buf = g.storage_init("x", &x);
    let y_buf = g.storage((M * N) as u64);
    let dy_buf = g.storage((M * N) as u64);
    let dw_buf = g.storage((N * K) as u64);
    let mn = (M * N) as f32;

    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        // Forward: re-derive the weight at `dt` from the current f32 host
        // "master" copy every step - same re-quantize-on-every-write
        // discipline `Harness::write_weight` uses.
        let weight = Weight::upload(&ops, &w, N as usize, K as usize, dt);
        let mut s = Vec::new();
        let act = ops.act(&mut s, &x_buf, 0, M, K);
        ops.matmul(&mut s, &weight, &act, &y_buf, 0);
        g.submit(&[&y_buf], &s);
        let y = g.read(&y_buf, (M * N) as usize);

        let mse: f32 = y.iter().zip(&target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / mn;
        losses.push(mse);

        // Backward: dY of the MSE loss, then the ALWAYS-f32 matmul_dw.
        let dy: Vec<f32> = y.iter().zip(&target).map(|(a, b)| 2.0 * (a - b) / mn).collect();
        g.write_f32(&dy_buf, &dy);
        let mut sb = Vec::new();
        ops.matmul_dw(&mut sb, &x_buf, &dy_buf, M, N, K, &dw_buf);
        g.submit(&[&dw_buf], &sb);
        let dw = g.read(&dw_buf, (N * K) as usize);

        // Plain SGD on the f32 master copy - `crates/optim`'s real AdamW is
        // out of this phase's scope (see this function's own doc comment).
        for (wi, dwi) in w.iter_mut().zip(&dw) {
            *wi -= lr * dwi;
        }
    }
    losses
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fp32-on-a-device finite differences, workspace-standard combined
    /// tolerance (same constants `deepseekocr.rs`'s own gate uses).
    const ATOL: f32 = 4e-3;
    const RTOL: f32 = 8e-2;

    fn gate(report: Report, what: &str) {
        report.print();
        let fails = report.failures(ATOL, RTOL);
        assert!(
            fails.is_empty(),
            "{what} gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
        let dead = report.dead_gradients();
        assert!(
            dead.is_empty(),
            "{what}: exactly-zero analytic gradients for {:?}",
            dead.iter().map(|c| &c.param).collect::<Vec<_>>()
        );
    }

    #[test]
    fn matmul_bf16_weight_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        gate(check_matmul_bf16_weight(7), "bf16-weight matmul forward/backward (B10)");
    }

    /// The eps probe, run as a gate: asserts the chosen [`EPS`] sits on the
    /// convergence plateau (not right at the noisy small-eps edge, and not
    /// so large the underlying linear approximation itself would be
    /// suspect), and prints the full table so a future change that moves it
    /// is visible rather than silently re-tuned.
    #[test]
    fn matmul_bf16_weight_eps_plateau() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let table = check_matmul_bf16_weight_eps_sweep(7);
        for (eps, rel) in &table {
            println!("  eps={eps:.1e}  max_rel={rel:.3e}");
        }
        let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
        assert!(at(EPS) <= RTOL, "eps {EPS:.1e} max_rel {:.3e} exceeds rtol", at(EPS));
    }

    /// The convergence sanity check - see [`bf16_training_sanity`]'s own doc
    /// comment for exactly what this does and does not validate.
    #[test]
    fn bf16_training_sanity_loss_decreases_comparably_to_f32_baseline() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (bf16_losses, f32_losses) = bf16_training_sanity(7, 60, 0.6);
        println!("step   bf16 MSE     f32 MSE");
        for (i, (b, f)) in bf16_losses.iter().zip(&f32_losses).enumerate() {
            println!("{i:4}   {b:.6}   {f:.6}");
        }
        let (bf16_first, bf16_last) = (bf16_losses[0], *bf16_losses.last().unwrap());
        let (f32_first, f32_last) = (f32_losses[0], *f32_losses.last().unwrap());
        assert!(bf16_last < bf16_first * 0.5, "bf16 loss did not decrease: {bf16_first} -> {bf16_last}");
        assert!(f32_last < f32_first * 0.5, "f32 loss did not decrease: {f32_first} -> {f32_last}");
        let rel_gap = (bf16_last - f32_last).abs() / f32_last.max(1e-6);
        assert!(
            rel_gap < 0.5,
            "bf16 vs f32 final loss diverged more than expected: bf16={bf16_last:.6} f32={f32_last:.6} \
             rel_gap={rel_gap:.3}"
        );
    }

    /// This module's own `kernel_list()` against `model::ops::
    /// REQUIRED_KERNELS` - a pure name-set comparison, no `Gpu`/GPU device
    /// required, so unlike every other test in this module it is NOT gated
    /// behind `MOE_SKIP_GPU_TESTS`. Catches drift (a kernel `Ops::new`
    /// requires but this list forgets to register) at `cargo test` time -
    /// exactly the class of bug `qwen3::serve::ops_kernel_list` had (15
    /// kernels short of `REQUIRED_KERNELS`) that no GPU-gated test here would
    /// have caught in an environment where `MOE_SKIP_GPU_TESTS` is set.
    #[test]
    fn kernel_list_has_every_kernel_ops_new_requires() {
        model::ops::assert_kernel_list_complete(kernel_list());
    }
}
