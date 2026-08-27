// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient check for `vae::blocks::grad::Op::Mix` (`Builder::mix`): `y = a*x
//! + b*f`, reusing `edm_mix.wgsl`/`scale_row.wgsl` verbatim - SUPIR's
//! ZeroSFT/ZeroCrossAttn `control_scale` lerp is exactly this shape.
//!
//! A tiny graph containing ONE `Op::Mix`, gated the same way
//! `check_vlm_splice` (src/lib.rs) gates an input gradient: a directional
//! central difference of the REAL recorded forward against the analytic
//! `Trace::backward` output, not against a hand-derived formula - so a wrong
//! kernel param order in either `Builder::mix` or `Op::Mix`'s backward would
//! be caught by re-running the actual forward, not just re-deriving the same
//! algebra the implementation used.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use data::rng::Lcg;
use gradcheck::Check;
use vae::blocks::grad::BwdIds;
use vae::blocks::{BlockNames, Builder, MixIds, Tensors};

const K_EDM_MIX: usize = vae::blocks::NEXT_SLOT;
const K_SCALE_ROW: usize = vae::blocks::NEXT_SLOT + 1;
/// Where the shared reverse kernel set ([`vae::blocks::BWD_KERNELS`], which
/// `axpy` - dispatched by every `Op`'s backward, including `Op::Mix`'s -
/// lives inside) sits in [`KERNELS`].
const BWD_BASE: usize = vae::blocks::NEXT_SLOT + 2;
const N_KERNELS: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

static KERNELS: [(&str, &str); N_KERNELS] = kernel_set();
const fn kernel_set() -> [(&'static str, &'static str); N_KERNELS] {
    let mut k = vae::blocks::kernels_with::<N_KERNELS>();
    k[K_EDM_MIX] = ("edm_mix", kernels::EDM_MIX);
    k[K_SCALE_ROW] = ("scale_row", kernels::SCALE_ROW);
    let mut j = 0;
    while j < vae::blocks::BWD_KERNELS.len() {
        k[BWD_BASE + j] = vae::blocks::BWD_KERNELS[j];
        j += 1;
    }
    k
}

/// `edm_mix`'s own documented exactness property: `a=1, b=0` reproduces `x`
/// bit-for-bit (no rounding from a no-op lerp) - cheap to assert directly,
/// no finite differences needed.
#[test]
fn a_one_b_zero_reproduces_x_bit_for_bit() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let tensors: Tensors = std::collections::HashMap::new();
    let mut b = Builder::new(&gpu, &tensors, 1e-5, 32, BlockNames::diffusers(), false);
    b.set_mix_ids(MixIds { fwd: K_EDM_MIX, bwd: K_SCALE_ROW });

    let n = 32u32;
    let mut st = Lcg::new(99u64);
    let x: Vec<f32> = (0..n).map(|_| st.scaled(4.0)).collect();
    let f: Vec<f32> = (0..n).map(|_| st.scaled(4.0)).collect();
    let xb = gpu.storage_init("x", &x);
    let fb = gpu.storage_init("f", &f);

    let y = b.mix(n, 1.0, 0.0, &xb, &fb);
    let (steps, _taps) = b.finish();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let got = gpu.read(&y, n as usize);

    let differing = got.iter().zip(&x).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(differing, 0, "a=1,b=0 must reproduce x bit-for-bit, {differing} of {n} differed");
}

/// The gate: a directional central difference of the real recorded forward
/// against `Trace::backward`'s `Op::Mix` adjoint, on both `x` and `f`.
#[test]
fn op_mix_backward_matches_finite_differences_of_the_real_forward() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let tensors: Tensors = std::collections::HashMap::new();
    let mut builder = Builder::new(&gpu, &tensors, 1e-5, 32, BlockNames::diffusers(), false);
    builder.set_train(true);
    builder.set_mix_ids(MixIds { fwd: K_EDM_MIX, bwd: K_SCALE_ROW });

    let n = 24u32;
    let mut st = Lcg::new(4242u64);
    let x0: Vec<f32> = (0..n).map(|_| st.scaled(2.0)).collect();
    let f0: Vec<f32> = (0..n).map(|_| st.scaled(2.0)).collect();
    let (a, bc) = (0.63f32, -1.4f32);
    let xb = gpu.storage_init("x", &x0);
    let fb = gpu.storage_init("f", &f0);

    let y = builder.mix(n, a, bc, &xb, &fb);
    let trace = builder.trace();
    let (steps, _taps) = builder.finish();

    // L(x, f) = dot(w, mix(x, f, a, b)) for a fixed random weight w - the same
    // "contract to one scalar via a random direction" recipe every other
    // gradcheck in this repo uses.
    let w: Vec<f32> = (0..n).map(|_| st.scaled(1.0)).collect();
    let dw = gpu.storage_init("dw", &w);

    let grads = trace.alloc_grads(&gpu);
    let reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, &y, &dw);
    let clears: Vec<&gpu_core::DeviceBuffer> = reverse.clears.iter().collect();
    gpu.submit(&clears, &reverse.steps);
    gpu.poll_wait();
    let dx = reverse.d(&xb).expect("Op::Mix must reach x").clone();
    let df = reverse.d(&fb).expect("Op::Mix must reach f").clone();
    let dx = gpu.read(&dx, n as usize);
    let df = gpu.read(&df, n as usize);

    let forward = |xv: &[f32], fv: &[f32]| -> f32 {
        gpu.write_f32(&xb, xv);
        gpu.write_f32(&fb, fv);
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let out = gpu.read(&y, n as usize);
        out.iter().zip(&w).map(|(o, wi)| o * wi).sum()
    };

    let eps = 5e-3f32;
    let mut checks = Vec::new();
    for _ in 0..4 {
        let vx: Vec<f32> = (0..n).map(|_| if st.signed() < 0.0 { -1.0 } else { 1.0 }).collect();
        let vf: Vec<f32> = (0..n).map(|_| if st.signed() < 0.0 { -1.0 } else { 1.0 }).collect();

        let analytic: f32 =
            dx.iter().zip(&vx).map(|(d, v)| d * v).sum::<f32>() + df.iter().zip(&vf).map(|(d, v)| d * v).sum::<f32>();

        let xp: Vec<f32> = x0.iter().zip(&vx).map(|(x, v)| x + eps * v).collect();
        let fp: Vec<f32> = f0.iter().zip(&vf).map(|(f, v)| f + eps * v).collect();
        let lp = forward(&xp, &fp);

        let xm: Vec<f32> = x0.iter().zip(&vx).map(|(x, v)| x - eps * v).collect();
        let fm: Vec<f32> = f0.iter().zip(&vf).map(|(f, v)| f - eps * v).collect();
        let lm = forward(&xm, &fm);

        let numeric = (lp - lm) / (2.0 * eps);
        let abs_err = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        checks.push(Check { param: "mix(x,f)".into(), analytic, numeric, abs_err, rel_err: abs_err / denom });
    }

    for c in &checks {
        assert!(c.within(4e-3, 8e-2), "Op::Mix backward vs FD: {c:?}");
    }
}
