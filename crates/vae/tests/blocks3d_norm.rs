// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FUSED channel-axis norm (`l2norm_scale2d`) computes the SAME function
//! as the composed `nchw_nlc` -> `l2norm_scale` -> `nlc_nchw` form - bit for
//! bit, not within a tolerance.
//!
//! Swedish Embedded AB implements correctness gates for fused GPU kernels for
//! its clients. If your team needs expertise in proving that a kernel rewrite
//! changed the schedule and nothing else, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Bits, and not a cosine floor, because that is exactly the statement the
//! code makes: the two permutes are exact rearrangements, and both arms fold a
//! position's sum of squares over ASCENDING channel index, so the fused kernel
//! performs the identical sequence of roundings on the identical values. A
//! tolerance would be a weaker claim than the one being made, and - since a
//! cosine is scale invariant - would not even see a uniformly mis-scaled
//! result.
//!
//! Checkpoint-free, so it runs in every lane, and run on the CPU JIT as well
//! as on whatever `Gpu::new` selects: `l2norm_scale2d` is barrier-free and
//! array-free and therefore declares `@cpu yes`, which is a claim about
//! `backend-cpu` that only a run on `backend-cpu` can hold up.
//!
//! Both of the builder's norms are covered, because they are two different
//! `(gain, eps)` pairs into one dispatch site: [`Builder3d::pixel_norm`] (a
//! synthesized uniform gain, the LTX video VAE's `PixelNorm`) and
//! [`Builder3d::rms_norm`] (a learned gain, the Wan VAE's norm). A shape whose
//! four extents are all different is used deliberately, so an axis swap cannot
//! hide behind a square.

use std::collections::HashMap;

use data::rng::Lcg;
use gpu_core::Gpu;
use vae::blocks3d::{Builder3d, T3, KERNELS};

const SWITCH: &str = "BRAIN_VAE3D_SPLIT_NORM";

/// The arm is a process-wide environment variable read once per
/// [`Builder3d::new`], so two tests must not be in opposite arms at the same
/// time - the suite runs a binary's tests in parallel, and one test clearing
/// the variable while another relies on it would silently compare an arm
/// against itself.
static ARM: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn without_fusion<T>(f: impl FnOnce() -> T) -> T {
    std::env::set_var(SWITCH, "1");
    let out = f();
    std::env::remove_var(SWITCH);
    out
}

/// `0.0 == -0.0` is true for two different results and `NaN == NaN` is false
/// for the same one, so `==` over `f32` is not the relation "these two graphs
/// produced the same answer".
fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "length mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// `[c, t, h, w]`, every extent different so a transposed axis cannot pass.
const SHAPE: (u32, u32, u32, u32) = (48, 3, 5, 7);

fn input() -> Vec<f32> {
    let (c, t, h, w) = SHAPE;
    let n = (c * t * h * w) as usize;
    // Signed, and deliberately not tiny: a per-position sum of squares whose
    // terms all have the same magnitude would round the same way in any order,
    // which is precisely the property this file must NOT accidentally rely on.
    let mut r = Lcg::new(0x3D_00_11_22);
    r.vec_scaled(n, 3.0)
}

fn gamma() -> Vec<f32> {
    let mut r = Lcg::new(0xA1_1E_5C_7E);
    (0..SHAPE.0 as usize).map(|_| 0.5 + r.unit()).collect()
}

/// Build and run a graph that applies both norms in sequence, returning its
/// output. `device` is passed straight through to the shared handle helper.
fn run(gpu: &Gpu, x_host: &[f32], g_host: &[f32]) -> Vec<f32> {
    let (c, t, h, w) = SHAPE;
    let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    tensors.insert("norm.gamma".to_string(), (vec![c as usize], g_host.to_vec()));

    let mut b = Builder3d::new(gpu, &tensors, false);
    let x_in = gpu.storage(x_host.len() as u64);
    gpu.write_f32(&x_in, x_host);
    let x = T3 { buf: x_in, c, t, h, w };

    let a = b.pixel_norm(&x, 1e-8);
    let y = b.rms_norm("norm", &a);
    let out = y.buf.clone();
    let n = x_host.len();
    let (steps, _) = b.finish();
    gpu.submit(&[], &steps);
    gpu.read(&out, n)
}

fn both_arms_agree(device: Option<&str>, label: &str) {
    let x = input();
    let g = gamma();
    let gpu = match device {
        Some(d) => Gpu::open(Some(d), &KERNELS),
        None => gpu_core::testgpu::dev(&KERNELS),
    };

    let _arm = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let split = without_fusion(|| run(&gpu, &x, &g));
    let split_again = without_fusion(|| run(&gpu, &x, &g));
    assert_eq!(
        differing_bits(&split, &split_again),
        0,
        "{label}: two runs of the SAME arm must already agree bit for bit, or this file cannot tell a fusion bug from ordinary nondeterminism"
    );

    let fused = run(&gpu, &x, &g);
    assert!(fused.iter().all(|v| v.is_finite()), "{label}: the fused arm produced non-finite output");
    // Guard against the degenerate pass where every value is zero and any
    // kernel at all would agree.
    assert!(fused.iter().any(|v| v.abs() > 1e-3), "{label}: the fused arm produced no signal to compare");
    println!("{label}: {} of {} output words differ", differing_bits(&split, &fused), split.len());
    assert_eq!(
        differing_bits(&split, &fused),
        0,
        "{label}: the fused channel-axis norm changed the result. It permutes nothing and folds the same values in the same order, so any differing bit is an indexing or a fold-order bug"
    );
}

#[test]
fn the_fused_channel_norm_is_bit_identical_to_the_composed_form() {
    both_arms_agree(None, "default device");
}

#[test]
fn the_fused_channel_norm_is_bit_identical_on_the_cpu_backend() {
    both_arms_agree(Some("cpu"), "cpu backend");
}
