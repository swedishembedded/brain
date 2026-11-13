// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified perceptual-loss training for its clients. If your team needs expertise in
// training image generators against a perceptual metric, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The gradient of LPIPS with respect to its input images ([`lpips::Lpips::distance_grad`]), the trunk and heads frozen.
//!
//! The oracle is the f64 transcription of the reference in `reference.rs`, central differences taken on IT (an f64
//! forward has no rounding noise to divide by the step), against the f32 device gradient. Weights are random
//! ones of the trunk's released shapes ([`lpips::import::random`]), for AlexNet and for VGG16, so none of this needs
//! the model store.
//!
//! Tolerance. The device gradient is a chain of f32 accumulations through five conv layers, the channel
//! normalisation and the head; no entry is wrong by more than the round-off of that chain, and a defect (a dropped
//! term, a missing factor, a wrong sign) moves entries by a fraction of their own size. The bound is Higham's
//! (`brain_testutil::rounding`): `k` accumulated terms of absolute sum `S` are within `gamma_k S` of exact, two
//! evaluations within twice that. `k` is the number of terms on the longest chain the backward walks (every conv's
//! `cin k^2` contraction down to the input, the 1x1 head, the channel norm and the sum over taps), and `S` is the
//! largest gradient magnitude, which no entry's chain of terms can fall below by more than cancellation allows.

use brain_testutil::rounding::agreement_bound;
use gpu_core::{DeviceBuffer, Gpu};
use lpips::config::Trunk;
use lpips::import::Tensors;
use lpips::Lpips;

use crate::reference::{distance_of, reference, taps};

const TRUNKS: [Trunk; 2] = [Trunk::Alex, Trunk::Vgg16];

/// The test image side. AlexNet's pools leave a 3x3 last tap at 47. VGG16's reference forward is 13 convolutions in
/// f64 and is run dozens of times by the finite differences, so its side is small: 24 gives taps of 24, 12, 6, 3 and
/// 1 pixels wide, every pool and every head exercised.
fn side(trunk: Trunk) -> usize {
    match trunk {
        Trunk::Alex => 47,
        Trunk::Vgg16 => 24,
    }
}

fn metric(trunk: Trunk) -> Lpips {
    let g: Gpu = gpu_core::testgpu::dev(lpips::GRAD_PIPELINES);
    Lpips::with_trunk(g, trunk, &weights(trunk)).expect("lpips on the test device")
}

fn weights(trunk: Trunk) -> Tensors {
    use std::sync::OnceLock;
    static SEED: OnceLock<u64> = OnceLock::new();
    let seed = *SEED.get_or_init(|| {
        let s = data::rng::random_seed();
        eprintln!("lpips backward test weight seed: {s}");
        s
    });
    lpips::import::random(trunk, seed)
}

/// An interleaved `[0, 1]` RGB image of uniform noise.
fn noise(trunk: Trunk, seed: u64) -> Vec<f32> {
    data::rng::Lcg::new(seed).vec_unit(3 * side(trunk) * side(trunk))
}

/// The two interleaved images as the planar `[2, 3, h, w]` batch in `[-1, 1]` the gradient entry point takes.
fn pair(gpu: &Gpu, trunk: Trunk, a: &[f32], b: &[f32]) -> DeviceBuffer {
    let plane = side(trunk) * side(trunk);
    let mut v = vec![0.0f32; 2 * 3 * plane];
    for (n, img) in [a, b].iter().enumerate() {
        for p in 0..plane {
            for c in 0..3 {
                v[(n * 3 + c) * plane + p] = 2.0 * img[p * 3 + c] - 1.0;
            }
        }
    }
    gpu.storage_init("lpips.test.pair", &v)
}

/// Terms on the longest chain the backward walks.
fn chain_terms(trunk: Trunk) -> usize {
    let convs = trunk.convs();
    convs.iter().map(|c| (c.cin * c.k * c.k) as usize).sum::<usize>() + convs.iter().map(|c| c.cout as usize).sum::<usize>() + 5
}

fn f64_total(trunk: Trunk, t: &Tensors, a: &[f32], b: &[f32]) -> f64 {
    reference(trunk, t, a, b, side(trunk), side(trunk), None).iter().sum()
}

/// The device total agrees with the f64 reference, and the entry point that also differentiates returns the same
/// distance as the one that does not.
#[test]
fn the_gradient_entry_point_computes_the_same_distance() {
    for trunk in TRUNKS {
        let n = side(trunk) as u32;
        let (a, b) = (noise(trunk, 1), noise(trunk, 2));
        let mut m = metric(trunk);
        let plain = m.distance(&a, &b, n, n, None).expect("distance");
        let p = pair(m.gpu(), trunk, &a, &b);
        let (with_grad, _) = m.distance_grad(&p, n, n).expect("distance_grad");
        let want = f64_total(trunk, &weights(trunk), &a, &b);
        let tol = agreement_bound(chain_terms(trunk), want);
        assert!((plain.total - want).abs() <= tol, "{trunk:?}: distance {} vs f64 reference {want} (tolerance {tol:e})", plain.total);
        assert!((with_grad.total - want).abs() <= tol, "{trunk:?}: distance_grad {} vs f64 reference {want} (tolerance {tol:e})", with_grad.total);
    }
}

/// Central differences of the f64 reference against the device gradient, entry by entry, on both images.
#[test]
fn the_alexnet_input_gradient_matches_central_differences_of_the_f64_reference() {
    input_gradient_matches(Trunk::Alex);
}

#[test]
fn the_vgg16_input_gradient_matches_central_differences_of_the_f64_reference() {
    input_gradient_matches(Trunk::Vgg16);
}

fn input_gradient_matches(trunk: Trunk) {
    let n = side(trunk);
    let (a, b) = (noise(trunk, 3), noise(trunk, 4));
    let mut m = metric(trunk);
    let gpu = m.gpu().share();
    let p = pair(&gpu, trunk, &a, &b);
    let (_, grad) = m.distance_grad(&p, n as u32, n as u32).expect("distance_grad");
    let plane = n * n;
    let g = gpu.read(grad, 2 * 3 * plane);
    assert!(g.iter().all(|v| v.is_finite()), "{trunk:?}: a non-finite gradient entry");
    // The device works in [-1, 1], the oracle in [0, 1]: d/dv = 2 d/dx.
    let in_unit = |which: usize, idx: usize| {
        let (p, c) = (idx / 3, idx % 3);
        2.0 * g[(which * 3 + c) * plane + p] as f64
    };
    let scale = g.iter().fold(0.0f64, |m, v| m.max(2.0 * (*v as f64).abs()));
    let tol = agreement_bound(2 * chain_terms(trunk), scale);
    assert!(tol < 0.01 * scale, "{trunk:?}: the tolerance {tol:e} would hide a one percent error on the largest entry {scale:e}");
    assert!(scale > 1e-6, "{trunk:?}: an all-zero gradient proves nothing");

    let t = weights(trunk);
    // The image that is not varied has its taps computed once.
    let (taps_a, taps_b) = (taps(trunk, &t, &a, n, n), taps(trunk, &t, &b, n, n));
    // VGG16's f64 reference is thirteen convolutions per evaluation, so it checks fewer entries.
    let entries = if trunk == Trunk::Alex { 12 } else { 6 };
    let eps = 1e-4f32;
    let mut rng = data::rng::Rng::new(data::rng::random_seed());
    let mut checked = 0;
    let mut tried = 0;
    while checked < entries && tried < 60 {
        tried += 1;
        let which = (rng.next_u64() % 2) as usize;
        let idx = (rng.next_u64() % (3 * plane as u64)) as usize;
        let base = [&a, &b][which];
        let total = |img: &[f32]| {
            let varied = taps(trunk, &t, img, n, n);
            let (fa, fb) = if which == 0 { (&varied, &taps_b) } else { (&taps_a, &varied) };
            distance_of(&t, fa, fb, n, n, None).iter().sum::<f64>()
        };
        let (mut hi, mut lo) = (base.clone(), base.clone());
        hi[idx] += eps;
        lo[idx] -= eps;
        // The step actually taken in f32, not the one asked for.
        let (dh, dl) = ((hi[idx] - base[idx]) as f64, (base[idx] - lo[idx]) as f64);
        let (l0, lh, ll) = (total(base), total(&hi), total(&lo));
        let (ahead, behind) = ((lh - l0) / dh, (l0 - ll) / dl);
        // A ReLU or max-pool kink inside the step: the loss is not differentiable there, draw another entry.
        if (ahead - behind).abs() > 1e-3 * ahead.abs().max(behind.abs()).max(1e-3) {
            continue;
        }
        let fd = (lh - ll) / (dh + dl);
        let analytic = in_unit(which, idx);
        assert!(
            (analytic - fd).abs() <= tol,
            "{trunk:?} image {which} entry {idx}: analytic {analytic:e} vs central difference {fd:e} (tolerance {tol:e}, largest entry {scale:e})"
        );
        checked += 1;
    }
    assert_eq!(checked, entries, "{trunk:?}: only {checked} smooth entries found in {tried} draws");
}

/// An image against itself is the metric's minimum: its gradient is exactly zero, and in particular finite where a
/// feature position is zero in every channel (the channel norm's reciprocal is `1e10` there).
#[test]
fn identical_images_have_a_zero_gradient() {
    for trunk in TRUNKS {
        let n = side(trunk);
        let a = noise(trunk, 5);
        let mut m = metric(trunk);
        let gpu = m.gpu().share();
        let p = pair(&gpu, trunk, &a, &a);
        let (d, grad) = m.distance_grad(&p, n as u32, n as u32).expect("distance_grad");
        assert_eq!(d.total, 0.0);
        assert!(gpu.read(grad, 6 * n * n).iter().all(|v| *v == 0.0), "{trunk:?}: a nonzero gradient at the minimum");
    }
}

/// A device built without the backward kernels is refused by name, not dispatched into a missing kernel.
#[test]
fn a_device_without_the_backward_kernels_is_refused() {
    let gpu = gpu_core::testgpu::dev(lpips::PIPELINES);
    let mut m = Lpips::new(gpu, &weights(Trunk::Alex)).expect("the forward needs only PIPELINES");
    let n = side(Trunk::Alex);
    let p = pair(m.gpu(), Trunk::Alex, &noise(Trunk::Alex, 6), &noise(Trunk::Alex, 7));
    let err = m.distance_grad(&p, n as u32, n as u32).map(|_| ()).unwrap_err();
    assert!(err.contains("GRAD_PIPELINES"), "{err}");
}

/// An input below the trunk's smallest side is refused by name, and weights of the other trunk are refused at
/// construction rather than convolved.
#[test]
fn a_too_small_image_and_the_wrong_trunk_are_refused() {
    let mut m = metric(Trunk::Vgg16);
    let below = Trunk::Vgg16.min_side() as usize - 1;
    let img = vec![0.5f32; 3 * below * below];
    let err = m.distance(&img, &img, below as u32, below as u32, None).unwrap_err();
    assert!(err.contains("vgg"), "{err}");
    let gpu = gpu_core::testgpu::dev(lpips::PIPELINES);
    let err = Lpips::with_trunk(gpu, Trunk::Vgg16, &weights(Trunk::Alex)).map(|_| ()).unwrap_err();
    assert!(err.contains("features.0.weight") && err.contains("[64, 3, 3, 3]"), "{err}");
}
