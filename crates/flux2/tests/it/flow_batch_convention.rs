// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rectified-flow convention a FLUX.2 training batch is built in, checked in
//! FP64 against the formula and against the Euler step the sampler takes.
//!
//! `pipeline` starts from pure noise at σ = 1, evaluates the DiT with model time
//! `t = σ` and steps `x += (σ' − σ)·v` down to σ = 0. A batch is only training
//! the model the sampler runs if its noised input is the point that integration
//! passes through at σ, its target is the velocity the step multiplies, and its
//! time input is σ itself (not 1 − σ, not 1000·σ). Each of those is asserted
//! here for [`flux2::modelgrad::make_flow_batch`] and its paired form.
//!
//! Swedish Embedded AB implements training pipelines that optimise the model
//! deployment runs for its clients. If your team needs expertise in diffusion
//! and flow-matching objectives, you can procure our services by sending an
//! email to info@swedishembedded.com.

use data::rng::Lcg;
use flux2::modelgrad::{make_flow_batch, make_flow_batch_paired, Batch, Cfg};

fn draw(rng: &mut Lcg, n: usize) -> Vec<f64> {
    rng.vec(n).into_iter().map(f64::from).collect()
}

struct Example {
    x0: Vec<f64>,
    refs: Vec<f64>,
    ctx: Vec<f64>,
    noise: Vec<f64>,
}

fn example(cfg: &Cfg, seed: u64) -> Example {
    let mut rng = Lcg::new(seed);
    let cin = cfg.in_channels;
    Example {
        x0: draw(&mut rng, cfg.n_gen() * cin),
        refs: draw(&mut rng, cfg.n_ref() * cin),
        ctx: draw(&mut rng, cfg.txt_len * cfg.context_in_dim),
        noise: draw(&mut rng, cfg.n_gen() * cin),
    }
}

fn batch(cfg: &Cfg, e: &Example, sigma: f64) -> Batch<f64> {
    if cfg.refs.is_empty() {
        make_flow_batch(cfg, &e.x0, &e.ctx, sigma, &e.noise)
    } else {
        make_flow_batch_paired(cfg, &e.x0, &e.refs, &e.ctx, sigma, &e.noise)
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// FP64 rounding of a handful of operations on values of magnitude below 2.
const TOL: f64 = 1e-14;

/// **The noised input, the velocity target and the model time are the
/// formula's, caption-only and paired.** `x_σ = (1 − σ)·x₀ + σ·ε` over the
/// generated rows, every reference token appended clean after them, target
/// `v = ε − x₀`, `t = σ`, the caption unchanged and the RoPE tables the joint
/// layout's (the ones the sampler builds), with no loss weights.
#[test]
fn a_flow_batch_is_the_rectified_flow_formula_in_fp64() {
    for cfg in [Cfg::tiny(), Cfg::tiny_paired()] {
        let cin = cfg.in_channels;
        let e = example(&cfg, 0x0f10_7c0e + cfg.n_ref() as u64);
        let tables = cfg.layout().rope(cfg.axes_dim, cfg.rope_theta);
        for sigma in [0.05, 0.45, 0.7175, 1.0] {
            let b = batch(&cfg, &e, sigma);
            let n_gen = cfg.n_gen() * cin;
            assert_eq!(b.img.len(), cfg.n_img() * cin, "the image half holds generated then reference rows");
            let want_noised: Vec<f64> = (0..n_gen).map(|i| (1.0 - sigma) * e.x0[i] + sigma * e.noise[i]).collect();
            assert!(max_abs_diff(&b.img[..n_gen], &want_noised) <= TOL, "x_sigma at sigma {sigma}");
            assert_eq!(&b.img[n_gen..], &e.refs[..], "reference tokens enter clean at every sigma");
            let want_velocity: Vec<f64> = (0..n_gen).map(|i| e.noise[i] - e.x0[i]).collect();
            assert!(max_abs_diff(&b.target, &want_velocity) <= TOL, "velocity target at sigma {sigma}");
            assert_eq!(b.t, sigma, "the model time input is sigma itself");
            assert_eq!(b.ctx, e.ctx, "the caption features pass through unchanged");
            assert!(b.w.is_empty(), "a plain flow batch carries no loss weights");
            let cast = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
            assert_eq!(b.cos, cast(&tables.cos), "RoPE cos tables are the joint layout's");
            assert_eq!(b.sin, cast(&tables.sin), "RoPE sin tables are the joint layout's");
        }
    }
}

/// **The target is the velocity the sampler's Euler step integrates.** The
/// interpolant is linear in σ, so the exact finite difference of the noised
/// input between any two noise levels is the target, and one step
/// `x += (σ' − σ)·v` from the batch at σ lands on the batch at σ'. At σ = 1 the
/// input is pure noise (where sampling starts) and a step to σ = 0 recovers
/// `x₀`. A target of the opposite sign fails each of these.
#[test]
fn the_velocity_target_is_the_step_the_euler_sampler_takes() {
    let cfg = Cfg::tiny();
    let e = example(&cfg, 0x000e_01e5);
    let n_gen = cfg.n_gen() * cfg.in_channels;
    let at = |sigma: f64| batch(&cfg, &e, sigma);

    let start = at(1.0);
    assert!(max_abs_diff(&start.img, &e.noise) <= TOL, "sampling starts from pure noise at sigma 1");

    for (from, to) in [(1.0, 0.9581), (0.9581, 0.7175), (0.7175, 0.25), (0.25, 0.0)] {
        let b = at(from);
        let stepped: Vec<f64> = (0..n_gen).map(|i| b.img[i] + (to - from) * b.target[i]).collect();
        assert!(max_abs_diff(&stepped, &at(to).img) <= 4.0 * TOL, "Euler step {from} -> {to} with the target velocity");
        let slope: Vec<f64> = (0..n_gen).map(|i| (at(to).img[i] - b.img[i]) / (to - from)).collect();
        assert!(max_abs_diff(&slope, &b.target) <= 1e-12, "d x_sigma / d sigma is the target between {from} and {to}");
    }

    let clean: Vec<f64> = (0..n_gen).map(|i| start.img[i] - start.target[i]).collect();
    assert!(max_abs_diff(&clean, &e.x0) <= TOL, "one step from sigma 1 to 0 with the exact velocity recovers x0");
}
