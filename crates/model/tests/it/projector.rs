// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device MLP projector computes the host reference and its exact
//! gradient: forward equal to `forward_host` for the plain stack at two
//! depths and for the hybrid split, and every parameter's and every input
//! stream's gradient equal to central finite differences of the host
//! function.

use std::collections::HashMap;

use data::rng::{random_seed, Rng};
use model::projector::{forward_host, MlpProjector, ProjectorConfig, PROJECTOR_PIPELINES};

const ROWS: usize = 3;

fn configs() -> [ProjectorConfig; 6] {
    [
        ProjectorConfig::from_type("mlp_gelu", 2, 6, 8).unwrap(),
        // Janus-Pro's generation head: the last linear widens to the image
        // vocabulary.
        ProjectorConfig::from_type("mlp_gelu", 2, 6, 8).unwrap().with_out_dim(11).unwrap(),
        ProjectorConfig::from_type("mlp_gelu", 3, 6, 8).unwrap(),
        ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 2, 6, 8).unwrap(),
        // A checkpoint without biases (Inkling 2's merger): no bias parameter exists, none is fed.
        ProjectorConfig::from_type("mlp2x_gelu", 2, 6, 8).unwrap().without_bias(),
        ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 2, 6, 8).unwrap().without_bias(),
    ]
}

fn random(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

#[test]
fn the_device_projector_matches_the_host_function_and_its_gradient() {
    let g = gpu_core::testgpu::dev(PROJECTOR_PIPELINES);
    for cfg in configs() {
        let mut rng = Rng::new(random_seed());
        let weights: HashMap<String, Vec<f32>> = cfg.param_list().into_iter().map(|(n, len)| (n, random(len, &mut rng))).collect();
        let inputs: Vec<Vec<f32>> = (0..cfg.inputs()).map(|_| random(ROWS * cfg.input_dim as usize, &mut rng)).collect();
        // The loss is the output against a fixed random direction.
        let dir = random(ROWS * cfg.out_dim as usize, &mut rng);
        let loss = |w: &HashMap<String, Vec<f32>>, x: &[Vec<f32>]| -> f64 {
            let refs: Vec<&[f32]> = x.iter().map(Vec::as_slice).collect();
            forward_host(&cfg, w, &refs, ROWS).iter().zip(&dir).map(|(a, b)| *a as f64 * *b as f64).sum()
        };

        let proj = MlpProjector::new(&g, cfg, ROWS as u32, &weights).unwrap();
        let x_dev: Vec<_> = inputs.iter().map(|x| g.storage_init("x", x)).collect();
        let x_refs: Vec<&_> = x_dev.iter().collect();
        g.submit(&[], &proj.forward(&g, &x_refs));
        let got = g.read(proj.out(), ROWS * cfg.out_dim as usize);
        let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
        let want = forward_host(&cfg, &weights, &refs, ROWS);
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "{cfg:?}: forward differs by {worst}");

        proj.zero_grads(&g);
        let d_out = g.storage_init("d_out", &dir);
        let d_in: Vec<_> = inputs.iter().map(|x| g.storage(x.len() as u64)).collect();
        let d_in_refs: Vec<&_> = d_in.iter().collect();
        g.submit(&[], &proj.backward(&g, &x_refs, &d_out, Some(&d_in_refs)));

        let eps = 1e-3f32;
        let check = |label: &str, analytic: &[f32], numeric: &dyn Fn(usize) -> f64| {
            for (j, &a) in analytic.iter().enumerate() {
                let n = numeric(j);
                let err = (a as f64 - n).abs();
                assert!(err <= 2e-3 + 2e-2 * n.abs(), "{cfg:?} {label}[{j}]: analytic {a} numeric {n}");
            }
        };
        for (name, len) in cfg.param_list() {
            let analytic = g.read(proj.grad(&name), len);
            check(&name, &analytic, &|j| {
                let (mut p, mut m) = (weights.clone(), weights.clone());
                p.get_mut(&name).unwrap()[j] += eps;
                m.get_mut(&name).unwrap()[j] -= eps;
                (loss(&p, &inputs) - loss(&m, &inputs)) / (2.0 * eps as f64)
            });
        }
        for (s, d) in d_in.iter().enumerate() {
            let analytic = g.read(d, inputs[s].len());
            check(&format!("input {s}"), &analytic, &|j| {
                let (mut p, mut m) = (inputs.clone(), inputs.clone());
                p[s][j] += eps;
                m[s][j] -= eps;
                (loss(&weights, &p) - loss(&weights, &m)) / (2.0 * eps as f64)
            });
        }
    }
}

#[test]
fn an_unknown_projector_or_an_odd_hybrid_width_is_refused() {
    assert!(ProjectorConfig::from_type("linear", 1, 4, 4).is_err());
    assert!(ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 2, 4, 7).is_err());
    assert!(ProjectorConfig::from_type("mlp_gelu", 0, 4, 4).is_err());
    let hybrid = ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 1, 4, 8).unwrap();
    assert!(hybrid.with_out_dim(6).is_err(), "a one-linear hybrid's output is its two halves");
}

/// A frozen projector (inference: no gradient buffers) computes the same
/// forward, and refuses a backward it has nowhere to accumulate into.
#[test]
fn a_frozen_projector_forwards_the_same_and_refuses_backward() {
    let g = gpu_core::testgpu::dev(PROJECTOR_PIPELINES);
    let cfg = ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 2, 6, 8).unwrap();
    let mut rng = Rng::new(random_seed());
    let weights: HashMap<String, Vec<f32>> = cfg.param_list().into_iter().map(|(n, len)| (n, random(len, &mut rng))).collect();
    let x: Vec<_> = (0..2).map(|_| g.storage_init("x", &random(ROWS * 6, &mut rng))).collect();
    let refs: Vec<&_> = x.iter().collect();
    let run = |p: &MlpProjector| {
        g.submit(&[], &p.forward(&g, &refs));
        g.read(p.out(), ROWS * 8)
    };
    let (live, frozen) = (MlpProjector::new(&g, cfg, ROWS as u32, &weights).unwrap(), MlpProjector::new_frozen(&g, cfg, ROWS as u32, &weights).unwrap());
    assert_eq!(run(&frozen), run(&live));
    let d_out = g.storage(ROWS as u64 * 8);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| frozen.backward(&g, &refs, &d_out, None)));
    assert!(refused.is_err(), "a frozen projector has no gradients to accumulate");
}

/// A bias-free stack lists weights only, and a projector sized for more rows than a call uses gives that call's rows
/// the result (and gradients) of an exactly sized one: the resident projector of a serving tower.
#[test]
fn a_bias_free_projector_has_no_bias_and_a_capacity_projector_serves_fewer_rows() {
    let cfg = ProjectorConfig::from_type("mlp2x_gelu", 2, 6, 8).unwrap().without_bias();
    let names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["in.weight", "layers.1.weight"]);

    let g = gpu_core::testgpu::dev(PROJECTOR_PIPELINES);
    let mut rng = Rng::new(random_seed());
    let weights: HashMap<String, Vec<f32>> = cfg.param_list().into_iter().map(|(n, len)| (n, random(len, &mut rng))).collect();
    let (exact, roomy) = (MlpProjector::new(&g, cfg, ROWS as u32, &weights).unwrap(), MlpProjector::new(&g, cfg, 3 * ROWS as u32, &weights).unwrap());
    assert_eq!(roomy.capacity(), 3 * ROWS as u32);
    let x = random(ROWS * 6, &mut rng);
    let mut padded = x.clone();
    padded.extend(random(2 * ROWS * 6, &mut rng));
    let (x_exact, x_roomy) = (g.storage_init("x", &x), g.storage_init("x", &padded));
    g.submit(&[], &exact.forward(&g, &[&x_exact]));
    g.submit(&[], &roomy.forward_rows(&g, &[&x_roomy], ROWS as u32));
    let n = ROWS * 8;
    assert_eq!(g.read(roomy.out(), n), g.read(exact.out(), n), "the first rows of a roomy projector are the exact one's");

    let dir = random(n, &mut rng);
    let (d_exact, d_roomy) = (g.storage_init("d", &dir), g.storage_init("d", &dir));
    let (di_exact, di_roomy) = (g.storage(x.len() as u64), g.storage(padded.len() as u64));
    g.submit(&[], &exact.backward(&g, &[&x_exact], &d_exact, Some(&[&di_exact])));
    g.submit(&[], &roomy.backward_rows(&g, &[&x_roomy], &d_roomy, Some(&[&di_roomy]), ROWS as u32));
    for (name, len) in cfg.param_list() {
        assert_eq!(g.read(roomy.grad(&name), len), g.read(exact.grad(&name), len), "{name} gradient");
    }
    assert_eq!(g.read(&di_roomy, x.len()), g.read(&di_exact, x.len()), "input gradient");

    // `resized` shares the parameters: an update through the original is seen by the copy, nothing is re-uploaded.
    let shared = exact.resized(&g, 2 * ROWS as u32);
    g.write_f32(exact.param("in.weight"), &vec![0.0; weights["in.weight"].len()]);
    g.submit(&[], &shared.forward_rows(&g, &[&x_roomy], ROWS as u32));
    assert!(g.read(shared.out(), n).iter().all(|v| *v == 0.0), "a zero first layer gives a zero output");
}

/// A projector whose weights have no gradient buffers still passes the gradient to its input (a frozen merger in
/// front of a trained decoder) and accumulates nothing.
#[test]
fn a_projector_over_frozen_weights_differentiates_through() {
    let g = gpu_core::testgpu::dev(PROJECTOR_PIPELINES);
    let cfg = ProjectorConfig::from_type("mlp2x_gelu", 2, 6, 8).unwrap().without_bias();
    let mut rng = Rng::new(random_seed());
    let weights: HashMap<String, Vec<f32>> = cfg.param_list().into_iter().map(|(n, len)| (n, random(len, &mut rng))).collect();
    let live = MlpProjector::new(&g, cfg, ROWS as u32, &weights).unwrap();
    let params = cfg.param_list().into_iter().map(|(n, _)| (n.clone(), live.param(&n).clone())).collect();
    let through = MlpProjector::with_buffers(&g, cfg, ROWS as u32, params, HashMap::new(), true).unwrap();
    let x = g.storage_init("x", &random(ROWS * 6, &mut rng));
    let d_out = g.storage_init("d", &random(ROWS * 8, &mut rng));
    let (a, b) = (g.storage(ROWS as u64 * 6), g.storage(ROWS as u64 * 6));
    g.submit(&[], &live.forward(&g, &[&x]));
    g.submit(&[], &through.forward(&g, &[&x]));
    g.submit(&[], &live.backward(&g, &[&x], &d_out, Some(&[&a])));
    g.submit(&[], &through.backward(&g, &[&x], &d_out, Some(&[&b])));
    assert_eq!(g.read(&a, ROWS * 6), g.read(&b, ROWS * 6));
}
