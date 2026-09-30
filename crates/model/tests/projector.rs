// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device MLP projector computes the host reference and its exact
//! gradient: forward equal to `forward_host` for the plain stack at two
//! depths and for the hybrid split, and every parameter's and every input
//! stream's gradient equal to central finite differences of the host
//! function.

use std::collections::HashMap;

use data::rng::Rng;
use model::projector::{forward_host, MlpProjector, ProjectorConfig, PROJECTOR_PIPELINES};

const ROWS: usize = 3;

fn configs() -> [ProjectorConfig; 3] {
    [
        ProjectorConfig::from_type("mlp_gelu", 2, 6, 8).unwrap(),
        ProjectorConfig::from_type("mlp_gelu", 3, 6, 8).unwrap(),
        ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 2, 6, 8).unwrap(),
    ]
}

fn random(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

#[test]
fn the_device_projector_matches_the_host_function_and_its_gradient() {
    let g = gpu_core::testgpu::dev(PROJECTOR_PIPELINES);
    for cfg in configs() {
        let mut rng = Rng::new(11);
        let weights: HashMap<String, Vec<f32>> = cfg.param_list().into_iter().map(|(n, len)| (n, random(len, &mut rng))).collect();
        let inputs: Vec<Vec<f32>> = (0..cfg.inputs()).map(|_| random(ROWS * cfg.input_dim as usize, &mut rng)).collect();
        // The loss is the output against a fixed random direction.
        let dir = random(ROWS * cfg.n_embed as usize, &mut rng);
        let loss = |w: &HashMap<String, Vec<f32>>, x: &[Vec<f32>]| -> f64 {
            let refs: Vec<&[f32]> = x.iter().map(Vec::as_slice).collect();
            forward_host(&cfg, w, &refs, ROWS).iter().zip(&dir).map(|(a, b)| *a as f64 * *b as f64).sum()
        };

        let proj = MlpProjector::new(&g, cfg, ROWS as u32, &weights).unwrap();
        let x_dev: Vec<_> = inputs.iter().map(|x| g.storage_init("x", x)).collect();
        let x_refs: Vec<&_> = x_dev.iter().collect();
        g.submit(&[], &proj.forward(&g, &x_refs));
        let got = g.read(proj.out(), ROWS * cfg.n_embed as usize);
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
}
