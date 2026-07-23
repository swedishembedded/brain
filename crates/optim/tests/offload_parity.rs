// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The offloaded (host/CPU) AdamW must match the on-GPU AdamW element-wise, so
//! offloading optimiser state to RAM changes only *where* the moments live, not
//! the training trajectory.

use std::collections::HashMap;

use gpu_core::Gpu;
use optim::{OffloadAdam, Optim};
use paramstore::{ParamStore, Role};

const ADAMW: usize = 0;
const GRADNORM_SQ: usize = 1;
const GRAD_SCALE: usize = 2;
const CLIP_COEF: usize = 3;
const GRAD_SCALE_BUF: usize = 4;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    ]
}

fn run(clip: Option<f32>) {
    let n = 4096usize;
    let w0: Vec<f32> = (0..n).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
    let init: HashMap<String, Vec<f32>> = [("w".to_string(), w0.clone())].into_iter().collect();

    // GPU-optimised reference.
    let gpu = Gpu::new_cpu(&kernels()); // CPU backend runs the same adamw kernel via JIT
    let ps_g = ParamStore::new(&gpu, vec![("w".to_string(), n)], &init);
    let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

    // Offloaded (host) optimiser on an identical store.
    let ps_o =
        ParamStore::new_with_roles(&gpu, vec![("w".to_string(), n, Role::Offload)], &init);
    let mut off = OffloadAdam::new(&gpu, &ps_o);

    for t in 1..=6u32 {
        // Same synthetic grad into both stores' grad buffers.
        let g: Vec<f32> = (0..n).map(|i| (((i as u32 + t) % 13) as f32 / 13.0 - 0.5) * 0.1).collect();
        gpu.write(ps_g.g("w"), bytemuck::cast_slice(&g));
        gpu.write(ps_o.g("w"), bytemuck::cast_slice(&g));

        let (lr, wd, b1, b2, eps, scale) = (1e-3, 0.01, 0.9, 0.999, 1e-8, 2.0);
        opt.step(&gpu, &ps_g, t, lr, wd, b1, b2, eps, clip, scale);
        off.step(&gpu, &ps_o, t, lr, wd, b1, b2, eps, clip, scale);
    }

    let wg = gpu.read(ps_g.w("w"), n);
    let wo = gpu.read(ps_o.w("w"), n);
    let maxd = wg.iter().zip(&wo).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let scale = wg.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    let rel = maxd / scale;
    eprintln!("clip={clip:?}: offload vs gpu adamw  max-abs {maxd:.2e}  rel {rel:.2e}");
    assert!(rel < 1e-4, "offload adamw diverges from gpu adamw (rel {rel:.2e})");
}

#[test]
fn offload_adamw_matches_gpu_noclip() {
    run(None);
}

#[test]
fn offload_adamw_matches_gpu_clip() {
    run(Some(1.0));
}
