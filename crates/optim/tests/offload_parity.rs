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

        let (lr, wd, scale) = (1e-3, 0.01, 2.0);
        let adam = optim::Adam { beta1: 0.8, beta2: 0.95, eps: 1e-6 };
        opt.step(&gpu, &ps_g, t, lr, wd, adam, clip, scale);
        off.step(&gpu, &ps_o, t, lr, wd, adam, clip, scale);
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

/// `torch.optim.AdamW` after `clip_grad_norm_`, on a gradient averaged by
/// `scale` (a `1/K` accumulation factor): the clip sees the averaged
/// gradient, as it does in torch when the loss is divided by K.
fn torch_reference(w: &mut [f32], grads: &[Vec<f32>], lr: f32, wd: f32, adam: optim::Adam, clip: f32, scale: f32) {
    let (mut m, mut v) = (vec![0f32; w.len()], vec![0f32; w.len()]);
    for (t, g) in grads.iter().enumerate() {
        let t = t as i32 + 1;
        let avg: Vec<f32> = g.iter().map(|x| x * scale).collect();
        let norm = avg.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
        let coef = (clip / (norm + 1e-6)).min(1.0);
        for i in 0..w.len() {
            let gi = avg[i] * coef;
            m[i] = adam.beta1 * m[i] + (1.0 - adam.beta1) * gi;
            v[i] = adam.beta2 * v[i] + (1.0 - adam.beta2) * gi * gi;
            let (mh, vh) = (m[i] / (1.0 - adam.beta1.powi(t)), v[i] / (1.0 - adam.beta2.powi(t)));
            w[i] -= lr * wd * w[i];
            w[i] -= lr * mh / (vh.sqrt() + adam.eps);
        }
    }
}

/// `extra_scale` and the clip mean the same on the device and the host
/// optimiser, and what they mean is torch's: the gradient is averaged by
/// `extra_scale` and the clip bounds the averaged gradient's norm. A large ε
/// makes the gradient's scale visible through Adam's normalisation.
#[test]
fn scale_and_clip_follow_torch_on_device_and_host() {
    let n = 1024usize;
    let w0: Vec<f32> = (0..n).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
    let init: HashMap<String, Vec<f32>> = [("w".to_string(), w0.clone())].into_iter().collect();
    let gpu = Gpu::new_cpu(&kernels());
    let ps_g = ParamStore::new(&gpu, vec![("w".to_string(), n)], &init);
    let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);
    let ps_o = ParamStore::new_with_roles(&gpu, vec![("w".to_string(), n, Role::Offload)], &init);
    let mut off = OffloadAdam::new(&gpu, &ps_o);

    let (lr, wd, clip, scale) = (1e-2f32, 0.01f32, 0.5f32, 0.25f32);
    let adam = optim::Adam { beta1: 0.9, beta2: 0.99, eps: 1e-1 };
    let grads: Vec<Vec<f32>> = (1..=3u32).map(|t| (0..n).map(|i| (((i as u32 * 3 + t) % 17) as f32 / 17.0 - 0.5) * 0.4).collect()).collect();
    for (t, g) in grads.iter().enumerate() {
        gpu.write(ps_g.g("w"), bytemuck::cast_slice(g));
        gpu.write(ps_o.g("w"), bytemuck::cast_slice(g));
        opt.step(&gpu, &ps_g, t as u32 + 1, lr, wd, adam, Some(clip), scale);
        off.step(&gpu, &ps_o, t as u32 + 1, lr, wd, adam, Some(clip), scale);
    }
    let mut want = w0;
    torch_reference(&mut want, &grads, lr, wd, adam, clip, scale);
    for (what, got) in [("device", gpu.read(ps_g.w("w"), n)), ("host", gpu.read(ps_o.w("w"), n))] {
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{what} AdamW is off torch's scaled, clipped step by {worst}");
    }
}
