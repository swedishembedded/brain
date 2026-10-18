// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gates for MiniCPM-V's vision-side backward
//! (`minicpmv::vision_train`): the NaViT tower with its window merger, the
//! merger, and LoRA on the tower's linears.
//!
//! The loss is `L = sum(out * R)` over the visual rows for a fixed random
//! `R`, so `dL/dout = R` is exact and every parameter's analytic gradient
//! comes from one backward. Each check perturbs one tensor along a random
//! +-1 direction and re-runs the real forward. Two packed views of different
//! shapes exercise the ragged spans, and the 16x and 4x modes exercise the
//! window merger and its absence.

use data::rng::Rng;
use gpu_core::Gpu;
use minicpmv::vision::{Tower, View};
use minicpmv::vision_train::{train_pipelines, TowerLora, LORA_LEAVES};
use minicpmv::{Downsample, MiniCpmVConfig};

use crate::{Check, Report};

/// Random tower weights: LayerNorm gains near 1, everything else small.
fn tiny_weights(cfg: &MiniCpmVConfig, rng: &mut Rng) -> std::collections::HashMap<String, Vec<f32>> {
    minicpmv::vision::tensor_manifest(cfg)
        .into_iter()
        .map(|(name, n)| {
            let gain = name.contains("norm") && name.ends_with(".weight");
            let v = (0..n).map(|_| if gain { 1.0 + 0.2 * (rng.next_f32() - 0.5) } else { 0.6 * (rng.next_f32() - 0.5) }).collect();
            (name, v)
        })
        .collect()
}

/// Two views of different shapes, each reducible by 4 in both directions.
fn views_and_patches(cfg: &MiniCpmVConfig, rng: &mut Rng) -> (Vec<View>, Vec<f32>) {
    let views = vec![View { h: 4, w: 4 }, View { h: 4, w: 8 }];
    let n: u32 = views.iter().map(|v| v.patches()).sum();
    let pp = 3 * cfg.vision.patch * cfg.vision.patch;
    (views, (0..n * pp).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
}

fn loss(out: &[f32], r: &[f32]) -> f32 {
    out.iter().zip(r).map(|(a, b)| a * b).sum()
}

/// Directional central difference of `L` in tensor `name` (read and written
/// through `get`/`set`), against `analytic`, the best of `n_dirs` directions.
fn directional(name: &str, analytic: &[f32], w0: &[f32], rng: &mut Rng, eps: f32, mut set: impl FnMut(&[f32]), mut eval: impl FnMut() -> f32) -> Check {
    let mut best: Option<Check> = None;
    for _ in 0..3 {
        let v: Vec<f32> = (0..w0.len()).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
        let a: f32 = analytic.iter().zip(&v).map(|(g, d)| g * d).sum();
        set(&w0.iter().zip(&v).map(|(w, d)| w + eps * d).collect::<Vec<_>>());
        let lp = eval();
        set(&w0.iter().zip(&v).map(|(w, d)| w - eps * d).collect::<Vec<_>>());
        let lm = eval();
        set(w0);
        let numeric = (lp - lm) / (2.0 * eps);
        let abs_err = (a - numeric).abs();
        let rel_err = abs_err / a.abs().max(numeric.abs()).max(1e-3);
        let cand = Check { param: name.to_string(), analytic: a, numeric, abs_err, rel_err };
        if best.as_ref().is_none_or(|b| cand.rel_err < b.rel_err) {
            best = Some(cand);
        }
    }
    best.expect("one direction")
}

/// Every vision parameter group, the whole tower trainable, in `mode`.
fn check_tower(seed: u64, mode: Downsample) -> Report {
    let cfg = MiniCpmVConfig::tiny();
    let mut rng = Rng::new(seed);
    let w = tiny_weights(&cfg, &mut rng);
    let tower = Tower::new_train_on(Gpu::new(train_pipelines()), cfg.clone(), &w, &|_| true);
    let (views, patches) = views_and_patches(&cfg, &mut rng);
    let (out, pass) = tower.encode_train(&patches, &views, mode);
    let r: Vec<f32> = (0..out.len()).map(|_| rng.next_f32() - 0.5).collect();
    tower.zero_grads();
    tower.backward(&pass, &r);
    let mut names = vec![
        "patch.weight", "patch.bias", "pos.weight", "blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.bias", "blocks.2.attn.proj.weight",
        "blocks.2.norm2.weight", "post_norm.weight", "post_norm.bias", "merger.pre_norm.weight", "merger.fc1.weight", "merger.fc2.bias",
    ];
    if mode == Downsample::X16 {
        names.extend(["wm.norm1.weight", "wm.attn.qkv.weight", "wm.attn.proj.bias", "wm.pre_norm.bias", "wm.fc1.weight", "wm.fc2.weight"]);
    }
    let checks = names
        .into_iter()
        .map(|name| {
            let analytic = tower.read_grad(name);
            let w0 = tower.read_weight(name);
            directional(name, &analytic, &w0, &mut rng, 1e-2, |x| tower.write_weight(name, x), || loss(&tower.encode(&patches, &views, mode), &r))
        })
        .collect();
    Report { checks }
}

/// The tower with its window merger (16x), every parameter group.
pub fn check_minicpmv_vision(seed: u64) -> Report {
    check_tower(seed, Downsample::X16)
}

/// The tower without its window merger (4x), every parameter group.
pub fn check_minicpmv_vision_4x(seed: u64) -> Report {
    check_tower(seed, Downsample::X4)
}

/// LoRA on every tower linear: the factors' gradients, projected from the
/// dense weight gradients, against differences of the folded forward. `B`
/// starts non-zero so both factors have a gradient to check.
pub fn check_minicpmv_vision_lora(seed: u64) -> Report {
    let cfg = MiniCpmVConfig::tiny();
    let mut rng = Rng::new(seed);
    let w = tiny_weights(&cfg, &mut rng);
    let mut tower = Tower::new_train_on(Gpu::new(train_pipelines()), cfg.clone(), &w, &|n| n.starts_with("blocks."));
    tower.ps.freeze_where(|_| true);
    let mut init_rng = Rng::new(seed ^ 0x5eed);
    let lora = TowerLora::new(&tower, LORA_LEAVES, 3, 6.0, &mut |n| (0..n).map(|_| 0.4 * (init_rng.next_f32() - 0.5)).collect()).unwrap();
    for (name, _) in lora.ps.params.clone() {
        if name.ends_with(".lora_b") {
            let b: Vec<f32> = (0..lora.ps.numel(&name)).map(|_| 0.2 * (rng.next_f32() - 0.5)).collect();
            tower.gpu.write_f32(lora.ps.w(&name), &b);
        }
    }
    let (views, patches) = views_and_patches(&cfg, &mut rng);
    lora.fold(&tower);
    let (out, pass) = tower.encode_train(&patches, &views, Downsample::X16);
    let r: Vec<f32> = (0..out.len()).map(|_| rng.next_f32() - 0.5).collect();
    tower.zero_grads();
    lora.ps.zero_grads(&tower.gpu);
    tower.backward(&pass, &r);
    lora.project(&tower);
    let names = ["blocks.0.attn.qkv.weight.lora_a", "blocks.0.attn.qkv.weight.lora_b", "blocks.1.mlp.fc2.weight.lora_a", "blocks.2.attn.proj.weight.lora_b", "blocks.2.mlp.fc1.weight.lora_b"];
    let checks = names
        .into_iter()
        .map(|name| {
            let analytic = lora.ps.read_grad(&tower.gpu, name);
            let w0 = lora.ps.read_weight(&tower.gpu, name);
            directional(name, &analytic, &w0, &mut rng, 1e-2, |x| tower.gpu.write_f32(lora.ps.w(name), x), || {
                lora.fold(&tower);
                loss(&tower.encode(&patches, &views, Downsample::X16), &r)
            })
        })
        .collect();
    Report { checks }
}
