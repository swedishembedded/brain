// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image DiT profiler: ms per full forward + effective GFLOP/s, on CPU or a
//! single GPU. Reports vs the Tesla P40 fp32 peak (11.76 TFLOP/s/card).
//!
//! Usage:
//!   BRAIN_ZIMAGE_DIT=<z_image_turbo_bf16.safetensors> \
//!     zimage_bench <cpu|gpu> [h w cap_len reps]
//!
//! (Full 6B fp32 exceeds a single 24 GB P40, so `gpu` here is for sizes/configs
//! that fit one card; the 2-GPU sharded path is `zimage-shard`.)

use std::time::Instant;

use zimage::{import::import_comfy, ZImageConfig, ZImageDit, ZImageDitShard};

const P40_FP32_TFLOPS: f64 = 11.76;

/// Analytical FLOPs of one forward (the matmul-dominated ops), in GFLOP.
fn forward_gflop(cfg: &ZImageConfig, n_img: u64, ncap: u64) -> f64 {
    let d = cfg.dim as u64;
    let hd = (cfg.dim * 8 / 3) as u64;
    // MAC per block at T tokens: 4·T·D² (qkv+out) + 2·T²·D (attn) + 3·T·D·Hd (mlp).
    let block_mac = |t: u64| 4 * t * d * d + 2 * t * t * d + 3 * t * d * hd;
    let nref = cfg.n_refiner_layers as u64;
    let ntot = n_img + ncap;
    let mac = nref * block_mac(n_img)      // noise_refiner (image)
        + nref * block_mac(ncap)           // context_refiner (caption)
        + cfg.n_layers as u64 * block_mac(ntot); // main layers
    2.0 * mac as f64 / 1e9 // FLOP = 2·MAC
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device = args.get(1).map(|s| s.as_str()).unwrap_or("cpu");

    // `probe`: allocate ~Ngb GB on BRAIN_GPU_INDEX, hold it, so nvidia-smi shows
    // which physical card the index selected + how much fits before OOM.
    if device == "probe" {
        use gpu_core::Gpu;
        let gb: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let idx = std::env::var("BRAIN_GPU_INDEX").unwrap_or_default();
        eprintln!("probe: BRAIN_GPU_INDEX={idx}, allocating {gb} × 1 GB…");
        let gpu = Gpu::new_wgpu(&[("add2", kernels::ADD2)]);
        let mut bufs = Vec::new();
        for i in 0..gb {
            bufs.push(gpu.storage(256 * 1024 * 1024)); // 1 GB = 256M f32
            gpu.poll_wait();
            eprintln!("  allocated {} GB", i + 1);
        }
        eprintln!("probe: {gb} GB resident — check `nvidia-smi` now (holding 20s)");
        std::thread::sleep(std::time::Duration::from_secs(20));
        return;
    }
    let h: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let w: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let cap_len: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32);
    let reps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);

    let dit_path = match std::env::var("BRAIN_ZIMAGE_DIT") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("set BRAIN_ZIMAGE_DIT to the z_image_turbo_bf16 safetensors");
            std::process::exit(1);
        }
    };

    let mut cfg = ZImageConfig::turbo();
    // Optional: run a layer-reduced model (fits one 24 GB card for single-GPU
    // profiling; extrapolate ms to the full 30 layers). Uses real layer weights.
    if let Ok(n) = std::env::var("BRAIN_ZIMAGE_LAYERS") {
        if let Ok(n) = n.parse::<u32>() {
            cfg.n_layers = n;
        }
    }
    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let (f, ft, ht, wt) = (1u32, 1u32, h / ps, w / ps);
    let n_img = (ft * ht * wt) as u64;
    let gflop = forward_gflop(&cfg, n_img, cap_len as u64);

    eprintln!("loading weights…");
    let t0 = Instant::now();
    let tensors = checkpoint::safetensors::read(&dit_path).expect("read DiT weights");
    let weights = import_comfy(tensors, &cfg);
    eprintln!("  loaded + imported in {:.1}s", t0.elapsed().as_secs_f64());

    let latent = vec![0.1f32; (cfg.in_channels * f * h * w) as usize];
    let cap = vec![0.02f32; (cap_len * cfg.cap_feat_dim) as usize];

    // Warmup + timed reps (min-of-N wall clock). `shard` spans 2 GPUs.
    let time = |fwd: &dyn Fn() -> Vec<f32>| -> f64 {
        let _ = fwd();
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = fwd();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        best
    };

    eprintln!("building resident graph on {device}…");
    let t0 = Instant::now();
    let (best, cards) = if device == "shard" {
        let dit = ZImageDitShard::build(cfg.clone(), weights, f, h, w, cap_len);
        eprintln!("  built in {:.1}s", t0.elapsed().as_secs_f64());
        (time(&|| dit.forward(&latent, &cap, 0.5)), 2.0)
    } else {
        let dit = ZImageDit::build(cfg.clone(), weights, f, h, w, cap_len, Some(device));
        eprintln!("  built in {:.1}s", t0.elapsed().as_secs_f64());
        (time(&|| dit.forward(&latent, &cap, 0.5)), 1.0)
    };

    let gflops = gflop / best;
    let ntot = n_img + cap_len as u64;
    let peak = P40_FP32_TFLOPS * 1e3 * cards;
    println!("\n=== Z-Image DiT forward — {device} ({cards} card(s)) ===");
    println!("size: latent {h}x{w} -> {n_img} image + {cap_len} caption = {ntot} tokens");
    println!("work: {gflop:.1} GFLOP/forward");
    println!("time: {:.1} ms/forward (best of {reps})", best * 1e3);
    println!("rate: {gflops:.0} GFLOP/s  ({:.1}% of {cards}×P40 fp32 peak {peak:.0} GFLOP/s)", 100.0 * gflops / peak);
}
