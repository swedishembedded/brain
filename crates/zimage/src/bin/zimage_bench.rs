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

use zimage::{import::import_comfy, ZImageConfig, ZImageDit, ZImageDitI8, ZImageDitShard};

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

    // `probe`: allocate ~Ngb GB on the ambient card (registry-resolved:
    // `--device gpu<i>` / BRAIN_GPU_INDEX), hold it, so nvidia-smi shows which
    // physical card the canonical index selected + how much fits before OOM.
    if device == "probe" {
        use gpu_core::Gpu;
        let gb: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let idx = gpu_core::devices::current_gpu().map(|i| i.to_string()).unwrap_or_else(|| "0 (default)".into());
        eprintln!("probe: gpu{idx}, allocating {gb} × 1 GB…");
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

    // `probe2`: create TWO devices via new_wgpu_multi, put 3 GB on dev0 and 6 GB
    // on dev1, hold — nvidia-smi then reveals which physical card each maps to.
    if device == "probe2" {
        use gpu_core::Gpu;
        let gpus = Gpu::new_wgpu_multi(&[("add2", kernels::ADD2)], 2);
        let mut hold = Vec::new();
        for _ in 0..3 { hold.push(gpus[0].storage(256 * 1024 * 1024)); gpus[0].poll_wait(); }
        for _ in 0..6 { hold.push(gpus[1].storage(256 * 1024 * 1024)); gpus[1].poll_wait(); }
        eprintln!("probe2: dev0=3 GB, dev1=6 GB — nvidia-smi should show one card ~3 GB, other ~6 GB (holding 20s)");
        std::thread::sleep(std::time::Duration::from_secs(20));
        return;
    }
    // `train`: measure a real training step's dominant cost — the forward and
    // backward GEMM sweep across the whole 34-block DiT — with the actual
    // register-tiled kernels (matmul_reg2 fwd, matmul_dx_reg + matmul_dw_reg
    // bwd). No 6B load needed: GEMM time depends only on shapes, so we drive
    // correctly-shaped scratch. Backward = dx (dY@W) + dW (dY^T@X) per linear =
    // 2× the forward FLOP; this measures whether the bwd kernels hold the same
    // ~34%-of-peak regime as the forward. Args: train [h w cap_len reps].
    if device == "train" {
        use gpu_core::Gpu;
        let h: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
        let w: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
        let cap_len: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
        let reps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
        let mut cfg = ZImageConfig::turbo();
        if let Ok(n) = std::env::var("BRAIN_ZIMAGE_LAYERS").and_then(|n| n.parse::<u32>().map_err(|_| std::env::VarError::NotPresent)) {
            cfg.n_layers = n;
        }
        let (dim, hidden) = (cfg.dim as usize, (cfg.dim * 8 / 3) as usize);
        let ps = cfg.patch_size;
        let n_img = ((h / ps) * (w / ps)) as usize;
        let ncap = cap_len as usize;
        let ntot = n_img + ncap;

        const K_FWD: usize = 0;
        const K_DX: usize = 1;
        const K_DW: usize = 2;
        let kk = [("matmul_reg2", kernels::MATMUL_REG2), ("matmul_dx_reg", kernels::MATMUL_DX_REG), ("matmul_dw_reg", kernels::MATMUL_DW_REG)];
        let gpu = Gpu::new_wgpu(&kk);
        // Shared scratch, sized to the largest shape (reused across all linears —
        // we time compute, not numerics). max activation ntot×hidden, max weight
        // hidden×dim.
        let big_act = (ntot * hidden) as u64;
        let big_w = (hidden * dim) as u64;
        let xb = gpu.storage(big_act);
        let yb = gpu.storage(big_act);
        let wb = gpu.storage(big_w);
        let dyb = gpu.storage(big_act);
        let dxb = gpu.storage(big_act);
        let dwb = gpu.storage(big_w);
        let d128 = |x: usize| ((x + 127) / 128) as u32;
        // (in, out) of the 7 linears in one block.
        let linears = [(dim, dim), (dim, dim), (dim, dim), (dim, dim), (dim, hidden), (dim, hidden), (hidden, dim)];
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();
        let push_block = |steps_f: &mut Vec<_>, steps_b: &mut Vec<_>, t: usize| {
            for &(inp, out) in &linears {
                let (m, kdim, n) = (t as u32, inp as u32, out as u32);
                // forward y[m,n] = x[m,k] @ W[n,k]^T
                steps_f.push(gpu.step(K_FWD, &[&xb, &wb, &yb], &[m, kdim, n], d128(t) * d128(out) * 256));
                // dX[m,k] = dY[m,n] @ W[n,k]
                steps_b.push(gpu.step(K_DX, &[&dyb, &wb, &dxb], &[m, kdim, n, 0], d128(t) * d128(inp) * 256));
                // dW[n,k] += dY[m,n]^T @ X[m,k]
                steps_b.push(gpu.step(K_DW, &[&dyb, &xb, &dwb], &[m, kdim, n], d128(out) * d128(inp) * 256));
            }
        };
        for _ in 0..cfg.n_refiner_layers {
            push_block(&mut fwd, &mut bwd, n_img);
        }
        for _ in 0..cfg.n_refiner_layers {
            push_block(&mut fwd, &mut bwd, ncap);
        }
        for _ in 0..cfg.n_layers {
            push_block(&mut fwd, &mut bwd, ntot);
        }
        let time = |steps: &[gpu_core::Step]| -> f64 {
            gpu.submit(&[], steps);
            gpu.poll_wait();
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                gpu.submit(&[], steps);
                gpu.poll_wait();
                best = best.min(t0.elapsed().as_secs_f64());
            }
            best
        };
        let fwd_s = time(&fwd);
        let bwd_s = time(&bwd);
        let gflop = forward_gflop(&cfg, n_img as u64, ncap as u64);
        let peak = P40_FP32_TFLOPS * 1e3;
        println!("\n=== Z-Image DiT training GEMM sweep — 1×P40 ===");
        println!("config: {} main + {} refiner layers, {ntot} tokens ({n_img} img + {ncap} cap)", cfg.n_layers, 2 * cfg.n_refiner_layers);
        println!("fwd GEMM: {:.1} ms  ({:.0} GFLOP/s, {:.1}% peak) — {gflop:.0} GFLOP", fwd_s * 1e3, gflop / fwd_s, 100.0 * gflop / fwd_s / peak);
        println!("bwd GEMM: {:.1} ms  ({:.0} GFLOP/s, {:.1}% peak) — {:.0} GFLOP (dx+dw = 2×fwd)", bwd_s * 1e3, 2.0 * gflop / bwd_s, 100.0 * 2.0 * gflop / bwd_s / peak, 2.0 * gflop);
        println!("bwd/fwd ratio: {:.2}× (ideal 2.0)", bwd_s / fwd_s);
        println!("fwd+bwd GEMM: {:.1} ms/step", (fwd_s + bwd_s) * 1e3);
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
    let (best, cards) = if device == "int8" {
        let dit = ZImageDitI8::build(cfg.clone(), weights, f, h, w, cap_len);
        eprintln!("  built in {:.1}s", t0.elapsed().as_secs_f64());
        (time(&|| dit.forward(&latent, &cap, 0.5)), 1.0)
    } else if device == "shard" {
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
