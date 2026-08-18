// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 profiler: per-kernel-kind timing of the video DiT block stack at
//! the REAL 22B checkpoint's shape, and of the real (small-component) video
//! VAE decoder - this port's mandatory profile-per-kernel-kind discipline,
//! published BEFORE any implementation code is touched, so an optimization
//! pass is aimed at a measured share of time rather than a guess.
//!
//! Method follows the shared `gpu_core::profile`/`gpu_core::roof` facility
//! (`crates/vqgan/src/bin/vqgan_bench.rs`, `crates/qwen3/src/bin/qwen_bench.rs`
//! are the precedent) rather than a hand-rolled per-kind loop: one whole-pass
//! timing plus a per-kind breakdown, graded against this DEVICE's own
//! measured roofline (never a hardcoded peak - this host has no discrete
//! card, only an integrated GPU, so a P40 literal would be a statement about
//! hardware that is not here).
//!
//! ## Why the DiT bench replays FEWER than 48 layers, at REAL width
//!
//! The real 22B checkpoint is 48 layers at `inner_dim=4096`/`num_heads=32`
//! (this port's own architecture note; `crate::config`'s doc has every other
//! flag). Building that many DISTINCT weight sets, even zero-filled, is ~1.07 GB per
//! layer in f32 (16*dim^2 + 23*dim floats, `crate::dit::dit_tensor_manifest`)
//! - 48 layers is ~51 GB, and this host has ~24 GB free (`free -h`, checked
//! before writing this bench), so that is a real OOM, not a caution.
//!
//! [`LtxBlock::build_steps`] sidesteps the WEIGHT half of that entirely: a
//! dispatch's cost is a pure function of its shape, not the values in its
//! buffers (the same argument `crate::dit::random_tiny_weights`'s doc makes
//! for using random weights at all), so replaying ONE block's already-
//! uploaded weights `N` times profiles identically to `N` distinct-weight
//! blocks - this bench uploads exactly one layer's weights regardless of `N`.
//!
//! That does NOT make `N=48` free, though: chaining `N` layers into ONE
//! combined submit (this bench's whole point - a single-submit graph is what
//! `wan_bench`'s "whole graph" number and `gpu_core::profile`'s device-timed
//! path both assume) keeps every layer's ACTIVATION scratch alive at once,
//! because every dispatch's buffers stay referenced until the combined
//! `Vec<Step>` is submitted. One layer's own scratch (every temp buffer
//! inside `self_attn_and_text_ca` + `mlp_sublayer`, dominated by the
//! `[heads, tokens, tokens]` self-attention score matrix and the FFN's
//! `[tokens, 4*dim]` hidden buffers) is ~0.51 GB at this bench's default
//! `tokens=512`/`ctx_len=256`. 48 layers chained into ONE submit would need
//! ~24 GB of CONCURRENTLY live scratch - this host's entire free RAM
//! (`free -h`, checked before writing this bench), with nothing left for the
//! OS, the weights, or the two other agents building in this same tree. So
//! the default below is `layers=8` (~4.1 GB of scratch) - "4-8 real-width
//! layers" per this milestone's own scoping - not because the WEIGHTS don't
//! fit (they always do, by construction above) but because a single-submit
//! PROFILE of more layers needs more concurrent scratch than this host has
//! spare. Override `layers`/`tokens` if run on a bigger box; the per-kernel-
//! kind SHARES (not just the totals) are the number that matters and are not
//! expected to move much with layer count, since every layer dispatches the
//! identical shape sequence.
//!
//! Usage:
//!   ltxv_bench dit [reps] [layers] [tokens] [ctx_len]   video DiT block stack
//!   ltxv_bench vae [reps] [frames] [height] [width]     real video VAE decode
//!
//! `vae` needs `BRAIN_LTXV_VAE=<path to ltx-2.5-video-vae-conv-bf16.safetensors>`.

use std::time::Instant;

use gpu_core::profile::profile;
use gpu_core::roof;
use gpu_core::{DeviceBuffer, Gpu};
use ltxv::block::{open_device, LtxBlock};
use ltxv::config::LtxDitConfig;
use ltxv::dit::random_tiny_weights;
use ltxv::rope::ltx_rope_tables;
use ltxv::vae3d::{LtxVaeConfig, LtxVaeDecoder};

/// The real LTX-2.5 video-stream DiT shape - 48 layers, video stream 32
/// heads x 128 = 4096 dim, every other flag at M3's one implemented point
/// (`LtxDitConfig::assert_supported`). `num_layers` is the caller's `layers`
/// argument, not necessarily 48 - see this module's doc for why.
fn real_video_dit_config(num_layers: u32) -> LtxDitConfig {
    LtxDitConfig {
        inner_dim: 4096,
        num_heads: 32,
        num_layers,
        in_channels: 128,
        out_channels: 128,
        cross_attention_dim: 4096,
        ff_bias: false,
        cross_attention_adaln: true,
        use_prompt_adaln_single: false,
        use_keyframes_abs_pos_embedding: true,
        norm_eps: 1e-6,
        positional_embedding_theta: 10000.0,
        positional_embedding_max_pos: [20, 2048, 2048],
        timestep_scale_multiplier: 1000,
        use_middle_indices_grid: true,
        apply_gated_attention: false,
    }
}

fn upload_zeros(gpu: &Gpu, n: usize) -> DeviceBuffer {
    let b = gpu.storage(n as u64);
    gpu.write_f32(&b, &vec![0f32; n]);
    b
}

fn bench_dit(reps: usize, layers: u32, t: u32, ctx_len: u32) {
    let cfg = real_video_dit_config(layers);
    cfg.assert_supported();
    let dim = cfg.inner_dim as usize;

    println!(
        "\n=== ltxv video DiT: real width (inner_dim {}, heads {}, head_dim {}) - {} of 48 real layers chained, {} tokens, {} context ===",
        cfg.inner_dim, cfg.num_heads, cfg.head_dim(), layers, t, ctx_len
    );

    // ONE block's weights at real width - see this file's module doc for why
    // that is sufficient regardless of how many layers are chained below.
    let one_layer = LtxDitConfig { num_layers: 1, ..cfg };
    let weights = random_tiny_weights(&one_layer, 7);

    let gpu = open_device(Some("gpu"));
    let t0 = Instant::now();
    let blk = LtxBlock::on(gpu.share(), &cfg, &weights, "transformer_blocks.0", t, ctx_len);
    eprintln!("one block built (weights uploaded) in {:.2} s", t0.elapsed().as_secs_f64());

    // RoPE tables + the per-token adaLN raw table - built once, shared by
    // every chained layer, exactly as `LtxDit::forward`'s own preamble does
    // (`crate::dit`'s doc). Zero-filled positions/timesteps: shape-only, per
    // `crate::rope::ltx_rope_tables`'s doc every axis divides by a config
    // constant (`positional_embedding_max_pos`), never by a position value,
    // so an all-zero grid is well-defined, not degenerate.
    let positions = vec![0f32; 3 * t as usize * 2];
    let rope = ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, &cfg.positional_embedding_max_pos, &positions, t as usize);
    let mut cos_bufs = Vec::with_capacity(rope.heads);
    let mut sin_bufs = Vec::with_capacity(rope.heads);
    for h in 0..rope.heads {
        let (c, s) = rope.head(h);
        let cb = gpu.storage(c.len() as u64);
        gpu.write_f32(&cb, c);
        let sb = gpu.storage(s.len() as u64);
        gpu.write_f32(&sb, s);
        cos_bufs.push(cb);
        sin_bufs.push(sb);
    }
    let adaln_table = vec![0f32; t as usize * cfg.adaln_rows() as usize * dim];
    let ctx_buf = upload_zeros(&gpu, ctx_len as usize * dim);

    // Chain `layers` calls of the SAME block over a device-resident buffer -
    // ONE combined graph, no host round trip between layers (this module's
    // doc explains why that differs from `LtxDit::forward`'s own per-block
    // readback).
    let mut steps = Vec::new();
    let mut cur = upload_zeros(&gpu, t as usize * dim);
    for _ in 0..layers {
        let (s, out) = blk.build_steps(&cur, &adaln_table, &ctx_buf, &cos_bufs, &sin_bufs, t);
        steps.extend(s);
        cur = out;
    }

    let roofs = roof::ensure(&gpu);
    match roofs {
        Some(r) => println!("measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte", r.gflops, r.gbs, r.cache_gbs, r.ridge()),
        None => println!("roofline unmeasured - utilisation columns print '-' rather than a guess"),
    }
    let p = profile(&gpu, "ltxv DiT block stack (real width)", &steps, reps);
    p.print_top(roofs, 20);
}

// ------------------------------------------------------------------ vae ---

fn bench_vae(reps: usize, frames: u32, height: u32, width: u32) {
    let path = std::env::var("BRAIN_LTXV_VAE").unwrap_or_else(|_| panic!("set BRAIN_LTXV_VAE=<path to ltx-2.5-video-vae-conv-bf16.safetensors>"));
    let cfg = LtxVaeConfig::conv25();
    let lat_t = cfg.latent_frames(frames).unwrap_or_else(|| panic!("{frames} frames must be 1 + 8k"));
    let (lh, lw) = (height / 32, width / 32);
    assert!(lh > 0 && lw > 0 && height % 32 == 0 && width % 32 == 0, "height/width must be multiples of 32");

    println!("\n=== ltxv video VAE decode (real weights): latent [{}, {lat_t}, {lh}, {lw}] -> {frames} frames at {width}x{height} ===", cfg.latent_channels);
    let t0 = Instant::now();
    let raw = checkpoint::safetensors::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let weights = ltxv::import::import_vae(raw, &cfg).unwrap_or_else(|e| panic!("importing {path}: {e}"));
    let dec = LtxVaeDecoder::build(&cfg, &weights, lat_t, lh, lw, Some("gpu"));
    drop(weights);
    eprintln!("built in {:.1} s (real checkpoint, {} tensors)", t0.elapsed().as_secs_f64(), cfg.tensor_manifest().len());

    let roofs = roof::ensure(dec.gpu());
    match roofs {
        Some(r) => println!("measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte", r.gflops, r.gbs, r.cache_gbs, r.ridge()),
        None => println!("roofline unmeasured - utilisation columns print '-' rather than a guess"),
    }
    let p = profile(dec.gpu(), "ltxv video VAE decode (real weights)", dec.steps(), reps);
    p.print_top(roofs, 20);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("dit");
    let arg = |i: usize, d: u32| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    match mode {
        "dit" => bench_dit(arg(2, 2) as usize, arg(3, 8), arg(4, 1024), arg(5, 256)),
        "vae" => bench_vae(arg(2, 2) as usize, arg(3, 17), arg(4, 384), arg(5, 384)),
        other => {
            eprintln!("unknown mode {other} (dit|vae)");
            std::process::exit(1);
        }
    }
}
