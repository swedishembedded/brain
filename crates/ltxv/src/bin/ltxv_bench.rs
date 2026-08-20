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
//! timing plus a per-kind breakdown, graded against the selected DEVICE's own
//! measured roofline (never a hardcoded peak - an integrated GPU with no
//! discrete card present would make a P40 literal a statement about
//! hardware that is not there).
//!
//! ## Why the DiT bench replays FEWER than 48 layers, at REAL width
//!
//! The real 22B checkpoint is 48 layers at `inner_dim=4096`/`num_heads=32`
//! (this port's own architecture note; `crate::config`'s doc has every other
//! flag). Building that many DISTINCT weight sets, even zero-filled, is ~1.07 GB per
//! layer in f32 (16*dim^2 + 23*dim floats,
//! `crate::dit::dit_tensor_manifest`) - 48 layers is ~51 GB, well past the
//! free RAM a modest box has (`free -h`), so that is a real OOM, not a
//! caution.
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
//! `[tokens, 4*dim]` hidden buffers) is on the order of ~1 GB at this
//! bench's default `tokens=1024`/`ctx_len=256` (the score matrix term grows
//! quadratically with `tokens`, so this is NOT a fixed per-layer constant -
//! rerun with `--layers 1` and watch `nvidia-smi`/RSS if a different
//! `tokens` needs a fresh bound). 48 layers chained into ONE submit would
//! need on the order of ~48 GB of CONCURRENTLY live scratch - easily a
//! modest box's entire free RAM (`free -h`, checked before writing this
//! bench), with nothing left for the OS, the weights, or any other
//! build/test activity sharing the same machine. So the default below is
//! `layers=8` (a fraction of that) -
//! "4-8 real-width layers" per this milestone's own scoping - not because
//! the WEIGHTS don't fit (they always do, by construction above) but because
//! a single-submit PROFILE of more layers needs more concurrent scratch than
//! a modest box has spare. Override `layers`/`tokens` if run on a bigger box;
//! the per-kernel-kind SHARES (not just the totals) are the number that
//! matters and are not
//! expected to move much with layer count, since every layer dispatches the
//! identical shape sequence.
//!
//! Usage:
//!   ltxv_bench dit [reps] [layers] [tokens] [ctx_len]     video DiT block stack (fp32, synthetic weights)
//!   ltxv_bench vae [reps] [frames] [height] [width]       real video VAE decode
//!   ltxv_bench streamed [layers] [tokens] [ctx_len] [reuse_cache]  real int8 checkpoint, forward_q_streamed's own stage breakdown; reuse_cache=1 shares one block-weight cache across two calls (a generation's cache-miss vs cache-hit shape)
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

/// The real LTX-2.5 video-stream DiT shape - `LtxDitConfig::ltx25_22b()`
/// (48 layers, video stream 32 heads x 128 = 4096 dim, gated attention ON)
/// with only `num_layers` overridden to the caller's `layers` argument (not
/// necessarily 48 - see this module's doc for why). Gated attention is now
/// implemented and parity-proven (`LtxBlock::forward`), so this profiles the
/// REAL op sequence, not a reduced one - the private config this function
/// used to hand-transcribe (and had drifted to `apply_gated_attention:
/// false`, silently profiling a stale op sequence) is gone in favour of the
/// one source of truth in `config.rs`.
fn real_video_dit_config(num_layers: u32) -> LtxDitConfig {
    LtxDitConfig { num_layers, ..LtxDitConfig::ltx25_22b() }
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
    assert!(lh > 0 && lw > 0 && height.is_multiple_of(32) && width.is_multiple_of(32), "height/width must be multiples of 32");

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

// -------------------------------------------------------------- streamed ---

/// The REAL production forward path (`ltxv::dit::forward_q_streamed`,
/// int8 compute, streaming each of `layers` blocks fresh off the real GGUF
/// on every call - `RealDit::forward` in `crate::pipeline` dispatches
/// exactly this), timed end to end AND with the stage_time breakdown
/// (`forward_q_streamed`'s own instrumentation, added this milestone) that
/// splits GGUF read+dequant, host int8 quantize+upload, and GPU forward+wait
/// into three separately attributable totals. This is what Phase 8 step 2
/// needed and `bench_dit` above cannot answer: `bench_dit` profiles the
/// FP32 reference block stack with synthetic weights already resident on
/// device, never touching the checkpoint file or the quantized tier at all.
///
/// `layers` defaults to 4, not the real 48: this crate's own "small shapes
/// first" convention (`bin/ltxv_bench.rs`'s module doc, this port's roadmap).
/// A handful of layers is enough to attribute where each block's own time
/// goes, and scales linearly to the full 48 (each block streams/quantizes/
/// runs independently, so per-block cost does not change with layer count).
///
/// Needs `BRAIN_LTXV_DIT=<path to the real distilled Q8_0 or Q4_K_M GGUF>`,
/// the same env var `crate::pipeline::Paths::dit` resolves.
///
/// The `reuse_cache` argument controls whether a fresh cache is used per
/// call (the pre-Phase-9 behavior, `reuse_cache=false`) or one cache is
/// shared across TWO calls (`reuse_cache=true`, the real `denoise` loop's
/// own shape - see `crate::pipeline::RealDit`'s doc) so the cache-hit path's
/// real speedup is visible in this same harness rather than only inferred.
///
/// Uses [`LtxDitConfig::ltx25_22b`] UNMODIFIED (previously overrode
/// `use_embeddings_connector: false`, the same "profiles a reduced op
/// sequence, not the real one" bug this module's own doc already records
/// once for `apply_gated_attention` - the connector routing this was
/// silently skipping is a real, non-cached ~2.7s of every real forward call,
/// found while diagnosing a real-generation quality issue). `ctx_len` must
/// therefore be a multiple of 128 (the connector's own `num_registers`
/// requirement, `crate::block::EmbeddingsConnector`'s assertion) - the
/// default below already satisfies this.
///
/// Prints each call's raw output stats (mean/std/min/max/nonfinite count) -
/// a cheap sanity check this bench had none of before: a degenerate
/// (all-zero, saturated, or NaN) DiT output is visible here without needing
/// a full generation + VAE decode to notice.
fn bench_streamed(layers: u32, t: u32, ctx_len: u32, reuse_cache: bool) {
    let path = std::env::var("BRAIN_LTXV_DIT").unwrap_or_else(|_| panic!("set BRAIN_LTXV_DIT=<path to the real ltx-2.5-22b-distilled-transformer GGUF>"));
    // stage_time (gpu_core::profile) only prints under BRAIN_PROFILE; this
    // bench's whole purpose is that breakdown, so turn it on unconditionally
    // rather than asking the caller to remember an extra env var.
    std::env::set_var("BRAIN_PROFILE", "1");

    let cfg = LtxDitConfig { num_layers: layers, ..LtxDitConfig::ltx25_22b() };
    cfg.assert_supported();
    println!("\n=== ltxv real-checkpoint streamed forward (int8): {layers} of 48 real layers, {t} tokens, {ctx_len} context (reuse_cache={reuse_cache}) ===");

    let t0 = Instant::now();
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = ltxv::dit::load_head_tensors_from_source(&src, &cfg);
    eprintln!("GGUF opened + head tensors loaded in {:.2} s", t0.elapsed().as_secs_f64());

    let dim = cfg.in_channels as usize;
    let latent = vec![0f32; t as usize * dim];
    let timesteps = vec![500.0f32; t as usize];
    let positions = vec![0f32; 3 * t as usize * 2];
    let keyframes_mask = vec![0f32; t as usize];
    let context = vec![0f32; ctx_len as usize * cfg.cross_attention_dim as usize];
    let context_valid = vec![1f32; ctx_len as usize];

    let call = |label: &str, cache: &ltxv::block::GenerationCache| {
        let t1 = Instant::now();
        let out = ltxv::dit::forward_q_streamed(
            &cfg,
            &src,
            &head,
            Some("gpu"),
            ltxv::block::QTier::Int8,
            &latent,
            &timesteps,
            &positions,
            &keyframes_mask,
            &context,
            ctx_len as usize,
            t as usize,
            &context_valid,
            cache,
        );
        let wall = t1.elapsed().as_secs_f64();
        println!("[{label}] wall time for {layers} layers: {wall:.2} s ({:.2} s/layer)", wall / layers.max(1) as f64);
        let n = out.len() as f64;
        let mean = out.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = out.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let (min, max) = out.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let nan_count = out.iter().filter(|v| !v.is_finite()).count();
        println!("[{label}] OUTPUT STATS: len={} mean={mean:.6} std={:.6} min={min:.6} max={max:.6} nonfinite={nan_count}", out.len(), var.sqrt());
    };
    // `reuse_cache` is exactly "do both calls share one cache?" - a second,
    // default-constructed `GenerationCache` is the honest way to express "no",
    // now that the cache holds connector routing as well as block weights and
    // "empty" is its own default.
    let cache = ltxv::block::GenerationCache::default();
    call("call 1 (always a cache miss - the first forward of a generation)", &cache);
    if reuse_cache {
        call("call 2 (cache hit on every layer - every OTHER forward of a generation)", &cache);
    } else {
        call("call 2 (its OWN fresh cache - the pre-cache per-call cost)", &ltxv::block::GenerationCache::default());
    }
    println!("(the `stage forward_q_streamed: block ...` lines above split GGUF read+dequant / int8 quantize / GPU upload+forward+wait, summed over these {layers} layers - the first two are `cache misses only`: on a cache-hit call they are near-zero by construction, not merely small)");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("dit");
    let arg = |i: usize, d: u32| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    match mode {
        "dit" => bench_dit(arg(2, 2) as usize, arg(3, 8), arg(4, 1024), arg(5, 256)),
        "vae" => bench_vae(arg(2, 2) as usize, arg(3, 17), arg(4, 384), arg(5, 384)),
        "streamed" => bench_streamed(arg(2, 4), arg(3, 512), arg(4, 256), a.get(5).map(|s| s == "1").unwrap_or(false)),
        other => {
            eprintln!("unknown mode {other} (dit|vae|streamed)");
            std::process::exit(1);
        }
    }
}
