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
//!   ltxv_bench vae [reps] [frames] [height] [width]       real video VAE decode, WHOLE-clip path only (past `ltxv::vae3d::should_tile`'s ceiling a generation tiles instead, which this does not measure - it says so and stops rather than reporting a number for a path production would not take)
//!   ltxv_bench streamed [layers] [tokens] [ctx_len] [reuse_cache] [resident] [distinct_timesteps] [warm_reps]  real int8 checkpoint, forward_q_streamed's own stage breakdown; reuse_cache=1 shares one block-weight cache across two calls (a generation's cache-miss vs cache-hit shape); resident=1 additionally shares ONE device session, so a warm call re-uploads nothing (a real generation's actual shape - see crate::devres); distinct_timesteps sizes the host adaLN stage (1 = plain t2v, 2 = anchored or long-form, tokens = no dedup at all, i.e. what this stage cost before the dedup existed); warm_reps repeats the cache-hit call so a headline is a best-of-N and not one sample (the first warm call is the warm-up and is excluded)
//!   ltxv_bench streamed-av [layers] [video_tokens] [ctx_len] [audio_tokens] [reuse_cache] [resident] [distinct_timesteps] [warm_reps]  the same, for the JOINT audio+video forward - read against `streamed` at the same video token count to get what audio costs
//!   ltxv_bench decode <latent.bin> <whole|tiled> [h0 h1 w0 w1]   decode a DUMPED latent (see ltxv::latentdump), optionally a latent-cell crop
//!
//! `vae` and `decode` need `BRAIN_LTXV_VAE=<path to ltx-2.5-video-vae-conv-bf16.safetensors>`.
//!
//! ## What `decode` is for
//!
//! A generation is two stages that fail in visually identical ways - a bad
//! latent and a bad decode both look like smeared, warped video. `decode`
//! separates them: it takes a latent dumped by
//! `BRAIN_LTXV_LATENT_DUMP=<path>` on a real run and decodes it AGAIN,
//! through either path, at either a full or a cropped extent, printing the
//! per-latent-frame latent statistics and the per-pixel-frame
//! frame-to-frame difference curve. Two `decode` runs over the SAME dump
//! differ only in the decoder, so any difference between their curves is the
//! decoder's; a curve that is already broken in the whole-path arm is the
//! DiT's.
//!
//! The crop exists because the shape that motivates all of this
//! (25 frames at 1920x1088, 52.2 Mpx) cannot be whole-decoded on a 24 GiB
//! card at all - but a spatial crop of its latent can, which is what makes a
//! whole-vs-tiled comparison possible at the real resolution.

use std::time::Instant;

use gpu_core::profile::profile;
use gpu_core::roof;
use gpu_core::{DeviceBuffer, Gpu, Step};
use ltxv::block::{LtxBlock, KERNELS};
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

    let gpu = Gpu::open(Some("gpu"), &KERNELS);
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
    report(&gpu, "ltxv DiT block stack (real width)", &steps, reps, roofs);
}

/// One profile pass, printed against the device's MEASURED roofline, followed
/// by the DEFECT rows - every kernel running under its own roof's defect floor.
///
/// The same shape `wan_bench`, `qwen_bench`, `unet_bench`, `vqgan_bench` and
/// `mm3_bench` all use, and the reason this exists: this was the only `*_bench`
/// in the tree printing `print_top` alone, so this model's roof-floor defects
/// were invisible to its own harness while every other model's were not. A
/// profile that ranks kernels but never says which are BELOW their roof leaves
/// the reader to do the roofline arithmetic by hand, which is exactly what
/// nobody does.
fn report(gpu: &Gpu, label: &str, steps: &[Step], reps: usize, roofs: Option<roof::Roofs>) {
    let p = profile(gpu, label, steps, reps);
    p.print_top(roofs, 20);
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<24} {:>5.1}% of its {} roof (floor {:.0}%) - {:.1}% of this pass",
                row.name,
                pct,
                bound.as_str(),
                bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
}

// ------------------------------------------------------------------ vae ---

fn bench_vae(reps: usize, frames: u32, height: u32, width: u32) {
    let path = std::env::var("BRAIN_LTXV_VAE").unwrap_or_else(|_| panic!("set BRAIN_LTXV_VAE=<path to ltx-2.5-video-vae-conv-bf16.safetensors>"));
    let cfg = LtxVaeConfig::conv25();
    let lat_t = cfg.latent_frames(frames).unwrap_or_else(|| panic!("{frames} frames must be 1 + 8k"));
    let (lh, lw) = (height / 32, width / 32);
    assert!(lh > 0 && lw > 0 && height.is_multiple_of(32) && width.is_multiple_of(32), "height/width must be multiples of 32");

    println!("\n=== ltxv video VAE decode (real weights, WHOLE-clip path): latent [{}, {lat_t}, {lh}, {lw}] -> {frames} frames at {width}x{height} ===", cfg.latent_channels);

    // This bench measures `LtxVaeDecoder`, the whole-clip path. A generation
    // picks between that and the overlapping-tile path by `should_tile`
    // (`ltxv::pipeline::decode_video`), so past the ceiling the two disagree
    // about what "the VAE decode" even is. Building the whole graph anyway
    // ends in a wgpu out-of-memory panic that leaks the device - a stack
    // trace where the honest answer is one line, and worse, a number from a
    // smaller run that silently reads as production's.
    if ltxv::vae3d::should_tile(frames, height, width) {
        let px = frames as u64 * height as u64 * width as u64;
        println!(
            "this geometry is {:.1} Mpx, past the {:.1} Mpx whole-decode ceiling, so a real generation TILES here and this bench would not measure it.\n\
             Bench a geometry under the ceiling, or set BRAIN_LTXV_VAE_TILE=0 to force the whole path and measure it anyway (expect an out-of-memory abort if it does not fit).",
            px as f64 / 1e6,
            ltxv::vae3d::WHOLE_DECODE_MAX_PIXELS as f64 / 1e6,
        );
        return;
    }

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
    report(dec.gpu(), "ltxv video VAE decode (real weights)", dec.steps(), reps, roofs);
}

// ---------------------------------------------------------------- decode ---

/// Decode a latent dumped by `BRAIN_LTXV_LATENT_DUMP` (see
/// [`ltxv::latentdump`]) through an explicitly chosen path, over an optional
/// latent-cell crop, and print what the two halves of a generation each
/// contribute to a temporal-stability defect: the per-latent-frame latent
/// statistics (the DiT's output, before any decoder touches it) and the
/// per-pixel-frame frame-to-frame difference curve (the decoded result).
fn bench_decode(path: &str, mode: &str, crop: Option<(u32, u32, u32, u32)>) {
    let vae = std::env::var("BRAIN_LTXV_VAE").unwrap_or_else(|_| panic!("set BRAIN_LTXV_VAE=<path to ltx-2.5-video-vae-conv-bf16.safetensors>"));
    let (shape, data) = ltxv::latentdump::read(path).unwrap_or_else(|e| panic!("{e}"));
    let cfg = LtxVaeConfig::conv25();

    // The latent's own statistics, per latent frame. This is the DiT's output
    // and no decoder has touched it: a frame whose std or extremes are out of
    // line with its neighbours is a generation defect, full stop.
    println!("\n=== latent {path}: [{}, {}, {}, {}] ===", shape.c, shape.t, shape.h, shape.w);
    let plane = (shape.h * shape.w) as usize;
    for ti in 0..shape.t as usize {
        let (mut n, mut sum, mut sq, mut lo, mut hi) = (0usize, 0f64, 0f64, f32::INFINITY, f32::NEG_INFINITY);
        for ci in 0..shape.c as usize {
            for &v in &data[(ci * shape.t as usize + ti) * plane..(ci * shape.t as usize + ti + 1) * plane] {
                n += 1;
                sum += v as f64;
                sq += (v as f64) * (v as f64);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        let mean = sum / n as f64;
        println!("  latent frame {ti}: mean {mean:+.5} std {:.5} min {lo:+.4} max {hi:+.4}", (sq / n as f64 - mean * mean).max(0.0).sqrt());
    }
    // Adjacent-latent-frame distance, the latent-space twin of the pixel
    // frame-to-frame curve.
    for ti in 1..shape.t as usize {
        let (mut acc, mut n) = (0f64, 0usize);
        for ci in 0..shape.c as usize {
            let a = &data[(ci * shape.t as usize + ti - 1) * plane..(ci * shape.t as usize + ti) * plane];
            let b = &data[(ci * shape.t as usize + ti) * plane..(ci * shape.t as usize + ti + 1) * plane];
            for (x, y) in a.iter().zip(b) {
                acc += (x - y).abs() as f64;
                n += 1;
            }
        }
        println!("  latent frames {} -> {ti}: mean |delta| {:.5}", ti - 1, acc / n as f64);
    }

    // `BRAIN_LTXV_DECODE_LF_SUBST=dst=src` overwrites latent frame `dst` with
    // latent frame `src` before decoding. Decoding once with and once without
    // it, and differencing the two results per pixel frame, measures which
    // pixel frames a given latent frame actually controls - the decoder's
    // temporal receptive field, observed rather than derived from the
    // `1 + 8k` rule. That is the measurement that tells a defect confined to
    // one latent frame apart from one confined to a range of pixel frames.
    let mut data = data;
    if let Ok(spec) = std::env::var("BRAIN_LTXV_DECODE_LF_SUBST") {
        let (d, s) = spec.split_once('=').unwrap_or_else(|| panic!("BRAIN_LTXV_DECODE_LF_SUBST wants dst=src, got {spec}"));
        let (d, s): (usize, usize) = (d.trim().parse().unwrap(), s.trim().parse().unwrap());
        assert!(d < shape.t as usize && s < shape.t as usize, "latent frames {d}/{s} out of range for t={}", shape.t);
        for ci in 0..shape.c as usize {
            let (dst, src) = ((ci * shape.t as usize + d) * plane, (ci * shape.t as usize + s) * plane);
            let row: Vec<f32> = data[src..src + plane].to_vec();
            data[dst..dst + plane].copy_from_slice(&row);
        }
        println!("  substituted latent frame {d} <- {s}");
    }

    // `BRAIN_LTXV_DECODE_UPSAMPLE=<spatial upscaler path>` runs the real x2
    // latent upscaler over the latent before decoding, with NO refinement
    // pass. That isolates the upscaler: whatever the decoded result looks
    // like is what the two-stage path hands its stage 2 as a seed, before
    // any denoising has had a chance to help or hurt it.
    let (mut shape, mut data) = (shape, data);
    if let Ok(up) = std::env::var("BRAIN_LTXV_DECODE_UPSAMPLE") {
        let raw = checkpoint::safetensors::read(&up).unwrap_or_else(|e| panic!("reading {up}: {e}"));
        let ucfg = ltxv::upsampler::LatentUpsamplerConfig::spatial_x2();
        let uw = ltxv::import::import_upsampler(raw, &ucfg).unwrap_or_else(|e| panic!("importing {up}: {e}"));
        let ups = ltxv::upsampler::LatentUpsampler::build(&ucfg, &uw, shape.t, shape.h, shape.w, Some("gpu"));
        // The same `upsample_video` sandwich production takes; set
        // `BRAIN_LTXV_DECODE_UPSAMPLE_RAW=1` to skip it, which is how the
        // "no un-normalize" row of that function's own table was measured.
        data = if std::env::var("BRAIN_LTXV_DECODE_UPSAMPLE_RAW").is_ok() {
            ups.upsample(&data)
        } else {
            let vraw2 = checkpoint::safetensors::read(&vae).unwrap_or_else(|e| panic!("reading {vae}: {e}"));
            let vw = ltxv::import::import_vae(vraw2, &cfg).unwrap_or_else(|e| panic!("importing {vae}: {e}"));
            let (m, sd) = ltxv::vae3d::per_channel_statistics(&vw);
            ltxv::upsampler::upsample_video(&ups, &m, &sd, &data)
        };
        let (_, _, nh, nw) = ups.out_shape();
        shape = ltxv::latentdump::LatentShape { c: shape.c, t: shape.t, h: nh, w: nw };
        println!("  upsampled x2 -> [{}, {}, {}, {}]", shape.c, shape.t, shape.h, shape.w);
        for ti in 0..shape.t as usize {
            let plane2 = (shape.h * shape.w) as usize;
            let (mut n, mut sum, mut sq) = (0usize, 0f64, 0f64);
            for ci in 0..shape.c as usize {
                for &v in &data[(ci * shape.t as usize + ti) * plane2..(ci * shape.t as usize + ti + 1) * plane2] {
                    n += 1;
                    sum += v as f64;
                    sq += (v as f64) * (v as f64);
                }
            }
            let mean = sum / n as f64;
            println!("    upsampled latent frame {ti}: mean {mean:+.5} std {:.5}", (sq / n as f64 - mean * mean).max(0.0).sqrt());
        }
    }
    let plane = (shape.h * shape.w) as usize;

    // The crop, in latent cells. A crop keeps every latent frame (the axis the
    // defect under investigation lives on) and narrows the spatial extent
    // until a whole-clip decode fits on one card.
    let (h0, h1, w0, w1) = crop.unwrap_or((0, shape.h, 0, shape.w));
    assert!(h0 < h1 && h1 <= shape.h && w0 < w1 && w1 <= shape.w, "crop {h0}..{h1} x {w0}..{w1} out of range for {}x{}", shape.h, shape.w);
    let (lh, lw) = (h1 - h0, w1 - w0);
    let mut cropped = Vec::with_capacity(shape.c as usize * shape.t as usize * (lh * lw) as usize);
    for ci in 0..shape.c as usize {
        for ti in 0..shape.t as usize {
            let base = (ci * shape.t as usize + ti) * plane;
            for y in h0..h1 {
                let row = base + (y * shape.w) as usize;
                cropped.extend_from_slice(&data[row + w0 as usize..row + w1 as usize]);
            }
        }
    }

    let frames = 1 + 8 * (shape.t - 1);
    let (ph, pw) = (lh * 32, lw * 32);
    println!("\ndecoding [{}, {}, {lh}, {lw}] ({h0}..{h1} x {w0}..{w1}) -> {frames} frames at {pw}x{ph}, path = {mode}", shape.c, shape.t);

    let raw = checkpoint::safetensors::read(&vae).unwrap_or_else(|e| panic!("reading {vae}: {e}"));
    let weights = ltxv::import::import_vae(raw, &cfg).unwrap_or_else(|e| panic!("importing {vae}: {e}"));
    let t0 = Instant::now();
    let pixels = match mode {
        "whole" => {
            let dec = LtxVaeDecoder::build(&cfg, &weights, shape.t, lh, lw, Some("gpu"));
            dec.decode(&cropped)
        }
        "tiled" => {
            let dec = ltxv::vae3d::LtxVaeTiledDecoder::auto(&cfg, &weights, shape.t, lh, lw, Some("gpu"));
            println!("  {} tiles, overlap waste {:.3}x", dec.plan().tiles().len(), dec.plan().overlap_waste());
            dec.decode_with(&cropped, |_, _| {})
        }
        other => panic!("unknown decode path {other} (whole|tiled)"),
    };
    println!("  decoded in {:.1} s", t0.elapsed().as_secs_f64());

    let diffs = ltxv::clipmetric::frame_to_frame_diffs(&pixels, frames as usize, ph as usize, pw as usize);
    println!("\nframe-to-frame mean |delta| (128x128 probe, 0-255 units):");
    for (i, d) in diffs.iter().enumerate() {
        println!("  frame {:>3} <- {:>3}: {d:8.3}", i + 1, i);
    }
    println!("blowup ratio (max/median) = {:.2}", ltxv::clipmetric::blowup_ratio(&diffs));

    if let Ok(out) = std::env::var("BRAIN_LTXV_PIXEL_DUMP") {
        ltxv::latentdump::write(&out, ltxv::latentdump::LatentShape { c: 3, t: frames, h: ph, w: pw }, &pixels).unwrap_or_else(|e| panic!("{e}"));
        println!("wrote decoded pixels to {out}");
    }
}

// -------------------------------------------------------------- streamed ---

/// Cumulative DEVICE kernel time on `gpu` since it was opened, in ms - the sum
/// of the same per-kernel table `BRAIN_PROFILE` prints.
///
/// A resident session holds ONE device for its whole life, so this counter is
/// cumulative across every forward that ran on it and a single call's device
/// time is the DIFFERENCE between two readings. That difference is what
/// separates "the card is the bottleneck" from "the host is", and it is not
/// derivable from wall clock: wall clock around a forward includes the host
/// patchify/adaLN/connector stages, the weight streaming and the readback,
/// none of which is a kernel. `None` on a backend that cannot timestamp
/// kernels - reported as unavailable rather than as zero.
fn device_kernel_ms(gpu: &Gpu) -> Option<f64> {
    Some(gpu.kernel_times()?.iter().map(|(_, ms, _)| ms).sum())
}

/// One call's wall/device split, printed the same way by both streamed
/// benches: `prev` carries the previous reading of [`device_kernel_ms`] so the
/// per-call figure is a difference and not a total.
fn report_call(label: &str, layers: u32, wall: f64, device_ms: Option<f64>, prev: &std::cell::Cell<f64>) {
    print!("[{label}] wall time for {layers} layers: {wall:.2} s ({:.2} s/layer)", wall / f64::from(layers.max(1)));
    match device_ms {
        Some(total) => {
            let this = total - prev.get();
            prev.set(total);
            println!(" | device {:.2} s ({:.1}% of wall), host {:.2} s", this / 1e3, 100.0 * this / 1e3 / wall.max(1e-9), wall - this / 1e3);
        }
        None => println!(" | device time unavailable on this backend"),
    }
}

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
/// silently skipping is a real, non-cached part of every real forward call,
/// found while diagnosing a real-generation quality issue). `ctx_len` must
/// therefore be a multiple of 128 (the connector's own `num_registers`
/// requirement, `crate::block::EmbeddingsConnector`'s assertion) - the
/// default below already satisfies this.
///
/// Prints each call's raw output stats (mean/std/min/max/nonfinite count) -
/// a cheap sanity check this bench had none of before: a degenerate
/// (all-zero, saturated, or NaN) DiT output is visible here without needing
/// a full generation + VAE decode to notice.
fn bench_streamed(layers: u32, t: u32, ctx_len: u32, reuse_cache: bool, resident: bool, distinct_timesteps: u32, warm_reps: u32) {
    let path = std::env::var("BRAIN_LTXV_DIT").unwrap_or_else(|_| panic!("set BRAIN_LTXV_DIT=<path to the real ltx-2.5-22b-distilled-transformer GGUF>"));
    // stage_time (gpu_core::profile) only prints under BRAIN_PROFILE; this
    // bench's whole purpose is that breakdown, so turn it on unconditionally
    // rather than asking the caller to remember an extra env var.
    std::env::set_var("BRAIN_PROFILE", "1");

    let cfg = LtxDitConfig { num_layers: layers, ..LtxDitConfig::ltx25_22b() };
    cfg.assert_supported();
    println!("\n=== ltxv real-checkpoint streamed forward (int8): {layers} of 48 real layers, {t} tokens, {ctx_len} context, {distinct_timesteps} distinct timesteps (reuse_cache={reuse_cache}, device_resident={resident}) ===");

    let t0 = Instant::now();
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = ltxv::dit::load_head_tensors_from_source(&src, &cfg);
    eprintln!("GGUF opened + head tensors loaded in {:.2} s", t0.elapsed().as_secs_f64());

    let dim = cfg.in_channels as usize;
    let latent = vec![0f32; t as usize * dim];
    // `distinct_timesteps` is the ONE input that decides how much work the
    // host adaLN-single stage does: it computes one row per DISTINCT per-token
    // timestep (`dit::adaln::RowTable`) and uploads that many rows, so the
    // stage's cost is proportional to this number, not to `t`.
    //
    // 1 is a plain text-to-video step (`denoise_mask` all ones - every token
    // gets the schedule's sigma); 2 is anything anchored or carrying a
    // long-form context (the frozen tokens sit at `0 * sigma`); `t` is the
    // degenerate case where no two tokens agree, which is what this stage cost
    // BEFORE the dedup existed and is therefore the honest measurement of what
    // the fallback costs. Interleaved rather than blocked, so a run cannot
    // accidentally measure a contiguous-split special case.
    let distinct = distinct_timesteps.clamp(1, t.max(1));
    let timesteps: Vec<f32> = (0..t).map(|i| 500.0 + (i % distinct) as f32).collect();
    let positions = vec![0f32; 3 * t as usize * 2];
    let keyframes_mask = vec![0f32; t as usize];
    let context = vec![0f32; ctx_len as usize * cfg.cross_attention_dim as usize];
    let context_valid = vec![1f32; ctx_len as usize];

    // ONE session for both calls when `resident` - which is exactly what a real
    // generation does (`crate::pipeline::RealDit` holds one per card for the
    // whole denoise loop). A transient session is byte-for-byte the
    // pre-residency path: a fresh device per call, every block re-uploaded.
    // `None`, not `Some("gpu")`: `Gpu::open(None, ..)` goes through `Gpu::new`,
    // which honours `BRAIN_DEVICE` - so this bench can be pointed at brain's
    // native Vulkan backend (`BRAIN_DEVICE=vulkan`), which does NOT have
    // wgpu's doubled resident-buffer cost and therefore has a completely
    // different residency budget. wgpu is still the default, so an unset
    // environment measures exactly what it measured before.
    let dev: Option<&str> = None;
    let session = if resident {
        ltxv::devres::DitSession::resident(&cfg, ltxv::block::QTier::Int8, dev, t as usize)
    } else {
        ltxv::devres::DitSession::transient(dev)
    };
    let prev_device = std::cell::Cell::new(0f64);
    let call = |label: &str, cache: &ltxv::block::GenerationCache| -> f64 {
        let t1 = Instant::now();
        let out = ltxv::dit::forward_q_streamed_in(
            &session,
            &cfg,
            &src,
            &head,
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
        // Only a RESIDENT session can be asked: a transient one's
        // `device_for_call` OPENS a device, which would both cost seconds and
        // report a counter no forward ever ran on.
        let device_ms = session.is_resident().then(|| device_kernel_ms(&session.device_for_call())).flatten();
        report_call(label, layers, wall, device_ms, &prev_device);
        let n = out.len() as f64;
        let mean = out.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = out.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let (min, max) = out.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
        let nan_count = out.iter().filter(|v| !v.is_finite()).count();
        println!("[{label}] OUTPUT STATS: len={} mean={mean:.6} std={:.6} min={min:.6} max={max:.6} nonfinite={nan_count}", out.len(), var.sqrt());
        let rs = session.stats();
        println!("[{label}] DEVICE RESIDENCY: slots={} device_hits={} device_uploads={}", rs.slots, rs.hits, rs.uploads);
        wall
    };
    // `reuse_cache` is exactly "do both calls share one cache?" - a second,
    // default-constructed `GenerationCache` is the honest way to express "no",
    // now that the cache holds connector routing as well as block weights and
    // "empty" is its own default.
    let cache = ltxv::block::GenerationCache::default();
    call("call 1 (always a cache miss - the first forward of a generation)", &cache);
    if reuse_cache {
        // Every warm call is a repetition of the same measurement, and the
        // FIRST of them is a warm-up: it is the call that pays for whatever the
        // cold call left cold (the device residency window's own last uploads,
        // the allocator pool settling). Best-of-N over the rest is what gets
        // reported; the run prints them all so a reader can see the spread
        // rather than trust a single sample.
        let mut warm = Vec::new();
        for r in 0..warm_reps.max(1) {
            warm.push(call(&format!("call {} (cache hit on every layer - every OTHER forward of a generation)", r + 2), &cache));
        }
        let best = warm.iter().skip(1).fold(f64::INFINITY, |a, &b| a.min(b));
        if best.is_finite() {
            println!("[warm] best of {} (first warm call excluded as warm-up): {best:.2} s", warm.len() - 1);
        }
    } else {
        call("call 2 (its OWN fresh cache - the pre-cache per-call cost)", &ltxv::block::GenerationCache::default());
    }
    println!("(the `stage forward_q_streamed: block ...` lines above split GGUF read+dequant / int8 quantize / GPU upload+forward+wait, summed over these {layers} layers - the first two are `cache misses only`: on a cache-hit call they are near-zero by construction, not merely small)");
}

/// Wall time of the REAL base vocoder over `t` mel frames, best of `reps`.
///
/// Real weights, because this stage's cost is dominated by how its boundary
/// handling and its convolutions are dispatched rather than by any shape the
/// weights could be faked at - and the file is small enough to read per run.
/// Reports wall time and the audio seconds it produced, so a run can be read
/// as a real-time factor without the harness asserting one.
fn bench_vocoder(reps: usize, t: u32) {
    let path = std::env::var("BRAIN_LTXV_AUDIO_VAE").unwrap_or_else(|_| {
        eprintln!("ltxv_bench vocoder: set BRAIN_LTXV_AUDIO_VAE to the audio VAE safetensors");
        std::process::exit(1);
    });
    let cfg = ltxv::vocoder::VocoderConfig::ltx25();
    let w = ltxv::import::import_vocoder(checkpoint::safetensors::read(&path).expect("read audio vae"), &cfg).expect("import vocoder");
    let (channels, mel_bins) = (2u32, 64u32);
    // Shape-correct scratch: the timing is a function of the dispatch shapes,
    // not of the values (the same argument `random_tiny_weights`'s own doc
    // makes), and a deterministic ramp keeps two runs comparable.
    let mel: Vec<f32> = (0..(channels * t * mel_bins) as usize).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    let mut best = f64::MAX;
    for r in 0..reps.max(1) {
        let t0 = Instant::now();
        let wave = ltxv::vocoder::synthesize(&cfg, &w, &mel, channels, t, mel_bins, None);
        let wall = t0.elapsed().as_secs_f64();
        best = best.min(wall);
        println!("[rep {r}] {wall:.3} s for {t} mel frames -> {} samples", wave.len());
    }
    let seconds_of_audio = f64::from(t) / 100.0; // 16 kHz / hop 160
    println!("[vocoder] best of {reps}: {best:.3} s wall for {seconds_of_audio:.2} s of 16 kHz stereo audio (RTF {:.2})", best / seconds_of_audio);
}

/// One JOINT audio+video forward at the real 22B config, off the real GGUF.
///
/// The number this exists to produce is what audio COSTS: the same clip's
/// video tokens run through a model that also carries an audio stream and
/// bidirectional cross-attention every block, against `streamed`'s
/// video-only figure at the same `tokens`. Reports the load and the forward
/// separately, because they are paid at completely different rates - the load
/// once per process, the forward once per denoise step per CFG branch.
fn bench_av(tokens: u32, ctx_len: u32, frames: u32, fps: u32) {
    let path = std::env::var("BRAIN_LTXV_DIT").unwrap_or_else(|_| {
        eprintln!("ltxv_bench av: set BRAIN_LTXV_DIT to the 22B AV transformer GGUF");
        std::process::exit(1);
    });
    let cfg = ltxv::LtxAvDitConfig::ltx25();
    println!("[av] host fp32 expansion: {} GiB", ltxv::av_stream::host_floats(&cfg) * 4 / (1 << 30));
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).expect("open the AV GGUF");
    let t0 = Instant::now();
    let w = ltxv::av_stream::AvWeights::load(&src, cfg).expect("load the AV weights");
    println!("[av] weight load + dequant: {:.1} s", t0.elapsed().as_secs_f64());

    let ta = ltxv::audio::latent_frames(frames as usize, fps as usize);
    let vdim = cfg.video.in_channels as usize;
    let adim = ltxv::audio::TOKEN_DIM as usize;
    let tv = tokens as usize;
    // Shape-correct scratch. A forward's cost is a function of its shapes, not
    // of the values in its buffers.
    let v_latent: Vec<f32> = (0..tv * vdim).map(|i| ((i % 71) as f32 / 71.0) - 0.5).collect();
    let a_latent: Vec<f32> = (0..ta * adim).map(|i| ((i % 53) as f32 / 53.0) - 0.5).collect();
    // The frame/height/width factorisation only has to MULTIPLY to `tokens`;
    // the positions it yields are real ones either way.
    let (lh, lw) = (22usize, 40usize);
    let lat_t = tv / (lh * lw);
    let v_positions = ltxv::pipeline::real_pixel_positions(lat_t, lh, lw, f64::from(fps));
    let a_positions = ltxv::audio::positions(ta);
    let context: Vec<f32> = (0..(ctx_len as usize * cfg.video.cross_attention_dim as usize)).map(|i| ((i % 37) as f32 / 37.0) - 0.5).collect();
    // The audio stream's own connector is built for the AUDIO aggregate
    // head's narrower output, so its context is a different width - see
    // `ltxv::pipeline::TextContext`.
    let a_context: Vec<f32> = (0..(ctx_len as usize * cfg.audio.connector_inner_dim() as usize)).map(|i| ((i % 29) as f32 / 29.0) - 0.5).collect();

    let d = ltxv::av_stream::AvDenoiser::new(w, None);
    let v_timesteps = vec![1.0f32; lat_t * lh * lw];
    let v_keyframes_mask = vec![0f32; lat_t * lh * lw];
    let a_timesteps = vec![1.0f32; ta];
    let context_valid = vec![1.0f32; ctx_len as usize];
    // The SAME step struct `streamed-av` hands the quantized arm, which is
    // what makes the two rows of this comparison a comparison.
    let step = ltxv::dit::AvStreamedStep {
        v_latent: &v_latent,
        v_timesteps: &v_timesteps,
        v_positions: &v_positions,
        v_keyframes_mask: &v_keyframes_mask,
        v_context: &context,
        v_context_len: ctx_len as usize,
        tv: lat_t * lh * lw,
        v_sigma: 1.0,
        v_context_valid: &context_valid,
        a_latent: &a_latent,
        a_timesteps: &a_timesteps,
        a_positions: &a_positions,
        a_context: &a_context,
        a_context_len: ctx_len as usize,
        ta,
        a_sigma: 1.0,
        a_context_valid: &context_valid,
    };
    for r in 0..2 {
        let t1 = Instant::now();
        let (v, a) = d.forward(&step);
        let nonfinite = v.iter().chain(&a).filter(|x| !x.is_finite()).count();
        println!(
            "[av rep {r}] {:.1} s for {} video tokens + {ta} audio tokens (video out {}, audio out {}, nonfinite {nonfinite})",
            t1.elapsed().as_secs_f64(),
            lat_t * lh * lw,
            v.len(),
            a.len()
        );
    }
}

/// The STREAMED, int8, device-resident joint audio+video forward
/// (`ltxv::dit::av_forward_q_streamed_in`) - the AV counterpart of
/// [`bench_streamed`], and the number the audio-visual path is supposed to be
/// read against.
///
/// Deliberately the same harness shape as `streamed`, argument for argument,
/// because the whole claim being measured is a RATIO: what an AV forward costs
/// against a video-only forward at the SAME video token count. Two calls, the
/// first a guaranteed cache miss (a generation's first forward) and the second
/// a hit on every layer (every other forward), with one shared
/// `GenerationCache` and - at `resident=1` - one shared `AvDitSession`, which
/// is exactly what a real denoise loop holds.
///
/// `audio_tokens` is the audio stream's own length. A real clip's is
/// `round(frames / fps * 25)` (`ltxv::audio::latent_frames`); passing it
/// explicitly keeps this harness from having to reconstruct a clip geometry it
/// does not otherwise need.
fn bench_streamed_av(layers: u32, tv: u32, ctx_len: u32, ta: u32, reuse_cache: bool, resident: bool, distinct_timesteps: u32, warm_reps: u32) {
    let path = std::env::var("BRAIN_LTXV_DIT").unwrap_or_else(|_| panic!("set BRAIN_LTXV_DIT=<path to the real ltx-2.5-22b-distilled-transformer GGUF>"));
    std::env::set_var("BRAIN_PROFILE", "1");

    let mut cfg = ltxv::LtxAvDitConfig::ltx25();
    cfg.video.num_layers = layers;
    cfg.assert_supported();
    let per_block = ltxv::block::cached_av_block_bytes(&cfg.video, &cfg.audio, ltxv::block::QTier::Int8);
    println!("\n=== ltxv real-checkpoint STREAMED audio+video forward (int8): {layers} of 48 real layers, {tv} video + {ta} audio tokens, {ctx_len} context, {distinct_timesteps} distinct timesteps (reuse_cache={reuse_cache}, device_resident={resident}) ===");
    println!("[av] cached int8 bytes per AV block: {} MiB (all {} layers: {} MiB)", per_block / (1 << 20), layers, per_block * u64::from(layers) / (1 << 20));

    let t0 = Instant::now();
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = ltxv::dit::load_av_head_tensors_from_source(&src, &cfg);
    eprintln!("GGUF opened + AV head tensors loaded in {:.2} s", t0.elapsed().as_secs_f64());

    // Shape-correct scratch: a dispatch's cost is a function of its shape.
    let v_latent: Vec<f32> = (0..tv as usize * cfg.video.in_channels as usize).map(|i| ((i % 71) as f32 / 71.0) - 0.5).collect();
    let a_latent: Vec<f32> = (0..ta as usize * cfg.audio.in_channels as usize).map(|i| ((i % 53) as f32 / 53.0) - 0.5).collect();
    let distinct = distinct_timesteps.clamp(1, tv.max(1));
    let v_timesteps: Vec<f32> = (0..tv).map(|i| 500.0 + (i % distinct) as f32).collect();
    let a_timesteps: Vec<f32> = (0..ta).map(|i| 500.0 + (i % distinct.min(ta.max(1))) as f32).collect();
    let (lh, lw) = (22usize, 40usize);
    let lat_t = (tv as usize).div_ceil(lh * lw).max(1);
    let mut v_positions = ltxv::pipeline::real_pixel_positions(lat_t, lh, lw, 24.0);
    // `real_pixel_positions` fills a whole `lat_t * lh * lw` grid; the bench's
    // token count need not be a multiple of one, so keep the leading `tv`
    // positions of each of the three axes.
    let grid = lat_t * lh * lw;
    let mut trimmed = Vec::with_capacity(3 * tv as usize * 2);
    for axis in 0..3 {
        trimmed.extend_from_slice(&v_positions[axis * grid * 2..axis * grid * 2 + tv as usize * 2]);
    }
    v_positions = trimmed;
    let a_positions = ltxv::audio::positions(ta as usize);
    let v_context = vec![0f32; ctx_len as usize * cfg.video.cross_attention_dim as usize];
    let a_context = vec![0f32; ctx_len as usize * cfg.audio.connector_inner_dim() as usize];
    let context_valid = vec![1f32; ctx_len as usize];

    let dev: Option<&str> = None;
    let session = if resident {
        ltxv::devres::AvDitSession::resident(&cfg, ltxv::block::QTier::Int8, dev, tv as usize)
    } else {
        ltxv::devres::AvDitSession::transient(dev)
    };
    let step = ltxv::dit::AvStreamedStep {
        v_latent: &v_latent,
        v_timesteps: &v_timesteps,
        v_positions: &v_positions,
        v_keyframes_mask: &vec![0f32; tv as usize],
        v_context: &v_context,
        v_context_len: ctx_len as usize,
        tv: tv as usize,
        v_sigma: 1.0,
        v_context_valid: &context_valid,
        a_latent: &a_latent,
        a_timesteps: &a_timesteps,
        a_positions: &a_positions,
        a_context: &a_context,
        a_context_len: ctx_len as usize,
        ta: ta as usize,
        a_sigma: 1.0,
        a_context_valid: &context_valid,
    };
    let prev_device = std::cell::Cell::new(0f64);
    let call = |label: &str, cache: &ltxv::block::GenerationCache| -> f64 {
        let t1 = Instant::now();
        let (v, a) = ltxv::dit::av_forward_q_streamed_in(&session, &cfg, &src, &head, ltxv::block::QTier::Int8, &step, cache);
        let wall = t1.elapsed().as_secs_f64();
        let device_ms = session.is_resident().then(|| device_kernel_ms(&session.device_for_call())).flatten();
        report_call(label, layers, wall, device_ms, &prev_device);
        for (name, out) in [("video", &v), ("audio", &a)] {
            let n = out.len() as f64;
            let mean = out.iter().map(|&x| x as f64).sum::<f64>() / n;
            let var = out.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
            let (min, max) = out.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &x| (mn.min(x), mx.max(x)));
            let nonfinite = out.iter().filter(|x| !x.is_finite()).count();
            println!("[{label}] {name} OUTPUT STATS: len={} mean={mean:.6} std={:.6} min={min:.6} max={max:.6} nonfinite={nonfinite}", out.len(), var.sqrt());
        }
        let rs = session.stats();
        println!("[{label}] DEVICE RESIDENCY: slots={} device_hits={} device_uploads={}", rs.slots, rs.hits, rs.uploads);
        wall
    };
    let cache = ltxv::block::GenerationCache::default();
    call("call 1 (cache miss on every layer - a generation's first forward)", &cache);
    if reuse_cache {
        // See `bench_streamed`: the first warm call is the warm-up and is
        // excluded from the reported best-of-N.
        let mut warm = Vec::new();
        for r in 0..warm_reps.max(1) {
            warm.push(call(&format!("call {} (cache hit on every layer - every OTHER forward)", r + 2), &cache));
        }
        let best = warm.iter().skip(1).fold(f64::INFINITY, |a, &b| a.min(b));
        if best.is_finite() {
            println!("[warm] best of {} (first warm call excluded as warm-up): {best:.2} s", warm.len() - 1);
        }
    } else {
        call("call 2 (its OWN fresh cache)", &ltxv::block::GenerationCache::default());
    }
    println!("[av] host cache after both calls: {} MiB over {} blocks", cache.block_byte_len() / (1 << 20), cache.stats().blocks);
}

/// Peak resident set size of this process, from the kernel's own high-water
/// mark (`/proc/self/status`'s `VmHWM`), in MiB.
///
/// Host RSS is a first-class result for every mode below, not a footnote: what
/// separates the streamed int8 tiers from expanding a checkpoint to host fp32
/// is mostly a memory claim, and a run that got the timing it wanted while
/// quadrupling the host footprint has not made the path cheaper. Sampling it
/// externally needs a wrapper the caller has to remember (and a `wait4` that
/// this box has no `time(1)` for), so the harness reports its own high-water
/// mark and a run cannot forget to. `None` off Linux, where the file does not
/// exist - reported as unavailable rather than as zero.
fn peak_rss_mib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("dit");
    let arg = |i: usize, d: u32| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    match mode {
        "dit" => bench_dit(arg(2, 2) as usize, arg(3, 8), arg(4, 1024), arg(5, 256)),
        "vae" => bench_vae(arg(2, 2) as usize, arg(3, 17), arg(4, 384), arg(5, 384)),
        "vocoder" => bench_vocoder(arg(2, 3) as usize, arg(3, 100)),
        "av" => bench_av(arg(2, 880), arg(3, 256), arg(4, 25), arg(5, 24)),
        "streamed" => bench_streamed(arg(2, 4), arg(3, 512), arg(4, 256), a.get(5).map(|s| s == "1").unwrap_or(false), a.get(6).map(|s| s == "1").unwrap_or(false), arg(7, 1), arg(8, 1)),
        "streamed-av" => bench_streamed_av(arg(2, 4), arg(3, 512), arg(4, 256), arg(5, 128), a.get(6).map(|s| s == "1").unwrap_or(false), a.get(7).map(|s| s == "1").unwrap_or(false), arg(8, 1), arg(9, 1)),
        "decode" => {
            let path = a.get(2).map(|s| s.as_str()).unwrap_or_else(|| panic!("usage: ltxv_bench decode <latent.bin> <whole|tiled> [h0 h1 w0 w1]"));
            let crop = (a.len() >= 8).then(|| (arg(4, 0), arg(5, 0), arg(6, 0), arg(7, 0)));
            bench_decode(path, a.get(3).map(|s| s.as_str()).unwrap_or("tiled"), crop);
        }
        other => {
            eprintln!("unknown mode {other} (dit|vae|vocoder|av|streamed|streamed-av|decode)");
            std::process::exit(1);
        }
    }
    match peak_rss_mib() {
        Some(mib) => println!("[{mode}] host peak RSS: {mib} MiB"),
        None => println!("[{mode}] host peak RSS: unavailable on this platform"),
    }
}
