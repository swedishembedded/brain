// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax Music 3 profiler: where a denoise step, a vocoder chunk and an AR
//! frame actually spend their time, per kernel kind.
//!
//! This is the INSTRUMENT, not an optimisation. Every confident hypothesis
//! about this engine's hot spots has so far been killed by a measurement, so
//! the first move on anything slow is to profile per KERNEL KIND and publish
//! the table. Nothing in this crate had one, so every statement about where
//! MiniMax Music 3's time goes was arithmetic or inference. This binary makes
//! it a measurement.
//!
//! Method, and why each part is there:
//!
//! * **Per-kernel DEVICE time, not host-bracketed slices.** Host wall clock
//!   around a drained slice measures launch + execute + fence, whose floor is
//!   roughly constant and therefore inflates small kernels in inverse
//!   proportion to their size - up to 29x measured elsewhere in this tree,
//!   enough to invert a ranking. `gpu_core::profile::profile_live` reads
//!   `Gpu::kernel_times()` instead: real timestamps written inside the
//!   production passes.
//! * **FLOP/byte volume from `gpu_core::cost`**, via the online
//!   `Gpu::ops_counters()` ledger, so a rate reflects exactly the kernel
//!   variants that were dispatched. No local matmul model: this crate's DiT
//!   dispatches conv, layernorm, rope, three attention kernels and a swiglu
//!   besides its GEMMs, and `2*m*k*n` is silently wrong for every one of them.
//! * **Utilisation against the DEVICE'S OWN measured roofline**
//!   (`gpu_core::roof`), never a hardcoded P40 peak - the same table has to
//!   mean something on whatever card `--device` selected.
//! * **Warm-up never enters the statistics** and the reported number is
//!   **best-of-N**, not a single sample or a mean: the minimum is the least
//!   contaminated sample.
//! * **Every timed region is `poll_wait()`-bracketed.** A `submit` with an
//!   empty clear list only appends to the pending list; an unbracketed loop
//!   times host-side recording and reports it as device throughput. That
//!   mistake once produced a bandwidth figure above the card's physical roof.
//!
//! ## Why `profile_live` and not `gpu_core::profile::profile`
//!
//! `profile` re-submits one flat `&[Step]`. Neither of this crate's device
//! stages hands one out: `dit::forward_resident` submits per sub-layer and
//! reads a buffer back to the HOST in the middle of every block, and
//! `vocoder::forward` records its tape privately and submits it itself. The
//! live variant accumulates the identical two ledgers across whatever submits
//! the closure makes, so the table is the same table.
//!
//! One consequence to read carefully: the whole-pass number here is WALL CLOCK
//! around the closure, so it also contains host math and host<->device
//! readbacks. `total - sum of kernel device time` is therefore NOT just launch
//! and sync for these passes. That gap is itself a finding, not noise.
//!
//! ## Weights
//!
//! Random by default (`--random` is the default; `dit_train::random_weights`
//! and friends), because a dispatch's cost is a function of its SHAPE, not of
//! the values in its buffers - the same argument every other weight-free bench
//! in this tree makes. Set `BRAIN_MINIMAXMUSIC3_DIT` / `_VOCODER` / `_DEPTH` to
//! profile the real checkpoint instead. Which was used is printed in the header
//! of every run, because "random weights" is a claim the reader must be able to
//! check rather than assume.
//!
//! ## The depth decoder is HOST math
//!
//! `depth_decoder` is `model::hostmath` from end to end - no `Step`, no
//! dispatch, nothing for a GPU timestamp to attach to. Reporting a device table
//! for it would be a fiction, so the `depth` mode reports wall clock per call
//! class plus an analytic FLOP/byte model, explicitly labelled as host, and
//! never divides by the device roofline.
//!
//! Usage:
//!   mm3_bench [--device <cpu|gpu|gpu0|gpu1>] <mode> [args]
//!
//!   mm3_bench dit     [length] [reps]   one `dit::forward_resident`, DitConfig::real()
//!   mm3_bench vocoder [length] [reps]   one `vocoder::forward`, VocoderConfig::real()
//!   mm3_bench depth   [frames] [reps]   the RVQ depth decoder's per-frame AR loop (HOST)
//!   mm3_bench gemm    [reps]            A/B the GEMM kernels at the DiT's shapes, correctness AND speed
//!   mm3_bench all     [reps]            all three at real chunk dims
//!
//! `length` is in LATENT frames. A full denoise chunk is `CHUNK_FRAMES = 200`
//! AR frames, which the condition encoder resamples to 689 latent frames - the
//! default, and the only length whose numbers describe a real generation. Pass
//! a small one (e.g. `mm3_bench dit 64`) for quick iteration on the harness
//! itself; the shares will not be the real shares.

use std::time::Instant;

use gpu_core::roof::Roofs;
use gpu_core::Gpu;

use minimaxmusic3::config::{DepthDecoderConfig, DitConfig, VocoderConfig};
use minimaxmusic3::{depth_decoder, dit, dit_train, train, vocoder};

/// Latent frames in one full `denoise::CHUNK_FRAMES` (200 AR frame) chunk -
/// what `condition_encoder::latent_length` yields, and what
/// `tests/vocoder_real_chunk.rs` pins as this model's real chunk shape.
const REAL_CHUNK_LATENTS: usize = 689;

/// Where the source of a component's weights is recorded, so the header can
/// state it rather than leave the reader to assume.
enum Source {
    Random,
    Checkpoint(String),
}

impl Source {
    fn describe(&self) -> String {
        match self {
            Source::Random => "RANDOM weights (shape-correct; cost is a function of shape, not values)".to_string(),
            Source::Checkpoint(d) => format!("REAL checkpoint at {d}"),
        }
    }
}

/// The directory `var` points at, if it is set AND exists - otherwise `None`,
/// so an unset or stale env var degrades to random weights with a printed
/// reason rather than a panic halfway through a long run.
fn checkpoint_dir(var: &str) -> Option<String> {
    let dir = std::env::var(var).ok()?;
    if std::path::Path::new(&dir).exists() {
        Some(dir)
    } else {
        eprintln!("{var}={dir} does not exist - falling back to random weights");
        None
    }
}

/// The device header: which backend actually ran, and its own measured
/// roofline. Both are printed before any table because every utilisation
/// number below divides by the second, and rule 1 of this harness is that a
/// CPU-JIT run must not be readable as a GPU run.
fn device_header(gpu: &Gpu) -> Option<Roofs> {
    let kind = gpu.kind();
    println!("backend: {kind}   ({} dispatch-timing)", if gpu.set_kernel_timing(false) { "device-timestamp" } else { "NO" });
    // `adapter_info`'s bool is `is_software`, NOT `is_discrete`: a box with no
    // real card still serves `--device gpu` through a software rasteriser, and
    // a timing number is uninterpretable without knowing which it was.
    if let Some((name, software)) = gpu_core::adapter_info() {
        println!("adapter: {name}{}", if software { "  [SOFTWARE RASTERISER - not a real card]" } else { "" });
    }
    if kind == "cpu" {
        println!(
            "WARNING: this is the CPU JIT, not a GPU. Its numbers are NOT a GPU baseline and are \
             not labelled as one - they are here so the harness can be exercised without a card."
        );
    }
    let roofs = gpu_core::roof::ensure(gpu);
    match roofs {
        Some(r) => println!(
            "measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte",
            r.gflops, r.gbs, r.cache_gbs, r.ridge()
        ),
        None => println!("roofline unmeasured on this device - utilisation columns print '-' rather than a guess"),
    }
    roofs
}

/// Print one live profile, plus the §E.2 defect lines under it.
fn report(p: &gpu_core::profile::PassProfile, roofs: Option<Roofs>) {
    if !p.device_timed {
        println!(
            "\n=== {}: {} dispatches, {:.2} ms (best of N, wall clock) ===",
            p.label,
            p.dispatches,
            p.total_secs * 1e3
        );
        println!(
            "This backend cannot write timestamps inside a pass, so there is NO per-kernel time \
             to report and none is invented. Dispatch counts and FLOP/byte volume follow; the \
             time column would be a fabrication."
        );
        for r in &p.rows {
            println!(
                "  {:<26} {:>6} calls  {:>12.3} GFLOP  {:>12.3} GB{}",
                r.name,
                r.calls,
                r.flops.max(r.int_ops) as f64 / 1e9,
                r.bytes as f64 / 1e9,
                if r.covered { "" } else { "   (no cost formula)" },
            );
        }
        return;
    }
    p.print(roofs);
    println!(
        "NB: the whole-pass number is WALL CLOCK around the closure, so `total - sum of kernel \
         device time` = {:.2} ms is host math, readbacks and launch gaps TOGETHER on this pass, \
         not launch alone.",
        (p.total_secs - p.summed_secs) * 1e3
    );
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<24} {:>5.1}% of its {} roof (floor {:.0}%) - {:.1}% of this pass's device time",
                row.name,
                pct,
                bound.as_str(),
                bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// dit
// ---------------------------------------------------------------------------

/// One `dit::forward_resident` at `DitConfig::real()`, profiled per kernel kind.
///
/// This is the model's cost centre: `denoise::denoise_chunk` evaluates the DiT
/// TWICE per Euler step (the conditional and the zero-condition CFG branch), so
/// this one number multiplied by `2 * steps * chunks` is most of a generation.
fn dit_mode(device: Option<&str>, length: usize, reps: usize) {
    let cfg = DitConfig::real();
    let rows = length + 1;
    println!(
        "\n########## DiT - {} layers, inner {}, {} heads x {}, ff_inner {}, rotary {} ##########",
        cfg.num_layers, cfg.inner_dim(), cfg.num_attention_heads, cfg.attention_head_dim, cfg.ff_inner_dim, cfg.rotary_dim,
    );
    println!("chunk: {length} latent frames -> {rows} transformer rows (the prepended Fourier-timestep token)");

    let (w, src) = match checkpoint_dir("BRAIN_MINIMAXMUSIC3_DIT") {
        Some(dir) => match dit::import(&dir, &cfg) {
            Ok(w) => (w, Source::Checkpoint(dir)),
            Err(e) => {
                eprintln!("dit import failed ({e}) - falling back to random weights");
                (dit_train::random_weights(&cfg, 0xD17), Source::Random)
            }
        },
        None => (dit_train::random_weights(&cfg, 0xD17), Source::Random),
    };
    println!("weights: {}", src.describe());

    let gpu = Gpu::open(device, dit::PIPELINES);
    let roofs = device_header(&gpu);

    // The weight upload is a per-CHUNK cost, not a per-evaluation one (that is
    // exactly what `Resident` exists to hoist), so it is timed and reported
    // separately rather than folded into the forward.
    let t0 = Instant::now();
    let res = dit::Resident::new(&gpu, &cfg, &w, length);
    gpu.poll_wait();
    let upload = t0.elapsed().as_secs_f64();
    let weight_bytes: usize = 4 * w.blocks.iter().map(|b| {
        b.attn.wq.len() + b.attn.wk.len() + b.attn.wv.len() + b.attn.wo.len() + b.ff_in_w.len() + b.ff_out_w.len() + b.norm1_w.len() * 2 + b.norm2_w.len() * 2
    }).sum::<usize>();
    println!(
        "Resident::new (once per chunk): {:.2} s for {:.2} GB of block weights ({:.2} GB/s)",
        upload,
        weight_bytes as f64 / 1e9,
        weight_bytes as f64 / upload / 1e9,
    );

    let mut r = data::rng::Lcg::new(0xBEEF);
    let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
    let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);

    let p = gpu_core::profile::profile_live(&gpu, "DIT FORWARD (one CFG branch, one Euler step)", reps, || {
        let out = dit::forward_resident(&gpu, &cfg, &w, &res, &latents, &condition, 0.5, length);
        debug_assert_eq!(out.len(), cfg.in_channels as usize * length);
    });
    report(&p, roofs);
    println!(
        "\none forward: {:.2} ms (best of {reps}).  A denoise chunk = 2 CFG branches x N Euler \
         steps, so a 30-step chunk is {:.1} s of DiT alone, plus {:.2} s of Resident upload.",
        p.total_secs * 1e3,
        60.0 * p.total_secs,
        upload,
    );
}

// ---------------------------------------------------------------------------
// vocoder
// ---------------------------------------------------------------------------

fn vocoder_mode(device: Option<&str>, length: usize, reps: usize) {
    let cfg = VocoderConfig::real();
    let upsample: usize = cfg.upsampling_ratios.iter().product::<u32>() as usize;
    println!(
        "\n########## Vocoder - latent {} ch, in {}, hidden {}, ratios {:?} ({}x), {} Hz ##########",
        cfg.latent_channels, cfg.decoder_input_dim, cfg.decoder_hidden_dim, cfg.upsampling_ratios, upsample, cfg.sampling_rate,
    );
    println!("chunk: {length} latents -> {} samples/channel ({:.2} s of stereo audio)", length * upsample, (length * upsample) as f64 / f64::from(cfg.sampling_rate));

    let (w, src) = match checkpoint_dir("BRAIN_MINIMAXMUSIC3_VOCODER") {
        Some(dir) => match vocoder::import(&dir, &cfg) {
            Ok(w) => (w, Source::Checkpoint(dir)),
            Err(e) => {
                eprintln!("vocoder import failed ({e}) - falling back to random weights");
                (train::random_weights(&cfg, 0x0C), Source::Random)
            }
        },
        None => (train::random_weights(&cfg, 0x0C), Source::Random),
    };
    println!("weights: {}", src.describe());

    let gpu = Gpu::open(device, vocoder::PIPELINES);
    let roofs = device_header(&gpu);

    let mut r = data::rng::Lcg::new(0xC0DE);
    let latents = r.vec_scaled(cfg.latent_channels as usize * length, 0.1);

    let p = gpu_core::profile::profile_live(&gpu, "VOCODER FORWARD (one chunk)", reps, || {
        let out = vocoder::forward(&gpu, &cfg, &w, &latents, 1, length);
        debug_assert_eq!(out.len(), 2 * length * upsample);
    });
    report(&p, roofs);
    println!(
        "\none chunk: {:.2} ms (best of {reps}) for {:.2} s of audio -> {:.2}x realtime.",
        p.total_secs * 1e3,
        (length * upsample) as f64 / f64::from(cfg.sampling_rate),
        (length * upsample) as f64 / f64::from(cfg.sampling_rate) / p.total_secs,
    );
}

// ---------------------------------------------------------------------------
// gemm A/B
// ---------------------------------------------------------------------------

/// A/B the GEMM kernels at the DiT's OWN shapes, for CORRECTNESS and speed.
///
/// A faster kernel that disagrees is not a faster kernel, so `max|delta|`
/// against a HOST f64 oracle is printed beside every timing - kernel-to-kernel
/// agreement cannot tell you which one is wrong, and one A/B elsewhere in this
/// tree reported all three candidates disagreeing when the *harness* was
/// dispatching a `@workgroup_size(256)` kernel at 64 threads.
///
/// `dit::linear_step` selects through `model::block::gemm_variant`, so what is
/// swept here is also what the selection rule is choosing between. The shapes
/// are the six linears every one of the 36 blocks dispatches.
fn gemm_mode(device: Option<&str>, reps: usize) {
    let kernels: &[(&str, &str)] = &[
        ("matmul", kernels::MATMUL),
        ("matmul_reg3", kernels::MATMUL_REG3),
        ("matmul_gemv", kernels::MATMUL_GEMV),
    ];
    let gpu = Gpu::open(device, kernels);
    let _ = device_header(&gpu);
    let coop = gpu.caps().workgroup_reductions;
    if !coop {
        println!("(this device reports workgroup_reductions=false, so `dit::linear_step` keeps the reference `matmul`)");
    }

    // Correctness first, at a shape small enough for an f64 host oracle.
    {
        // `m <= 32` so `matmul_gemv` is in the comparison at all (see
        // [`gemm_candidates`]); the real shapes below are far past that and
        // exercise only the two that can run there.
        let (m, k, n) = (17u32, 61u32, 23u32);
        let mut r = data::rng::Lcg::new(7);
        let x = r.vec_scaled((m * k) as usize, 1.0);
        let w = r.vec_scaled((n * k) as usize, 1.0);
        let mut want = vec![0.0f64; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut acc = 0.0f64;
                for t in 0..k as usize {
                    acc += f64::from(x[i * k as usize + t]) * f64::from(w[j * k as usize + t]);
                }
                want[i * n as usize + j] = acc;
            }
        }
        let xb = gpu.storage(u64::from(m * k));
        let wb = gpu.storage(u64::from(n * k));
        gpu.write_f32(&xb, &x);
        gpu.write_f32(&wb, &w);
        for (name, threads) in gemm_candidates(m, n) {
            let Some(ki) = gpu.kernel_index(name) else { continue };
            let out = gpu.storage(u64::from(m * n));
            gpu.submit(&[], &[gpu.step(ki, &[&xb, &wb, &out], &[m, k, n], threads)]);
            gpu.poll_wait();
            let got = gpu.read(&out, (m * n) as usize);
            let err = got.iter().zip(&want).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0f64, f64::max);
            println!("  oracle [{m},{k},{n}]  {name:<16} max|delta| {err:.3e}");
            // A real gate, not a printout: benchmarking a kernel that computes
            // the wrong thing is worse than not benchmarking it.
            assert!(err < 1e-3, "{name} diverges from the f64 host oracle at [{m},{k},{n}]: {err:.3e}");
        }
    }

    let cfg = DitConfig::real();
    let (inner, ff) = (cfg.inner_dim(), cfg.ff_inner_dim);
    let rows = REAL_CHUNK_LATENTS as u32 + 1;
    println!("\n{:<34} {:>10} {:>12} {:>9} {:>10}", "shape [m, k, n]  (site)", "ms", "GFLOP/s", "%roof", "kernel");
    let roofs = gpu_core::roof::ensure(&gpu);
    for (m, k, n, site) in [
        (rows, inner, inner, "attn to_q/k/v/out (x4/block)"),
        (rows, inner, 2 * ff, "ff_in, fused"),
        (rows, inner, ff, "ff_in, one half"),
        (rows, ff, inner, "ff_out"),
    ] {
        let xb = gpu.storage(u64::from(m) * u64::from(k));
        let wb = gpu.storage(u64::from(n) * u64::from(k));
        let flops = 2.0 * f64::from(m) * f64::from(k) * f64::from(n);
        for (name, threads) in gemm_candidates(m, n) {
            let Some(ki) = gpu.kernel_index(name) else { continue };
            let out = gpu.storage(u64::from(m) * u64::from(n));
            let st = [gpu.step(ki, &[&xb, &wb, &out], &[m, k, n], threads)];
            let t = gpu_core::profile::best_of(&gpu, &st, reps);
            let pct = roofs
                .map(|r| format!("{:.1}%", 100.0 * (flops / t) as f32 / (r.gflops * 1e9)))
                .unwrap_or_else(|| "-".into());
            println!(
                "[{m:>4},{k:>6},{n:>6}]  {site:<24} {:>10.3} {:>12.1} {:>9} {name}",
                t * 1e3,
                flops / t / 1e9,
                pct,
            );
        }
    }
}

/// The GEMM candidates and their dispatch geometry - thread counts copied from
/// `model::block::gemm_variant`, never guessed, because dispatching a
/// register-tiled 256-thread kernel at one thread per output silently computes
/// a fraction of the answer.
///
/// `matmul_gemv` is offered only at `m <= 32`: its accumulator is a
/// `array<f32, 2048>` workgroup array indexed `[m*64 + t]`, so its own header
/// states `REQUIRES m <= 32`. Every DiT linear runs at `m = rows` (690 at a
/// real chunk), far outside that, which is exactly why `gemm_variant`'s
/// decode-regime branch never fires here.
fn gemm_candidates(m: u32, n: u32) -> Vec<(&'static str, u32)> {
    let mut c = vec![("matmul", m * n), ("matmul_reg3", m.div_ceil(128) * n.div_ceil(128) * 256)];
    if m <= 32 {
        c.push(("matmul_gemv", n * 64));
    }
    c
}

// ---------------------------------------------------------------------------
// depth decoder - HOST math
// ---------------------------------------------------------------------------

/// Analytic FLOPs and streamed bytes of one `depth_decoder::step` at position
/// `pos` (0-based), from the shapes in `depth_decoder`'s own code.
///
/// Hand-written because `gpu_core::cost` costs DISPATCHES, and this component
/// issues none: there is no `Step` to look up. Same conventions as that module
/// so the two numbers are comparable - 1 MAC = 2 ops, transcendentals count 1,
/// bytes are streaming traffic at 4 B/element.
fn depth_step_cost(cfg: &DepthDecoderConfig, pos: usize) -> (u64, u64) {
    let d = u64::from(cfg.hidden_size);
    let inter = u64::from(cfg.intermediate_size);
    let layers = u64::from(cfg.num_layers);
    let s = pos as u64 + 1; // keys/values attended, this position included

    // Per layer: q,k,v,o are [d,d]; gate,up,down are [inter,d]/[d,inter].
    let proj = 4 * 2 * d * d + 3 * 2 * d * inter;
    // Attention over `s` keys: scores and apply are each 2 ops per (key, dim),
    // summed over all heads, and heads*head_dim == d.
    let attn = 2 * 2 * s * d;
    // Two RMSNorms and a SwiGLU, counted the way `cost` counts them.
    let elementwise = 2 * 6 * d + 3 * inter;
    let flops = layers * (proj + attn + elementwise) + 6 * d;

    // The weights are the traffic: a GEMV touches every weight exactly once and
    // the activations are negligible beside them.
    let bytes = layers * 4 * (4 * d * d + 3 * d * inter);
    (flops, bytes)
}

/// The depth decoder's real per-frame AR loop, on the HOST.
///
/// Mirrors `pipeline::generate_depth_codes`: one `KvCache` per CFG branch, a
/// seed `step` on each from the Global LLM's hidden state, then
/// `num_codebooks - 1` rounds of (step x2, audio_head x2, embedding lookup +
/// projection). That is 16 `step` calls per frame at `num_codebooks = 8` - the
/// number the roadmap's budget arithmetic is about, now measured.
fn depth_mode(frames: usize, reps: usize) {
    let cfg = DepthDecoderConfig::real();
    println!(
        "\n########## RVQ depth decoder - hidden {}, inter {}, {} layers, {} heads, {} codebooks ##########",
        cfg.hidden_size, cfg.intermediate_size, cfg.num_layers, cfg.num_attention_heads, cfg.num_codebooks,
    );
    println!(
        "THIS COMPONENT IS HOST MATH (`model::hostmath`, AVX2+rayon over {} cores). It issues NO \n\
         device dispatch, so there is no GPU timestamp to attach to and no GPU table is printed \n\
         for it - wall clock and an analytic FLOP/byte model are the honest instruments here.",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    );

    let (w, src) = match checkpoint_dir("BRAIN_MINIMAXMUSIC3_DEPTH") {
        Some(dir) => match depth_decoder::import(&dir, &cfg) {
            Ok(w) => (w, Source::Checkpoint(dir)),
            Err(e) => {
                eprintln!("depth import failed ({e}) - falling back to random weights");
                (depth_decoder::random_weights(&cfg, 0xDD), Source::Random)
            }
        },
        None => (depth_decoder::random_weights(&cfg, 0xDD), Source::Random),
    };
    println!("weights: {}", src.describe());

    let d = cfg.hidden_size as usize;
    let books = cfg.num_codebooks as usize;
    let vocab = cfg.audio_vocab_size as usize;
    let mut r = data::rng::Lcg::new(0xDEF7);
    let hidden_cond = r.vec_scaled(d, 0.2);
    let hidden_uncond = r.vec_scaled(d, 0.2);

    // Wall clock per CALL CLASS - the host analogue of a per-kernel-kind table.
    // Warm-up is a full extra frame and never enters the statistics.
    let one_frame = |acc: &mut [f64; 4]| {
        let mut cache_c = depth_decoder::KvCache::new(&cfg);
        let mut cache_u = depth_decoder::KvCache::new(&cfg);
        let t = Instant::now();
        let pc = depth_decoder::projection(&w, &cfg, &hidden_cond);
        let pu = depth_decoder::projection(&w, &cfg, &hidden_uncond);
        acc[0] += t.elapsed().as_secs_f64();
        let t = Instant::now();
        depth_decoder::step(&w, &cfg, &mut cache_c, &pc);
        depth_decoder::step(&w, &cfg, &mut cache_u, &pu);
        acc[1] += t.elapsed().as_secs_f64();

        let mut row = pc.clone();
        for index in 1..books {
            let t = Instant::now();
            let h_c = depth_decoder::step(&w, &cfg, &mut cache_c, &row);
            let h_u = depth_decoder::step(&w, &cfg, &mut cache_u, &row);
            acc[1] += t.elapsed().as_secs_f64();

            // Both branches' heads run in the real loop (the logits are
            // CFG-blended), so both run here.
            let t = Instant::now();
            let l_c = depth_decoder::audio_head(&w, &cfg, index - 1, &h_c);
            let l_u = depth_decoder::audio_head(&w, &cfg, index - 1, &h_u);
            acc[2] += t.elapsed().as_secs_f64();
            std::hint::black_box(&l_u);
            // The real loop samples from the CFG-blended logits; any code in
            // range costs the same lookup, and sampling is not what is being
            // measured, so take a deterministic one.
            let code = l_c.len() % vocab;

            if index < books - 1 {
                let t = Instant::now();
                let e = depth_decoder::audio_embedding_row(&w, &cfg, code + (index - 1) * vocab);
                row = depth_decoder::projection(&w, &cfg, &e);
                acc[3] += t.elapsed().as_secs_f64();
            }
        }
    };

    let mut warm = [0.0f64; 4];
    one_frame(&mut warm); // never counted

    let mut best = f64::INFINITY;
    let mut best_acc = [0.0f64; 4];
    for _ in 0..reps.max(1) {
        let mut acc = [0.0f64; 4];
        let t0 = Instant::now();
        for _ in 0..frames {
            one_frame(&mut acc);
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
            best_acc = acc;
        }
    }

    // Analytic volume for one frame: 2 seed steps at position 0, then
    // 2*(books-1) steps walking positions 1..books-1.
    let (mut flops, mut bytes) = (0u64, 0u64);
    for _ in 0..2 {
        let (f, b) = depth_step_cost(&cfg, 0);
        flops += f;
        bytes += b;
    }
    for index in 1..books {
        let (f, b) = depth_step_cost(&cfg, index);
        flops += 2 * f;
        bytes += 2 * b;
    }
    let head = 2 * (books as u64 - 1) * (2 * u64::from(cfg.audio_vocab_size) * u64::from(cfg.hidden_size));
    let head_bytes = 2 * (books as u64 - 1) * 4 * u64::from(cfg.audio_vocab_size) * u64::from(cfg.hidden_size);
    let per_frame = best / frames as f64;

    println!("\n=== DEPTH DECODER, one AR frame (HOST, wall clock, best of {reps} over {frames} frames) ===");
    println!("{:<34} {:>10} {:>8} {:>10}", "host call class", "ms/frame", "%", "calls/frame");
    println!("{}", "-".repeat(70));
    let classes: [(&str, usize); 4] = [
        ("projection (seed x2)", 2),
        ("step (transformer block stack)", 2 + 2 * (books - 1)),
        ("audio_head", 2 * (books - 1)),
        ("embedding lookup + projection", books - 2),
    ];
    for ((name, n), secs) in classes.iter().zip(&best_acc) {
        println!(
            "{:<34} {:>10.2} {:>7.1}% {:>10}",
            name,
            1e3 * secs / frames as f64,
            100.0 * secs / best,
            n,
        );
    }
    println!("{}", "-".repeat(70));
    println!(
        "{:<34} {:>10.2}   ({:.2} GFLOP, {:.2} GB of weights streamed -> {:.1} GFLOP/s, {:.1} GB/s)",
        "WHOLE FRAME",
        1e3 * per_frame,
        (flops + head) as f64 / 1e9,
        (bytes + head_bytes) as f64 / 1e9,
        (flops + head) as f64 / per_frame / 1e9,
        (bytes + head_bytes) as f64 / per_frame / 1e9,
    );
    println!(
        "No device roofline is applied: these are HOST rates against host DRAM, and grading them \
         against a GPU's roof would be meaningless."
    );
    println!(
        "Extrapolated: one 200-frame denoise chunk = {:.1} s of depth decoding; a 4-minute track \
         (6000 AR frames) = {:.2} h.",
        200.0 * per_frame,
        6000.0 * per_frame / 3600.0,
    );
}

// ---------------------------------------------------------------------------

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut device: Option<String> = None;
    let mut pos: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--device" => {
                device = argv.get(i + 1).cloned();
                i += 2;
            }
            s if s.starts_with("--device=") => {
                device = Some(s["--device=".len()..].to_string());
                i += 1;
            }
            s => {
                pos.push(s.to_string());
                i += 1;
            }
        }
    }
    let dev = device.as_deref();
    let num = |i: usize, d: usize| pos.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);

    match pos.first().map(String::as_str) {
        Some("dit") => dit_mode(dev, num(1, REAL_CHUNK_LATENTS), num(2, 3)),
        Some("vocoder") => vocoder_mode(dev, num(1, REAL_CHUNK_LATENTS), num(2, 3)),
        // Frames default to 4, not 200: at real dims one frame is 16 GEMV
        // passes over ~2.3 GB of weights, so a whole chunk is minutes of host
        // math. The per-frame number is what extrapolates; the frame count only
        // has to be big enough to average.
        Some("depth") => depth_mode(num(1, 4), num(2, 3)),
        Some("gemm") => gemm_mode(dev, num(1, 5)),
        Some("all") => {
            let reps = num(1, 3);
            dit_mode(dev, REAL_CHUNK_LATENTS, reps);
            vocoder_mode(dev, REAL_CHUNK_LATENTS, reps);
            depth_mode(4, reps);
        }
        other => {
            if let Some(o) = other {
                eprintln!("unknown mode {o:?}");
            }
            eprintln!("usage: mm3_bench [--device <cpu|gpu|gpu0|gpu1>] <dit|vocoder|depth|gemm|all> [args]");
            eprintln!("  dit     [length={REAL_CHUNK_LATENTS}] [reps=3]   one dit::forward_resident at DitConfig::real()");
            eprintln!("  vocoder [length={REAL_CHUNK_LATENTS}] [reps=3]   one vocoder::forward at VocoderConfig::real()");
            eprintln!("  depth   [frames=4] [reps=3]     the RVQ depth decoder's per-frame AR loop (HOST math)");
            eprintln!("  gemm    [reps=5]                A/B the GEMM kernels at the DiT's own shapes, correctness AND speed");
            eprintln!("  all     [reps=3]");
            std::process::exit(2);
        }
    }
}
