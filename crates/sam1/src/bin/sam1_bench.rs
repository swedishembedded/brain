// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM-1 ViT-B tower profiler: where a forward's time goes, per kernel kind,
//! and how the CPU and wgpu backends compare at the real DeepSeek-OCR shape.
//!
//! Profile per kernel kind before touching anything, and publish the table.
//!
//! **Weight-free.** The tower's cost depends only on shape, so this drives
//! `sam1::init_dense` (random weights at the real geometry) instead of the
//! ~450 MB mmproj checkpoint -- seconds to build, correctness is not the
//! question this binary answers (the wgpu backend's known corruption,
//! `crates/sam1/tests/wgpu_block_count_corruption.rs`, is a VALUES bug, not a
//! hang, so its forward's wall time is still representative for comparison
//! purposes even though its output cannot be trusted).
//!
//! **Best-of-N wall clock**, not device kernel-time-summed: `SamEncoder::forward`
//! submits per-stage (patch, one submit per block, neck) rather than exposing
//! one flat `Vec<Step>`, so `gpu_core::profile::profile`'s single-submit path
//! does not apply here. `Gpu::kernel_times()` (`set_kernel_timing` +
//! `reset_kernel_times`) still gives the accumulated per-kernel-kind DEVICE
//! time across every submit in one forward -- the same accounting
//! `BRAIN_PROFILE`'s table uses -- so that is the per-kernel breakdown source.
//!
//! **Windowed vs global, isolated by a one-block tower.** A block's own
//! per-forward cost cannot be read off the 12-block real config directly (patch
//! embed / neck / compressor cost is shared, and blocks 0..11 differ only in
//! whether they are windowed or global). Building two `n_layers=1` configs that
//! are identical except `global_attn_layers` (mirroring
//! `wgpu_block_count_corruption.rs`'s own `cfg(n_layers)` helper) isolates it:
//! everything outside the one block is IDENTICAL between the two runs, so the
//! wall-clock delta is that one block's own windowed-vs-global cost, not a
//! guess from kernel names alone (the same kernels run at both spans, just
//! different `T`).
//!
//! Usage: `sam1_bench [reps]` (default 5). Runs CPU always; adds wgpu if
//! `MOE_SKIP_GPU_TESTS` is unset.

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::Gpu;
use sam1::{init_dense, SamEncoder, SamViTConfig, PIPELINES};

fn synthetic_image(cfg: &SamViTConfig) -> Vec<f32> {
    let n = (3 * cfg.image_h() * cfg.image_w()) as usize;
    // A fixed, non-degenerate pattern -- shape/timing does not depend on the
    // pixel values, so this need not be a real photo, only finite and varied
    // enough that no kernel takes a zero/NaN fast path.
    (0..n).map(|i| ((i % 251) as f32 / 251.0) - 0.5).collect()
}

/// Best-of-`reps` wall-clock forward, after one untimed warmup.
fn best_forward(enc: &SamEncoder, reps: usize) -> f64 {
    let _ = enc.forward();
    enc.gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let obj = enc.forward();
        enc.gpu.poll_wait();
        let dt = t0.elapsed().as_secs_f64();
        assert!(obj.is_finite(), "forward produced a non-finite objective");
        best = best.min(dt);
    }
    best
}

/// Per-kernel-kind device time table for one forward, via `kernel_times()` --
/// the same accumulator `BRAIN_PROFILE` reads, retrieved as data rather than
/// printed at drop.
fn kernel_table(enc: &SamEncoder) -> Vec<(String, f64, u64)> {
    let timed = enc.gpu.set_kernel_timing(true);
    enc.gpu.reset_kernel_times();
    let _ = enc.forward();
    enc.gpu.poll_wait();
    if !timed {
        return Vec::new();
    }
    let mut rows = enc.gpu.kernel_times().unwrap_or_default();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    rows
}

fn print_kernel_table(label: &str, rows: &[(String, f64, u64)]) {
    if rows.is_empty() {
        println!("{label}: this backend does not report per-kernel device time");
        return;
    }
    let total: f64 = rows.iter().map(|r| r.1).sum();
    println!("{label} (sum of per-kernel device time {total:.1} ms):");
    for (name, ms, calls) in rows.iter().take(12) {
        println!("  {name:<20} {ms:>9.1} ms  {calls:>6} calls  ({:>4.1}%)", 100.0 * ms / total);
    }
}

/// `SamViTConfig::deepseek_ocr()` truncated to one block -- mirrors
/// `wgpu_block_count_corruption.rs`'s `cfg(n_layers)` helper, but keeping
/// exactly one block so `global_attn_layers` alone decides windowed vs global.
fn one_block_cfg(global: bool) -> SamViTConfig {
    let base = SamViTConfig::deepseek_ocr();
    SamViTConfig { n_layers: 1, global_attn_layers: if global { vec![0] } else { vec![] }, ..base }
}

fn bench_backend(label: &str, gpu: Gpu, reps: usize) {
    let cfg = SamViTConfig::deepseek_ocr();
    let init: HashMap<String, Vec<f32>> = init_dense(&cfg, 7);
    let image = synthetic_image(&cfg);

    println!("\n=== {label}: full 12-block tower, 1024x1024, d_model=768 ===");
    let t_build = Instant::now();
    let enc = SamEncoder::new_on(gpu.share(), cfg.clone(), &init, 7, false);
    enc.write_image(&image);
    println!("built in {:.2}s", t_build.elapsed().as_secs_f64());

    let best = best_forward(&enc, reps);
    println!("best of {reps}: {:.1} ms/forward", best * 1e3);

    let rows = kernel_table(&enc);
    print_kernel_table(label, &rows);
    drop(enc);

    // Windowed vs global, isolated -- see this file's header.
    let win_init = init_dense(&one_block_cfg(false), 7);
    let win = SamEncoder::new_on(gpu.share(), one_block_cfg(false), &win_init, 7, false);
    win.write_image(&image);
    let t_win = best_forward(&win, reps);
    drop(win);

    let glob_init = init_dense(&one_block_cfg(true), 7);
    let glob = SamEncoder::new_on(gpu.share(), one_block_cfg(true), &glob_init, 7, false);
    glob.write_image(&image);
    let t_glob = best_forward(&glob, reps);
    drop(glob);

    println!(
        "one-block tower (patch+neck+compressor shared, only the block differs): \
         windowed (14x14 spans, {} windows tiling the 64x64 grid) {:.1} ms, \
         global (one 4096-row span) {:.1} ms ({:.1}x)",
        (64u32.div_ceil(14)) * (64u32.div_ceil(14)),
        t_win * 1e3,
        t_glob * 1e3,
        t_glob / t_win.max(1e-9)
    );
}

/// `sam1_bench profile` -- one full-tower forward on the CPU backend, with
/// `BRAIN_PROFILE`'s own per-kernel wall-time accumulator (the CPU backend has
/// no device-timestamp path, so `Gpu::kernel_times()` -- this file's
/// `kernel_table` -- reports nothing there; `dump_profile()` is the CPU
/// backend's own accounting, gated the same way). Fast (one forward, no
/// one-block/wgpu legs) so it fits comfortably inside one blocking call.
fn profile_mode() {
    // Must be set before `CpuBackend::new` builds (it reads the env once at
    // construction), so set it here rather than relying on the caller's shell.
    std::env::set_var("BRAIN_PROFILE", "1");
    let cfg = SamViTConfig::deepseek_ocr();
    let init: HashMap<String, Vec<f32>> = init_dense(&cfg, 7);
    let image = synthetic_image(&cfg);
    let gpu = Gpu::new_cpu(PIPELINES);
    let t_build = Instant::now();
    let enc = SamEncoder::new_on(gpu.share(), cfg, &init, 7, false);
    enc.write_image(&image);
    println!("built in {:.2}s", t_build.elapsed().as_secs_f64());
    // Best-of-3 wall clock: this machine may run other CPU-heavy work
    // concurrently (a separate agent's builds/tests in another worktree), so a
    // single sample can be contamined by contention the same way §F.1 warns a
    // per-group drain contaminates a kernel table. The per-kernel table below
    // is still just the LAST forward's accumulation (BRAIN_PROFILE sums across
    // every forward since the backend was built, so this is really "3 forwards
    // worth", divide by 3 for a per-forward per-kernel estimate).
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let obj = enc.forward();
        let dt = t0.elapsed().as_secs_f64();
        assert!(obj.is_finite());
        println!("forward: {:.1} ms", dt * 1e3);
        best = best.min(dt);
    }
    println!("best of 3: {:.1} ms/forward", best * 1e3);
    enc.gpu.dump_profile();
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("profile") {
        profile_mode();
        return;
    }
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5);

    bench_backend("CPU", Gpu::new_cpu(PIPELINES), reps);

    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        println!("\nMOE_SKIP_GPU_TESTS set -- skipping the wgpu comparison.");
        return;
    }
    // wgpu is known to CORRUPT this tower's output at 3+ blocks
    // (crates/sam1/tests/wgpu_block_count_corruption.rs) -- not shipped, not
    // used for anything but this timing reference. This model is served on the
    // CPU backend regardless of what this number shows.
    bench_backend("wgpu (NOT shipped -- known output corruption, timing reference only)", Gpu::new_wgpu(PIPELINES), reps);
}
