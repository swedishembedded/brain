// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measures the non-ReBAR Pascal "2x resident per storage buffer" cost cited
//! by `docs/lessons.md` §14 and `crates/qwen/src/q8.rs` — directly, via
//! `nvidia-smi` memory deltas around known allocations, rather than by
//! inference from a model's total footprint.
//!
//! ```text
//! CARGO_HOME=/data/resources/cargo-home cargo test --release -p brain-gpu-core \
//!   -- --ignored --nocapture --test-threads=1 vram_overhead
//! ```
//!
//! One GPU device per process (`AGENTS.md`), so this is its own test binary
//! run with `--test-threads=1` and is `#[ignore]`d out of the fast lane, same
//! shape as `bench_matmul.rs`. Needs `nvidia-smi` on `$PATH` (skips cleanly
//! otherwise).
//!
//! **Result** (P40 ×2, non-ReBAR, measured 2026-08-07 — see
//! `docs/lessons.md` and `docs/models/omni/status.md` M1 for the write-up):
//! the doubling is real, exactly 2.00x, upload-triggered (allocation alone is
//! 1.00x), independent of `COPY_SRC`/`COPY_DST` usage flags and independent of
//! upload chunk size — but it is **specific to the default wgpu backend's
//! Vulkan HAL**. brain's own native Vulkan backend (`crates/backend-vulkan`,
//! whose `with_staging` reuses one shared, bounded staging buffer — see
//! `crates/vulkan/src/context.rs`) measures a clean **1.00x**. The fix is
//! `--device vulkan` (or `BRAIN_DEVICE=vulkan`), not a wgpu-level change.
//!
//! Six probes, each isolating one candidate cause:
//!   1. `storage_init` (the exact path model weight import takes) at two sizes
//!      — does the overhead scale with size (a per-buffer cost) or stay fixed?
//!   2. Same size, `COPY_DST` only vs `COPY_DST|COPY_SRC` (raw `buffer()` +
//!      `write_f32`) — does dropping the read-back flag change it?
//!   3. Allocate-only, no `write_f32` at all — does the doubling need an
//!      upload, or does it happen on allocation alone?
//!   4. Same size via `write_f32_chunked` (64 MiB chunks) instead of one
//!      `write_f32` — does bounding the largest single `write_buffer` call
//!      bound the resident staging cost?
//!   5. Same probe, wgpu vs. brain's native Vulkan backend — the one that
//!      actually resolved the question (see "Result" above).

use std::process::Command;
use std::time::Duration;

use gpu_core::{BufUsage, Gpu};

const KERNELS: &[(&str, &str)] = &[];
/// This box has two P40s; a probe checks every index present so it never
/// silently reads a delta of zero because a backend enumerated cards in a
/// different order than `nvidia-smi` (which is exactly how probe 5's finding
/// was made: brain's native Vulkan backend put its buffer on index 1).
const GPU_INDICES: &[u32] = &[0, 1];

fn nvidia_smi_used_mib(index: u32) -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", &index.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn nvidia_smi_all_used_mib() -> Vec<(u32, u64)> {
    GPU_INDICES.iter().filter_map(|&i| nvidia_smi_used_mib(i).map(|m| (i, m))).collect()
}

/// Settle time for the driver's memory accounting to catch up with an
/// allocation/free before the next `nvidia-smi` read — chosen generously
/// since this is a one-shot diagnostic, not a hot loop.
fn settle() {
    std::thread::sleep(Duration::from_millis(800));
}

struct Probe {
    label: &'static str,
    logical_mib: u64,
    /// Whichever GPU index actually changed, and by how much — the backend
    /// under test may not place the buffer on the index the caller expected.
    landed_on: Option<(u32, i64)>,
}

impl Probe {
    fn ratio(&self) -> f64 {
        self.landed_on.map(|(_, d)| d as f64 / self.logical_mib as f64).unwrap_or(f64::NAN)
    }
}

/// The delta with the largest magnitude across every card checked — the one
/// the allocation actually landed on. A card whose baseline shifted for an
/// unrelated reason (another process) would show a small delta by comparison,
/// not the ~one full buffer size expected here.
fn dominant_delta(before: &[(u32, u64)], after: &[(u32, u64)]) -> Option<(u32, i64)> {
    before
        .iter()
        .filter_map(|&(i, b)| after.iter().find(|&&(j, _)| j == i).map(|&(_, a)| (i, a as i64 - b as i64)))
        .max_by_key(|&(_, d)| d.abs())
}

enum Upload {
    /// `storage_init` — the exact path model weight import takes.
    Init,
    /// `buffer()` with explicit usage, then `write_f32`.
    WriteAfter(BufUsage),
    /// Allocate via `storage()` and never write anything.
    AllocOnly,
    /// `storage()` alloc, then `write_f32_chunked` in `chunk_words`-sized pieces.
    Chunked(usize),
}

fn probe_storage(gpu: &Gpu, label: &'static str, mib: u64, upload: Upload) -> Probe {
    let before = nvidia_smi_all_used_mib();
    let n = (mib * 1024 * 1024 / 4) as usize; // f32 elements
    let buf = match upload {
        Upload::Init => {
            let data = vec![0.5f32; n];
            gpu.storage_init(label, &data)
        }
        Upload::WriteAfter(u) => {
            let data = vec![0.5f32; n];
            let b = gpu.buffer(label, (n * 4) as u64, u);
            gpu.write_f32(&b, &data);
            b
        }
        Upload::AllocOnly => gpu.storage(n as u64),
        Upload::Chunked(chunk_words) => {
            let data = vec![0.5f32; n];
            let b = gpu.storage(n as u64);
            gpu.write_f32_chunked(&b, &data, chunk_words);
            b
        }
    };
    gpu.flush();
    gpu.poll_wait();
    settle();
    let after = nvidia_smi_all_used_mib();
    drop(buf);
    Probe { label, logical_mib: mib, landed_on: dominant_delta(&before, &after) }
}

fn print_row(p: &Probe) {
    match p.landed_on {
        Some((idx, delta)) => {
            eprintln!("{:<28} {:>10} gpu{:<2} {:>10} {:>7.2}x", p.label, p.logical_mib, idx, delta, p.ratio())
        }
        None => eprintln!("{:<28} {:>10} {:>17}", p.label, p.logical_mib, "no nvidia-smi delta"),
    }
}

#[test]
#[ignore]
fn measure_storage_buffer_resident_overhead() {
    if Command::new("nvidia-smi").arg("-L").output().map(|o| !o.status.success()).unwrap_or(true) {
        eprintln!("skip: nvidia-smi not available");
        return;
    }
    let Ok(Ok(gpu)) = std::panic::catch_unwind(|| Gpu::new_on_index(0, KERNELS)) else {
        eprintln!("skip: could not build a GPU 0 device");
        return;
    };
    if let Some((name, discrete)) = gpu_core::adapter_info() {
        eprintln!("wgpu adapter: {name} (discrete={discrete})");
    }
    settle(); // let device init settle before the first baseline read

    let mut rows = vec![
        // 1. storage_init at two sizes — scaling check.
        probe_storage(&gpu, "wgpu-256mib", 256, Upload::Init),
        probe_storage(&gpu, "wgpu-1024mib", 1024, Upload::Init),
        // 2. COPY_DST only (no COPY_SRC), raw `buffer()`+`write_f32` path —
        //    isolates whether COPY_SRC (readback capability inference-only
        //    buffers never use) is the doubling cause.
        probe_storage(&gpu, "wgpu-1024mib-nocopysrc", 1024, Upload::WriteAfter(BufUsage::STORAGE | BufUsage::COPY_DST)),
        // 3. Allocation only, no upload — isolates alloc-time placement from
        //    the upload path.
        probe_storage(&gpu, "wgpu-1024mib-alloconly", 1024, Upload::AllocOnly),
    ];
    // 4. The chunked-upload fix candidate: same 1024 MiB in 64 MiB pieces.
    const CHUNK_MIB: u64 = 64;
    rows.push(probe_storage(&gpu, "wgpu-1024mib-chunked64", 1024, Upload::Chunked((CHUNK_MIB * 1024 * 1024 / 4) as usize)));

    // 5. Same probe, on brain's own native Vulkan backend instead of wgpu.
    // Drop the wgpu device fully first (one GPU device per process is the
    // house rule, and a live second device would confound the delta anyway).
    drop(gpu);
    settle();
    match Gpu::try_new_vulkan(KERNELS) {
        Ok(vk_gpu) => rows.push(probe_storage(&vk_gpu, "native-vulkan-1024mib", 1024, Upload::Init)),
        Err(e) => eprintln!("skip native-vulkan probe: {e}"),
    }

    eprintln!("\n{:<28} {:>10} {:>4} {:>10} {:>8}", "probe", "logical", "on", "delta", "ratio");
    eprintln!("{:<28} {:>10} {:>4} {:>10} {:>8}", "", "MiB", "", "MiB", "x");
    for p in &rows {
        print_row(p);
    }

    let landed: Vec<&Probe> = rows.iter().filter(|p| p.landed_on.is_some()).collect();
    assert!(!landed.is_empty(), "no probe produced an nvidia-smi delta on any tracked GPU index");

    // Report only — this is a diagnostic, not a gate. The plan of record's M1
    // decision (fix vs. hybrid-CPU-expert fallback) is made from these numbers
    // by a human/agent reading the printed table, not by this assertion.
    let by_label: std::collections::HashMap<&str, f64> = landed.iter().map(|p| (p.label, p.ratio())).collect();
    let get = |l: &str| by_label.get(l).copied();
    eprintln!();
    if let (Some(a), Some(b)) = (get("wgpu-1024mib"), get("wgpu-1024mib-nocopysrc")) {
        eprintln!("COPY_SRC effect:      with={a:.2}x without={b:.2}x (delta {:.2}x)", a - b);
    }
    if let (Some(a), Some(b)) = (get("wgpu-1024mib"), get("wgpu-1024mib-alloconly")) {
        eprintln!("upload effect:        with-upload={a:.2}x alloc-only={b:.2}x (delta {:.2}x)", a - b);
    }
    if let (Some(a), Some(b)) = (get("wgpu-1024mib"), get("wgpu-1024mib-chunked64")) {
        eprintln!("chunked-upload fix:   one-shot={a:.2}x chunked({CHUNK_MIB}MiB)={b:.2}x (delta {:.2}x)", a - b);
    }
    if let (Some(a), Some(b)) = (get("wgpu-1024mib"), get("native-vulkan-1024mib")) {
        eprintln!("backend fix:          wgpu={a:.2}x native-vulkan={b:.2}x (delta {:.2}x)  <-- the answer", a - b);
    }
}
