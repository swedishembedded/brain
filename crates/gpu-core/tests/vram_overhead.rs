// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measures the non-ReBAR Pascal "resident bytes doubled per storage buffer" cost, also
//! noted in `crates/qwen3/src/q8.rs` — directly, via `nvidia-smi` memory
//! deltas around known allocations, rather than by inference from a model's
//! total footprint.
//!
//! ```text
//! cargo test --release -p brain-gpu-core \
//!   -- --ignored --nocapture --test-threads=1 vram_overhead
//! ```
//!
//! One GPU device per process (`AGENTS.md`), so this is its own test binary
//! run with `--test-threads=1` and is `#[ignore]`d out of the fast lane, same
//! shape as `bench_matmul.rs`. Needs `nvidia-smi` on `$PATH` (skips cleanly
//! otherwise).
//!
//! **Result** (P40 ×2, measured 2026-08-07): the doubling was real and exact,
//! upload-triggered (allocation alone costs no extra resident bytes),
//! independent of `COPY_SRC`/`COPY_DST` usage flags and independent of upload
//! chunk size - and **specific to the default wgpu backend's Vulkan HAL**.
//! brain's own native Vulkan backend (`crates/backend-vulkan`, whose
//! `with_staging` reuses one shared, bounded staging buffer - see
//! `crates/vulkan/src/context.rs`) measured **no overhead at all**.
//!
//! **Fixed** (2026-08-21, same box): `wgpu-hal`'s Vulkan backend asked
//! `gpu-allocator` for `MemoryLocation::CpuToGpu` for every `MAP_WRITE`
//! buffer, whose preferred property bits include `DEVICE_LOCAL`; this card
//! exposes a `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory type drawn
//! from the whole 24 GiB VRAM heap, so every staged byte was allocated in
//! video memory - and the staging copy stays resident alongside the
//! destination it is copied into until the submission consuming it retires,
//! which is why peak resident was 2N for an N-byte upload and why chunking
//! could not bound it (the chunks are all live at once). A pure upload
//! staging buffer
//! (`MAP_WRITE` and nothing beyond `COPY_SRC` - exactly what `wgpu_core`
//! allocates for `Queue::write_buffer`) is now steered at host-visible,
//! non-device-local memory. **wgpu now measures no overhead on any probe
//! here**, equal to native Vulkan. The fix lives in the dependency, not in brain:
//! see the workspace root `Cargo.toml`'s `[patch]` notes for how it is
//! consumed and for the full root-cause write-up it points at.
//! `--device vulkan` is no longer needed to avoid the doubling.
//!
//! **It was not free, and this file is why we know.** The upload-throughput
//! probes below were added with the placement fix, precisely so the cost side
//! could not be assumed - and they immediately found that host-staged uploads
//! ran at roughly HALF the throughput of the old VRAM-staged ones, for a 1 GiB
//! buffer at the 4 MiB chunk size real weight upload uses. Two causes, both
//! since fixed, and both found by measuring rather than reasoning:
//!
//! 1. **Host memory is expensive to ALLOCATE, and device memory is not.**
//!    `vkAllocateMemory` from a device-local heap hands back an address range
//!    the driver already owns, in a time flat in the size. From a host heap it
//!    has to commit and pin the pages, at a cost linear in the size and at a
//!    rate slower than the upload those pages exist to carry. With a
//!    fresh allocation behind every `write_buffer`, pinning rather than
//!    copying became the whole cost. `wgpu-hal` now recycles upload staging
//!    buffers instead of allocating one per write.
//! 2. **These probes were not submitting the upload at all.** `write_buffer`
//!    only copies into a staging buffer and records a copy; nothing reaches
//!    the device until a `queue.submit`, and `backend-wgpu`'s `flush` used to
//!    return early when no DISPATCH was pending, treating uploads as not
//!    being work. Tracing the Vulkan staging allocator underneath this file
//!    showed all 1536 staging buffers of a 1 GiB chunked upload created
//!    before the first was released - one unbroken run of allocations, then
//!    one unbroken run of frees at teardown. See
//!    `crates/backend-wgpu/tests/upload_flush.rs`, which pins the contract.
//!
//! With both fixed, on the same card, the host-staged path uploads several
//! times faster than the original VRAM-staged one at every chunk size probed
//! (4 MiB, 64 MiB, and one single write), with the mid chunk size fastest. So
//! the placement fix is no longer a trade: it costs half the VRAM AND uploads
//! faster. The roadmap ledger has the full before/during/after table.
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

/// How an upload is issued, for [`probe_upload_gbs`].
#[derive(Clone, Copy)]
enum How {
    /// One `write_f32` for the whole buffer.
    OneShot,
    /// `write_f32_chunked` in N-MiB pieces, no submit until the end.
    Chunked(u64),
    /// N-MiB pieces with a submit after each, bounding live staging to one piece.
    ChunkedFlushed(u64),
}

/// Host-to-device upload throughput for the same path the probes above
/// measure the RESIDENT cost of, in GB/s - the other half of the same trade.
///
/// Where a staging buffer lives decides both numbers at once: host memory
/// costs no VRAM but adds a DMA hop, device-local host-visible memory costs
/// VRAM but lets the CPU write straight into it. A change that wins on the
/// resident side and quietly loses here would be no win at all, so the two
/// are measured together rather than one being assumed.
///
/// Allocation of the DESTINATION is excluded (the buffer is created before the
/// clock starts); whatever the backend has to allocate for staging is not, and
/// should not be - that cost is part of the upload. The best of `reps` is
/// reported, so a concurrent job on the same card can only make this number
/// pessimistic, never optimistic.
///
/// `how` picks the granularity, and the three are not variations on one path -
/// they price three different staging behaviours:
///
/// * [`How::OneShot`] stages the whole buffer at once.
/// * [`How::Chunked`] splits the write into `c`-MiB pieces, which is what
///   `write_at`'s own doc tells a caller streaming a large tensor to do and
///   therefore what the real weight-import path looks like. It bounds the
///   largest single staging allocation but NOT the total live at once:
///   `wgpu_core` allocates a fresh staging buffer per write and holds every
///   one of them until the next submission, so N chunks still means N chunks'
///   worth of staging resident and, in host memory, page-pinned.
/// * [`How::ChunkedFlushed`] submits after each chunk, so at most one chunk of
///   staging is ever live. This is what a hand-written backend does with one
///   reused staging buffer, and it is the probe that says whether the cost of
///   host-memory staging is the placement itself or merely the volume of it.
fn probe_upload_gbs(gpu: &Gpu, mib: u64, reps: usize, how: How) -> f64 {
    let n = (mib * 1024 * 1024 / 4) as usize;
    let data = vec![0.5f32; n];
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let b = gpu.storage(n as u64);
        let t0 = std::time::Instant::now();
        match how {
            How::OneShot => gpu.write_f32(&b, &data),
            How::Chunked(c) => gpu.write_f32_chunked(&b, &data, (c * 1024 * 1024 / 4) as usize),
            How::ChunkedFlushed(c) => {
                let cw = (c * 1024 * 1024 / 4) as usize;
                for (i, part) in data.chunks(cw).enumerate() {
                    gpu.write_f32_at(&b, (i * cw) as u64, part);
                    gpu.flush();
                }
            }
        }
        gpu.flush();
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
        drop(b);
        gpu.flush();
    }
    (n as f64 * 4.0) / best / 1e9
}

/// Allocate a buffer, drop it, and report the VRAM still held afterwards, in
/// MiB - the question "does dropping a device buffer actually give the memory
/// back?", which is separate from how much a live buffer costs.
///
/// It is asked because on the native Vulkan backend the answer used to be NO:
/// its buffer type had no `Drop`, so every buffer any model ever allocated
/// stayed on the card until the whole device was destroyed. That is invisible
/// to a resident model (its weights live forever anyway) and fatal to anything
/// with a working SET - `omni`'s bf16 Thinker drops each streamed layer's
/// ~2.4 GiB before loading the next, and instead accumulated all of them until
/// a 24 GB card reported `ERROR_OUT_OF_DEVICE_MEMORY` mid-request.
///
/// `reps` allocations are made and dropped in sequence, so a backend that
/// frees nothing shows `reps x mib` held rather than one buffer's worth -
/// which also distinguishes a real leak from allocator retention of a single
/// block for reuse.
fn probe_drop_releases(gpu: &Gpu, label: &'static str, mib: u64, reps: usize) -> Probe {
    settle();
    let before = nvidia_smi_all_used_mib();
    let n = (mib * 1024 * 1024 / 4) as usize;
    for _ in 0..reps {
        let b = gpu.storage(n as u64);
        gpu.write_f32(&b, &vec![0.5f32; n]);
        gpu.flush();
        gpu.poll_wait();
        drop(b);
        // A backend that defers destruction to a safe point needs one call
        // past the drop to reach it; production loops hit this via their next
        // dispatch or readback.
        gpu.flush();
    }
    gpu.poll_wait();
    settle();
    let after = nvidia_smi_all_used_mib();
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
        brain_testutil::skip_unavailable("nvidia-smi not available");
        return;
    }
    let Ok(Ok(gpu)) = std::panic::catch_unwind(|| Gpu::new_on_index(0, KERNELS)) else {
        brain_testutil::skip_unavailable("could not build a GPU 0 device");
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
    /// `paramstore::UPLOAD_CHUNK_WORDS` (1 << 20 words = 4 MiB) - the chunk
    /// size every real weight upload in this repo actually uses, and the one
    /// number in this file that describes production rather than a probe.
    /// Whether staging is sub-allocated from a pooled block or gets its own
    /// `VkDeviceMemory` turns on how this compares to the allocator's block
    /// size, so the two chunk sizes measure genuinely different paths.
    const PROD_CHUNK_MIB: u64 = 4;
    rows.push(probe_storage(&gpu, "wgpu-1024mib-chunked64", 1024, Upload::Chunked((CHUNK_MIB * 1024 * 1024 / 4) as usize)));
    let wgpu_upload_gbs = probe_upload_gbs(&gpu, 1024, 3, How::OneShot);
    let wgpu_upload_chunked_gbs = probe_upload_gbs(&gpu, 1024, 3, How::Chunked(CHUNK_MIB));
    let wgpu_upload_prod_gbs = probe_upload_gbs(&gpu, 1024, 3, How::Chunked(PROD_CHUNK_MIB));
    let wgpu_upload_bounded_gbs = probe_upload_gbs(&gpu, 1024, 3, How::ChunkedFlushed(PROD_CHUNK_MIB));

    // 5. Same probe, on brain's own native Vulkan backend instead of wgpu.
    // Drop the wgpu device fully first (one GPU device per process is the
    // house rule, and a live second device would confound the delta anyway).
    drop(gpu);
    settle();
    let mut vk_upload_gbs = None;
    match Gpu::try_new_vulkan(KERNELS) {
        Ok(vk_gpu) => {
            rows.push(probe_storage(&vk_gpu, "native-vulkan-1024mib", 1024, Upload::Init));
            vk_upload_gbs = Some((probe_upload_gbs(&vk_gpu, 1024, 3, How::OneShot), probe_upload_gbs(&vk_gpu, 1024, 3, How::Chunked(PROD_CHUNK_MIB))));
            // 6. Does dropping give it back? Four 1 GiB buffers allocated and
            //    dropped one at a time: ~0 MiB held if Drop frees, ~4096 if
            //    nothing is ever released.
            rows.push(probe_drop_releases(&vk_gpu, "native-vulkan-drop-x4", 1024, 4));
        }
        Err(e) => brain_testutil::skip_unavailable(&format!("native-vulkan probe: {e}")),
    }

    eprintln!("\n{:<28} {:>10} {:>4} {:>10} {:>8}", "probe", "logical", "on", "delta", "ratio");
    eprintln!("{:<28} {:>10} {:>4} {:>10} {:>8}", "", "MiB", "", "MiB", "x");
    for p in &rows {
        print_row(p);
    }

    let landed: Vec<&Probe> = rows.iter().filter(|p| p.landed_on.is_some()).collect();
    assert!(!landed.is_empty(), "no probe produced an nvidia-smi delta on any tracked GPU index");

    // Report only - this is a diagnostic, not a gate. The fix-vs.-hybrid-
    // CPU-expert-fallback decision is made from these numbers by a
    // human/agent reading the printed table, not by this assertion.
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
    eprintln!("upload 1 GiB one-shot:      wgpu={wgpu_upload_gbs:.2} GB/s");
    eprintln!("upload 1 GiB, {CHUNK_MIB}MiB chunks: wgpu={wgpu_upload_chunked_gbs:.2} GB/s");
    eprintln!("upload 1 GiB, {PROD_CHUNK_MIB}MiB chunks:  wgpu={wgpu_upload_prod_gbs:.2} GB/s  <-- the size real weight upload uses");
    eprintln!("  same, submitting per chunk: wgpu={wgpu_upload_bounded_gbs:.2} GB/s  <-- live staging bounded to one chunk");
    if let Some((one, prod)) = vk_upload_gbs {
        eprintln!("  same, native-vulkan:      one-shot={one:.2} GB/s {PROD_CHUNK_MIB}MiB-chunked={prod:.2} GB/s");
    }
    // Unlike the ratios above (a diagnostic a human reads), this one IS a
    // gate: "a dropped buffer frees" is a correctness property, not a
    // measurement. 4 GiB allocated and dropped must not still be held.
    if let Some(held) = rows.iter().find(|p| p.label == "native-vulkan-drop-x4").and_then(|p| p.landed_on) {
        let (idx, delta) = held;
        eprintln!("drop releases:        gpu{idx} still holds {delta} MiB after 4x1024 MiB allocated and dropped");
        assert!(
            delta < 1024,
            "native Vulkan backend held {delta} MiB on gpu{idx} after allocating and dropping 4x1024 MiB -- \
             dropped buffers are not being freed, which is what walks a streaming model to OOM"
        );
    }
}
