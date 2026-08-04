// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic device prefix-scan and stable LSD radix sort, built from the
//! `scan_block`/`scan_add`/`sort_hist`/`sort_scatter` kernels. Atomic-free and
//! barrier-free by construction: scans are block-sequential + recursive
//! add-back, the sort ranks per 256-element chunk with private offset tables.
//!
//! Callers *record* steps into a `Vec<Step>` (nothing executes until
//! `Gpu::submit`), so scans/sorts compose into larger frame pipelines.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::Kernels;

/// Elements per `scan_block` run. Fixed by convention; the kernels take it as
/// a parameter so it can be tuned without a WGSL change.
pub const SCAN_BLOCK_LEN: usize = 256;

/// Keys per `sort_hist`/`sort_scatter` chunk (one invocation each).
pub const SORT_CHUNK_LEN: usize = 256;

/// Per-level block-sum buffers for an exclusive scan of up to `max_n`
/// elements. The last level always has a single element: after the recorded
/// scan ran, it holds the grand total of the scanned range.
pub struct ScanScratch {
    levels: Vec<(DeviceBuffer, usize)>,
    max_n: usize,
}

impl ScanScratch {
    pub fn new(gpu: &Gpu, max_n: usize) -> ScanScratch {
        assert!(max_n >= 1);
        let mut levels = Vec::new();
        let mut n = max_n;
        loop {
            let nb = n.div_ceil(SCAN_BLOCK_LEN);
            levels.push((gpu.storage(nb as u64), nb));
            if nb <= 1 {
                break;
            }
            n = nb;
        }
        ScanScratch { levels, max_n }
    }

    /// Buffer whose element 0 holds the grand total once the scan has run.
    pub fn total(&self) -> &DeviceBuffer {
        &self.levels.last().unwrap().0
    }
}

/// Record an in-place exclusive prefix scan of `data[0..n]` (u32 payload).
/// After execution `ScanScratch::total()` holds the sum of the scanned range.
pub fn record_scan(
    gpu: &Gpu,
    ks: &Kernels,
    data: &DeviceBuffer,
    n: usize,
    scratch: &ScanScratch,
    steps: &mut Vec<Step>,
) {
    assert!(n >= 1 && n <= scratch.max_n, "scan n={n} exceeds scratch max {}", scratch.max_n);
    record_scan_level(gpu, ks, data, n, scratch, 0, steps);
}

fn record_scan_level(
    gpu: &Gpu,
    ks: &Kernels,
    data: &DeviceBuffer,
    n: usize,
    scratch: &ScanScratch,
    level: usize,
    steps: &mut Vec<Step>,
) {
    let sums = &scratch.levels[level].0;
    let nb = n.div_ceil(SCAN_BLOCK_LEN);
    let params = [n as u32, SCAN_BLOCK_LEN as u32];
    steps.push(gpu.step(ks.scan_block, &[data, sums], &params, nb as u32));
    if nb > 1 {
        record_scan_level(gpu, ks, sums, nb, scratch, level + 1, steps);
        steps.push(gpu.step(ks.scan_add, &[data, sums], &params, n as u32));
    }
}

/// Scratch for a radix sort of up to `max_n` (key, value) pairs: the
/// column-major digit histogram plus the scan scratch to turn it into global
/// scatter offsets.
pub struct SortScratch {
    hist: DeviceBuffer,
    hist_scan: ScanScratch,
    max_n: usize,
}

impl SortScratch {
    pub fn new(gpu: &Gpu, max_n: usize) -> SortScratch {
        assert!(max_n >= 1);
        let max_chunks = max_n.div_ceil(SORT_CHUNK_LEN);
        let hist_n = 256 * max_chunks;
        SortScratch {
            hist: gpu.storage(hist_n as u64),
            hist_scan: ScanScratch::new(gpu, hist_n),
            max_n,
        }
    }
}

/// Record a stable LSD radix sort of the pairs `(keys_a, vals_a)[0..n]` by the
/// low `key_bits` key bits, ping-ponging through the equally-sized B buffers.
/// Returns `true` when the sorted result ends up in the B buffers (odd number
/// of 8-bit digit passes).
#[allow(clippy::too_many_arguments)]
pub fn record_sort_pairs(
    gpu: &Gpu,
    ks: &Kernels,
    keys_a: &DeviceBuffer,
    vals_a: &DeviceBuffer,
    keys_b: &DeviceBuffer,
    vals_b: &DeviceBuffer,
    n: usize,
    key_bits: u32,
    scratch: &SortScratch,
    steps: &mut Vec<Step>,
) -> bool {
    assert!(n >= 1 && n <= scratch.max_n, "sort n={n} exceeds scratch max {}", scratch.max_n);
    assert!((1..=32).contains(&key_bits));
    let passes = key_bits.div_ceil(8);
    let n_chunks = n.div_ceil(SORT_CHUNK_LEN);
    let hist_n = 256 * n_chunks;
    let mut src = (keys_a, vals_a);
    let mut dst = (keys_b, vals_b);
    for pass in 0..passes {
        let params = [n as u32, 8 * pass, n_chunks as u32, SORT_CHUNK_LEN as u32];
        steps.push(gpu.step(ks.sort_hist, &[src.0, &scratch.hist], &params, n_chunks as u32));
        record_scan(gpu, ks, &scratch.hist, hist_n, &scratch.hist_scan, steps);
        steps.push(gpu.step(
            ks.sort_scatter,
            &[src.0, src.1, &scratch.hist, dst.0, dst.1],
            &params,
            n_chunks as u32,
        ));
        core::mem::swap(&mut src, &mut dst);
    }
    passes % 2 == 1
}
