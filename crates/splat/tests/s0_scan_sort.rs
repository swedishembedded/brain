// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the generic device scan + radix-sort primitives, run
//! on the CPU (Cranelift JIT) backend so it needs no GPU; the same checks run
//! on wgpu unless MOE_SKIP_GPU_TESTS is set. Device results are compared
//! against plain Rust references (iterator scan, stable sort).

use gpu_core::{BufUsage, DeviceBuffer, Gpu};
use splat::sort::{record_scan, record_sort_pairs, ScanScratch, SortScratch};
use splat::Kernels;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn ubuf(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.buffer(
        "u",
        (data.len().max(1) * 4) as u64,
        BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC,
    );
    g.write(&b, data);
    b
}

fn read_u32(g: &Gpu, b: &DeviceBuffer, n: usize) -> Vec<u32> {
    g.read(b, n).iter().map(|x| x.to_bits()).collect()
}

fn check_scan(g: &Gpu, ks: &Kernels, n: usize, seed: u64) {
    let mut r = Lcg(seed);
    let data: Vec<u32> = (0..n).map(|_| r.next_u32() % 1000).collect();
    let expect: Vec<u32> = data
        .iter()
        .scan(0u32, |acc, &v| {
            let out = *acc;
            *acc += v;
            Some(out)
        })
        .collect();
    let total: u32 = data.iter().sum();

    let buf = ubuf(g, &data);
    let scratch = ScanScratch::new(g, n);
    let mut steps = Vec::new();
    record_scan(g, ks, &buf, n, &scratch, &mut steps);
    g.submit(&[], &steps);

    assert_eq!(read_u32(g, &buf, n), expect, "scan mismatch n={n}");
    assert_eq!(read_u32(g, scratch.total(), 1)[0], total, "scan total n={n}");
}

fn check_sort(g: &Gpu, ks: &Kernels, n: usize, key_bits: u32, seed: u64) {
    let mut r = Lcg(seed);
    let mask = if key_bits == 32 { u32::MAX } else { (1u32 << key_bits) - 1 };
    let keys: Vec<u32> = (0..n).map(|_| r.next_u32() & mask).collect();
    let vals: Vec<u32> = (0..n as u32).collect();
    let mut expect: Vec<(u32, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
    expect.sort_by_key(|&(k, _)| k); // stable: preserves val order for equal keys

    let ka = ubuf(g, &keys);
    let va = ubuf(g, &vals);
    let kb = ubuf(g, &vec![0u32; n]);
    let vb = ubuf(g, &vec![0u32; n]);
    let scratch = SortScratch::new(g, n);
    let mut steps = Vec::new();
    let in_b = record_sort_pairs(g, ks, &ka, &va, &kb, &vb, n, key_bits, &scratch, &mut steps);
    g.submit(&[], &steps);

    let (rk, rv) = if in_b { (&kb, &vb) } else { (&ka, &va) };
    let got_k = read_u32(g, rk, n);
    let got_v = read_u32(g, rv, n);
    for i in 0..n {
        assert_eq!(
            (got_k[i], got_v[i]),
            expect[i],
            "sort mismatch at {i} (n={n}, bits={key_bits})"
        );
    }
}

fn run_all(g: &Gpu) {
    let ks = Kernels::at(0);
    for (i, &n) in [1usize, 5, 255, 256, 257, 4096, 65535, 65536, 65537, 1_000_000].iter().enumerate() {
        check_scan(g, &ks, n, 0x5eed + i as u64);
    }
    // duplicate-heavy (8-bit keys), medium, and full-width keys; stability
    // asserted via the original-index payloads.
    for (i, &(n, bits)) in
        [(1usize, 32u32), (255, 8), (256, 8), (257, 16), (65537, 16), (200_000, 32)].iter().enumerate()
    {
        check_sort(g, &ks, n, bits, 0xab1e + i as u64);
    }
}

#[test]
fn scan_sort_cpu() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    run_all(&g);
}

#[test]
fn scan_sort_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let g = gpu_core::testgpu::dev(splat::PIPELINES);
    run_all(&g);
}

/// Large-size smoke: 4M-element scan (3 recursion levels) on CPU.
#[test]
fn scan_4m_cpu() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    check_scan(&g, &ks, 4_000_000, 0xbead);
}
