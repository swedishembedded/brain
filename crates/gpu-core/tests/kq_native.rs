// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native affine K-quant kernels (`kernels_cuda`): the
//! decode GEMVs (`cu/matmul_kq_gemv.cu`) and the prefill GEMMs (the int8
//! GEMM's block in `cu/matmul_i8_dp4a.cu`), which `gpu_core::native_upgrade`
//! substitutes for `matmul_kq_gemv.wgsl` and `matmul_kq_dyn.wgsl` (Q4_K's
//! 4-bit and Q5_K's 8-bit code slots) on a CUDA device.
//!
//! Swedish Embedded AB implements native kernels for serving GGUF K-quant
//! models. If your team needs expertise in 4-bit LLM decode that is both fast
//! and accountable to an oracle, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The native kernels sum a 32-element group exactly in integers and round
//! twice per group; the WGSL tiers round differently. So the claim
//! is accuracy, not bits: against an f64 oracle over the same inputs, every
//! output is within the fp32 summation bound of its terms, and no worse than
//! the WGSL tier's own error on the same case. The cases cover what the vector
//! loads and the scale indexing can get wrong: `K` under one 16-lane stride
//! and with ragged strides, an odd group count (half a `wsm` word, a partial
//! `wd` super-block), `n` off the block's 16 weight rows, every row count the
//! kernel serves, and the real Qwen3.8-27B projection widths.

use gpu_core::{Dispatch, Gpu};

/// Rows of x the native GEMV serves: the decode regime of the WGSL GEMV.
const NATIVE_ROWS: u32 = backend_api::select::DECODE_REGIME_MAX_ROWS;

/// Which of the two WGSL kernels a case dispatches.
#[derive(Clone, Copy)]
enum Kernel {
    /// `matmul_kq_gemv`: one workgroup per weight row.
    Gemv,
    /// `matmul_kq_dyn`: one workgroup per 128 x 128 output tile.
    Dyn,
}

impl Kernel {
    fn workgroups(self, m: u32, n: u32) -> u32 {
        match self {
            Kernel::Gemv => n,
            Kernel::Dyn => m.div_ceil(128) * n.div_ceil(128),
        }
    }
}

/// The WGSL kernel under its upgraded name and under a name no table knows.
fn device(kernel: Kernel, bits: u32) -> Gpu {
    let (base, src) = match kernel {
        Kernel::Gemv => ("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV),
        Kernel::Dyn => ("matmul_kq_dyn", kernels::MATMUL_KQ_DYN),
    };
    let (name, src) = kernels::template::interned(base, src, &[("CODE_BITS", bits)]).expect("the K-quant kernel template");
    gpu_core::testgpu::dev(Box::leak(Box::new([(name, src), ("kq_reference", src)])))
}

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled() && gpu.caps().numeric.int8_dot
}

/// One K-quant GEMV's inputs and its f64 oracle.
struct Case {
    m: u32,
    k: u32,
    n: u32,
    xq: Vec<u32>,
    wq: Vec<u32>,
    sx: Vec<f32>,
    wsm: Vec<u32>,
    wd: Vec<u32>,
}

impl Case {
    fn new(bits: u32, m: u32, k: u32, n: u32, seed: u64) -> Case {
        let mut r = data::rng::Lcg::new(seed);
        let (mu, ku, nu) = (m as usize, k as usize, n as usize);
        let groups = ku / 32;
        // Q4_K codes are any nibble; Q5_K codes are below 32 in their byte.
        let code_mask = if bits == 4 { u32::MAX } else { 0x1f1f_1f1f };
        let f16 = |v: f32| u32::from(half::f16::from_f32(v).to_bits());
        Case {
            m,
            k,
            n,
            xq: (0..mu * ku / 4).map(|_| r.next_u32()).collect(),
            wq: (0..nu * ku * bits as usize / 32).map(|_| r.next_u32() & code_mask).collect(),
            sx: (0..mu).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect(),
            // Q4_K/Q5_K sub-scales and mins are 6-bit.
            wsm: (0..nu * groups.div_ceil(2)).map(|_| r.next_u32() & 0x3f3f_3f3f).collect(),
            wd: (0..nu * groups.div_ceil(8)).map(|_| f16(1e-4 + (r.next_u32() % 100) as f32 * 1e-5) | (f16((r.next_u32() % 100) as f32 * 1e-5) << 16)).collect(),
        }
    }

    /// `(value, sum of the magnitudes of its terms)` of each output `outs`, in f64.
    fn oracle(&self, bits: u32, outs: &[usize]) -> Vec<(f64, f64)> {
        let (k, n) = (self.k as usize, self.n as usize);
        let groups = k / 32;
        let x = |m: usize, c: usize| f64::from((self.xq[m * k / 4 + c / 4] >> (8 * (c % 4))) as u8 as i8);
        let code = |row: usize, c: usize| {
            let per = 32 / bits as usize;
            f64::from((self.wq[row * k * bits as usize / 32 + c / per] >> (bits as usize * (c % per))) & ((1 << bits) - 1))
        };
        let f16 = |b: u32| f64::from(half::f16::from_bits(b as u16).to_f32());
        outs.iter()
            .map(|&o| {
                let (m, row) = (o / n, o % n);
                let (mut v, mut mag) = (0.0f64, 0.0f64);
                for g in 0..groups {
                    let sm = self.wsm[row * groups.div_ceil(2) + g / 2] >> (16 * (g % 2));
                    let d = self.wd[row * groups.div_ceil(8) + g / 8];
                    let (ds, dm) = (f16(d & 0xffff) * f64::from(sm & 0xff), f16(d >> 16) * f64::from((sm >> 8) & 0xff));
                    let a: f64 = (0..32).map(|i| x(m, 32 * g + i) * code(row, 32 * g + i)).sum();
                    let s: f64 = (0..32).map(|i| x(m, 32 * g + i)).sum();
                    v += ds * a - dm * s;
                    mag += (ds * a).abs() + (dm * s).abs();
                }
                let sx = f64::from(self.sx[m]);
                (v * sx, mag * sx)
            })
            .collect()
    }
}

/// Both tiers' outputs for `case`: (native, WGSL).
fn run(gpu: &Gpu, kernel: Kernel, case: &Case) -> (Vec<f32>, Vec<f32>) {
    let as_f32 = |v: &[u32]| v.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>();
    let (m, k, n) = (case.m, case.k, case.n);
    let xq = gpu.storage_init("xq", &as_f32(&case.xq));
    let wq = gpu.storage_init("wq", &as_f32(&case.wq));
    let sx = gpu.storage_init("sx", &case.sx);
    let wsm = gpu.storage_init("wsm", &as_f32(&case.wsm));
    let wd = gpu.storage_init("wd", &as_f32(&case.wd));
    // The group sums the WGSL tier reads (the native kernel computes its own).
    let xgs: Vec<f32> = (0..m as usize * k as usize / 32)
        .map(|i| {
            let (row, g) = (i / (k as usize / 32), i % (k as usize / 32));
            (0..8).flat_map(|w| case.xq[row * k as usize / 4 + 8 * g + w].to_le_bytes()).map(|b| f32::from(b as i8)).sum()
        })
        .collect();
    let xgs = gpu.storage_init("xgs", &xgs);
    let outs = [gpu.storage((m * n) as u64), gpu.storage((m * n) as u64)];
    let steps: Vec<_> = (0..2).map(|kind| gpu.dispatch(kind, &[&xq, &wq, &sx, &wsm, &wd, &xgs, &outs[kind]], &[m, k, n], Dispatch::Workgroups(kernel.workgroups(m, n)))).collect();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    (gpu.read(&outs[0], (m * n) as usize), gpu.read(&outs[1], (m * n) as usize))
}

/// Native within the fp32 summation bound of the oracle, and no further from
/// it than the WGSL tier (up to the same bound).
fn check(gpu: &Gpu, kernel: Kernel, bits: u32, m: u32, k: u32, n: u32, seed: u64) {
    let case = Case::new(bits, m, k, n, seed);
    let (native, wgsl) = run(gpu, kernel, &case);
    // One rounding per group term and per accumulation step: a group count's
    // worth of f32 epsilons over the terms' magnitude.
    let groups = f64::from(k / 32);
    // Every output of a small case; about 4096 evenly spaced ones of a large
    // one (an odd stride, so the sample lands on every weight row position of
    // a block).
    let total = (m * n) as usize;
    let outs: Vec<usize> = (0..total).step_by(total.div_ceil(4096) | 1).collect();
    for (&o, &(want, mag)) in outs.iter().zip(&case.oracle(bits, &outs)) {
        let bound = (groups + 4.0) * f64::from(f32::EPSILON) * mag + 1e-30;
        let (e_nat, e_wgsl) = ((f64::from(native[o]) - want).abs(), (f64::from(wgsl[o]) - want).abs());
        let ctx = format!("bits={bits} m={m} k={k} n={n} out={o}: oracle {want}, native {}, wgsl {}", native[o], wgsl[o]);
        assert!(e_nat <= bound, "native outside the summation bound {bound}: {ctx}");
        assert!(e_nat <= e_wgsl + bound, "native less accurate than the WGSL tier: {ctx}");
    }
}

#[test]
fn the_native_kernels_are_offered_exactly_where_they_serve() {
    for (bits, gemv, gemm) in [(4u32, "matmul_kq4_gemv", "matmul_kq4_dp4a"), (8, "matmul_kq8_gemv", "matmul_kq8_dp4a")] {
        let gpu = device(Kernel::Gemv, bits);
        if !is_cuda(&gpu) {
            assert_eq!(gpu.native_kernel_for(0, &[1, 5120, 1024]), None, "only a CUDA device takes a native kernel");
            brain_testutil::skip_unavailable("native K-quant kernels need a CUDA device with a packed int8 dot");
            return;
        }
        for m in 1..=NATIVE_ROWS {
            assert_eq!(gpu.native_kernel_for(0, &[m, 5120, 1024]), Some(gemv), "bits={bits} m={m}");
        }
        assert_eq!(gpu.native_kernel_for(0, &[NATIVE_ROWS + 1, 5120, 1024]), None, "past the decode regime");
        assert_eq!(gpu.native_kernel_for(0, &[1, 5136, 1024]), None, "K not a whole number of groups");
        assert_eq!(gpu.native_kernel_for(1, &[1, 5120, 1024]), None, "the reference alias is not upgraded");
        let gpu = device(Kernel::Dyn, bits);
        for m in [1u32, 33, 512, 4096] {
            assert_eq!(gpu.native_kernel_for(0, &[m, 5120, 1024]), Some(gemm), "bits={bits} m={m}");
        }
        assert_eq!(gpu.native_kernel_for(0, &[512, 5136, 1024]), None, "K not a whole number of groups");
    }
}

#[test]
fn the_native_kernels_meet_the_oracle_over_shapes() {
    for bits in [4u32, 8] {
        let gpu = device(Kernel::Gemv, bits);
        if !is_cuda(&gpu) {
            brain_testutil::skip_unavailable("native K-quant kernels need a CUDA device with a packed int8 dot");
            return;
        }
        // 32 and 96 are under one stride or chunk (96: an odd group count);
        // 544 and 1056 leave ragged strides and chunks; 2624 spans several.
        for (k, n) in [(32u32, 1u32), (96, 7), (544, 17), (1056, 33), (2624, 16)] {
            for m in [1u32, 2, 3, 5, 8, 9, 17, NATIVE_ROWS] {
                check(&gpu, Kernel::Gemv, bits, m, k, n, u64::from(bits * 1000 + m * 31 + k + n));
            }
        }
        let gpu = device(Kernel::Dyn, bits);
        // Partial row and column tiles, one tile, several tiles.
        for (k, n) in [(32u32, 1u32), (96, 130), (544, 128), (1056, 257)] {
            for m in [1u32, 33, 128, 129, 300] {
                check(&gpu, Kernel::Dyn, bits, m, k, n, u64::from(bits * 1000 + m * 31 + k + n));
            }
        }
    }
}

/// The Qwen3.8-27B projections `(K, N)` at decode and at a prefill round.
#[test]
fn the_real_27b_shapes_meet_the_oracle() {
    let gemv = device(Kernel::Gemv, 4);
    if !is_cuda(&gemv) {
        brain_testutil::skip_unavailable("native K-quant kernels need a CUDA device with a packed int8 dot");
        return;
    }
    let gemm = device(Kernel::Dyn, 4);
    for (k, n) in [(5120u32, 48u32), (5120, 1024), (5120, 6144), (6144, 5120), (5120, 10240), (17408, 5120), (5120, 17408)] {
        for m in [1u32, 8] {
            check(&gemv, Kernel::Gemv, 4, m, k, n, u64::from(k + n + m));
        }
        check(&gemm, Kernel::Dyn, 4, 256, k, n, u64::from(k + n));
    }
}
