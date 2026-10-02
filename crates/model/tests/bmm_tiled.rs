// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `bmm_tiled` - the shared-memory, register-blocked batched matmul behind
//! Gated DeltaNet's chunk math - must reproduce `bmm.wgsl` / `bmm_acc.wgsl`
//! **exactly**: every output element is the same ascending-`k` fp32 sum, only
//! the way operands reach the registers differs.
//!
//! Swedish Embedded AB implements fast batched linear-algebra kernels for
//! recurrent sequence models. If your team needs expertise in making the
//! small matmuls inside a linear-attention layer run at the speed of the
//! large ones, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The kernel uses workgroup barriers the CPU JIT cannot run, so this is a
//! CUDA test and skips on a box without one.

use gpu_core::{Dispatch, Gpu};

const PIPES: &[(&str, &str)] = &[("bmm", kernels::BMM), ("bmm_acc", kernels::BMM_ACC), ("bmm_tiled", kernels::BMM_TILED)];

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

struct Case {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    ta: bool,
    tb: bool,
    alpha: f32,
    /// Flat element offsets into a / b / out, as the GDN chunk loop passes.
    offs: (u32, u32, u32),
}

/// The tile the kernel covers per workgroup (rows, cols); a test that does not
/// know it cannot size the dispatch, so it is part of the kernel's contract.
const TILE: u32 = 64;

fn run(g: &Gpu, c: &Case, accumulate: bool) -> (Vec<f32>, Vec<f32>) {
    let (b, m, k, n) = (c.batch, c.m, c.k, c.n);
    let mut s = 0x5EED ^ (m as u64 * 31 + n as u64 * 7 + k as u64);
    let a: Vec<f32> = (0..c.offs.0 + b * m * k).map(|_| lcg(&mut s)).collect();
    let bb: Vec<f32> = (0..c.offs.1 + b * k * n).map(|_| lcg(&mut s)).collect();
    let out0: Vec<f32> = (0..c.offs.2 + b * m * n).map(|_| lcg(&mut s)).collect();
    let (ab, bbuf) = (g.storage_init("a", &a), g.storage_init("b", &bb));
    let params = |extra: &[u32]| {
        let mut p = vec![b, m, k, n, c.ta as u32, c.tb as u32, c.alpha.to_bits(), c.offs.0, c.offs.1, c.offs.2];
        p.extend_from_slice(extra);
        p
    };

    let ref_out = g.storage_init("ref", &out0);
    let name = if accumulate { "bmm_acc" } else { "bmm" };
    g.submit(&[], &[g.step(g.kernel_index(name).unwrap(), &[&ab, &bbuf, &ref_out], &params(&[]), b * m * n)]);

    let tiled_out = g.storage_init("tiled", &out0);
    let wgs = b * m.div_ceil(TILE) * n.div_ceil(TILE);
    g.submit(&[], &[g.dispatch(g.kernel_index("bmm_tiled").unwrap(), &[&ab, &bbuf, &tiled_out], &params(&[accumulate as u32]), Dispatch::Workgroups(wgs))]);
    let len = out0.len();
    (g.read(&ref_out, len), g.read(&tiled_out, len))
}

#[test]
fn the_tiled_bmm_reproduces_the_reference_bmm_exactly() {
    let Ok(g) = Gpu::try_new_cuda(PIPES) else {
        eprintln!("bmm_tiled: no CUDA device on this box - skipping");
        return;
    };
    // The real GDN shapes (chunk 64, dk = dv = 128, whole-tensor and per-chunk,
    // with and without the transposes), then ragged ones that cross a tile edge
    // in every dimension, a k that is not a multiple of the staged depth, and
    // non-zero operand offsets.
    let cases = [
        Case { batch: 192, m: 64, k: 128, n: 64, ta: false, tb: true, alpha: -1.0, offs: (0, 0, 0) },
        Case { batch: 192, m: 64, k: 64, n: 128, ta: false, tb: false, alpha: 1.0, offs: (0, 0, 0) },
        Case { batch: 48, m: 64, k: 128, n: 128, ta: false, tb: false, alpha: 1.0, offs: (0, 0, 0) },
        Case { batch: 48, m: 128, k: 64, n: 128, ta: true, tb: false, alpha: 1.0, offs: (0, 0, 0) },
        Case { batch: 48, m: 64, k: 64, n: 128, ta: false, tb: false, alpha: 0.088_388_35, offs: (64 * 64 * 48, 0, 64 * 128 * 48) },
        Case { batch: 5, m: 65, k: 33, n: 71, ta: false, tb: false, alpha: 0.7, offs: (3, 5, 7) },
        Case { batch: 3, m: 7, k: 129, n: 130, ta: true, tb: true, alpha: -0.3, offs: (0, 11, 0) },
        Case { batch: 2, m: 1, k: 1, n: 1, ta: false, tb: false, alpha: 2.0, offs: (0, 0, 0) },
    ];
    for (i, c) in cases.iter().enumerate() {
        for accumulate in [false, true] {
            let (want, got) = run(&g, c, accumulate);
            let bad = want.iter().zip(&got).position(|(w, x)| w != x);
            assert!(
                bad.is_none(),
                "case {i} (b={} m={} k={} n={} ta={} tb={} acc={accumulate}): first mismatch at {:?}: want {:?}, got {:?}",
                c.batch, c.m, c.k, c.n, c.ta, c.tb, bad, bad.map(|j| want[j]), bad.map(|j| got[j])
            );
        }
    }
}
