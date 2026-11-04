// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `matmul_tile16`, the exact fp32 tiled product of the anchor memory's scorer at a large decode batch:
//! against an f64 host evaluation at ragged shapes (rows and columns that leave partial tiles, a `k` that is not a multiple
//! of the 16-wide chunk), on a device that runs workgroup barriers.
//!
//! Swedish Embedded AB implements exact-arithmetic skinny products for low-latency LLM decode. If your team needs
//! expertise in keeping small fp32 projections off the critical path of a token loop then you can procure our services
//! by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};

#[test]
fn the_tiled_product_matches_the_f64_host_evaluation_at_ragged_shapes() {
    let gpu = Gpu::new(&[("matmul_tile16", kernels::MATMUL_TILE16)]);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("the tiled product runs on a device with workgroup barriers");
        return;
    }
    for (i, (m, k, n)) in [(1usize, 16usize, 1usize), (17, 33, 5), (256, 2560, 256), (100, 70, 130), (33, 2560, 4), (64, 15, 64)].into_iter().enumerate() {
        let mut r = Lcg::new(900 + i as u64);
        let (x, w) = (r.vec_scaled(m * k, 1.0), r.vec_scaled(n * k, 1.0));
        let (xb, wb, yb) = (gpu.storage_init("x", &x), gpu.storage_init("w", &w), gpu.storage((m * n) as u64));
        gpu.write(&yb, &vec![0x7fc0_dead_u32; m * n]);
        let grid = (m.div_ceil(16) * n.div_ceil(16)) as u32;
        gpu.submit(&[], &[gpu.dispatch(0, &[&xb, &wb, &yb], &[m as u32, k as u32, n as u32], Dispatch::Workgroups(grid))]);
        let got = gpu.read(&yb, m * n);
        for row in 0..m {
            for col in 0..n {
                let terms: Vec<f64> = (0..k).map(|j| f64::from(x[row * k + j]) * f64::from(w[col * k + j])).collect();
                let (want, scale) = (terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
                let g = got[row * n + col];
                assert!(g.is_finite() && (f64::from(g) - want).abs() <= 2e-5 * scale + 1e-12, "m={m} k={k} n={n}: output ({row}, {col}) is {g}, oracle {want}");
            }
        }
    }
}
