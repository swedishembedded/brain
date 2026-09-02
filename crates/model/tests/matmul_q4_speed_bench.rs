// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A/B speed measurement for M5.5's two new kernels against the ones they
//! stand next to, at every shape the correctness suite already swept.
//! `#[ignore]`d - this is a measurement, not a correctness gate, and it is
//! run manually (`cargo test -p brain-model --test matmul_q4_speed_bench --
//! --ignored --nocapture`) to produce the numbers recorded in this
//! campaign's kernel-performance roadmap. Bracketed with `poll_wait()` via
//! `gpu_core::profile::best_of` so this times the device, not the host.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::int4::quantize_weight_q4;

const KERNELS: &[(&str, &str)] = &[
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
    ("matmul_q4_dyn_reg", kernels::MATMUL_Q4_DYN_REG),
    ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
    ("matmul_q4_gemv_reg", kernels::MATMUL_Q4_GEMV_REG),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn quant_x(g: &Gpu, k_maxr: usize, k_qp: usize, x: &gpu_core::DeviceBuffer, m: u32, k: u32) -> (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer) {
    let sx = g.storage(m as u64);
    let xq = g.storage((m * k / 4) as u64);
    let steps = [g.step(k_maxr, &[x, &sx], &[m, k], m), g.step(k_qp, &[x, &sx, &xq], &[m, k], m * k / 4)];
    g.submit(&[], &steps);
    (xq, sx)
}

const REPS: usize = 30;

/// The exact `MREG` bucket ladder `gpu_core::upgrade`'s real UPGRADES row
/// uses (power-of-two, complete for `matmul_q4_gemv`'s own `m <= 32`
/// contract) - duplicated here (that table's own constant is
/// `pub(crate)` to `gpu-core`) so this bench measures the SAME
/// specialisation production dispatch would actually pick per `m`, not a
/// single worst-case build. Measuring `matmul_q4_gemv_reg`'s own plain
/// registered name (its un-templated `MREG = 32` default) at every `m`
/// would repeat exactly the mistake this file exists to avoid: a variant
/// compiled for 32 rows is a documented regression at 1 row.
const MREG_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 32];

fn smallest_bucket_covering(m: u32) -> u32 {
    *MREG_BUCKETS.iter().find(|&&b| m <= b).expect("m must be <= the largest bucket (32)")
}

/// GEMV-shaped: `matmul_q4_gemv` vs `matmul_q4_gemv_reg` across the decode
/// regime, at a shape close to a real model's `d_model` (2048). The `_reg`
/// side dispatches the SAME per-`m` `MREG` bucket the `gpu_core::upgrade`
/// table would pick in production, built directly via
/// `kernels::template::interned` (bypassing the upgrade table's own
/// capability probe, which this bench does not need - it already knows this
/// device has `int8_dot`).
#[test]
#[ignore]
fn gemv_vs_gemv_reg_across_decode_rows() {
    let variants: Vec<(&'static str, &'static str)> = MREG_BUCKETS
        .iter()
        .map(|&b| kernels::template::interned("matmul_q4_gemv_reg", kernels::MATMUL_Q4_GEMV_REG, &[("MREG", b)]).unwrap())
        .collect();
    let mut kernels_with_buckets = KERNELS.to_vec();
    kernels_with_buckets.extend(variants.iter().copied());
    let g = gpu_core::Gpu::new(&kernels_with_buckets);
    if !g.caps().numeric.int8_dot {
        eprintln!("skipping: no packed int8 dot on this device");
        return;
    }
    let (k_maxr, k_qp, k_gemv) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, "matmul_q4_gemv"));
    let (k, n) = (2048u32, 2048u32);
    println!("\nmatmul_q4_gemv vs matmul_q4_gemv_reg (per-m MREG bucket), k={k} n={n}, m swept:\n");
    for &m in &[1u32, 2, 4, 8, 16, 32] {
        let bucket = smallest_bucket_covering(m);
        let reg_name = kernels::template::interned("matmul_q4_gemv_reg", kernels::MATMUL_Q4_GEMV_REG, &[("MREG", bucket)]).unwrap().0;
        let k_reg = idx(&g, reg_name);

        let mut rng = Lcg::new(9000 + u64::from(m));
        let x_h = rng.vec_scaled((m * k) as usize, 1.0);
        let w_h = rng.vec_scaled((n * k) as usize, 1.0);
        let x = g.storage_init("x", &x_h);
        let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m, k);
        let (wq, sw) = quantize_weight_q4(&w_h, n as usize, k as usize);
        let wqb = g.storage(wq.len() as u64);
        g.write(&wqb, &wq);
        let swb = g.storage_init("sw", &sw);
        let out = g.storage((m * n) as u64);

        let st_gemv = vec![g.step(k_gemv, &[&xq, &wqb, &sx, &swb, &out], &[m, k, n], n * 64)];
        let st_reg = vec![g.step(k_reg, &[&xq, &wqb, &sx, &swb, &out], &[m, k, n], n * 64)];
        let t_gemv = gpu_core::profile::best_of(&g, &st_gemv, REPS);
        let t_reg = gpu_core::profile::best_of(&g, &st_reg, REPS);
        let bytes = u64::from(n) * u64::from(k) / 8 + u64::from(n) * u64::from(k) / 32 * 4; // packed weight + scales, dominant term
        let gbs_gemv = bytes as f64 / t_gemv / 1e9;
        let gbs_reg = bytes as f64 / t_reg / 1e9;
        println!(
            "m={m:>3}  matmul_q4_gemv {:>8.4} ms ({:>7.2} GB/s)   matmul_q4_gemv_reg {:>8.4} ms ({:>7.2} GB/s)   speedup {:.2}x",
            t_gemv * 1e3,
            gbs_gemv,
            t_reg * 1e3,
            gbs_reg,
            t_gemv / t_reg
        );
    }
}

/// Tiled/prefill-shaped: `matmul_q4_dyn` vs `matmul_q4_dyn_reg` across a
/// sweep of row counts at a fixed `k, n` close to a real model's
/// `d_model`/`intermediate` shape.
#[test]
#[ignore]
fn dyn_vs_dyn_reg_across_prefill_rows() {
    let g = gpu_core::testgpu::dev(KERNELS);
    if !g.caps().numeric.int8_dot {
        eprintln!("skipping: no packed int8 dot on this device");
        return;
    }
    let (k_maxr, k_qp, k_dyn, k_reg) =
        (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, "matmul_q4_dyn"), idx(&g, "matmul_q4_dyn_reg"));
    let (k, n) = (2048u32, 2048u32);
    println!("\nmatmul_q4_dyn vs matmul_q4_dyn_reg, k={k} n={n}, m swept:\n");
    for &m in &[32u32, 64, 128, 256, 512, 1024, 2048] {
        let mut rng = Lcg::new(9100 + u64::from(m));
        let x_h = rng.vec_scaled((m * k) as usize, 1.0);
        let w_h = rng.vec_scaled((n * k) as usize, 1.0);
        let x = g.storage_init("x", &x_h);
        let (xq, sx) = quant_x(&g, k_maxr, k_qp, &x, m, k);
        let (wq, sw) = quantize_weight_q4(&w_h, n as usize, k as usize);
        let wqb = g.storage(wq.len() as u64);
        g.write(&wqb, &wq);
        let swb = g.storage_init("sw", &sw);
        let out = g.storage((m * n) as u64);
        let tile_threads = m.div_ceil(128) * n.div_ceil(128) * 256;

        let st_dyn = vec![g.step(k_dyn, &[&xq, &wqb, &sx, &swb, &out], &[m, k, n], m * n)];
        let st_reg = vec![g.step(k_reg, &[&xq, &wqb, &sx, &swb, &out], &[m, k, n], tile_threads)];
        let t_dyn = gpu_core::profile::best_of(&g, &st_dyn, REPS);
        let t_reg = gpu_core::profile::best_of(&g, &st_reg, REPS);
        let int_ops = 8u64 * u64::from(m) * u64::from(k) * u64::from(n);
        let gops_dyn = int_ops as f64 / t_dyn / 1e9;
        let gops_reg = int_ops as f64 / t_reg / 1e9;
        println!(
            "m={m:>5}  matmul_q4_dyn {:>9.4} ms ({:>9.1} GOP/s)   matmul_q4_dyn_reg {:>9.4} ms ({:>9.1} GOP/s)   speedup {:.2}x",
            t_dyn * 1e3,
            gops_dyn,
            t_reg * 1e3,
            gops_reg,
            t_dyn / t_reg
        );
    }
}
