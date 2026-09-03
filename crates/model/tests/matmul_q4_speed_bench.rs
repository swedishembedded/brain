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
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::int4::quantize_weight_q4;
use model::ops::{Ops, Weight};

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

/// Quantize activations on-device via the existing int8 path (`max_abs_row` +
/// `quant_pack`) - the same helper `crates/model/tests/matmul_q4_gemm.rs`
/// uses, duplicated here rather than shared across `tests/` binaries (each
/// integration test file is its own crate).
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

/// GEMV-shaped, at qwen35's own real decode leaf shapes rather than a
/// generic `d_model` stand-in - `gemv_vs_gemv_reg_across_decode_rows` above
/// answers "does `_reg` win at all", this answers "does it win at the exact
/// shapes the qwen35 Q4 tier (M24) would actually dispatch". `m=1` only:
/// `qwen35::serve::Engine`'s `DECODE_WINDOW_CAPACITY` is pinned to 1 (GDN
/// state has no snapshot/restore), so decode never batches rows.
#[test]
#[ignore]
fn gemv_vs_gemv_reg_at_qwen35_decode_shapes() {
    const SHAPES: &[(&str, u32, u32)] = &[
        ("mlp.gate/up", 5120, 17408),
        ("mlp.down", 17408, 5120),
        ("gdn.in_proj_qkv", 5120, 10240),
        ("gqa.q_proj", 5120, 12288),
        ("gqa.o_proj/gdn.out_proj", 6144, 5120),
    ];
    let m: u32 = 1;
    let bucket = smallest_bucket_covering(m);
    let variant = kernels::template::interned("matmul_q4_gemv_reg", kernels::MATMUL_Q4_GEMV_REG, &[("MREG", bucket)]).unwrap();
    let mut kernels_with_bucket = KERNELS.to_vec();
    kernels_with_bucket.push(variant);
    let g = gpu_core::Gpu::new(&kernels_with_bucket);
    if !g.caps().numeric.int8_dot {
        eprintln!("skipping: no packed int8 dot on this device");
        return;
    }
    let (k_maxr, k_qp, k_gemv) = (idx(&g, "max_abs_row"), idx(&g, "quant_pack"), idx(&g, "matmul_q4_gemv"));
    let k_reg = idx(&g, variant.0);
    println!("\nmatmul_q4_gemv vs matmul_q4_gemv_reg#MREG={bucket} at qwen35 decode shapes, m=1:\n");
    let mut any_reg_loses = false;
    for &(label, k, n) in SHAPES {
        let mut rng = Lcg::new(9200 + u64::from(k) + u64::from(n));
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
        let bytes = u64::from(n) * u64::from(k) / 8 + u64::from(n) * u64::from(k) / 32 * 4;
        let gbs_gemv = bytes as f64 / t_gemv / 1e9;
        let gbs_reg = bytes as f64 / t_reg / 1e9;
        let speedup = t_gemv / t_reg;
        if speedup < 1.0 {
            any_reg_loses = true;
        }
        println!(
            "{label:<26} k={k:>6} n={n:>6}  gemv {:>8.4} ms ({:>7.2} GB/s)   gemv_reg {:>8.4} ms ({:>7.2} GB/s)   speedup {speedup:.2}x{}",
            t_gemv * 1e3,
            gbs_gemv,
            t_reg * 1e3,
            gbs_reg,
            if speedup < 1.0 { "  <-- reg LOSES" } else { "" }
        );
    }
    if any_reg_loses {
        println!(
            "\nRESULT: matmul_q4_gemv_reg does NOT win at every qwen35 decode shape - \
             do not add it to gpu_core::upgrade's UPGRADES table blindly; measure the \
             plain matmul_q4_gemv path for M24's tok/s claim instead."
        );
    } else {
        println!(
            "\nRESULT: matmul_q4_gemv_reg wins at every qwen35 decode shape - safe to add \
             the UPGRADES row (gpu_core::upgrade.rs) gating qwen35's Q4 tier onto it."
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

/// The FACADE measurement `dyn_vs_dyn_reg_across_prefill_rows` above does not
/// take: that test dispatches the raw `matmul_q4_dyn`/`matmul_q4_dyn_reg`
/// kernel names directly, never through `model::ops::Ops::matmul`. This
/// builds a real `Ops`, uploads a real `Weight::Q4`, and times `Ops::matmul`
/// itself (activation already quantized outside the timed region, exactly
/// like the raw-kernel bench above isolates matmul-only cost) at qwen35's own
/// prefill-relevant shapes and `m=128` - M26's own `MAX_PREFILL_TOKENS`
/// chunk size on the real two-card 27B resident. `Ops::matmul_kernel` also
/// asserts which physical kernel the facade actually chose, so a future
/// regression in `Ops::bind`'s `(PackedInt8, Q4)` arm shows up here as a
/// changed kernel name, not just a changed number.
#[test]
#[ignore]
fn ops_facade_confirms_the_dyn_reg_speedup_at_qwen35_prefill_shapes() {
    let g = gpu_core::testgpu::dev(model::ops::kernel_list());
    if !g.caps().numeric.int8_dot {
        eprintln!("skipping: no packed int8 dot on this device");
        return;
    }
    let ops = Ops::new(g).expect("Ops::new: canonical kernel_list() must satisfy REQUIRED_KERNELS");
    let gpu = ops.gpu();
    let (k_maxr, k_qp, k_dyn) = (idx(gpu, "max_abs_row"), idx(gpu, "quant_pack"), idx(gpu, "matmul_q4_dyn"));

    const SHAPES: &[(&str, u32, u32)] = &[
        ("mlp.gate/up", 5120, 17408),
        ("mlp.down", 17408, 5120),
        ("gdn.in_proj_qkv", 5120, 10240),
        ("gqa.q_proj", 5120, 12288),
        ("gqa.o_proj/gdn.out_proj", 6144, 5120),
    ];
    let m: u32 = 128; // M26's MAX_PREFILL_TOKENS chunk size on the real resident.
    println!("\nOps::matmul (Q4) at qwen35 prefill shapes, m={m}, chunk-sized like M26's real resident:\n");
    for &(label, k, n) in SHAPES {
        let mut rng = Lcg::new(9300 + u64::from(k) + u64::from(n));
        let x_h = rng.vec_scaled((m * k) as usize, 1.0);
        let w_h = rng.vec_scaled((n * k) as usize, 1.0);
        let x = gpu.storage_init("x", &x_h);
        let weight = Weight::upload(&ops, &w_h, n as usize, k as usize, Dtype::Q4);
        assert_eq!(weight.dtype(), Dtype::Q4, "this device must support int8_dot for this bench to mean anything");

        let chosen = ops.matmul_kernel(&weight, m);
        assert_eq!(
            chosen,
            "matmul_q4_dyn_reg",
            "Ops::matmul_kernel chose {chosen:?} at m={m} -- the facade must dispatch matmul_q4_dyn_reg, \
             not the naive matmul_q4_dyn, for this to be the speedup this bench measures"
        );

        // Quantize the activation ONCE, outside the timed region - the same
        // isolation the raw-kernel bench above applies. `ops.act` builds its
        // OWN xq/sx internally (private to `Act`), so the raw-kernel
        // baseline below quantizes independently via the same
        // `max_abs_row`/`quant_pack` pair on the identical `x` buffer -
        // deterministic, so both sides see the same quantized activation.
        let mut setup_steps = Vec::new();
        let act = ops.act(&mut setup_steps, &x, 0, m, k);
        gpu.submit(&[], &setup_steps);

        let mut facade_steps = Vec::new();
        let out = gpu.storage((m * n) as u64);
        ops.matmul(&mut facade_steps, &weight, &act, &out, 0);
        let t_facade = gpu_core::profile::best_of(gpu, &facade_steps, REPS);

        // Raw-kernel baseline: the naive `matmul_q4_dyn` at its own m*n
        // dispatch geometry, driven by hand on an independently-quantized
        // copy of the SAME `x`/weight bytes - the exact comparison
        // `dyn_vs_dyn_reg_across_prefill_rows` makes, repeated here at
        // qwen35's real shapes so the facade's win is checked against real
        // data, not assumed to transfer from a separate run.
        let (xq, sx) = quant_x(gpu, k_maxr, k_qp, &x, m, k);
        let (wq, sw) = quantize_weight_q4(&w_h, n as usize, k as usize);
        let wqb = gpu.storage(wq.len() as u64);
        gpu.write(&wqb, &wq);
        let swb = gpu.storage_init("sw_naive", &sw);
        let out_naive = gpu.storage((m * n) as u64);
        let st_naive = vec![gpu.step(k_dyn, &[&xq, &wqb, &sx, &swb, &out_naive], &[m, k, n], m * n)];
        let t_naive = gpu_core::profile::best_of(gpu, &st_naive, REPS);

        let speedup = t_naive / t_facade;
        println!(
            "{label:<26} k={k:>6} n={n:>6}  Ops::matmul(reg) {:>9.4} ms   raw matmul_q4_dyn {:>9.4} ms   speedup {speedup:.2}x",
            t_facade * 1e3,
            t_naive * 1e3,
        );
    }
}
