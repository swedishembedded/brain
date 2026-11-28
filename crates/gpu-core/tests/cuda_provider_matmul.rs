// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The first native CUDA kernel has to earn its place: it must be
//! **faster than the dispatch it displaces**, run at a stated fraction
//! of the device's own measured roof, and still agree with the portable WGSL
//! reference within the binary32 round-off bound of its reduction
//! (`brain_testutil::rounding`).
//!
//! Swedish Embedded AB implements measured performance contracts for GPU
//! compute stacks. If your team needs expertise in proving that a tuned
//! kernel is both faster AND still correct - rather than assuming it - you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # Why both halves, in one test
//!
//! A tuned kernel that is wrong is worthless, and a tuned kernel that is
//! right but no faster is worse than worthless: it is a second
//! implementation to maintain plus a `Tuned` tier claim the policy ratchet
//! will happily believe. Neither half alone is a reason to ship a kernel, so
//! neither half is asserted alone here.
//!
//! The reference side of both halves is the reference provider's `matmul`
//! dispatch as the SAME device serves it: what the tuned kernel displaces.
//! On a CUDA device that is not `wgsl_cuda`'s translation of `matmul.wgsl`
//! but the native register-tiled `matmul_f32_reg` `gpu_core::native_upgrade`
//! substitutes for it, bit for bit (`tests/f32_reg_native.rs`); the
//! translation itself runs only under `--no-native-kernels`. A ratio against
//! that side therefore measures one native kernel against another, and says
//! nothing card-independent; the floor is the tuned kernel's own fraction of
//! the device's ceiling, and against the displaced dispatch it must only be
//! faster. Comparing in-process means the two sides differ in exactly one
//! thing (which kernel ran), not in device, driver, allocation or input.
//!
//! # Skipped without a device, by design
//!
//! The card is an argument (`cargo test -p brain-gpu-core --test cuda_provider_matmul --
//! --device gpu1 --backend cuda`); without one nothing here opens a card.
//! `Gpu::try_new_cuda` returning `Err` is an ordinary outcome (most machines
//! are not NVIDIA machines) and every assertion here is about a real launch,
//! so there is nothing to assert without one - the same skip-if-absent shape
//! the rest of this repo's real-hardware tests use.
//!
//! # Deliberately small
//!
//! One timed shape sized to the device's multiprocessor count, a few
//! megabytes of device memory, a handful of milliseconds of device time. This
//! is a contract check, not a benchmark: the floor is the compute-bound
//! defect line against the device's measured roof, not a ratio taken on one
//! card.

use std::sync::Arc;

use backend_api::select::{self, Dtype, KernelSelector};
use backend_api::{DType, ImplSource};
use gpu_core::provider::cuda::{CudaProvider, SKINNY_MAX_ROWS, SKINNY_MIN_ROWS};
use gpu_core::provider::parity::{self, ParityCase, Tolerance};
use gpu_core::provider::{LowerCtx, Operand, OpRequest, OperatorProvider, Pass, Role};
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL)];

/// The tuned kernel's floor is a fraction of the device's own fp32 ceiling, measured by `gpu_core::roof`, at a
/// shape that fills that device: the one statement about a GEMM that means the same thing on every card. A ratio
/// against the displaced dispatch does not: it is bounded by how far below the ceiling that dispatch sits, which
/// depends on which kernel serves it on the device (a native upgrade or a mechanical translation) and on the card's
/// memory system. Where the displaced dispatch already reaches a large fraction of the ceiling, no tuned kernel can
/// clear a ratio taken where it reaches a small one.
///
/// The fraction is the compute-bound defect line (`roof::Bound::defect_pct`): below it a kernel is a defect, not
/// untuned. The tuned kernel must also beat the displaced dispatch at that shape outright.
// perf-number: the floor is derived from the device's measured roof; the fraction achieved is printed by the run.
fn roof_floor_pct() -> f32 {
    gpu_core::roof::Bound::Compute.defect_pct()
}

/// The fp32 rate a kernel that keeps the reference's rounding can reach, GFLOP/s: half the fused roof. The backend
/// compiles with `--fmad=false`, so the kernel's `acc + a * b` rounds twice, as the reference does, and issues a
/// multiply and an add where a fused kernel issues one instruction; the roof probe (`roof_fma.wgsl`) times explicit
/// fused multiply-adds, which is the rate of one instruction per two FLOPs.
fn unfused_ceiling_gflops(roofs: &gpu_core::roof::Roofs) -> f64 {
    f64::from(roofs.gflops) / 2.0
}

/// Output tiles (64 x 64, the tuned kernel's block) per multiprocessor the timed shape provides, so that the last
/// partial wave of blocks costs a small part of the run on any card: the shape follows the device, not one card.
const TILES_PER_UNIT: u32 = 8;

/// The reduction depth of the timed shape: long enough that each block's staged loads are amortized over many
/// multiply-adds, short enough that the run is a few milliseconds.
const TIMED_K: u32 = 512;

/// `(m, n, k)` for the timed comparison: square, with at least [`TILES_PER_UNIT`] output tiles per multiprocessor.
fn timed_shape(units: u32) -> (u32, u32, u32) {
    let side_tiles = ((TILES_PER_UNIT * units) as f64).sqrt().ceil() as u32;
    (side_tiles * 64, side_tiles * 64, TIMED_K)
}

/// The binary32 bound on the distance between two evaluations of `out = x @ W^T` (`x` is `[m, k]`, `w` is `[n, k]`),
/// each a length-`k` dot product summed in any order: `brain_testutil::rounding::agreement_bound` at that output's
/// own `S = sum |x_i w_i|`, the largest over the outputs (the bound grows with `S`, so the largest one covers all).
fn dot_agreement_bound(xs: &[f32], ws: &[f32], m: usize, n: usize, k: usize) -> f64 {
    brain_testutil::rounding::agreement_bound(k, max_abs_dot(xs, ws, m, n, k))
}

/// The largest `sum_i |x[r, i] * w[c, i]|` over every output `(r, c)`.
fn max_abs_dot(xs: &[f32], ws: &[f32], m: usize, n: usize, k: usize) -> f64 {
    let mut worst = 0f64;
    for r in 0..m {
        for c in 0..n {
            let s: f64 = (0..k).map(|i| f64::from((xs[r * k + i] * ws[c * k + i]).abs())).sum();
            worst = worst.max(s);
        }
    }
    worst
}

/// Timed repetitions per trial, and trials per side. The fastest trial is
/// compared, not the mean or the median: a machine shared with other work
/// produces outliers in one direction only.
const REPS: u32 = 8;
const TRIALS: usize = 5;

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed ^ 0xD1B5_4A32_D192_ED03;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// What one timed run of `REPS` identical dispatches cost.
struct Timed {
    /// Host wall time from the first submit to the synchronise. Includes every
    /// launch the host issues, so on a busy machine it measures how long the
    /// host thread waited for a core as much as the kernel.
    wall: std::time::Duration,
    /// Time the kernels spent executing on the device, from the driver's own
    /// events around each launch (`Gpu::kernel_times`). A host that lost its
    /// core between two launches leaves this unchanged.
    device: std::time::Duration,
}

/// The `x`, `w`, `out` buffers of one `(m, n, k)` product.
struct Product<'a> {
    shape: (u32, u32, u32),
    x: &'a backend_api::DeviceBuffer,
    w: &'a backend_api::DeviceBuffer,
    y: &'a backend_api::DeviceBuffer,
}

/// Lower one `Op::MatMul` through `p`, then submit it `REPS` times and
/// synchronise - `REPS` identical dispatches, timed on the host and on the
/// device.
fn time_provider(gpu: &Gpu, p: &dyn OperatorProvider, product: &Product) -> Timed {
    let (m, n, k) = product.shape;
    let (x, w, y) = (product.x, product.w, product.y);
    let caps = gpu.caps();
    let bind = |v: select::KernelVariant| -> (usize, &'static str) {
        assert_eq!(v, select::KernelVariant::Reference, "this test's selector is AlwaysReference");
        (gpu.kernel_index("matmul").expect("'matmul' is registered"), "matmul")
    };
    let operands = [
        Operand { role: Role::Act, buf: x, range: (0, (m as u64) * k as u64), dtype: DType::F32 },
        Operand { role: Role::Weight, buf: w, range: (0, 0), dtype: DType::F32 },
        Operand { role: Role::Out, buf: y, range: (0, (m as u64) * n as u64), dtype: DType::F32 },
    ];
    let attrs = [m, k, n];
    let req = OpRequest {
        op: select::Op::MatMul,
        shape: select::OpShape { m, n, k, dtype: Dtype::F32 },
        pass: Pass::Forward,
        operands: &operands,
        attrs: &attrs,
        group: 32,
        bind: &bind,
    };
    let mut steps = Vec::new();
    {
        let mut ctx = LowerCtx { gpu, caps: &caps, steps: &mut steps, capture: false };
        p.lower(&mut ctx, &req).expect("both providers must lower this request");
    }
    // Warm up: the first dispatch of a kernel is where this backend compiles
    // it, and a compile in the timed region would be measuring NVRTC.
    gpu.submit(&[], &steps);
    gpu.poll_wait();

    assert!(gpu.set_kernel_timing(true), "the CUDA backend times kernels on the device");
    gpu.reset_kernel_times();
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        gpu.submit(&[], &steps);
    }
    gpu.poll_wait();
    let wall = t0.elapsed();
    // Only this provider's kernel ran since the reset, so the sum is its time.
    let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().map(|(_, ms, _)| ms).sum();
    gpu.set_kernel_timing(false);
    Timed { wall, device: std::time::Duration::from_secs_f64(ms / 1e3) }
}

/// The fastest run of each measure. On a card another process is using, a
/// neighbour's time slice only ever LENGTHENS a run, so the minimum is the
/// statistic it cannot inflate; a median of a few runs swung the measured
/// speedup widely on one shared card within minutes.
fn fastest(runs: &[Timed]) -> Timed {
    Timed {
        wall: runs.iter().map(|t| t.wall).min().expect("at least one run"),
        device: runs.iter().map(|t| t.device).min().expect("at least one run"),
    }
}

/// **The milestone's red test.** One native CUDA `Op::MatMul` kernel,
/// selected against the compute capability the driver was ASKED for, must
/// reach [`roof_floor_pct`] of the device's measured fp32 roof at a shape that
/// fills the device, beat the dispatch it displaces there, and still agree with it
/// within the binary32 bound of the reduction.
fn the_tuned_cuda_matmul_clears_its_roof_floor_and_agrees_with_the_dispatch_it_displaces() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        brain_testutil::skip_unavailable("cuda_provider_matmul: no CUDA device on this box");
        return;
    };
    let caps = gpu.caps();
    let cc = caps.arch.compute_capability.expect(
        "the CUDA backend must report the compute capability it queried - every tier decision keys on it",
    );
    let provider = CudaProvider::for_gpu(&gpu).expect("a CUDA handle must yield a CUDA provider");
    eprintln!("cuda_provider_matmul: compute capability {}.{}", cc.0, cc.1);

    let roofs = gpu_core::roof::ensure(&gpu).expect("a CUDA device's fp32 roof is measurable");
    let units = caps.compute_units.expect("a CUDA device reports its multiprocessors");
    let (m, n, k) = timed_shape(units);
    let shape = select::OpShape { m, n, k, dtype: Dtype::F32 };
    let scratch = gpu.storage(1);
    let bind = |_: select::KernelVariant| -> (usize, &'static str) { (0, "matmul") };
    let probe_operands = [
        Operand { role: Role::Act, buf: &scratch, range: (0, 1), dtype: DType::F32 },
        Operand { role: Role::Weight, buf: &scratch, range: (0, 0), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &scratch, range: (0, 1), dtype: DType::F32 },
    ];
    let probe = OpRequest {
        op: select::Op::MatMul,
        shape,
        pass: Pass::Forward,
        operands: &probe_operands,
        attrs: &[],
        group: 32,
        bind: &bind,
    };
    assert!(provider.accepts(&probe, false), "a plain f32 MatMul is what this provider exists for");
    assert_eq!(provider.source(&probe), ImplSource::Tuned, "a native kernel is the Tuned tier");
    eprintln!("cuda_provider_matmul: tuned kernel {:?}", provider.kernel_name(select::Op::MatMul, Dtype::F32));

    let (xs, ws) = (seeded(1, (m * k) as usize), seeded(2, (n * k) as usize));
    let x = gpu.storage_init("x", &xs);
    let w = gpu.storage_init("w", &ws);
    let y_ref = gpu.storage((m as u64) * n as u64);
    let y_got = gpu.storage((m as u64) * n as u64);
    let displaced = Product { shape: (m, n, k), x: &x, w: &w, y: &y_ref };
    let tuned = Product { shape: (m, n, k), x: &x, w: &w, y: &y_got };

    let reference_selector: Arc<dyn KernelSelector> =
        Arc::new(select::CachedSelector::new(select::AlwaysReference));
    let reference = gpu_core::provider::wgsl::WgslProvider::new(reference_selector);

    let mut t_ref = Vec::new();
    let mut t_got = Vec::new();
    for _ in 0..TRIALS {
        t_ref.push(time_provider(&gpu, &reference, &displaced));
        t_got.push(time_provider(&gpu, &provider, &tuned));
    }
    let (t_ref, t_got) = (fastest(&t_ref), fastest(&t_got));
    let speedup = t_ref.device.as_secs_f64() / t_got.device.as_secs_f64();
    let flops = 2 * u64::from(m) * u64::from(n) * u64::from(k) * u64::from(REPS);
    let ceiling = unfused_ceiling_gflops(&roofs);
    let pct = |t: &Timed| 100.0 * flops as f64 / t.device.as_secs_f64() / 1e9 / ceiling;
    eprintln!(
        "cuda_provider_matmul: {m}x{n}x{k} f32, {REPS} dispatches - device time: displaced {:?} ({:.1}% of the {:.0} GFLOP/s unfused ceiling), \
         tuned {:?} ({:.1}%), speedup {speedup:.2}x (host wall time: displaced {:?}, tuned {:?}, {:.2}x)",
        t_ref.device,
        pct(&t_ref),
        ceiling,
        t_got.device,
        pct(&t_got),
        t_ref.wall,
        t_got.wall,
        t_ref.wall.as_secs_f64() / t_got.wall.as_secs_f64()
    );

    let expect = gpu.read(&y_ref, (m * n) as usize);
    let got = gpu.read(&y_got, (m * n) as usize);
    let maxabs = expect.iter().zip(got.iter()).map(|(a, b)| f64::from((a - b).abs())).fold(0f64, f64::max);
    let bar = dot_agreement_bound(&xs, &ws, m as usize, n as usize, k as usize);
    eprintln!("cuda_provider_matmul: maxabs vs the displaced dispatch {maxabs:e} over {} outputs (bound {bar:e})", expect.len());

    assert!(maxabs <= bar, "tuned CUDA matmul disagrees with the dispatch it displaces by {maxabs:e} (bound {bar:e})");
    assert!(speedup > 1.0, "tuned CUDA matmul is {speedup:.2}x the dispatch it displaces: not faster");
    let floor = f64::from(roof_floor_pct());
    assert!(
        pct(&t_got) >= floor,
        "tuned CUDA matmul reaches {:.1}% of this device's unfused fp32 ceiling at {m}x{n}x{k} (floor {floor:.0}%) - \
         a native kernel below the defect line is a maintenance cost with a Tuned tier claim attached",
        pct(&t_got)
    );
}

/// **The production wiring.** `ProviderRegistry::for_gpu` is what every
/// model's `Ops` is built with, so this is the assertion that a real forward
/// pass on this device reaches the native kernel - not merely that the
/// provider works when a test constructs one by hand.
///
/// Asserted from the DISPATCH RECORD rather than from the output, because
/// every tier computes the same numbers: which tier answered is exactly the
/// fact nothing else can observe.
fn the_production_registry_routes_a_real_matmul_to_the_tuned_kernel() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        brain_testutil::skip_unavailable("cuda_provider_matmul: no CUDA device on this box");
        return;
    };
    let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
    let registry = gpu_core::provider::ProviderRegistry::for_gpu(&gpu, selector);

    // Wide enough that the 64 x 64 tile fills the device (the narrow shapes belong to the skinny kernel, below).
    let (m, n, k) = (512u32, 1024u32, 64u32);
    let x = gpu.storage_init("x", &seeded(7, (m * k) as usize));
    let w = gpu.storage_init("w", &seeded(8, (n * k) as usize));
    let y = gpu.storage((m * n) as u64);
    let bind = |_: select::KernelVariant| -> (usize, &'static str) {
        (gpu.kernel_index("matmul").expect("'matmul' is registered"), "matmul")
    };
    let operands = [
        Operand { role: Role::Act, buf: &x, range: (0, (m * k) as u64), dtype: DType::F32 },
        Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
        Operand { role: Role::Out, buf: &y, range: (0, (m * n) as u64), dtype: DType::F32 },
    ];
    let attrs = [m, k, n];
    let req = OpRequest {
        op: select::Op::MatMul,
        shape: select::OpShape { m, n, k, dtype: Dtype::F32 },
        pass: Pass::Forward,
        operands: &operands,
        attrs: &attrs,
        group: 32,
        bind: &bind,
    };
    let caps = gpu.caps();
    let mut steps = Vec::new();
    let lowered = {
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        registry.dispatch(&mut ctx, &req)
    };
    let choice = lowered.choice.expect("ProviderRegistry::dispatch always attaches its record");
    assert_eq!(choice.provider, "cuda", "the production chain must reach the CUDA provider, declines: {:?}", choice.declined);
    assert_eq!(choice.source, ImplSource::Tuned);
    assert_eq!(
        choice.arch.compute_capability,
        caps.arch.compute_capability,
        "the dispatch record must carry the capability this device was QUERIED for"
    );
    assert_eq!(lowered.kernels, vec!["native:matmul_f32_tiled"]);
    assert_eq!(lowered.pushed, 1);

    // And it computes the right thing at this shape, not merely the right
    // tier - a record naming a kernel that produced garbage is worse than no
    // record at all.
    gpu.submit(&[], &steps);
    let got = gpu.read(&y, (m * n) as usize);
    let xs = seeded(7, (m * k) as usize);
    let ws = seeded(8, (n * k) as usize);
    let mut maxabs = 0f64;
    for r in 0..m as usize {
        for c in 0..n as usize {
            let want: f64 = (0..k as usize).map(|i| f64::from(xs[r * k as usize + i]) * f64::from(ws[c * k as usize + i])).sum();
            maxabs = maxabs.max((want - f64::from(got[r * n as usize + c])).abs());
        }
    }
    let bar = brain_testutil::rounding::accumulation_bound(k as usize, max_abs_dot(&xs, &ws, m as usize, n as usize, k as usize));
    assert!(maxabs <= bar, "production dispatch is {maxabs:e} from an f64 oracle (bound {bar:e})");
}

/// The shared cross-provider parity harness, driven by the tuned provider
/// over the fixed `Op::MatMul` case table - (decode-shaped cases are asserted
/// declined, see below), the tile
/// crossover, a multi-tile shape, a non-tile-multiple shape and a non-zero
/// row offset. Same oracle (the WGSL reference provider), same seeded
/// inputs, tolerance widened from the table's own `BitIdentical` to the
/// binary32 agreement bound of a length-`k` dot product.
fn the_tuned_cuda_matmul_clears_every_shared_parity_case() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        brain_testutil::skip_unavailable("cuda_provider_matmul: no CUDA device on this box");
        return;
    };
    let provider = CudaProvider::for_gpu(&gpu).expect("a CUDA handle must yield a CUDA provider");
    let scratch = gpu.storage(1);
    for case in parity::cases_for(select::Op::MatMul) {
        let case = ParityCase {
            // The skinny kernel splits the contraction over warps, so it differs from the reference's single ascending
            // sum by the summation error: the agreement bound of `k` terms, each `|x w| <= 1` because the harness
            // seeds every input in [-1, 1), so `S <= k`.
            tol: Tolerance::Numeric {
                atol: brain_testutil::rounding::agreement_bound(case.shape.k as usize, f64::from(case.shape.k)) as f32,
                rtol: 0.0,
                cosine_min: 0.999_999,
            },
            ..*case
        };
        let bind = |_: select::KernelVariant| -> (usize, &'static str) { (0, "matmul") };
        // The operand ROLES/dtypes are what `accepts` gates on, so the probe
        // has to carry the same three-f32 bundle the harness itself binds -
        // an empty list would be refused for the wrong reason and prove
        // nothing about the shape.
        let operands = [
            Operand { role: Role::Act, buf: &scratch, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: &scratch, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &scratch, range: (0, 1), dtype: DType::F32 },
        ];
        let probe = OpRequest {
            op: case.op,
            shape: case.shape,
            pass: case.pass,
            operands: &operands,
            attrs: &[],
            group: 32,
            bind: &bind,
        };
        if case.shape.m <= select::DECODE_REGIME_MAX_ROWS {
            // The decode regime belongs to the skinny-M GEMV: the 64x64 tile
            // would run a handful of blocks for a one-row output.
            assert!(!provider.accepts(&probe, false), "the tiled kernel must leave decode-regime {:?} to the GEMV", case.shape);
            continue;
        }
        assert!(
            provider.accepts(&probe, false),
            "the tuned provider must take every prefill-regime MatMul parity case at f32, including {:?}",
            case.shape
        );
        parity::assert_provider_parity(&gpu, &provider, &case);
    }
}

/// **A few rows against a short weight** run on the narrow-tile kernel and match an f64 oracle: the decoder step's
/// shapes at batch 33..128 (24 x 2048 gates, 512-row projections), tails in every dimension, and a shape wide enough
/// that the 64 x 64 tile fills the device and keeps its kernel.
fn a_starved_tile_grid_runs_the_skinny_kernel_and_agrees_with_an_f64_oracle() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        brain_testutil::skip_unavailable("cuda_provider_matmul: no CUDA device on this box");
        return;
    };
    let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
    let registry = gpu_core::provider::ProviderRegistry::for_gpu(&gpu, selector);
    let caps = gpu.caps();
    let units = caps.compute_units.expect("a CUDA device reports its multiprocessors");
    let bind = |_: select::KernelVariant| -> (usize, &'static str) { (gpu.kernel_index("matmul").expect("'matmul' is registered"), "matmul") };
    let cases = [(33u32, 24u32, 2048u32), (64, 24, 2048), (64, 512, 512), (64, 1120, 512), (100, 7, 33), (128, 384, 31), (48, 1, 1), (65, 8, 32), (64, 16384, 512), (16, 512, 512), (24, 24, 2048), (32, 1120, 512)];
    for (m, n, k) in cases {
        let xs = seeded(11 + u64::from(m), (m * k) as usize);
        let ws = seeded(12 + u64::from(n), (n * k) as usize);
        let x = gpu.storage_init("x", &xs);
        let w = gpu.storage_init("w", &ws);
        let y = gpu.storage((m * n) as u64);
        let operands = [
            Operand { role: Role::Act, buf: &x, range: (0, u64::from(m * k)), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: &w, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &y, range: (0, u64::from(m * n)), dtype: DType::F32 },
        ];
        let attrs = [m, k, n];
        let req = OpRequest { op: select::Op::MatMul, shape: select::OpShape { m, n, k, dtype: Dtype::F32 }, pass: Pass::Forward, operands: &operands, attrs: &attrs, group: 32, bind: &bind };
        let mut steps = Vec::new();
        let lowered = {
            let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
            registry.dispatch(&mut ctx, &req)
        };
        let starved = (SKINNY_MIN_ROWS..=SKINNY_MAX_ROWS).contains(&m) && m.div_ceil(64) * n.div_ceil(64) < units / 2;
        // A decode-regime product whose tile grid fills the device is the portable decode GEMV's (`matmul` through
        // this test's binding): the tiled kernel declines it, and the skinny one is claimed only for a starved grid.
        let want_kernel = match (starved, m > select::DECODE_REGIME_MAX_ROWS) {
            (true, _) => "native:matmul_f32_skinny",
            (false, true) => "native:matmul_f32_tiled",
            (false, false) => "matmul",
        };
        assert_eq!(lowered.kernels, vec![want_kernel], "m={m} n={n} k={k} on {units} multiprocessors");
        gpu.submit(&[], &steps);
        let got = gpu.read(&y, (m * n) as usize);
        // Each output against its own binary32 bound, `gamma_k * sum |x_i w_i|`: the error of a sum grows with the
        // magnitude of its terms, so one bar for every output is too tight for the large ones or too loose for the rest.
        let mut worst: Option<(f64, f64)> = None;
        for r in 0..m as usize {
            for c in (0..n as usize).step_by(if n > 4096 { 97 } else { 1 }) {
                let terms = (0..k as usize).map(|i| f64::from(xs[r * k as usize + i]) * f64::from(ws[c * k as usize + i]));
                let (want, s) = terms.fold((0f64, 0f64), |(sum, s), t| (sum + t, s + t.abs()));
                let err = (want - f64::from(got[r * n as usize + c])).abs();
                let bar = brain_testutil::rounding::accumulation_bound(k as usize, s);
                if worst.is_none_or(|(e, b)| err * b > e * bar) {
                    worst = Some((err, bar));
                }
            }
        }
        let (err, bar) = worst.expect("every case has an output");
        assert!(err <= bar, "m={m} n={n} k={k}: {err:e} from an f64 oracle (bound {bar:e})");
    }
}

gpu_core::card_tests!(
    the_tuned_cuda_matmul_clears_its_roof_floor_and_agrees_with_the_dispatch_it_displaces,
    the_production_registry_routes_a_real_matmul_to_the_tuned_kernel,
    the_tuned_cuda_matmul_clears_every_shared_parity_case,
    a_starved_tile_grid_runs_the_skinny_kernel_and_agrees_with_an_f64_oracle,
);

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before any test runs.
brain_testutil::configure_test_binary!();
