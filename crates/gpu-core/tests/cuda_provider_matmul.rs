// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The first hand-written CUDA kernel has to earn its place: it must be
//! **faster than the generated tier it displaces** and it must still agree
//! with the portable WGSL reference to the absolute bar this project asserts
//! everywhere else.
//!
//! Swedish Embedded AB implements measured performance contracts for GPU
//! compute stacks. If your team needs expertise in proving that a hand-tuned
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
//! The reference side of both halves is the SAME device's generated tier -
//! `wgsl_cuda`'s translation of `matmul.wgsl`, which
//! `backend-cuda`'s own golden gate already holds to `wgsl-cpu`. Comparing
//! against it in-process means the two sides differ in exactly one thing
//! (which kernel ran), not in device, driver, allocation or input.
//!
//! # Skipped without a device, by design
//!
//! `Gpu::try_new_cuda` returning `Err` is an ordinary outcome (most machines
//! are not NVIDIA machines) and every assertion here is about a real launch,
//! so there is nothing to assert without one - the same skip-if-absent shape
//! the rest of this repo's real-hardware tests use.
//!
//! # Deliberately small
//!
//! One shape, a few hundred kilobytes of device memory, a handful of
//! milliseconds of device time. This is a contract check, not a benchmark:
//! the floor below is set well under what was measured so that it states
//! "the tuned tier is materially faster", not "this box was idle".

use std::sync::Arc;

use backend_api::select::{self, Dtype, KernelSelector};
use backend_api::{DType, ImplSource};
use gpu_core::provider::cuda::CudaProvider;
use gpu_core::provider::parity::{self, ParityCase, Tolerance};
use gpu_core::provider::{LowerCtx, Operand, OpRequest, OperatorProvider, Pass, Role};
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL)];

/// How much faster than the generated tier the tuned kernel must be, as a
/// ratio of wall time for the identical dispatch.
///
/// Set BELOW the ratio actually measured, on purpose. The number this gate
/// has to defend is "a hand-written kernel replaced a mechanical translation
/// and it was worth doing"; a floor pinned to one box's best case would
/// instead fail whenever that box is busy, which says nothing about the
/// kernel, and this one was measured on a machine also running an unrelated
/// job on the same cards. The ratio is far more stable under that contention
/// than either side's absolute time, because both sides contend equally.
/// What the run actually reports is printed, so the margin is never a
/// mystery.
// perf-number: the asserted floor of this gate IS this constant, and the
// ratio actually achieved against it is printed by the run itself.
const SPEEDUP_FLOOR: f64 = 8.0;

/// The absolute agreement bar this project asserts across backends
/// (`deepseek2/tests/backend_parity.rs`, `gpt2/tests/cuda_backend_parity.rs`).
/// Bit-identity is deliberately NOT the bar: a tuned kernel is free to
/// reassociate, and the moment one does, a zero-delta assertion would have to
/// be weakened under pressure rather than having been honest from the start.
const MAXABS: f32 = 1e-6;

/// `(m, n, k)` for the timed comparison. Small enough that the whole run is a
/// few milliseconds and under a megabyte of device memory, large enough that
/// the generated tier's uncoalesced weight reads actually cost something.
const SHAPE: (u32, u32, u32) = (512, 512, 512);

/// Timed repetitions per trial, and trials per side. The median of the
/// trials is compared, not the mean: a machine shared with other work
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

/// Lower one `Op::MatMul` through `p`, then submit it `REPS` times and
/// synchronise - the wall time of `REPS` identical dispatches.
fn time_provider(gpu: &Gpu, p: &dyn OperatorProvider, x: &backend_api::DeviceBuffer, w: &backend_api::DeviceBuffer, y: &backend_api::DeviceBuffer) -> std::time::Duration {
    let (m, n, k) = SHAPE;
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

    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        gpu.submit(&[], &steps);
    }
    gpu.poll_wait();
    t0.elapsed()
}

fn median(mut v: Vec<std::time::Duration>) -> std::time::Duration {
    v.sort();
    v[v.len() / 2]
}

/// **The milestone's red test.** One hand-written CUDA `Op::MatMul` kernel,
/// selected against the compute capability the driver was ASKED for, must
/// clear a measured speedup floor against the generated tier and still agree
/// with it to `MAXABS`.
#[test]
fn the_tuned_cuda_matmul_is_faster_than_the_generated_tier_and_still_agrees_with_it() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        eprintln!("cuda_provider_matmul: no CUDA device on this box - skipping");
        return;
    };
    let caps = gpu.caps();
    let cc = caps.arch.compute_capability.expect(
        "the CUDA backend must report the compute capability it queried - every tier decision keys on it",
    );
    let provider = CudaProvider::for_gpu(&gpu).expect("a CUDA handle must yield a CUDA provider");
    eprintln!("cuda_provider_matmul: compute capability {}.{}", cc.0, cc.1);

    let (m, n, k) = SHAPE;
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
        bind: &bind,
    };
    assert!(provider.accepts(&probe, false), "a plain f32 MatMul is what this provider exists for");
    assert_eq!(provider.source(&probe), ImplSource::Tuned, "a hand-written kernel is the Tuned tier");
    eprintln!("cuda_provider_matmul: tuned kernel {:?}", provider.kernel_name(select::Op::MatMul));

    let x = gpu.storage_init("x", &seeded(1, (m * k) as usize));
    let w = gpu.storage_init("w", &seeded(2, (n * k) as usize));
    let y_ref = gpu.storage((m as u64) * n as u64);
    let y_got = gpu.storage((m as u64) * n as u64);

    let reference_selector: Arc<dyn KernelSelector> =
        Arc::new(select::CachedSelector::new(select::AlwaysReference));
    let reference = gpu_core::provider::wgsl::WgslProvider::new(reference_selector);

    let mut t_ref = Vec::new();
    let mut t_got = Vec::new();
    for _ in 0..TRIALS {
        t_ref.push(time_provider(&gpu, &reference, &x, &w, &y_ref));
        t_got.push(time_provider(&gpu, &provider, &x, &w, &y_got));
    }
    let (t_ref, t_got) = (median(t_ref), median(t_got));
    let speedup = t_ref.as_secs_f64() / t_got.as_secs_f64();
    eprintln!(
        "cuda_provider_matmul: {m}x{n}x{k} f32, {REPS} dispatches - generated {:?}, tuned {:?}, speedup {speedup:.2}x",
        t_ref, t_got
    );

    let expect = gpu.read(&y_ref, (m * n) as usize);
    let got = gpu.read(&y_got, (m * n) as usize);
    let maxabs = expect.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("cuda_provider_matmul: maxabs vs the generated tier {maxabs:e} over {} outputs", expect.len());

    assert!(maxabs < MAXABS, "tuned CUDA matmul disagrees with the generated tier by {maxabs:e} (bar {MAXABS:e})");
    assert!(
        speedup >= SPEEDUP_FLOOR,
        "tuned CUDA matmul is only {speedup:.2}x the generated tier (floor {SPEEDUP_FLOOR:.2}x) - \
         a hand-written kernel that is not materially faster is a maintenance cost with a Tuned tier claim attached"
    );
}

/// **The production wiring.** `ProviderRegistry::for_gpu` is what every
/// model's `Ops` is built with, so this is the assertion that a real forward
/// pass on this device reaches the hand-written kernel - not merely that the
/// provider works when a test constructs one by hand.
///
/// Asserted from the DISPATCH RECORD rather than from the output, because
/// every tier computes the same numbers: which tier answered is exactly the
/// fact nothing else can observe.
#[test]
fn the_production_registry_routes_a_real_matmul_to_the_tuned_kernel() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        eprintln!("cuda_provider_matmul: no CUDA device on this box - skipping");
        return;
    };
    let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
    let registry = gpu_core::provider::ProviderRegistry::for_gpu(&gpu, selector);

    let (m, n, k) = (64u32, 64u32, 64u32);
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
    let mut maxabs = 0f32;
    for r in 0..m as usize {
        for c in 0..n as usize {
            let want: f32 = (0..k as usize).map(|i| xs[r * k as usize + i] * ws[c * k as usize + i]).sum();
            maxabs = maxabs.max((want - got[r * n as usize + c]).abs());
        }
    }
    assert!(maxabs < MAXABS, "production dispatch disagrees with a host oracle by {maxabs:e}");
}

/// The shared cross-provider parity harness, driven by the tuned provider
/// over the fixed `Op::MatMul` case table - decode-shaped, the tile
/// crossover, a multi-tile shape, a non-tile-multiple shape and a non-zero
/// row offset. Same oracle (the WGSL reference provider), same seeded
/// inputs, tolerance widened from the table's own `BitIdentical` to the
/// absolute `MAXABS` bar a tuned kernel is held to.
#[test]
fn the_tuned_cuda_matmul_clears_every_shared_parity_case() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        eprintln!("cuda_provider_matmul: no CUDA device on this box - skipping");
        return;
    };
    let provider = CudaProvider::for_gpu(&gpu).expect("a CUDA handle must yield a CUDA provider");
    let scratch = gpu.storage(1);
    for case in parity::cases_for(select::Op::MatMul) {
        let case = ParityCase {
            tol: Tolerance::Numeric { atol: MAXABS, rtol: 0.0, cosine_min: 0.999_999 },
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
            bind: &bind,
        };
        assert!(
            provider.accepts(&probe, false),
            "the tuned provider must take every shared MatMul parity case at f32, including {:?}",
            case.shape
        );
        parity::assert_provider_parity(&gpu, &provider, &case);
    }
}
