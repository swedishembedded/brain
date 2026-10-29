// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The int8 prefill GEMM on tensor cores must be **routed to by the production
//! registry**, must **agree with the portable DP4A kernel it displaces** on
//! the real Qwen3.8-27B GEMM shapes, and must be **much faster** than it.
//!
//! Swedish Embedded AB implements measured performance contracts for GPU
//! compute stacks. If your team needs expertise in proving that a tensor-core
//! kernel is both faster AND still numerically faithful to the portable path
//! it replaces, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What "agrees" means here
//!
//! Both kernels compute the same integer group sums exactly (int8 x int8 into
//! int32 is associative), so they can differ only in how the f32 running
//! total across groups is rounded: the portable kernel's separate multiply and
//! add (the backend compiles with `--fmad=false`) against the tensor-core
//! kernel's single fused multiply-add. The bar is therefore stated relative to
//! the output's own scale ([`REL_TOL`] of the row-wise RMS), not as an absolute
//! number that would silently stop meaning anything at another weight scale.
//! An independent f64 host oracle over a sample of outputs checks both kernels
//! against the truth, so "agrees with the reference" cannot hide a bug the two
//! share.
//!
//! # Skipped without a device, by design
//!
//! `Gpu::try_new_cuda` returning `Err` is an ordinary outcome, and every
//! assertion here is about a real launch.
//!
//! The card is an argument (`cargo test -p brain-gpu-core --test
//! cuda_provider_matmul_i8 -- --device gpu1 --backend cuda`); without one
//! nothing here opens a card.

use std::sync::Arc;

use backend_api::select::{self, Dtype, KernelSelector};
use backend_api::DType;
use gpu_core::provider::{LowerCtx, Operand, OpRequest, ProviderRegistry, Pass, Role};
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[("matmul_i8_dyn", kernels::MATMUL_I8_DYN)];

/// Largest `|tensor-core - portable|` allowed, as a fraction of the RMS of the
/// portable output. The two differ by f32 rounding of a running sum of at most
/// `K/32` terms, which is ~1e-6 of the sum's magnitude at the deepest K here;
/// the bar leaves an order of magnitude over what is measured (printed by the
/// run) without being loose enough to hide a wrong group, row or scale.
const REL_TOL: f32 = 2.0e-5;

/// Outputs checked against the f64 host oracle per shape, and the bar for them:
/// the f32 kernels each round a K/32-term running sum, so the oracle bar is the
/// same relative scale.
const ORACLE_SAMPLES: usize = 2048;

/// The reference side is the native DP4A GEMM that `native_upgrade` substitutes for the WGSL kernel (several times
/// under the tensor cores at a 256-row round), not the scalar WGSL kernel this floor was first set against.
/// How much faster than the DP4A kernel the tensor-core kernel must be at the
/// real prefill shapes, by device time. Set far under what is measured (on
/// an idle part with int8 tensor cores) so a busy shared device cannot fail it, but high
/// enough that a kernel which fell back to scalar work would.
// perf-number: the asserted floor of this gate; the achieved ratio is printed.
const SPEEDUP_FLOOR: f64 = 2.0;

/// Device-timed trials per side; the best is compared. The cards of a shared
/// box are time-sliced between processes, which only ever inflates a kernel's
/// measured time, so the minimum over enough trials is the honest figure.
const TRIALS: usize = 12;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn i8(&mut self) -> i8 {
        (self.next() >> 56) as i8
    }
    /// Uniform in `[lo, hi)`.
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * (hi - lo)
    }
}

fn pack(bytes: &[i8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect()
}

struct Problem {
    m: u32,
    n: u32,
    k: u32,
    xq: Vec<i8>,
    wq: Vec<i8>,
    sx: Vec<f32>,
    sw: Vec<f32>,
}

impl Problem {
    /// Q8_0-like data: full-range int8, per-group weight scales around the
    /// magnitude real checkpoints carry, per-group (32 along K) activation scales.
    fn new(m: u32, n: u32, k: u32, seed: u64) -> Problem {
        let mut r = Rng(seed);
        let xq = (0..m * k).map(|_| r.i8()).collect();
        let wq = (0..n * k).map(|_| r.i8()).collect();
        let sx = (0..m * (k / 32)).map(|_| r.f32(0.001, 0.02)).collect();
        let sw = (0..n * (k / 32)).map(|_| r.f32(0.0005, 0.01)).collect();
        Problem { m, n, k, xq, wq, sx, sw }
    }

    fn oracle(&self, row: usize, col: usize) -> f64 {
        let k = self.k as usize;
        let ng = k / 32;
        let mut acc = 0f64;
        for g in 0..ng {
            let mut s = 0i64;
            for i in g * 32..g * 32 + 32 {
                s += self.xq[row * k + i] as i64 * self.wq[col * k + i] as i64;
            }
            acc += s as f64 * (self.sw[col * ng + g] * self.sx[row * ng + g]) as f64;
        }
        acc
    }
}

struct Device<'a> {
    gpu: &'a Gpu,
    x: backend_api::DeviceBuffer,
    w: backend_api::DeviceBuffer,
    sx: backend_api::DeviceBuffer,
    sw: backend_api::DeviceBuffer,
    y: backend_api::DeviceBuffer,
}

impl<'a> Device<'a> {
    fn upload(gpu: &'a Gpu, p: &Problem) -> Device<'a> {
        let up = |words: &[u32]| {
            let b = gpu.storage(words.len() as u64);
            gpu.write(&b, words);
            b
        };
        Device {
            gpu,
            x: up(&pack(&p.xq)),
            w: up(&pack(&p.wq)),
            sx: gpu.storage_init("sx", &p.sx),
            sw: gpu.storage_init("sw", &p.sw),
            y: gpu.storage(p.m as u64 * p.n as u64),
        }
    }

    /// Dispatch `p` through `registry` and read the output back, with the
    /// kernel that answered.
    fn run(&self, registry: &ProviderRegistry, p: &Problem) -> (Vec<f32>, Vec<&'static str>) {
        let kernels = self.dispatch(registry, p);
        (self.gpu.read(&self.y, (p.m * p.n) as usize), kernels)
    }

    /// Record and submit `p` through `registry` without waiting for it.
    fn dispatch(&self, registry: &ProviderRegistry, p: &Problem) -> Vec<&'static str> {
        let gpu = self.gpu;
        let (m, n, k) = (p.m, p.n, p.k);
        let bind = |_: select::KernelVariant| -> (usize, &'static str) {
            (gpu.kernel_index("matmul_i8_dyn").expect("registered"), "matmul_i8_dyn")
        };
        let kg = k as u64 / 4;
        let operands = [
            Operand { role: Role::Act, buf: &self.x, range: (0, m as u64 * kg), dtype: DType::I8 },
            Operand { role: Role::Weight, buf: &self.w, range: (0, 0), dtype: DType::I8 },
            Operand { role: Role::ActScale, buf: &self.sx, range: (0, m as u64 * (k as u64 / 32)), dtype: DType::F32 },
            Operand { role: Role::WeightScale, buf: &self.sw, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &self.y, range: (0, m as u64 * n as u64), dtype: DType::F32 },
        ];
        let attrs = [m, kg as u32, n];
        let req = OpRequest {
            op: select::Op::MatMul,
            shape: select::OpShape { m, n, k, dtype: Dtype::I8 },
            pass: Pass::Forward,
            operands: &operands,
            attrs: &attrs,
            group: 32,
            bind: &bind,
        };
        let caps = gpu.caps();
        let mut steps = Vec::new();
        let lowered = {
            let mut ctx = LowerCtx { gpu, caps: &caps, steps: &mut steps, capture: false };
            registry.dispatch(&mut ctx, &req)
        };
        gpu.submit(&[], &steps);
        lowered.kernels
    }

    /// Device milliseconds per dispatch of the kernels whose name contains `kernel_family`, from the backend's own
    /// event-bracketed kernel table - immune to host load. Best of `trials`.
    fn device_ms(&self, registry: &ProviderRegistry, p: &Problem, kernel_family: &str, trials: usize) -> f64 {
        let gpu = self.gpu;
        assert!(gpu.set_kernel_timing(true), "this backend cannot time kernels");
        let mut best = f64::MAX;
        for _ in 0..trials {
            gpu.reset_kernel_times();
            let reps = 4;
            for _ in 0..reps {
                self.dispatch(registry, p);
            }
            gpu.poll_wait();
            let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(name, _, _)| name.contains(kernel_family)).map(|(_, ms, _)| ms).sum();
            best = best.min(ms / reps as f64);
        }
        gpu.set_kernel_timing(false);
        best
    }
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len().max(1) as f64).sqrt() as f32
}

/// The production registry, and the portable reference alone, over one device.
fn registries(gpu: &Gpu) -> (ProviderRegistry, ProviderRegistry) {
    let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
    (ProviderRegistry::for_gpu(gpu, selector.clone()), ProviderRegistry::reference(selector))
}

/// **The milestone's red test.** On every real 27B prefill GEMM shape (and
/// ragged ones) the production registry must answer with the tensor-core
/// kernel, agree with the portable kernel to [`REL_TOL`], and agree with an
/// f64 host oracle.
fn the_int8_tensor_core_gemm_is_routed_to_and_agrees_with_the_portable_kernel_on_real_shapes() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        eprintln!("cuda_provider_matmul_i8: no CUDA device on this box - skipping");
        return;
    };
    let cc = gpu.caps().arch.compute_capability.expect("the CUDA backend reports its capability");
    if cc < kernels_cuda::MMA_S8_MIN_CC {
        eprintln!("cuda_provider_matmul_i8: compute capability {}.{} has no int8 MMA - skipping", cc.0, cc.1);
        return;
    }
    let (production, reference) = registries(&gpu);

    // (m, n, k): Qwen3.8-27B d_model 5120, ff 17408, GDN/attention widths; a
    // prefill round of 256 rows; then ragged rows/columns (a partial row tile,
    // a partial column tile, odd N) and the row counts either side of where the
    // tensor-core GEMM takes over from the GEMV (the smallest batch past
    // `select::I8_GEMV_MAX_ROWS`, the top of the decode regime, the first
    // prefill count).
    let shapes: &[(u32, u32, u32)] = &[
        (256, 5120, 5120),
        (256, 17408, 5120),
        (256, 5120, 17408),
        (256, 10240, 5120),
        (256, 6144, 6144),
        (300, 5136, 5120),
        (9, 6144, 5120),
        (16, 5120, 5120),
        (17, 17408, 5120),
        (32, 6144, 5120),
        (33, 6144, 5120),
        (64, 5120, 17408),
        (100, 129, 128),
        (128, 10240, 5120),
        (129, 5120, 5120),
    ];
    let mut worst = 0f32;
    for (i, &(m, n, k)) in shapes.iter().enumerate() {
        let p = Problem::new(m, n, k, 0x1000 + i as u64);
        let d = Device::upload(&gpu, &p);
        let (want, ref_kernels) = d.run(&reference, &p);
        assert_eq!(ref_kernels, vec!["matmul_i8_dyn"], "the reference registry must run the portable kernel");
        let (got, kernels_used) = d.run(&production, &p);
        if m <= backend_api::select::DECODE_REGIME_MAX_ROWS {
            // The decode regime belongs to the int8 GEMV slot (`gpu_core::native_tune` chooses its candidate per
            // shape class), so the provider leaves it alone.
            assert_eq!(kernels_used, vec!["matmul_i8_dyn"], "{m} rows is the GEMV slot's: the provider must decline it");
            continue;
        }
        assert!(
            kernels_used.len() == 1 && kernels_used[0].starts_with("native:matmul_i8_"),
            "{m}x{n}x{k}: the production chain must route to a tensor-core kernel, got {kernels_used:?}"
        );

        let scale = rms(&want);
        let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("cuda_provider_matmul_i8: {m}x{n}x{k}: max|tc - portable| = {maxdiff:e} ({:.2e} of rms {scale:e})", maxdiff / scale);
        assert!(maxdiff <= REL_TOL * scale, "{m}x{n}x{k}: tensor-core GEMM differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
        worst = worst.max(maxdiff / scale);

        let mut rng = Rng(77 + i as u64);
        let mut oracle_worst = 0f64;
        for _ in 0..ORACLE_SAMPLES.min((m * n) as usize) {
            let (r, c) = ((rng.next() % m as u64) as usize, (rng.next() % n as u64) as usize);
            let truth = p.oracle(r, c);
            oracle_worst = oracle_worst.max((got[r * n as usize + c] as f64 - truth).abs());
        }
        assert!(oracle_worst <= (REL_TOL * scale) as f64, "{m}x{n}x{k}: tensor-core GEMM is {oracle_worst:e} from the f64 oracle");
    }
    eprintln!("cuda_provider_matmul_i8: worst relative difference over all shapes {worst:.2e} (bar {REL_TOL:.0e})");
}

/// One launch of the registry kernel `name` on `p`, bypassing the row-count routing.
fn run_named(gpu: &Gpu, name: &str, p: &Problem) -> Vec<f32> {
    use backend_api::BindKind;
    const BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];
    let kernel = kernels_cuda::get(name).unwrap_or_else(|| panic!("{name} is registered"));
    // The warpgroup GEMM's stage ring is dynamic shared memory above the default limit, in an arch-specific build.
    let launch = if name == "matmul_i8_wgmma" {
        backend_api::CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::I8_WGMMA_DYNAMIC_SHARED_BYTES,
            shared_opt_in: true,
            arch: backend_api::ArchFeatures::Required,
            ..backend_api::CudaLaunch::NONE
        }
    } else {
        backend_api::CudaLaunch::NONE
    };
    let id = gpu.native_kernel_launched(kernel, BINDINGS, launch).unwrap_or_else(|| panic!("{name} is offered on this device"));
    let d = Device::upload(gpu, p);
    let kg = p.k as u64 / 4;
    let operands = [(&d.x, (0, p.m as u64 * kg)), (&d.w, (0, 0)), (&d.sx, (0, p.m as u64 * (p.k as u64 / 32))), (&d.sw, (0, 0)), (&d.y, (0, p.m as u64 * p.n as u64))];
    let bufs: Vec<&backend_api::DeviceBuffer> = operands.iter().map(|o| o.0).collect();
    let offs: Vec<(u64, u64)> = operands.iter().map(|o| o.1).collect();
    let blocks = p.m.div_ceil(kernel.tile.0) * p.n.div_ceil(kernel.tile.1);
    let step = gpu.step_native_sliced(id, &bufs, &offs, &[p.m, kg as u32, p.n], blocks).expect("native step");
    gpu.submit(&[&d.y], std::slice::from_ref(&step));
    gpu.read(&d.y, (p.m * p.n) as usize)
}

/// Every tile of the int8 GEMM folds an output the same way, so the result for a row
/// does not depend on how many rows share the launch: the decode-batch tiles and the
/// 64x64 prefill tile give the same bits. That is what lets
/// a prompt be prefilled in slices, in a batch with other prompts, or whole, and what
/// lets a speculative verify step reproduce plain decode.
fn every_tile_of_the_int8_gemm_gives_the_same_bits() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        eprintln!("cuda_provider_matmul_i8: no CUDA device on this box - skipping");
        return;
    };
    let cc = gpu.caps().arch.compute_capability.expect("the CUDA backend reports its capability");
    if cc < kernels_cuda::MMA_S8_MIN_CC {
        eprintln!("cuda_provider_matmul_i8: compute capability {}.{} has no int8 MMA - skipping", cc.0, cc.1);
        return;
    }
    for (i, &(m, n, k)) in [(200u32, 2048u32, 1024u32), (19, 4096, 2048), (577, 1088, 5120)].iter().enumerate() {
        let p = Problem::new(m, n, k, 0x7117 + i as u64);
        let reference = run_named(&gpu, "matmul_i8_mma", &p);
        let mut tiles = vec!["matmul_i8_mma_m128", "matmul_i8_mma_m32", "matmul_i8_mma_m16"];
        // The warpgroup GEMM folds each group with the same two fused multiply-adds in the same order.
        if cc >= kernels_cuda::WGMMA_MIN_CC {
            tiles.push("matmul_i8_wgmma");
        }
        for name in tiles {
            let got = run_named(&gpu, name, &p);
            let differing = reference.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differing, 0, "{m}x{n}x{k}: {name} differs from matmul_i8_mma in {differing} of {} outputs", reference.len());
        }
    }
}

/// Every candidate of the prefill slot - not only the one the device's tuned table names - agrees with an f64
/// oracle on shapes with partial row and column tiles, and runs where it serves. A candidate that is wrong on a
/// shape the tuner never timed is a defect waiting for the next model, so each is forced through the production
/// chain (`gpu_core::force_candidate`) and checked, and each must be seen to have run at least once.
fn every_prefill_candidate_agrees_with_the_f64_oracle_on_ragged_shapes() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        return brain_testutil::skip_unavailable("cuda_provider_matmul_i8: no CUDA device on this box");
    };
    let cc = gpu.caps().arch.compute_capability.expect("the CUDA backend reports its capability");
    if cc < kernels_cuda::MMA_S8_MIN_CC {
        return brain_testutil::skip_unvalidated_capability("int8-tensor-core-mma", &format!("compute capability {}.{} has no int8 MMA", cc.0, cc.1));
    }
    let (production, _) = registries(&gpu);
    let offered: Vec<String> = gpu_core::native_candidates_offered().into_iter().filter(|(slot, _)| slot == "matmul_i8_prefill").map(|(_, k)| k).collect();
    assert!(offered.len() >= 2, "the prefill slot offers a choice on this device, got {offered:?}");
    let mut ran: Vec<&str> = Vec::new();
    // (m, n, k): a partial row tile, a partial column tile, odd N, K at 64 and at 128 granularity, a wide round, and the
    // rows of a large decode batch (33-128) the weight-streaming candidates serve.
    for (i, &(m, n, k)) in [(300u32, 5136u32, 5120u32), (129, 2050, 1024), (1100, 643, 4096), (64, 4096, 5184), (2050, 256, 2048), (40, 5136, 5120), (100, 2050, 4096), (33, 643, 1280)].iter().enumerate() {
        let p = Problem::new(m, n, k, 0x2000 + i as u64);
        let d = Device::upload(&gpu, &p);
        for cand in &offered {
            gpu_core::force_candidate("matmul_i8_prefill", cand);
            let (got, used) = d.run(&production, &p);
            gpu_core::clear_forced_candidates();
            assert_eq!(used.len(), 1, "{m}x{n}x{k}: one kernel answers, got {used:?}");
            if used[0] == format!("native:{cand}") {
                ran.push(cand);
            }
            let scale = rms(&got);
            let mut rng = Rng(177 + i as u64);
            let mut worst = 0f64;
            // The corners are where a partial tile goes wrong.
            for &(r, c) in [(0usize, 0usize), (m as usize - 1, n as usize - 1), (m as usize - 1, 0), (0, n as usize - 1)].iter() {
                worst = worst.max((got[r * n as usize + c] as f64 - p.oracle(r, c)).abs());
            }
            for _ in 0..512 {
                let (r, c) = ((rng.next() % m as u64) as usize, (rng.next() % n as u64) as usize);
                worst = worst.max((got[r * n as usize + c] as f64 - p.oracle(r, c)).abs());
            }
            assert!(worst <= (REL_TOL * scale) as f64, "{m}x{n}x{k}: {} is {worst:e} from the f64 oracle (bar {:e})", used[0], REL_TOL * scale);
        }
    }
    for cand in &offered {
        assert!(ran.contains(&cand.as_str()), "candidate {cand} never ran on any shape it was forced on (ran: {ran:?})");
    }
}

/// A native tensor-core kernel that is not materially faster than the
/// DP4A kernel it displaces is a maintenance cost with a `Tuned` claim
/// attached. Device time, at the real prefill round shape.
fn the_int8_tensor_core_gemm_is_materially_faster_than_the_portable_kernel() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        return brain_testutil::skip_unavailable("cuda_provider_matmul_i8: no CUDA device on this box");
    };
    let cc = gpu.caps().arch.compute_capability.expect("the CUDA backend reports its capability");
    if cc < kernels_cuda::MMA_S8_MIN_CC {
        return brain_testutil::skip_unvalidated_capability("int8-tensor-core-mma", &format!("compute capability {}.{} has no int8 MMA", cc.0, cc.1));
    }
    let (production, reference) = registries(&gpu);
    for &(m, n, k) in &[(256u32, 5120u32, 5120u32), (256, 17408, 5120), (256, 5120, 17408)] {
        let p = Problem::new(m, n, k, 5);
        let d = Device::upload(&gpu, &p);
        d.run(&reference, &p); // compile outside the timed region
        d.run(&production, &p);
        let t_ref = d.device_ms(&reference, &p, "matmul_i8_", TRIALS);
        let t_tc = d.device_ms(&production, &p, "matmul_i8_", TRIALS);
        let tops = |ms: f64| 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
        let speedup = t_ref / t_tc;
        eprintln!(
            "cuda_provider_matmul_i8: {m}x{n}x{k}: portable {t_ref:.3} ms ({:.1} TOPS), tensor-core {t_tc:.3} ms ({:.1} TOPS), {speedup:.1}x",
            tops(t_ref),
            tops(t_tc)
        );
        assert!(speedup >= SPEEDUP_FLOOR, "{m}x{n}x{k}: tensor-core GEMM is only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
    }
}

/// Developer sweep, not a gate: compile the tensor-core kernel at alternative
/// tile / warp / pipeline configurations (the `BRAIN_I8_*` macros the source
/// exposes) and print device-timed TOPS at the real prefill shapes, so a tile
/// choice is a measurement rather than a guess.
///
///   cargo test --release -p brain-gpu-core --test cuda_provider_matmul_i8 -- --device gpu1 --backend cuda --ignored sweep
fn sweep_tile_configurations() {
    // (BM, BN, WARPS_M, WARPS_N, STAGES)
    let configs: &[(u32, u32, u32, u32, u32)] = &[
        (128, 64, 4, 2, 3),
        (128, 64, 4, 2, 2),
        (64, 64, 2, 2, 4),
        (64, 128, 2, 4, 3),
        (128, 128, 4, 2, 2),
        (256, 64, 4, 2, 2),
    ];
    sweep(configs, &[(256u32, 5120u32, 5120u32), (256, 17408, 5120), (256, 5120, 17408), (512, 5120, 5120)]);
}

/// [`sweep_tile_configurations`] at the row counts of a batched DECODE step
/// (16-128 sequences) on the 27B's projection shapes, where a block's 64x64
/// output tile leaves most of a 132-SM part without a block and the kernel
/// streams the weights at a fraction of the memory bandwidth: smaller weight
/// tiles (more blocks) and deeper pipelines (more bytes in flight).
///
///   cargo test --release -p brain-gpu-core --test cuda_provider_matmul_i8 -- --backend cuda --ignored sweep_skinny
fn sweep_skinny_tile_configurations() {
    let configs: &[(u32, u32, u32, u32, u32)] = &[
        (64, 64, 2, 2, 4),
        (64, 32, 2, 1, 4),
        (64, 32, 2, 1, 6),
        (64, 16, 2, 1, 4),
        (64, 16, 2, 1, 6),
        (32, 32, 2, 1, 6),
        (32, 16, 2, 1, 6),
        (16, 32, 1, 1, 6),
        (16, 16, 1, 1, 8),
        (64, 32, 4, 1, 4),
    ];
    let mut shapes = Vec::new();
    for m in [16u32, 32, 64, 128] {
        for (n, k) in [(5120u32, 5120u32), (17408, 5120), (5120, 17408), (10240, 5120)] {
            shapes.push((m, n, k));
        }
    }
    sweep(configs, &shapes);
}

/// [`sweep_skinny_tile_configurations`] at the 3-16 rows of the small-batch decode step (the rows the dp4a GEMV
/// serves today, where its per-row f32 fold keeps the arithmetic pipes busy), weights cycled past L2.
///
///   cargo test --release -p brain-gpu-core --test cuda_provider_matmul_i8 -- --backend cuda --ignored sweep_decode
fn sweep_decode_tile_configurations() {
    let configs: &[(u32, u32, u32, u32, u32)] = &[
        (16, 16, 1, 1, 8),
        (16, 32, 1, 1, 8),
        (16, 32, 1, 2, 6),
        (16, 64, 1, 2, 6),
        (16, 64, 1, 4, 6),
        (16, 128, 1, 4, 4),
        (16, 32, 1, 1, 12),
        (16, 16, 1, 1, 14),
    ];
    let mut shapes = Vec::new();
    for m in [4u32, 8, 16] {
        for (n, k) in [(5120u32, 5120u32), (10240, 5120), (17408, 5120), (5120, 17408)] {
            shapes.push((m, n, k));
        }
    }
    sweep(configs, &shapes);
}

/// Developer sweep at the prefill round widths and the 27B's projection shapes (cuda.md, prefill GEMM): tile,
/// warp split and pipeline depth against the rows of one round.
///
///   cargo test --release -p brain-gpu-core --test cuda_provider_matmul_i8 -- --device gpu1 --backend cuda --ignored sweep_prefill
fn sweep_prefill_configurations() {
    let configs: &[(u32, u32, u32, u32, u32)] = &[(64, 64, 2, 2, 4), (64, 64, 2, 2, 3), (128, 64, 4, 2, 3), (128, 128, 4, 2, 2), (128, 128, 2, 4, 2)];
    let mut shapes = Vec::new();
    for m in [256u32, 1024, 4096] {
        for (n, k) in [(5120u32, 5120u32), (17408, 5120), (5120, 17408)] {
            shapes.push((m, n, k));
        }
    }
    sweep(configs, &shapes);
}

/// Developer report, not a gate: open a handle that registers the int8 GEMM kernels, let the tuner measure every
/// prefill candidate on its class grid and print the selection table (candidate times, achieved TOPS, fraction of
/// the measured roofs).
///
///   cargo test --release -p brain-gpu-core --test cuda_provider_matmul_i8 -- --device gpu1 --backend cuda --ignored prefill_selection_report --nocapture
fn prefill_selection_report() {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        return brain_testutil::skip_unavailable("prefill_selection_report: no CUDA device on this box");
    };
    let _ = &gpu;
    eprintln!("{}", gpu_core::native_selection_report());
}

/// Distinct weight bytes cycled per timed pass in [`sweep`], well past L2.
const SWEEP_FOOTPRINT_BYTES: u64 = 768 << 20;

fn sweep(configs: &[(u32, u32, u32, u32, u32)], shapes: &[(u32, u32, u32)]) {
    use backend_api::{BindKind, NativeSpec};
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        return brain_testutil::skip_unavailable("sweep: no CUDA device on this box");
    };
    let src = kernels_cuda::get("matmul_i8_mma").expect("registered").src;
    let bindings: &'static [BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];
    for &(m, n, k) in shapes {
        let p = Problem::new(m, n, k, 9);
        let d = Device::upload(&gpu, &p);
        // Decode reads every weight from HBM once per token: cycle through enough copies
        // of the weights to exceed L2 several times, or a small projection reports L2 bandwidth.
        let kg = k as u64 / 4;
        let wbytes = n as u64 * kg * 4 + n as u64 * (kg / 8) * 4;
        let copies = (SWEEP_FOOTPRINT_BYTES / wbytes).max(1) as usize;
        let extra: Vec<(backend_api::DeviceBuffer, backend_api::DeviceBuffer)> = (1..copies)
            .map(|_| {
                let w = gpu.storage(n as u64 * kg);
                let s = gpu.storage(n as u64 * (kg / 8));
                (w, s)
            })
            .collect();
        for &(bm, bn, wm, wn, st) in configs {
            let text: &'static str = Box::leak(
                format!("#define BRAIN_I8_BM {bm}\n#define BRAIN_I8_BN {bn}\n#define BRAIN_I8_WARPS_M {wm}\n#define BRAIN_I8_WARPS_N {wn}\n#define BRAIN_I8_STAGES {st}\n{src}").into_boxed_str(),
            );
            // The source's static shared-memory arrays: two int8 tiles and two group-scale arrays per stage.
            let smem = st * (bm * 64 + bn * 64 + bn * 8 + bm * 8);
            let Some(id) = gpu.register_native(&NativeSpec::Cuda { name: "native:matmul_i8_mma", src: text, entry: "brain_matmul_i8_mma", block_dim: wm * wn * 32, bindings, shared_bytes: smem, launch: backend_api::CudaLaunch::NONE }) else {
                eprintln!("sweep: {m}x{n}x{k} cfg {bm}x{bn} w{wm}x{wn} s{st}: declined");
                continue;
            };
            let attrs = [m, kg as u32, n];
            let blocks = m.div_ceil(bm) * n.div_ceil(bn);
            let steps: Vec<_> = (0..copies)
                .map(|c| {
                    let (w, sw) = if c == 0 { (&d.w, &d.sw) } else { (&extra[c - 1].0, &extra[c - 1].1) };
                    let bufs: [&backend_api::DeviceBuffer; 5] = [&d.x, w, &d.sx, sw, &d.y];
                    let offs = [(0, m as u64 * kg), (0, 0), (0, m as u64), (0, 0), (0, m as u64 * n as u64)];
                    gpu.step_native_sliced(id, &bufs, &offs, &attrs, blocks).expect("native step")
                })
                .collect();
            gpu.submit(&[], &steps);
            gpu.poll_wait();
            gpu.set_kernel_timing(true);
            let mut best = f64::MAX;
            for _ in 0..7 {
                gpu.reset_kernel_times();
                gpu.submit(&[], &steps);
                gpu.poll_wait();
                let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().map(|(_, ms, _)| ms).sum();
                best = best.min(ms / copies as f64);
            }
            gpu.set_kernel_timing(false);
            // A TOPS figure is only worth reading beside a checked output (knowledge 319): the whole
            // output is read back and a spread of positions, including both last rows and columns, is
            // compared with the f64 oracle.
            gpu.submit(&[&d.y], &steps[..1]);
            let got = gpu.read(&d.y, (m * n) as usize);
            let scale = rms(&got);
            let worst = [(0, 0), (m - 1, n - 1), (m / 2, n / 3), (m - 1, 0), (0, n - 1), (m / 3, n - 1), (m - 1, n / 2)]
                .iter()
                .map(|&(r, c)| (got[(r * n + c) as usize] as f64 - p.oracle(r as usize, c as usize)).abs())
                .fold(0f64, f64::max);
            eprintln!(
                "sweep: {m}x{n}x{k} cfg {bm}x{bn} w{wm}x{wn} s{st}: {best:.4} ms  {:.1} TOPS  {:.0} GB/s of weights+scales  (oracle err {:.1e} of rms{})",
                2.0 * m as f64 * n as f64 * k as f64 / (best * 1e-3) / 1e12,
                wbytes as f64 / (best * 1e-3) / 1e9,
                worst / scale as f64,
                if worst > 2e-5 * scale as f64 { "  WRONG" } else { "" }
            );
        }
    }
}

/// The selector hands rows to the tensor-core GEMM from the capability the kernel
/// itself needs; the two constants are one fact stated in two crates.
fn the_selector_and_the_kernel_agree_on_the_tensor_core_floor() {
    assert_eq!(select::I8_MMA_MIN_CC, kernels_cuda::MMA_S8_MIN_CC);
}

gpu_core::card_tests!(
    the_int8_tensor_core_gemm_is_routed_to_and_agrees_with_the_portable_kernel_on_real_shapes,
    the_int8_tensor_core_gemm_is_materially_faster_than_the_portable_kernel,
    every_tile_of_the_int8_gemm_gives_the_same_bits,
    every_prefill_candidate_agrees_with_the_f64_oracle_on_ragged_shapes,
    ; host: the_selector_and_the_kernel_agree_on_the_tensor_core_floor,
    ; ignored: sweep_tile_configurations, sweep_skinny_tile_configurations,
    sweep_decode_tile_configurations, sweep_prefill_configurations, prefill_selection_report,
);
