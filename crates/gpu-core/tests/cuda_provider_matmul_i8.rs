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

/// How much faster than the DP4A kernel the tensor-core kernel must be at the
/// real prefill shapes, by device time. Set far under what is measured on a
/// part with int8 tensor cores so a busy shared device cannot fail it, but high
/// enough that a kernel which fell back to scalar work would.
// perf-number: the asserted floor of this gate; the achieved ratio is printed.
const SPEEDUP_FLOOR: f64 = 5.0;

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
    /// magnitude real checkpoints carry, per-token activation scales.
    fn new(m: u32, n: u32, k: u32, seed: u64) -> Problem {
        let mut r = Rng(seed);
        let xq = (0..m * k).map(|_| r.i8()).collect();
        let wq = (0..n * k).map(|_| r.i8()).collect();
        let sx = (0..m).map(|_| r.f32(0.001, 0.02)).collect();
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
            acc += s as f64 * self.sw[col * ng + g] as f64;
        }
        acc * self.sx[row] as f64
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
            Operand { role: Role::ActScale, buf: &self.sx, range: (0, m as u64), dtype: DType::F32 },
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

    /// Device milliseconds per dispatch of `kernel`, from the backend's own
    /// event-bracketed kernel table - immune to host load. Best of `trials`.
    fn device_ms(&self, registry: &ProviderRegistry, p: &Problem, kernel_suffix: &str, trials: usize) -> f64 {
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
            let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(name, _, _)| name.ends_with(kernel_suffix)).map(|(_, ms, _)| ms).sum();
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
#[test]
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
    // a partial column tile, odd N) and the smallest row count past the decode
    // regime.
    let shapes: &[(u32, u32, u32)] = &[
        (256, 5120, 5120),
        (256, 17408, 5120),
        (256, 5120, 17408),
        (256, 10240, 5120),
        (256, 6144, 6144),
        (300, 5136, 5120),
        (33, 6144, 5120),
        (100, 129, 128),
    ];
    let mut worst = 0f32;
    for (i, &(m, n, k)) in shapes.iter().enumerate() {
        let p = Problem::new(m, n, k, 0x1000 + i as u64);
        let d = Device::upload(&gpu, &p);
        let (want, ref_kernels) = d.run(&reference, &p);
        assert_eq!(ref_kernels, vec!["matmul_i8_dyn"], "the reference registry must run the portable kernel");
        let (got, kernels_used) = d.run(&production, &p);
        assert_eq!(kernels_used, vec!["native:matmul_i8_mma"], "{m}x{n}x{k}: the production chain must route to the tensor-core kernel");

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

/// A hand-written tensor-core kernel that is not materially faster than the
/// DP4A kernel it displaces is a maintenance cost with a `Tuned` claim
/// attached. Device time, at the real prefill round shape.
#[test]
fn the_int8_tensor_core_gemm_is_materially_faster_than_the_portable_kernel() {
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
    for &(m, n, k) in &[(256u32, 5120u32, 5120u32), (256, 17408, 5120), (256, 5120, 17408)] {
        let p = Problem::new(m, n, k, 5);
        let d = Device::upload(&gpu, &p);
        d.run(&reference, &p); // compile outside the timed region
        d.run(&production, &p);
        let t_ref = d.device_ms(&reference, &p, "matmul_i8_dyn", 3);
        let t_tc = d.device_ms(&production, &p, "matmul_i8_mma", 3);
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
