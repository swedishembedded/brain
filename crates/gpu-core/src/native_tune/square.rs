// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device's large square GEMMs on brain's best kernels, each checked
//! against an f64 reference before it is timed: the compute rows of `brain
//! perf roofline`.
//!
//! Swedish Embedded AB implements tensor-core GEMM validation for its clients.
//! If your team needs to know what its matrix kernels reach on a new
//! accelerator, and that they are right, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Every candidate the tuner could choose for a prefill-sized product runs on
//! one square problem, graded by the tuner's own admission rule
//! ([`super::admit`]) and timed launch by launch ([`crate::launch_stats`]): the
//! int8 GEMMs with int8 activations, the dequantising ones with bf16
//! activations, and the FP8 linear's candidates. The vendor-library products
//! brain trains with (bf16 and TF32 tensor cores, fp32 CUDA cores) and brain's
//! own fp32 GEMM run on the same square, checked at the same sampled outputs.
//! Each precision reports its fastest admitted kernel and names the others.

use super::{sample_positions, time_class, Acts, Verdict, FP8, MIN_LAUNCHES, PREFILL, TUNED_SLOT_KERNELS, TUNING};
use crate::launch_stats::{self, LaunchStats};
use crate::native_upgrade::ActFormat;
use crate::provider::library::{F32Product, GemmPrecision, LibraryGemmProvider, Operand};
use crate::{DeviceBuffer, Gpu, Step};

/// The side of the square problem: large enough that every tile scheduler fills the device many times over, small
/// enough that the host's reference activations take a few seconds.
pub(crate) const SIDE: u32 = 8192;

/// One precision's fastest admitted kernel on the square problem.
#[derive(Clone, Debug)]
pub(crate) struct SquareGemm {
    /// `int8-tc`, `bf16-tc`, `fp8-tc`, `tf32-tc` or `fp32-gemm`.
    pub metric: &'static str,
    pub kernel: String,
    pub stats: LaunchStats,
    /// Operations a second at the median launch, in tera.
    pub tera: f64,
    /// How the output was checked.
    pub verified: String,
    /// The other admitted kernels of the precision and their rates; refused ones with the reason.
    pub others: Vec<String>,
}

/// The precision a candidate's activation form computes in.
fn metric_of(act: ActFormat) -> &'static str {
    match act {
        ActFormat::Int8Group32 => "int8-tc",
        ActFormat::Bf16 => "bf16-tc",
        ActFormat::Fp8Group128 => "fp8-tc",
    }
}

/// Operations of the square product.
fn ops() -> f64 {
    2.0 * f64::from(SIDE).powi(3)
}

/// Keep the faster of `best` and `cand` for `metric`, moving the slower into its `others`.
fn keep(found: &mut Vec<SquareGemm>, cand: SquareGemm) {
    match found.iter_mut().find(|g| g.metric == cand.metric) {
        Some(g) if cand.tera > g.tera => {
            let slower = format!("{} {:.0}", g.kernel, g.tera);
            let mut others = std::mem::take(&mut g.others);
            others.push(slower);
            others.extend(cand.others.iter().cloned());
            *g = SquareGemm { others, ..cand };
        }
        Some(g) => {
            g.others.push(format!("{} {:.0}", cand.kernel, cand.tera));
            g.others.extend(cand.others);
        }
        None => found.push(cand),
    }
}

/// Every precision's fastest admitted kernel on the `SIDE`-cubed product on `dev`, and for each precision no kernel
/// was admitted for, why. The handle is opened without tuning its slots (the candidates are timed here directly).
pub(crate) fn square_gemms(dev: &crate::devices::DeviceId) -> (Vec<SquareGemm>, Vec<(&'static str, String)>) {
    let mut kernels = TUNED_SLOT_KERNELS.to_vec();
    kernels.push(("matmul_reg3", kernels::MATMUL_REG3));
    TUNING.with(|t| t.set(true));
    let gpu = Gpu::new_on(dev, &kernels);
    TUNING.with(|t| t.set(false));
    let mut missing = Vec::new();
    if !gpu.set_kernel_timing(true) {
        let why = "the backend cannot time kernels on this device".to_owned();
        return (Vec::new(), ["int8-tc", "bf16-tc", "fp8-tc", "tf32-tc", "fp32-gemm"].map(|m| (m, why.clone())).to_vec());
    }
    let (m, kg, n) = (SIDE, SIDE / 4, SIDE);
    let acts = Acts::new(m, kg);
    let mut found: Vec<SquareGemm> = Vec::new();
    for (kind, grid) in [(crate::native_upgrade::PREFILL_KIND, &PREFILL), (crate::native_upgrade::FP8_KIND, &FP8)] {
        let params = [m, kg, n];
        let cands: Vec<&crate::native_upgrade::Active> = crate::native_upgrade::candidates(gpu.native_rows(), kind, &params).collect();
        let verdicts = time_class(&gpu, grid, &cands, (m, kg, n), &acts, true);
        for (a, (name, v)) in cands.iter().zip(&verdicts) {
            let metric = metric_of(a.input.act);
            match v {
                Verdict::Timed { us, min_us, stretched, .. } => keep(
                    &mut found,
                    SquareGemm {
                        metric,
                        kernel: name.clone(),
                        stats: LaunchStats { median_ms: us * 1e-3, min_ms: min_us * 1e-3, stretched: *stretched, samples: MIN_LAUNCHES },
                        tera: ops() / (us * 1e-6) / 1e12,
                        verified: format!("admitted against the f64 reference of its stored inputs at {} sampled outputs", sample_positions(m, n).len()),
                        others: Vec::new(),
                    },
                ),
                Verdict::Rejected(why) => {
                    let refused = format!("{name} REJECTED ({why})");
                    match found.iter_mut().find(|g| g.metric == metric) {
                        Some(g) => g.others.push(refused),
                        None => missing.push((metric, refused)),
                    }
                }
            }
        }
    }
    for g in library_and_fp32(&gpu, &acts, &mut missing) {
        keep(&mut found, g);
    }
    gpu.set_kernel_timing(false);
    // A precision that only failed is reported missing once, with every refusal.
    let mut why: Vec<(&'static str, String)> = Vec::new();
    for (metric, reason) in missing.into_iter().filter(|(metric, _)| !found.iter().any(|g| g.metric == *metric)) {
        match why.iter_mut().find(|(m, _)| *m == metric) {
            Some((_, r)) => {
                r.push_str("; ");
                r.push_str(&reason);
            }
            None => why.push((metric, reason)),
        }
    }
    for metric in ["int8-tc", "bf16-tc", "fp8-tc", "tf32-tc", "fp32-gemm"] {
        if !found.iter().any(|g| g.metric == metric) && !why.iter().any(|(m, _)| *m == metric) {
            why.push((metric, "no kernel for it serves this device".to_owned()));
        }
    }
    (found, why)
}

/// Tolerance of a sampled output, relative to the sum of its terms' magnitudes: a kernel that is right differs from
/// the f64 reference by its accumulation (and, for TF32, its operand) rounding; a wrong one by the size of the output.
const FP32_TOL: f64 = 1e-4;
const TF32_TOL: f64 = 2e-3;

/// The largest error of `got` (the full `m x n` output) at the sampled positions, relative to each output's term
/// magnitude, against `reference(r, c) -> (value, sum of |terms|)`; `Err` past `tol`.
fn check(got: &[f32], n: u32, positions: &[(u32, u32)], reference: impl Fn(u32, u32) -> (f64, f64), tol: f64) -> Result<String, String> {
    let mut worst = 0.0f64;
    for &(r, c) in positions {
        let (want, scale) = reference(r, c);
        let v = f64::from(got[r as usize * n as usize + c as usize]);
        if !v.is_finite() {
            return Err(format!("non-finite output at ({r}, {c})"));
        }
        worst = worst.max((v - want).abs() / scale.max(f64::MIN_POSITIVE));
    }
    if worst > tol {
        Err(format!("error {worst:.2e} of the term magnitude, over the {tol:.0e} bound"))
    } else {
        Ok(format!("within {worst:.1e} of the term magnitude of the f64 reference at {} sampled outputs", positions.len()))
    }
}

/// Run the product `step` once into `out` (poisoned first) and check it, then time it launch by launch.
#[allow(clippy::too_many_arguments)]
fn run(gpu: &Gpu, metric: &'static str, kernel: &str, steps: Vec<Step>, out: &DeviceBuffer, reference: &dyn Fn(u32, u32) -> (f64, f64), tol: f64, missing: &mut Vec<(&'static str, String)>) -> Option<SquareGemm> {
    let (m, n) = (SIDE, SIDE);
    gpu.write_f32(out, &vec![f32::NAN; (m * n) as usize]);
    gpu.submit(&[], &steps[..1]);
    gpu.poll_wait();
    let positions = sample_positions(m, n);
    let verified = match check(&gpu.read(out, (m * n) as usize), n, &positions, reference, tol) {
        Ok(v) => v,
        Err(why) => {
            missing.push((metric, format!("{kernel} REJECTED ({why})")));
            return None;
        }
    };
    let Some(stats) = launch_stats::measure(gpu, &steps) else {
        missing.push((metric, format!("{kernel}: no device timing")));
        return None;
    };
    Some(SquareGemm { metric, kernel: kernel.to_owned(), tera: ops() / (stats.median_ms * 1e-3) / 1e12, stats, verified, others: Vec::new() })
}

/// Launches timed per product.
const LAUNCHES: usize = 9;

/// The vendor-library products brain trains with (bf16 and TF32 tensor cores, fp32 CUDA cores) and brain's own fp32
/// GEMM, on `x x^T` of the square activations.
fn library_and_fp32(gpu: &Gpu, acts: &Acts, missing: &mut Vec<(&'static str, String)>) -> Vec<SquareGemm> {
    let (m, n, k) = (SIDE, SIDE, SIDE);
    let mut found = Vec::new();
    let out = gpu.storage(u64::from(m) * u64::from(n));
    let x = gpu.storage(u64::from(m) * u64::from(k));
    gpu.write_f32(&x, &acts.x);
    let exact = |r: u32, c: u32| {
        let (mut v, mut a) = (0.0f64, 0.0f64);
        for i in 0..k as usize {
            let t = acts.value(None, r, i) * acts.value(None, c, i);
            v += t;
            a += t.abs();
        }
        (v, a)
    };
    // brain's own fp32 GEMM, dispatched as the workspace dispatches it (`matmul_reg3`, redirected to the native kernel
    // where the backend has one).
    let tiles = crate::Dispatch::Workgroups(m.div_ceil(128) * n.div_ceil(128));
    let reg3 = (0..LAUNCHES).map(|_| gpu.dispatch(TUNED_SLOT_KERNELS.len(), &[&x, &x, &out], &[m, k, n], tiles)).collect();
    found.extend(run(gpu, "fp32-gemm", "matmul_reg3 (native matmul_f32_reg where served)", reg3, &out, &exact, FP32_TOL, missing));
    let lib = LibraryGemmProvider::for_gpu(gpu, GemmPrecision::Bf16TensorCore);
    if !lib.is_active() {
        let why = format!("{:?}", lib.path());
        missing.push(("tf32-tc", format!("no vendor-library tensor-core path: {why}")));
        return found;
    }
    fn whole(b: &DeviceBuffer) -> Operand<'_> {
        Operand { buf: b, range: (0, 0) }
    }
    for (metric, tf32, tol) in [("fp32-gemm", false, FP32_TOL), ("tf32-tc", true, TF32_TOL)] {
        let product = F32Product { m, n, k, trans_a: false, trans_b: true, alpha: 1.0, beta: 0.0, tf32 };
        let mut steps = Vec::new();
        for _ in 0..LAUNCHES {
            if !lib.gemm_f32(gpu, &mut steps, &whole(&x), &whole(&x), &whole(&out), product) {
                break;
            }
        }
        if steps.len() == LAUNCHES {
            found.extend(run(gpu, metric, if tf32 { "vendor library TF32 GEMM" } else { "vendor library fp32 GEMM" }, steps, &out, &exact, tol, missing));
        } else {
            missing.push((metric, "the vendor library declined the product".to_owned()));
        }
    }
    let xb = gpu.storage(u64::from(m) * u64::from(k) / 2);
    gpu.write(&xb, &acts.bf16);
    let bf16 = |r: u32, c: u32| {
        let (mut v, mut a) = (0.0f64, 0.0f64);
        for i in 0..k as usize {
            let t = acts.value(Some(ActFormat::Bf16), r, i) * acts.value(Some(ActFormat::Bf16), c, i);
            v += t;
            a += t.abs();
        }
        (v, a)
    };
    let mut steps = Vec::new();
    for _ in 0..LAUNCHES {
        if !lib.forward_bf16(gpu, &mut steps, &xb, &whole(&xb), &whole(&out), m, n, k) {
            break;
        }
    }
    if steps.len() == LAUNCHES {
        found.extend(run(gpu, "bf16-tc", "vendor library bf16 GEMM", steps, &out, &bf16, FP32_TOL, missing));
    } else {
        missing.push(("bf16-tc", "the vendor library declined the bf16 product".to_owned()));
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sampled output within the tolerance of its term magnitude passes and says how close it came; one far off,
    /// or not finite, is refused with the reason.
    #[test]
    fn a_square_product_is_checked_relative_to_its_term_magnitude() {
        let n = 4u32;
        let positions = [(0, 0), (1, 3)];
        let mut got = vec![0.0f32; 16];
        got[0] = 1.0;
        got[7] = 2.0;
        let reference = |r: u32, c: u32| if (r, c) == (0, 0) { (1.0 + 1e-6, 100.0) } else { (2.0, 10.0) };
        assert!(check(&got, n, &positions, reference, FP32_TOL).is_ok());
        got[7] = 3.0;
        assert!(check(&got, n, &positions, reference, FP32_TOL).unwrap_err().contains("over the"));
        got[7] = f32::NAN;
        assert!(check(&got, n, &positions, reference, FP32_TOL).unwrap_err().contains("non-finite"));
    }

    /// Per precision the fastest kernel is kept and the slower ones are named beside it.
    #[test]
    fn each_precision_keeps_its_fastest_kernel_and_names_the_rest() {
        let g = |metric, kernel: &str, tera| SquareGemm {
            metric,
            kernel: kernel.into(),
            stats: LaunchStats { median_ms: 1.0, min_ms: 1.0, stretched: 0.0, samples: 1 },
            tera,
            verified: String::new(),
            others: Vec::new(),
        };
        let mut found = Vec::new();
        keep(&mut found, g("int8-tc", "a", 100.0));
        keep(&mut found, g("int8-tc", "b", 300.0));
        keep(&mut found, g("bf16-tc", "c", 50.0));
        keep(&mut found, g("int8-tc", "d", 200.0));
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].kernel, "b");
        assert_eq!(found[0].others, vec!["a 100".to_owned(), "d 200".to_owned()]);
    }
}
