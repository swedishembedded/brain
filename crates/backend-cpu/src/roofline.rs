// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CPU backend's own roofline probe - real measured GFLOP/s **and GB/s**
//! over every core the process may use, not a guess. Both halves are needed to say which bound a kernel is under:
//! a decoder's steady state is GEMV-shaped at 0.5 FLOP/byte, so the compute
//! roof alone would grade every one of its hot kernels against the wrong
//! ceiling.
//!
//! `CpuBackend::caps` (`crate::lib::CpuBackend::caps`) explicitly leaves
//! `peak_gflops`/`peak_bandwidth_gbs` as `None` and defers to
//! `gpu_core::roof`, but `gpu_core::roof::ensure` skips the CPU device class
//! by default: its calibration loop has a known-bad interaction with this
//! crate's own rayon dispatch (see the comment on `roof::ensure` for the
//! deadlock it caused). Rather than fight that skip, this module measures a
//! CPU roofline directly, in three parts, each fast enough to be one rung of
//! `brain perf roofline`'s "first result within 10 seconds" report:
//!
//! * **the silicon** ([`measure_fma`]): one thread pinned to each allowed core,
//!   each running independent SIMD FMA chains held in registers (NEON on
//!   aarch64, AVX2+FMA on x86-64 where present), so the number is what the
//!   cores can issue together, not what one core or one kernel reaches;
//! * **memory** ([`measure_memory`]): STREAM copy and triad, one thread a core,
//!   each over arrays it touched first itself (so the pages sit on its own
//!   node), sized past every last-level cache;
//! * **brain's own kernel**: the min-of-N wall time over a representative NCHW
//!   conv2d mix lifted from `fast_conv::tests::bench_conv_gflops`, which says
//!   how far the CPU backend's real code sits under the silicon.
//!
//! [`published_peak`] derives the vendor figure the measurements are compared
//! with, for the CPUs it knows.
//!
//! Swedish Embedded AB implements fast, honest hardware-capacity reporting
//! for teams that need to know what an edge box can actually do before they
//! ship a model to it. If your team needs a roofline number for CPU, GPU, or
//! NPU targets, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::fast_conv::{conv2d, ConvParams};
use crate::par;
use std::sync::Barrier;
use std::time::Instant;

/// Measured CPU roofline over every core the process may use.
#[derive(Clone, Debug)]
pub struct CpuRoofline {
    /// Threads the FMA and memory probes ran, one pinned to each allowed core.
    pub threads: usize,
    /// The SIMD path the FMA probe issued (`neon`, `avx2+fma`, `scalar`).
    pub simd: &'static str,
    /// Peak fp32 rate of the SIMD FMA probe on all threads together, GFLOP/s.
    pub fma_gflops: f32,
    /// fp32 rate of brain's own conv2d over a representative mix, GFLOP/s:
    /// what the backend's real code reaches, graded against `fma_gflops`.
    pub conv_gflops: f32,
    /// STREAM copy (`a = b`), GB/s, counting one read and one write a word.
    pub copy_gbs: f32,
    /// STREAM triad (`a = b + s * c`), GB/s, counting two reads and one write a word.
    pub triad_gbs: f32,
    /// Bytes the memory probes stream over, every thread's arrays together.
    pub working_set_bytes: u64,
    /// The published peak of these cores, where the CPU is known.
    pub peak: Option<CpuPeak>,
}

/// A published (or derived-from-published) peak of the cores the process may use.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuPeak {
    /// cores x maximum clock x fp32 FLOP the core issues a cycle.
    pub fp32_gflops: f64,
    /// The memory's published bandwidth over the sockets these cores sit on, where known.
    pub memory_gbs: Option<f64>,
    pub source: String,
}

/// What the published peak is derived from: the identity of the cores and of the system around them.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuId {
    /// `(implementer, part)` from the core's ID register (aarch64), `None` elsewhere.
    pub part: Option<(u32, u32)>,
    /// The SoC's JEDEC identity (`/sys/devices/soc0/soc_id`), where the platform publishes one.
    pub soc: Option<String>,
    /// Cores the process may use.
    pub cores: usize,
    /// The cores' maximum clock, MHz.
    pub max_mhz: Option<f64>,
    /// NUMA nodes holding at least one of those cores (one per socket on the systems listed).
    pub cpu_nodes: usize,
}

impl CpuId {
    /// The core's microarchitecture name, where [`CORE_ISSUE`] lists it.
    pub fn core_name(&self) -> Option<&'static str> {
        let part = self.part?;
        CORE_ISSUE.iter().find(|c| c.part == part).map(|c| c.name)
    }
}

/// fp32 FLOP one core issues a cycle, per microarchitecture: SIMD FMA pipes x lanes x 2.
struct CoreIssue {
    part: (u32, u32),
    name: &'static str,
    flop_per_cycle: f64,
    source: &'static str,
}

const CORE_ISSUE: &[CoreIssue] = &[
    // Arm Neoverse V2 Software Optimization Guide: four 128-bit SIMD/FP pipes, each with FMA.
    CoreIssue { part: (0x41, 0xd4f), name: "Neoverse-V2", flop_per_cycle: 32.0, source: "Arm Neoverse V2 optimization guide (4 x 128-bit FMA pipes)" },
    // Arm Neoverse V1 Software Optimization Guide: two 256-bit SVE pipes (four 128-bit NEON), each with FMA.
    CoreIssue { part: (0x41, 0xd40), name: "Neoverse-V1", flop_per_cycle: 32.0, source: "Arm Neoverse V1 optimization guide (4 x 128-bit FMA pipes)" },
    // Arm Neoverse N2 Software Optimization Guide: two 128-bit SIMD/FP pipes with FMA.
    CoreIssue { part: (0x41, 0xd49), name: "Neoverse-N2", flop_per_cycle: 16.0, source: "Arm Neoverse N2 optimization guide (2 x 128-bit FMA pipes)" },
];

/// Memory bandwidth per socket, per SoC.
struct SocMemory {
    soc: &'static str,
    name: &'static str,
    gbs_per_socket: f64,
    source: &'static str,
}

// perf-number: vendor datasheet figure, the comparison column of `brain perf roofline`, never a measured claim.
const SOC_MEMORY: &[SocMemory] = &[SocMemory {
    soc: "jep106:036b:0241",
    name: "NVIDIA Grace",
    gbs_per_socket: 500.0,
    source: "NVIDIA Grace CPU datasheet (LPDDR5X, per Grace)",
}];

/// The published peak of the cores `id` describes: `None` for a core
/// microarchitecture [`CORE_ISSUE`] does not list, or with no maximum clock.
pub fn derive_peak(id: &CpuId) -> Option<CpuPeak> {
    let core = CORE_ISSUE.iter().find(|c| Some(c.part) == id.part)?;
    let mhz = id.max_mhz?;
    let soc = id.soc.as_deref().and_then(|s| SOC_MEMORY.iter().find(|m| m.soc == s));
    let memory_gbs = soc.map(|m| m.gbs_per_socket * id.cpu_nodes.max(1) as f64);
    let mut source = format!("{} cores x {mhz:.0} MHz x {} FLOP/cycle: {}", id.cores, core.flop_per_cycle, core.source);
    if let Some(m) = soc {
        source += &format!("; {} x {} memory: {}", id.cpu_nodes.max(1), m.name, m.source);
    }
    Some(CpuPeak { fp32_gflops: id.cores as f64 * mhz * 1e-3 * core.flop_per_cycle, memory_gbs, source })
}

/// The identity of the cores this process may use, read from the kernel.
pub fn identify() -> CpuId {
    let cores = par::allowed_cores();
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |key: &str| {
        cpuinfo.lines().find(|l| l.split(':').next().is_some_and(|k| k.trim() == key)).and_then(|l| l.split(':').nth(1)).map(|v| v.trim().to_owned())
    };
    let hex = |v: String| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok();
    let part = field("CPU implementer").and_then(hex).zip(field("CPU part").and_then(hex));
    let soc = std::fs::read_to_string("/sys/devices/soc0/soc_id").ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
    let first = cores.first().copied().unwrap_or(0);
    let max_mhz = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{first}/cpufreq/cpuinfo_max_freq")).ok().and_then(|s| s.trim().parse::<f64>().ok()).map(|khz| khz / 1e3);
    let cpu_nodes = cpu_nodes_of(&cores);
    CpuId { part, soc, cores: cores.len(), max_mhz, cpu_nodes }
}

/// NUMA nodes holding at least one of `cores` (1 where the kernel publishes no topology).
fn cpu_nodes_of(cores: &[usize]) -> usize {
    let Ok(dir) = std::fs::read_dir("/sys/devices/system/node") else { return 1 };
    let nodes = dir
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().strip_prefix("node").is_some_and(|n| n.parse::<u32>().is_ok()))
        .filter(|e| std::fs::read_to_string(e.path().join("cpulist")).is_ok_and(|l| parse_cpulist(&l).iter().any(|c| cores.contains(c))))
        .count();
    nodes.max(1)
}

/// A kernel CPU list (`0-71,144,150-151`) as core numbers.
fn parse_cpulist(list: &str) -> Vec<usize> {
    list.trim()
        .split(',')
        .filter(|r| !r.is_empty())
        .flat_map(|r| match r.split_once('-') {
            Some((a, b)) => a.parse::<usize>().ok().zip(b.parse::<usize>().ok()).map_or(0..0, |(a, b)| a..b + 1),
            None => r.parse::<usize>().map_or(0..0, |a| a..a + 1),
        })
        .collect()
}

/// The last-level cache bytes the allowed cores share, summed over every distinct cache: what the memory probe's
/// working set must be well past. `None` where the kernel publishes no cache topology.
pub fn last_level_cache_bytes() -> Option<u64> {
    let mut seen: Vec<String> = Vec::new();
    let mut total = 0u64;
    for core in par::allowed_cores() {
        let base = format!("/sys/devices/system/cpu/cpu{core}/cache");
        let level = |i: usize| std::fs::read_to_string(format!("{base}/index{i}/level")).ok().and_then(|l| l.trim().parse::<u32>().ok());
        let Some(last) = (0..8).filter(|&i| level(i).is_some()).max_by_key(|&i| level(i)) else { continue };
        let shared = std::fs::read_to_string(format!("{base}/index{last}/shared_cpu_list")).unwrap_or_default();
        if seen.contains(&shared) {
            continue;
        }
        let size = std::fs::read_to_string(format!("{base}/index{last}/size")).ok()?;
        let size = size.trim();
        let bytes = match size.strip_suffix('K') {
            Some(k) => k.parse::<u64>().ok()? << 10,
            None => match size.strip_suffix('M') {
                Some(m) => m.parse::<u64>().ok()? << 20,
                None => size.parse::<u64>().ok()?,
            },
        };
        total += bytes;
        seen.push(shared);
    }
    (total > 0).then_some(total)
}

/// Everything at once: the FMA probe, the memory probes and brain's conv mix,
/// over every core the process may use. A few seconds at most.
pub fn measure() -> CpuRoofline {
    let cores = par::allowed_cores();
    let (fma_gflops, simd) = measure_fma(&cores);
    let mem = measure_memory(&cores);
    let id = identify();
    CpuRoofline {
        threads: cores.len(),
        simd,
        fma_gflops,
        conv_gflops: measure_conv(),
        copy_gbs: mem.copy_gbs,
        triad_gbs: mem.triad_gbs,
        working_set_bytes: mem.working_set_bytes,
        peak: derive_peak(&id),
    }
}

/// Independent accumulators per thread: enough to cover the FMA latency of
/// every pipe (four pipes x four cycles on the widest core listed).
const FMA_CHAINS: usize = 16;

/// Grow the FMA trip count until one run takes at least this long.
const MIN_FMA_SECONDS: f64 = 0.05;

/// fp32 FLOP one thread issues per iteration of the FMA chain, and the SIMD path it runs.
fn fma_shape() -> (u64, &'static str) {
    #[cfg(target_arch = "aarch64")]
    {
        ((FMA_CHAINS * 4 * 2) as u64, "neon")
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            ((FMA_CHAINS * 8 * 2) as u64, "avx2+fma")
        } else {
            ((FMA_CHAINS * 2) as u64, "scalar")
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        ((FMA_CHAINS * 2) as u64, "scalar")
    }
}

/// `iters` iterations of [`FMA_CHAINS`] independent SIMD FMA chains, `acc += x * y` with a product far below the
/// accumulator's resolution growth (no overflow, no denormals, nothing the compiler may fold: `y` is opaque to it),
/// returning a value the caller keeps observable. The accumulator is the instruction's addend, so each step is one
/// fused instruction in place, with no register copy beside it.
fn fma_chains(iters: u64) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32};
        // SAFETY: NEON is part of the aarch64 base architecture; the intrinsics only read and write registers.
        unsafe {
            let (x, y) = (vdupq_n_f32(1.0), vdupq_n_f32(std::hint::black_box(1e-12)));
            let one = vdupq_n_f32(1.0);
            // Sixteen named accumulators, each its own register for the whole loop.
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (one, one, one, one, one, one, one, one);
            let (mut a8, mut a9, mut a10, mut a11, mut a12, mut a13, mut a14, mut a15) = (one, one, one, one, one, one, one, one);
            for _ in 0..iters {
                a0 = vfmaq_f32(a0, x, y);
                a1 = vfmaq_f32(a1, x, y);
                a2 = vfmaq_f32(a2, x, y);
                a3 = vfmaq_f32(a3, x, y);
                a4 = vfmaq_f32(a4, x, y);
                a5 = vfmaq_f32(a5, x, y);
                a6 = vfmaq_f32(a6, x, y);
                a7 = vfmaq_f32(a7, x, y);
                a8 = vfmaq_f32(a8, x, y);
                a9 = vfmaq_f32(a9, x, y);
                a10 = vfmaq_f32(a10, x, y);
                a11 = vfmaq_f32(a11, x, y);
                a12 = vfmaq_f32(a12, x, y);
                a13 = vfmaq_f32(a13, x, y);
                a14 = vfmaq_f32(a14, x, y);
                a15 = vfmaq_f32(a15, x, y);
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15].iter().map(|a| vaddvq_f32(*a)).sum()
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the features were detected at run time just above.
            return unsafe { fma_chains_avx2(iters) };
        }
        fma_chains_scalar(iters)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        fma_chains_scalar(iters)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_chains_avx2(iters: u64) -> f32 {
    use std::arch::x86_64::{__m256, _mm256_fmadd_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let (x, y) = (_mm256_set1_ps(1.0), _mm256_set1_ps(std::hint::black_box(1e-12)));
    let mut acc: [__m256; FMA_CHAINS] = [_mm256_set1_ps(1.0); FMA_CHAINS];
    for _ in 0..iters {
        for a in acc.iter_mut() {
            *a = _mm256_fmadd_ps(x, y, *a);
        }
    }
    let mut out = [0f32; 8];
    let mut total = 0.0;
    for a in acc {
        // SAFETY: `out` holds eight f32.
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), a) };
        total += out.iter().sum::<f32>();
    }
    total
}

#[cfg(not(target_arch = "aarch64"))]
fn fma_chains_scalar(iters: u64) -> f32 {
    let y = std::hint::black_box(1e-12f32);
    let mut acc = [1.0f32; FMA_CHAINS];
    for _ in 0..iters {
        for a in acc.iter_mut() {
            *a = 1.0f32.mul_add(y, *a);
        }
    }
    acc.iter().sum()
}

/// Wall time of one synchronised run on every core: each thread waits at a barrier, runs `work(i)`, and waits again;
/// thread 0 times the span between the two, which covers the slowest thread.
fn timed_on_every_core(cores: &[usize], work: impl Fn(usize) + Sync) -> f64 {
    let barrier = Barrier::new(cores.len());
    let times = par::per_core(cores, |i| {
        barrier.wait();
        let t = Instant::now();
        work(i);
        barrier.wait();
        t.elapsed().as_secs_f64()
    });
    times[0]
}

/// Peak fp32 GFLOP/s of the SIMD FMA probe with one thread pinned to each of `cores`, and the SIMD path it ran.
pub fn measure_fma(cores: &[usize]) -> (f32, &'static str) {
    let (flop_per_iter, simd) = fma_shape();
    // Calibrate on one core: long enough that thread start-up and the barrier are noise.
    let mut iters: u64 = 1 << 14;
    loop {
        let t = Instant::now();
        std::hint::black_box(fma_chains(iters));
        let secs = t.elapsed().as_secs_f64();
        if secs >= MIN_FMA_SECONDS || iters >= 1 << 32 {
            break;
        }
        iters = ((iters as f64 * MIN_FMA_SECONDS / secs.max(1e-9)).ceil() as u64).max(iters * 2);
    }
    let best = (0..3)
        .map(|_| timed_on_every_core(cores, |_| {
            std::hint::black_box(fma_chains(iters));
        }))
        .fold(f64::INFINITY, f64::min);
    let flop = cores.len() as f64 * iters as f64 * flop_per_iter as f64;
    ((flop / best / 1e9) as f32, simd)
}

/// fp32 words per array per thread: three arrays of 4 MiB a thread is past a core's share of any last-level cache
/// listed, and the whole working set grows with the core count, so a wide machine is not measured in its caches.
const STREAM_FLOATS_PER_THREAD: usize = 1 << 20;

/// Timed repeats of each memory probe; the fastest (least contended) is kept.
const STREAM_REPEATS: usize = 4;

/// The memory probes' results.
pub struct Memory {
    pub copy_gbs: f32,
    pub triad_gbs: f32,
    pub working_set_bytes: u64,
}

/// STREAM copy and triad with one thread pinned to each of `cores`, each over three arrays it allocated and touched
/// first itself, so their pages are on its own node. STREAM's counting: copy moves 8 bytes a word, triad 12.
pub fn measure_memory(cores: &[usize]) -> Memory {
    let n = STREAM_FLOATS_PER_THREAD;
    let threads = cores.len();
    // Each thread's arrays live in its own slot, allocated and first written on that thread.
    let slots: Vec<std::sync::Mutex<Option<[Vec<f32>; 3]>>> = (0..threads).map(|_| std::sync::Mutex::new(None)).collect();
    par::per_core(cores, |i| {
        let fill = |seed: usize| (0..n).map(|j| ((i + j + seed) % 17) as f32).collect::<Vec<f32>>();
        *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some([fill(0), fill(5), fill(11)]);
    });
    let run = |triad: bool| -> f64 {
        (0..STREAM_REPEATS)
            .map(|_| {
                timed_on_every_core(cores, |i| {
                    let mut slot = slots[i].lock().unwrap_or_else(|e| e.into_inner());
                    let [a, b, c] = slot.as_mut().expect("filled above");
                    if triad {
                        for ((a, b), c) in a.iter_mut().zip(b.iter()).zip(c.iter()) {
                            *a = *b + 1.000_001 * *c;
                        }
                    } else {
                        a.copy_from_slice(b);
                    }
                    std::hint::black_box(&a[n - 1]);
                })
            })
            .fold(f64::INFINITY, f64::min)
    };
    let copy = run(false);
    let triad = run(true);
    let words = (threads * n) as f64;
    Memory {
        copy_gbs: (words * 8.0 / copy / 1e9) as f32,
        triad_gbs: (words * 12.0 / triad / 1e9) as f32,
        working_set_bytes: (threads * n * 3 * 4) as u64,
    }
}

/// (cin, h, w, cout, k, stride, pad) - three of `bench_conv_gflops`'s real
/// yolov8n@640 shapes (the strided 3x3 stem, a steady-state 3x3, and a 1x1
/// pointwise conv), chosen to keep the mix well under a second while
/// still spanning the stride/kernel mix that shapes the aggregate number.
const SHAPES: [(usize, usize, usize, usize, usize, usize, usize); 3] = [
    (3, 640, 640, 16, 3, 2, 1),  // stem: 3->16 3x3 s2 @640
    (64, 80, 80, 64, 3, 1, 1),   // steady state: 64->64 3x3 s1 @80
    (128, 80, 80, 128, 1, 1, 0), // pointwise: 1x1 128->128 @80
];

/// Timed repeats per shape; the contention-robust minimum is kept, same as
/// `bench_conv_gflops`. Lower than that ignored bench's `n=30` on purpose -
/// this must run in well under a second, and a min-of-6 is already stable on
/// real hardware (see `crates/backend-cpu/tests/it/roofline.rs` for the
/// cross-check against the full ignored bench's real output).
const REPEATS: usize = 6;

/// Same deterministic LCG as `fast_conv::tests::lcg` / `bench_conv_gflops` -
/// input values are irrelevant to a throughput measurement, only their shape
/// is, so there is no reason for this to differ from the proven bench.
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn params(cin: usize, h: usize, w: usize, cout: usize, k: usize, stride: usize, pad: usize) -> ConvParams {
    let ho = (h + 2 * pad - k) / stride + 1;
    let wo = (w + 2 * pad - k) / stride + 1;
    ConvParams { n: 1, cin, h, w, cout, k, stride, pad, dilation: 1, ho, wo }
}

/// This host's real fp32 conv throughput via `fast_conv::conv2d` (SIMD GEMM +
/// the pool's tile parallelism, the same routine every CPU conv dispatch
/// actually runs), GFLOP/s.
pub fn measure_conv() -> f32 {
    let mut total_flop = 0.0f64;
    let mut total_min = 0.0f64;
    for &(cin, h, w, cout, k, stride, pad) in &SHAPES {
        let p = params(cin, h, w, cout, k, stride, pad);
        let mut s = 999u32;
        let x: Vec<f32> = (0..p.x_len()).map(|_| lcg(&mut s)).collect();
        let wt: Vec<f32> = (0..p.w_len()).map(|_| lcg(&mut s)).collect();
        let mut y = vec![0.0f32; p.y_len()];
        conv2d(&p, &x, &wt, &mut y); // warm - first call pays any one-time cost
        let mut best = f64::INFINITY;
        for _ in 0..REPEATS {
            let t = Instant::now();
            conv2d(&p, &x, &wt, &mut y);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let flop = 2.0 * (p.cout * p.cin * p.k * p.k * p.ho * p.wo) as f64;
        total_flop += flop;
        total_min += best;
    }
    (total_flop / total_min / 1e9) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grace(cores: usize, nodes: usize) -> CpuId {
        CpuId { part: Some((0x41, 0xd4f)), soc: Some("jep106:036b:0241".into()), cores, max_mhz: Some(3366.0), cpu_nodes: nodes }
    }

    #[test]
    fn a_grace_peak_is_its_cores_times_clock_times_issue_width_and_its_sockets_memory() {
        let p = derive_peak(&grace(288, 4)).expect("Neoverse V2 is listed");
        assert!((p.fp32_gflops - 288.0 * 3.366 * 32.0).abs() < 1e-6, "{p:?}");
        assert_eq!(p.memory_gbs, Some(2000.0));
        let one = derive_peak(&grace(72, 1)).unwrap();
        assert!((p.fp32_gflops / one.fp32_gflops - 4.0).abs() < 1e-9);
        assert_eq!(one.memory_gbs, Some(500.0));
    }

    #[test]
    fn an_unknown_core_or_soc_has_no_figure_rather_than_a_guess() {
        assert_eq!(derive_peak(&CpuId { part: Some((0x41, 0xfff)), ..grace(8, 1) }), None);
        assert_eq!(derive_peak(&CpuId { max_mhz: None, ..grace(8, 1) }), None);
        let other_soc = derive_peak(&CpuId { soc: None, ..grace(8, 1) }).unwrap();
        assert_eq!(other_soc.memory_gbs, None, "the same core in another system has its own memory");
    }

    #[test]
    fn a_cpu_list_parses_ranges_and_singles() {
        assert_eq!(parse_cpulist("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpulist(""), Vec::<usize>::new());
    }
}
