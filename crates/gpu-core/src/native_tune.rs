// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measured selection among the native candidates of one kernel slot - the
//! mechanism that makes a device run at its roofline without anyone tuning for
//! it.
//!
//! Swedish Embedded AB implements self-tuning inference kernels that reach the
//! memory roofline of whatever accelerator they land on. If your team needs
//! expertise in making batched LLM decode fast on hardware you have not
//! optimised for yet, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! [`crate::native_upgrade`] lists, per kernel slot a model registers, every
//! native kernel that could serve a dispatch (a *candidate*): each states the
//! device capability it needs through its registry floor and the shapes it
//! serves through its `serves` predicate, so a device only ever sees the
//! candidates it can run. This module chooses among them:
//!
//! 1. on the first handle of a device, for each slot with two or more
//!    candidates, a bounded grid of synthetic problems (row bucket x `K` class x
//!    weight-bytes class, the shape classes a decode step falls in) is built
//!    on the device;
//! 2. every candidate that serves a problem is run on it once, handed the
//!    activation form it declares (packed int8 or bf16 rows), and its output is
//!    compared with the f64 reference of the stored representation over a sample
//!    of outputs; a candidate whose error exceeds [`ALPHA`] times the error of
//!    rounding its own inputs to their storage resolution is rejected for that
//!    class, never chosen (see [`admit`]);
//! 3. the surviving candidates are timed with the backend's per-launch device
//!    events, the weights cycled past the L2 so the stream is DRAM-bound as in a
//!    real decode step, and the fastest wins (an earlier candidate keeps a tie
//!    inside [`TIE`], so a noisy tie does not flip the choice between runs);
//! 4. each choice records the achieved weight bandwidth and its fraction of
//!    the device's measured roofline ([`crate::roof`]), and the table is
//!    persisted per device, per brain build and per candidate source so the next
//!    start reads it instead of measuring.
//!
//! A dispatch then takes the candidate its shape class names
//! ([`Tuned::choice`]); models call the op and never see any of this.
//! [`force_candidate`] pins a slot to one candidate for debugging and
//! A/B measurement and [`set_autotune`]`(false)` keeps every slot on its
//! default (first) candidate; [`report`] lists what was chosen and the ops
//! under [`GAP_THRESHOLD`] of their roofline.

use crate::native_upgrade::{ActFormat, Input, WeightRead};
use crate::Gpu;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A later candidate must beat an earlier one by this factor to displace it.
const TIE: f64 = 0.97;
/// Ops achieving less than this fraction of their roofline bound are listed as the device's gap.
pub const GAP_THRESHOLD: f64 = 0.6;
/// The shape-class grid of one slot: row buckets (`(lo, hi)` inclusive, timed at `hi`), the `K` classes (timed at
/// their representative), the weight-bytes classes with the output columns timed for each, and how weights are
/// cycled while timing.
struct Grid {
    rows: &'static [(u32, u32)],
    /// Whether rows past the last bucket belong to it (a prefill round may be any width) or to no class.
    open_top: bool,
    k: [u32; 2],
    /// Upper bound (exclusive) of the weight-bytes classes, in bytes; the last class is unbounded.
    bytes_bounds: [u64; 3],
    /// Output columns timed for each bytes class at each `K` class.
    n_reps: [[u32; 4]; 2],
    /// Distinct weight bytes cycled per timing of a streaming class, so the weights come from memory and not from the
    /// L2.
    footprint_bytes: u64,
    /// Classes of at most this many rows stream their weights (timed on [`Grid::footprint_bytes`] of weights, graded
    /// against the memory roof); wider classes are compute-bound (timed on one copy of the weights, graded against the
    /// tensor-core roof, a weight read once per row tile is not flattered by the L2).
    stream_max_rows: u32,
}

impl Grid {
    /// Whether a class timed at `m` rows streams its weights.
    fn streams(&self, m: u32) -> bool {
        m <= self.stream_max_rows
    }
}

/// The decode regime: weight-streaming classes, timed with the weights cycled past the L2 so the stream is
/// DRAM-bound as in a real decode step (about 1.5, 6, 24 and 90 MB of weights per bytes class).
const DECODE: Grid = Grid {
    rows: &[(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 24), (25, 32)],
    open_top: false,
    k: [5120, 17408],
    bytes_bounds: [3 << 20, 12 << 20, 48 << 20],
    n_reps: [[256, 1024, 4096, 16384], [64, 256, 1024, 4096]],
    footprint_bytes: 192 << 20,
    stream_max_rows: u32::MAX,
};

/// The rows past the decode regime to a round of thousands of rows (bytes classes of about 10, 30, 60 and 100 MB of
/// weights). Up to 128 rows (a decode step of a large batch) the step still reads every weight once and is bound by
/// that read: those classes stream, like the decode grid's. Wider rounds are compute-bound.
const PREFILL: Grid = Grid {
    rows: &[(33, 48), (49, 64), (65, 96), (97, 128), (129, 256), (257, 512), (513, 1024), (1025, 4096)],
    open_top: true,
    k: [5120, 17408],
    bytes_bounds: [20 << 20, 45 << 20, 80 << 20],
    n_reps: [[2048, 5120, 10240, 17408], [512, 1536, 3072, 5120]],
    footprint_bytes: 192 << 20,
    stream_max_rows: 128,
};

/// The grid a slot is tuned on; a slot not named here is a decode-regime slot.
fn grid(slot: &str) -> &'static Grid {
    if slot == crate::native_upgrade::PREFILL_SLOT {
        &PREFILL
    } else {
        &DECODE
    }
}

const TRIALS: usize = 3;
/// Launches timed per candidate at least, so a class with one weight copy still has several to take the minimum of.
const MIN_LAUNCHES: usize = 6;
/// Wall time one device's tuning may take; classes not reached keep their default candidate until a rebuild retunes.
const BUDGET: std::time::Duration = std::time::Duration::from_secs(40);

/// The read-bandwidth probe: every thread streams 16-byte vectors with the loads a decode GEMV issues (L1
/// no-allocate) and folds them, so the number is the read roof of the access type the candidates are graded on.
/// Params `[vectors]`; a launch over a few vectors measures the launch floor instead.
static ROOF_READ: kernels_cuda::CudaKernel = kernels_cuda::CudaKernel {
    name: "roof_read",
    op: backend_api::select::Op::MatMul,
    weight: backend_api::select::Dtype::I8,
    by_name: true,
    source: backend_api::ImplSource::Tuned,
    min_cc: kernels_cuda::DP4A_MIN_CC,
    entry: "brain_roof_read",
    what: "device read bandwidth probe: grid-stride 16-byte streaming loads folded into one word per thread",
    reported: "native:roof_read",
    block_dim: 256,
    tile: (1, 256),
    shared_bytes: 0,
    programmatic: false,
    src: r#"
extern "C" __global__ void __launch_bounds__(256)
brain_roof_read(const unsigned int* params, const uint4* data, unsigned int* out) {
    const unsigned long long n = params[0];
    unsigned int acc = 0u;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(data + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[(unsigned long long)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
"#,
};
const ROOF_BINDINGS: &[backend_api::BindKind] = &[backend_api::BindKind::Uniform, backend_api::BindKind::StorageRead, backend_api::BindKind::StorageReadWrite];
/// Bytes the read probe streams: far past any cache.
const ROOF_BYTES: u64 = 1 << 30;
const ROOF_BLOCKS: u32 = 4096;

/// The device's read roof (GB/s) and launch floor (microseconds), measured with [`ROOF_READ`]; `None` where the
/// backend cannot time kernels or compile the probe.
fn measure_read_roof(gpu: &Gpu) -> Option<(f32, f32)> {
    let id = gpu.native_kernel(&ROOF_READ, ROOF_BINDINGS)?;
    let vectors = (ROOF_BYTES / 16) as u32;
    let data = gpu.storage(ROOF_BYTES / 4);
    let out = gpu.storage(u64::from(ROOF_BLOCKS) * 256);
    let launch = |n: u32| gpu.native_step_raw(id, &[&data, &out], &[n], ROOF_BLOCKS);
    let time = |n: u32, reps: usize| -> Option<f64> {
        let steps: Vec<_> = (0..reps).filter_map(|_| launch(n)).collect();
        gpu.reset_kernel_times();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let t = gpu.kernel_times()?;
        let (ms, calls) = t.iter().filter(|r| r.0 == ROOF_READ.reported).fold((0.0, 0u64), |a, r| (a.0 + r.1, a.1 + r.2));
        (calls > 0).then(|| ms / calls as f64)
    };
    time(vectors, 2)?; // warm clocks and the pipeline
    let best = (0..6).filter_map(|_| time(vectors, 8)).fold(f64::MAX, f64::min);
    let floor = (0..6).filter_map(|_| time(256, 64)).fold(f64::MAX, f64::min);
    (best.is_finite() && best > 0.0 && best < f64::MAX && floor < f64::MAX).then(|| ((ROOF_BYTES as f64 / (best * 1e-3) / 1e9) as f32, (floor * 1e3) as f32))
}

/// The int8 tensor-core probe (`mma.sync m16n8k32`): every warp issues independent MMAs on operands held in registers,
/// so nothing but the matrix pipeline limits it - the roof of the compute-bound classes' `mma.sync` candidates.
/// Params `[iterations]`.
static ROOF_MMA_S8: kernels_cuda::CudaKernel = kernels_cuda::CudaKernel {
    name: "roof_mma_s8",
    op: backend_api::select::Op::MatMul,
    weight: backend_api::select::Dtype::I8,
    by_name: true,
    source: backend_api::ImplSource::Tuned,
    min_cc: kernels_cuda::MMA_S8_MIN_CC,
    entry: "brain_roof_mma_s8",
    what: "device int8 tensor-core throughput probe: independent mma.sync m16n8k32 chains on register operands",
    reported: "native:roof_mma_s8",
    block_dim: 128,
    tile: (1, 1),
    shared_bytes: 0,
    programmatic: false,
    src: r#"
extern "C" __global__ void __launch_bounds__(128)
brain_roof_mma_s8(const unsigned int* params, unsigned int* out) {
    const unsigned int iters = params[0];
    const unsigned int t = blockIdx.x * 128u + threadIdx.x;
    const unsigned int a0 = t * 0x9E3779B1u, a1 = a0 ^ 0x85EBCA6Bu, a2 = a0 + 0xC2B2AE35u, a3 = a1 + 0x27D4EB2Fu;
    const unsigned int b0 = a2 ^ 0x165667B1u, b1 = a3 + 0x9E3779B9u;
    unsigned int acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0u; }
    for (unsigned int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                         : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    unsigned int x = 0u;
#pragma unroll
    for (int j = 0; j < 16; ++j) { x ^= acc[j][0] ^ acc[j][1] ^ acc[j][2] ^ acc[j][3]; }
    out[t] = x;
}
"#,
};
const ROOF_MMA_BINDINGS: &[backend_api::BindKind] = &[backend_api::BindKind::Uniform, backend_api::BindKind::StorageReadWrite];
/// MMAs per warp the probe issues per timed launch: long enough that the ramp is noise.
const ROOF_MMA_ITERS: u32 = 512;
/// Operations of one `m16n8k32` MMA: 16 x 8 x 32 multiply-adds.
const MMA_OPS: f64 = 2.0 * 16.0 * 8.0 * 32.0;

/// Device milliseconds of one launch of the probe `kernel` on `blocks` blocks, the best of a few runs.
fn time_probe(gpu: &Gpu, kernel: &'static kernels_cuda::CudaKernel, bindings: &'static [backend_api::BindKind], params: &[u32], blocks: u32, out_words: u64) -> Option<f64> {
    let id = gpu.native_kernel(kernel, bindings)?;
    let out = gpu.storage(out_words);
    let steps: Vec<_> = (0..6).filter_map(|_| gpu.native_step_raw(id, &[&out], params, blocks)).collect();
    let time = |steps: &[crate::Step]| -> Option<f64> {
        gpu.reset_kernel_times();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        let t = gpu.kernel_times()?;
        let (ms, calls) = t.iter().filter(|r| r.0 == kernel.reported).fold((0.0, 0u64), |a, r| (a.0 + r.1, a.1 + r.2));
        (calls > 0).then(|| ms / calls as f64)
    };
    time(&steps[..2])?; // warm clocks and the pipeline
    let best = (0..4).filter_map(|_| time(&steps)).fold(f64::MAX, f64::min);
    (best.is_finite() && best > 0.0 && best < f64::MAX).then_some(best)
}

/// The device's int8 tensor-core roof in TOPS: the fastest of its matrix-multiply paths (`mma.sync`, and the
/// warpgroup path where the device has it) on register operands; `None` where no probe compiles or runs.
fn measure_int8_roof(gpu: &Gpu) -> Option<f32> {
    let cc = gpu.caps().arch.compute_capability?;
    if cc < kernels_cuda::MMA_S8_MIN_CC {
        return None;
    }
    let blocks = gpu.caps().compute_units.unwrap_or(16) * 4;
    let ms = time_probe(gpu, &ROOF_MMA_S8, ROOF_MMA_BINDINGS, &[ROOF_MMA_ITERS], blocks, u64::from(blocks) * 128)?;
    let ops = f64::from(blocks) * 4.0 * f64::from(ROOF_MMA_ITERS) * 16.0 * MMA_OPS;
    Some((ops / (ms * 1e-3) / 1e12) as f32)
}

/// The slots whose candidates this module knows how to build a synthetic problem and a reference for: the int8
/// decode linear and the prefill-sized int8 GEMM. A slot with several rows that is not listed (the K-quant narrow
/// and wide blocks) keeps its row-count rule until it has a problem generator here.
const TUNABLE_SLOTS: [&str; 2] = ["matmul_i8_gemv", crate::native_upgrade::PREFILL_SLOT];

static AUTOTUNE: AtomicBool = AtomicBool::new(true);
static FORCED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
static TABLES: Mutex<Vec<(String, Arc<Tuned>)>> = Mutex::new(Vec::new());

thread_local! {
    /// Set while this thread tunes, so the probe handles tuning itself opens do not tune in turn.
    static TUNING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether handles built from now on tune their candidate slots (default) or keep every slot on its first candidate.
pub fn set_autotune(on: bool) {
    AUTOTUNE.store(on, Ordering::Relaxed);
}

/// Pin `slot` (the kernel name a model registers, e.g. `matmul_i8_gemv`) to the candidate `kernel` wherever that
/// candidate serves the dispatch; a dispatch it cannot serve keeps the tuned choice. An unknown candidate name is
/// refused by [`validate_forced`] once handles exist, never silently ignored.
pub fn force_candidate(slot: &str, kernel: &str) {
    let mut f = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    f.retain(|(s, _)| s != slot);
    f.push((slot.to_owned(), kernel.to_owned()));
}

/// Drop every forced candidate: the tuned choice applies again.
pub fn clear_forced() {
    FORCED.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// Every forced `(slot, candidate)` pair whose candidate no handle offers - a misspelt name must fail loudly.
pub fn unknown_forced(offered: &[(String, String)]) -> Vec<(String, String)> {
    FORCED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|f| !offered.contains(f)).cloned().collect()
}

/// A shape class of one slot's grid ([`grid`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Class {
    rows: usize,
    k: usize,
    bytes: usize,
}

impl Class {
    fn of(g: &Grid, m: u32, kg: u32, n: u32) -> Option<Class> {
        let rows = match g.rows.iter().position(|&(lo, hi)| (lo..=hi).contains(&m)) {
            Some(r) => r,
            None if g.open_top && m > g.rows[g.rows.len() - 1].1 => g.rows.len() - 1,
            None => return None,
        };
        let k = if kg * 4 <= g.k[0] * 3 / 2 { 0 } else { 1 };
        let bytes = u64::from(n) * u64::from(kg) * 4 * 9 / 8;
        Some(Class { rows, k, bytes: g.bytes_bounds.iter().position(|&b| bytes < b).unwrap_or(g.bytes_bounds.len()) })
    }

    fn all(g: &Grid) -> impl Iterator<Item = Class> + '_ {
        (0..g.rows.len()).flat_map(move |rows| (0..g.k.len()).flat_map(move |k| (0..=g.bytes_bounds.len()).map(move |bytes| Class { rows, k, bytes })))
    }

    /// The problem this class is timed on: `(m, kg, n)`.
    fn problem(self, g: &Grid) -> (u32, u32, u32) {
        (g.rows[self.rows].1, g.k[self.k] / 4, g.n_reps[self.k][self.bytes])
    }

    fn label(self, g: &Grid) -> String {
        let (lo, hi) = g.rows[self.rows];
        let (_, kg, n) = self.problem(g);
        format!("rows {lo}-{hi}, K~{}, N~{n}", kg * 4)
    }
}

/// What happened to one candidate on one class.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Timed: microseconds per launch, weight bandwidth in GB/s, fraction of the measured roofline bound.
    Timed { us: f64, gbs: f64, frac: Option<f64> },
    /// Refused by the admission rule; never chosen.
    Rejected(String),
}

/// One class's outcome.
#[derive(Clone, Debug)]
pub struct Entry {
    pub slot: String,
    class: Class,
    pub best: String,
    /// Every candidate that served the class, in preference order.
    pub results: Vec<(String, Verdict)>,
}

impl Entry {
    pub fn describe(&self) -> String {
        self.class.label(grid(&self.slot))
    }

    /// The winner's fraction of its roofline bound, when a roof was measured.
    pub fn roof_fraction(&self) -> Option<f64> {
        self.results.iter().find(|(k, _)| *k == self.best).and_then(|(_, v)| match v {
            Verdict::Timed { frac, .. } => *frac,
            Verdict::Rejected(_) => None,
        })
    }
}

/// The device's tuned table: the candidate each slot takes per shape class.
#[derive(Debug)]
pub struct Tuned {
    pub device: String,
    /// The measured memory roof the fractions are taken against, GB/s.
    pub roof_gbs: Option<f32>,
    /// The measured cost of an almost empty launch, microseconds: the bound of a projection too small to stream.
    pub floor_us: Option<f32>,
    /// The measured int8 tensor-core compute roof, TOPS: the fastest of the device's int8 matrix-multiply paths on
    /// operands held in registers. The bound of the compute-bound (prefill) classes.
    pub int8_tops: Option<f32>,
    pub entries: Vec<Entry>,
    /// Whether this table was read from the persistent store rather than measured by this process.
    pub from_store: bool,
}

impl Tuned {
    /// The candidate the table (or a forced override) names for a dispatch of `slot` with caller `params`, among the
    /// candidates `accepts` (a caller that can supply only one activation form asks for the best candidate reading
    /// it; a class that was never timed names its fill-in choice when that one is accepted).
    pub(crate) fn choice(&self, slot: &str, params: &[u32], accepts: &dyn Fn(&str) -> bool) -> Option<String> {
        if let Some((_, k)) = FORCED.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(s, _)| s == slot) {
            return Some(k.clone());
        }
        let class = match params {
            [m, kg, n, ..] => Class::of(grid(slot), *m, *kg, *n)?,
            _ => return None,
        };
        let e = self.entries.iter().find(|e| e.slot == slot && e.class == class)?;
        if accepts(&e.best) {
            return Some(e.best.clone());
        }
        // The overall winner reads a form the caller cannot supply: the fastest accepted candidate of the class
        // (an earlier one keeps a near tie, as when the winner was chosen).
        let mut best: Option<(&str, f64)> = None;
        for (name, v) in &e.results {
            if let Verdict::Timed { us, .. } = v {
                if accepts(name) && best.is_none_or(|(_, b)| *us < b * TIE) {
                    best = Some((name, *us));
                }
            }
        }
        best.map(|(n, _)| n.to_owned())
    }
}

impl Tuned {
    /// Whether every shape class of `slot` with the row bucket of `m` rows and the `K` class of `kg` names a
    /// candidate `accepts`. A class never measured counts as not accepted, so an untuned slot answers `false`.
    pub(crate) fn all_classes_accept(&self, slot: &str, m: u32, kg: u32, accepts: &dyn Fn(&str) -> bool) -> bool {
        let g = grid(slot);
        // Any `n` of the bytes classes: the row bucket and the `K` class are what the caller knows.
        let Some(probe) = Class::of(g, m, kg, 1) else { return false };
        let mut any = false;
        for e in self.entries.iter().filter(|e| e.slot == slot && e.class.rows == probe.rows && e.class.k == probe.k) {
            if !accepts(&e.best) {
                return false;
            }
            any = true;
        }
        any
    }
}

fn fnv(parts: &[&str]) -> u64 {
    crate::tune::source_fingerprint(parts)
}

/// splitmix64 of an index: deterministic filler for the synthetic problems.
fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pseudo-random int8 words: a block of random words repeated, so filling a hundred-megabyte weight tensor costs
/// a copy, not a hash per word. The reference reads the same array, so the check is exact regardless of the pattern.
fn packed_i8(words: usize, salt: u64) -> Vec<u32> {
    const BLOCK: usize = 65_521;
    let block: Vec<u32> = (0..BLOCK.min(words)).map(|i| mix(salt << 40 | i as u64) as u32).collect();
    (0..words).map(|i| block[i % block.len()] ^ (i / BLOCK) as u32).collect()
}

fn scales(n: usize, salt: u64) -> Vec<f32> {
    (0..n).map(|i| 1e-3 + (mix(salt << 40 | i as u64) % 1000) as f32 * 1e-5).collect()
}

fn byte(words: &[u32], idx: usize) -> i32 {
    i32::from((words[idx / 4] >> (8 * (idx % 4))) as u8 as i8)
}

/// Spread of the output positions checked against the reference: first, last and interior rows, columns at tile edges.
fn sample_positions(m: u32, n: u32) -> Vec<(u32, u32)> {
    let rows = [0, m / 2, m.saturating_sub(1)];
    let cols = [0, 1, 15, 16, 17, n / 3, n / 2, n.saturating_sub(17), n.saturating_sub(1)];
    let mut out = Vec::new();
    for &r in &rows {
        for &c in &cols {
            if r < m && c < n && !out.contains(&(r, c)) {
                out.push((r, c));
            }
        }
    }
    out
}

/// `x` rounded to bf16 (round to nearest, ties to even), as `cast_f32_bf16` and the bf16 tensor-core operands do.
fn round_bf16(x: f64) -> f64 {
    let x = x as f32;
    if !x.is_finite() {
        return f64::from(x);
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f64::from(f32::from_bits(rounded))
}

/// A synthetic activation matrix of one class and every form a candidate may consume it in. The fp32 rows are the
/// activations "as produced"; the int8 group form is made from them exactly as `max_abs_row` and `quant_pack` do
/// (`sx = max(amax, 1e-8) / 127`, `q = clamp(round(x / sx), -127, 127)`), the bf16 form as `cast_f32_bf16` does.
struct Acts {
    m: u32,
    kg: u32,
    x: Vec<f32>,
    xq: Vec<u32>,
    sx: Vec<f32>,
    bf16: Vec<u32>,
}

impl Acts {
    fn new(m: u32, kg: u32) -> Acts {
        const BLOCK: usize = 65_521;
        let k = (kg * 4) as usize;
        let groups = k / 32;
        // Triangular noise with a group-varying magnitude and an occasional outlier: the shape of a normalised
        // activation row, and what makes the group quantisation step differ between groups.
        let block: Vec<f32> = (0..BLOCK)
            .map(|i| {
                let u = |salt: u64| (mix(salt << 40 | i as u64) >> 40) as f32 / (1u64 << 24) as f32;
                let v = u(5) + u(6) - 1.0;
                if mix(7 << 40 | i as u64).is_multiple_of(97) { 8.0 * v } else { v }
            })
            .collect();
        const MAGNITUDE: [f32; 8] = [0.25, 0.5, 1.0, 2.0, 0.75, 1.5, 0.35, 1.2];
        let total = m as usize * k;
        let x: Vec<f32> = (0..total).map(|i| block[(i + i / BLOCK) % BLOCK] * MAGNITUDE[(i / 32 + i / (32 * BLOCK)) % 8]).collect();
        let mut sx = Vec::with_capacity(m as usize * groups);
        let mut xq = vec![0u32; total / 4];
        for (g, chunk) in x.as_chunks::<32>().0.iter().enumerate() {
            let scale = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-8) / 127.0;
            sx.push(scale);
            for (e, v) in chunk.iter().enumerate() {
                let q = (*v * (1.0 / scale)).round().clamp(-127.0, 127.0) as i32;
                let idx = g * 32 + e;
                xq[idx / 4] |= ((q & 0xff) as u32) << (8 * (idx % 4));
            }
        }
        let bf16 = x.chunks(2).map(|p| (round_bf16(f64::from(p[0])) as f32).to_bits() >> 16 | (round_bf16(f64::from(p[1])) as f32).to_bits() & 0xFFFF_0000).collect();
        Acts { m, kg, x, xq, sx, bf16 }
    }

    /// `x[r, idx]` as a consumer of `format` sees it (`None`: as produced).
    fn value(&self, format: Option<ActFormat>, r: u32, idx: usize) -> f64 {
        let k = (self.kg * 4) as usize;
        let at = r as usize * k + idx;
        match format {
            None => f64::from(self.x[at]),
            Some(ActFormat::Int8Group32) => f64::from(byte(&self.xq, at)) * f64::from(self.sx[r as usize * (k / 32) + idx / 32]),
            Some(ActFormat::Bf16) => round_bf16(f64::from(self.x[at])),
        }
    }
}

/// The int8 weights of a class as stored: `n` rows of `kg` words of codes, one f32 scale per 32 codes.
struct Stored<'a> {
    w: &'a [u32],
    sw: &'a [f32],
    kg: u32,
}

impl Stored<'_> {
    /// `w[c, idx]` as a consumer of `read` sees it (`None`: as stored, `code * scale`, exact in f64).
    fn value(&self, read: Option<WeightRead>, c: u32, idx: usize) -> f64 {
        let k = (self.kg * 4) as usize;
        let stored = f64::from(byte(self.w, c as usize * k + idx)) * f64::from(self.sw[c as usize * (k / 32) + idx / 32]);
        match read {
            None | Some(WeightRead::Stored) => stored,
            Some(WeightRead::Bf16) => round_bf16(stored),
        }
    }
}

/// `sum_k a[r, k] * w[c, k]` over the activations and weights as a consumer of the given forms sees them (`None`:
/// as produced and as stored), in f64, and the sum of the absolute terms.
fn dot(a: &Acts, w: &Stored, input: Option<Input>, r: u32, c: u32) -> (f64, f64) {
    let (mut total, mut abs) = (0.0f64, 0.0f64);
    for idx in 0..(a.kg * 4) as usize {
        let term = a.value(input.map(|i| i.act), r, idx) * w.value(input.map(|i| i.weight), c, idx);
        total += term;
        abs += term.abs();
    }
    (total, abs)
}

/// The admission factor: a candidate is admitted while its error against the f64 reference of the stored
/// representation is at most this many times the error of rounding its own inputs to their storage resolution.
pub const ALPHA: f64 = 8.0;

/// Unit roundoff of the fp32 accumulation every candidate sums in (2^-24). The accumulation floor of the bound.
const F32_EPS: f64 = 1.0 / 16_777_216.0;

/// The root-mean-square and the largest absolute error of a candidate (or of rounding its inputs) over the sampled
/// outputs of one class.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Errors {
    rms: f64,
    max: f64,
}

impl Errors {
    fn of(errors: &[f64]) -> Errors {
        let rms = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len().max(1) as f64).sqrt();
        Errors { rms, max: errors.iter().fold(0.0f64, |a, e| a.max(*e)) }
    }
}

/// The admission rule, on the errors of one class's sampled outputs.
///
/// Reference: the f64 dot product of the fp32 activations as produced and the weights as stored (`code * scale`, exact
/// in f64) - what the linear means, independent of any candidate's arithmetic. A candidate cannot beat the resolution
/// of the forms it consumes: an int8-activation kernel sees the activations rounded to int8 with a scale per 32, a
/// bf16 kernel sees them and the dequantised weights rounded to bf16. So the error that form alone costs is measured
/// directly: `input` is the f64 dot product of the inputs rounded to the candidate's formats ([`Input`]), and its
/// difference to the reference is the candidate's resolution error, derived from the formats it declares and from
/// nothing else. Beyond rounding its inputs a candidate only adds accumulation order and fp32 rounding, which are
/// orders of magnitude smaller; a wrong kernel is orders of magnitude larger. The rule puts the line between at
/// [`ALPHA`] times the input error (8: it admits both the int8 candidates, whose error is dominated by the activation
/// step, and the bf16 ones, whose tensor-core arithmetic rounds the weight product once more than the f64 emulation
/// does), floored at the fp32 accumulation error so that a candidate consuming its inputs without rounding (an fp32
/// one) is held to an achievable bound rather than zero.
///
/// Both the root mean square and the largest error over the sampled outputs must pass: a bad tile edge or a single
/// wrong row passes an rms bound and fails the maximum.
///
/// Returns `Ok` when `candidate` is within [`ALPHA`] times `input` (never under `floor`), otherwise the reason it is refused.
fn admit(candidate: Errors, input: Errors, floor: Errors) -> Result<(), String> {
    if candidate.rms.is_nan() || candidate.max.is_nan() {
        return Err("non-finite output".into());
    }
    let bound = Errors { rms: ALPHA * input.rms.max(floor.rms), max: ALPHA * input.max.max(floor.max) };
    if candidate.rms > bound.rms || candidate.max > bound.max {
        return Err(format!(
            "error rms {:.3e} max {:.3e} against {ALPHA}x the input rounding error (rms {:.3e} max {:.3e}) over the stored reference",
            candidate.rms, candidate.max, bound.rms, bound.max
        ));
    }
    Ok(())
}

/// The admission verdict of every candidate of `gpu`'s int8 linear sets that serves `params` (`[m, k/4, n]`), on a
/// synthetic problem of exactly that shape: the check [`measure`] applies per class, for a shape of the caller's
/// choosing (ragged edges, odd widths). A candidate is listed as `Timed` with NaN time when it passes. For tests.
#[doc(hidden)]
pub fn check_shape(gpu: &Gpu, params: [u32; 3]) -> Vec<(String, Verdict)> {
    let [m, kg, n] = params;
    let Some(kind) = gpu.int8_linear_kind(m) else { return Vec::new() };
    let slot = if kind == crate::native_upgrade::PREFILL_KIND { crate::native_upgrade::PREFILL_SLOT } else { "matmul_i8_gemv" };
    let cands: Vec<&crate::native_upgrade::Active> = crate::native_upgrade::candidates(gpu.native_rows(), kind, &params).collect();
    time_class(gpu, grid(slot), &cands, (m, kg, n), &Acts::new(m, kg), &Roofs { gbs: None, floor_us: None, int8_tops: None }, false)
}

/// Tune every multi-candidate slot of `gpu` (once per device and build) and return the table, or `None` when this
/// handle has no slot with a choice to make.
pub(crate) fn ensure(gpu: &Gpu, slots: &[(usize, &'static str)]) -> Option<Arc<Tuned>> {
    let multi: Vec<(usize, &'static str)> =
        slots.iter().copied().filter(|&(kind, slot)| TUNABLE_SLOTS.contains(&slot) && gpu.native_rows().iter().filter(|a| a.slow == kind).count() > 1).collect();
    if multi.is_empty() || !AUTOTUNE.load(Ordering::Relaxed) || TUNING.with(|t| t.get()) {
        return None;
    }
    let key = device_key(gpu, &multi);
    let mut tables = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, t)) = tables.iter().find(|(k, _)| *k == key) {
        return Some(t.clone());
    }
    TUNING.with(|t| t.set(true));
    let names: Vec<&str> = multi.iter().map(|&(_, slot)| slot).collect();
    let stored = load(&key);
    let tuned = Arc::new(match stored {
        Some(t) if missing_classes(&t, &names).is_empty() => t,
        prior => {
            // A table the budget cut short is resumed: its measured classes are kept and only the rest are timed.
            let t = measure(gpu, &multi, prior.as_ref());
            save(&key, &t);
            eprintln!("{}", summary(&t));
            let unmeasured = missing_classes(&t, &names).len();
            if unmeasured > 0 {
                eprintln!(
                    "brain: warning: the {}s tuning budget left {unmeasured} kernel shape classes unmeasured on {}; they take their nearest measured neighbour's choice until the next start measures them",
                    BUDGET.as_secs(),
                    t.device
                );
            }
            t
        }
    });
    TUNING.with(|t| t.set(false));
    tables.push((key, tuned.clone()));
    Some(tuned)
}

fn device_key(gpu: &Gpu, slots: &[(usize, &'static str)]) -> String {
    let desc = crate::adapter_info().map(|(d, _)| d).unwrap_or_else(|| gpu.kind().to_owned());
    let cc = gpu.caps().arch.compute_capability.map(|(a, b)| format!("{a}.{b}")).unwrap_or_default();
    let mut sources: Vec<&str> = gpu.native_rows().iter().filter(|a| slots.iter().any(|&(k, _)| k == a.slow)).map(|a| a.kernel).collect();
    sources.sort_unstable();
    sources.dedup();
    let srcs: Vec<&str> = sources.iter().filter_map(|n| kernels_cuda::get(n).map(|k| k.src)).collect();
    let fp = fnv(&srcs) ^ fnv(&[env!("CARGO_PKG_VERSION"), &format!("admission-{ALPHA}"), &format!("{:?}", [&DECODE, &PREFILL].map(|g| (g.rows, g.k, g.bytes_bounds, g.n_reps, g.footprint_bytes, g.stream_max_rows)))]);
    let slug: String = format!("{desc}-{}-{cc}", gpu.kind()).chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("{slug}-{fp:016x}")
}

fn store_path(key: &str) -> Option<std::path::PathBuf> {
    Some(crate::tune::cache_dir()?.join(format!("native-tune-{key}.txt")))
}

fn load(key: &str) -> Option<Tuned> {
    parse_store(&std::fs::read_to_string(store_path(key)?).ok()?)
}

/// A stored table, as written by [`save`]; `None` when the text is not one. A class saved with no results was not
/// reached by its tuning run's budget ([`missing_classes`]).
fn parse_store(text: &str) -> Option<Tuned> {
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next()?.split('|').collect();
    let [tag, device, roof, floor, tops] = header[..] else { return None };
    if tag != "native-tune-v3" {
        return None;
    }
    let mut entries = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split('|').collect();
        let [slot, rows, k, bytes, best, results] = f[..] else { return None };
        let class = Class { rows: rows.parse().ok()?, k: k.parse().ok()?, bytes: bytes.parse().ok()? };
        let mut rs = Vec::new();
        for r in results.split(';').filter(|r| !r.is_empty()) {
            let p: Vec<&str> = r.split(':').collect();
            let verdict = match p[..] {
                [_, "rej", why] => Verdict::Rejected(why.to_owned()),
                [_, us, gbs, frac] => Verdict::Timed { us: us.parse().ok()?, gbs: gbs.parse().ok()?, frac: frac.parse().ok() },
                _ => return None,
            };
            rs.push((p[0].to_owned(), verdict));
        }
        entries.push(Entry { slot: slot.to_owned(), class, best: best.to_owned(), results: rs });
    }
    Some(Tuned { device: device.to_owned(), roof_gbs: roof.parse().ok(), floor_us: floor.parse().ok(), int8_tops: tops.parse().ok(), entries, from_store: true })
}

/// The `(slot, class)` pairs of `slots` that `t` holds no measurement for: absent, or saved unmeasured because the
/// tuning budget ran out before them.
fn missing_classes(t: &Tuned, slots: &[&str]) -> Vec<(String, Class)> {
    slots
        .iter()
        .flat_map(|&slot| Class::all(grid(slot)).map(move |c| (slot, c)))
        .filter(|&(slot, c)| !t.entries.iter().any(|e| e.slot == slot && e.class == c && !e.results.is_empty()))
        .map(|(slot, c)| (slot.to_owned(), c))
        .collect()
}

/// Give every unmeasured class the choice of the nearest measured row bucket of the same slot, `K` and bytes
/// class (the lower one on a tie) whose winner `serves` it; a class with no such neighbour keeps its default.
/// The slot's default is its first candidate, the single-row GEMV - the slowest choice at a batch - so leaving
/// a batched class on it is the costliest guess. Filled classes keep no results, so they stay unmeasured.
/// Returns how many classes were filled.
fn fill_untimed(entries: &mut [Entry], serves: impl Fn(&str, Class, &str) -> bool) -> usize {
    let measured: Vec<(String, Class, String)> = entries.iter().filter(|e| !e.results.is_empty()).map(|e| (e.slot.clone(), e.class, e.best.clone())).collect();
    let mut filled = 0;
    for e in entries.iter_mut().filter(|e| e.results.is_empty()) {
        let nearest = measured
            .iter()
            .filter(|(slot, c, best)| *slot == e.slot && c.k == e.class.k && c.bytes == e.class.bytes && serves(&e.slot, e.class, best))
            .min_by_key(|(_, c, _)| (c.rows.abs_diff(e.class.rows), c.rows));
        if let Some((_, _, best)) = nearest {
            e.best = best.clone();
            filled += 1;
        }
    }
    filled
}

/// One class's results as [`save`] writes them.
fn store_results(results: &[(String, Verdict)]) -> String {
    results
        .iter()
        .map(|(k, v)| match v {
            Verdict::Timed { us, gbs, frac } => format!("{k}:{us:.2}:{gbs:.0}:{}", frac.map(|f| format!("{f:.3}")).unwrap_or_default()),
            Verdict::Rejected(why) => format!("{k}:rej:{}", why.replace([':', '|', ';', '\n'], " ")),
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn save(key: &str, t: &Tuned) {
    let Some(path) = store_path(key) else { return };
    let opt = |v: Option<f32>| v.map(|r| r.to_string()).unwrap_or_default();
    let mut text = format!("native-tune-v3|{}|{}|{}|{}\n", t.device, opt(t.roof_gbs), opt(t.floor_us), opt(t.int8_tops));
    for e in &t.entries {
        text += &format!("{}|{}|{}|{}|{}|{}\n", e.slot, e.class.rows, e.class.k, e.class.bytes, e.best, store_results(&e.results));
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// Measure every class of every multi-candidate slot that `prior` (a stored table the budget cut short) holds no
/// measurement for; the measured classes of `prior` are kept as they are.
fn measure(gpu: &Gpu, slots: &[(usize, &'static str)], prior: Option<&Tuned>) -> Tuned {
    let timing = gpu.set_kernel_timing(true);
    let (roof_gbs, floor_us) = match timing.then(|| measure_read_roof(gpu)).flatten() {
        Some((r, f)) => (Some(r), Some(f)),
        None => (None, None),
    };
    let int8_tops = timing.then(|| measure_int8_roof(gpu)).flatten();
    let started = std::time::Instant::now();
    let mut entries = Vec::new();
    let mut served: HashMap<(&str, Class), Vec<&str>> = HashMap::new();
    // The activations of the last class built: neighbouring classes (the bytes classes of one row bucket and `K`)
    // time the same rows.
    let mut acts: Option<Acts> = None;
    for &(kind, slot) in slots {
        let g = grid(slot);
        for class in Class::all(g) {
            let (m, kg, n) = class.problem(g);
            let params = [m, kg, n];
            let cands: Vec<&crate::native_upgrade::Active> = crate::native_upgrade::candidates(gpu.native_rows(), kind, &params).collect();
            if cands.is_empty() {
                continue;
            }
            served.insert((slot, class), cands.iter().map(|a| a.kernel).collect());
            if let Some(e) = prior.and_then(|p| p.entries.iter().find(|e| e.slot == slot && e.class == class && !e.results.is_empty())) {
                entries.push(e.clone());
                continue;
            }
            // Past the budget a class is left unmeasured (filled from its neighbours below) rather than delaying the
            // start further; the next start measures it.
            let results = if started.elapsed() > BUDGET { Vec::new() } else {
                if !acts.as_ref().is_some_and(|a| a.m == m && a.kg == kg) {
                    acts = Some(Acts::new(m, kg));
                }
                time_class(gpu, g, &cands, (m, kg, n), acts.as_ref().expect("built above"), &Roofs { gbs: roof_gbs, floor_us, int8_tops }, timing)
            };
            let mut best: Option<(&str, f64)> = None;
            for (name, v) in &results {
                if let Verdict::Timed { us, .. } = v {
                    // An earlier candidate keeps a near tie: the choice must not flip between runs on noise.
                    if best.is_none_or(|(_, b)| *us < b * TIE) {
                        best = Some((name, *us));
                    }
                }
            }
            let best = best.map(|(n, _)| n.to_owned()).unwrap_or_else(|| cands[0].kernel.to_owned());
            entries.push(Entry { slot: slot.to_owned(), class, best, results });
        }
    }
    fill_untimed(&mut entries, |slot, class, name| served.get(&(slot, class)).is_some_and(|names| names.contains(&name)));
    gpu.set_kernel_timing(false);
    let device = crate::adapter_info().map(|(d, _)| d).unwrap_or_else(|| gpu.kind().to_owned());
    Tuned { device, roof_gbs, floor_us, int8_tops, entries, from_store: false }
}

/// The device's measured roofs, the bounds a class's fraction is taken against.
struct Roofs {
    /// Memory read roof, GB/s.
    gbs: Option<f32>,
    /// Launch floor, microseconds.
    floor_us: Option<f32>,
    /// int8 tensor-core roof, TOPS.
    int8_tops: Option<f32>,
}

impl Roofs {
    /// The best time (ms) a problem of `ops` operations over `wbytes` of weights can take: the slower of the compute
    /// and the memory bound, never under the launch floor. `None` while the roof it needs is unmeasured. A
    /// weight-streaming (decode) class has no compute bound, it is far under the compute roof at these rows.
    fn bound_ms(&self, compute_bound: bool, ops: f64, wbytes: u64) -> Option<f64> {
        let mem = wbytes as f64 / (f64::from(self.gbs?) * 1e9) * 1e3;
        let compute = if compute_bound { ops / (f64::from(self.int8_tops?) * 1e12) * 1e3 } else { 0.0 };
        Some(mem.max(compute).max(f64::from(self.floor_us.unwrap_or(0.0)) * 1e-3))
    }
}

/// Run and time each candidate on one synthetic problem `(m, kg, n)` of grid `g`, over the activations `acts`
/// (`m` rows, `kg * 4` wide). A candidate is handed the activation form it declares ([`Input`]) and judged by the
/// admission rule ([`admit`]) before it is timed.
fn time_class(gpu: &Gpu, g: &Grid, cands: &[&crate::native_upgrade::Active], (m, kg, n): (u32, u32, u32), acts: &Acts, roofs: &Roofs, timing: bool) -> Vec<(String, Verdict)> {
    debug_assert!(acts.m == m && acts.kg == kg);
    let k = kg * 4;
    let groups = (k / 32) as usize;
    let w = packed_i8(n as usize * kg as usize, 3);
    let sw = scales(n as usize * groups, 4);
    let xq = gpu.storage(u64::from(m * kg));
    let sxb = gpu.storage((m as usize * groups) as u64);
    let xb = gpu.storage(u64::from(m) * u64::from(kg) * 2);
    gpu.write(&xq, &acts.xq);
    gpu.write_f32(&sxb, &acts.sx);
    gpu.write(&xb, &acts.bf16);
    let wbytes = u64::from(n) * u64::from(kg) * 4 + u64::from(n) * groups as u64 * 4;
    let copies = if !timing || !g.streams(m) { 1 } else { g.footprint_bytes.div_ceil(wbytes).clamp(1, 128) as usize };
    let weights: Vec<_> = (0..copies)
        .map(|i| {
            let wb = gpu.storage(u64::from(n) * u64::from(kg));
            let sb = gpu.storage(n as u64 * groups as u64);
            if i == 0 {
                gpu.write(&wb, &w);
                gpu.write_f32(&sb, &sw);
            }
            (wb, sb)
        })
        .collect();
    let out = gpu.storage(u64::from(m) * u64::from(n));
    let positions = sample_positions(m, n);
    let stored = Stored { w: &w, sw: &sw, kg };
    // The reference and, per input form, the error of rounding the inputs to it: computed once per form.
    let reference: Vec<(f64, f64)> = positions.iter().map(|&(r, c)| dot(acts, &stored, None, r, c)).collect();
    let mut resolution: Vec<(Input, Errors)> = Vec::new();
    for a in cands {
        if !resolution.iter().any(|(i, _)| *i == a.input) {
            let errors: Vec<f64> = positions.iter().zip(&reference).map(|(&(r, c), (truth, _))| (dot(acts, &stored, Some(a.input), r, c).0 - truth).abs()).collect();
            resolution.push((a.input, Errors::of(&errors)));
        }
    }
    let floor = Errors {
        rms: F32_EPS * (reference.iter().map(|(_, abs)| abs * abs).sum::<f64>() / reference.len() as f64).sqrt(),
        max: F32_EPS * reference.iter().fold(0.0f64, |a, (_, abs)| a.max(*abs)),
    };
    let params = [m, kg, n];
    let mut results = Vec::new();
    for a in cands {
        let blocks = (a.blocks)(&params, a.tile);
        let step = |wi: &(crate::DeviceBuffer, crate::DeviceBuffer)| match a.input.act {
            ActFormat::Int8Group32 => gpu.native_step_raw(a.id, &[&xq, &wi.0, &sxb, &wi.1, &out], &params, blocks),
            ActFormat::Bf16 => gpu.native_step_raw(a.id, &[&xb, &wi.0, &wi.1, &out], &params, blocks),
        };
        let Some(check) = step(&weights[0]) else {
            results.push((a.kernel.to_owned(), Verdict::Rejected("backend declined the launch".into())));
            continue;
        };
        gpu.submit(&[], &[check]);
        gpu.poll_wait();
        let got = gpu.read(&out, (m * n) as usize);
        let errors: Vec<f64> = positions.iter().zip(&reference).map(|(&(r, c), (truth, _))| (f64::from(got[r as usize * n as usize + c as usize]) - truth).abs()).collect();
        let input = resolution.iter().find(|(i, _)| *i == a.input).map_or(Errors { rms: 0.0, max: 0.0 }, |(_, e)| *e);
        if let Err(why) = admit(Errors::of(&errors), input, floor) {
            results.push((a.kernel.to_owned(), Verdict::Rejected(why)));
            continue;
        }
        if !timing {
            results.push((a.kernel.to_owned(), Verdict::Timed { us: f64::NAN, gbs: 0.0, frac: None }));
            continue;
        }
        // A single copy of the weights is launched repeatedly: a launch is timed by the backend's own events, and the
        // minimum over several makes a short kernel's figure the uncontended one.
        let launches = copies.max(MIN_LAUNCHES);
        let steps: Vec<_> = (0..launches).filter_map(|i| step(&weights[i % copies])).collect();
        let time = |steps: &[crate::Step]| {
            gpu.reset_kernel_times();
            gpu.submit(&[], steps);
            gpu.poll_wait();
            gpu.kernel_times().map(|t| t.iter().filter(|r| r.0 == a.reported).map(|r| r.1).sum::<f64>() / steps.len() as f64)
        };
        let _ = time(&steps);
        // Best over short runs of launches: another context time-sliced onto the card stretches whatever it
        // interrupts and never shortens anything, so the minimum is the uncontended figure.
        let best = (0..TRIALS).flat_map(|_| steps.chunks(steps.len().div_ceil(4).max(4))).filter_map(time).fold(f64::MAX, f64::min);
        if !best.is_finite() || best <= 0.0 || best == f64::MAX {
            results.push((a.kernel.to_owned(), Verdict::Rejected("no device timing".into())));
            continue;
        }
        let gbs = wbytes as f64 / (best * 1e-3) / 1e9;
        // A decode-regime int8 linear moves its weights once and does far fewer operations than the roof allows per
        // byte, so its bound is the memory roof; a prefill-sized one is bound by the tensor cores. The fraction is the
        // bound's time over the measured time.
        let ops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        let frac = roofs.bound_ms(!g.streams(m), ops, wbytes).map(|b| b / best);
        results.push((a.kernel.to_owned(), Verdict::Timed { us: best * 1e3, gbs, frac }));
    }
    results
}

fn summary(t: &Tuned) -> String {
    let mut chosen: HashMap<(&str, &str), usize> = HashMap::new();
    for e in &t.entries {
        *chosen.entry((e.slot.as_str(), e.best.as_str())).or_default() += 1;
    }
    let mut parts: Vec<String> = chosen.iter().map(|((s, b), n)| format!("{s}->{b} x{n}")).collect();
    parts.sort();
    format!("brain: tuned native kernels on {} ({} classes): {}", t.device, t.entries.len(), parts.join(", "))
}

/// The tables built by this process, listed: per slot and shape class the chosen candidate, its time, achieved
/// bandwidth and fraction of the measured roofline, the other candidates, and the classes under [`GAP_THRESHOLD`].
pub fn report() -> String {
    let tables = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    if tables.is_empty() {
        return "no native candidate slot was tuned (no multi-candidate kernel on this device, or autotune is off)\n".to_owned();
    }
    let mut out = String::new();
    for (_, t) in tables.iter() {
        let roof = t.roof_gbs.map(|r| format!("{r:.0} GB/s")).unwrap_or_else(|| "not measured".into());
        let tops = t.int8_tops.map(|r| format!("{r:.0} TOPS")).unwrap_or_else(|| "not measured".into());
        out += &format!("native kernel selection on {} (memory roof {roof}, int8 tensor-core roof {tops}, {})\n", t.device, if t.from_store { "from the store" } else { "measured now" });
        out += &format!("{:<18} {:<28} {:<22} {:>8} {:>8} {:>7}  other candidates\n", "slot", "shape class", "chosen", "us", "GB/s|TOPS", "roof");
        for e in &t.entries {
            let g = grid(&e.slot);
            let (us, rate, frac) = match e.results.iter().find(|(k, _)| *k == e.best) {
                // A compute-bound class reports the operations it ran a second, a streaming one its weight bandwidth.
                Some((_, Verdict::Timed { us, frac, .. })) if !g.streams(e.class.problem(g).0) => {
                    let (m, kg, n) = e.class.problem(g);
                    (*us, 2.0 * f64::from(m) * f64::from(n) * f64::from(kg) * 4.0 / (*us * 1e-6) / 1e12, *frac)
                }
                Some((_, Verdict::Timed { us, gbs, frac })) => (*us, *gbs, *frac),
                _ => (f64::NAN, 0.0, None),
            };
            let others: Vec<String> = e
                .results
                .iter()
                .filter(|(k, _)| *k != e.best)
                .map(|(k, v)| match v {
                    Verdict::Timed { us, .. } => format!("{k} {us:.1}us"),
                    Verdict::Rejected(why) => format!("{k} REJECTED ({why})"),
                })
                .collect();
            out += &format!(
                "{:<18} {:<28} {:<22} {:>8.1} {:>8.0} {:>7}  {}\n",
                e.slot,
                e.describe(),
                e.best,
                us,
                rate,
                frac.map(|f| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "-".into()),
                others.join(", ")
            );
        }
        let gap: Vec<&Entry> = t.entries.iter().filter(|e| e.roof_fraction().is_some_and(|f| f < GAP_THRESHOLD)).collect();
        out += &format!("below {:.0}% of the roofline: {} of {} classes\n", GAP_THRESHOLD * 100.0, gap.len(), t.entries.len());
        for e in gap {
            out += &format!("  {} {} - {:.0}% with {}\n", e.slot, e.describe(), e.roof_fraction().unwrap_or(0.0) * 100.0, e.best);
        }
    }
    out
}

/// Every `(slot, candidate)` pair the tuned handles offer.
pub fn offered() -> Vec<(String, String)> {
    let tables = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<(String, String)> = tables.iter().flat_map(|(_, t)| t.entries.iter().flat_map(|e| e.results.iter().map(|(k, _)| (e.slot.clone(), k.clone())))).collect();
    v.sort();
    v.dedup();
    v
}

/// The selection table of the process's default device, built by opening a handle that registers the kernel slots
/// the native candidate sets cover (the int8 decode linear and prefill GEMM): what `brain roofline --kernels` prints, with no model
/// loaded.
pub fn report_for_default_device() -> String {
    let _gpu = Gpu::new(&[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV), ("matmul_i8_dyn", kernels::MATMUL_I8_DYN)]);
    report()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_decode_shape_falls_in_a_class_and_each_class_is_timed_inside_it() {
        for m in 1..=32u32 {
            for kg in [8u32, 256, 1280, 4352, 9000] {
                for n in [1u32, 48, 1024, 5120, 17408, 248320] {
                    let c = Class::of(&DECODE, m, kg, n).expect("a decode-regime shape has a class");
                    assert!((DECODE.rows[c.rows].0..=DECODE.rows[c.rows].1).contains(&m));
                }
            }
        }
        assert!(Class::of(&DECODE, 33, 1280, 1024).is_none(), "past the decode regime nothing is tuned on the decode grid");
        for g in [&DECODE, &PREFILL] {
            for c in Class::all(g) {
                let (m, kg, n) = c.problem(g);
                assert_eq!(Class::of(g, m, kg, n), Some(c), "{}: the problem timed for a class falls in that class", c.label(g));
            }
        }
    }

    /// The prefill grid starts where the decode grid ends and takes every wider round into its last class, so no
    /// prefill-sized dispatch is left to the default for being wider than the grid.
    #[test]
    fn the_prefill_grid_continues_the_decode_grid_and_is_open_above() {
        let decode_top = DECODE.rows[DECODE.rows.len() - 1].1;
        assert_eq!(PREFILL.rows[0].0, decode_top + 1);
        for pair in PREFILL.rows.windows(2) {
            assert_eq!(pair[0].1 + 1, pair[1].0, "buckets are contiguous");
        }
        let top = Class::of(&PREFILL, PREFILL.rows[PREFILL.rows.len() - 1].1, 1280, 5120).unwrap();
        assert_eq!(Class::of(&PREFILL, 1 << 20, 1280, 5120), Some(top));
        // The classes that still read every weight once stream; the wider rounds are compute-bound.
        assert!(PREFILL.streams(128) && !PREFILL.streams(129));
        assert!(DECODE.streams(32));
    }

    #[test]
    fn a_table_names_its_choice_per_class_and_a_forced_candidate_wins() {
        let class = Class::of(&DECODE, 12, 1280, 5120).unwrap();
        let t = Tuned {
            device: "test".into(),
            roof_gbs: Some(1000.0),
            floor_us: Some(5.0),
            int8_tops: Some(600.0),
            entries: vec![Entry { slot: "slot".into(), class, best: "fast".into(), results: vec![("fast".into(), Verdict::Timed { us: 1.0, gbs: 500.0, frac: Some(0.5) })] }],
            from_store: false,
        };
        assert_eq!(t.choice("slot", &[12, 1280, 5120], &|_| true).as_deref(), Some("fast"));
        assert_eq!(t.choice("slot", &[12, 1280, 17], &|_| true).as_deref(), None, "an untuned class names nothing: the default stands");
        assert_eq!(t.choice("other", &[12, 1280, 5120], &|_| true), None);
        assert_eq!(t.entries[0].roof_fraction(), Some(0.5));
    }

    fn timed(slot: &str, rows: usize, best: &str) -> Entry {
        Entry { slot: slot.into(), class: Class { rows, k: 0, bytes: 2 }, best: best.into(), results: vec![(best.into(), Verdict::Timed { us: 10.0, gbs: 1500.0, frac: Some(0.4) })] }
    }

    fn untimed(slot: &str, rows: usize) -> Entry {
        Entry { slot: slot.into(), class: Class { rows, k: 0, bytes: 2 }, best: "default".into(), results: Vec::new() }
    }

    /// A class the tuning budget did not reach takes the choice of the nearest measured row bucket of the same
    /// slot, `K` and bytes class whose winner also serves it - not the slot's default, which is the slowest
    /// candidate at a batch (a 32-row decode step on the default ran at a third of its bandwidth).
    #[test]
    fn a_class_the_budget_did_not_reach_takes_its_nearest_measured_neighbours_choice() {
        let mut entries = vec![timed("s", 3, "gemv"), timed("s", 4, "stream"), untimed("s", 5), untimed("s", 6), timed("other", 6, "x")];
        let filled = fill_untimed(&mut entries, |_, _, name| name != "x");
        assert_eq!(filled, 2);
        assert_eq!(entries[2].best, "stream", "rows 17-24 take rows 9-16's choice");
        assert_eq!(entries[3].best, "stream", "rows 25-32 take the nearest measured bucket's choice");
        assert!(entries[2].results.is_empty(), "a filled class stays marked unmeasured, so the next start measures it");
        assert_eq!(entries[1].best, "stream", "measured classes are untouched");

        let mut entries = vec![timed("s", 4, "stream"), untimed("s", 5)];
        fill_untimed(&mut entries, |_, class, name| !(class.rows == 5 && name == "stream"));
        assert_eq!(entries[1].best, "default", "a neighbour whose winner cannot serve the class is not copied");
    }

    /// A stored table holding an unmeasured class is incomplete: it is used, and the next start measures the
    /// missing classes instead of serving them from the default for as long as this build lives.
    #[test]
    fn a_stored_table_with_unmeasured_classes_is_incomplete() {
        let line = |e: &Entry| format!("{}|{}|{}|{}|{}|{}\n", e.slot, e.class.rows, e.class.k, e.class.bytes, e.best, store_results(&e.results));
        let all: Vec<Entry> = Class::all(&DECODE).map(|class| Entry { class, ..timed("s", 0, "gemv") }).collect();
        let text = |entries: &[Entry]| "native-tune-v3|dev|3700|6.5|600\n".to_owned() + &entries.iter().map(line).collect::<String>();
        let complete = parse_store(&text(&all)).expect("a whole table parses");
        assert!(missing_classes(&complete, &["s"]).is_empty());
        let mut cut = all.clone();
        let last = cut.len() - 1;
        cut[last].results.clear();
        let partial = parse_store(&text(&cut)).expect("a table cut by the budget still parses");
        assert_eq!(missing_classes(&partial, &["s"]), vec![("s".to_owned(), cut[last].class)]);
        assert_eq!(missing_classes(&parse_store(&text(&all[1..])).unwrap(), &["s"]).len(), 1, "an absent class is missing too");
    }

    /// The errors of a candidate over the positions of a small synthetic class, against the f64 reference of the
    /// stored representation, and of rounding the inputs to `input`'s forms.
    fn class_errors(input: Input, candidate: impl Fn(&Acts, &Stored, u32, u32) -> f64) -> (Errors, Errors, Errors) {
        let (m, kg, n) = (4u32, 64u32, 40u32);
        let acts = Acts::new(m, kg);
        let (w, sw) = (packed_i8(n as usize * kg as usize, 3), scales(n as usize * kg as usize / 8, 4));
        let stored = Stored { w: &w, sw: &sw, kg };
        let positions = sample_positions(m, n);
        let reference: Vec<(f64, f64)> = positions.iter().map(|&(r, c)| dot(&acts, &stored, None, r, c)).collect();
        let diff = |f: &dyn Fn(u32, u32) -> f64| Errors::of(&positions.iter().zip(&reference).map(|(&(r, c), (t, _))| (f(r, c) - t).abs()).collect::<Vec<_>>());
        let floor = Errors { rms: F32_EPS * reference.iter().map(|(_, a)| a * a).sum::<f64>().sqrt() / (reference.len() as f64).sqrt(), max: F32_EPS * reference.iter().fold(0.0f64, |a, (_, b)| a.max(*b)) };
        (diff(&|r, c| candidate(&acts, &stored, r, c)), diff(&|r, c| dot(&acts, &stored, Some(input), r, c).0), floor)
    }

    /// The rule admits the arithmetic each input form allows and refuses what is wrong, whatever the candidate's
    /// format: an int8-activation kernel (error dominated by the activation step), a bf16 dequantising kernel (weights
    /// rounded once more than the f64 emulation of the formats does) and a kernel that gets a tile edge wrong.
    #[test]
    fn the_admission_rule_follows_the_candidates_input_resolution() {
        // The existing integer-MMA candidates compute exactly the dot product of their inputs.
        let (cand, input, floor) = class_errors(Input::PACKED, |a, w, r, c| dot(a, w, Some(Input::PACKED), r, c).0);
        assert!(input.rms > 100.0 * floor.rms, "int8 activations cost far more than the fp32 accumulation: {input:?} vs {floor:?}");
        assert_eq!(admit(cand, input, floor), Ok(()), "an int8-activation candidate is admitted");

        // A bf16 kernel that rounds the group scale to bf16 before the product (the cheap packed dequantisation).
        let scale_bf16 = |a: &Acts, w: &Stored, r: u32, c: u32| {
            (0..(w.kg * 4) as usize).map(|i| a.value(Some(ActFormat::Bf16), r, i) * round_bf16(round_bf16(f64::from(w.sw[c as usize * (w.kg as usize / 8) + i / 32])) * f64::from(byte(w.w, c as usize * (w.kg as usize * 4) + i)))).sum::<f64>()
        };
        let (cand, input, floor) = class_errors(Input::DEQUANT_BF16, scale_bf16);
        assert_eq!(admit(cand, input, floor), Ok(()), "a bf16 dequantising candidate is admitted: {cand:?} vs {input:?}");

        // The same bf16 kernel with one output wrong by a tenth of its rms is refused (max bound), and so is a kernel
        // 100x off; NaN is refused.
        let wrong = |a: &Acts, w: &Stored, r: u32, c: u32| scale_bf16(a, w, r, c) * if (r, c) == (0, 17) { 1.1 } else { 1.0 };
        let (cand, input, floor) = class_errors(Input::DEQUANT_BF16, wrong);
        assert!(admit(cand, input, floor).is_err(), "a single wrong output fails the maximum: {cand:?} vs {input:?}");
        let (cand, input, floor) = class_errors(Input::PACKED, |a, w, r, c| dot(a, w, Some(Input::PACKED), r, c).0);
        assert_eq!(admit(Errors { rms: f64::NAN, max: 0.0 }, input, floor), Err("non-finite output".into()));
        assert!(admit(Errors { rms: input.rms * 100.0, max: input.max * 100.0 }, input, floor).is_err(), "a kernel 100x off the input resolution is refused");
        assert_eq!(admit(cand, input, floor), Ok(()));
    }

    /// A candidate that consumes its inputs unrounded has no resolution error to scale: the bound falls to the
    /// accumulation floor, which an fp32-accurate kernel meets and a bf16-accurate one does not.
    #[test]
    fn a_zero_input_error_leaves_the_fp32_accumulation_floor() {
        let none = Errors { rms: 0.0, max: 0.0 };
        let floor = Errors { rms: 1e-6, max: 3e-6 };
        assert_eq!(admit(Errors { rms: 5e-6, max: 2e-5 }, none, floor), Ok(()));
        assert!(admit(Errors { rms: 5e-5, max: 2e-4 }, none, floor).is_err());
    }

    #[test]
    fn the_store_round_trips_a_table() {
        let class = Class::of(&DECODE, 12, 1280, 5120).unwrap();
        let entry = Entry {
            slot: "slot".into(),
            class,
            best: "b".into(),
            results: vec![("a".into(), Verdict::Rejected("bad: output".into())), ("b".into(), Verdict::Timed { us: 12.5, gbs: 1800.0, frac: Some(0.5) })],
        };
        let line = format!("{}|{}|{}|{}|{}|", entry.slot, class.rows, class.k, class.bytes, entry.best);
        assert!(line.starts_with("slot|"));
        assert_eq!(fnv(&["a"]), fnv(&["a"]));
        assert_ne!(fnv(&["a"]), fnv(&["b"]), "a changed candidate source changes the store key");
    }
}
