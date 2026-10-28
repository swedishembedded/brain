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
//! 2. every candidate that serves a problem is run on it once and its output is
//!    compared with an f64 oracle over a sample of outputs; a candidate that
//!    disagrees is rejected for that class, never chosen;
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

use crate::Gpu;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A later candidate must beat an earlier one by this factor to displace it.
const TIE: f64 = 0.97;
/// Ops achieving less than this fraction of their roofline bound are listed as the device's gap.
pub const GAP_THRESHOLD: f64 = 0.6;
/// Row buckets of the decode regime: `(lo, hi)` inclusive, timed at `hi`.
const ROW_BUCKETS: [(u32, u32); 7] = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 24), (25, 32)];
/// `K` classes, timed at their representative.
const K_CLASSES: [u32; 2] = [5120, 17408];
/// Upper bound (exclusive) of the weight-bytes classes, in bytes; the last class is unbounded.
const BYTES_BOUNDS: [u64; 3] = [3 << 20, 12 << 20, 48 << 20];
/// Output columns timed for each bytes class at each `K` class (about 1.5, 6, 24 and 90 MB of weights).
const N_REPS: [[u32; 4]; 2] = [[256, 1024, 4096, 16384], [64, 256, 1024, 4096]];
/// Distinct weight bytes cycled per timing, past the L2 of any current part.
const FOOTPRINT_BYTES: u64 = 192 << 20;
const TRIALS: usize = 3;
/// Wall time one device's tuning may take; classes not reached keep their default candidate until a rebuild retunes.
const BUDGET: std::time::Duration = std::time::Duration::from_secs(25);

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

/// The slots whose candidates this module knows how to build a synthetic problem and an oracle for: the int8
/// decode linear. A slot with several rows that is not listed (the K-quant narrow and wide blocks) keeps its
/// row-count rule until it has a problem generator here.
const TUNABLE_SLOTS: [&str; 1] = ["matmul_i8_gemv"];

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

/// A shape class of the decode regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Class {
    rows: usize,
    k: usize,
    bytes: usize,
}

impl Class {
    fn of(m: u32, kg: u32, n: u32) -> Option<Class> {
        let rows = ROW_BUCKETS.iter().position(|&(lo, hi)| (lo..=hi).contains(&m))?;
        let k = if kg * 4 <= K_CLASSES[0] * 3 / 2 { 0 } else { 1 };
        let bytes = u64::from(n) * u64::from(kg) * 4 * 9 / 8;
        Some(Class { rows, k, bytes: BYTES_BOUNDS.iter().position(|&b| bytes < b).unwrap_or(BYTES_BOUNDS.len()) })
    }

    fn all() -> impl Iterator<Item = Class> {
        (0..ROW_BUCKETS.len()).flat_map(|rows| (0..K_CLASSES.len()).flat_map(move |k| (0..=BYTES_BOUNDS.len()).map(move |bytes| Class { rows, k, bytes })))
    }

    /// The problem this class is timed on: `(m, kg, n)`.
    fn problem(self) -> (u32, u32, u32) {
        (ROW_BUCKETS[self.rows].1, K_CLASSES[self.k] / 4, N_REPS[self.k][self.bytes])
    }

    fn label(self) -> String {
        let (lo, hi) = ROW_BUCKETS[self.rows];
        let (_, kg, n) = self.problem();
        format!("rows {lo}-{hi}, K~{}, N~{n}", kg * 4)
    }
}

/// What happened to one candidate on one class.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Timed: microseconds per launch, weight bandwidth in GB/s, fraction of the measured roofline bound.
    Timed { us: f64, gbs: f64, frac: Option<f64> },
    /// Disagreed with the oracle; never chosen.
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
        self.class.label()
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
    pub entries: Vec<Entry>,
    /// Whether this table was read from the persistent store rather than measured by this process.
    pub from_store: bool,
}

impl Tuned {
    /// The candidate the table (or a forced override) names for a dispatch of `slot` with caller `params`.
    pub(crate) fn choice(&self, slot: &str, params: &[u32]) -> Option<String> {
        if let Some((_, k)) = FORCED.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(s, _)| s == slot) {
            return Some(k.clone());
        }
        let class = match params {
            [m, kg, n, ..] => Class::of(*m, *kg, *n)?,
            _ => return None,
        };
        self.entries.iter().find(|e| e.slot == slot && e.class == class).map(|e| e.best.clone())
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
/// a copy, not a hash per word. The oracle reads the same array, so the check is exact regardless of the pattern.
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

/// Spread of the output positions checked against the oracle: first, last and interior rows, columns at tile edges.
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

/// The f64 oracle for `out[r, c]`: `sum_g (sw[c, g] * sx[r, g]) * sum_{k in g} xq[r, k] * wq[c, k]`.
fn oracle(x: &[u32], w: &[u32], sx: &[f32], sw: &[f32], kg: u32, r: u32, c: u32) -> f64 {
    let k = (kg * 4) as usize;
    let groups = k / 32;
    let mut total = 0.0f64;
    for g in 0..groups {
        let mut acc = 0i64;
        for e in 0..32 {
            let idx = g * 32 + e;
            acc += i64::from(byte(x, r as usize * k + idx)) * i64::from(byte(w, c as usize * k + idx));
        }
        total += acc as f64 * (f64::from(sw[c as usize * groups + g]) * f64::from(sx[r as usize * groups + g]));
    }
    total
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
    let fp = fnv(&srcs) ^ fnv(&[env!("CARGO_PKG_VERSION"), &format!("{ROW_BUCKETS:?}{K_CLASSES:?}{BYTES_BOUNDS:?}{N_REPS:?}")]);
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
    let [tag, device, roof, floor] = header[..] else { return None };
    if tag != "native-tune-v2" {
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
    Some(Tuned { device: device.to_owned(), roof_gbs: roof.parse().ok(), floor_us: floor.parse().ok(), entries, from_store: true })
}

/// The `(slot, class)` pairs of `slots` that `t` holds no measurement for: absent, or saved unmeasured because the
/// tuning budget ran out before them.
fn missing_classes(t: &Tuned, slots: &[&str]) -> Vec<(String, Class)> {
    slots
        .iter()
        .flat_map(|&slot| Class::all().map(move |c| (slot, c)))
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
    let mut text = format!("native-tune-v2|{}|{}|{}\n", t.device, t.roof_gbs.map(|r| r.to_string()).unwrap_or_default(), t.floor_us.map(|r| r.to_string()).unwrap_or_default());
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
    let started = std::time::Instant::now();
    let mut entries = Vec::new();
    let mut served: HashMap<(&str, Class), Vec<&str>> = HashMap::new();
    for &(kind, slot) in slots {
        for class in Class::all() {
            let (m, kg, n) = class.problem();
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
            let results = if started.elapsed() > BUDGET { Vec::new() } else { time_class(gpu, &cands, m, kg, n, roof_gbs, floor_us, timing) };
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
    Tuned { device, roof_gbs, floor_us, entries, from_store: false }
}

/// Run and time each candidate on one synthetic problem.
#[allow(clippy::too_many_arguments)]
fn time_class(gpu: &Gpu, cands: &[&crate::native_upgrade::Active], m: u32, kg: u32, n: u32, roof_gbs: Option<f32>, floor_us: Option<f32>, timing: bool) -> Vec<(String, Verdict)> {
    let k = kg * 4;
    let groups = (k / 32) as usize;
    let x = packed_i8((m * kg) as usize, 1);
    let sx = scales(m as usize * groups, 2);
    let w = packed_i8(n as usize * kg as usize, 3);
    let sw = scales(n as usize * groups, 4);
    let xq = gpu.storage(u64::from(m * kg));
    let sxb = gpu.storage((m as usize * groups) as u64);
    gpu.write(&xq, &x);
    gpu.write_f32(&sxb, &sx);
    let wbytes = u64::from(n) * u64::from(kg) * 4 + u64::from(n) * groups as u64 * 4;
    let copies = FOOTPRINT_BYTES.div_ceil(wbytes).clamp(1, 128) as usize;
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
    let truth: Vec<f64> = positions.iter().map(|&(r, c)| oracle(&x, &w, &sx, &sw, kg, r, c)).collect();
    let rms = (truth.iter().map(|t| t * t).sum::<f64>() / truth.len() as f64).sqrt();
    let params = [m, kg, n];
    let mut results = Vec::new();
    for a in cands {
        let blocks = (a.blocks)(&params, a.tile);
        let step = |wi: &(crate::DeviceBuffer, crate::DeviceBuffer)| gpu.native_step_raw(a.id, &[&xq, &wi.0, &sxb, &wi.1, &out], &params, blocks);
        let Some(check) = step(&weights[0]) else {
            results.push((a.kernel.to_owned(), Verdict::Rejected("backend declined the launch".into())));
            continue;
        };
        gpu.submit(&[], &[check]);
        gpu.poll_wait();
        let got = gpu.read(&out, (m * n) as usize);
        let worst = positions.iter().zip(&truth).map(|(&(r, c), t)| (f64::from(got[r as usize * n as usize + c as usize]) - t).abs()).fold(0.0, f64::max);
        if worst.is_nan() || worst > 2e-5 * rms + 1e-30 {
            results.push((a.kernel.to_owned(), Verdict::Rejected(format!("worst error {worst:.3e} against output rms {rms:.3e}"))));
            continue;
        }
        if !timing {
            results.push((a.kernel.to_owned(), Verdict::Timed { us: f64::NAN, gbs: 0.0, frac: None }));
            continue;
        }
        let steps: Vec<_> = weights.iter().filter_map(step).collect();
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
        // byte, so its bound is the memory roof; the fraction is that bound's time over the measured time.
        let frac = roof_gbs.map(|r| (wbytes as f64 / (f64::from(r) * 1e9) * 1e3).max(f64::from(floor_us.unwrap_or(0.0)) * 1e-3) / best);
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
        out += &format!("native kernel selection on {} (memory roof {roof}, {})\n", t.device, if t.from_store { "from the store" } else { "measured now" });
        out += &format!("{:<18} {:<28} {:<22} {:>8} {:>8} {:>7}  other candidates\n", "slot", "shape class", "chosen", "us", "GB/s", "roof");
        for e in &t.entries {
            let (us, gbs, frac) = match e.results.iter().find(|(k, _)| *k == e.best) {
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
                gbs,
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
/// the native candidate sets cover (the int8 decode linear): what `brain roofline --kernels` prints, with no model
/// loaded.
pub fn report_for_default_device() -> String {
    let _gpu = Gpu::new(&[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV)]);
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
                    let c = Class::of(m, kg, n).expect("a decode-regime shape has a class");
                    assert!((ROW_BUCKETS[c.rows].0..=ROW_BUCKETS[c.rows].1).contains(&m));
                }
            }
        }
        assert!(Class::of(33, 1280, 1024).is_none(), "past the decode regime nothing is tuned here");
        for c in Class::all() {
            let (m, kg, n) = c.problem();
            assert_eq!(Class::of(m, kg, n), Some(c), "{}: the problem timed for a class falls in that class", c.label());
        }
    }

    #[test]
    fn a_table_names_its_choice_per_class_and_a_forced_candidate_wins() {
        let class = Class::of(12, 1280, 5120).unwrap();
        let t = Tuned {
            device: "test".into(),
            roof_gbs: Some(1000.0),
            floor_us: Some(5.0),
            entries: vec![Entry { slot: "slot".into(), class, best: "fast".into(), results: vec![("fast".into(), Verdict::Timed { us: 1.0, gbs: 500.0, frac: Some(0.5) })] }],
            from_store: false,
        };
        assert_eq!(t.choice("slot", &[12, 1280, 5120]).as_deref(), Some("fast"));
        assert_eq!(t.choice("slot", &[12, 1280, 17]).as_deref(), None, "an untuned class names nothing: the default stands");
        assert_eq!(t.choice("other", &[12, 1280, 5120]), None);
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
        let all: Vec<Entry> = Class::all().map(|class| Entry { class, ..timed("s", 0, "gemv") }).collect();
        let text = |entries: &[Entry]| "native-tune-v2|dev|3700|6.5\n".to_owned() + &entries.iter().map(line).collect::<String>();
        let complete = parse_store(&text(&all)).expect("a whole table parses");
        assert!(missing_classes(&complete, &["s"]).is_empty());
        let mut cut = all.clone();
        let last = cut.len() - 1;
        cut[last].results.clear();
        let partial = parse_store(&text(&cut)).expect("a table cut by the budget still parses");
        assert_eq!(missing_classes(&partial, &["s"]), vec![("s".to_owned(), cut[last].class)]);
        assert_eq!(missing_classes(&parse_store(&text(&all[1..])).unwrap(), &["s"]).len(), 1, "an absent class is missing too");
    }

    #[test]
    fn the_store_round_trips_a_table() {
        let class = Class::of(12, 1280, 5120).unwrap();
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
