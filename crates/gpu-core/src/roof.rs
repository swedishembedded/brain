// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device's own roofline - measured, never assumed.
//!
//! Every "% of peak" this engine printed used to divide by a literal
//! (`PEAK_TFLOPS = 11.76`, `PEAK_GBPS = 346.0`) copied into each bench and test.
//! Those are one card's numbers. On anything else the utilisation column was a
//! fiction, and since the whole optimisation method here is "rank against the
//! roof, fix the top row, re-profile", a wrong roof silently invalidates the
//! method rather than producing an obviously wrong answer.
//!
//! No compute API reports either roof, so both are **measured**:
//!
//! * **compute** - [`kernels::ROOF_FMA`], a dependency-free FMA chain held in
//!   registers with no memory traffic in the loop. It measures the *silicon*,
//!   deliberately not "the best GEMM we have written": grading brain's kernels
//!   against brain's best kernel would hide exactly the gap worth closing.
//! * **bandwidth** - `axpy` (`out[i] += s * in[i]`) over buffers far larger than
//!   any cache: the classic STREAM-triad shape, 12 bytes moved per element
//!   (two reads and a write), already in the kernel tree.
//!
//! Both probes self-calibrate their trip count until the timed region is long
//! enough to dominate launch and drain, and both are `poll_wait`-bracketed:
//! a bare-submit loop once reported a bandwidth above the card's physical
//! peak, by timing the host.
//!
//! Results are cached per (backend, physical device) - see [`DeviceKey`] -
//! with the same key discipline [`crate::tune`] uses beyond that (adapter
//! slug + a fingerprint of the probe sources), so a probe-kernel edit
//! invalidates old numbers by construction instead of silently reusing them.
//! The physical-device half of that key comes from
//! [`backend_api::Backend::identity`], which today only `backend-wgpu`
//! exposes; on every other backend two devices still collapse into one
//! shared, first-device-wins entry, exactly as the whole module did before
//! that identity existed.

use crate::Gpu;
use std::sync::Mutex;
use std::time::Instant;

/// A device's measured roofline. Both halves are required - a memory-bound
/// kernel reported as a FLOP rate is meaningless, and vice versa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roofs {
    /// Peak fp32 arithmetic rate, GFLOP/s.
    pub gflops: f32,
    /// Peak DRAM bandwidth, GB/s - a working set far larger than any cache.
    pub gbs: f32,
    /// Peak bandwidth for a CACHE-RESIDENT working set, GB/s.
    ///
    /// The roofline is HIERARCHICAL: a kernel whose data fits in L2 is bounded
    /// by L2 bandwidth, not by DRAM, and can legitimately exceed the DRAM roof
    /// several times over. Grading such a kernel against `gbs` produces a
    /// utilisation above the roof that looks like a broken measurement and is
    /// not one - which is exactly what `paged_decode_scores_batched` did.
    ///
    /// Measured with the same triad over a working set small enough to stay
    /// resident. On a device whose cache is smaller than the probe this simply
    /// converges to `gbs`, which is a conservative and self-consistent failure.
    pub cache_gbs: f32,
    /// Peak packed-int8 (`dot4I8Packed`) rate, GOP/s. `None` where the device
    /// has no int8 dot path.
    ///
    /// Without it an int8 kernel is graded against the *fp32* roof, which
    /// flatters it by whatever factor the device's DP4A path exceeds fp32:
    /// `matmul_i8_dyn` read about a third of the roof that way, and a kernel
    /// that looks like a third of the machine may be a tenth of it.
    pub int8_gops: Option<f32>,
    /// Peak native-`f16` (`enable f16;`, real f16 registers - B11) FMA rate,
    /// GFLOP/s. `None` where the device has not been *measured* to run f16
    /// arithmetic fast (`caps().numeric.f16`, which per its own doc "stays
    /// false until the autotuner measures it" - availability of the WGSL
    /// extension alone is never enough, since e.g. Pascal exposes it at 1/64
    /// rate). Never a guess: this is the same real, `poll_wait`-bracketed FMA
    /// chain as `gflops`, just with every accumulator declared `f16`, so the
    /// two rates are directly comparable and a device that claims fast f16
    /// must measure `f16_gflops >= gflops`.
    pub f16_gflops: Option<f32>,
}

impl Roofs {
    /// FLOP per byte at the ridge point - below it no kernel can be
    /// compute-bound however it is tiled, above it bandwidth cannot be the
    /// limit.
    pub fn ridge(&self) -> f32 {
        if self.gbs > 0.0 { self.gflops / self.gbs } else { 0.0 }
    }

    /// The memory roof that applies to a kernel achieving `gbs` of *logical*
    /// traffic - DRAM if it is under that roof, the cache roof if it is above
    /// it (the data is being served from cache, which is a real and legitimate
    /// place for it to come from), and `None` if it is above even that, which
    /// means the accounting or the timing is wrong.
    pub fn memory_roof(&self, achieved_gbs: f64) -> Option<(f32, bool)> {
        if achieved_gbs as f32 <= self.gbs * 1.05 {
            Some((self.gbs, false))
        } else if achieved_gbs as f32 <= self.cache_gbs * 1.05 {
            Some((self.cache_gbs, true))
        } else {
            None
        }
    }

    /// Which roof a kernel doing `flops` and moving `bytes` is graded against.
    pub fn classify(&self, flops: u64, bytes: u64) -> Bound {
        if bytes == 0 {
            return Bound::Compute;
        }
        let intensity = flops as f32 / bytes as f32;
        if intensity >= self.ridge() { Bound::Compute } else { Bound::Memory }
    }

    /// The compute roof for work that is `int_ops` integer ops and `flops`
    /// float ops. Integer-dominant work is graded against the measured DP4A
    /// rate; where that was not measured the fp32 roof stands in and the caller
    /// is told, because silently using it OVERSTATES an int8 kernel by whatever
    /// factor the device's int8 path exceeds fp32.
    pub fn compute_roof(&self, flops: u64, int_ops: u64) -> (f32, bool) {
        if int_ops > flops {
            match self.int8_gops {
                Some(g) if g > 0.0 => (g, true),
                _ => (self.gflops, false),
            }
        } else {
            (self.gflops, true)
        }
    }

    /// The shortest time this much work can take on this device: the roofline
    /// LOWER BOUND, in seconds.
    ///
    /// Unlike [`Self::utilisation_of`], which grades ONE kernel and can treat
    /// it as int-dominant or float-dominant, this is for an aggregate - a
    /// pipeline stage, a whole generation - where both kinds of work are
    /// present in quantity and neither may be dropped. `max(flops, int_ops)`
    /// against a single roof is wrong there in both directions: an int8 DiT
    /// still runs its attention and its norms in fp32, and those fp32 ops
    /// graded against the (much faster) DP4A roof simply vanish.
    ///
    /// The two op classes share the same SMs, so their times ADD; memory
    /// overlaps with compute, so it is a `max` against the sum. Both halves
    /// assume perfect utilisation, which is what makes this a bound and not a
    /// prediction - the number to track is the ratio a real run achieves
    /// against it.
    ///
    /// `None` when there is no work at all, or when a needed roof is unknown.
    pub fn seconds_at_roof(&self, flops: u64, int_ops: u64, bytes: u64) -> Option<f64> {
        if flops == 0 && int_ops == 0 && bytes == 0 {
            return None;
        }
        if self.gflops <= 0.0 {
            return None;
        }
        // An int8 rate the device never reported falls back to the fp32 roof,
        // which UNDER-states int8 throughput rather than over-stating it.
        let int_roof = self.int8_gops.filter(|g| *g > 0.0).unwrap_or(self.gflops);
        let compute = flops as f64 / (self.gflops as f64 * 1e9) + int_ops as f64 / (int_roof as f64 * 1e9);
        let memory = if self.gbs > 0.0 { bytes as f64 / (self.gbs as f64 * 1e9) } else { 0.0 };
        Some(compute.max(memory))
    }

    /// Which roof an aggregate is against, decided by which side takes longer
    /// rather than by an intensity against one ridge point - a mix of fp32 and
    /// int8 work has two ridge points, and comparing it to either alone
    /// misclassifies it.
    pub fn bound_of(&self, flops: u64, int_ops: u64, bytes: u64) -> Bound {
        let int_roof = self.int8_gops.filter(|g| *g > 0.0).unwrap_or(self.gflops);
        if self.gflops <= 0.0 || self.gbs <= 0.0 {
            return Bound::Compute;
        }
        let compute = flops as f64 / self.gflops as f64 + int_ops as f64 / int_roof as f64;
        let memory = bytes as f64 / self.gbs as f64;
        if compute >= memory { Bound::Compute } else { Bound::Memory }
    }

    /// Percent of the relevant roof a kernel achieved. `seconds` must be a
    /// `poll_wait`-bracketed device time.
    pub fn utilisation(&self, flops: u64, bytes: u64, seconds: f64) -> Option<f32> {
        self.utilisation_of(flops, 0, bytes, seconds)
    }

    /// [`Self::utilisation`] with integer ops kept separate, so int8 work is
    /// graded against the int8 roof.
    pub fn utilisation_of(&self, flops: u64, int_ops: u64, bytes: u64, seconds: f64) -> Option<f32> {
        if seconds <= 0.0 {
            return None;
        }
        let work = flops.max(int_ops);
        match self.classify(work, bytes) {
            Bound::Compute if self.gflops > 0.0 => {
                let (roof, _) = self.compute_roof(flops, int_ops);
                Some(100.0 * (work as f64 / seconds / 1e9) as f32 / roof)
            }
            Bound::Memory if self.gbs > 0.0 => {
                // Graded against whichever memory roof actually applies - DRAM,
                // or the cache roof when the kernel is plainly cache-resident.
                let achieved = bytes as f64 / seconds / 1e9;
                let roof = self.memory_roof(achieved).map(|(r, _)| r).unwrap_or(self.gbs);
                Some(100.0 * achieved as f32 / roof)
            }
            _ => None,
        }
    }
}

/// Which side of the ridge a kernel sits on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Compute,
    Memory,
}

impl Bound {
    /// The band this workstream holds each class to. A well-tuned fp32 GEMM on
    /// old silicon lands well short of peak, so parity with peak is not the
    /// target for either class, and stating a band per class (the values
    /// returned below) is what makes the goal falsifiable.
    pub fn target_pct(self) -> f32 {
        match self {
            Bound::Compute => 60.0,
            Bound::Memory => 70.0,
        }
    }

    /// Below this, treat the kernel as a defect rather than as untuned.
    pub fn defect_pct(self) -> f32 {
        match self {
            Bound::Compute => 30.0,
            Bound::Memory => 35.0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Bound::Compute => "compute",
            Bound::Memory => "memory",
        }
    }
}

/// The probe kernel set - the two sources whose fingerprint keys the cache.
const PROBE_KERNELS: &[(&str, &str)] = &[
    ("roof_fma", kernels::ROOF_FMA),
    ("axpy", kernels::AXPY),
    ("roof_dp4a", kernels::ROOF_DP4A),
];
const K_FMA: usize = 0;
const K_AXPY: usize = 1;
const K_DP4A: usize = 2;
/// Eight `dot4I8Packed` per iteration, each a 4-wide dot = 8 int ops.
const DP4A_OPS_PER_ITER: u64 = 64;

/// Threads the FMA probe launches. Deliberately far more than any device has
/// lanes, so the probe never depends on knowing the topology - an
/// under-subscribed device would report its *latency*, not its throughput.
const FMA_THREADS: u32 = 1 << 20;
/// Eight independent accumulators, one FMA each = 16 FLOP per thread-iteration.
const FMA_FLOPS_PER_ITER: u64 = 16;

/// Elements per bandwidth-probe buffer. Two buffers of 64 Mi f32 = 512 MiB
/// resident, and 768 MiB of traffic per pass - far past any LLC, so the number
/// is DRAM bandwidth rather than cache bandwidth.
const BW_ELEMS: u64 = 64 << 20;
/// Elements per CACHE-roof probe buffer. Two buffers of 1 MiB = 2 MiB resident,
/// which fits a Pascal GP102's 3 MiB L2 and is large enough that the dispatch is
/// still bandwidth-bound rather than launch-bound. A device with a smaller cache
/// measures its DRAM rate here instead, which `measure` clamps to stay monotonic.
const CACHE_ELEMS: u64 = 256 << 10;
/// `out[i] = out[i] + s * in[i]` reads two words and writes one.
const BW_BYTES_PER_ELEM: u64 = 12;
/// A tiny working set needs many more repetitions to clear the launch floor.
const MAX_BW_PASSES: usize = 4096;

/// Grow a probe's trip count until the timed region is at least this long, so
/// launch latency and queue drain cannot dominate the measurement.
const MIN_PROBE_SECONDS: f64 = 0.05;

/// Wall-clock ceiling for ONE calibration loop (compute, int8, or bandwidth).
/// On expiry the loop returns `None` - "roofline unmeasured", the same
/// contract `ensure`'s doc already promises callers (render `-`, never a
/// guess) - rather than blocking indefinitely. Every rung of a healthy probe
/// clears `MIN_PROBE_SECONDS` in one or two dispatches, so this is a
/// generous multiple of that, not a tight budget. Override with
/// `BRAIN_ROOF_BUDGET_S`.
const DEFAULT_ROOF_BUDGET_S: f64 = 10.0;

fn roof_budget() -> std::time::Duration {
    let secs = std::env::var("BRAIN_ROOF_BUDGET_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_ROOF_BUDGET_S);
    std::time::Duration::from_secs_f64(secs)
}

/// Continuous work issued before ANY rung of the probe, so the ceiling is
/// measured at the device's operating point rather than at its idle clock.
///
/// **A roof measured on a parked GPU is worse than no roof**, because it is
/// cached in memory AND persisted to disk (see `persist`), so one cold probe
/// poisons the denominator of every later "% of roof" claim on that machine
/// for good. It is also self-evidently wrong in a way nothing checked: on an
/// integrated Arc (Meteor Lake) a cold probe put the fp32 roof BELOW the rate
/// `matmul_reg2` - a real memory-fed GEMM, which cannot exceed the silicon -
/// was measured at on the same device, by more than a factor of two. The
/// driver's own `gt_act_freq_mhz` sat near its floor for the whole probe
/// against a ceiling several times higher, which is all of the explanation.
/// `tests/roofline.rs::real_work_can_never_beat_the_roof` pins the invariant.
///
/// The `roof_fma` chain is register-resident and dependency-free, so this
/// phase is the most direct way there is to ask the device for its clock.
/// Override with `BRAIN_ROOF_WARMUP_S`; `0` restores the old cold behaviour.
const DEFAULT_ROOF_WARMUP_S: f64 = 2.0;

/// Trip count for one warm-up dispatch: `1 Mi` threads x 64 iterations x 16
/// FLOP is about a GFLOP of work, i.e. milliseconds on any real GPU and still
/// bounded on a slow one. Deliberately NOT the calibration loop's self-scaling
/// count - this phase is not a measurement, and a warm-up whose own dispatch
/// length grew with the device would be unbounded.
const WARMUP_ITERS: u32 = 64;

fn roof_warmup() -> std::time::Duration {
    let secs = std::env::var("BRAIN_ROOF_WARMUP_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(DEFAULT_ROOF_WARMUP_S);
    std::time::Duration::from_secs_f64(secs)
}

/// Which (backend, physical device) a cached/persisted [`Roofs`] belongs to.
///
/// `identity` is `None` on every backend that has not been wired to expose
/// [`backend_api::Backend::identity`] (today: everything except
/// `backend-wgpu`), and two keys with `identity: None` are always treated as
/// the SAME device - the process-wide, first-device-wins behaviour this
/// whole module had before per-device keying existed, preserved exactly for
/// those backends rather than silently changed.
#[derive(Clone)]
struct DeviceKey {
    backend: &'static str,
    identity: Option<backend_api::GpuIdentity>,
}

impl DeviceKey {
    fn of(gpu: &Gpu) -> DeviceKey {
        DeviceKey { backend: gpu.kind(), identity: gpu.identity() }
    }

    /// Same backend, and - where either side has an identity - the SAME
    /// physical card by [`backend_api::GpuIdentity::same_device`]. Two keys
    /// that both carry `None` match (see the struct doc); one `Some` and one
    /// `None` never match, since that can only happen by comparing a
    /// pre-identity-plumbing key against a post-plumbing one within the same
    /// process, which should not be trusted to mean "the same card".
    fn matches(&self, other: &DeviceKey) -> bool {
        self.backend == other.backend
            && match (&self.identity, &other.identity) {
                (Some(a), Some(b)) => a.same_device(b),
                (None, None) => true,
                _ => false,
            }
    }
}

static CACHE: Mutex<Vec<(DeviceKey, Roofs)>> = Mutex::new(Vec::new());

/// Serialises the actual measurement (`ensure`'s cache-miss path): two racing
/// `ensure` calls used to BOTH probe, each measuring a device contended by the
/// other, and the loser persisted a too-low roof to disk - permanently
/// poisoning the denominator every later "%-of-roof" claim divides by (the
/// crate's own `tests/roofline.rs` explains why that race is not benign and
/// guarded itself with a mutex production lacked). Never held across the
/// `CACHE` lock (see `ensure`'s reentrancy note).
///
/// Process-wide rather than per-device: on a multi-GPU box this over-
/// serialises (two INDEPENDENT cards' one-time calibration passes cannot run
/// concurrently), but never mis-measures - the correctness property the
/// races above are guarded against - so it is left process-wide rather than
/// grown into a per-device lock registry that nothing here currently needs.
static MEASURE: Mutex<()> = Mutex::new(());

/// Which `DeviceKey`s have already concluded "unprobeable" this process: the
/// probe allocates 512 MiB and can take the full `roof_budget()` before
/// reaching that conclusion, and re-running it on every later `ensure` call
/// on the SAME device repays that cost for the same answer. Per-device (not
/// one process-wide flag): a multi-GPU box where one card genuinely cannot be
/// probed (contended, driver fault, whatever) must not also silently refuse
/// to probe every OTHER card in the process - that would be exactly the
/// "one shared/clobbered entry" bug this module's per-device keying exists
/// to remove, just relocated into this flag instead of `CACHE`.
static MEASURE_FAILED: Mutex<Vec<DeviceKey>> = Mutex::new(Vec::new());

/// The roofs for this process's device, measuring them once if needed.
///
/// Returns `None` when the device cannot be probed - callers must then print
/// `-`, never a guess. Set `BRAIN_NO_ROOF=1` to skip probing entirely (useful
/// when a run must not spend the probe's fraction of a second, or to reproduce
/// pre-roofline output).
///
/// Defaults to skipped (as if `BRAIN_NO_ROOF=1`) on the CPU backend, where the
/// probe's calibration loop has a known-bad interaction with `backend-cpu`'s
/// rayon dispatch - a per-call-site opt-out is what every new caller was
/// once asked to add by hand; this applies it once at the source instead.
/// Set `BRAIN_NO_ROOF=0` to force the probe to run anyway, even on the CPU
/// backend.
pub fn ensure(gpu: &Gpu) -> Option<Roofs> {
    match std::env::var("BRAIN_NO_ROOF").as_deref() {
        Ok(v) if v != "0" => return None,
        Ok("0") => {} // explicit opt-in, overrides the CPU-default-off below
        _ => {
            if gpu.caps().class == backend_api::DeviceClass::Cpu {
                return None;
            }
        }
    }
    // Each access below takes and releases `CACHE`'s lock separately, NEVER
    // held across `measure(gpu)`: `measure` reads `gpu.caps()` (to gate the
    // int8/f16 probes on `numeric.int8_dot`/`numeric.f16`), and `Gpu::caps`
    // overlays `known()`, which locks this SAME `CACHE` - a `std::sync::Mutex`
    // is not reentrant, so holding a guard here across that call self-deadlocks
    // the calling thread forever on any cold-cache (first-ever, or
    // on-disk-store-miss) call. Confirmed on real hardware: `caps_expose_the_
    // roofs_only_after_something_measured_them` hung indefinitely (gdb: thread
    // parked in `Mutex::lock` on `roof::CACHE`, called from `roof::known` <-
    // `Gpu::caps` <- `roof::measure` <- this function, one level up, on the
    // SAME thread) - not a GPU/driver issue at all, despite presenting
    // exactly like the driver hangs this module's `ensure`/wait-bound work
    // was written to guard against. A benign race remains (two threads both
    // missing the cache both call `measure`; last write wins) - acceptable,
    // matching this function's own existing double-checked-init shape.
    let key = DeviceKey::of(gpu);
    if let Some(r) = cached_for(&key) {
        return Some(r);
    }
    if measure_failed(&key) {
        return None; // already concluded unprobeable this process - see MEASURE_FAILED
    }
    // Serialise the miss path: only one thread measures; the others wait and
    // re-check the cache - measuring a device the winner is saturating would
    // record (and persist) a contended, too-low roof. This guard is NOT held
    // while taking the CACHE lock's guard beyond a single statement, so the
    // known()/caps() reentrancy hazard above cannot involve it.
    let _measuring = MEASURE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = cached_for(&key) {
        return Some(r); // the winner filled it while this thread waited
    }
    let store = persist::store(&key);
    if let Some(r) = store.as_ref().and_then(|s| s.load()) {
        cache_insert(&key, r);
        return Some(r);
    }
    let Some(r) = measure(gpu) else {
        mark_measure_failed(&key);
        return None;
    };
    if let Some(s) = store.as_ref() {
        s.save(r);
    }
    cache_insert(&key, r);
    Some(r)
}

/// The in-memory record, but only if it was measured for the SAME `key` -
/// backend AND (where either side has one) physical device - see
/// [`DeviceKey::matches`] and `persist::store`'s doc. A process that builds
/// several devices (several backends, or several physical cards behind one
/// backend) must not serve one's number as another's.
fn cached_for(key: &DeviceKey) -> Option<Roofs> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| k.matches(key)).map(|(_, r)| *r)
}

/// Record (or replace) `key`'s measurement - an upsert, since a later
/// [`reprofile`] must overwrite an earlier entry for the SAME device rather
/// than accumulate a duplicate.
fn cache_insert(key: &DeviceKey, r: Roofs) {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    match cache.iter_mut().find(|(k, _)| k.matches(key)) {
        Some(slot) => slot.1 = r,
        None => cache.push((key.clone(), r)),
    }
}

fn measure_failed(key: &DeviceKey) -> bool {
    MEASURE_FAILED.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|k| k.matches(key))
}

fn mark_measure_failed(key: &DeviceKey) {
    let mut failed = MEASURE_FAILED.lock().unwrap_or_else(|e| e.into_inner());
    if !failed.iter().any(|k| k.matches(key)) {
        failed.push(key.clone());
    }
}

/// Redirect where measured roofs persist (`None` restores the default
/// `~/.cache/brain` / `BRAIN_PIPELINE_CACHE_DIR` resolution). A TEST seam:
/// tests that exercise `ensure()` point this at a temp dir so they never
/// mutate the developer's real cache - see `tests/roofline.rs`.
pub fn set_cache_dir(dir: Option<std::path::PathBuf>) {
    persist::set_dir_override(dir);
}

/// Whatever has already been measured, without measuring. This is what
/// [`Gpu::caps`] overlays onto [`backend_api::DeviceCaps`] - reading caps must
/// never have the side effect of running kernels. `identity` should be the
/// SAME handle's [`Gpu::identity`] - passing `None` for a device that has one
/// would look up the shared no-identity slot instead of this specific card's.
pub fn known(backend: &'static str, identity: Option<&backend_api::GpuIdentity>) -> Option<Roofs> {
    cached_for(&DeviceKey { backend, identity: identity.cloned() })
}

/// Force a fresh measurement of `gpu`'s roofline, bypassing (and then
/// overwriting) both the in-memory and on-disk cache [`ensure`] would
/// otherwise serve. `BRAIN_NO_ROOF=1` still wins - a caller asking to skip
/// probing entirely is honored even here, so this is "reprofile if roofing is
/// allowed at all", not an unconditional override of that opt-out.
///
/// The one entry point `brain models list --reprofile` (and `brain models
/// profile`) needs: every OTHER caller wants [`ensure`]'s cache-first
/// behavior, and should keep calling that, not this.
pub fn reprofile(gpu: &Gpu) -> Option<Roofs> {
    if matches!(std::env::var("BRAIN_NO_ROOF").as_deref(), Ok(v) if v != "0") {
        return None;
    }
    let key = DeviceKey::of(gpu);
    let r = measure(gpu)?;
    if let Some(s) = persist::store(&key) {
        s.save(r);
    }
    cache_insert(&key, r);
    Some(r)
}

/// Run both probes now, ignoring and not updating the cache. Exposed so a test
/// can assert the measurement itself rather than a memoised value.
pub fn measure(gpu: &Gpu) -> Option<Roofs> {
    let g = gpu.new_like(PROBE_KERNELS);
    // Bring the device to its operating point first, or every rung below
    // measures the idle clock and persists it as this machine's ceiling.
    warm_up(&g);
    let gflops = measure_compute(&g)?;
    let gbs = measure_bandwidth(&g, BW_ELEMS)?;
    // The same triad over a working set chosen to stay in cache. Never report a
    // cache roof BELOW the DRAM roof: on a device whose cache is smaller than
    // the probe (or which has none worth the name) the two converge, and the
    // max keeps the hierarchy monotonic so `memory_roof` cannot invert.
    let cache_gbs = measure_bandwidth(&g, CACHE_ELEMS)?.max(gbs);
    // `None` where the device has no int8 dot path - never a guess, and never
    // fp32's number standing in for it.
    let int8_gops = gpu.caps().numeric.int8_dot.then(|| measure_int8(&g)).flatten();
    // Same rule as int8: `None` unless the device has already been VERIFIED
    // to run native f16 fast (`caps().numeric.f16`) - this stays `false`
    // (hence this probe stays unrun) on every backend until something wires
    // up that verification; see `Roofs::f16_gflops`'s own doc. Built off `g`
    // rather than `gpu` so the f16 kernel compiles onto the SAME already-warm
    // device the other probes just measured, not a fresh cold one.
    let f16_gflops = gpu.caps().numeric.f16.then(|| measure_f16(&g)).flatten();
    Some(Roofs { gflops, gbs, cache_gbs, int8_gops, f16_gflops })
}

/// A single `submit`+`poll_wait` in this probe must complete within this
/// budget or the whole measurement is abandoned (`best_of` returns `None`).
/// Generous relative to a healthy measurement (clean rounds in this probe
/// take well under a second even on a slow iGPU) but far below the
/// multi-minute stalls this exists to bound - see `poll_wait_timeout`'s doc.
/// Only bounds anything on backends that actually implement `poll_wait_timeout`
/// (currently `backend-wgpu`); on others this is a no-op budget the default
/// trait method ignores.
const PER_DISPATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Time one submit of `steps`, best of `reps`, `poll_wait`-bracketed. `None`
/// if any single submit (including the warmup) does not complete within
/// `PER_DISPATCH_TIMEOUT` - the caller must not keep using `gpu` for further
/// timed measurements after that (see `poll_wait_timeout`'s doc: a timeout
/// leaves completion state unknown on backends where reuse would not be
/// safe), so every caller here treats `None` as "abandon this probe."
///
/// Stops repping once `deadline` passes (keeping whatever best it already
/// timed): the calibration loops used to check their budget only BETWEEN
/// `best_of` calls, so one call issued just under the wire could legally run
/// `1 + reps` dispatches, each up to `PER_DISPATCH_TIMEOUT`, past the
/// deadline - an effectively unbounded overshoot the roofline wall-clock test
/// could only paper over with slack.
///
/// The bracketing is the whole point: `submit` only appends to the pending list
/// on the wgpu backend, so an unbracketed loop times host-side recording and
/// reports a rate above the hardware roof (§E.0).
fn best_of(gpu: &Gpu, steps: &[crate::Step], reps: usize, deadline: Instant) -> Option<f64> {
    gpu.submit(&[], steps);
    if !gpu.poll_wait_timeout(PER_DISPATCH_TIMEOUT) {
        return None;
    }
    let mut best = f64::INFINITY;
    for i in 0..reps {
        let t0 = Instant::now();
        gpu.submit(&[], steps);
        if !gpu.poll_wait_timeout(PER_DISPATCH_TIMEOUT) {
            return None;
        }
        best = best.min(t0.elapsed().as_secs_f64());
        if i + 1 < reps && Instant::now() >= deadline {
            break; // budget exhausted mid-call: report what was measured
        }
    }
    Some(best)
}

/// Issue the FMA probe back to back until [`roof_warmup`] has elapsed, so the
/// rungs that follow time a device that is awake. Nothing is recorded: this is
/// the ramp, not the measurement.
///
/// Adds at most `roof_warmup()` plus one dispatch to `measure`'s wall clock -
/// the deadline is checked before each submit, and `WARMUP_ITERS` bounds how
/// long any single one can be. A device that times out mid-warm-up returns
/// early and lets the first real rung report the failure.
fn warm_up(gpu: &Gpu) {
    let budget = roof_warmup();
    if budget.is_zero() {
        return;
    }
    let inp = gpu.storage(FMA_THREADS as u64);
    let out = gpu.storage(FMA_THREADS as u64);
    let (c, d) = (backend_api::f(0.5), backend_api::f(0.5));
    let step = gpu.step(K_FMA, &[&inp, &out], &[FMA_THREADS, WARMUP_ITERS, c, d], FMA_THREADS);
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        gpu.submit(&[], std::slice::from_ref(&step));
        if !gpu.poll_wait_timeout(PER_DISPATCH_TIMEOUT) {
            return;
        }
    }
}

fn measure_compute(gpu: &Gpu) -> Option<f32> {
    let inp = gpu.storage(FMA_THREADS as u64);
    let out = gpu.storage(FMA_THREADS as u64);
    gpu.write_f32(&inp, &vec![1.0f32; FMA_THREADS as usize]);

    // c = 0.5, d = 0.5 has fixed point 1.0: the chain converges immediately and
    // then stays exactly 1.0 - no overflow, no denormals (slow on some
    // hardware, which would understate the roof), no NaNs.
    let (c, d) = (backend_api::f(0.5), backend_api::f(0.5));

    let deadline = Instant::now() + roof_budget();
    let mut iters: u32 = 256;
    loop {
        if Instant::now() >= deadline {
            return None;
        }
        let step = gpu.step(K_FMA, &[&inp, &out], &[FMA_THREADS, iters, c, d], FMA_THREADS);
        let secs = best_of(gpu, std::slice::from_ref(&step), 3, deadline)?;
        if secs >= MIN_PROBE_SECONDS || iters >= (1 << 20) {
            let flops = FMA_THREADS as u64 * iters as u64 * FMA_FLOPS_PER_ITER;
            return Some((flops as f64 / secs / 1e9) as f32);
        }
        // Scale straight to the target rather than doubling blindly, then round
        // up, so a fast device converges in one extra step instead of ten.
        let want = (iters as f64 * MIN_PROBE_SECONDS / secs.max(1e-9)).ceil();
        // NOT `clamp(lo, hi)`: once `iters*2` passes the ceiling the arguments
        // invert and `clamp` panics with `min > max`. Grow by at least 2x, cap
        // at the ceiling, in that order.
        iters = (want as u32).max(iters.saturating_mul(2)).min(1 << 20);
    }
}

/// Peak packed-int8 rate, same calibration loop as the fp32 probe.
fn measure_int8(gpu: &Gpu) -> Option<f32> {
    let inp = gpu.storage(FMA_THREADS as u64);
    let out = gpu.storage(FMA_THREADS as u64);
    // Four int8 lanes packed per u32; values chosen so the accumulator cannot
    // overflow i32 across the loop.
    let (a, b) = (0x01010101u32, 0x01010101u32);
    let deadline = Instant::now() + roof_budget();
    let mut iters: u32 = 256;
    loop {
        if Instant::now() >= deadline {
            return None;
        }
        let step = gpu.step(K_DP4A, &[&inp, &out], &[FMA_THREADS, iters, a, b], FMA_THREADS);
        let secs = best_of(gpu, std::slice::from_ref(&step), 3, deadline)?;
        if secs >= MIN_PROBE_SECONDS || iters >= (1 << 20) {
            let ops = FMA_THREADS as u64 * iters as u64 * DP4A_OPS_PER_ITER;
            return Some((ops as f64 / secs / 1e9) as f32);
        }
        let want = (iters as f64 * MIN_PROBE_SECONDS / secs.max(1e-9)).ceil();
        iters = (want as u32).max(iters.saturating_mul(2)).min(1 << 20);
    }
}

/// Peak native-`f16` FMA rate (B11's `enable f16;` register-typed arithmetic,
/// NOT the storage-tier `#w=f16` decode, which stays fp32 compute the whole
/// time and needs no gate at all). Same calibration loop as the fp32 probe
/// and the SAME dispatch shape (`kernels::template::native_f16_poc::ROOF_FMA`
/// mirrors `kernels::ROOF_FMA` accumulator-for-accumulator - see that
/// constant's own doc), so the two rates are directly comparable.
///
/// Deliberately built on its OWN single-kernel device handle
/// (`gpu.new_like`) rather than folded into the shared `PROBE_KERNELS` list
/// every `measure()` call compiles unconditionally: `enable f16;` WGSL only
/// compiles where `wgpu::Features::SHADER_F16` was granted at device
/// creation, and other backends' compilers are not guaranteed to accept it
/// at all (the CPU JIT's `wgsl_cpu::Ty` lattice has no f16 entry and silently
/// aliases it to f32 instead - see `native_f16_variant`'s own doc for the
/// proof). Keeping this kernel out of `PROBE_KERNELS` is what keeps the
/// probe INERT - never even attempted - on any backend `measure`'s caller
/// has not gated true, rather than relying on every non-wgpu backend to
/// happen to reject or safely alias source it was never meant to see.
///
/// No caps gate in here - the caller (`measure`) applies it, exactly as
/// `measure_int8` leaves its own `int8_dot` gate to its caller.
fn measure_f16(gpu: &Gpu) -> Option<f32> {
    const K_FMA_F16: usize = 0; // this handle's only compiled kernel
    let (name, src) =
        kernels::template::native_f16_variant("roof_fma_f16", kernels::template::native_f16_poc::ROOF_FMA);
    let g = gpu.new_like(&[(name, src)]);
    let inp = g.storage(FMA_THREADS as u64);
    let out = g.storage(FMA_THREADS as u64);
    g.write_f32(&inp, &vec![1.0f32; FMA_THREADS as usize]);
    // Same fixed point as the fp32 probe (`c = d = 0.5` -> steady state
    // `1.0`): no overflow, no denormals, no NaNs, sourced from the uniform so
    // the loop cannot be constant-folded away.
    let (c, d) = (backend_api::f(0.5), backend_api::f(0.5));
    let deadline = Instant::now() + roof_budget();
    let mut iters: u32 = 256;
    loop {
        if Instant::now() >= deadline {
            return None;
        }
        let step = g.step(K_FMA_F16, &[&inp, &out], &[FMA_THREADS, iters, c, d], FMA_THREADS);
        let secs = best_of(&g, std::slice::from_ref(&step), 3, deadline)?;
        if secs >= MIN_PROBE_SECONDS || iters >= (1 << 20) {
            let flops = FMA_THREADS as u64 * iters as u64 * FMA_FLOPS_PER_ITER;
            return Some((flops as f64 / secs / 1e9) as f32);
        }
        let want = (iters as f64 * MIN_PROBE_SECONDS / secs.max(1e-9)).ceil();
        iters = (want as u32).max(iters.saturating_mul(2)).min(1 << 20);
    }
}

fn measure_bandwidth(gpu: &Gpu, elems: u64) -> Option<f32> {
    let out = gpu.storage(elems);
    let inp = gpu.storage(elems);
    // Leave both buffers at their zero-initialised contents: axpy's rate depends
    // on bytes moved, not on values, and writing 512 MiB from the host would
    // cost far more than the measurement.

    // One pass over 512 MiB is already only milliseconds on a fast card;
    // repeat the same dispatch until the timed region clears the launch floor.
    let deadline = Instant::now() + roof_budget();
    let mut passes = 1usize;
    loop {
        if Instant::now() >= deadline {
            return None;
        }
        let steps: Vec<crate::Step> = (0..passes)
            .map(|_| {
                gpu.step(
                    K_AXPY,
                    &[&out, &inp],
                    &[elems as u32, backend_api::f(1.0)],
                    elems as u32,
                )
            })
            .collect();
        let secs = best_of(gpu, &steps, 3, deadline)?;
        if secs >= MIN_PROBE_SECONDS || passes >= MAX_BW_PASSES {
            let bytes = elems * BW_BYTES_PER_ELEM * passes as u64;
            return Some((bytes as f64 / secs / 1e9) as f32);
        }
        let want = (passes as f64 * MIN_PROBE_SECONDS / secs.max(1e-9)).ceil();
        // See the note in `measure_compute`: `clamp` panics when the bounds
        // invert, which they do as soon as `passes * 2` exceeds the ceiling.
        passes = (want as usize).max(passes * 2).min(MAX_BW_PASSES);
    }
}

/// Per-adapter persistence, mirroring [`crate::tune`]'s discipline: keyed by the
/// adapter slug plus a fingerprint of the probe sources, so editing a probe
/// invalidates old numbers by filename rather than by trusting them.
mod persist {
    use super::{DeviceKey, Roofs};
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Process-local override of the persist directory. Set via
    /// [`super::set_cache_dir`] - primarily a TEST seam: without it, any test
    /// that reaches `ensure()` writes the developer's real `~/.cache/brain/`
    /// (state outside the repo), and the only alternative was mutating
    /// process-wide env vars from test threads.
    static DIR_OVERRIDE: Mutex<Option<PathBuf>> = Mutex::new(None);

    pub fn set_dir_override(dir: Option<PathBuf>) {
        *DIR_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = dir;
    }

    pub struct RoofStore {
        path: PathBuf,
    }

    /// `key.backend` keys the record alongside the adapter, because a roof is
    /// a property of the (device, backend) PAIR, not of the silicon alone:
    /// the same P40 measured roughly HALF the fp32 roof through
    /// `backend-vulkan` that it did through `backend-wgpu`, while the two
    /// compiled kernels under different naga runtime-check settings. Without
    /// this, whichever backend measured first silently published its number
    /// as the other's roof, and every "% of roof" on that backend was wrong
    /// by that ratio.
    ///
    /// `key.identity`, where the backend exposes one (today: `backend-wgpu`
    /// only - see `backend_api::Backend::identity`), additionally keys the
    /// filename on the PHYSICAL card: a box with two GPUs behind the same
    /// backend used to publish ONE shared file, so whichever card measured
    /// first silently became "the roof" for both - a plugged-in-and-measured
    /// second card would then either overwrite that file (poisoning the
    /// first card's numbers) or read it back as its own (poisoning the
    /// second's). The identity slug (PCI bus / UUID / `vendor:device:ordinal`,
    /// see `identity_slug`'s own doc for the priority) makes the two cards'
    /// files distinct by construction.
    ///
    /// The human-readable part of the slug prefers `key.identity`'s own
    /// device name where present - correct even for the SECOND device on a
    /// multi-GPU box - and falls back to `adapter_info()` (this PROCESS's
    /// first-built wgpu adapter) only where no identity exists, unchanged
    /// from before this existed.
    pub fn store(key: &DeviceKey) -> Option<RoofStore> {
        let dir = cache_dir()?;
        let hash = crate::tune::source_fingerprint(
            &super::PROBE_KERNELS.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
        );
        let desc = match &key.identity {
            Some(id) => id.name.clone(),
            None => crate::adapter_info()?.0,
        };
        let mut slug: String = desc
            .chars()
            .chain(std::iter::once('-'))
            .chain(key.backend.chars())
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        if let Some(id) = &key.identity {
            slug.push('-');
            slug.push_str(&identity_slug(id));
        }
        Some(RoofStore { path: dir.join(format!("roof-{slug}-{hash:016x}.txt")) })
    }

    /// A filesystem-safe, per-physical-card suffix: the strongest identity key
    /// this device reported, preferring (in order) the Vulkan `deviceUUID`,
    /// the PCI bus id, and finally the `(vendor:device, ordinal)` fallback -
    /// the same priority `GpuIdentity::same_device` itself uses, so two
    /// identities this crate considers "the same card" always produce the
    /// same slug and two it considers different always produce different ones.
    fn identity_slug(id: &backend_api::GpuIdentity) -> String {
        let raw = if let Some(uuid) = id.uuid {
            uuid.iter().map(|b| format!("{b:02x}")).collect::<String>()
        } else if let Some(pci) = &id.pci_bus {
            pci.clone()
        } else {
            format!("{:04x}_{:04x}_{}", id.vendor_id, id.device_id, id.ordinal)
        };
        raw.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
    }

    fn cache_dir() -> Option<PathBuf> {
        if let Some(d) = DIR_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return Some(d);
        }
        // ONE resolution, shared with the tune store (this was a verbatim copy).
        crate::tune::cache_dir()
    }

    impl RoofStore {
        /// A missing, unparseable or non-finite record is ignored, never
        /// trusted - the same rule the tune store applies.
        pub fn load(&self) -> Option<Roofs> {
            let text = std::fs::read_to_string(&self.path).ok()?;
            let (mut gflops, mut gbs, mut cache, mut int8, mut f16) = (None, None, None, None, None);
            for line in text.lines() {
                let (k, v) = line.split_once('=')?;
                let v: f32 = v.trim().parse().ok()?;
                match k.trim() {
                    "gflops" => gflops = Some(v),
                    "gbs" => gbs = Some(v),
                    "cache_gbs" => cache = Some(v),
                    "int8_gops" => int8 = Some(v),
                    "f16_gflops" => f16 = Some(v),
                    _ => {}
                }
            }
            // A record written before `cache_gbs` existed parses to `None` here
            // and is simply re-measured - a stale cache is ignored, never
            // patched up with a default.
            // `int8_gops`/`f16_gflops` are legitimately absent - on a device
            // without DP4A, or on any record written before `f16_gflops`
            // existed at all - so a record missing either is still valid,
            // unlike the other three.
            let r = Roofs { gflops: gflops?, gbs: gbs?, cache_gbs: cache?, int8_gops: int8, f16_gflops: f16 };
            (r.gflops.is_finite()
                && r.gflops > 0.0
                && r.gbs.is_finite()
                && r.gbs > 0.0
                && r.cache_gbs >= r.gbs)
                .then_some(r)
        }

        pub fn save(&self, r: Roofs) {
            if let Some(parent) = self.path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let tmp = self.path.with_extension("tmp");
            let body = format!(
                "gflops={}\ngbs={}\ncache_gbs={}\n{}{}",
                r.gflops,
                r.gbs,
                r.cache_gbs,
                r.int8_gops.map(|v| format!("int8_gops={v}\n")).unwrap_or_default(),
                r.f16_gflops.map(|v| format!("f16_gflops={v}\n")).unwrap_or_default(),
            );
            if std::fs::write(&tmp, body).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A record written by a pre-f16 build of this crate has no
        /// `f16_gflops=` line at all - `save`'s own format never wrote one
        /// before `Roofs::f16_gflops` existed. `load` must still parse it
        /// (never crash / never treat the missing line as corruption) and
        /// come back with `f16_gflops: None`, exactly the same "legitimately
        /// absent" treatment `int8_gops` already got when THIS format grew a
        /// DP4A line on top of the original three-field one.
        #[test]
        fn old_persisted_record_without_f16_gflops_line_loads_as_none() {
            let dir = std::env::temp_dir()
                .join(format!("brain-roof-persist-oldfmt-{}-{:?}", std::process::id(), std::thread::current().id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("roof-old-format.txt");
            // Byte-for-byte what `save` wrote before this change: three
            // required fields plus the (already-optional) int8 line, and
            // nothing else.
            std::fs::write(&path, "gflops=1000.0\ngbs=100.0\ncache_gbs=400.0\nint8_gops=5000.0\n").unwrap();

            let store = RoofStore { path: path.clone() };
            let r = store.load().expect("an old-format record (no f16_gflops line) must still load");
            assert_eq!(r.f16_gflops, None, "a record predating f16_gflops must come back None, not crash or invent a value");
            assert_eq!(r.gflops, 1000.0);
            assert_eq!(r.gbs, 100.0);
            assert_eq!(r.cache_gbs, 400.0);
            assert_eq!(r.int8_gops, Some(5000.0));

            std::fs::remove_dir_all(&dir).ok();
        }

        /// Even older: a record from before `int8_gops` existed either (the
        /// original three-field format). Both new-since fields come back
        /// `None`.
        #[test]
        fn very_old_persisted_record_without_int8_or_f16_lines_loads_as_none_for_both() {
            let dir = std::env::temp_dir().join(format!(
                "brain-roof-persist-veryoldfmt-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("roof-very-old-format.txt");
            std::fs::write(&path, "gflops=2000.0\ngbs=200.0\ncache_gbs=800.0\n").unwrap();

            let store = RoofStore { path: path.clone() };
            let r = store.load().expect("the original three-field record must still load");
            assert_eq!(r.int8_gops, None);
            assert_eq!(r.f16_gflops, None);

            std::fs::remove_dir_all(&dir).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_and_classification_split_at_the_ridge() {
        let r = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0), f16_gflops: None };
        // A P40's ridge is ~34 FLOP/byte - anything streaming is left of it.
        assert!((r.ridge() - 33.99).abs() < 0.1, "ridge {}", r.ridge());
        // axpy: 2 FLOP per 12 bytes.
        assert_eq!(r.classify(2, 12), Bound::Memory);
        // A square GEMM at 1024: 2*n^3 flops over ~3*n^2*4 bytes = 170 FLOP/byte.
        assert_eq!(r.classify(2 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 4), Bound::Compute);
        // A kernel that moves nothing can only be compute-bound.
        assert_eq!(r.classify(100, 0), Bound::Compute);
    }

    #[test]
    fn utilisation_is_measured_against_the_kernels_own_roof() {
        let r = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0), f16_gflops: None };
        // A streaming kernel at exactly the bandwidth roof reads as fully
        // utilised, even though its FLOP rate is negligible - the whole point
        // of classifying.
        let bytes = 346_000_000_000u64;
        let u = r.utilisation(bytes / 6, bytes, 1.0).unwrap();
        assert!((u - 100.0).abs() < 0.5, "{u}");
        // Zero-length regions produce nothing rather than infinity.
        assert!(r.utilisation(1, 1, 0.0).is_none());
    }

    /// The bound for an AGGREGATE (a pipeline stage, a whole generation) must
    /// count both op classes. An int8 diffusion transformer still runs its
    /// attention and its norms in fp32, and grading the stage by
    /// `max(flops, int_ops)` against one roof makes whichever class loses that
    /// comparison disappear entirely - which is how an int8 denoise step came
    /// out cheaper than its fp32 attention alone.
    #[test]
    fn a_mixed_precision_stage_is_not_graded_against_one_roof() {
        let r = Roofs { gflops: 10_000.0, gbs: 250.0, cache_gbs: 1000.0, int8_gops: Some(40_000.0), f16_gflops: None };
        // Exactly one second of fp32 work at that roof, plus exactly one
        // second of int8 work at that roof.
        let (fp, int) = (10_000_000_000_000u64, 40_000_000_000_000u64);
        let t = r.seconds_at_roof(fp, int, 0).unwrap();
        assert!((t - 2.0).abs() < 1e-9, "the two classes share the SMs, so their times add: {t}");
        // Each class alone is half of it - so neither can have been dropped.
        assert!((r.seconds_at_roof(fp, 0, 0).unwrap() - 1.0).abs() < 1e-9);
        assert!((r.seconds_at_roof(0, int, 0).unwrap() - 1.0).abs() < 1e-9);
        // Memory overlaps with compute rather than adding to it.
        assert!((r.seconds_at_roof(fp, int, 250_000_000_000).unwrap() - 2.0).abs() < 1e-9);
        // ...until it dominates.
        assert!((r.seconds_at_roof(fp, int, 2_500_000_000_000).unwrap() - 10.0).abs() < 1e-9);
        assert_eq!(r.bound_of(fp, int, 2_500_000_000_000), Bound::Memory);
        assert_eq!(r.bound_of(fp, int, 250_000_000_000), Bound::Compute);
        // No work at all is not "instant", it is nothing to report.
        assert!(r.seconds_at_roof(0, 0, 0).is_none());
        // An unmeasured int8 roof falls back to fp32, which UNDER-states int8
        // throughput - the safe direction for a lower bound on time.
        let no_i8 = Roofs { int8_gops: None, ..r };
        assert!(no_i8.seconds_at_roof(0, int, 0).unwrap() > r.seconds_at_roof(0, int, 0).unwrap());
    }

    #[test]
    fn target_bands_are_stated_per_class() {
        assert!(Bound::Compute.target_pct() > Bound::Compute.defect_pct());
        assert!(Bound::Memory.target_pct() > Bound::Memory.defect_pct());
    }

    fn skip_gpu() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").map(|v| v != "0").unwrap_or(false)
    }

    /// The f16 probe (B11-shaped, `enable f16;` real registers) has two
    /// halves to prove:
    ///
    /// 1. On every device and backend that exists TODAY, `caps().numeric.f16`
    ///    is unconditionally `false` (its own doc: "stays false until the
    ///    autotuner measures it" - nothing in this codebase sets it yet), so
    ///    `measure()` must gate the probe off and report `f16_gflops: None` -
    ///    checked here against a REAL device, not assumed.
    /// 2. Where hardware genuinely DOES run native f16 (checked the honest
    ///    way `backend-wgpu`'s own `tests/native_f16.rs` does - the adapter's
    ///    `wgpu::Features::SHADER_F16`, not the always-false caps flag), the
    ///    measurement mechanism itself must be self-consistent: a real f16
    ///    ALU is never slower than the same silicon's fp32 path, so
    ///    `f16_gflops >= gflops` is asserted, not merely "is Some". This half
    ///    calls `measure_f16`/`measure_compute` directly, bypassing the
    ///    (currently permanently-closed) `caps().numeric.f16` gate - exactly
    ///    as `native_f16.rs` measures the real mechanism directly rather than
    ///    through a flag nothing sets.
    ///
    /// The two probes are independent calibration loops on separate device
    /// handles seconds apart, so the comparison samples the machine at two
    /// different moments - and the fast lane runs suites in parallel, so
    /// concurrent load can independently depress either side by more than
    /// the f16-vs-fp32 margin (observed once: f16 265 vs fp32 279 GFLOP/s
    /// under a 4-thread lane, passing in isolation immediately after). The
    /// halves are therefore interleaved for several rounds and compared
    /// best-vs-best: both sides then sample the same clock/thermal window,
    /// and one depressed sample cannot decide a hardware-invariant verdict.
    #[test]
    fn f16_roof_is_none_while_uncapped_and_never_slower_than_fp32_where_hardware_supports_it() {
        if skip_gpu() {
            return;
        }
        let gpu = crate::testgpu::dev(PROBE_KERNELS);

        // (1) The gated, production path, on a real device.
        assert!(
            !gpu.caps().numeric.f16,
            "this test's premise (nothing sets NumericSupport.f16 yet) no longer holds - \
             see Roofs::f16_gflops' own doc for what changes if it does"
        );
        if let Some(r) = measure(&gpu) {
            assert!(r.f16_gflops.is_none(), "f16_gflops must stay None while caps().numeric.f16 is false, got {:?}", r.f16_gflops);
        }

        // (2) The raw mechanism, on real f16-capable hardware only.
        if gpu.kind() != "wgpu" {
            return; // native f16 compute exists only on this backend
        }
        let probe = backend_wgpu::WgpuBackend::new(&[("axpy", kernels::AXPY)]);
        if !probe.supports_shader_f16() {
            brain_testutil::skip_unavailable(
                "f16_roof_is_none_while_uncapped_and_never_slower_than_fp32_where_hardware_supports_it: \
                 this adapter does not report wgpu::Features::SHADER_F16",
            );
            return;
        }
        let (mut best_fp32, mut best_fp16) = (None, None);
        for _ in 0..5 {
            if let Some(v) = measure_compute(&gpu) {
                if best_fp32.is_none_or(|b| v > b) {
                    best_fp32 = Some(v);
                }
            }
            if let Some(v) = measure_f16(&gpu) {
                if best_fp16.is_none_or(|b| v > b) {
                    best_fp16 = Some(v);
                }
            }
        }
        let (Some(fp32), Some(fp16)) = (best_fp32, best_fp16) else {
            return; // unprobeable device - callers print `-`, never a guess
        };
        assert!(fp16.is_finite() && fp16 > 0.0, "f16 roof {fp16}");
        assert!(
            fp16 >= fp32,
            "native f16 measured {fp16:.0} GFLOP/s, SLOWER than fp32's {fp32:.0} GFLOP/s on hardware \
             that reports SHADER_F16 - a real f16 ALU cannot be slower than fp32 on the same silicon"
        );
        eprintln!("native f16 roof: {fp16:.0} GFLOP/s vs fp32 {fp32:.0} GFLOP/s ({:.2}x)", fp16 / fp32);
    }

    /// Two distinct physical GPUs (synthetic identities here - this sandbox
    /// has one card; a real multi-GPU box would use `backend_wgpu::
    /// enumerate_gpus()`'s own two entries instead) must land in two
    /// DISTINCT cache slots, in memory and on disk - never one clobbering
    /// the other, which is exactly the "first-device-wins" bug per-device
    /// keying exists to remove. Needs no live device at all: `DeviceKey`,
    /// `cache_insert`/`cached_for` and `persist::store` are pure functions of
    /// the identity data once a `GpuIdentity` exists, which is why this can
    /// construct two by hand rather than needing real hardware.
    #[test]
    fn two_distinct_gpu_identities_never_clobber_each_others_cache_entry() {
        let id_a = backend_api::GpuIdentity {
            name: "Synthetic Test GPU".into(),
            vendor_id: 0x10de,
            device_id: 0x1234,
            uuid: None,
            pci_bus: Some("0000:01:00.0".into()),
            ordinal: 0,
            vram_bytes: 0,
            class: backend_api::DeviceClass::DiscreteGpu,
        };
        let id_b = backend_api::GpuIdentity { pci_bus: Some("0000:02:00.0".into()), ..id_a.clone() };
        assert!(!id_a.same_device(&id_b), "the two synthetic identities must be distinct PCI devices");

        let key_a = DeviceKey { backend: "wgpu", identity: Some(id_a) };
        let key_b = DeviceKey { backend: "wgpu", identity: Some(id_b) };
        assert!(!key_a.matches(&key_b), "DeviceKey must not treat two distinct physical cards as the same device");

        // In-memory cache: inserting both must produce two independent slots.
        let ra = Roofs { gflops: 111.0, gbs: 11.0, cache_gbs: 22.0, int8_gops: None, f16_gflops: None };
        let rb = Roofs { gflops: 222.0, gbs: 33.0, cache_gbs: 44.0, int8_gops: Some(999.0), f16_gflops: None };
        cache_insert(&key_a, ra);
        cache_insert(&key_b, rb);
        assert_eq!(cached_for(&key_a), Some(ra), "card A's cache entry must be card A's own measurement");
        assert_eq!(cached_for(&key_b), Some(rb), "card B's cache entry must be card B's own measurement, not A's");

        // On-disk persistence: two distinct files, each loading back its own
        // card's numbers.
        let dir = std::env::temp_dir()
            .join(format!("brain-roof-devicekey-test-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        persist::set_dir_override(Some(dir.clone()));

        persist::store(&key_a).expect("a synthetic identity still resolves to a store").save(ra);
        persist::store(&key_b).expect("a synthetic identity still resolves to a store").save(rb);

        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(files.len(), 2, "two distinct physical identities must persist to two distinct files, got {files:?}");
        assert_eq!(persist::store(&key_a).unwrap().load(), Some(ra), "card A must read back its own file");
        assert_eq!(persist::store(&key_b).unwrap().load(), Some(rb), "card B must read back its own file, not A's (a clobber)");

        persist::set_dir_override(None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
