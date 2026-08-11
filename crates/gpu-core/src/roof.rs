// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device's own roofline — measured, never assumed.
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
//! * **compute** — [`kernels::ROOF_FMA`], a dependency-free FMA chain held in
//!   registers with no memory traffic in the loop. It measures the *silicon*,
//!   deliberately not "the best GEMM we have written": grading brain's kernels
//!   against brain's best kernel would hide exactly the gap worth closing.
//! * **bandwidth** — `axpy` (`out[i] += s * in[i]`) over buffers far larger than
//!   any cache: the classic STREAM-triad shape, 12 bytes moved per element
//!   (two reads and a write), already in the kernel tree.
//!
//! Both probes self-calibrate their trip count until the timed region is long
//! enough to dominate launch and drain, and both are `poll_wait`-bracketed —
//! a bare-submit loop once reported 377 GB/s on a ~346 GB/s card by timing
//! the host.
//!
//! Results are cached per adapter, with the same key discipline
//! [`crate::tune`] uses (adapter slug + a fingerprint of the probe sources), so
//! a probe-kernel edit invalidates old numbers by construction instead of
//! silently reusing them.

use crate::Gpu;
use std::sync::Mutex;
use std::time::Instant;

/// A device's measured roofline. Both halves are required — a memory-bound
/// kernel reported as a FLOP rate is meaningless, and vice versa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roofs {
    /// Peak fp32 arithmetic rate, GFLOP/s.
    pub gflops: f32,
    /// Peak DRAM bandwidth, GB/s — a working set far larger than any cache.
    pub gbs: f32,
    /// Peak bandwidth for a CACHE-RESIDENT working set, GB/s.
    ///
    /// The roofline is HIERARCHICAL: a kernel whose data fits in L2 is bounded
    /// by L2 bandwidth, not by DRAM, and can legitimately exceed the DRAM roof
    /// several times over. Grading such a kernel against `gbs` produces a
    /// number above 100% that looks like a broken measurement and is not one —
    /// which is exactly what `paged_decode_scores_batched` did at 292%.
    ///
    /// Measured with the same triad over a working set small enough to stay
    /// resident. On a device whose cache is smaller than the probe this simply
    /// converges to `gbs`, which is a conservative and self-consistent failure.
    pub cache_gbs: f32,
    /// Peak packed-int8 (`dot4I8Packed`) rate, GOP/s. `None` where the device
    /// has no int8 dot path.
    ///
    /// Without it an int8 kernel is graded against the *fp32* roof, which
    /// flatters it by whatever factor the device's DP4A path exceeds fp32 —
    /// `matmul_i8_dyn` read "36.1% of roof" that way, and a kernel that looks
    /// like a third of the machine may be a tenth of it.
    pub int8_gops: Option<f32>,
}

impl Roofs {
    /// FLOP per byte at the ridge point — below it no kernel can be
    /// compute-bound however it is tiled, above it bandwidth cannot be the
    /// limit.
    pub fn ridge(&self) -> f32 {
        if self.gbs > 0.0 { self.gflops / self.gbs } else { 0.0 }
    }

    /// The memory roof that applies to a kernel achieving `gbs` of *logical*
    /// traffic — DRAM if it is under that roof, the cache roof if it is above
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
                // Graded against whichever memory roof actually applies — DRAM,
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
    /// The band this workstream holds each class to. A well-tuned fp32 GEMM
    /// on old silicon lands 60-80% of peak, so 99% is not the target for
    /// either class and stating one per class is what makes the goal
    /// falsifiable.
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

/// The probe kernel set — the two sources whose fingerprint keys the cache.
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
/// lanes, so the probe never depends on knowing the topology — an
/// under-subscribed device would report its *latency*, not its throughput.
const FMA_THREADS: u32 = 1 << 20;
/// Eight independent accumulators, one FMA each = 16 FLOP per thread-iteration.
const FMA_FLOPS_PER_ITER: u64 = 16;

/// Elements per bandwidth-probe buffer. Two buffers of 64 Mi f32 = 512 MiB
/// resident, and 768 MiB of traffic per pass — far past any LLC, so the number
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
/// On expiry the loop returns `None` — "roofline unmeasured", the same
/// contract `ensure`'s doc already promises callers (render `-`, never a
/// guess) — rather than blocking indefinitely. Every rung of a healthy probe
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

static CACHE: Mutex<Option<Roofs>> = Mutex::new(None);

/// Serialises the actual measurement (`ensure`'s cache-miss path): two racing
/// `ensure` calls used to BOTH probe, each measuring a device contended by the
/// other, and the loser persisted a too-low roof to disk — permanently
/// poisoning the denominator every later "%-of-roof" claim divides by (the
/// crate's own `tests/roofline.rs` explains why that race is not benign and
/// guarded itself with a mutex production lacked). Never held across the
/// `CACHE` lock (see `ensure`'s reentrancy note).
static MEASURE: Mutex<()> = Mutex::new(());

/// One failed measurement is remembered for the process lifetime: the probe
/// allocates 512 MiB and can take the full `roof_budget()` before concluding
/// "unprobeable", and re-running it on every later `ensure` call repays that
/// cost for the same answer. (Per-DEVICE keying of this flag, the cache and
/// the persist file is real remaining work — it needs a canonical device
/// identity plumbed through `backend_api::Backend`, which no backend exposes
/// yet; until then all of this module is process-wide, first-device-wins.)
static MEASURE_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The roofs for this process's device, measuring them once if needed.
///
/// Returns `None` when the device cannot be probed — callers must then print
/// `-`, never a guess. Set `BRAIN_NO_ROOF=1` to skip probing entirely (useful
/// when a run must not spend the ~0.2 s, or to reproduce pre-roofline output).
///
/// Defaults to skipped (as if `BRAIN_NO_ROOF=1`) on the CPU backend, where the
/// probe's calibration loop has a known-bad interaction with `backend-cpu`'s
/// rayon dispatch — a per-call-site opt-out is what every new caller was
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
    // int8 probe on `numeric.int8_dot`), and `Gpu::caps` overlays `known()`,
    // which locks this SAME `CACHE` — a `std::sync::Mutex` is not reentrant,
    // so holding a guard here across that call self-deadlocks the calling
    // thread forever on any cold-cache (first-ever, or on-disk-store-miss)
    // call. Confirmed on real hardware: `caps_expose_the_roofs_only_after_
    // something_measured_them` hung indefinitely (gdb: thread parked in
    // `Mutex::lock` on `roof::CACHE`, called from `roof::known` <-
    // `Gpu::caps` <- `roof::measure` <- this function, one level up, on the
    // SAME thread) — not a GPU/driver issue at all, despite presenting
    // exactly like the driver hangs this module's `ensure`/wait-bound work
    // was written to guard against. A benign race remains (two threads both
    // missing the cache both call `measure`; last write wins) — acceptable,
    // matching this function's own existing double-checked-init shape.
    if let Some(r) = *CACHE.lock().unwrap_or_else(|e| e.into_inner()) {
        return Some(r);
    }
    if MEASURE_FAILED.load(std::sync::atomic::Ordering::Relaxed) {
        return None; // already concluded unprobeable this process — see MEASURE_FAILED
    }
    // Serialise the miss path: only one thread measures; the others wait and
    // re-check the cache — measuring a device the winner is saturating would
    // record (and persist) a contended, too-low roof. This guard is NOT held
    // while taking the CACHE lock's guard beyond a single statement, so the
    // known()/caps() reentrancy hazard above cannot involve it.
    let _measuring = MEASURE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = *CACHE.lock().unwrap_or_else(|e| e.into_inner()) {
        return Some(r); // the winner filled it while this thread waited
    }
    let store = persist::store();
    if let Some(r) = store.as_ref().and_then(|s| s.load()) {
        *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        return Some(r);
    }
    let Some(r) = measure(gpu) else {
        MEASURE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    if let Some(s) = store.as_ref() {
        s.save(r);
    }
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    Some(r)
}

/// Redirect where measured roofs persist (`None` restores the default
/// `~/.cache/brain` / `BRAIN_PIPELINE_CACHE_DIR` resolution). A TEST seam:
/// tests that exercise `ensure()` point this at a temp dir so they never
/// mutate the developer's real cache — see `tests/roofline.rs`.
pub fn set_cache_dir(dir: Option<std::path::PathBuf>) {
    persist::set_dir_override(dir);
}

/// Whatever has already been measured, without measuring. This is what
/// [`Gpu::caps`] overlays onto [`backend_api::DeviceCaps`] — reading caps must
/// never have the side effect of running kernels.
pub fn known() -> Option<Roofs> {
    *CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run both probes now, ignoring and not updating the cache. Exposed so a test
/// can assert the measurement itself rather than a memoised value.
pub fn measure(gpu: &Gpu) -> Option<Roofs> {
    let g = gpu.new_like(PROBE_KERNELS);
    let gflops = measure_compute(&g)?;
    let gbs = measure_bandwidth(&g, BW_ELEMS)?;
    // The same triad over a working set chosen to stay in cache. Never report a
    // cache roof BELOW the DRAM roof: on a device whose cache is smaller than
    // the probe (or which has none worth the name) the two converge, and the
    // max keeps the hierarchy monotonic so `memory_roof` cannot invert.
    let cache_gbs = measure_bandwidth(&g, CACHE_ELEMS)?.max(gbs);
    // `None` where the device has no int8 dot path — never a guess, and never
    // fp32's number standing in for it.
    let int8_gops = gpu.caps().numeric.int8_dot.then(|| measure_int8(&g)).flatten();
    Some(Roofs { gflops, gbs, cache_gbs, int8_gops })
}

/// A single `submit`+`poll_wait` in this probe must complete within this
/// budget or the whole measurement is abandoned (`best_of` returns `None`).
/// Generous relative to a healthy measurement (clean rounds in this probe
/// take well under a second even on a slow iGPU) but far below the
/// multi-minute stalls this exists to bound — see `poll_wait_timeout`'s doc.
/// Only bounds anything on backends that actually implement `poll_wait_timeout`
/// (currently `backend-wgpu`); on others this is a no-op budget the default
/// trait method ignores.
const PER_DISPATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Time one submit of `steps`, best of `reps`, `poll_wait`-bracketed. `None`
/// if any single submit (including the warmup) does not complete within
/// `PER_DISPATCH_TIMEOUT` — the caller must not keep using `gpu` for further
/// timed measurements after that (see `poll_wait_timeout`'s doc: a timeout
/// leaves completion state unknown on backends where reuse would not be
/// safe), so every caller here treats `None` as "abandon this probe."
///
/// Stops repping once `deadline` passes (keeping whatever best it already
/// timed): the calibration loops used to check their budget only BETWEEN
/// `best_of` calls, so one call issued just under the wire could legally run
/// `1 + reps` dispatches, each up to `PER_DISPATCH_TIMEOUT`, past the
/// deadline — an effectively unbounded overshoot the roofline wall-clock test
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

fn measure_compute(gpu: &Gpu) -> Option<f32> {
    let inp = gpu.storage(FMA_THREADS as u64);
    let out = gpu.storage(FMA_THREADS as u64);
    gpu.write_f32(&inp, &vec![1.0f32; FMA_THREADS as usize]);

    // c = 0.5, d = 0.5 has fixed point 1.0: the chain converges immediately and
    // then stays exactly 1.0 — no overflow, no denormals (slow on some
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

fn measure_bandwidth(gpu: &Gpu, elems: u64) -> Option<f32> {
    let out = gpu.storage(elems);
    let inp = gpu.storage(elems);
    // Leave both buffers at their zero-initialised contents: axpy's rate depends
    // on bytes moved, not on values, and writing 512 MiB from the host would
    // cost far more than the measurement.

    // One pass over 512 MiB is already ~2 ms on a fast card; repeat the same
    // dispatch until the timed region clears the launch floor.
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
    use super::Roofs;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Process-local override of the persist directory. Set via
    /// [`super::set_cache_dir`] — primarily a TEST seam: without it, any test
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

    pub fn store() -> Option<RoofStore> {
        let (desc, _) = crate::adapter_info()?;
        let dir = cache_dir()?;
        let hash = crate::tune::source_fingerprint(
            &super::PROBE_KERNELS.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
        );
        let slug: String =
            desc.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        Some(RoofStore { path: dir.join(format!("roof-{slug}-{hash:016x}.txt")) })
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
        /// trusted — the same rule the tune store applies.
        pub fn load(&self) -> Option<Roofs> {
            let text = std::fs::read_to_string(&self.path).ok()?;
            let (mut gflops, mut gbs, mut cache, mut int8) = (None, None, None, None);
            for line in text.lines() {
                let (k, v) = line.split_once('=')?;
                let v: f32 = v.trim().parse().ok()?;
                match k.trim() {
                    "gflops" => gflops = Some(v),
                    "gbs" => gbs = Some(v),
                    "cache_gbs" => cache = Some(v),
                    "int8_gops" => int8 = Some(v),
                    _ => {}
                }
            }
            // A record written before `cache_gbs` existed parses to `None` here
            // and is simply re-measured — a stale cache is ignored, never
            // patched up with a default.
            // `int8_gops` is legitimately absent on a device without DP4A, so a
            // record missing it is valid — unlike the other three.
            let r = Roofs { gflops: gflops?, gbs: gbs?, cache_gbs: cache?, int8_gops: int8 };
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
                "gflops={}\ngbs={}\ncache_gbs={}\n{}",
                r.gflops,
                r.gbs,
                r.cache_gbs,
                r.int8_gops.map(|v| format!("int8_gops={v}\n")).unwrap_or_default(),
            );
            if std::fs::write(&tmp, body).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_and_classification_split_at_the_ridge() {
        let r = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0) };
        // A P40's ridge is ~34 FLOP/byte — anything streaming is left of it.
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
        let r = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0) };
        // A streaming kernel at exactly the bandwidth roof reads 100%, even
        // though its FLOP rate is negligible — the whole point of classifying.
        let bytes = 346_000_000_000u64;
        let u = r.utilisation(bytes / 6, bytes, 1.0).unwrap();
        assert!((u - 100.0).abs() < 0.5, "{u}");
        // Zero-length regions produce nothing rather than infinity.
        assert!(r.utilisation(1, 1, 0.0).is_none());
    }

    #[test]
    fn target_bands_are_stated_per_class() {
        assert!(Bound::Compute.target_pct() > Bound::Compute.defect_pct());
        assert!(Bound::Memory.target_pct() > Bound::Memory.defect_pct());
    }
}
