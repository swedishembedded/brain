// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel-kind profiling of a recorded pass — the one implementation.
//!
//! The prescribed shape: group contiguous runs of one kernel *in submit
//! order*, time each group, and publish the table alongside the whole-pass
//! number. Four benches had grown their own
//! copy of it (`vqgan_bench`, `unet_bench`, `flux2_bench`, `zimage_bench`) and a
//! fifth was about to, which is precisely the duplication AGENTS.md's "one
//! implementation" rule exists to stop — the copies had already drifted on the
//! two things that matter most:
//!
//! * **the roof they divide by.** Each carried its own `PEAK_TFLOPS = 11.76`
//!   literal, so every utilisation column was a statement about one card. Here
//!   the denominator is the *measured* [`crate::roof::Roofs`] of whatever ran.
//! * **coverage honesty.** `vqgan_bench`'s `WHOLE PASS` row summed FLOPs across
//!   all rows regardless of whether `cost` had a formula for them, so a partly
//!   covered pass silently *under*-reported its rate instead of declaring
//!   itself incomplete. Here an uncovered pass says so.
//!
//! ## RESOLVED: the time source
//!
//! [`profile`] now uses DEVICE time wherever the backend can give it:
//! `Gpu::set_kernel_timing(true)` + one timed submit of the WHOLE pass (the same
//! single compute pass production runs — no group slicing, no per-group drain,
//! no launch+fence floor folded into a kernel's number) yields per-kernel device
//! totals directly via `kernel_times()`. `PassProfile::device_timed` records
//! which path a given profile actually took. Validated: summed kernel device
//! time and the whole-pass number agree to within a fraction of a percent,
//! where the old host-bracketed-slice method was off by more than an order of
//! magnitude on small kernels and inverted the ranking outright.
//!
//! Only backends without a device-timestamp path fall back to the OLD
//! host-wall-clock-per-group method below, which still carries the launch+fence
//! floor this section used to warn about — check `device_timed` before trusting
//! a per-kernel share as more than a rough guide on such a run.
//!
//! Two numbers come out of every profile and they are not interchangeable.
//! [`PassProfile::total_secs`] is ONE submit of the whole list — the number that
//! decides whether a change worked. [`PassProfile::summed_secs`] is the sum of
//! the per-group timings, each of which pays its own queue drain; on a VQGAN
//! backward that inflates the total by a large fraction. **Rank with the
//! table, decide with the pass.**

use crate::roof::{Bound, Roofs};

use crate::{Gpu, Step};
use std::collections::HashMap;
use std::time::Instant;

/// Above this fraction of a roof, the number is impossible and is reported as an
/// error (`!`) rather than as a utilisation. A few percent of headroom keeps
/// measurement noise on a kernel genuinely at the roof from tripping it.
pub const IMPOSSIBLE_PCT: f32 = 105.0;

/// One kernel kind's contribution to a pass.
#[derive(Clone, Debug)]
pub struct KernelRow {
    pub name: String,
    /// Summed time of this kind's groups — an UPPER BOUND (each group drains).
    pub secs: f64,
    pub calls: usize,
    pub flops: u64,
    /// Integer ops (the DP4A int8 path), kept SEPARATE from `flops`: folding
    /// them together loses which roof the row must be graded against, and an
    /// int8 kernel graded against fp32 reads several times better than it is.
    pub int_ops: u64,
    pub bytes: u64,
    /// False when any step of this kind has no `cost` formula, in which case
    /// the rates are not reported at all rather than reported as zero.
    pub covered: bool,
}

impl KernelRow {
    /// `None` when the row has no measurable time. A device timestamp can round
    /// to zero for a kernel that runs below the timer's resolution, and dividing
    /// by it printed `inf GFLOP/s` — a number that reads as "infinitely fast"
    /// where the truth is "too short to measure".
    pub fn gflops(&self) -> Option<f64> {
        let work = self.flops.max(self.int_ops);
        (self.covered && work > 0 && self.secs > 0.0).then(|| work as f64 / self.secs / 1e9)
    }
    pub fn gbs(&self) -> Option<f64> {
        (self.covered && self.bytes > 0 && self.secs > 0.0)
            .then(|| self.bytes as f64 / self.secs / 1e9)
    }
    /// Which roof this kernel is graded against, given the device's ridge.
    pub fn bound(&self, roofs: Roofs) -> Option<Bound> {
        self.covered.then(|| roofs.classify(self.flops.max(self.int_ops), self.bytes))
    }

    /// True when this row's compute is graded against a roof that was NOT
    /// measured for its kind — int8 work falling back to the fp32 roof, which
    /// overstates it. The table marks those rather than printing a clean number.
    pub fn roof_is_substituted(&self, roofs: Roofs) -> bool {
        self.bound(roofs) == Some(Bound::Compute)
            && !roofs.compute_roof(self.flops, self.int_ops).1
    }
    /// True when this row's logical traffic exceeds the DRAM roof but not the
    /// cache roof — i.e. it is being served from cache, which is a real place
    /// for data to come from and not a broken measurement.
    pub fn cache_resident(&self, roofs: Roofs) -> bool {
        self.gbs().and_then(|g| roofs.memory_roof(g)).map(|(_, c)| c).unwrap_or(false)
    }

    /// Percent of *its own class's* roof. This is the number this
    /// workstream's per-class target bands are stated against — reporting a
    /// memory-bound kernel as a fraction of a FLOP peak is meaningless.
    pub fn utilisation(&self, roofs: Roofs) -> Option<f32> {
        (self.covered && self.secs > 0.0)
            .then(|| roofs.utilisation_of(self.flops, self.int_ops, self.bytes, self.secs))?
    }
}

/// A profiled pass: the whole-pass time, and the per-kind breakdown.
#[derive(Clone, Debug)]
pub struct PassProfile {
    pub label: String,
    /// ONE submit of the entire step list — the number a change is judged by.
    pub total_secs: f64,
    pub dispatches: usize,
    /// Rows, slowest first.
    pub rows: Vec<KernelRow>,
    /// Sum of the per-group timings; exceeds `total_secs` by the drain cost.
    pub summed_secs: f64,
    /// How many contiguous groups the pass has (= how many extra drains the
    /// group timings paid for).
    pub groups: usize,
    /// True only when every step in the pass had a `cost` formula.
    pub fully_covered: bool,
    /// True when per-kernel times are DEVICE times from timestamp queries
    /// written inside the production single compute pass. False means they are
    /// host-bracketed group times, which inflate small kernels by more than an
    /// order of magnitude and
    /// must not be used to attribute time between kernels (`lessons.md` #31).
    pub device_timed: bool,
}

impl PassProfile {
    /// Whole-pass FLOP rate. `None` when the pass is not fully covered — a
    /// partial numerator over the full denominator is a fiction, and the
    /// honest answer to "what rate did this pass achieve" is "unknown".
    pub fn gflops(&self) -> Option<f64> {
        let flops: u64 = self.rows.iter().map(|r| r.flops.max(r.int_ops)).sum();
        (self.fully_covered && flops > 0).then(|| flops as f64 / self.total_secs / 1e9)
    }

    /// Percent inflation the per-group timings carry over the real pass.
    pub fn drain_overhead_pct(&self) -> f64 {
        if self.total_secs > 0.0 {
            100.0 * (self.summed_secs - self.total_secs) / self.total_secs
        } else {
            0.0
        }
    }

    /// The §F.1 table. `roofs` is the *measured* roofline of the device that
    /// ran; where it is `None` the utilisation columns print `-` rather than
    /// dividing by an assumed peak.
    pub fn print(&self, roofs: Option<Roofs>) {
        self.print_top(roofs, usize::MAX);
    }

    /// [`Self::print`] showing at most `top` rows.
    pub fn print_top(&self, roofs: Option<Roofs>, top: usize) {
        println!(
            "\n=== {}: {} dispatches, {:.2} ms ===",
            self.label,
            self.dispatches,
            self.total_secs * 1e3
        );
        println!(
            "{:<26} {:>9} {:>6} {:>6} {:>9} {:>11} {:>9} {:>7} {:>7}",
            "kernel", "ms", "n", "%", "ms/call", "GFLOP/s", "GB/s", "bound", "%roof"
        );
        println!("{}", "-".repeat(100));
        let dash = "-".to_string();
        for r in self.rows.iter().take(top) {
            let gfs = r.gflops().map(|v| format!("{v:.1}")).unwrap_or_else(|| dash.clone());
            let gbs = r.gbs().map(|v| format!("{v:.1}")).unwrap_or_else(|| dash.clone());
            let (bound, util) = match roofs {
                Some(rf) => (
                    r.bound(rf).map(|b| b.as_str().to_string()).unwrap_or_else(|| dash.clone()),
                    // A rate ABOVE the device's roof is not an achievement, it
                    // is a defect in the measurement — either the cost formula
                    // overstates the work (a data-dependent kernel costed by an
                    // upper bound, a synthetic harness whose buffers alias so
                    // the "streaming" byte estimate is fiction) or the timed
                    // region was the host (a bare-submit loop once reported a
                    // bandwidth above what the card can physically deliver).
                    // Printing it as a percentage
                    // launders a broken number into a flattering one.
                    r.utilisation(rf)
                        .map(|u| {
                            if u > IMPOSSIBLE_PCT {
                                format!("!{u:.0}%")
                            } else if r.roof_is_substituted(rf) {
                                // int8 work with no measured int8 roof: the
                                // number is against fp32 and OVERSTATES it.
                                format!("~{u:.1}%")
                            } else if r.cache_resident(rf) {
                                // Above the DRAM roof but under the cache roof:
                                // the data really is coming from cache, and the
                                // percentage is against THAT roof.
                                format!("{u:.1}%c")
                            } else {
                                format!("{u:.1}%")
                            }
                        })
                        .unwrap_or_else(|| dash.clone()),
                ),
                None => (dash.clone(), dash.clone()),
            };
            println!(
                "{:<26} {:>9.2} {:>6} {:>5.1}% {:>9.3} {:>11} {:>9} {:>7} {:>7}",
                r.name,
                r.secs * 1e3,
                r.calls,
                100.0 * r.secs / self.summed_secs,
                r.secs * 1e3 / r.calls as f64,
                gfs,
                gbs,
                bound,
                util,
            );
        }
        match self.gflops() {
            Some(g) => {
                let pk = roofs
                    .map(|rf| format!("{:.1}%", 100.0 * g as f32 / rf.gflops))
                    .unwrap_or_else(|| dash.clone());
                println!(
                    "{:<26} {:>9.2} {:>6} {:>6} {:>9} {:>11.1} {:>9} {:>7} {:>7}",
                    "WHOLE PASS",
                    self.total_secs * 1e3,
                    self.dispatches,
                    "",
                    "",
                    g,
                    "",
                    "",
                    pk
                );
            }
            None => {
                // Name them, don't just count them: an uncovered kind is a
                // work item (`gpu_core::cost` needs a formula), and a count
                // alone leaves the reader unable to act — especially since the
                // uncovered kinds are usually cheap enough to fall outside the
                // printed top rows, which is exactly where they hide.
                let missing: Vec<&str> =
                    self.rows.iter().filter(|r| !r.covered).map(|r| r.name.as_str()).collect();
                println!(
                    "{:<26} {:>9.2} {:>6}   (rate unavailable — no cost formula for: {})",
                    "WHOLE PASS",
                    self.total_secs * 1e3,
                    self.dispatches,
                    missing.join(", "),
                );
            }
        }
        println!("{}", "-".repeat(100));
        if self.device_timed {
            println!(
                "sum of kernel device time {:.2} ms vs whole pass {:.2} ms — the gap is launch, \
                 sync and gaps between dispatches, not kernel work.",
                self.summed_secs * 1e3,
                self.total_secs * 1e3,
            );
        } else {
            println!(
                "sum of groups {:.2} ms vs whole pass {:.2} ms — {:.0}% drain inflation over {} groups.",
                self.summed_secs * 1e3,
                self.total_secs * 1e3,
                self.drain_overhead_pct(),
                self.groups,
            );
        }
        println!("Rank with the table; decide with the whole-pass number.");
        if self.device_timed {
            println!(
                "Per-kernel times are DEVICE times (timestamp queries inside the production \
                 single compute pass), so the shares above are attributable."
            );
        } else {
            println!(
                "NOTE per-kernel times are HOST-bracketed (launch+execute+fence) — this device \
                 cannot write timestamps inside a pass — and inflate small kernels up to ~30x. \
                 Do not attribute time between kernels from them (lessons #31)."
            );
        }
        if let Some(rf) = roofs {
            let bogus: Vec<&str> = self
                .rows
                .iter()
                .filter(|r| r.utilisation(rf).is_some_and(|u| u > IMPOSSIBLE_PCT))
                .map(|r| r.name.as_str())
                .collect();
            if !bogus.is_empty() {
                println!(
                    "!! {} row(s) report above even the CACHE roof — the accounting or the \
                     timing is wrong, not the kernel: {}",
                    bogus.len(),
                    bogus.join(", "),
                );
            }
        }
    }

    /// Every row over `pct` of the pass that sits below its class's defect
    /// line — i.e. the rows this workstream calls defects rather than merely
    /// untuned. Returns `(row, bound, achieved %)`.
    pub fn defects(&self, roofs: Roofs, pct: f64) -> Vec<(&KernelRow, Bound, f32)> {
        self.rows
            .iter()
            .filter(|r| 100.0 * r.secs / self.summed_secs >= pct)
            .filter_map(|r| {
                let b = r.bound(roofs)?;
                let u = r.utilisation(roofs)?;
                (u < b.defect_pct()).then_some((r, b, u))
            })
            .collect()
    }
}

/// Whether `BRAIN_PROFILE` asks for profiling output - see
/// [`backend_api::profile_enabled`], which every backend shares.
pub use backend_api::profile_enabled as enabled;

/// Wall time of a host-side STAGE to stderr under `BRAIN_PROFILE` - the coarse
/// timeline above the per-kernel tables.
///
/// Device timestamps ([`crate::Gpu::kernel_times`]) attribute time BETWEEN
/// kernels and nothing else; a serving path whose cost is checkpoint I/O, host
/// dequantization or a cross-device residual hop spends that time where no
/// kernel is running, so a kernel table alone reports a tiny total and silently
/// omits the dominant term. These two views are complementary and neither
/// substitutes for the other.
pub fn stage_time(name: &str, since: Instant) {
    if enabled() {
        eprintln!("stage {name}: {:.1} ms", since.elapsed().as_secs_f64() * 1e3);
    }
}

/// Time one submit of `steps`, best of `reps`.
///
/// Every timed region is `poll_wait`-bracketed. This is not defensive style: on
/// the wgpu backend `submit` with an empty clear list only appends to the
/// pending list, so an unbracketed loop times host-side recording and reports a
/// rate above the physical roof - it once produced a bandwidth figure larger
/// than the card's datasheet peak.
pub fn best_of(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    gpu.submit(&[], steps);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// Contiguous runs of one kernel kind, in submit order: `(kind, start, len)`.
///
/// Grouping by *contiguous run* rather than by kind keeps graph order, so the
/// sum of the parts is comparable with the whole — a scatter of the same kind
/// across the graph would otherwise be timed as one artificial block.
pub fn groups(steps: &[Step]) -> Vec<(usize, usize, usize)> {
    let mut g: Vec<(usize, usize, usize)> = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        let k = s.meta().map(|m| m.kernel).unwrap_or(usize::MAX);
        match g.last_mut() {
            Some((gk, _, len)) if *gk == k => *len += 1,
            _ => g.push((k, i, 1)),
        }
    }
    g
}

/// Profile a recorded pass: one whole-pass timing plus a per-kind breakdown.
///
/// `steps` must have been recorded through a handle of `gpu`'s kernel set —
/// kernel indices resolve through this handle's pipeline names.
pub fn profile(gpu: &Gpu, label: &str, steps: &[Step], reps: usize) -> PassProfile {
    // Whole-pass time is always the un-timed production flush: it is what a
    // change is judged by, and it must not include the timestamp resolve.
    let total = best_of(gpu, steps, reps);
    let gs = groups(steps);

    // DEVICE time where the backend can give it (`lessons.md` #31). One timed
    // submit of the WHOLE pass — same single compute pass as production — yields
    // per-kernel totals directly, so there is no group slicing, no drain per
    // group, and no launch+fence floor folded into a kernel's number.
    let device_times: Option<HashMap<String, (f64, u64)>> = gpu.set_kernel_timing(true).then(|| {
        gpu.reset_kernel_times();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        let t = gpu
            .kernel_times()
            .unwrap_or_default()
            .into_iter()
            .map(|(n, ms, calls)| (n, (ms / 1e3, calls)))
            .collect();
        gpu.set_kernel_timing(false);
        t
    });
    // An EMPTY timing map means the backend claimed timing but recorded
    // nothing — treating it as device-timed would zero every row and render
    // each share as NaN%. Fall through to the host-timed path instead.
    if let Some(d) = device_times.as_ref().filter(|d| !d.is_empty()) {
        let mut per: HashMap<usize, (f64, usize, u64, u64, u64, bool)> = HashMap::new();
        for (k, start, len) in &gs {
            accumulate(gpu, &mut per, *k, &steps[*start..*start + *len], 0.0);
        }
        let mut rows = finish_rows(gpu, per);
        // Overwrite the (unused) host times with the measured device ones.
        // Rows carry the CALLER's kernel name, but the backend's timing map is
        // keyed by the PHYSICAL pipeline that ran — for an upgrade-redirected
        // kernel (e.g. max_abs_row -> max_abs_rows) those differ, and the
        // caller-name lookup silently reported zero time for the redirected kernel
        // while inflating every other row's share. Translate through the
        // upgrade map first. (If two caller slots redirect to one physical
        // kernel their combined time lands on each row — a visible
        // over-attribution rather than an invisible zero.)
        for r in rows.iter_mut() {
            let physical = match gpu.kernel_index(&r.name) {
                Some(k) => gpu.physical_kernel_names(k),
                None => vec![r.name.as_str()],
            };
            let hits: Vec<_> = physical.iter().filter_map(|n| d.get(*n)).collect();
            if hits.is_empty() {
                r.secs = 0.0;
            } else {
                r.secs = hits.iter().map(|(s, _)| *s).sum();
                r.calls = hits.iter().map(|(_, c)| *c as usize).sum();
            }
        }
        rows.sort_by(|a, b| b.secs.partial_cmp(&a.secs).unwrap());
        let summed: f64 = rows.iter().map(|r| r.secs).sum();
        let fully_covered = rows.iter().all(|r| r.covered);
        // A timing map that matched NO row would still divide by zero in every
        // share — treat that, too, as "no device timing" and use host timing.
        if summed > 0.0 {
            return PassProfile {
                label: label.to_string(),
                total_secs: total,
                dispatches: steps.len(),
                rows,
                summed_secs: summed,
                groups: gs.len(),
                fully_covered,
                device_timed: true,
            };
        }
    }

    let mut per: HashMap<usize, (f64, usize, u64, u64, u64, bool)> = HashMap::new();
    for (k, start, len) in &gs {
        let t = best_of(gpu, &steps[*start..*start + *len], reps);
        accumulate(gpu, &mut per, *k, &steps[*start..*start + *len], t);
    }

    let mut rows = finish_rows(gpu, per);
    rows.sort_by(|a, b| b.secs.partial_cmp(&a.secs).unwrap());

    let summed = rows.iter().map(|r| r.secs).sum();
    let fully_covered = rows.iter().all(|r| r.covered);
    PassProfile {
        label: label.to_string(),
        total_secs: total,
        dispatches: steps.len(),
        rows,
        summed_secs: summed,
        groups: gs.len(),
        fully_covered,
        device_timed: false,
    }
}

/// Profile a pass that SUBMITS ITSELF - the case [`profile`] cannot take.
///
/// [`profile`] needs one flat `&[Step]` it can re-submit. Several models never
/// hand one out: `minimaxmusic3::dit::forward_resident` submits per sub-layer
/// and reads a buffer back to the host in the middle of every block,
/// `minimaxmusic3::vocoder::forward` records its tape privately, and
/// `sam1::SamEncoder::forward` submits per stage. Both halves of a
/// [`PassProfile`] still exist for those passes - they are just not in a step
/// list, so they come off the `Gpu` handle instead:
///
/// * per-kernel DEVICE time from [`Gpu::kernel_times`] - the same accumulator
///   [`profile`]'s device-timed path reads, but summed over EVERY submit the
///   closure makes rather than over one;
/// * per-kernel FLOP/byte volume from [`Gpu::ops_counters`], which folds
///   [`crate::cost::kernel_cost`] over every submitted step online.
///
/// Same table, same two modules, no third copy of either.
///
/// `run` is called `reps + 2` times: **once un-timed to warm up** (a warm-up
/// must never enter the statistics - the first call pays shader specialisation,
/// allocator growth and first-touch page faults no later call repeats), `reps`
/// times for the best-of-N whole-pass wall clock, and once more with both
/// ledgers armed. Best-of-N, not mean: the minimum is the least contaminated
/// sample.
///
/// [`PassProfile::total_secs`] here is WALL CLOCK around the whole closure, so
/// unlike [`profile`]'s it also contains any host math and any readback the
/// pass does between submits. That is deliberate - it is what the caller
/// actually waits for - but it means `total_secs - summed_secs` is *not* just
/// launch and sync for a pass with host work in it. Read the two numbers as
/// "what it costs" and "what the device did".
///
/// `device_timed` is false on a backend with no timestamp path (the CPU JIT):
/// the rows still carry dispatch counts and FLOP/byte volume, but every `secs`
/// is 0.0 and the shares are meaningless. Check it before printing.
pub fn profile_live(gpu: &Gpu, label: &str, reps: usize, mut run: impl FnMut()) -> PassProfile {
    // Warm-up - never counted (`crates/perf` states the rule and why).
    run();
    gpu.poll_wait();

    let mut total = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t0 = Instant::now();
        run();
        gpu.poll_wait();
        total = total.min(t0.elapsed().as_secs_f64());
    }

    // One more pass with both ledgers armed. Arming AFTER the timed reps keeps
    // the online tally's per-dispatch lock out of the number a change is
    // judged by.
    let timed = gpu.set_kernel_timing(true);
    if timed {
        gpu.reset_kernel_times();
    }
    gpu.reset_ops_counters();
    run();
    gpu.poll_wait();
    let report = gpu.ops_counters();
    let times: HashMap<String, f64> = if timed {
        gpu.kernel_times()
            .unwrap_or_default()
            .into_iter()
            .map(|(n, ms, _calls)| (n, ms / 1e3))
            .collect()
    } else {
        HashMap::new()
    };
    gpu.set_kernel_timing(false);

    // Rows carry the CALLER's kernel name; the backend's timing map is keyed by
    // the PHYSICAL pipeline that ran, which differs for an upgrade-redirected
    // kernel. Translate, exactly as `profile` does. (Two caller slots redirected
    // onto one physical kernel would each show the combined time - a visible
    // over-attribution rather than an invisible zero.)
    let secs_of = |name: &str| -> f64 {
        let physical = match gpu.kernel_index(name) {
            Some(k) => gpu.physical_kernel_names(k),
            None => vec![name],
        };
        physical.iter().filter_map(|n| times.get(*n)).sum()
    };

    let covered = report.by_kernel.iter().map(|(name, kc)| KernelRow {
        name: name.clone(),
        secs: secs_of(name),
        calls: kc.calls as usize,
        flops: kc.cost.flops,
        int_ops: kc.cost.int_ops,
        bytes: kc.cost.bytes,
        covered: true,
    });
    // An uncovered kind keeps its time and its call count but reports NO rates -
    // unmeasured is null, never zero-pretending-complete.
    let uncovered = report.uncovered.iter().map(|(name, calls)| KernelRow {
        name: name.clone(),
        secs: secs_of(name),
        calls: *calls as usize,
        flops: 0,
        int_ops: 0,
        bytes: 0,
        covered: false,
    });
    let mut rows: Vec<KernelRow> = covered.chain(uncovered).collect();
    rows.sort_by(|a, b| b.secs.partial_cmp(&a.secs).unwrap());

    let summed: f64 = rows.iter().map(|r| r.secs).sum();
    let fully_covered = rows.iter().all(|r| r.covered);
    PassProfile {
        label: label.to_string(),
        total_secs: total,
        dispatches: report.steps as usize,
        rows,
        // Every dispatch went through ONE accumulated device-time ledger, so
        // there is no per-group drain to declare and no group count to give.
        groups: 0,
        summed_secs: summed,
        fully_covered,
        device_timed: timed && summed > 0.0,
    }
}

/// Fold one group's `cost` volume (and `t` seconds) into the per-kernel map.
fn accumulate(
    gpu: &Gpu,
    per: &mut HashMap<usize, (f64, usize, u64, u64, u64, bool)>,
    k: usize,
    slice: &[Step],
    t: f64,
) {
    let name = gpu.kernel_name(k).unwrap_or("?");
    let (mut fl, mut io, mut by, mut covered) = (0u64, 0u64, 0u64, true);
    for st in slice {
        let m = st.meta();
        let params = m.as_ref().and_then(|m| m.params.as_deref());
        let threads = m.as_ref().map(|m| m.threads).unwrap_or(0);
        match crate::cost::kernel_cost(name, params, threads) {
            Some(c) => {
                fl += c.flops;
                io += c.int_ops;
                by += c.bytes;
            }
            None => covered = false,
        }
    }
    let e = per.entry(k).or_insert((0.0, 0, 0, 0, 0, true));
    e.0 += t;
    e.1 += slice.len();
    e.2 += fl;
    e.3 += io;
    e.4 += by;
    e.5 &= covered;
}

fn finish_rows(gpu: &Gpu, per: HashMap<usize, (f64, usize, u64, u64, u64, bool)>) -> Vec<KernelRow> {
    per.into_iter()
        .map(|(k, (secs, calls, flops, int_ops, bytes, covered))| KernelRow {
            name: gpu.kernel_name(k).unwrap_or("?").to_string(),
            secs,
            calls,
            flops,
            int_ops,
            bytes,
            covered,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, secs: f64, flops: u64, bytes: u64, covered: bool) -> KernelRow {
        KernelRow { name: name.into(), secs, calls: 1, flops, int_ops: 0, bytes, covered }
    }

    fn pass(rows: Vec<KernelRow>, total: f64) -> PassProfile {
        let summed = rows.iter().map(|r| r.secs).sum();
        let fully_covered = rows.iter().all(|r| r.covered);
        PassProfile {
            label: "t".into(),
            total_secs: total,
            dispatches: rows.len(),
            rows,
            summed_secs: summed,
            groups: 1,
            fully_covered,
            device_timed: false,
        }
    }

    #[test]
    fn an_uncovered_pass_reports_no_rate_rather_than_a_low_one() {
        // This is the bug the extraction fixes: summing a covered kernel's
        // FLOPs over the whole pass time, while an uncovered kernel contributes
        // none, silently under-reports instead of admitting it cannot tell.
        let p = pass(vec![row("a", 1.0, 1_000_000_000, 0, true), row("b", 1.0, 0, 0, false)], 2.0);
        assert!(p.gflops().is_none(), "a partly covered pass must not report a rate");

        let q = pass(vec![row("a", 1.0, 2_000_000_000, 0, true)], 1.0);
        assert_eq!(q.gflops(), Some(2.0));
    }

    #[test]
    fn the_group_sum_is_reported_as_an_upper_bound_on_the_pass() {
        // A real VQGAN backward, as measured: the group times below sum to
        // markedly more than the whole-pass time they came from.
        let p = pass(vec![row("a", 0.65809, 0, 1, true)], 0.45607);
        assert!((p.drain_overhead_pct() - 44.3).abs() < 0.5, "{}", p.drain_overhead_pct());
    }

    #[test]
    fn defects_are_rows_under_their_own_classs_floor() {
        let roofs = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0) };
        // col2im-shaped: a long row moving ~3.3 GB, i.e. a small fraction of
        // the bandwidth roof.
        let col2im = row("col2im", 0.14275, 0, 3_312_000_000, true);
        // A GEMM fast in absolute terms but still under the compute floor.
        let gemm = row("matmul_dx_reg", 0.07322, 144_600_000_000, 1_000_000, true);
        let p = pass(vec![col2im, gemm], 0.456);

        let d = p.defects(roofs, 5.0);
        let names: Vec<&str> = d.iter().map(|(r, _, _)| r.name.as_str()).collect();
        assert!(names.contains(&"col2im"), "{names:?}");
        assert_eq!(d.iter().find(|(r, _, _)| r.name == "col2im").unwrap().1, Bound::Memory);
        assert_eq!(d.iter().find(|(r, _, _)| r.name == "matmul_dx_reg").unwrap().1, Bound::Compute);

        // A row under the share threshold is not reported however bad it is.
        assert!(p.defects(roofs, 99.0).len() <= 1);
    }
}
