// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel-kind profiling of a recorded pass — the one implementation.
//!
//! `docs/kernel-checklist.md` §F.1 prescribes a specific shape for this: group
//! contiguous runs of one kernel *in submit order*, time each group, and publish
//! the table alongside the whole-pass number. Four benches had grown their own
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
//! ## KNOWN DEFECT: the time source (`docs/lessons.md` #31)
//!
//! Group times here are HOST wall-clock around a `poll_wait`-bracketed submit of
//! that slice, so they measure **launch + execute + fence**. The floor of that is
//! roughly constant, which inflates small kernels in inverse proportion to their
//! size — measured against the backend's GPU timestamp queries on one
//! `qwen_bench serve` run: `matmul_reg3` 1.5x, `rmsnorm_rows` **19.7x**,
//! `paged_decode_scores_batched` **17x**, `paged_decode_apply_batched` **29x**.
//!
//! The resulting RANKING is wrong, not merely imprecise: by device time
//! `matmul_reg3` is 94.8% of that pass, where this table said 53.8% and promoted
//! two sub-2% kernels to "DEFECT" rows. **Do not attribute time between kernels
//! from this table for small kernels** — cross-check with `BRAIN_PROFILE=1`.
//! [`PassProfile::total_secs`] is unaffected, and remains the number a change is
//! judged by.
//!
//! The fix is timestamp queries inside the production single-pass flush, so
//! attribution and the whole-pass number come from the same execution;
//! `BRAIN_PROFILE` today only times dispatches in a one-pass-per-dispatch mode
//! whose absolute times are not production times either.
//!
//! Two numbers come out of every profile and they are not interchangeable.
//! [`PassProfile::total_secs`] is ONE submit of the whole list — the number that
//! decides whether a change worked. [`PassProfile::summed_secs`] is the sum of
//! the per-group timings, each of which pays its own queue drain; on a VQGAN
//! backward that inflates the total by ~44%. **Rank with the table, decide with
//! the pass** (`docs/lessons.md` #21).

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

    /// Percent of *its own class's* roof. This is the number the target bands
    /// in `.todo/saturate-the-gpu.md` are stated against — reporting a
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
    /// host-bracketed group times, which inflate small kernels up to ~30x and
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
                    // region was the host (`docs/kernel-checklist.md` §E.0,
                    // which exists because a bare-submit loop once reported
                    // 377 GB/s on a ~346 GB/s card). Printing it as a percentage
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
        println!("Rank with the table; decide with the whole-pass number (docs/lessons.md #21).");
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

/// Time one submit of `steps`, best of `reps`.
///
/// Every timed region is `poll_wait`-bracketed. This is not defensive style: on
/// the wgpu backend `submit` with an empty clear list only appends to the
/// pending list, so an unbracketed loop times host-side recording and reports a
/// rate above the physical roof (`docs/kernel-checklist.md` §E.0 — it once
/// produced 377 GB/s on a ~346 GB/s card).
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
    if let Some(d) = &device_times {
        let mut per: HashMap<usize, (f64, usize, u64, u64, u64, bool)> = HashMap::new();
        for (k, start, len) in &gs {
            accumulate(gpu, &mut per, *k, &steps[*start..*start + *len], 0.0);
        }
        let mut rows = finish_rows(gpu, per);
        // Overwrite the (unused) host times with the measured device ones.
        for r in rows.iter_mut() {
            if let Some((secs, calls)) = d.get(&r.name) {
                r.secs = *secs;
                r.calls = *calls as usize;
            } else {
                r.secs = 0.0;
            }
        }
        rows.sort_by(|a, b| b.secs.partial_cmp(&a.secs).unwrap());
        let summed = rows.iter().map(|r| r.secs).sum();
        let fully_covered = rows.iter().all(|r| r.covered);
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
        // The measured VQGAN backward: 658.09 ms of groups over a 456.07 ms pass.
        let p = pass(vec![row("a", 0.65809, 0, 1, true)], 0.45607);
        assert!((p.drain_overhead_pct() - 44.3).abs() < 0.5, "{}", p.drain_overhead_pct());
    }

    #[test]
    fn defects_are_rows_under_their_own_classs_floor() {
        let roofs = Roofs { gflops: 11760.0, gbs: 346.0, cache_gbs: 1200.0, int8_gops: Some(40000.0) };
        // col2im-shaped: 142.75 ms moving ~3.3 GB => ~23 GB/s, 6.7% of the roof.
        let col2im = row("col2im", 0.14275, 0, 3_312_000_000, true);
        // A GEMM at 2 TFLOP/s: 17% of peak — under the 30% compute floor too.
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
