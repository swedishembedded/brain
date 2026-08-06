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
    pub bytes: u64,
    /// False when any step of this kind has no `cost` formula, in which case
    /// the rates are not reported at all rather than reported as zero.
    pub covered: bool,
}

impl KernelRow {
    pub fn gflops(&self) -> Option<f64> {
        (self.covered && self.flops > 0).then(|| self.flops as f64 / self.secs / 1e9)
    }
    pub fn gbs(&self) -> Option<f64> {
        (self.covered && self.bytes > 0).then(|| self.bytes as f64 / self.secs / 1e9)
    }
    /// Which roof this kernel is graded against, given the device's ridge.
    pub fn bound(&self, roofs: Roofs) -> Option<Bound> {
        self.covered.then(|| roofs.classify(self.flops, self.bytes))
    }
    /// Percent of *its own class's* roof. This is the number the target bands
    /// in `.todo/saturate-the-gpu.md` are stated against — reporting a
    /// memory-bound kernel as a fraction of a FLOP peak is meaningless.
    pub fn utilisation(&self, roofs: Roofs) -> Option<f32> {
        self.covered.then(|| roofs.utilisation(self.flops, self.bytes, self.secs))?
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
}

impl PassProfile {
    /// Whole-pass FLOP rate. `None` when the pass is not fully covered — a
    /// partial numerator over the full denominator is a fiction, and the
    /// honest answer to "what rate did this pass achieve" is "unknown".
    pub fn gflops(&self) -> Option<f64> {
        let flops: u64 = self.rows.iter().map(|r| r.flops).sum();
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
        println!(
            "sum of groups {:.2} ms vs whole pass {:.2} ms — {:.0}% drain inflation over {} groups.",
            self.summed_secs * 1e3,
            self.total_secs * 1e3,
            self.drain_overhead_pct(),
            self.groups,
        );
        println!("Rank with the table; decide with the whole-pass number (docs/lessons.md #21).");
        if let Some(rf) = roofs {
            let bogus: Vec<&str> = self
                .rows
                .iter()
                .filter(|r| r.utilisation(rf).is_some_and(|u| u > IMPOSSIBLE_PCT))
                .map(|r| r.name.as_str())
                .collect();
            if !bogus.is_empty() {
                println!(
                    "!! {} row(s) report ABOVE the device roof — the accounting or the timing is \
                     wrong, not the kernel: {}",
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
    let total = best_of(gpu, steps, reps);
    let gs = groups(steps);

    let mut per: HashMap<usize, (f64, usize, u64, u64, bool)> = HashMap::new();
    for (k, start, len) in &gs {
        let t = best_of(gpu, &steps[*start..*start + *len], reps);
        let name = gpu.kernel_name(*k).unwrap_or("?");
        let (mut fl, mut by, mut covered) = (0u64, 0u64, true);
        for st in &steps[*start..*start + *len] {
            let m = st.meta();
            let params = m.as_ref().and_then(|m| m.params.as_deref());
            let threads = m.as_ref().map(|m| m.threads).unwrap_or(0);
            match crate::cost::kernel_cost(name, params, threads) {
                Some(c) => {
                    fl += c.flops.max(c.int_ops);
                    by += c.bytes;
                }
                None => covered = false,
            }
        }
        let e = per.entry(*k).or_insert((0.0, 0, 0, 0, true));
        e.0 += t;
        e.1 += *len;
        e.2 += fl;
        e.3 += by;
        e.4 &= covered;
    }

    let mut rows: Vec<KernelRow> = per
        .into_iter()
        .map(|(k, (secs, calls, flops, bytes, covered))| KernelRow {
            name: gpu.kernel_name(k).unwrap_or("?").to_string(),
            secs,
            calls,
            flops,
            bytes,
            covered,
        })
        .collect();
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, secs: f64, flops: u64, bytes: u64, covered: bool) -> KernelRow {
        KernelRow { name: name.into(), secs, calls: 1, flops, bytes, covered }
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
        let roofs = Roofs { gflops: 11760.0, gbs: 346.0 };
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
