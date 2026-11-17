// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The graded kernel table: each tuned class's chosen candidate against the
//! roofline bound of its class, the device's gap list, and a pass or fail per
//! slot - what `brain perf roofline --kernels` prints.
//!
//! Swedish Embedded AB implements roofline validation of inference kernels for
//! its clients. If your team needs to know which of its kernels leave the
//! hardware idle, and by how much, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # The bound of a class
//!
//! A class's roof time is the shortest time its problem can take on the
//! device: the bytes it must move (weights and their scales once, the
//! activations in their narrowest form, the fp32 output) over the memory roof,
//! and for a compute-bound class also its operations over the tensor-core roof
//! of the slot's precision, whichever is longer. The fraction is that roof time
//! over the measured median, so it says how much of the hardware the kernel
//! used, and it cannot pass the whole roof:
//!
//! * every roof is the **largest** rate measured for it on the device - the
//!   tuner's probe, a rate measured outside the tuner ([`ExtraRoofs`]: the
//!   device rows' best kernels), or any candidate the table timed. A kernel
//!   that outruns a probe proves the probe was not the ceiling, and the roof is
//!   raised to it (the report names who raised it) instead of grading that
//!   kernel above its roof;
//! * the launch floor is NOT folded into the bound. A class whose roof time is
//!   below the floor is launch-bound: no kernel can approach its roof, so it is
//!   shown with its own `launch` bound and the floor beside it, and kept out of
//!   the gap list and the pass/fail, which are about the roofs.
//!
//! A class whose winner timed noisily ([`Verdict::noisy`]) is graded but not
//! counted in the gap list or the verdict, and the report says how many.

use super::{grid, Class, Entry, MoeProblem, Tuned, Verdict, GAP_THRESHOLD, MOE_SLOT};

/// Rates measured outside the tuner that may raise its roofs: what the device
/// rows' best kernels reached (`crate::device_rates`).
#[derive(Clone, Debug, Default)]
pub struct ExtraRoofs {
    /// The device memory's read bandwidth, GB/s, and how it was measured.
    pub memory_gbs: Option<(f64, String)>,
    /// The int8 tensor-core rate, TOPS, and how it was measured.
    pub int8_tops: Option<(f64, String)>,
    /// The FP8 tensor-core rate, TOPS, and how it was measured.
    pub fp8_tops: Option<(f64, String)>,
}

/// A roof a class is graded against, and where its value came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Roof {
    pub value: f64,
    pub source: String,
}

/// Which bound a class is under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassBound {
    /// Its bytes over the memory roof take longest.
    Memory,
    /// Its operations over the tensor-core roof take longest.
    Compute,
    /// Its roof time is under the launch floor: no kernel can approach the roof.
    Launch,
}

impl ClassBound {
    pub fn as_str(self) -> &'static str {
        match self {
            ClassBound::Memory => "memory",
            ClassBound::Compute => "compute",
            ClassBound::Launch => "launch",
        }
    }

    /// The share of its roof a class of this bound is held to ([`crate::roof::Bound::target_pct`]); a launch-bound
    /// class has none.
    pub fn target(self) -> Option<f64> {
        match self {
            ClassBound::Memory => Some(f64::from(crate::roof::Bound::Memory.target_pct()) / 100.0),
            ClassBound::Compute => Some(f64::from(crate::roof::Bound::Compute.target_pct()) / 100.0),
            ClassBound::Launch => None,
        }
    }
}

/// The work of one class's problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cost {
    /// Bytes the problem must move at least.
    pub bytes: f64,
    /// Multiply-add operations, counted as two.
    pub ops: f64,
    /// Whether the class is graded as compute-bound (a weight-streaming class is not: it has far fewer operations
    /// than the compute roof allows a byte).
    pub compute: bool,
}

/// Which tensor-core roof a slot's compute-bound classes are graded against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    Int8,
    Fp8,
}

fn precision(slot: &str) -> Precision {
    if slot == crate::native_upgrade::FP8_SLOT { Precision::Fp8 } else { Precision::Int8 }
}

/// The work of `slot`'s class `class`, on the problem the tuner timed it on.
pub(crate) fn cost(slot: &str, class: Class) -> Cost {
    let g = grid(slot);
    let (m, kg, n) = class.problem(g);
    let (m, n, k) = (f64::from(m), f64::from(n), f64::from(kg) * 4.0);
    if slot == MOE_SLOT {
        let p = MoeProblem::new(m as u32, kg, n as u32);
        let (ne, slots, rows) = (f64::from(p.ne), f64::from(p.slots()), f64::from(p.slots() / p.xdiv));
        let bytes = ne * n * k * (1.0 + 4.0 / 32.0) + rows * k * (1.0 + 4.0 / 32.0) + slots * n * 4.0;
        return Cost { bytes, ops: 2.0 * slots * n * k, compute: true };
    }
    // Weights once with their scales; the activations in the narrowest form any candidate reads (one byte an element,
    // a scale per group); the fp32 output.
    let (weight_scales, act_group) = if g.fp8 { ((n / 128.0).ceil() * (k / 128.0) * 4.0, 128.0) } else { (n * k / 32.0 * 4.0, 32.0) };
    let bytes = n * k + weight_scales + m * k * (1.0 + 4.0 / act_group) + m * n * 4.0;
    Cost { bytes, ops: 2.0 * m * n * k, compute: !g.streams(m as u32) }
}

/// The roof time of `cost` in microseconds, and which side bounds it; `None` while a roof it needs is unmeasured.
pub(crate) fn roof_us(cost: Cost, memory_gbs: Option<f64>, tops: Option<f64>) -> Option<(f64, ClassBound)> {
    let memory = cost.bytes / (memory_gbs.filter(|r| *r > 0.0)? * 1e9) * 1e6;
    if !cost.compute {
        return Some((memory, ClassBound::Memory));
    }
    let compute = cost.ops / (tops.filter(|r| *r > 0.0)? * 1e12) * 1e6;
    Some(if compute >= memory { (compute, ClassBound::Compute) } else { (memory, ClassBound::Memory) })
}

/// The fraction of its roof a class reached: roof time over measured time, never above one (the roofs are raised to
/// any measured rate, so a value past one is rounding).
pub(crate) fn fraction(roof_us: f64, measured_us: f64) -> f64 {
    (roof_us / measured_us).min(1.0)
}

/// `(rate, who)`: the largest of the candidates, `None` if none.
fn largest(candidates: impl Iterator<Item = (f64, String)>) -> Option<(f64, String)> {
    candidates.filter(|(v, _)| v.is_finite() && *v > 0.0).fold(None, |best: Option<(f64, String)>, c| if best.as_ref().is_none_or(|b| c.0 > b.0) { Some(c) } else { best })
}

/// The roof from a probe, an outside measurement and the rates the table's candidates reached: the largest, named.
fn raise(probe: Option<(f64, String)>, extra: Option<(f64, String)>, achieved: Option<(f64, String)>) -> Option<Roof> {
    let base = largest(probe.into_iter().chain(extra));
    match (base, achieved) {
        (Some(b), Some(a)) if a.0 > b.0 => Some(Roof { value: a.0, source: format!("raised by {} (above the {})", a.1, b.1) }),
        (Some(b), _) => Some(Roof { value: b.0, source: b.1 }),
        (None, Some(a)) => Some(Roof { value: a.0, source: format!("{}, no probe", a.1) }),
        (None, None) => None,
    }
}

/// One graded class.
#[derive(Clone, Debug)]
pub struct ClassRow {
    pub slot: String,
    pub class: String,
    pub chosen: String,
    /// The chosen candidate's median and fastest launch and the share of its launches stretched; `None` when unmeasured.
    pub us: Option<f64>,
    pub min_us: Option<f64>,
    pub stretched: Option<f64>,
    /// GB/s for a streaming class, TOPS for a compute-bound one.
    pub rate: Option<f64>,
    pub rate_unit: &'static str,
    pub roof_us: Option<f64>,
    pub fraction: Option<f64>,
    pub bound: Option<ClassBound>,
    pub floor_us: Option<f64>,
    pub noisy: bool,
    /// The other candidates: `(name, median microseconds or the reason it was refused)`.
    pub others: Vec<(String, String)>,
}

/// The verdict of one slot's classes of one bound.
#[derive(Clone, Debug)]
pub struct SlotVerdict {
    pub slot: String,
    pub bound: ClassBound,
    /// Classes graded (not noisy, measured).
    pub classes: usize,
    pub median: f64,
    pub min: f64,
    pub target: f64,
    pub pass: bool,
    /// The lowest classes: `(class, fraction, chosen)`.
    pub slowest: Vec<(String, f64, String)>,
}

/// One device's graded table.
#[derive(Clone, Debug)]
pub struct TableReport {
    pub device: String,
    pub from_store: bool,
    /// The share of floor-probe launches another context stretched while it was measured.
    pub busy: Option<f64>,
    pub memory_roof: Option<Roof>,
    pub int8_roof: Option<Roof>,
    pub fp8_roof: Option<Roof>,
    pub floor_us: Option<f64>,
    pub rows: Vec<ClassRow>,
    /// Indices into `rows` below [`GAP_THRESHOLD`], lowest first.
    pub gaps: Vec<usize>,
    pub latency_bound: usize,
    pub noisy: usize,
    pub slots: Vec<SlotVerdict>,
}

/// Slots that ran no class this table holds are left out; the slowest classes a verdict names.
const SLOWEST: usize = 3;

/// `t` graded against its roofs, raised by `extra`.
pub fn grade(t: &Tuned, extra: &ExtraRoofs) -> TableReport {
    let timed = |e: &Entry| e.results.iter().filter_map(|(k, v)| match v {
        Verdict::Timed { us, .. } if us.is_finite() && *us > 0.0 && !v.noisy() => Some((k.clone(), *us)),
        _ => None,
    }).collect::<Vec<_>>();
    let achieved = |want_compute: bool, prec: Option<Precision>| {
        largest(t.entries.iter().filter(|e| prec.is_none_or(|p| precision(&e.slot) == p)).flat_map(|e| {
            let c = cost(&e.slot, e.class);
            let label = e.describe();
            timed(e).into_iter().filter(move |_| c.compute == want_compute).map(move |(k, us)| {
                let rate = if want_compute { c.ops / (us * 1e-6) / 1e12 } else { c.bytes / (us * 1e-6) / 1e9 };
                (rate, format!("{k} at {label}"))
            })
        }))
    };
    let memory_roof = raise(t.roof_gbs.map(|r| (f64::from(r), "read probe".to_owned())), extra.memory_gbs.clone(), achieved(false, None));
    let int8_roof = raise(t.int8_tops.map(|r| (f64::from(r), "mma.sync probe".to_owned())), extra.int8_tops.clone(), achieved(true, Some(Precision::Int8)));
    let fp8_roof = raise(None, extra.fp8_tops.clone(), achieved(true, Some(Precision::Fp8)));
    let floor_us = t.floor_us.map(f64::from);
    let rows: Vec<ClassRow> = t
        .entries
        .iter()
        .map(|e| {
            let c = cost(&e.slot, e.class);
            let tops = match precision(&e.slot) {
                Precision::Int8 => int8_roof.as_ref(),
                Precision::Fp8 => fp8_roof.as_ref(),
            };
            let roof = roof_us(c, memory_roof.as_ref().map(|r| r.value), tops.map(|r| r.value));
            let chosen = e.chosen();
            let (us, min_us, stretched) = match chosen {
                Some(Verdict::Timed { us, min_us, stretched, .. }) if us.is_finite() => (Some(*us), Some(*min_us), Some(*stretched)),
                _ => (None, None, None),
            };
            let bound = roof.map(|(r, b)| if floor_us.is_some_and(|f| r < f) { ClassBound::Launch } else { b });
            let others = e
                .results
                .iter()
                .filter(|(k, _)| *k != e.best)
                .map(|(k, v)| {
                    (k.clone(), match v {
                        Verdict::Timed { us, .. } => format!("{us:.1}us"),
                        Verdict::Rejected(why) => format!("REJECTED ({why})"),
                    })
                })
                .collect();
            ClassRow {
                slot: e.slot.clone(),
                class: e.describe(),
                chosen: e.best.clone(),
                us,
                min_us,
                stretched,
                rate: us.map(|us| if c.compute { c.ops / (us * 1e-6) / 1e12 } else { c.bytes / (us * 1e-6) / 1e9 }),
                rate_unit: if c.compute { "TOPS" } else { "GB/s" },
                roof_us: roof.map(|(r, _)| r),
                fraction: roof.zip(us).map(|((r, _), us)| fraction(r, us)),
                bound,
                floor_us,
                noisy: chosen.is_some_and(Verdict::noisy),
                others,
            }
        })
        .collect();
    let counted = |r: &ClassRow| !r.noisy && r.fraction.is_some() && r.bound.is_some_and(|b| b != ClassBound::Launch);
    let mut gaps: Vec<usize> = (0..rows.len()).filter(|&i| counted(&rows[i]) && rows[i].fraction.is_some_and(|f| f < GAP_THRESHOLD)).collect();
    gaps.sort_by(|&a, &b| rows[a].fraction.partial_cmp(&rows[b].fraction).unwrap_or(std::cmp::Ordering::Equal));
    let mut slots = Vec::new();
    let mut names: Vec<&str> = rows.iter().map(|r| r.slot.as_str()).collect();
    names.dedup();
    for slot in names {
        for bound in [ClassBound::Memory, ClassBound::Compute] {
            let mut graded: Vec<&ClassRow> = rows.iter().filter(|r| r.slot == slot && r.bound == Some(bound) && counted(r)).collect();
            if graded.is_empty() {
                continue;
            }
            graded.sort_by(|a, b| a.fraction.partial_cmp(&b.fraction).unwrap_or(std::cmp::Ordering::Equal));
            let fr: Vec<f64> = graded.iter().filter_map(|r| r.fraction).collect();
            let median = if fr.len() % 2 == 1 { fr[fr.len() / 2] } else { (fr[fr.len() / 2 - 1] + fr[fr.len() / 2]) / 2.0 };
            let target = bound.target().expect("a roof-bound class has a target");
            slots.push(SlotVerdict {
                slot: slot.to_owned(),
                bound,
                classes: graded.len(),
                median,
                min: fr[0],
                target,
                pass: median >= target,
                slowest: graded.iter().take(SLOWEST).map(|r| (r.class.clone(), r.fraction.unwrap_or(0.0), r.chosen.clone())).collect(),
            });
        }
    }
    TableReport {
        device: t.device.clone(),
        from_store: t.from_store,
        busy: t.busy.map(f64::from),
        memory_roof,
        int8_roof,
        fp8_roof,
        floor_us,
        latency_bound: rows.iter().filter(|r| r.bound == Some(ClassBound::Launch)).count(),
        noisy: rows.iter().filter(|r| r.noisy).count(),
        rows,
        gaps,
        slots,
    }
}

fn pct(f: f64) -> String {
    format!("{:.0}%", f * 100.0)
}

fn roof_text(r: &Option<Roof>, unit: &str) -> String {
    match r {
        Some(r) => format!("{:.0} {unit} ({})", r.value, r.source),
        None => "not measured".to_owned(),
    }
}

impl TableReport {
    /// The plain table, every line opening with `prefix` (an accelerator id, so `| grep gpu3` keeps whole lines).
    pub fn render(&self, prefix: &str) -> String {
        let mut out = String::new();
        for line in self.header_lines(prefix) {
            out += &line;
            out.push('\n');
        }
        for i in 0..self.rows.len() {
            out += &self.row_line(i, prefix);
            out.push('\n');
        }
        out += &self.gap_header(prefix);
        out.push('\n');
        for &i in &self.gaps {
            let r = &self.rows[i];
            out += &format!("{}gap   {} {} - {} of its {} roof with {}\n", lead(prefix), r.slot, r.class, pct(r.fraction.unwrap_or(0.0)), r.bound.map(ClassBound::as_str).unwrap_or("-"), r.chosen);
        }
        for v in &self.slots {
            out += &verdict_line(v, prefix);
            out.push('\n');
        }
        out
    }

    /// The table's header: the device and whether it was busy, the roofs, the column names.
    pub fn header_lines(&self, prefix: &str) -> [String; 3] {
        let p = lead(prefix);
        let busy = match self.busy {
            Some(b) if b > f64::from(super::BUSY_LIMIT) => format!("; CARD BUSY: {} of the floor probe's launches stretched by another context, the table is not stored", pct(b)),
            Some(b) => format!("; {} of the floor probe's launches stretched", pct(b)),
            None => String::new(),
        };
        [
            format!("{p}native kernel selection on {} ({}{busy})", self.device, if self.from_store { "from the store" } else { "measured now" }),
            format!(
                "{p}roofs: memory {}; int8 tensor core {}; fp8 tensor core {}; launch floor {}",
                roof_text(&self.memory_roof, "GB/s"),
                roof_text(&self.int8_roof, "TOPS"),
                roof_text(&self.fp8_roof, "TOPS"),
                self.floor_us.map(|f| format!("{f:.1} us")).unwrap_or_else(|| "not measured".into())
            ),
            format!(
                "{p}{:<18} {:<28} {:<26} {:>9} {:>9} {:>7} {:>10} {:>9} {:>5} {:<7} {:>6}  other candidates",
                "slot", "shape class", "chosen", "median us", "min us", "stretch", "GB/s|TOPS", "roof us", "roof", "bound", "floor"
            ),
        ]
    }

    /// The plain line of class row `i`.
    pub fn row_line(&self, i: usize, prefix: &str) -> String {
        let r = &self.rows[i];
        let num = |v: Option<f64>, d: usize| v.map(|v| format!("{v:.d$}")).unwrap_or_else(|| "-".into());
        let others: Vec<String> = r.others.iter().map(|(k, v)| format!("{k} {v}")).collect();
        format!(
            "{}{:<18} {:<28} {:<26} {:>9} {:>9} {:>7} {:>10} {:>9} {:>5} {:<7} {:>6}  {}{}",
            lead(prefix),
            r.slot,
            r.class,
            r.chosen,
            num(r.us, 1),
            num(r.min_us, 1),
            r.stretched.map(pct).unwrap_or_else(|| "-".into()),
            num(r.rate, 0),
            num(r.roof_us, 1),
            r.fraction.map(pct).unwrap_or_else(|| "-".into()),
            r.bound.map(ClassBound::as_str).unwrap_or("-"),
            num(r.floor_us, 1),
            if r.noisy { "NOISY " } else { "" },
            others.join(", ")
        )
    }

    /// The gap list's heading: how many roof-bound classes are under [`GAP_THRESHOLD`] and how many were not counted.
    pub fn gap_header(&self, prefix: &str) -> String {
        let graded = self.rows.iter().filter(|r| !r.noisy && r.fraction.is_some() && r.bound.is_some_and(|b| b != ClassBound::Launch)).count();
        format!(
            "{}gap: {} of {graded} roof-bound classes below {} of their roof ({} launch-bound and {} noisy classes not counted)",
            lead(prefix),
            self.gaps.len(),
            pct(GAP_THRESHOLD),
            self.latency_bound,
            self.noisy
        )
    }
}

/// `prefix` and a space, or nothing.
fn lead(prefix: &str) -> String {
    if prefix.is_empty() { String::new() } else { format!("{prefix} ") }
}

/// The plain line of one slot's verdict.
pub fn verdict_line(v: &SlotVerdict, prefix: &str) -> String {
    let slowest: Vec<String> = v.slowest.iter().map(|(c, f, k)| format!("{c} {} ({k})", pct(*f))).collect();
    format!(
        "{}validate {} {}-bound: {} classes, median {} of the roof, min {}, target {}: {}; slowest: {}",
        lead(prefix),
        v.slot,
        v.bound.as_str(),
        v.classes,
        pct(v.median),
        pct(v.min),
        pct(v.target),
        if v.pass { "PASS" } else { "FAIL" },
        slowest.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn timed(us: f64) -> Verdict {
        Verdict::Timed { us, gbs: 0.0, min_us: us * 0.98, stretched: 0.0 }
    }

    fn table(entries: Vec<Entry>) -> Tuned {
        Tuned { device: "dev".into(), roof_gbs: Some(3000.0), floor_us: Some(4.0), int8_tops: Some(1000.0), busy: Some(0.0), entries, from_store: false, index: OnceLock::new() }
    }

    fn entry(slot: &str, m: u32, kg: u32, n: u32, results: Vec<(&str, Verdict)>) -> Entry {
        let class = Class::of(grid(slot), m, kg, n).expect("a class of the grid");
        Entry { slot: slot.into(), class, best: results[0].0.into(), results: results.into_iter().map(|(k, v)| (k.to_owned(), v)).collect() }
    }

    /// The bound is the slower of bytes over the memory roof and, for a compute-bound class, operations over the
    /// tensor-core roof; it needs the roofs it uses.
    #[test]
    fn a_class_bound_is_its_bytes_or_its_operations_over_the_roof_that_takes_longer() {
        let streaming = Cost { bytes: 3e9, ops: 1e9, compute: false };
        let (us, bound) = roof_us(streaming, Some(3000.0), None).unwrap();
        assert!((us - 1e3).abs() < 1e-6 && bound == ClassBound::Memory, "{us} {bound:?}");
        let gemm = Cost { bytes: 3e6, ops: 2e12, compute: true };
        let (us, bound) = roof_us(gemm, Some(3000.0), Some(1000.0)).unwrap();
        assert!((us - 2000.0).abs() < 1e-6 && bound == ClassBound::Compute, "{us} {bound:?}");
        assert_eq!(roof_us(gemm, Some(3000.0), None), None, "a compute-bound class needs its compute roof");
        let thin = Cost { bytes: 3e10, ops: 2e12, compute: true };
        assert_eq!(roof_us(thin, Some(3000.0), Some(1000.0)).map(|r| r.1), Some(ClassBound::Memory));
    }

    /// The fraction cannot pass one, whatever the inputs.
    #[test]
    fn a_fraction_never_exceeds_the_roof() {
        assert_eq!(fraction(10.0, 20.0), 0.5);
        assert_eq!(fraction(30.0, 20.0), 1.0);
    }

    /// A candidate that outruns the probe raises the roof to its own rate (and is named), so the class it won is at
    /// the whole roof rather than above it; a class whose roof time is under the launch floor is launch-bound and out of the
    /// gap list however low its fraction.
    #[test]
    fn a_kernel_above_the_probe_raises_the_roof_and_a_class_under_the_floor_is_launch_bound() {
        // rows 1, K 5120, N 4096: about 23.7 MB of weights in 5 us, past the probe's memory roof.
        let fast = entry("matmul_i8_gemv", 1, 1280, 4096, vec![("gemv", timed(5.0))]);
        // rows 1, K 5120, N 256: ~1.5 MB, a roof time of ~0.3 us under a 4 us floor.
        let tiny = entry("matmul_i8_gemv", 1, 1280, 256, vec![("gemv", timed(9.0))]);
        // rows 1, K 17408, N 4096: ~80 MB, graded against the raised roof.
        let big = entry("matmul_i8_gemv", 1, 4352, 4096, vec![("gemv", timed(60.0)), ("other", timed(80.0))]);
        let r = grade(&table(vec![fast, tiny, big]), &ExtraRoofs::default());
        let roof = r.memory_roof.as_ref().unwrap();
        assert!(roof.value > 3000.0 && roof.source.contains("raised by gemv"), "{roof:?}");
        assert!(r.rows.iter().all(|row| row.fraction.is_none_or(|f| f <= 1.0)), "{:?}", r.rows);
        assert_eq!(r.rows[0].fraction, Some(1.0));
        assert_eq!(r.rows[1].bound, Some(ClassBound::Launch));
        assert_eq!(r.latency_bound, 1);
        assert!(!r.gaps.contains(&1), "a launch-bound class is not a roof gap");
        assert_eq!(r.rows[2].others, vec![("other".to_owned(), "80.0us".to_owned())]);
    }

    /// An outside measurement raises a roof the same way, and the verdict of a slot is its graded classes' median
    /// against the bound's target, naming the slowest.
    #[test]
    fn a_slot_passes_when_the_median_of_its_classes_reaches_the_target() {
        let at = |frac: f64, n: u32| {
            let c = cost("matmul_i8_gemv", Class::of(grid("matmul_i8_gemv"), 1, 4352, n).unwrap());
            entry("matmul_i8_gemv", 1, 4352, n, vec![("k", timed(c.bytes / 4000e9 * 1e6 / frac))])
        };
        let extra = ExtraRoofs { memory_gbs: Some((4000.0, "hbm read".into())), ..Default::default() };
        let r = grade(&table(vec![at(0.9, 1024), at(0.8, 4096)]), &extra);
        assert_eq!(r.memory_roof.as_ref().map(|m| m.source.as_str()), Some("hbm read"));
        let v = &r.slots[0];
        assert_eq!((v.bound, v.classes, v.pass), (ClassBound::Memory, 2, true));
        assert!((v.median - 0.85).abs() < 1e-9 && (v.min - 0.8).abs() < 1e-9, "{v:?}");
        let failing = grade(&table(vec![at(0.5, 1024), at(0.3, 4096)]), &extra);
        assert!(!failing.slots[0].pass);
        assert_eq!(failing.gaps.len(), 2);
        assert!(failing.rows[failing.gaps[0]].fraction < failing.rows[failing.gaps[1]].fraction, "the gap list is lowest first");
    }

    /// A noisy winner is shown but neither a gap nor part of a verdict, and a noisy candidate never raises a roof.
    #[test]
    fn a_noisy_class_is_flagged_and_not_counted() {
        let noisy = Verdict::Timed { us: 2000.0, gbs: 0.0, min_us: 10.0, stretched: 0.9 };
        let fast_noisy = Verdict::Timed { us: 0.5, gbs: 0.0, min_us: 0.4, stretched: 0.5 };
        let e = entry("matmul_i8_gemv", 1, 4352, 4096, vec![("k", noisy), ("x", fast_noisy)]);
        let r = grade(&table(vec![e]), &ExtraRoofs::default());
        assert!(r.rows[0].noisy);
        assert_eq!((r.noisy, r.gaps.len(), r.slots.len()), (1, 0, 0));
        assert_eq!(r.memory_roof.as_ref().map(|m| m.value), Some(3000.0), "a noisy timing raises nothing");
        assert!(r.render("gpu3").lines().all(|l| l.starts_with("gpu3 ")), "every line is greppable by its accelerator");
    }
}
