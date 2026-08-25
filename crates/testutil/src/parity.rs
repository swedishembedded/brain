// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared stage-parity report: cosine + max_abs per tap, one floor for the
//! whole run.
//!
//! Every goldens test in this workspace wants the identical four things - a
//! cosine, a max absolute difference, a printed line per stage, and one
//! assertion at the end naming the worst stage, which is the same drift risk
//! that produced [`crate::testdata`]. [`Report`] is the strictest of the
//! private copies that predated this module: the only one that refused a
//! degenerate all-zero reference tap and treated a `NaN` cosine as a failure
//! rather than letting `partial_cmp` swallow it.
//!
//! Not every parity suite can reach these types. `crates/diffusion`'s
//! scheduler ladder gates a max RELATIVE deviation over a sigma schedule, where
//! a cosine is meaningless; the `clip` and `t5encoder` suites print their
//! comparison lines to **stderr** with a `cosine=` label and their own column
//! widths, and also split one stage by row population (content rows vs
//! right-pad rows). Reproducing all of those here would make the format a
//! parameter of the helper, which is how a shared helper stops being one.
//!
//! ## Why there are three report types and not one
//!
//! The printed lines of a parity suite are the evidence its numbers were
//! certified from, so a shared helper may not quietly re-format them. The
//! copies this module absorbed fall into three shapes that differ in *when*
//! they print and *what* they gate on, and each shape's format is load-bearing
//! for a suite whose numbers are already on the record:
//!
//! * [`Report`] streams one line per tap as it goes and gates on a single
//!   cosine floor. Best when the taps are slow: a hang is attributable.
//! * [`Table`] accumulates silently and gates on a cosine floor **and** a
//!   relative-L2 ceiling, leaving the caller to print an aligned table once
//!   every row is known. Best when a suite interleaves its own progress
//!   `println!`s with the comparisons.
//! * [`WorstTable`] is [`Table`] plus its own printing, and names both
//!   extremes: at a parity level where every cosine rounds to 1.000000000,
//!   relative L2 is the only discriminating column left.
//!
//! ## Why both numbers, always
//!
//! Cosine alone cannot see a scale error (a tensor twice as large is cosine
//! 1.0); max_abs alone cannot see a shape/permutation error (two tensors with
//! the same value distribution in the wrong order can have a small max_abs on a
//! smooth field). Reporting one without the other has certified a wrong port
//! before.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

/// Cosine similarity and max absolute difference between two equal-length
/// buffers, accumulated in `f64` so a long tensor's dot product does not lose
/// the low bits it is being asked to certify.
pub fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        mx = mx.max((x - y).abs());
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom, mx)
}

/// Relative L2 error, `||got - want|| / ||want||` - the third number, and the
/// only one of the three that is scale-free AND sees every element.
///
/// Cosine is blind to a uniform scale, max_abs is dominated by whichever single
/// element is worst, and a tensor can look fine on both while a broad fraction
/// of its energy is in the wrong place.
///
/// An all-zero reference has no norm to divide by. `0.0` when `got` is also
/// zero (the two really are equal); [`f64::INFINITY`] otherwise, so that a
/// nonzero result against an empty reference is reported as maximally wrong
/// rather than as a perfect score. [`Report::check`] refuses the degenerate
/// reference outright, but [`Table`] and [`WorstTable`] gate on this number
/// alone, so it has to carry the verdict itself.
pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch {} vs {}", got.len(), want.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in got.iter().zip(want.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (num / den).sqrt()
}

/// One parity run's accumulated per-stage results and the cosine floor they are
/// all held to.
pub struct Report {
    /// `(name, cosine, max_abs)` in the order the stages were checked.
    pub rows: Vec<(String, f64, f64)>,
    /// Every stage must reach at least this cosine.
    pub floor: f64,
    /// Column width for the printed stage name.
    width: usize,
}

impl Report {
    /// A report holding every stage to `floor`.
    pub fn new(floor: f64) -> Report {
        Report { rows: Vec::new(), floor, width: 24 }
    }

    /// Same, with a wider name column (long tap names, e.g. per-layer MoE
    /// internals, otherwise push the numbers out of alignment).
    pub fn wide(floor: f64, width: usize) -> Report {
        Report { rows: Vec::new(), floor, width }
    }

    /// Compare one stage and record it.
    ///
    /// A reference tap that is identically ~zero compares "perfectly" against
    /// anything of the same length and therefore proves nothing - that is a
    /// defect in the fixture or in the tap selection, so it fails loudly here
    /// rather than inflating the pass count.
    pub fn check(&mut self, name: &str, got: &[f32], want: &[f32]) {
        let (c, m) = compare(got, want);
        let r = rel_l2(got, want);
        println!(
            "  {:<w$} cos {c:.10}  rel_l2 {r:.3e}  max_abs {m:.3e}  n={}",
            name,
            want.len(),
            w = self.width
        );
        assert!(
            want.iter().any(|v| v.abs() > 1e-6),
            "{name}: the REFERENCE tap is all ~zero -- degenerate comparison"
        );
        self.rows.push((name.to_string(), c, m));
    }

    /// [`Self::check`] against a named tensor of a loaded golden file.
    pub fn against(&mut self, name: &str, got: &[f32], golden: &HashMap<String, StTensor>) {
        let want = golden.get(name).unwrap_or_else(|| panic!("golden tap {name} missing"));
        self.check(name, got, &want.data);
    }

    /// The `(name, cosine, max_abs)` of the worst stage so far.
    pub fn worst(&self) -> &(String, f64, f64) {
        self.rows.iter().min_by(|a, b| a.1.total_cmp(&b.1)).expect("no stages compared")
    }

    /// Print the summary and assert every stage cleared the floor. A `NaN`
    /// cosine counts as a failure: `partial_cmp`-based ordering would otherwise
    /// silently sort it away.
    pub fn finish(self, label: &str) {
        let worst = self.worst().clone();
        println!("  [{label}] {} taps, worst cosine {:.10} at {}", self.rows.len(), worst.1, worst.0);
        let bad: Vec<&(String, f64, f64)> =
            self.rows.iter().filter(|r| r.1.is_nan() || r.1 < self.floor).collect();
        assert!(bad.is_empty(), "{label}: {} tap(s) below cosine {}: {bad:?}", bad.len(), self.floor);
    }
}

/// A parity run that gates on cosine **and** relative L2 and prints nothing
/// until the caller asks.
///
/// Two floors rather than one because they fail on different things: a stage
/// uniformly double the reference is cosine 1.0 and fails only the relative-L2
/// ceiling, while a permuted stage keeps its magnitude and fails only the
/// cosine floor. Both are recorded per row, and every violation of either is
/// collected so one run names every bad stage instead of stopping at the first.
pub struct Table {
    /// `(name, cosine, max_abs, rel_l2)` in the order the stages were checked.
    pub rows: Vec<(String, f64, f64, f64)>,
    /// One human-readable line per floor violation, in the order they happened.
    pub failures: Vec<String>,
    cos_floor: f64,
    rel_ceiling: f64,
}

impl Table {
    /// A table holding every stage to `cos_floor` and `rel_ceiling`.
    pub fn new(cos_floor: f64, rel_ceiling: f64) -> Table {
        Table { rows: Vec::new(), failures: Vec::new(), cos_floor, rel_ceiling }
    }

    /// Compare one stage and record it. Nothing is printed; see
    /// [`Self::print`].
    pub fn check(&mut self, label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: {} values, golden has {}", got.len(), want.len());
        let (c, m) = compare(got, want);
        let r = rel_l2(got, want);
        self.rows.push((label.to_string(), c, m, r));
        // NaN-safe: a NaN cosine must fail, so this is an explicit NaN check
        // plus `<`, never `!(>=)`.
        if c.is_nan() || c < self.cos_floor {
            self.failures
                .push(format!("{label}: cosine {c:.10} < {}, max_abs {m:.3e}", self.cos_floor));
        }
        if r.is_nan() || r > self.rel_ceiling {
            self.failures.push(format!(
                "{label}: rel_l2 {r:.3e} > {:.0e}, max_abs {m:.3e}",
                self.rel_ceiling
            ));
        }
    }

    /// Whether a stage of this name was compared - the check that a golden is
    /// not sitting in the fixture with no tap pointed at it.
    pub fn has(&self, name: &str) -> bool {
        self.rows.iter().any(|(k, ..)| k == name)
    }

    /// The header and one right-aligned row per stage.
    pub fn print(&self) {
        println!("\n{:<40} {:>14} {:>11} {:>11}", "stage", "cosine", "max_abs", "rel_l2");
        for (k, c, mx, rl) in &self.rows {
            println!("{k:<40} {c:>14.10} {mx:>11.3e} {rl:>11.3e}");
        }
    }

    /// `(name, cosine)` of the lowest-cosine stage.
    ///
    /// Seeded at `("", 1.0)`: a run in which every stage is exactly cosine 1.0
    /// has no worst stage to name, and printing an arbitrary tap's name there
    /// would read as if that tap were the weak one.
    pub fn worst_cosine(&self) -> (String, f64) {
        let mut worst = (String::new(), 1.0f64);
        for (k, c, ..) in &self.rows {
            if *c < worst.1 {
                worst = (k.clone(), *c);
            }
        }
        worst
    }

    /// `(name, rel_l2)` of the worst-magnitude stage, seeded at `("", 0.0)` for
    /// the same reason as [`Self::worst_cosine`].
    pub fn worst_rel_l2(&self) -> (String, f64) {
        let mut worst = (String::new(), 0.0f64);
        for (k, _, _, r) in &self.rows {
            if *r > worst.1 {
                worst = (k.clone(), *r);
            }
        }
        worst
    }

    /// Assert every stage cleared both floors, naming all of them at once.
    pub fn assert_clean(&self) {
        assert!(
            self.failures.is_empty(),
            "{} failed:\n  {}",
            self.failures.len(),
            self.failures.join("\n  ")
        );
    }
}

/// [`Table`] that owns its own printing and names both extremes.
///
/// The floors are passed to [`Self::finish`] rather than to the constructor
/// because the suites this serves reuse one accumulator shape across sections
/// held to different tolerances (a quantizer unit at 1e-6 and a full 128x128
/// forward at 1e-4, in the same file).
pub struct WorstTable {
    /// `(name, cosine, max_abs, rel_l2)` in the order the stages were checked.
    pub rows: Vec<(String, f64, f64, f64)>,
    width: usize,
}

impl WorstTable {
    /// An empty table whose printed name column is `width` wide.
    pub fn new(width: usize) -> WorstTable {
        WorstTable { rows: Vec::new(), width }
    }

    /// Compare one stage and record it. Nothing is printed; see
    /// [`Self::finish`].
    pub fn add(&mut self, label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: {} values vs golden {}", got.len(), want.len());
        let (c, m) = compare(got, want);
        self.rows.push((label.to_string(), c, m, rel_l2(got, want)));
    }

    /// Print every stage, then assert the worst cosine clears `floor` and the
    /// worst relative L2 clears `rel_floor`.
    pub fn finish(&self, title: &str, floor: f64, rel_floor: f64) {
        println!("\n=== {title} ===");
        let mut worst = (f64::INFINITY, String::new());
        let mut worst_rel = (0.0f64, String::new());
        for (name, cos, mad, rel) in &self.rows {
            println!(
                "  {name:<w$} cosine {cos:.9} (1-cos {:.2e})  max|Δ| {mad:.3e}  relL2 {rel:.3e}",
                1.0 - cos,
                w = self.width
            );
            if *cos < worst.0 {
                worst = (*cos, name.clone());
            }
            if *rel > worst_rel.0 {
                worst_rel = (*rel, name.clone());
            }
        }
        println!("  worst: {} at cosine {:.9} (1-cos {:.2e})", worst.1, worst.0, 1.0 - worst.0);
        println!("  worst relative L2: {} at {:.3e}", worst_rel.1, worst_rel.0);
        assert!(worst.0 >= floor, "{title}: {} cosine {:.9} < {floor}", worst.1, worst.0);
        assert!(
            worst_rel.0 <= rel_floor,
            "{title}: {} relative L2 {:.3e} > {rel_floor:.0e} - the direction matches but the \
             MAGNITUDE does not",
            worst_rel.1,
            worst_rel.0
        );
    }
}

/// Read a safetensors golden into a name-keyed map - the shape every parity
/// test wants it in.
pub fn load(path: &std::path::Path) -> HashMap<String, StTensor> {
    checkpoint::safetensors::read(path.to_str().expect("utf-8 path"))
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_buffers_are_cosine_one_and_zero_max_abs() {
        let a = [1.0f32, -2.0, 3.5];
        assert_eq!(compare(&a, &a), (1.0, 0.0));
    }

    /// Cosine cannot see a pure scale error; max_abs can. The pair is the point.
    #[test]
    fn a_scaled_copy_is_cosine_one_but_not_max_abs_zero() {
        let a = [1.0f32, -2.0, 3.5];
        let b = [2.0f32, -4.0, 7.0];
        let (c, m) = compare(&a, &b);
        assert!((c - 1.0).abs() < 1e-12, "{c}");
        assert!((m - 3.5).abs() < 1e-6, "{m}");
        // ...and rel_l2 sees it too: half the reference's norm is missing.
        assert!((rel_l2(&a, &b) - 0.5).abs() < 1e-12, "{}", rel_l2(&a, &b));
    }

    #[test]
    fn rel_l2_is_zero_on_equality_and_defined_on_an_all_zero_reference() {
        let a = [1.0f32, -2.0, 3.5];
        assert_eq!(rel_l2(&a, &a), 0.0);
        assert_eq!(rel_l2(&[0.0; 3], &[0.0; 3]), 0.0);
        // A nonzero result against an empty reference is maximally wrong, not
        // perfect: `Table`/`WorstTable` gate on this number with no separate
        // degenerate-reference guard behind it.
        assert_eq!(rel_l2(&a, &[0.0; 3]), f64::INFINITY);
    }

    /// Both floors, because they catch different defects.
    #[test]
    fn a_table_gates_a_permutation_on_cosine_and_a_scale_on_rel_l2() {
        let mut t = Table::new(0.9999, 1e-3);
        t.check("permuted", &[1.0, 2.0], &[2.0, 1.0]);
        // Cosine 1.0 exactly, and only the magnitude is wrong: this row is the
        // one a cosine-only ladder waves through.
        t.check("scaled", &[2.0, 4.0], &[1.0, 2.0]);
        assert!(t.failures.iter().any(|f| f.starts_with("permuted: cosine ")), "{:?}", t.failures);
        assert!(
            t.failures.iter().filter(|f| f.starts_with("scaled: ")).count() == 1
                && t.failures.iter().any(|f| f.starts_with("scaled: rel_l2 ")),
            "{:?}",
            t.failures
        );
        assert!(t.has("scaled") && !t.has("absent"));
    }

    /// An all-1.0 run must not accuse an arbitrary tap of being the worst.
    #[test]
    fn a_tables_worst_is_unnamed_when_every_stage_is_exact() {
        let mut t = Table::new(0.9999, 1e-3);
        t.check("a", &[1.0, 0.0], &[1.0, 0.0]);
        t.check("b", &[3.0, 0.0], &[3.0, 0.0]);
        assert_eq!(t.worst_cosine(), (String::new(), 1.0));
        assert_eq!(t.worst_rel_l2(), (String::new(), 0.0));
        t.assert_clean();
    }

    #[test]
    fn a_worst_table_fails_on_magnitude_even_at_cosine_one() {
        let mut w = WorstTable::new(30);
        w.add("scaled", &[2.0, 4.0], &[1.0, 2.0]);
        let e = std::panic::catch_unwind(move || w.finish("t", 0.9999, 1e-4)).unwrap_err();
        let msg = e.downcast_ref::<String>().map(String::as_str).unwrap_or("");
        assert!(msg.contains("MAGNITUDE"), "{msg:?}");
    }

    #[test]
    fn a_stage_below_the_floor_fails() {
        let mut r = Report::new(0.9999);
        r.check("good", &[1.0, 2.0], &[1.0, 2.0]);
        r.check("bad", &[1.0, 2.0], &[2.0, -1.0]);
        let e = std::panic::catch_unwind(move || r.finish("t")).unwrap_err();
        let msg = e.downcast_ref::<String>().map(String::as_str).unwrap_or("");
        assert!(msg.contains("bad"), "{msg:?}");
    }

    /// The guard that separates this helper from the nine it replaces.
    #[test]
    fn an_all_zero_reference_tap_is_refused() {
        let mut r = Report::new(0.9999);
        let e = std::panic::catch_unwind(move || r.check("z", &[0.0, 0.0], &[0.0, 0.0])).unwrap_err();
        let msg = e.downcast_ref::<String>().map(String::as_str).unwrap_or("");
        assert!(msg.contains("degenerate"), "{msg:?}");
    }
}
