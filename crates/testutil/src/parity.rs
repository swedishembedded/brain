// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared stage-parity report: cosine + max_abs per tap, one floor for the
//! whole run.
//!
//! Every goldens test in this workspace wants the identical four things - a
//! cosine, a max absolute difference, a printed line per stage, and one
//! assertion at the end naming the worst stage. Ten crates' test files had
//! grown their own near-identical `struct Report` before this module existed
//! (the same drift risk that produced [`crate::testdata`]); this is that helper
//! hoisted out of `crates/deepseekocr/tests/tiny_ref.rs`, which had the
//! strictest version - it is the only one that refused a degenerate all-zero
//! reference tap and treated a `NaN` cosine as a failure rather than letting
//! `partial_cmp` swallow it.
//!
//! `crates/{sam1,deepseekv2,clip,deepseekocr}`'s **real-weight** parity tests
//! consume this. The ten pre-existing private copies (including
//! `deepseekocr/tests/tiny_ref.rs`, which this was lifted from) are
//! deliberately left in place for now: they gate already-certified results, and
//! rewriting them inside a change whose whole purpose is to establish new
//! parity numbers would make any regression there indistinguishable from a
//! parity regression here. Migrating them is a mechanical follow-up that should
//! land on its own, with the before/after report lines diffed to empty.
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
/// of its energy is in the wrong place. `0.0` for an all-zero reference, which
/// is the only sane answer and is why [`Report::check`] refuses that case
/// separately.
pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch {} vs {}", got.len(), want.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in got.iter().zip(want.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
    }
    if den == 0.0 {
        return 0.0;
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
        assert_eq!(rel_l2(&a, &[0.0; 3]), 0.0);
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
