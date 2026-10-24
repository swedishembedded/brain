// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two-way coverage of an imported tensor map against a model's manifest.
//!
//! Every port validates the same contract at the checkpoint boundary: every
//! tensor the model's config declares is present exactly once with the right
//! shape and element count, and no source tensor is left unused. Never a
//! zero-fill, never a silent drop. The contract was restated by hand in each
//! model crate's `import.rs`, about twenty times, with the same messages
//! drifting apart; this is the one implementation.
//!
//! Three entry points, because loaders differ in what they can see before
//! reading data:
//!
//! * [`validate`] - a materialized `name -> (shape, values)` map (an eager
//!   importer).
//! * [`validate_shapes`] - only shapes are known (a streaming loader that must
//!   refuse a mismatched checkpoint BEFORE reading any weight).
//! * [`validate_counts`] - only shapes are known, and the source is free to
//!   store a 2-D weight flattened; the graph depends on how many values there
//!   are, not how the source punctuated them.
//!
//! Swedish Embedded AB implements checkpoint-import boundaries where a wrong
//! file fails by naming the tensor instead of loading halfway, for its
//! clients. If your team needs expertise in format-agnostic weight
//! conversion and its validation, you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::HashMap;

/// A materialized tensor map: canonical name -> (shape, fp32 values). The
/// shape every eager importer in the workspace uses (crate-local aliases are
/// this same type).
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Everything a manifest declares is present with that exact shape and that
/// many values, and nothing outside it is present.
///
/// `who` is the message prefix (`"clip import"`, `"ltxv vit import"`), so an
/// error still names the part of the model that refused.
pub fn validate(map: &Tensors, manifest: &[(String, Vec<usize>)], who: &str) -> Result<(), String> {
    for (name, shape) in manifest {
        let Some((got, data)) = map.get(name) else {
            return Err(format!("{who}: missing tensor {name}"));
        };
        if got != shape {
            return Err(format!("{who}: {name} shape {got:?}, expected {shape:?}"));
        }
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(format!("{who}: {name} has {} values, expected {n}", data.len()));
        }
    }
    reject_extra(map.keys().map(String::as_str), manifest, who)
}

/// [`validate`] over shapes only, for a loader that has not read any data:
/// every declared tensor is present with that exact shape, and `present`
/// carries nothing the manifest does not name.
pub fn validate_shapes(shape_of: impl Fn(&str) -> Option<Vec<usize>>, present: &[String], manifest: &[(String, Vec<usize>)], who: &str) -> Result<(), String> {
    for (name, shape) in manifest {
        match shape_of(name) {
            None => return Err(format!("{who}: missing tensor {name}")),
            Some(got) if &got != shape => {
                return Err(format!("{who}: {name} shape {got:?}, expected {shape:?}"));
            }
            Some(_) => {}
        }
    }
    reject_extra(present.iter().map(String::as_str), manifest, who)
}

/// [`validate`] over shapes only, comparing element counts: a source that
/// stored a 2-D weight flattened still matches. The declared shape is what an
/// error reports, so a genuine width mismatch stays readable.
pub fn validate_counts(shape_of: impl Fn(&str) -> Option<Vec<usize>>, present: &[String], manifest: &[(String, Vec<usize>)], who: &str) -> Result<(), String> {
    for (name, want) in manifest {
        let Some(got) = shape_of(name) else {
            return Err(format!("{who}: missing tensor {name}"));
        };
        let n: usize = want.iter().product();
        let got_n: usize = got.iter().product();
        if got_n != n {
            return Err(format!("{who}: {name} has {got_n} elements (shape {got:?}), expected {n} (shape {want:?})"));
        }
    }
    reject_extra(present.iter().map(String::as_str), manifest, who)
}

/// Refuse any source tensor the manifest does not name, listing them sorted.
fn reject_extra<'a>(present: impl Iterator<Item = &'a str>, manifest: &[(String, Vec<usize>)], who: &str) -> Result<(), String> {
    let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
    let mut extra: Vec<&str> = present.filter(|k| !expected.contains(k)).collect();
    if extra.is_empty() {
        return Ok(());
    }
    extra.sort_unstable();
    extra.dedup();
    Err(format!("{who}: unused source tensors: {extra:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Vec<(String, Vec<usize>)> {
        vec![("a".to_string(), vec![2, 3]), ("b".to_string(), vec![4])]
    }

    fn map(extra: Option<&str>) -> Tensors {
        let mut m = Tensors::new();
        m.insert("a".to_string(), (vec![2, 3], vec![0.0; 6]));
        m.insert("b".to_string(), (vec![4], vec![0.0; 4]));
        if let Some(name) = extra {
            m.insert(name.to_string(), (vec![1], vec![0.0]));
        }
        m
    }

    #[test]
    fn a_complete_map_passes() {
        assert_eq!(validate(&map(None), &manifest(), "m import"), Ok(()));
        let shapes: Vec<String> = manifest().into_iter().map(|(n, _)| n).collect();
        assert_eq!(validate_shapes(|n| manifest().into_iter().find(|(k, _)| k == n).map(|(_, s)| s), &shapes, &manifest(), "m import"), Ok(()));
    }

    #[test]
    fn a_missing_tensor_errors_by_name() {
        let mut m = map(None);
        m.remove("b");
        let e = validate(&m, &manifest(), "m import").unwrap_err();
        assert_eq!(e, "m import: missing tensor b");
    }

    #[test]
    fn a_wrong_shape_errors_with_both_shapes() {
        let mut m = map(None);
        m.insert("a".to_string(), (vec![3, 2], vec![0.0; 6]));
        let e = validate(&m, &manifest(), "m import").unwrap_err();
        assert_eq!(e, "m import: a shape [3, 2], expected [2, 3]");
    }

    #[test]
    fn a_wrong_value_count_errors_with_the_expected_count() {
        let mut m = map(None);
        m.insert("b".to_string(), (vec![4], vec![0.0; 3]));
        let e = validate(&m, &manifest(), "m import").unwrap_err();
        assert_eq!(e, "m import: b has 3 values, expected 4");
    }

    #[test]
    fn an_unused_source_tensor_is_refused_sorted() {
        let mut m = map(Some("z"));
        m.insert("y".to_string(), (vec![1], vec![0.0]));
        let e = validate(&m, &manifest(), "m import").unwrap_err();
        assert_eq!(e, r#"m import: unused source tensors: ["y", "z"]"#);
    }

    #[test]
    fn the_shapes_only_form_does_not_read_values() {
        // A map whose values disagree with its shapes still passes the
        // shapes-only check: a streaming loader has not read them yet.
        let shapes = |n: &str| match n {
            "a" => Some(vec![2, 3]),
            "b" => Some(vec![4]),
            _ => None,
        };
        let present = vec!["a".to_string(), "b".to_string()];
        assert_eq!(validate_shapes(shapes, &present, &manifest(), "m import"), Ok(()));
        assert_eq!(validate_counts(shapes, &present, &manifest(), "m import"), Ok(()));
    }

    #[test]
    fn the_counts_form_accepts_a_flattened_weight() {
        let shapes = |n: &str| match n {
            "a" => Some(vec![6]),
            "b" => Some(vec![4]),
            _ => None,
        };
        let present = vec!["a".to_string(), "b".to_string()];
        assert_eq!(validate_counts(shapes, &present, &manifest(), "m import"), Ok(()));
        // The exact-shape form refuses the same map and reports both shapes.
        let e = validate_shapes(shapes, &present, &manifest(), "m import").unwrap_err();
        assert_eq!(e, "m import: a shape [6], expected [2, 3]");
    }

    #[test]
    fn the_counts_form_reports_element_counts() {
        let shapes = |n: &str| match n {
            "a" => Some(vec![6]),
            "b" => Some(vec![5]),
            _ => None,
        };
        let present = vec!["a".to_string(), "b".to_string()];
        let e = validate_counts(shapes, &present, &manifest(), "m import").unwrap_err();
        assert_eq!(e, "m import: b has 5 elements (shape [5]), expected 4 (shape [4])");
    }
}
