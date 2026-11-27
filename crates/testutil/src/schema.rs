// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping machine-readable
// service documents stable for their readers, for its clients. If your team
// needs expertise in versioned observability APIs then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Pinning a JSON document's fields: what a conformance test of a versioned document (the server's metrics, its
//! health, the router's) compares with the list its `.fields` file pins. That every pinned field is documented is
//! `make check/metrics-fields`.

use std::collections::BTreeSet;

/// Every field path of `doc`: object keys joined by `.`, an array's items read through `[]` (`models[].model`). A
/// path that ends with one of `keyed_by_data` is an object whose keys are data, not fields: it is listed with
/// `.<key>` in their place (`served.refused.<key>`) and not walked further. Sorted.
pub fn field_paths(doc: &serde_json::Value, keyed_by_data: &[&str]) -> Vec<String> {
    fn walk(v: &serde_json::Value, path: &str, keyed_by_data: &[&str], out: &mut BTreeSet<String>) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, child) in map {
                    let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                    out.insert(p.clone());
                    if keyed_by_data.iter().any(|suffix| p.ends_with(suffix)) {
                        out.insert(format!("{p}.<key>"));
                        continue;
                    }
                    walk(child, &p, keyed_by_data, out);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, &format!("{path}[]"), keyed_by_data, out);
                }
            }
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    walk(doc, "", keyed_by_data, &mut out);
    out.into_iter().collect()
}

/// The fields a `.fields` file pins (one path per line; blank lines and `#` comments skipped), each with `prefix`
/// taken off: the paths [`field_paths`] gives of the object the prefix names. A line without the prefix is an error in
/// the file.
pub fn pinned(fields: &str, prefix: &str) -> Vec<String> {
    let mut paths: Vec<String> = fields
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.strip_prefix(prefix).unwrap_or_else(|| panic!("the pinned field {l} is not under {prefix}")).to_string())
        .collect();
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_read_arrays_through_brackets_and_maps_keyed_by_data_as_one_field() {
        let doc = serde_json::json!({"a": 1, "list": [{"x": 1}, {"y": {"z": 2}}], "counts": {"oom": 3, "dry": 1}});
        assert_eq!(field_paths(&doc, &["counts"]), ["a", "counts", "counts.<key>", "list", "list[].x", "list[].y", "list[].y.z"]);
    }

    #[test]
    fn a_fields_file_pins_its_lines_under_a_prefix() {
        assert_eq!(pinned("# a comment\n\nreport.b\nreport.a\n", "report."), ["a", "b"]);
    }
}
