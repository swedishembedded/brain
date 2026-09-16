// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A minimal RFC 4180 reader, because the fields we need are quoted.
//!
//! Hand-rolled rather than pulled in, which is this workspace's habit for
//! small parsers (`crates/capture`'s V4L2 ioctls, `crates/wm-display`'s SDL2,
//! `crates/onnx`'s protobuf). The reason it cannot be a `split(',')` is
//! concrete: a Codex `neurons.csv` community-label field looks like
//!
//! ```text
//! 10000,LTCT,"description::Giant fiber,group::10000,synonyms::GF, Giant Fiber",ACH,0.51
//! ```
//!
//! Splitting that on commas silently shifts every later column left by four,
//! so a neuron's neurotransmitter would be read out of its cell-type field
//! and the import would succeed with plausible nonsense.

/// Split one CSV line into fields, honouring double quotes and `""` escapes.
///
/// Returns fields with quotes removed. A quoted field may contain commas and
/// escaped quotes; an unterminated quote consumes the rest of the line rather
/// than erroring, which is the same thing every reader does and keeps this
/// infallible for a caller that is already checking field COUNT.
pub fn split_line(line: &str, out: &mut Vec<String>) {
    out.clear();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    // "" inside a quoted field is one literal quote.
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => out.push(std::mem::take(&mut field)),
            _ => field.push(c),
        }
    }
    out.push(field);
}

/// Map a header row to column indices, so a schema change moves a column
/// rather than corrupting a field. Every reader here looks columns up by NAME.
pub struct Header {
    names: Vec<String>,
}

impl Header {
    pub fn new(line: &str) -> Header {
        let mut v = Vec::new();
        split_line(line, &mut v);
        // A UTF-8 BOM on the first header cell would otherwise make the first
        // column unfindable by name, and only the first column.
        if let Some(first) = v.first_mut() {
            *first = first.trim_start_matches('\u{feff}').to_string();
        }
        Header { names: v }
    }

    /// Index of a required column, or an error naming what was actually there.
    pub fn need(&self, name: &str) -> Result<usize, String> {
        self.names.iter().position(|n| n == name).ok_or_else(|| {
            format!("column {name:?} not found; header is [{}]", self.names.join(", "))
        })
    }

    /// Index of an optional column. `None` is a legitimate answer: Janelia's
    /// MANC export has no verified-neurotransmitter column at all, while
    /// BANC's does.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quoted_field_keeps_its_commas() {
        let mut v = Vec::new();
        split_line(r#"10000,LTCT,"description::Giant fiber,group::10000",ACH,0.51"#, &mut v);
        assert_eq!(v, ["10000", "LTCT", "description::Giant fiber,group::10000", "ACH", "0.51"]);
    }

    #[test]
    fn escaped_quotes_survive() {
        let mut v = Vec::new();
        split_line(r#"a,"he said ""hi""",b"#, &mut v);
        assert_eq!(v, ["a", r#"he said "hi""#, "b"]);
    }

    #[test]
    fn empty_fields_are_preserved_positionally() {
        let mut v = Vec::new();
        split_line("a,,b,", &mut v);
        assert_eq!(v, ["a", "", "b", ""], "a dropped empty field shifts every later column");
    }

    #[test]
    fn a_header_finds_columns_by_name_and_says_what_it_saw() {
        let h = Header::new("Root ID,Class,Predicted NT type");
        assert_eq!(h.need("Class").unwrap(), 1);
        assert_eq!(h.find("Verified NT type"), None);
        let err = h.need("Nerve").unwrap_err();
        assert!(err.contains("Root ID"), "the error must show the real header: {err}");
    }
}
