// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a pipeline stage leaves behind, so the next one can read it and a
//! person can check it.
//!
//! A pipeline whose stages hand each other values in memory can only be
//! debugged by running the whole thing again with a print statement in it.
//! One whose stages write their output down can be inspected at the point it
//! went wrong, resumed from there, and diffed between runs.
//!
//! One record per line, because that is the form that survives a stage
//! crashing halfway: everything written before the crash is still readable,
//! and `wc -l` says how far it got.
//!
//! Swedish Embedded AB builds data pipelines whose intermediate results are
//! artefacts rather than temporaries. If your team needs expertise in
//! machine-learning pipelines that can be audited a stage at a time, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::io::{BufRead, Write};
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{Error, Result};

/// Write `items` as one JSON record per line, replacing whatever was there.
///
/// The parent directory is created: a stage should not fail because the run
/// directory has not been laid out yet.
pub fn write_jsonl<T: Serialize>(path: impl AsRef<Path>, items: &[T]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| Error::Backend(format!("{}: {e}", parent.display())))?;
    }
    let file = std::fs::File::create(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    let mut out = std::io::BufWriter::new(file);
    for (i, item) in items.iter().enumerate() {
        let line = serde_json::to_string(item).map_err(|e| Error::Backend(format!("{}: record {i}: {e}", path.display())))?;
        writeln!(out, "{line}").map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    }
    out.flush().map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

/// Read back what [`write_jsonl`] wrote.
///
/// A malformed line is an error naming its 1-based number, never a silently
/// skipped record: a stage that quietly dropped half its input would show up
/// downstream as a weaker result rather than as a failure.
pub fn read_jsonl<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| Error::Backend(format!("{}: line {}: {e}", path.display(), i + 1)))?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .map_err(|e| Error::Backend(format!("{}: line {}: {e}", path.display(), i + 1)))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Row {
        a: String,
        b: usize,
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brain-artifact-{name}-{}", std::process::id())).join("stage.jsonl")
    }

    /// Round trip, including the directory the caller has not made yet.
    #[test]
    fn records_survive_a_write_and_a_read() {
        let p = scratch("roundtrip");
        let rows = vec![Row { a: "one".into(), b: 1 }, Row { a: "two".into(), b: 2 }];
        write_jsonl(&p, &rows).expect("write");
        assert_eq!(read_jsonl::<Row>(&p).expect("read"), rows);
        // One record per line is the contract, not an implementation detail:
        // a half-written file has to stay readable up to the crash.
        assert_eq!(std::fs::read_to_string(&p).expect("read").lines().count(), 2);
        let _ = std::fs::remove_dir_all(p.parent().expect("parent"));
    }

    /// A record that does not parse stops the stage and names the line. The
    /// alternative - skipping it - turns lost data into a weaker result
    /// downstream, which is the hardest kind of defect to find.
    #[test]
    fn a_malformed_record_is_an_error_naming_its_line() {
        let p = scratch("malformed");
        write_jsonl(&p, &[Row { a: "one".into(), b: 1 }]).expect("write");
        let mut text = std::fs::read_to_string(&p).expect("read");
        text.push_str("{ not json }\n");
        std::fs::write(&p, text).expect("write");
        let err = read_jsonl::<Row>(&p).expect_err("must refuse");
        assert!(format!("{err}").contains("line 2"), "{err}");
        let _ = std::fs::remove_dir_all(p.parent().expect("parent"));
    }
}
