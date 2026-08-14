// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! NDJSON step streaming: one [`TraceStep`] per line, each line an
//! independently-valid standalone JSON object. Intended for the
//! `sven | sven` piping use case, where a consumer can start parsing steps
//! before the producer has finished emitting them.

use std::io::{BufRead, Write};

use thiserror::Error;

use crate::model::TraceStep;

/// Errors from NDJSON step (de)serialization.
#[derive(Debug, Error)]
pub enum NdjsonError {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A step failed to serialize while writing.
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    /// A line failed to parse as a [`TraceStep`]; `line` is the 1-indexed
    /// line number so a producer can find and fix the offending record.
    #[error("line {line}: failed to parse step: {source}")]
    Parse {
        /// 1-indexed line number of the malformed record.
        line: usize,
        /// The underlying JSON parse error.
        #[source]
        source: serde_json::Error,
    },
}

/// Write `steps` to `writer` as NDJSON: one compact, standalone JSON object
/// per line.
pub fn write_steps_ndjson<W: Write>(mut writer: W, steps: &[TraceStep]) -> Result<(), NdjsonError> {
    for step in steps {
        let line = serde_json::to_string(step)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Read a stream of NDJSON step lines back into an ordered `Vec<TraceStep>`.
/// Blank/whitespace-only lines (including a trailing blank line) are
/// skipped rather than treated as errors. A malformed line fails with
/// [`NdjsonError::Parse`] naming its 1-indexed line number.
pub fn read_steps_ndjson<R: BufRead>(reader: R) -> Result<Vec<TraceStep>, NdjsonError> {
    let mut steps = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let step: TraceStep = serde_json::from_str(&line).map_err(|source| NdjsonError::Parse {
            line: index + 1,
            source,
        })?;
        steps.push(step);
    }
    Ok(steps)
}
