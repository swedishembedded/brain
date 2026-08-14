// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Header-only quick read, adapted from `sven-input`'s `chat_document.rs`
//! `ChatDocumentHeader` trick (find the marker before the large array, parse
//! only what precedes it) - for JSON instead of YAML.
//!
//! [`crate::model::Trajectory`] deliberately declares `subagent_trajectories`
//! and `steps` as its last Rust struct fields, and `serde_json` preserves
//! struct declaration order when serializing, so on any file this crate
//! wrote, those two keys come last. [`read_trajectory_header_fast`] exploits
//! that: it **streams** the file in chunks, scanning for the byte offset of
//! the first top-level `"subagent_trajectories"` or `"steps"` key (tracking
//! brace/bracket depth and string escaping so it can't be fooled by either
//! name appearing inside a nested `extra` value), stops reading there,
//! closes the prefix into a small valid JSON object, and deserializes only
//! that into [`TrajectoryHeader`] - never reading, let alone parsing, the
//! (possibly huge) step or subagent arrays.
//!
//! [`read_trajectory_header`] is the defensive public entry point: it tries
//! the fast path first and falls back to a full [`crate::model::Trajectory`]
//! parse (reduced to a header) on *any* failure - a foreign file where the
//! big arrays aren't last, a truncated file, whatever.

use std::fs;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::model::{AgentProfile, FinalMetrics, Trajectory};

/// Errors from the header-only reader.
#[derive(Debug, Error)]
pub enum HeaderReadError {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization failure.
    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),
    /// The fast-path heuristic could not locate a usable split point (e.g.
    /// neither big-array key is present at the top level, or the file is too
    /// short/odd to contain one).
    #[error("could not locate a top-level \"steps\"/\"subagent_trajectories\" key to split on")]
    NoSplitPoint,
}

/// Everything in a [`crate::model::Trajectory`] except `steps` and
/// `subagent_trajectories` - cheap to obtain even from a huge trajectory
/// file, for listing many trajectories without deserializing their step
/// arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryHeader {
    /// See [`Trajectory::schema_version`].
    pub schema_version: String,
    /// See [`Trajectory::session_id`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// See [`Trajectory::trajectory_id`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    /// See [`Trajectory::agent`].
    pub agent: AgentProfile,
    /// See [`Trajectory::notes`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// See [`Trajectory::final_metrics`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_metrics: Option<FinalMetrics>,
    /// See [`Trajectory::continued_trajectory_ref`] - header data: it is how
    /// a continuation segment is discovered without parsing any steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continued_trajectory_ref: Option<String>,
    /// See [`Trajectory::extra`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl From<&Trajectory> for TrajectoryHeader {
    fn from(t: &Trajectory) -> Self {
        Self {
            schema_version: t.schema_version.clone(),
            session_id: t.session_id.clone(),
            trajectory_id: t.trajectory_id.clone(),
            agent: t.agent.clone(),
            notes: t.notes.clone(),
            final_metrics: t.final_metrics.clone(),
            continued_trajectory_ref: t.continued_trajectory_ref.clone(),
            extra: t.extra.clone(),
        }
    }
}

/// The top-level keys that begin the "big arrays" tail of a trajectory
/// document; the header is everything before the first of them.
const SPLIT_KEYS: [&[u8]; 2] = [b"\"subagent_trajectories\"", b"\"steps\""];

/// Length of the longest split key - the scanner refuses to decide on a
/// depth-1 quote closer than this to the end of the buffered data unless the
/// file is at EOF, so a key can never be missed across a chunk boundary.
const MAX_KEY_LEN: usize = 23; // "subagent_trajectories" plus both quotes

enum ScanOutcome {
    /// Byte offset of the comma immediately preceding the first big-array
    /// key: the header is everything before it.
    Found(usize),
    /// Ran out of buffered bytes without a verdict; feed more data.
    NeedMore,
    /// The whole input was scanned and no usable split point exists.
    NotFound,
}

/// Incremental depth/string-tracking scanner for the split point. Feed it a
/// growing buffer (the file prefix read so far); it remembers where it
/// stopped, so each byte is examined once.
#[derive(Default)]
struct SplitScanner {
    depth: i32,
    in_string: bool,
    escape: bool,
    last_top_level_comma: Option<usize>,
    pos: usize,
}

impl SplitScanner {
    fn scan(&mut self, buf: &[u8], eof: bool) -> ScanOutcome {
        while self.pos < buf.len() {
            let i = self.pos;
            let b = buf[i];
            if self.in_string {
                if self.escape {
                    self.escape = false;
                } else if b == b'\\' {
                    self.escape = true;
                } else if b == b'"' {
                    self.in_string = false;
                }
                self.pos += 1;
                continue;
            }
            match b {
                b'"' => {
                    if self.depth == 1 {
                        let rest = &buf[i..];
                        if rest.len() < MAX_KEY_LEN && !eof {
                            // A split key could straddle the chunk boundary:
                            // don't decide until more bytes arrive.
                            return ScanOutcome::NeedMore;
                        }
                        if SPLIT_KEYS.iter().any(|k| rest.starts_with(k)) {
                            // No preceding comma would mean the big array is
                            // the very first key - impossible for a valid
                            // Trajectory (schema_version comes first), so
                            // treat it as "no usable split point".
                            return match self.last_top_level_comma {
                                Some(comma) => ScanOutcome::Found(comma),
                                None => ScanOutcome::NotFound,
                            };
                        }
                    }
                    self.in_string = true;
                }
                b'{' | b'[' => self.depth += 1,
                b'}' | b']' => self.depth -= 1,
                b',' if self.depth == 1 => self.last_top_level_comma = Some(i),
                _ => {}
            }
            self.pos += 1;
        }
        if eof {
            ScanOutcome::NotFound
        } else {
            ScanOutcome::NeedMore
        }
    }
}

/// The strict fast path: **stream** the file until the first top-level
/// `"subagent_trajectories"`/`"steps"` key, truncate before it, close the
/// object, and parse only that prefix into a [`TrajectoryHeader`] - the big
/// arrays are never read from disk, let alone parsed. Returns
/// [`HeaderReadError`] (never panics) if the heuristic can't find a usable
/// split point or the prefix doesn't parse - callers that want automatic
/// fallback should use [`read_trajectory_header`] instead.
pub fn read_trajectory_header_fast(path: &Path) -> Result<TrajectoryHeader, HeaderReadError> {
    const CHUNK: usize = 64 * 1024;

    let mut file = fs::File::open(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(CHUNK);
    let mut scanner = SplitScanner::default();
    let mut chunk = vec![0u8; CHUNK];
    let split_at = loop {
        let n = file.read(&mut chunk)?;
        buf.extend_from_slice(&chunk[..n]);
        let eof = n == 0;
        match scanner.scan(&buf, eof) {
            ScanOutcome::Found(comma) => break comma,
            ScanOutcome::NeedMore => continue,
            ScanOutcome::NotFound => return Err(HeaderReadError::NoSplitPoint),
        }
    };

    buf.truncate(split_at);
    buf.push(b'}');
    let header: TrajectoryHeader = serde_json::from_slice(&buf)?;
    Ok(header)
}

/// Defensive public entry point: try [`read_trajectory_header_fast`] first
/// (which reads only the header prefix); on any failure, fall back to one
/// full read + full [`Trajectory`] parse reduced to a [`TrajectoryHeader`].
/// Only errors if *both* the fast path and the full parse fail (e.g. a
/// genuinely truncated/corrupt file).
pub fn read_trajectory_header(path: &Path) -> Result<TrajectoryHeader, HeaderReadError> {
    if let Ok(header) = read_trajectory_header_fast(path) {
        return Ok(header);
    }
    let content = fs::read_to_string(path)?;
    let full: Trajectory = serde_json::from_str(&content)?;
    Ok(TrajectoryHeader::from(&full))
}
