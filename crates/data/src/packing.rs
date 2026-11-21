// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on packed rows - several documents per
// row with no attention, state or supervision crossing between them - for its clients. If your team needs expertise
// in efficient and exact training data pipelines then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Packed training rows: several whole documents laid out one after another in a row, each token tagged with its
//! document (segment) id.
//!
//! The contract a packed row carries, and that [`document_positions`] validates: a batch is `rows` rows of
//! `row_len` tokens; within a row the segment ids never decrease, and a new document starts wherever the id changes
//! (and at every row start). A model that reads packed rows resets everything that carries information along the
//! sequence at each document start - attention never reaches an earlier document, a convolution's state restarts,
//! positions count from 0 again - so a packed document computes exactly what it computes alone.

use std::fmt;

/// A per-token segment array that is not a valid packed batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackingError {
    /// The segment array is not a whole number of rows (or a row has no tokens).
    Shape { tokens: usize, row_len: usize },
    /// A segment id falls inside a row: a document cannot resume after another began.
    Decreasing { row: usize, at: usize, previous: u32, id: u32 },
}

impl fmt::Display for PackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackingError::Shape { tokens, row_len } => write!(f, "packed rows: {tokens} segment ids are not a whole number of rows of {row_len} tokens"),
            PackingError::Decreasing { row, at, previous, id } => {
                write!(f, "packed rows: row {row} token {at} has segment id {id} after {previous}; segment ids never decrease within a row (a document is one contiguous run)")
            }
        }
    }
}

impl std::error::Error for PackingError {}

/// Each token's position inside its document - 0 where a document starts (a change of segment id, or a row start),
/// one more on each following token - after checking `segments` is a valid packed batch of rows of `row_len` tokens.
pub fn document_positions(segments: &[u32], row_len: usize) -> Result<Vec<u32>, PackingError> {
    if row_len == 0 || !segments.len().is_multiple_of(row_len) {
        return Err(PackingError::Shape { tokens: segments.len(), row_len });
    }
    let mut positions = Vec::with_capacity(segments.len());
    for (row, ids) in segments.chunks(row_len).enumerate() {
        let mut pos = 0u32;
        for (at, &id) in ids.iter().enumerate() {
            if at > 0 {
                let previous = ids[at - 1];
                if id < previous {
                    return Err(PackingError::Decreasing { row, at, previous, id });
                }
                pos = if id == previous { pos + 1 } else { 0 };
            }
            positions.push(pos);
        }
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_restart_at_every_document_and_row() {
        let segments = [3, 3, 3, 4, 4, 9, /* row 2 */ 9, 9, 10, 11, 11, 11];
        assert_eq!(document_positions(&segments, 6).unwrap(), vec![0, 1, 2, 0, 1, 0, 0, 1, 0, 0, 1, 2]);
    }

    #[test]
    fn a_malformed_batch_is_refused_by_name() {
        assert_eq!(document_positions(&[0, 0, 1], 2), Err(PackingError::Shape { tokens: 3, row_len: 2 }));
        assert_eq!(document_positions(&[], 0), Err(PackingError::Shape { tokens: 0, row_len: 0 }));
        assert_eq!(document_positions(&[0, 1, 0, 2], 4), Err(PackingError::Decreasing { row: 0, at: 2, previous: 1, id: 0 }));
    }
}
