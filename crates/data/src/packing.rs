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
    /// The dataset is a plain token stream: it has no example boundaries, so no document to pack.
    NoExamples,
    /// A row was asked to hold an example the dataset does not have.
    UnknownExample { row: usize, example: usize, examples: usize },
    /// A row's examples together are longer than the row.
    RowOverflow { row: usize, tokens: usize, row_len: usize },
    /// A batch was asked for with a different number of rows than its configuration.
    RowCount { rows: usize, batch_size: usize },
}

impl fmt::Display for PackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackingError::Shape { tokens, row_len } => write!(f, "packed rows: {tokens} segment ids are not a whole number of rows of {row_len} tokens"),
            PackingError::Decreasing { row, at, previous, id } => {
                write!(f, "packed rows: row {row} token {at} has segment id {id} after {previous}; segment ids never decrease within a row (a document is one contiguous run)")
            }
            PackingError::NoExamples => write!(f, "packed rows: the dataset is a plain token stream with no example boundaries; packing needs whole examples (documents)"),
            PackingError::UnknownExample { row, example, examples } => write!(f, "packed rows: row {row} names example {example} of a dataset of {examples}"),
            PackingError::RowOverflow { row, tokens, row_len } => write!(f, "packed rows: row {row} holds {tokens} tokens of examples, more than its {row_len}"),
            PackingError::RowCount { rows, batch_size } => write!(f, "packed rows: {rows} rows asked for a batch of {batch_size}"),
        }
    }
}

impl std::error::Error for PackingError {}

/// One packed batch, row-major `[rows * row_len]`: input tokens, next-token targets (`IGNORE` = -1 where nothing is
/// supervised: masked tokens, padding, and the last token of every document, whose next token is another document's),
/// and each token's document (segment) id. Documents are numbered from 0 in the order they are laid out across the
/// batch; the padding that ends a row is a document of its own, so it never reads one it does not belong to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBatch {
    pub tokens: Vec<u32>,
    pub targets: Vec<i32>,
    pub segments: Vec<u32>,
    /// The examples laid out in each row, in order.
    pub rows: Vec<Vec<usize>>,
    /// Tokens that belong to an example (the rest is padding).
    pub useful_tokens: u64,
}

/// Greedy, order-preserving packing of whole examples into rows: each row takes the next examples while they fit,
/// and the first that does not fit opens the next row (it is held, never dropped or reordered). An example is never
/// split across rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowPacker {
    pending: Option<usize>,
}

impl RowPacker {
    /// A packer that has held nothing back.
    #[must_use]
    pub fn new() -> RowPacker {
        RowPacker::default()
    }

    /// A packer that resumes holding `pending` (what [`Self::pending`] reported), so a resumed run lays out the rows an
    /// uninterrupted one would.
    #[must_use]
    pub fn resuming(pending: Option<usize>) -> RowPacker {
        RowPacker { pending }
    }

    /// The example drawn but not yet laid out: it opens the next row.
    #[must_use]
    pub fn pending(&self) -> Option<usize> {
        self.pending
    }

    /// `rows` rows of at most `row_len` tokens, filled with the examples `next` yields (in that order) whose lengths
    /// `len_of` gives. A row left empty because `next` ran out is a row of padding.
    ///
    /// # Panics
    /// When an example is empty or longer than a row (a dataset of examples refuses those when it is built).
    pub fn fill(&mut self, rows: usize, row_len: usize, len_of: impl Fn(usize) -> usize, mut next: impl FnMut() -> Option<usize>) -> Vec<Vec<usize>> {
        let mut out = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::new();
            let mut used = 0usize;
            while let Some(example) = self.pending.take().or_else(&mut next) {
                let len = len_of(example);
                assert!(len >= 1 && len <= row_len, "packed rows: example {example} of {len} tokens cannot be laid out in a row of {row_len}");
                if used + len > row_len {
                    self.pending = Some(example);
                    break;
                }
                row.push(example);
                used += len;
            }
            out.push(row);
        }
        out
    }
}

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
