// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding for clients serving
// large models on modest hardware. If your team needs expertise in LLM
// serving throughput then you can procure our services by sending an email
// to info@swedishembedded.com.

//! The one primitive an attention-only decoder needs to speculate: score
//! several consecutive rows of a sequence in one forward. The scheduler's
//! verify ([`PagedDecoder::forward_batched_verify`]) is exactly that, and an
//! attention-only decoder has nothing to restore on rejection (the scheduler
//! truncates the block table, which drops the rejected rows' K/V), so its
//! `forward_batched_verify` forwards to [`RowsForward::forward_rows`] and its
//! `commit_verified` stays the default no-op. Decoders with recurrent state
//! need a snapshot/restore step on top of this; that is not provided here.

use super::PagedDecoder;
use crate::paged::BlockTable;

/// Proposals per sequence an attention-only decoder verifies by default.
pub const DEFAULT_MAX_DRAFT: usize = 4;

/// A decoder that can forward a run of rows of one sequence at once.
pub trait RowsForward: PagedDecoder {
    /// Forward `rows[i]` at the next `rows[i].len()` positions of `tables[i]`
    /// (appending those positions to the table) and return, per sequence and
    /// row, the top `cands` `(token id, logit)` of the distribution after that
    /// row, best first. Each row's result must equal what a plain decode step
    /// at that position returns, so greedy speculation is output-identical.
    fn forward_rows(&mut self, tables: &mut [&mut BlockTable], rows: &[Vec<u32>], cands: usize) -> Vec<Vec<Vec<(u32, f32)>>>;
}
