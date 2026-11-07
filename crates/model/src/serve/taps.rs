// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters that read a
// target's internals, verifiers and the state rollback a hybrid stack needs -
// for clients serving large models on modest hardware. If your team needs
// expertise in LLM serving throughput then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Hidden-state taps: the residual stream of a target at chosen layers, kept per cached position.
//!
//! A drafter that reads the target's internals (DFlash2 attends to the target's hidden states at a handful of its
//! layers) needs those rows for every position the target has consumed: the prompt, every decoded token and every
//! verify row. A decoder that can supply them implements [`PagedDecoder::enable_taps`] and
//! [`PagedDecoder::read_taps`]; this module is the storage it can use.
//!
//! # Where a tap row lives
//!
//! In a [`TapPool`] addressed exactly like the decoder's KV: by the physical block of a position in the sequence's
//! [`BlockTable`] and its offset in that block. That one choice settles everything the engine would otherwise have to
//! handle position by position:
//!
//! * **Rollback.** A rejected verify row's block-table entry is truncated away by the scheduler and the next write at
//!   that position replaces the row; nothing to undo.
//! * **Prefix reuse.** A cached prefix is its blocks, shared by reference; the tap rows of those blocks are shared with
//!   them, and the copy-on-write copy of a partly filled tail block ([`TapPool::copy_rows`]) carries them over.
//! * **Memory.** The pool is sized from the same block count as the KV pool ([`TapPool::bytes`]), so the placement
//!   charges it before anything loads.
//!
//! Rows are held as bf16 on the host: the drafter they feed was trained on bf16 hidden states, a row is
//! `layers x hidden` wide (100 KiB at five taps of 5120 in f32), and a verify moves a few rows a step.

use crate::half::f32_to_bf16;
use crate::paged::BlockTable;

/// Per-position storage of the residual stream at a fixed set of layers.
pub struct TapPool {
    layers: Vec<usize>,
    hidden: usize,
    block_size: u32,
    /// `[num_blocks * block_size, width]` bf16 bits.
    rows: Vec<u16>,
}

impl TapPool {
    /// A pool of `num_blocks` blocks of `block_size` positions retaining the residual of `layers`, each `hidden`
    /// wide.
    pub fn new(layers: Vec<usize>, hidden: usize, num_blocks: u32, block_size: u32) -> TapPool {
        let width = layers.len() * hidden;
        TapPool { layers, hidden, block_size, rows: vec![0; num_blocks as usize * block_size as usize * width] }
    }

    /// Host bytes a pool of this shape holds.
    pub fn bytes(n_layers: usize, hidden: usize, num_blocks: u32, block_size: u32) -> u64 {
        u64::from(num_blocks) * u64::from(block_size) * (n_layers * hidden) as u64 * 2
    }

    /// The tapped layers, in row order.
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// Values in one position's row: `layers x hidden`.
    pub fn width(&self) -> usize {
        self.layers.len() * self.hidden
    }

    fn at(&self, block: u32, offset: u32) -> usize {
        (block as usize * self.block_size as usize + offset as usize) * self.width()
    }

    /// Store the row of position `pos` of the sequence whose blocks are `blocks`; `row` is the tapped layers'
    /// residual concatenated in [`Self::layers`] order.
    pub fn write(&mut self, blocks: &[u32], pos: u32, row: &[f32]) {
        assert_eq!(row.len(), self.width(), "a tap row is {} values, got {}", self.width(), row.len());
        let (block, offset) = (blocks[(pos / self.block_size) as usize], pos % self.block_size);
        let at = self.at(block, offset);
        for (dst, &v) in self.rows[at..at + row.len()].iter_mut().zip(row) {
            *dst = f32_to_bf16(v);
        }
    }

    /// The rows of positions `from .. from + count` of `table`, `[count, width]` as f32.
    pub fn read(&self, table: &BlockTable, from: u32, count: u32) -> Result<Vec<f32>, String> {
        if from.checked_add(count).is_none_or(|end| end > table.len()) {
            return Err(format!("taps: positions {from}..{} of a sequence of {} cached positions", u64::from(from) + u64::from(count), table.len()));
        }
        let width = self.width();
        let mut out = Vec::with_capacity(count as usize * width);
        for pos in from..from + count {
            let (block, offset) = table.locate(pos, self.block_size);
            let at = self.at(block, offset);
            out.extend(self.rows[at..at + width].iter().map(|&b| f32::from_bits(u32::from(b) << 16)));
        }
        Ok(out)
    }

    /// Copy the first `rows` positions of block `from` over block `to` (the copy-on-write of a shared tail block).
    pub fn copy_rows(&mut self, from: u32, to: u32, rows: u32) {
        let n = rows as usize * self.width();
        let (src, dst) = (self.at(from, 0), self.at(to, 0));
        self.rows.copy_within(src..src + n, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged::BlockAllocator;

    #[test]
    fn a_row_reads_back_through_the_block_table_rounded_to_bf16() {
        let mut alloc = BlockAllocator::new(8, 4);
        let mut table = BlockTable::new();
        table.reserve(10, &mut alloc).unwrap();
        let mut pool = TapPool::new(vec![1, 5], 3, 8, 4);
        assert_eq!(pool.width(), 6);
        for pos in 0..10u32 {
            let row: Vec<f32> = (0..6).map(|i| pos as f32 + i as f32 * 0.5 + 0.1).collect();
            pool.write(table.blocks(), pos, &row);
        }
        let got = pool.read(&table, 3, 4).unwrap();
        for (j, chunk) in got.chunks(6).enumerate() {
            for (i, &v) in chunk.iter().enumerate() {
                let want = f32::from_bits(u32::from(f32_to_bf16((3 + j) as f32 + i as f32 * 0.5 + 0.1)) << 16);
                assert_eq!(v, want, "position {} value {i}", 3 + j);
            }
        }
        assert!(pool.read(&table, 8, 3).is_err(), "a read past the cached positions is refused, never zeros");
    }

    #[test]
    fn a_copied_tail_block_carries_its_rows() {
        let mut alloc = BlockAllocator::new(4, 4);
        let mut table = BlockTable::new();
        table.reserve(6, &mut alloc).unwrap();
        let mut pool = TapPool::new(vec![0], 2, 4, 4);
        for pos in 0..6u32 {
            pool.write(table.blocks(), pos, &[pos as f32, -(pos as f32)]);
        }
        let (from, to) = (table.blocks()[1], 3);
        pool.copy_rows(from, to, 2);
        let mut copy = BlockTable::new();
        copy.adopt_rows(&[table.blocks()[0], to], 6, &mut alloc);
        assert_eq!(pool.read(&copy, 4, 2).unwrap(), vec![4.0, -4.0, 5.0, -5.0]);
    }
}
