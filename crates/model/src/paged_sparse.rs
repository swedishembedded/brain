// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse attention over the paged KV cache: each query row reads an explicit
//! list of logical key positions (`paged_sparse_flash_decode.wgsl`).
//!
//! Swedish Embedded AB implements attention whose per-query cost is set by a
//! key budget instead of the context length, for clients serving contexts far
//! beyond what dense attention affords. If your team needs expertise in sparse
//! attention on paged KV caches, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The selection comes from any indexer - [`crate::lsa::select`] is one - as a
//! [`Selection`]; [`SparseRows::pack`] turns it into the device rows the kernel
//! reads, refusing a position the sequence does not hold yet, and
//! [`SparseAttention::step`] records the dispatch. K/V stay where the paged
//! cache put them: a position resolves through its sequence's block table
//! inside the kernel, so nothing is gathered into a contiguous copy.

use crate::lsa::Selection;
use gpu_core::{DeviceBuffer, Gpu, Step};

/// The per-query-head kernel, for a caller's pipeline list.
pub const KERNEL: (&str, &str) = ("paged_sparse_flash_decode", kernels::PAGED_SPARSE_FLASH_DECODE);

/// The per-KV-head kernel specialised for `group` query heads per KV head:
/// each staged K/V row serves the whole group, so K/V traffic falls by
/// `group`. Same bindings and parameters as [`KERNEL`], bit-identical result.
pub fn gqa_kernel(group: u32) -> Result<(&'static str, &'static str), String> {
    if group == 0 {
        return Err("sparse attention: a GQA group of 0 query heads".into());
    }
    kernels::template::interned("paged_sparse_flash_gqa", kernels::PAGED_SPARSE_FLASH_GQA, &[("GROUP", group)])
}

/// Widest head the kernel's register tile holds (its `HD`).
pub const MAX_HEAD_DIM: u32 = 128;

/// A selection packed for the kernel: `sel` is `[rows, width]` positions and
/// `row_meta` is `[rows, 3]` of (selected count, block-table row, live keys).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseRows {
    /// Query rows.
    pub rows: u32,
    /// Row stride of `sel`.
    pub width: u32,
    /// Selected positions, `[rows, width]`; padding is never read.
    pub sel: Vec<u32>,
    /// Per row: selected count, block-table row, live keys of its sequence.
    pub row_meta: Vec<u32>,
}

impl SparseRows {
    /// Pack `selection` for sequences whose block tables are rows
    /// `table_rows[r]` and whose caches hold `live[r]` keys.
    ///
    /// A selected position at or past `live[r]` is refused here; the kernel
    /// also masks one, so a selection that went stale after packing still
    /// cannot read a block the sequence does not own.
    pub fn pack(selection: &Selection, table_rows: &[u32], live: &[u32]) -> Result<SparseRows, String> {
        let rows = selection.rows();
        if table_rows.len() != rows || live.len() != rows {
            return Err(format!(
                "sparse attention: {rows} selected rows but {} table rows and {} live lengths",
                table_rows.len(),
                live.len()
            ));
        }
        let width = u32::try_from(selection.width).map_err(|_| format!("sparse attention: selection width {} overflows u32", selection.width))?;
        let mut sel = vec![0u32; rows * selection.width];
        let mut row_meta = Vec::with_capacity(rows * 3);
        for r in 0..rows {
            let ids = selection.row(r);
            for (dst, &j) in sel[r * selection.width..].iter_mut().zip(ids) {
                if j < 0 || j as u32 >= live[r] {
                    return Err(format!("sparse attention: row {r} selects position {j} of a sequence holding {} keys", live[r]));
                }
                *dst = j as u32;
            }
            row_meta.extend_from_slice(&[ids.len() as u32, table_rows[r], live[r]]);
        }
        Ok(SparseRows { rows: rows as u32, width, sel, row_meta })
    }
}

/// Head and cache geometry of one sparse attention dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseAttention {
    /// Query heads.
    pub n_heads: u32,
    /// Key/value heads; divides `n_heads`.
    pub n_kv_heads: u32,
    /// Channels per head, at most [`MAX_HEAD_DIM`].
    pub head_dim: u32,
    /// Keys per KV block.
    pub block_size: u32,
    /// Row stride of the block tables, in blocks.
    pub max_bt: u32,
}

/// The buffers one dispatch binds.
pub struct SparseBuffers<'a> {
    /// Queries, `[rows, n_heads * head_dim]`.
    pub q: &'a DeviceBuffer,
    /// Key pool, `[num_blocks * block_size, n_kv_heads * head_dim]`.
    pub pool_k: &'a DeviceBuffer,
    /// Value pool, same layout as `pool_k`.
    pub pool_v: &'a DeviceBuffer,
    /// Block tables, `[n_tables, max_bt]`.
    pub block_tables: &'a DeviceBuffer,
    /// [`SparseRows::sel`] on the device.
    pub sel: &'a DeviceBuffer,
    /// [`SparseRows::row_meta`] on the device.
    pub row_meta: &'a DeviceBuffer,
    /// Output context, `[rows, n_heads * head_dim]`.
    pub ctx: &'a DeviceBuffer,
}

impl SparseAttention {
    /// Refuse a geometry the kernel cannot run.
    pub fn check(&self) -> Result<(), String> {
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!("sparse attention: {} query heads do not group over {} KV heads", self.n_heads, self.n_kv_heads));
        }
        if self.head_dim == 0 || self.head_dim > MAX_HEAD_DIM {
            return Err(format!("sparse attention: head_dim {} is outside 1..={MAX_HEAD_DIM}", self.head_dim));
        }
        if self.block_size == 0 || self.max_bt == 0 {
            return Err("sparse attention: block_size and max_bt must be positive".into());
        }
        Ok(())
    }

    /// Record one dispatch over `rows` query rows of selection width `width`.
    /// `kernel` is the caller's pipeline index of [`KERNEL`] (`per_kv_head`
    /// false) or of [`gqa_kernel`] for this geometry's group (true), either
    /// possibly a bf16-pool variant.
    pub fn step(&self, g: &Gpu, kernel: usize, per_kv_head: bool, rows: u32, width: u32, b: &SparseBuffers) -> Result<Step, String> {
        self.check()?;
        let params = [rows, self.n_heads, self.n_kv_heads, self.head_dim, self.n_heads / self.n_kv_heads, self.block_size, self.max_bt, width];
        let bufs = [b.q, b.pool_k, b.pool_v, b.block_tables, b.sel, b.row_meta, b.ctx];
        let groups = if per_kv_head { self.n_kv_heads } else { self.n_heads };
        Ok(g.dispatch(kernel, &bufs, &params, gpu_core::Dispatch::Workgroups(rows * groups)))
    }
}

#[cfg(test)]
mod tests;
