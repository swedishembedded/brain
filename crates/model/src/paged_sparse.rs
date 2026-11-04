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

/// A packed (quantised) KV pool the per-KV-head kernel can read: four bytes
/// to a word and one f32 scale per `(token slot, kv head)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedKv {
    /// Signed int8 bytes (`paged_kv_append_i8_clipped_batched`).
    Int8,
    /// FP8 E4M3 bytes (`paged_kv_append_fp8_clipped_batched`).
    Fp8,
}

/// The per-KV-head kernel over a packed pool, specialised for `group` query
/// heads per KV head: [`gqa_kernel`]'s arithmetic after a dequantising stage,
/// a quarter of its bytes. Bind the pools' scales ([`SparseBuffers::scales`]).
pub fn gqa_kernel_kv(group: u32, kv: PackedKv) -> Result<(&'static str, &'static str), String> {
    if group == 0 {
        return Err("sparse attention: a GQA group of 0 query heads".into());
    }
    let int8 = kernels::template::interned("paged_sparse_flash_gqa_i8", kernels::PAGED_SPARSE_FLASH_GQA_I8, &[("GROUP", group)])?;
    match kv {
        PackedKv::Int8 => Ok(int8),
        PackedKv::Fp8 => kernels::template::fp8_decode_variant(int8.0, int8.1),
    }
}

/// How a sparse kernel splits a forward over workgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// [`KERNEL`]: one workgroup per (row, query head).
    PerHead,
    /// [`gqa_kernel`] / [`gqa_kernel_kv`]: one per (row, KV head).
    PerKvHead,
    /// [`rows_kernel_kv`]: one per row, every head.
    PerRow,
}

/// The whole-row kernel over a packed pool, specialised for the model's
/// geometry (`n_heads` must divide 64, `head_dim % 4 == 0`): each selected
/// key is resolved and its full K/V row staged once per query row and read
/// by every head - the prefill-shaped read. Bind the scales.
pub fn rows_kernel_kv(n_heads: u32, n_kv_heads: u32, head_dim: u32, kv: PackedKv) -> Result<(&'static str, &'static str), String> {
    if n_heads == 0 || n_kv_heads == 0 || !64u32.is_multiple_of(n_heads) || !n_heads.is_multiple_of(n_kv_heads) || head_dim == 0 || !head_dim.is_multiple_of(4) || !(head_dim / (64 / n_heads)).is_multiple_of(4) {
        return Err(format!("sparse attention: no whole-row kernel for {n_heads} heads over {n_kv_heads} KV heads of {head_dim}"));
    }
    let int8 = kernels::template::interned("paged_sparse_flash_rows_i8", kernels::PAGED_SPARSE_FLASH_ROWS_I8, &[("HEADS", n_heads), ("KVH", n_kv_heads), ("HD", head_dim)])?;
    match kv {
        PackedKv::Int8 => Ok(int8),
        PackedKv::Fp8 => kernels::template::fp8_decode_variant(int8.0, int8.1),
    }
}

/// Widest head the kernel's register tile holds (its `HD`).
pub const MAX_HEAD_DIM: u32 = 128;

/// Words of a selection row's header: selected count, block-table row,
/// live keys of the row's sequence. The positions follow.
pub const ROW_HEADER: u32 = 3;

/// A selection packed for the kernel: `sel` is `[rows, stride]`, each row
/// its [`ROW_HEADER`] then its logical positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseRows {
    /// Query rows.
    pub rows: u32,
    /// Row stride of `sel`, header included.
    pub stride: u32,
    /// Headers and selected positions; padding is never read.
    pub sel: Vec<u32>,
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
        let stride = u32::try_from(selection.width + ROW_HEADER as usize).map_err(|_| format!("sparse attention: selection width {} overflows u32", selection.width))?;
        let mut sel = vec![0u32; rows * stride as usize];
        for r in 0..rows {
            let ids = selection.row(r);
            let row = &mut sel[r * stride as usize..][..stride as usize];
            row[..3].copy_from_slice(&[ids.len() as u32, table_rows[r], live[r]]);
            for (dst, &j) in row[3..].iter_mut().zip(ids) {
                if j < 0 || j as u32 >= live[r] {
                    return Err(format!("sparse attention: row {r} selects position {j} of a sequence holding {} keys", live[r]));
                }
                *dst = j as u32;
            }
        }
        Ok(SparseRows { rows: rows as u32, stride, sel })
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
    /// `(scales_k, scales_v)`, `[num_blocks * block_size, n_kv_heads]` f32,
    /// for a packed pool read by [`gqa_kernel_kv`]; `None` for f32/bf16 pools.
    pub scales: Option<(&'a DeviceBuffer, &'a DeviceBuffer)>,
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

    /// Record one dispatch over `rows` query rows of selection row stride
    /// `stride` (header included). `kernel` is the caller's pipeline index of
    /// a kernel of `layout` for this geometry (possibly a bf16-pool or packed
    /// variant).
    pub fn step(&self, g: &Gpu, kernel: usize, layout: Layout, rows: u32, stride: u32, b: &SparseBuffers) -> Result<Step, String> {
        self.check()?;
        let params = [rows, self.n_heads, self.n_kv_heads, self.head_dim, self.n_heads / self.n_kv_heads, self.block_size, self.max_bt, stride];
        let groups = match layout {
            Layout::PerHead => self.n_heads,
            Layout::PerKvHead => self.n_kv_heads,
            Layout::PerRow => 1,
        };
        let grid = gpu_core::Dispatch::Workgroups(rows * groups);
        Ok(match b.scales {
            None if layout == Layout::PerRow => return Err("sparse attention: the whole-row kernel reads a packed pool".into()),
            None => g.dispatch(kernel, &[b.q, b.pool_k, b.pool_v, b.block_tables, b.sel, b.ctx], &params, grid),
            Some((sk, sv)) => {
                if layout == Layout::PerHead || !self.head_dim.is_multiple_of(4) {
                    return Err("sparse attention over a packed pool needs a per-KV-head or whole-row kernel and head_dim % 4 == 0".into());
                }
                g.dispatch(kernel, &[b.q, b.pool_k, b.pool_v, sk, sv, b.block_tables, b.sel, b.ctx], &params, grid)
            }
        })
    }
}

#[cfg(test)]
mod tests;
