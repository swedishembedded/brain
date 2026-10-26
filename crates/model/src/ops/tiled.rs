// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training over quantized
// checkpoints for its clients. If your team needs expertise in running
// forward, input-gradient and weight-gradient products over packed weights
// with tensor cores and a scratch budget that does not grow with the model,
// you can procure our services by sending an email to info@swedishembedded.com.

//! The quantized-execution contract of [`Ops`]: for EVERY [`Weight`] tier a forward product
//! `Y = X W^T`, an input gradient `dX = dY W` and a weight gradient `dW = dY^T X` in a chosen gradient
//! dtype, through one path.
//!
//! # Contract
//!
//! * A tier that is not dense (`F16`, `I8`, `Q4`, `NF4`, `KQuant`, `FP8`, `Micro`) is multiplied by
//!   decoding a TILE of its rows to f32 (`weight_decode`, one kernel for all tiers, see
//!   `kernels::template::weight_decode_variants`) and running the dense product on the tile through the
//!   same entry points an f32 weight takes: the vendor-library TF32 tensor-core GEMM on a device that has
//!   it, the portable WGSL kernels elsewhere. The decoded values equal the host decode exactly.
//! * The scratch is at most [`Ops::tile_budget_bytes`] of decoded f32 plus one `[m, tile]` product tile,
//!   chosen from the budget and `k` by [`tile_rows`], never from the size of the tensor: a weight larger
//!   than the budget is processed in several tiles of one reused scratch. Nothing here allocates a
//!   full-size copy of a quantized weight.
//! * `BF16` and `F32` are dense: their own kernels and library paths (`matmul_rows`, `matmul_dx_rows`) are the
//!   same contract with nothing to decode. NF4 is a tile-decoded tier like the rest.
//! * The weight gradient does not read the weight at all (`dW = dY^T X`); its shape comes from the weight
//!   and its dtype ([`GradDtype`]) from the caller. An fp32 gradient is one GEMM into the caller's buffer;
//!   a bf16 gradient is computed per row tile into a bounded fp32 scratch and added with round to nearest
//!   even ([`Ops::weight_grad`]).
//! * The kernel path is a function of operand formats, shape and device capabilities only
//!   ([`plan_linear`]): no model-crate conditional picks a kernel.

use gpu_core::{DeviceBuffer, Step};
use kernels::template::DecodeFormat;

use super::*;
use crate::fp8::Fp8Format;

/// Decoded-f32 scratch one tile of a quantized product may use unless the caller asks otherwise.
pub const DEFAULT_TILE_BUDGET_BYTES: u64 = 256 << 20;

/// The dtype a weight gradient is accumulated into: fp32 (one word per element) or bf16 (two per word,
/// `crate::half::pack_bf16`'s layout, accumulated with round to nearest even). One type with the training
/// precision descriptors of `paramstore`.
pub use paramstore::precision::GradDtype;

/// The padding entry of a distinct-id list: the embedding-gradient kernels skip it.
pub const UNIQ_PAD: u32 = u32::MAX;

/// The distinct values of `ids`, ascending, padded with [`UNIQ_PAD`] to `capacity` entries: the `uniq` list of an
/// [`EmbeddingGrad`] whose dispatch must keep one shape from batch to batch (a recorded tape passes `capacity` as
/// `n_uniq`). Panics when `ids` holds more distinct values than `capacity`, or holds the padding value itself.
pub fn uniq_padded(ids: &[u32], capacity: usize) -> Vec<u32> {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v.dedup();
    assert!(v.len() <= capacity, "uniq_padded: {} distinct ids for a list of {capacity}", v.len());
    assert!(v.last() != Some(&UNIQ_PAD), "uniq_padded: id {UNIQ_PAD} is the padding value, not a token");
    v.resize(capacity, UNIQ_PAD);
    v
}

/// One embedding-table gradient: `dw[uniq[u], :] += sum_{n : tokens[n] == uniq[u]} d_x[n, :]` over the `n_rows`
/// looked-up tokens, into `dw` (`[rows, width]`) stored as `grad`. `uniq` holds the `n_uniq` distinct token ids
/// (`model::block::uniq_u32` or [`uniq_padded`] of the same ids, which must all be present: a missing one's gradient
/// is dropped; [`UNIQ_PAD`] entries are skipped), so the cost is `distinct * width * n_rows` whatever the table's row
/// count.
pub struct EmbeddingGrad<'a> {
    pub tokens: &'a DeviceBuffer,
    pub uniq: &'a DeviceBuffer,
    pub n_uniq: u32,
    pub d_x: &'a DeviceBuffer,
    pub n_rows: u32,
    pub width: u32,
    pub dw: &'a DeviceBuffer,
    pub grad: GradDtype,
}

/// How a product against a weight runs, decided from formats, shapes and the device alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearPlan {
    /// The tier has a float-read kernel (or library path) of its own.
    Native,
    /// Decode `tile_rows` rows at a time to a scratch and run the dense product on each tile.
    TileDecode {
        /// Rows decoded per tile; the last tile may be shorter.
        tile_rows: u32,
        /// The scratch holds bf16 (the vendor-library tensor-core product reads it, half the bytes of
        /// f32); otherwise f32, read by the TF32 tensor-core product or the portable kernels.
        bf16: bool,
    },
}

/// The plan for a product over `rows` rows of a `k`-wide weight of `tier` under a scratch `budget_bytes`;
/// `bf16_scratch` says the device can decode this tier to bf16 for the library GEMM
/// ([`gpu_core::provider::library::LibraryGemmProvider::can_dequant_bf16`]).
///
/// `F32` and `BF16` are dense and keep their kernels and library paths. Every other tier - NF4 included -
/// decodes tiles.
pub fn plan_linear(tier: Dtype, rows: u32, k: u32, budget_bytes: u64, bf16_scratch: bool) -> LinearPlan {
    match tier {
        Dtype::F32 | Dtype::BF16 => LinearPlan::Native,
        _ => LinearPlan::TileDecode { tile_rows: tile_rows(rows, k, budget_bytes, if bf16_scratch { 2 } else { 4 }), bf16: bf16_scratch },
    }
}

/// Rows of a `k`-wide decoded tile of `elem_bytes` per element that fit `budget_bytes`, at least 1 and at
/// most `rows`.
pub fn tile_rows(rows: u32, k: u32, budget_bytes: u64, elem_bytes: u64) -> u32 {
    let fit = budget_bytes / (elem_bytes * k.max(1) as u64);
    fit.clamp(1, rows.max(1) as u64) as u32
}

impl Weight {
    /// The `weight_decode` format of this tier; `None` for `F32`, which is already dense.
    pub fn decode_format(&self) -> Option<DecodeFormat> {
        Some(match self {
            Weight::F32 { .. } => return None,
            Weight::BF16 { .. } => DecodeFormat::Bf16,
            Weight::F16 { .. } => DecodeFormat::F16,
            Weight::I8 { .. } => DecodeFormat::I8,
            Weight::Q4 { .. } => DecodeFormat::Q4,
            Weight::NF4 { .. } => DecodeFormat::Nf4,
            Weight::FP8 { fmt: Fp8Format::E4M3, .. } => DecodeFormat::Fp8E4m3,
            Weight::FP8 { fmt: Fp8Format::E5M2, .. } => DecodeFormat::Fp8E5m2,
            Weight::Micro { kind: MicroScaled::Nvfp4, .. } => DecodeFormat::Nvfp4,
            Weight::Micro { kind: MicroScaled::Mxfp4, .. } => DecodeFormat::Mxfp4,
            Weight::Micro { kind: MicroScaled::Mxfp8E4m3, .. } => DecodeFormat::Mxfp8E4m3,
            Weight::Micro { kind: MicroScaled::Mxfp8E5m2, .. } => DecodeFormat::Mxfp8E5m2,
            Weight::KQuant { affine: true, bits: 4, .. } => DecodeFormat::KqAffine4,
            Weight::KQuant { affine: true, .. } => DecodeFormat::KqAffine8,
            Weight::KQuant { affine: false, bits: 8, group: 16, .. } => DecodeFormat::KqSym8,
            Weight::KQuant { bits, group, .. } => {
                panic!("Weight::KQuant bits={bits} group={group} has no tile decode: only Q4_K, Q5_K and Q6_K layouts are defined")
            }
        })
    }
}

/// Activation rows up to which a stored FP8 or NVFP4 weight is read by its own float-read GEMV kernel (decode batches)
/// instead of the tile decode: the weight's bytes are streamed once for all of the rows.
pub const GEMV_MAX_ROWS: u32 = 8;
/// The FP8 and NVFP4 GEMVs' row cap: they are decode-bound, and past two rows the tile decode feeding the bf16
/// tensor-core product reads the weight once for all of the rows. A recording that holds no library product has no such
/// tile product, so there the GEMVs serve up to [`GEMV_MAX_ROWS`].
const LIBRARY_GEMV_MAX_ROWS: u32 = 2;
/// Workgroups the GEMV grid aims at (about four per multiprocessor of a large GPU) and the most 4-row rounds one
/// workgroup walks.
const GEMV_TARGET_WORKGROUPS: u32 = 528;
const GEMV_MAX_ROUNDS: u32 = 16;

impl Ops {
    /// The decode-regime product of an FP8 (E4M3, block or tensor scales) or NVFP4 weight against `act`'s few rows by
    /// the kernel that reads the stored bytes directly. `false` when the weight, its shape, the row count or the
    /// registered kernels do not allow it; the caller then takes the tile decode.
    pub(super) fn small_m_gemv(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), act: &Act, y: &DeviceBuffer, yoff: u64) -> bool {
        let (first, count) = rows;
        let (m, k) = (act.m, w.k());
        // The kernels reduce across lanes through workgroup memory and barriers; the CPU JIT cannot run them.
        if m == 0 || m > GEMV_MAX_ROWS || !self.gpu.caps().workgroup_reductions {
            return false;
        }
        let max_rows = if self.gpu.declines_library() { GEMV_MAX_ROWS } else { LIBRARY_GEMV_MAX_ROWS };
        // A workgroup covers 4 weight rows per round and amortises its decode table over `rpw` rounds, as many as keep
        // about 4 workgroups per multiprocessor busy: a tall weight (the LM head) walks many rounds, a short one few.
        let tiles = count.div_ceil(4);
        let rpw = (tiles / GEMV_TARGET_WORKGROUPS).clamp(1, GEMV_MAX_ROUNDS);
        let grid = gpu_core::Dispatch::Workgroups(tiles.div_ceil(rpw));
        let out = (yoff, m as u64 * count as u64);
        match w {
            Weight::FP8 { w: wb, s: sb, fmt: Fp8Format::E4M3, block, .. } if k.is_multiple_of(16) && block.is_multiple_of(16) && m <= max_rows => {
                let Some(idx) = self.gpu.kernel_index(&kernels::template::gemv_kernel_name("matmul_fp8_gemv", m)) else { return false };
                let cb = k.div_ceil(*block);
                let params = [m, k, count, first, *block, cb, act.xr0, rpw];
                s.push(self.gpu.dispatch_sliced(idx, &[&act.x, wb, sb, y], &[WHOLE, WHOLE, WHOLE, out], &params, grid));
                true
            }
            Weight::Micro { w: wb, s: sb, kind: MicroScaled::Nvfp4, global, .. } if k.is_multiple_of(64) && m <= max_rows => {
                let Some(idx) = self.gpu.kernel_index(&kernels::template::gemv_kernel_name("matmul_nvfp4_gemv", m)) else { return false };
                let params = [m, k, count, first, act.xr0, global.to_bits(), rpw, 0];
                s.push(self.gpu.dispatch_sliced(idx, &[&act.x, wb, sb, y], &[WHOLE, WHOLE, WHOLE, out], &params, grid));
                true
            }
            _ => false,
        }
    }

    /// Bytes of decoded f32 scratch a quantized product may use per tile (see the module doc).
    pub fn tile_budget_bytes(&self) -> u64 {
        self.tile_budget_bytes
    }

    /// Set the per-tile decode scratch budget. The tile shrinks to fit, down to one row.
    pub fn set_tile_budget_bytes(&mut self, bytes: u64) {
        assert!(bytes > 0, "Ops::set_tile_budget_bytes: the budget must be positive");
        self.tile_budget_bytes = bytes;
    }

    /// How a product over `rows` rows of `w` runs on this device (see [`plan_linear`]).
    pub fn linear_plan(&self, w: &Weight, rows: u32) -> LinearPlan {
        plan_linear(w.dtype(), rows, w.k(), self.tile_budget_bytes, self.gemm.can_dequant_bf16(1, w.k()))
    }

    pub(super) fn tiled_kernel(&self, name: &str) -> usize {
        self.gpu.kernel_index(name).unwrap_or_else(|| {
            panic!(
                "Ops: kernel '{name}' is not registered on this Gpu -- quantized products need \
                 model::ops::training_kernel_list() (the weight_decode variants and copy_cols)"
            )
        })
    }

    /// The planes `weight_decode` reads for `w`, in binding order (codes, f32 scales, sub-scales, super-scales);
    /// a tier binds a tiny dummy buffer for the planes it does not have.
    fn decode_planes<'a>(&'a self, w: &'a Weight) -> [&'a DeviceBuffer; 4] {
        let dummy = self.decode_dummy.get_or_init(|| self.gpu.storage(4));
        match w {
            Weight::BF16 { w: wb, .. } | Weight::F16 { w: wb, .. } => [wb, dummy, dummy, dummy],
            Weight::I8 { w: wb, s: sb, .. } | Weight::Q4 { w: wb, s: sb, .. } | Weight::NF4 { w: wb, s: sb, .. } | Weight::FP8 { w: wb, s: sb, .. } | Weight::Micro { w: wb, s: sb, .. } => {
                [wb, sb, dummy, dummy]
            }
            Weight::KQuant { w: wb, sz: KqScale::PlainF32(sb), .. } => [wb, sb, dummy, dummy],
            Weight::KQuant { w: wb, sz: KqScale::Packed { wsm, wd }, .. } => [wb, dummy, wsm, wd],
            Weight::F32 { .. } => unreachable!("an F32 weight is already dense"),
        }
    }

    fn check_decode_window(w: &Weight, row0: u32, rows: u32) {
        assert!(rows > 0 && row0 as u64 + rows as u64 <= w.n() as u64, "Ops::decode_tile: rows {row0}+{rows} outside the {} rows of the weight", w.n());
    }

    /// Push the step decoding rows `row0 .. row0 + rows` of `w` into `out` (`[rows, k]` f32).
    pub fn decode_tile(&self, s: &mut Vec<Step>, w: &Weight, row0: u32, rows: u32, out: &DeviceBuffer) {
        let fmt = w.decode_format().unwrap_or_else(|| panic!("Ops::decode_tile: an F32 weight is already dense"));
        Self::check_decode_window(w, row0, rows);
        let k = w.k();
        let [wb, sb, wsm, wd] = self.decode_planes(w);
        let block = if let Weight::FP8 { block, .. } = w { *block } else { 1 };
        let global = if let Weight::Micro { global, .. } = w { global.to_bits() } else { 0 };
        let params = [row0, rows, k, block, k.div_ceil(block), k.div_ceil(32).div_ceil(2), k.div_ceil(256), global];
        let name = fmt.kernel_name();
        s.push(self.gpu.step(self.tiled_kernel(&name), &[wb, sb, wsm, wd, out], &params, rows * k));
    }

    /// [`Self::decode_tile`] into a bf16 scratch (`rows * k / 2` words) for the library tensor-core product;
    /// `false` (nothing pushed) when the device declines ([`LibraryGemmProvider::can_dequant_bf16`]).
    pub fn decode_tile_bf16(&self, s: &mut Vec<Step>, w: &Weight, row0: u32, rows: u32, out: &DeviceBuffer) -> bool {
        let format = w.decode_format().unwrap_or_else(|| panic!("Ops::decode_tile_bf16: an F32 weight is already dense"));
        Self::check_decode_window(w, row0, rows);
        let [wb, sb, wsm, wd] = self.decode_planes(w);
        let whole = |buf| library::Operand { buf, range: WHOLE };
        let block = if let Weight::FP8 { block, .. } = w { *block } else { 1 };
        let window = library::Dequant {
            format,
            codes: whole(wb),
            scales: whole(sb),
            sub_scales: whole(wsm),
            super_scales: whole(wd),
            row0,
            rows,
            k: w.k(),
            block,
            global: if let Weight::Micro { global, .. } = w { *global } else { 1.0 },
        };
        self.gemm.dequant_bf16(&self.gpu, s, &window, out)
    }

    /// `dst[r * dst_stride + dst_off + j] = src[r * src_stride + src_off + j]` for `r < rows`, `j < cnt`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_cols(&self, s: &mut Vec<Step>, src: &DeviceBuffer, dst: &DeviceBuffer, rows: u32, cnt: u32, src_stride: u32, src_off: u32, dst_stride: u32, dst_off: u32) {
        s.push(self.gpu.step(self.tiled_kernel("copy_cols"), &[src, dst], &[rows, cnt, src_stride, src_off, dst_stride, dst_off], rows * cnt));
    }

    /// `words` 32-bit words of `buf` from word `start`, through a bounded device copy: only a window around the range
    /// is bound (so a buffer larger than one storage binding can be read) and only the range reaches the host.
    pub fn read_range(&self, buf: &DeviceBuffer, start: u64, words: usize) -> Vec<f32> {
        read_range_with(&self.gpu, self.tiled_kernel("copy_cols"), buf, start, words)
    }

    /// The values `w` stores, as host f32: a dense weight is read as it is, every other tier is
    /// decoded on the device a bounded tile of rows at a time ([`Self::decode_tile`], the scratch is
    /// [`Self::tile_budget_bytes`], not the size of the tensor), so one read serves every tier a
    /// model can hold.
    pub fn read_host(&self, w: &Weight) -> Vec<f32> {
        let (n, k) = (w.n(), w.k());
        if let Weight::F32 { w: dense, .. } = w {
            return self.gpu.read(dense, n as usize * k as usize);
        }
        let tile = tile_rows(n, k, self.tile_budget_bytes, 4);
        let scratch = self.gpu.storage_uninit(tile as u64 * k as u64);
        let mut out = Vec::with_capacity(n as usize * k as usize);
        let mut row = 0;
        while row < n {
            let rows = tile.min(n - row);
            let mut steps = Vec::new();
            self.decode_tile(&mut steps, w, row, rows, &scratch);
            self.gpu.submit(&[], &steps);
            out.extend(self.gpu.read(&scratch, rows as usize * k as usize));
            row += rows;
        }
        out
    }

    /// The tile loop shared by the forward and input-gradient products: one scratch sized by the plan,
    /// each tile of `rows` decoded into it as a dense weight view that `product` multiplies.
    fn for_each_tile(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), mut product: impl FnMut(&mut Vec<Step>, &Weight, u32, u32, u32)) {
        let (first, count) = rows;
        let k = w.k();
        let LinearPlan::TileDecode { tile_rows: tile, bf16 } = self.linear_plan(w, count) else {
            unreachable!("tile path taken for a tier with a native kernel")
        };
        if !bf16 {
            return self.for_each_tile_f32(s, w, rows, product);
        }
        let scratch = self.gpu.storage_uninit(tile as u64 * k as u64 / 2);
        let mut done = 0;
        while done < count {
            let c = tile.min(count - done);
            if !self.decode_tile_bf16(s, w, first + done, c, &scratch) {
                // The device declined the native decode kernel (nothing was pushed for this tile).
                assert_eq!(done, 0, "Ops: the bf16 tile decode was declined mid-loop");
                return self.for_each_tile_f32(s, w, rows, product);
            }
            product(s, &Weight::BF16 { w: scratch.clone(), n: c, k }, done, c, tile);
            done += c;
        }
    }

    pub(super) fn for_each_tile_f32(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), mut product: impl FnMut(&mut Vec<Step>, &Weight, u32, u32, u32)) {
        let (first, count) = rows;
        let k = w.k();
        let LinearPlan::TileDecode { tile_rows: tile, .. } = plan_linear(Dtype::I8, count, k, self.tile_budget_bytes, false) else { unreachable!() };
        let scratch = self.gpu.storage_uninit(tile as u64 * k as u64);
        let mut done = 0;
        while done < count {
            let c = tile.min(count - done);
            self.decode_tile(s, w, first + done, c, &scratch);
            let dense = Weight::F32 { w: scratch.clone(), n: c, k };
            product(s, &dense, done, c, tile);
            done += c;
        }
    }

    /// [`Ops::matmul_rows`] for a tier without a float-read kernel: `y[yoff ..] = act @ W[first.., :]^T`
    /// in row tiles, each a dense product on a decoded tile.
    pub(super) fn forward_tiled(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), act: &Act, y: &DeviceBuffer, yoff: u64) {
        let count = rows.1;
        let m = act.m;
        let mut ytile: Option<DeviceBuffer> = None;
        self.for_each_tile(s, w, rows, |s, dense, done, c, tile| {
            if tile >= count {
                return self.matmul_rows(s, dense, (0, c), act, y, yoff);
            }
            let yt = ytile.get_or_insert_with(|| self.gpu.storage_uninit(m as u64 * tile as u64)).clone();
            self.matmul_rows(s, dense, (0, c), act, &yt, 0);
            let dst_off = u32::try_from(yoff).expect("Ops::forward_tiled: output offset exceeds u32") + done;
            self.copy_cols(s, &yt, y, m, c, c, 0, count, dst_off);
        });
    }

    /// [`Ops::matmul_dx_rows`] for a tier without a float-read kernel: `dx[dx_row0 ..] (+)= dy @ W[first.., :]`
    /// with `dy` dense `[m, count]`, accumulating the row tiles' contributions into `dx`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dx_tiled(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), dy: &DeviceBuffer, m: u32, dx: &DeviceBuffer, dx_row0: u32, accumulate: bool) {
        let count = rows.1;
        let mut dytile: Option<DeviceBuffer> = None;
        self.for_each_tile(s, w, rows, |s, dense, done, c, tile| {
            let acc = accumulate || done > 0;
            if tile >= count {
                return self.matmul_dx_rows(s, dense, (0, c), dy, m, dx, dx_row0, acc);
            }
            let dt = dytile.get_or_insert_with(|| self.gpu.storage_uninit(m as u64 * tile as u64)).clone();
            self.copy_cols(s, dy, &dt, m, c, count, done, c, 0);
            self.matmul_dx_rows(s, dense, (0, c), &dt, m, dx, dx_row0, acc);
        });
    }

    /// The weight gradient of `Y = X W^T` for the shape of `w`: `dW[n, k] += dY[m, n]^T X[m, k]`, into `dw`
    /// stored as `grad`. It never reads the weight's values, so every tier shares it; the weight only
    /// supplies `n` and `k`. `dw` accumulates (the caller zeroes it once per backward pass, exactly like
    /// [`Ops::matmul_dw`]). An fp32 gradient is one product straight into `dw`; a bf16 gradient is
    /// computed per row tile into a bounded fp32 scratch and added with round to nearest even.
    #[allow(clippy::too_many_arguments)]
    pub fn weight_grad(&self, s: &mut Vec<Step>, w: &Weight, x: &DeviceBuffer, dy: &DeviceBuffer, m: u32, dw: &DeviceBuffer, grad: GradDtype) {
        let (n, k) = (w.n(), w.k());
        match grad {
            GradDtype::F32 => self.dw_f32(s, x, dy, m, n, k, dw, 0),
            GradDtype::Bf16 => {
                assert!(k.is_multiple_of(2), "Ops::weight_grad: a bf16 gradient needs an even K (got {k})");
                let tile = tile_rows(n, k, self.tile_budget_bytes, 4);
                // Zero-initialised once; the accumulate kernel clears each tile after consuming it.
                let scratch = self.gpu.storage(tile as u64 * k as u64);
                let dytile = (tile < n).then(|| self.gpu.storage_uninit(m as u64 * tile as u64));
                let mut done = 0;
                while done < n {
                    let c = tile.min(n - done);
                    match &dytile {
                        None => self.dw_f32_overwrite_or_add(s, x, dy, m, c, k, &scratch),
                        Some(dt) => {
                            self.copy_cols(s, dy, dt, m, c, n, done, c, 0);
                            self.dw_f32_overwrite_or_add(s, x, dt, m, c, k, &scratch);
                        }
                    }
                    self.accumulate_bf16(s, &scratch, dw, c as u64 * k as u64, done as u64 * k as u64);
                    done += c;
                }
            }
        }
    }

    /// `dw[elem0 .. elem0 + elems] += tile[0 .. elems]` into a bf16 gradient (round to nearest even), clearing the
    /// f32 `tile` so the next product can accumulate into it again. `elem0` and `elems` are even.
    pub fn accumulate_bf16(&self, s: &mut Vec<Step>, tile: &DeviceBuffer, dw: &DeviceBuffer, elems: u64, elem0: u64) {
        assert!(elems.is_multiple_of(2) && elem0.is_multiple_of(2), "Ops::accumulate_bf16: a bf16 word holds two elements (elems {elems}, elem0 {elem0})");
        let words = u32::try_from(elems / 2).expect("Ops::accumulate_bf16: tile exceeds u32 words");
        let elem0 = u32::try_from(elem0).expect("Ops::accumulate_bf16: gradient exceeds u32 elements");
        s.push(self.gpu.step(self.tiled_kernel("accum_bf16"), &[tile, dw], &[words, elem0], words));
    }

    /// The gradient of an embedding table: see [`EmbeddingGrad`]. Like [`Self::weight_grad`] it accumulates and never
    /// reads the table's values, so every tier shares it.
    pub fn embedding_grad(&self, s: &mut Vec<Step>, g: &EmbeddingGrad<'_>) {
        let params = [g.n_rows, g.width, g.n_uniq];
        let binds = [g.tokens, g.uniq, g.d_x, g.dw];
        match g.grad {
            GradDtype::F32 => s.push(self.gpu.step(self.tiled_kernel("emb_bwd_uniq"), &binds, &params, g.n_uniq * g.width)),
            GradDtype::Bf16 => {
                assert!(g.width.is_multiple_of(2), "Ops::embedding_grad: a bf16 gradient needs an even width (got {})", g.width);
                s.push(self.gpu.step(self.tiled_kernel("emb_bwd_uniq_bf16"), &binds, &params, g.n_uniq * (g.width / 2)));
            }
        }
    }

    /// `dw[row0 * k ..] += dy[m, n]^T x[m, k]` in fp32: the library TF32 product where the device has it
    /// (accumulating into `dw`), else the portable accumulating kernel.
    #[allow(clippy::too_many_arguments)]
    fn dw_f32(&self, s: &mut Vec<Step>, x: &DeviceBuffer, dy: &DeviceBuffer, m: u32, n: u32, k: u32, dw: &DeviceBuffer, row0: u32) {
        let range = if row0 == 0 { WHOLE } else { (row0 as u64 * k as u64, n as u64 * k as u64) };
        let (dy_op, x_op) = (library::Operand { buf: dy, range: WHOLE }, library::Operand { buf: x, range: WHOLE });
        if self.gemm.dw_tf32(&self.gpu, s, &dy_op, &x_op, &library::Operand { buf: dw, range }, m, n, k, true) {
            return;
        }
        assert_eq!(row0, 0, "Ops::dw_f32: the portable kernel writes whole gradients");
        self.matmul_dw(s, x, dy, m, n, k, dw);
    }

    /// The product of one gradient tile into the (zeroed) tile scratch: [`Self::dw_f32`] over the scratch.
    #[allow(clippy::too_many_arguments)]
    fn dw_f32_overwrite_or_add(&self, s: &mut Vec<Step>, x: &DeviceBuffer, dy: &DeviceBuffer, m: u32, c: u32, k: u32, scratch: &DeviceBuffer) {
        self.dw_f32(s, x, dy, m, c, k, scratch, 0);
    }
}

/// [`Ops::read_range`] on any handle that registered `copy_cols` (its index `copy_cols`).
pub(super) fn read_range_with(gpu: &gpu_core::Gpu, copy_cols: usize, buf: &DeviceBuffer, start: u64, words: usize) -> Vec<f32> {
    // A bound sub-range starts on a 256-byte boundary.
    const ALIGN_WORDS: u64 = 64;
    if words == 0 {
        return Vec::new();
    }
    let window0 = start / ALIGN_WORDS * ALIGN_WORDS;
    let lead = u32::try_from(start - window0).expect("an alignment remainder fits u32");
    let count = u32::try_from(words).expect("read_range: a range exceeds u32 words");
    let staging = gpu.storage_uninit(words as u64);
    let params = [1, count, lead + count, lead, count, 0];
    let copy = gpu.step_sliced(copy_cols, &[buf, &staging], &[(window0, (lead + count) as u64), WHOLE], &params, count);
    gpu.submit(&[], &[copy]);
    gpu.read(&staging, words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tiers_keep_their_kernels_and_quantized_tiers_tile_by_budget() {
        for tier in [Dtype::F32, Dtype::BF16] {
            assert_eq!(plan_linear(tier, 1000, 512, 1 << 20, true), LinearPlan::Native, "{tier:?}");
        }
        for tier in [Dtype::NF4, Dtype::F16, Dtype::I8, Dtype::Q4, Dtype::Q4K, Dtype::Q8K, Dtype::F8E4M3, Dtype::F8E5M2] {
            // 1 MiB of f32 over 512-wide rows is 512 rows per tile; a bf16 scratch fits twice the rows.
            assert_eq!(plan_linear(tier, 1000, 512, 1 << 20, false), LinearPlan::TileDecode { tile_rows: 512, bf16: false }, "{tier:?}");
            assert_eq!(plan_linear(tier, 1000, 512, 1 << 20, true), LinearPlan::TileDecode { tile_rows: 1000, bf16: true }, "{tier:?}");
        }
    }

    #[test]
    fn the_tile_is_bounded_by_the_budget_not_by_the_tensor() {
        assert_eq!(tile_rows(1_000_000, 4096, 64 << 20, 4), 4096);
        assert_eq!(tile_rows(10, 4096, 64 << 20, 4), 10, "never more rows than the weight has");
        assert_eq!(tile_rows(10, 1 << 20, 1024, 4), 1, "a row wider than the budget still makes progress");
    }
}
