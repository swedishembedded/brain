// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A learned position table resampled onto an image grid: the host-built tap plan (which table rows each patch reads
//! and with what weight), and the device gather and table-gradient steps that consume it.
//!
//! A vision tower trains one `side x side` table of position embeddings and serves images of any size, so each
//! patch's position embedding is a bilinear blend of the four table rows around the patch's coordinate on the table's
//! grid. The blend is `out[r] = sum_t weight[r, t] * table[idx[r, t]]`: the plan is `idx` and `weight`, the forward
//! is `rows_gather_weighted`, and the table's gradient is the transpose of the same sparse matrix,
//! `rows_gather_weighted_bwd`, which needs the plan in CSR form ([`TapPlan::transpose`]) because scattering into
//! shared rows takes atomics that core-compute WGSL does not have.
//!
//! [`bilinear_grid_plan`] reproduces `F.interpolate(mode="bilinear")` as a tap list, in f32 and in the same operation
//! order as the reference helper that does so for vision towers (`align_corners` false uses half-pixel centres,
//! `(i + 0.5) * side / size - 0.5`; an edge tap clamps to the border). Patches are emitted in spatial-merge-block
//! order (block row, block column, row in block, column in block), the order a tower that merges `m x m` patches into
//! one token consumes them; `merge = 1` is raster order.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Which table rows each output row reads and with what weight: `taps` entries per output row.
#[derive(Clone, Debug, PartialEq)]
pub struct TapPlan {
    pub taps: u32,
    /// `[rows * taps]` table row numbers.
    pub idx: Vec<u32>,
    /// `[rows * taps]` weights.
    pub weight: Vec<f32>,
}

/// A [`TapPlan`] transposed to table rows: for table row `i`, the edges `ptr[i] .. ptr[i + 1]` name the output rows
/// that read it (`edge_row`) and the weight of each read (`edge_w`), in output-row order.
#[derive(Clone, Debug, PartialEq)]
pub struct TapEdges {
    pub ptr: Vec<u32>,
    pub edge_row: Vec<u32>,
    pub edge_w: Vec<f32>,
}

impl TapPlan {
    /// The number of output rows.
    pub fn rows(&self) -> u32 {
        self.idx.len() as u32 / self.taps
    }

    /// Check the plan is well formed for a table of `table_rows` rows: at least one tap and one row, matching
    /// lengths, every table row in range, every weight finite.
    pub fn validate(&self, table_rows: u32) -> Result<(), String> {
        if self.taps == 0 || self.idx.is_empty() {
            return Err("tap plan: needs at least one tap and one output row".to_string());
        }
        if self.idx.len() != self.weight.len() || !self.idx.len().is_multiple_of(self.taps as usize) {
            return Err(format!("tap plan: {} indices, {} weights, {} taps per row", self.idx.len(), self.weight.len(), self.taps));
        }
        if let Some(i) = self.idx.iter().position(|&r| r >= table_rows) {
            return Err(format!("tap plan: tap {i} names table row {} of {table_rows}", self.idx[i]));
        }
        if let Some(i) = self.weight.iter().position(|w| !w.is_finite()) {
            return Err(format!("tap plan: tap {i} has a non-finite weight"));
        }
        Ok(())
    }

    /// The plan as CSR over table rows, for the table gradient. Edges of one table row stay in output-row order, so the
    /// gradient is summed in a fixed order.
    pub fn transpose(&self, table_rows: u32) -> Result<TapEdges, String> {
        self.validate(table_rows)?;
        let mut ptr = vec![0u32; table_rows as usize + 1];
        for &r in &self.idx {
            ptr[r as usize + 1] += 1;
        }
        for i in 0..table_rows as usize {
            ptr[i + 1] += ptr[i];
        }
        let mut next = ptr.clone();
        let mut edge_row = vec![0u32; self.idx.len()];
        let mut edge_w = vec![0f32; self.idx.len()];
        for (k, (&r, &w)) in self.idx.iter().zip(&self.weight).enumerate() {
            let slot = next[r as usize] as usize;
            next[r as usize] += 1;
            edge_row[slot] = k as u32 / self.taps;
            edge_w[slot] = w;
        }
        Ok(TapEdges { ptr, edge_row, edge_w })
    }
}

/// The two taps and weights of output coordinate `index` on an axis of length `size`, into a `side`-long table axis:
/// `F.interpolate(mode="bilinear")`'s kernel, in f32 and in the reference's operation order.
pub fn bilinear_axis_taps(index: u32, size: u32, side: u32, align_corners: bool) -> ([u32; 2], [f32; 2]) {
    let (index, size_f, side_f) = (index as f32, size as f32, side as f32);
    let src = if align_corners { index * (side_f - 1.0) / (size_f - 1.0).max(1.0) } else { (index + 0.5) * side_f / size_f - 0.5 };
    let floor = src.floor();
    let mut taps = [0u32; 2];
    let mut weights = [0f32; 2];
    for (k, offset) in [0.0f32, 1.0].into_iter().enumerate() {
        taps[k] = (floor as i64 + k as i64).clamp(0, i64::from(side) - 1) as u32;
        weights[k] = (1.0 - (src - floor - offset).abs()).max(0.0);
    }
    (taps, weights)
}

/// The bilinear tap plan resampling a `side x side` table onto each of `grids` (`(height, width)` in patches), the
/// grids' patches concatenated in order, each in spatial-merge-block order (see the module doc). Four taps per patch,
/// `idx = row_tap * side + column_tap` with the row tap major, weight the product of the two axis weights.
pub fn bilinear_grid_plan(grids: &[(u32, u32)], side: u32, merge: u32, align_corners: bool) -> Result<TapPlan, String> {
    if side == 0 || merge == 0 {
        return Err("bilinear plan: the table side and the merge size must be positive".to_string());
    }
    let mut plan = TapPlan { taps: 4, idx: Vec::new(), weight: Vec::new() };
    for &(h, w) in grids {
        if h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
            return Err(format!("bilinear plan: a {h}x{w} patch grid is not a non-empty multiple of the merge size {merge}"));
        }
        for block_row in 0..h / merge {
            for block_col in 0..w / merge {
                for in_row in 0..merge {
                    for in_col in 0..merge {
                        let (row_taps, row_w) = bilinear_axis_taps(block_row * merge + in_row, h, side, align_corners);
                        let (col_taps, col_w) = bilinear_axis_taps(block_col * merge + in_col, w, side, align_corners);
                        for (rt, rw) in row_taps.into_iter().zip(row_w) {
                            for (ct, cw) in col_taps.into_iter().zip(col_w) {
                                plan.idx.push(rt * side + ct);
                                plan.weight.push(rw * cw);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(plan)
}

/// The pipeline slots of the two kernels, resolved by name off the device handle.
#[derive(Clone, Copy, Debug)]
pub struct PosTableKernels {
    pub gather: usize,
    /// `None` on a device registered for inference only: [`DeviceTaps::table_grad_step`] then refuses.
    pub gather_bwd: Option<usize>,
}

impl PosTableKernels {
    /// `rows_gather_weighted` is required; `rows_gather_weighted_bwd` is taken when the device registers it.
    pub fn by_name(g: &Gpu) -> Result<PosTableKernels, String> {
        let gather = g.kernel_index("rows_gather_weighted").ok_or("postable: kernel \"rows_gather_weighted\" is not registered")?;
        Ok(PosTableKernels { gather, gather_bwd: g.kernel_index("rows_gather_weighted_bwd") })
    }
}

fn index_buffer(g: &Gpu, label: &str, data: &[u32]) -> DeviceBuffer {
    let b = g.buffer(label, data.len() as u64 * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
    g.write(&b, data);
    b
}

/// A [`TapPlan`] uploaded for one table: the forward's index and weight buffers and the backward's CSR buffers.
pub struct DeviceTaps {
    rows: u32,
    taps: u32,
    table_rows: u32,
    idx: DeviceBuffer,
    weight: DeviceBuffer,
    ptr: DeviceBuffer,
    edge_row: DeviceBuffer,
    edge_w: DeviceBuffer,
}

impl DeviceTaps {
    pub fn upload(g: &Gpu, plan: &TapPlan, table_rows: u32) -> Result<DeviceTaps, String> {
        let edges = plan.transpose(table_rows)?;
        Ok(DeviceTaps {
            rows: plan.rows(),
            taps: plan.taps,
            table_rows,
            idx: index_buffer(g, "postable.idx", &plan.idx),
            weight: g.storage_init("postable.weight", &plan.weight),
            ptr: index_buffer(g, "postable.ptr", &edges.ptr),
            edge_row: index_buffer(g, "postable.edge_row", &edges.edge_row),
            edge_w: g.storage_init("postable.edge_w", &edges.edge_w),
        })
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// `out[r, :] = sum_t weight[r, t] * table[idx[r, t], :]` for `[table_rows, width]` `table` into `[rows, width]`
    /// `out`.
    pub fn gather_step(&self, g: &Gpu, k: &PosTableKernels, table: &DeviceBuffer, out: &DeviceBuffer, width: u32) -> Step {
        g.step(k.gather, &[&self.idx, &self.weight, table, out], &[self.rows, self.taps, width], self.rows * width)
    }

    /// `dtable += transpose(plan) * dout` for `[rows, width]` `dout` into `[table_rows, width]` `dtable`
    /// (accumulating: zero it first).
    pub fn table_grad_step(&self, g: &Gpu, k: &PosTableKernels, dout: &DeviceBuffer, dtable: &DeviceBuffer, width: u32) -> Result<Step, String> {
        let bwd = k.gather_bwd.ok_or("postable: kernel \"rows_gather_weighted_bwd\" is not registered, so the table gradient is unavailable")?;
        Ok(g.step(bwd, &[&self.ptr, &self.edge_row, &self.edge_w, dout, dtable], &[self.table_rows, width], self.table_rows * width))
    }
}
