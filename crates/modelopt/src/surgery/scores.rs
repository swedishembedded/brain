// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How much each channel of a declared dimension matters, by the estimates
//! a channel selection ranks on (higher keeps first):
//!
//! * **activation RMS**: the root mean square of the channel's activation on
//!   calibration data;
//! * **Wanda**: the activation's norm times the summed magnitude of the
//!   weights that read it (`‖a_c‖ · Σ_o |W[o, c]|`), the per-weight metric
//!   `|W| · ‖X‖` summed over the channel;
//! * **Fisher**: the diagonal-Fisher estimate of the loss increase of
//!   zeroing the channel, `½ Σ_j F_j w_j²` over its weights;
//! * **group Taylor**: the squared first-order change of the loss when the
//!   whole channel is zeroed, `(Σ_j g_j w_j)²`, averaged over batches.
//!
//! The activation statistics are the model's to collect (only it knows where
//! a channel's activation is); the curvature ones run on any
//! [`model::Model`] through its gradients.

use model::surgery::{Axis, ChannelDim};
use model::Model;

use super::Checkpoint;
pub use crate::sensitivity::LoadBatch;

/// A channel score: what the selection ranks a dimension by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Score {
    ActRms,
    Wanda,
    Fisher,
    Taylor,
}

/// The root mean square of each channel from its summed squared activation over `count` rows.
pub fn act_rms(sum_sq: &[f64], count: u64) -> Vec<f64> {
    sum_sq.iter().map(|s| (s / count.max(1) as f64).sqrt()).collect()
}

/// Wanda channel scores: `sqrt(sum_sq[c]) * Σ |reader lines of c|`, where
/// `reader` is the tensor that reads the channel (`axis` indexes channels,
/// `block` lines each).
pub fn wanda(ck: &Checkpoint, sum_sq: &[f64], reader: &str, axis: Axis, block: usize) -> Result<Vec<f64>, String> {
    let w = ck.get(reader)?;
    let (rows, cols) = w.rows_cols();
    let width = sum_sq.len();
    let lines = match axis {
        Axis::Rows => rows,
        Axis::Cols => cols,
    };
    if lines != width * block {
        return Err(format!("{reader}: {lines} lines along {axis:?}, the activation has {width} x {block}"));
    }
    let mut mag = vec![0f64; width];
    for r in 0..rows {
        for c in 0..cols {
            let line = if axis == Axis::Rows { r } else { c };
            mag[line / block] += f64::from(w.data[r * cols + c].abs());
        }
    }
    Ok(mag.iter().zip(sum_sq).map(|(m, s)| m * s.sqrt()).collect())
}

/// The elements of channel `c` (lines `c*block..(c+1)*block`) in a `[rows, cols]` tensor.
fn channel_elements(rows: usize, cols: usize, axis: Axis, block: usize, c: usize) -> Box<dyn Iterator<Item = usize>> {
    match axis {
        Axis::Rows => Box::new(c * block * cols..(c + 1) * block * cols),
        Axis::Cols => Box::new((0..rows).flat_map(move |r| (c * block..(c + 1) * block).map(move |k| r * cols + k))),
    }
}

/// Fisher and group-Taylor scores of every channel of `dims` on `m`, one
/// forward and backward per calibration batch; `load(m, i)` puts batch `i`
/// of `batches` on the model. A tensor's shape is its [`Checkpoint`]
/// entry's (`shapes`); its values and gradients are the model's.
pub fn curvature<M: Model>(m: &M, shapes: &Checkpoint, dims: &[&ChannelDim], batches: usize, load: &mut LoadBatch<'_, M>) -> Result<Curvature, String> {
    let mut fisher: Vec<Vec<f64>> = dims.iter().map(|d| vec![0f64; d.width]).collect();
    let mut taylor: Vec<Vec<f64>> = fisher.clone();
    // Every line of every dimension, flattened: (dimension, tensor, its shape, its weights).
    let mut lines = Vec::new();
    for (di, d) in dims.iter().enumerate() {
        for l in &d.lines {
            lines.push((di, l, shapes.get(&l.tensor)?.rows_cols(), m.read_weight(&l.tensor)));
        }
    }
    let names: Vec<&str> = lines.iter().map(|(_, l, _, _)| l.tensor.as_str()).collect();
    // The first-order sums of the batch under way, per dimension and channel.
    let mut first: Vec<Vec<f64>> = fisher.clone();
    crate::sensitivity::for_each_gradient(m, &names, batches, load, &mut |_, t, g| {
        let (di, l, (rows, cols), w) = &lines[t];
        if g.len() != w.len() {
            return Err(format!("{}: gradient of {} values, the tensor {}", l.tensor, g.len(), w.len()));
        }
        for c in 0..dims[*di].width {
            for j in channel_elements(*rows, *cols, l.axis, l.block, c) {
                let (gj, wj) = (f64::from(g[j]), f64::from(w[j]));
                fisher[*di][c] += 0.5 * gj * gj * wj * wj;
                first[*di][c] += gj * wj;
            }
        }
        // The batch's last line closes it: square its first-order sums into the Taylor scores.
        if t + 1 == lines.len() {
            for (tay, f) in taylor.iter_mut().zip(first.iter_mut()) {
                tay.iter_mut().zip(f.iter_mut()).for_each(|(t, f)| *t += std::mem::take(f).powi(2));
            }
        }
        Ok(())
    })?;
    let n = batches as f64;
    for v in fisher.iter_mut().chain(taylor.iter_mut()).flatten() {
        *v /= n;
    }
    Ok(Curvature { fisher, taylor })
}

/// Per dimension and channel: the Fisher and the group-Taylor scores.
#[derive(Clone, Debug, PartialEq)]
pub struct Curvature {
    pub fisher: Vec<Vec<f64>>,
    pub taylor: Vec<Vec<f64>>,
}
