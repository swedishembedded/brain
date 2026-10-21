// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in training with factored optimizer
// state so that the second moment costs rows plus columns instead of rows times
// columns, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Adafactor's factored second moment (Shazeer and Stern, 2018, arXiv:1804.04235).
//!
//! A matrix `[n, k]` keeps a row statistic `R[n]` and a column statistic `C[k]` of the squared gradient
//! instead of an `[n, k]` second moment; the reconstruction is `V = R C^T / mean(R)`. A vector keeps the
//! plain second moment. There is no first moment. The update is `u = g / sqrt(V)`, clipped so its RMS does not
//! exceed `clip_threshold`, applied as `w <- w - lr (u + wd w)`.
//!
//! The statistics are tiny (`n + k` floats a matrix) and live in host memory beside the streamed state; the
//! update runs in three streamed passes over a tensor's gradient tiles (statistics, update RMS, apply), tiles
//! being whole rows so the row and column sums are exact.

use std::io;

/// `1 - t^(-decay)`: the second-moment decay of step `t` (1-based), growing towards 1.
pub fn beta2(decay: f32, t: u32) -> f32 {
    1.0 - (t as f32).powf(-decay)
}

/// Row and column statistics of one tensor, or the plain second moment of a vector (`cols` empty).
#[derive(Clone, Debug, PartialEq)]
pub struct Factored {
    pub rows: Vec<f32>,
    pub cols: Vec<f32>,
}

impl Factored {
    /// Zeroed statistics of a tensor of `numel` elements with rows of `row_len` (`None`: a vector).
    pub fn new(numel: usize, row_len: Option<usize>) -> Factored {
        match row_len {
            Some(k) => Factored { rows: vec![0.0; numel / k], cols: vec![0.0; k] },
            None => Factored { rows: vec![0.0; numel], cols: Vec::new() },
        }
    }

    /// Host bytes of the statistics.
    pub fn bytes(&self) -> u64 {
        4 * (self.rows.len() + self.cols.len()) as u64
    }

    pub fn is_matrix(&self) -> bool {
        !self.cols.is_empty()
    }
}

/// Accumulates the squared-gradient statistics of one step over a tensor's tiles.
pub struct SqAccumulator {
    rows: Vec<f32>,
    cols: Vec<f32>,
    row_len: usize,
    n_rows: usize,
    eps: f32,
}

impl SqAccumulator {
    pub fn new(f: &Factored, eps: f32) -> SqAccumulator {
        SqAccumulator { rows: vec![0.0; f.rows.len()], cols: vec![0.0; f.cols.len()], row_len: f.cols.len(), n_rows: f.rows.len(), eps }
    }

    /// Fold `g` (already multiplied by the step's gradient multiplier), starting at element `start`.
    /// For a matrix `start` and `g.len()` must be whole rows.
    pub fn add(&mut self, start: usize, g: &[f32]) {
        if self.row_len == 0 {
            for (a, &x) in self.rows[start..].iter_mut().zip(g) {
                *a = x * x + self.eps;
            }
            return;
        }
        let k = self.row_len;
        debug_assert!(start.is_multiple_of(k) && g.len().is_multiple_of(k), "a factored tile is whole rows");
        for (r, row) in g.chunks(k).enumerate() {
            let i = start / k + r;
            let mut sum = 0.0f32;
            for (c, &x) in row.iter().enumerate() {
                let sq = x * x + self.eps;
                sum += sq;
                self.cols[c] += sq;
            }
            self.rows[i] = sum;
        }
    }

    /// The exponential moving average of the step's statistics into `f` (matrices: row and column means).
    pub fn finish(self, f: &mut Factored, beta2: f32) {
        let ema = |old: &mut f32, new: f32| *old = beta2 * *old + (1.0 - beta2) * new;
        if self.row_len == 0 {
            for (o, n) in f.rows.iter_mut().zip(&self.rows) {
                ema(o, *n);
            }
            return;
        }
        for (o, n) in f.rows.iter_mut().zip(&self.rows) {
            ema(o, n / self.row_len as f32);
        }
        for (o, n) in f.cols.iter_mut().zip(&self.cols) {
            ema(o, n / self.n_rows as f32);
        }
    }
}

/// The preconditioned gradient `u = g / sqrt(V)` of a tile starting at `start`, in place.
pub fn precondition(f: &Factored, start: usize, g: &mut [f32]) {
    if !f.is_matrix() {
        for (x, v) in g.iter_mut().zip(&f.rows[start..]) {
            *x /= v.sqrt().max(f32::MIN_POSITIVE);
        }
        return;
    }
    let k = f.cols.len();
    let mean_r = f.rows.iter().sum::<f32>() / f.rows.len() as f32;
    for (r, row) in g.chunks_mut(k).enumerate() {
        let ri = f.rows[start / k + r];
        for (x, c) in row.iter_mut().zip(&f.cols) {
            *x /= (ri * c / mean_r).sqrt().max(f32::MIN_POSITIVE);
        }
    }
}

/// The factor `1 / max(1, rms(u) / threshold)` that keeps the update's RMS at or below `threshold`.
pub fn clip_scale(sum_sq_u: f64, numel: usize, threshold: f32) -> f32 {
    let rms = (sum_sq_u / numel as f64).sqrt() as f32;
    1.0 / (rms / threshold).max(1.0)
}

/// Serialize the statistics (rows then columns, little endian f32) for a checkpoint.
pub fn write_stats(f: &Factored, w: &mut dyn io::Write) -> io::Result<()> {
    for v in f.rows.iter().chain(&f.cols) {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Read the statistics [`write_stats`] wrote into `f`, in place.
pub fn read_stats(f: &mut Factored, r: &mut dyn io::Read) -> io::Result<()> {
    let mut b = [0u8; 4];
    for v in f.rows.iter_mut().chain(f.cols.iter_mut()) {
        r.read_exact(&mut b)?;
        *v = f32::from_le_bytes(b);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: for a gradient whose squares are exactly rank one (`g_ij^2 = a_i b_j`) the factored second moment
    /// reconstructs them, so the preconditioned gradient has unit magnitude everywhere.
    #[test]
    fn a_rank_one_second_moment_is_reconstructed_exactly() {
        let (n, k) = (3usize, 4usize);
        let a = [1.0f32, 4.0, 9.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let g: Vec<f32> = (0..n * k).map(|i| (a[i / k] * b[i % k]).sqrt()).collect();
        let mut f = Factored::new(n * k, Some(k));
        let mut acc = SqAccumulator::new(&f, 0.0);
        acc.add(0, &g);
        acc.finish(&mut f, 0.0);
        let mut u = g.clone();
        precondition(&f, 0, &mut u);
        assert!(u.iter().all(|x| (x - 1.0).abs() < 1e-5), "{u:?}");
    }

    /// Spec: statistics folded in whole-row tiles equal those folded in one tile, and the state is `n + k` floats.
    #[test]
    fn tiling_does_not_change_the_statistics_and_the_state_is_rows_plus_columns() {
        let (n, k) = (6usize, 5usize);
        let g: Vec<f32> = (0..n * k).map(|i| ((i * 13 % 17) as f32 - 8.0) / 7.0).collect();
        let run = |tile_rows: usize| {
            let mut f = Factored::new(n * k, Some(k));
            let mut acc = SqAccumulator::new(&f, 1e-30);
            for (t, chunk) in g.chunks(tile_rows * k).enumerate() {
                acc.add(t * tile_rows * k, chunk);
            }
            acc.finish(&mut f, 0.5);
            f
        };
        assert_eq!(run(6), run(1));
        assert_eq!(run(2), run(6));
        assert_eq!(run(6).bytes(), 4 * (n + k) as u64);
    }

    /// Spec: the update-clipping factor leaves a small update alone and scales a large one to the threshold.
    #[test]
    fn the_update_rms_is_clipped_to_the_threshold() {
        assert_eq!(clip_scale(4.0, 4, 1.0), 1.0, "rms 1 is at the threshold");
        assert_eq!(clip_scale(1.0, 4, 1.0), 1.0, "rms 0.5 is below it");
        assert!((clip_scale(16.0, 4, 1.0) - 0.5).abs() < 1e-6, "rms 2 is scaled to 1");
    }

    /// Spec: the decay schedule starts at 0 and grows towards 1.
    #[test]
    fn beta2_grows_towards_one() {
        assert_eq!(beta2(0.8, 1), 0.0);
        assert!(beta2(0.8, 2) > 0.4 && beta2(0.8, 100) > 0.97 && beta2(0.8, 100) < 1.0);
    }

    /// Spec: the statistics survive a checkpoint round trip.
    #[test]
    fn statistics_round_trip() {
        let mut f = Factored::new(12, Some(4));
        f.rows = vec![1.0, 2.0, 3.0];
        f.cols = vec![0.5, 0.25, 0.125, 1.0];
        let mut bytes = Vec::new();
        write_stats(&f, &mut bytes).unwrap();
        let mut g = Factored::new(12, Some(4));
        read_stats(&mut g, &mut bytes.as_slice()).unwrap();
        assert_eq!(f, g);
    }
}
