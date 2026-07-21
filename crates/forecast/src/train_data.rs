// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Leak-safe training-data assembly for the weekly fine-tuning pipeline.
//!
//! Turns the whole market's raw daily OHLCV into `(context → horizon)` training
//! windows and — the correctness-critical part — a **temporal split with an
//! embargo/purge gap** so no training window's forecast horizon can leak into a
//! validation/holdout window (and vice-versa). This is model-agnostic: it does the
//! windowing, calendar stamping, and splitting; each model applies its OWN
//! (already past-only, leak-safe) normalization to the extracted window.
//!
//! Leakage rules enforced here:
//! - Windows are assigned to a split by their **origin date** (the timestamp of the
//!   last context bar; the horizon is strictly in the future of the origin).
//! - Between splits sits an **embargo** of `>= horizon` trading days: the last train
//!   origin's horizon ends before the first val origin, so no realized label is shared.
//! - Normalization is never global — it is computed per window from PAST bars only
//!   (done downstream in each model's preprocess), so future values never enter the
//!   scaling of any input. This module never looks at a window's future to build its
//!   context.
//!
//! The reference Kronos recipe instead *overlaps* its split ranges (val starts months
//! before train ends) — convenient for lookback context but not a purged split. We use
//! an explicit embargo so the pipeline is correct by construction.

/// A `(year, month, day)` calendar date (daily bars; intraday unused here).
pub type Date = (i32, u32, u32);

/// One instrument's aligned daily history. `dates[i]` ↔ `ohlcv[i]` =
/// `[open, high, low, close, volume]`. Rows must be ascending in time.
#[derive(Clone, Debug)]
pub struct Series {
    pub ticker: String,
    pub dates: Vec<Date>,
    pub ohlcv: Vec<[f32; 5]>,
}

impl Series {
    pub fn len(&self) -> usize {
        self.ohlcv.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ohlcv.is_empty()
    }
}

/// A reference to one training window: `series[series_idx]`, context =
/// `[origin-context, origin)`, future = `[origin, origin+horizon)`. `origin` is the
/// index of the first future bar, so the origin *date* is `dates[origin-1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRef {
    pub series_idx: usize,
    pub origin: usize,
}

/// Days-from-civil weekday, Monday=0 (matches pandas `.weekday()` / the reference
/// `calc_time_stamps`). Used to build the calendar stamps Kronos consumes.
pub fn weekday(y: i32, m: u32, d: u32) -> u32 {
    let (y, m, d) = (y as i64, m as i64, d as i64);
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    (((days % 7) + 3 + 7) % 7) as u32
}

/// A monotone integer key for a date, for ordering/embargo math (no calendar
/// arithmetic needed — only relative order matters, and we count in *trading bars*,
/// not calendar days, via the global bar index below).
fn date_key(d: Date) -> i64 {
    d.0 as i64 * 10000 + d.1 as i64 * 100 + d.2 as i64
}

/// Every valid window in one series: an origin is valid when a full `context` of past
/// bars and a full `horizon` of future bars both exist.
pub fn enumerate_windows(series: &[Series], context: usize, horizon: usize) -> Vec<WindowRef> {
    let mut out = Vec::new();
    for (si, s) in series.iter().enumerate() {
        if s.len() < context + horizon {
            continue;
        }
        for origin in context..=(s.len() - horizon) {
            out.push(WindowRef { series_idx: si, origin });
        }
    }
    out
}

/// The three temporally-ordered, embargo-separated window sets.
#[derive(Clone, Debug, Default)]
pub struct Split {
    pub train: Vec<WindowRef>,
    pub val: Vec<WindowRef>,
    pub holdout: Vec<WindowRef>,
}

/// Config for [`temporal_split`].
#[derive(Clone, Copy, Debug)]
pub struct SplitConfig {
    /// Fraction of the global trading timeline used for training (by origin date).
    pub train_frac: f32,
    /// Fraction used for validation (best-checkpoint selection). The remainder,
    /// after embargoes, is the holdout used by the promotion gate.
    pub val_frac: f32,
    /// Embargo in **trading bars** between adjacent splits. Must be `>= horizon`
    /// to purge horizon leakage; a larger value (e.g. `context+horizon`) also
    /// prevents context overlap across the seam.
    pub embargo: usize,
}

impl Default for SplitConfig {
    fn default() -> Self {
        SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo: 0 }
    }
}

/// Split windows into train / val / holdout by **origin date**, with an embargo gap
/// (in trading bars along the *global* union calendar) between the sets so no
/// window's forecast horizon crosses a split boundary.
///
/// The global calendar is the sorted union of every date appearing in any series;
/// each origin date maps to a global bar index, and the timeline is cut at
/// `train_frac` / `val_frac` with `embargo` bars purged on each side of every cut.
pub fn temporal_split(series: &[Series], windows: &[WindowRef], horizon: usize, cfg: SplitConfig) -> Split {
    // Build the global ordered calendar (union of all trading dates).
    let mut all: Vec<i64> = series.iter().flat_map(|s| s.dates.iter().map(|&d| date_key(d))).collect();
    all.sort_unstable();
    all.dedup();
    if all.is_empty() {
        return Split::default();
    }
    let gidx = |key: i64| -> usize { all.partition_point(|&k| k < key) };
    let n = all.len();
    let embargo = cfg.embargo.max(horizon); // never below the purge minimum

    // Cut points on the global timeline.
    let train_end = ((n as f32) * cfg.train_frac) as usize;
    let val_end = ((n as f32) * (cfg.train_frac + cfg.val_frac)) as usize;

    let mut split = Split::default();
    for &w in windows {
        // origin date = the last context bar's date; horizon lives strictly after it.
        let origin_key = date_key(series[w.series_idx].dates[w.origin - 1]);
        let g = gidx(origin_key);
        // The window's realized future ends ~`horizon` bars after its origin; keep an
        // embargo band clear on both sides of each boundary so no label straddles it.
        if g + horizon <= train_end.saturating_sub(embargo) {
            split.train.push(w);
        } else if g >= train_end + embargo && g + horizon <= val_end.saturating_sub(embargo) {
            split.val.push(w);
        } else if g >= val_end + embargo {
            split.holdout.push(w);
        }
        // windows falling inside an embargo band are dropped (purged).
    }
    split
}

/// The extracted numeric content of one window (raw, un-normalized — each model
/// applies its own past-only scaling). `ctx`/`fut` are row-major `[len, 5]` OHLCV.
#[derive(Clone, Debug)]
pub struct Window {
    pub ticker: String,
    pub ctx: Vec<f32>,
    pub fut: Vec<f32>,
    pub ctx_dates: Vec<Date>,
    pub fut_dates: Vec<Date>,
}

impl Window {
    /// Calendar stamps `[len, 5]` = (minute, hour, weekday, day, month) for a date
    /// slice — daily bars carry minute=hour=0. Matches the reference stamp order.
    pub fn stamps(dates: &[Date]) -> Vec<u32> {
        let mut out = vec![0u32; dates.len() * 5];
        for (i, &(y, m, d)) in dates.iter().enumerate() {
            out[i * 5 + 2] = weekday(y, m, d);
            out[i * 5 + 3] = d;
            out[i * 5 + 4] = m;
        }
        out
    }
    pub fn ctx_stamps(&self) -> Vec<u32> {
        Self::stamps(&self.ctx_dates)
    }
    pub fn fut_stamps(&self) -> Vec<u32> {
        Self::stamps(&self.fut_dates)
    }
}

/// Materialize a window reference into its OHLCV + dates.
pub fn extract(series: &[Series], w: WindowRef, context: usize, horizon: usize) -> Window {
    let s = &series[w.series_idx];
    let flat = |rows: &[[f32; 5]]| -> Vec<f32> {
        let mut v = Vec::with_capacity(rows.len() * 5);
        for r in rows {
            v.extend_from_slice(r);
        }
        v
    };
    Window {
        ticker: s.ticker.clone(),
        ctx: flat(&s.ohlcv[w.origin - context..w.origin]),
        fut: flat(&s.ohlcv[w.origin..w.origin + horizon]),
        ctx_dates: s.dates[w.origin - context..w.origin].to_vec(),
        fut_dates: s.dates[w.origin..w.origin + horizon].to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(ticker: &str, n: usize, start_day: u32) -> Series {
        // one bar per day in 2025-01, ascending; simple ramp values.
        let dates: Vec<Date> = (0..n).map(|i| (2025, 1, start_day + i as u32)).collect();
        let ohlcv: Vec<[f32; 5]> =
            (0..n).map(|i| { let x = i as f32; [x, x + 1.0, x - 1.0, x + 0.5, 1000.0 + x] }).collect();
        Series { ticker: ticker.into(), dates, ohlcv }
    }

    #[test]
    fn weekday_matches_known_dates() {
        assert_eq!(weekday(2026, 7, 20), 0); // Monday
        assert_eq!(weekday(2026, 7, 21), 1); // Tuesday
        assert_eq!(weekday(1970, 1, 1), 3); // Thursday
    }

    #[test]
    fn enumerate_counts_valid_origins() {
        // n=20, context=5, horizon=3 -> origins 5..=17 = 13 windows.
        let s = vec![synth("A", 20, 1)];
        let w = enumerate_windows(&s, 5, 3);
        assert_eq!(w.len(), 13);
        assert_eq!(w[0].origin, 5);
        assert_eq!(w.last().unwrap().origin, 17);
        // too-short series contributes nothing.
        let s2 = vec![synth("B", 6, 1)];
        assert!(enumerate_windows(&s2, 5, 3).len() <= 1);
    }

    #[test]
    fn extract_shapes_and_dates() {
        let s = vec![synth("A", 20, 1)];
        let w = WindowRef { series_idx: 0, origin: 10 };
        let win = extract(&s, w, 5, 3);
        assert_eq!(win.ctx.len(), 5 * 5);
        assert_eq!(win.fut.len(), 3 * 5);
        // origin date is the last context bar (index origin-1 = 9 -> day 10).
        assert_eq!(win.ctx_dates.last().unwrap(), &(2025, 1, 10));
        assert_eq!(win.fut_dates[0], (2025, 1, 11));
        // stamps: weekday/day/month populated, minute/hour zero.
        let st = win.fut_stamps();
        assert_eq!(st[0 * 5 + 3], 11); // day
        assert_eq!(st[0 * 5 + 4], 1); // month
        assert_eq!(st[0 * 5], 0); // minute
    }

    #[test]
    fn temporal_split_has_no_horizon_leak_across_the_embargo() {
        // 200 bars, context 20, horizon 5, embargo 10 (>= horizon).
        let series = vec![synth("A", 200, 1)]; // Jan is short but date math only needs order
        let ctx = 20usize;
        let hor = 5usize;
        let windows = enumerate_windows(&series, ctx, hor);
        let cfg = SplitConfig { train_frac: 0.6, val_frac: 0.2, embargo: 10 };
        let sp = temporal_split(&series, &windows, hor, cfg);
        assert!(!sp.train.is_empty() && !sp.val.is_empty() && !sp.holdout.is_empty());

        // Core leak-safety invariant: the LAST realized future bar of any TRAIN window
        // must come strictly before the FIRST context bar of any VAL window — i.e. no
        // bar index is shared between a train window's horizon and a val window's input.
        let train_last_future = sp.train.iter().map(|w| w.origin + hor - 1).max().unwrap();
        let val_first_ctx = sp.val.iter().map(|w| w.origin - ctx).min().unwrap();
        assert!(train_last_future < val_first_ctx,
            "train future bar {train_last_future} leaks into val context starting {val_first_ctx}");

        // Same invariant across the val→holdout seam.
        let val_last_future = sp.val.iter().map(|w| w.origin + hor - 1).max().unwrap();
        let hold_first_ctx = sp.holdout.iter().map(|w| w.origin - ctx).min().unwrap();
        assert!(val_last_future < hold_first_ctx);
    }

    #[test]
    fn embargo_is_floored_at_horizon_even_if_zero_requested() {
        // Requesting embargo 0 must still purge >= horizon so labels can't straddle.
        let series = vec![synth("A", 120, 1)];
        let (ctx, hor) = (10usize, 8usize);
        let windows = enumerate_windows(&series, ctx, hor);
        let cfg = SplitConfig { train_frac: 0.5, val_frac: 0.25, embargo: 0 };
        let sp = temporal_split(&series, &windows, hor, cfg);
        if !sp.train.is_empty() && !sp.val.is_empty() {
            let tlf = sp.train.iter().map(|w| w.origin + hor - 1).max().unwrap();
            let vfc = sp.val.iter().map(|w| w.origin - ctx).min().unwrap();
            assert!(tlf < vfc);
        }
    }

    #[test]
    fn multi_symbol_windows_never_cross_series() {
        let series = vec![synth("A", 60, 1), synth("B", 60, 1)];
        let w = enumerate_windows(&series, 10, 5);
        // every window's series_idx is a valid index and content comes from one series.
        for wr in &w {
            assert!(wr.series_idx < 2);
            let win = extract(&series, *wr, 10, 5);
            assert_eq!(win.ctx.len(), 50);
        }
    }
}
