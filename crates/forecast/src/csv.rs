// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! OHLCV CSV in, a validated [`Panel`] out - the boundary where a file brain
//! did not itself produce becomes a tensor.
//!
//! A CSV handed to brain from outside is untrusted, so it is checked here,
//! once, **structurally and semantically**, before a single float reaches a
//! model. The alternative is what this module exists to prevent: a ragged row
//! or a NaN turning into a confusing panic deep inside the tokenizer's
//! normalization, where nothing left on the stack says "line 431 of your CSV
//! has five fields".
//!
//! Structural, in file order:
//!
//! - a header naming the six expected columns (case-insensitive, in order);
//! - every data row has exactly six comma-separated fields;
//! - every numeric field parses as `f32`, every timestamp parses as one of the
//!   three accepted ISO-8601 shapes.
//!
//! Semantic - the checks a shape-only parser passes and a forecaster then
//! chokes on:
//!
//! - timestamps strictly increasing (a duplicate or a backwards step means the
//!   rows are not one series in time order, and every window a forecaster cuts
//!   would silently straddle the discontinuity);
//! - every value finite (no `NaN`, no infinity);
//! - prices strictly positive and volume non-negative;
//! - the OHLC invariants `high >= max(open, close)`, `low <= min(open, close)`
//!   and `high >= low` - a bar violating them is not a bar, and a
//!   z-score-then-clip preprocessor would happily consume it;
//! - enough rows for the requested context + horizon, reported as a count
//!   rather than an empty-slice panic downstream.
//!
//! Every error names the 1-based **file line number** (header included, the
//! number an editor shows), the column, and what was expected.
//!
//! The calendar is derived here too, not guessed: Kronos consumes per-bar
//! `minute`/`hour`/`weekday`/`day`/`month` indices, and a bar index cannot be
//! mapped back to a date from a `start` + `freq` pair once a series has gaps.
//! [`OhlcvSeries::panel`] emits them as five known-future variates alongside
//! the five OHLCV ones, filling each one's `future` from the held-out rows'
//! own timestamps.

use crate::panel::{Item, Kind, Panel, Role, Variate};

/// The exact header this parser accepts, in order. The first column's name is
/// matched loosely (`timestamp`, `date`, `datetime` and `time` are all in
/// circulation and all mean the same thing here); the five that carry the bar
/// are matched exactly, because a file that spells them differently is a file
/// whose column ORDER we have no right to assume.
const TS_ALIASES: [&str; 4] = ["timestamp", "date", "datetime", "time"];
const OHLCV_HEADER: [&str; 5] = ["open", "high", "low", "close", "volume"];

/// The calendar variate names Kronos reads, in its own stamp order.
const CAL_NAMES: [&str; 5] = ["minute", "hour", "weekday", "day", "month"];

/// One parsed timestamp, already decomposed into the calendar indices a
/// forecaster consumes. `weekday` is Monday = 0 (pandas `.weekday()`, which is
/// what the Kronos reference feeds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub weekday: u32,
}

impl Stamp {
    /// Minutes since the epoch-ish origin used ONLY for ordering. Not a real
    /// epoch conversion (no leap seconds, no timezone): a strictly monotone
    /// map from (y, m, d, h, min) to an integer is all the ordering check
    /// needs, and a real calendar conversion would be a second thing to get
    /// wrong for no gain.
    fn order_key(&self) -> i64 {
        ((self.year as i64 * 12 + self.month as i64) * 31 + self.day as i64) * 1440 + self.hour as i64 * 60 + self.minute as i64
    }

    /// The calendar index for `name`, one of [`CAL_NAMES`].
    fn cal(&self, name: &str) -> f32 {
        match name {
            "minute" => self.minute as f32,
            "hour" => self.hour as f32,
            "weekday" => self.weekday as f32,
            "day" => self.day as f32,
            _ => self.month as f32,
        }
    }
}

/// Days in `month` of `year`, for the calendar range check.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap {
                29
            } else {
                28
            }
        }
    }
}

/// One validated bar: its timestamp and its five channels, in
/// [`OHLCV_HEADER`] order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    pub stamp: Stamp,
    /// `[open, high, low, close, volume]`.
    pub ohlcv: [f32; 5],
}

/// A whole validated CSV: every invariant in this module's docs already holds
/// for every bar, so a consumer never re-checks and never guards.
#[derive(Clone, Debug)]
pub struct OhlcvSeries {
    pub bars: Vec<Bar>,
}

/// Column index of `close` within [`Bar::ohlcv`] - the channel a univariate
/// score or chart is drawn from.
pub const CLOSE: usize = 3;

/// Parse and fully validate an OHLCV CSV. `Err` is a single human-readable
/// line naming the file line number and what was wrong.
pub fn parse_ohlcv(text: &str) -> Result<OhlcvSeries, String> {
    let mut lines = text.lines().enumerate();

    // Skip a UTF-8 BOM and any leading blank lines, so a file saved by a
    // spreadsheet is not rejected for a reason the user cannot see.
    let (header_no, header) = loop {
        let Some((i, l)) = lines.next() else {
            return Err("csv: no header row (file is empty)".to_string());
        };
        let l = l.strip_prefix('\u{feff}').unwrap_or(l).trim();
        if !l.is_empty() {
            break (i + 1, l);
        }
    };

    let cols: Vec<&str> = header.split(',').map(str::trim).collect();
    if cols.len() != 6 {
        return Err(format!("csv: line {header_no}: header has {} columns, expected 6 (timestamp,open,high,low,close,volume)", cols.len()));
    }
    if !TS_ALIASES.iter().any(|a| cols[0].eq_ignore_ascii_case(a)) {
        return Err(format!("csv: line {header_no}: first column is {:?}, expected one of {TS_ALIASES:?}", cols[0]));
    }
    for (i, want) in OHLCV_HEADER.iter().enumerate() {
        if !cols[i + 1].eq_ignore_ascii_case(want) {
            return Err(format!("csv: line {header_no}: column {} is {:?}, expected {want:?} (columns must be timestamp,open,high,low,close,volume in that order)", i + 2, cols[i + 1]));
        }
    }

    let mut bars: Vec<Bar> = Vec::new();
    let mut last_key: Option<i64> = None;
    for (i, line) in lines {
        let no = i + 1;
        let line = line.trim();
        if line.is_empty() {
            continue; // a trailing newline, or a blank separator row
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() != 6 {
            return Err(format!("csv: line {no}: {} fields, expected 6", f.len()));
        }
        let stamp = parse_stamp(f[0]).map_err(|e| format!("csv: line {no}: column 1 (timestamp): {e}"))?;
        let key = stamp.order_key();
        if let Some(prev) = last_key {
            if key <= prev {
                return Err(format!("csv: line {no}: timestamp {:?} is not after the previous row's - rows must be strictly increasing in time", f[0]));
            }
        }
        last_key = Some(key);

        let mut ohlcv = [0.0f32; 5];
        for (c, raw) in f[1..].iter().enumerate() {
            let v: f32 = raw
                .parse()
                .map_err(|_| format!("csv: line {no}: column {} ({}): {raw:?} is not a number", c + 2, OHLCV_HEADER[c]))?;
            if !v.is_finite() {
                return Err(format!("csv: line {no}: column {} ({}): {raw:?} is not finite", c + 2, OHLCV_HEADER[c]));
            }
            ohlcv[c] = v;
        }
        let [o, h, l, c, v] = ohlcv;
        for (name, price) in OHLCV_HEADER.iter().zip([o, h, l, c]) {
            if price <= 0.0 {
                return Err(format!("csv: line {no}: {name} is {price}, expected a positive price"));
            }
        }
        if v < 0.0 {
            return Err(format!("csv: line {no}: volume is {v}, expected >= 0"));
        }
        if h < l {
            return Err(format!("csv: line {no}: high {h} < low {l}"));
        }
        if h < o.max(c) {
            return Err(format!("csv: line {no}: high {h} is below open {o} / close {c} - not a valid bar"));
        }
        if l > o.min(c) {
            return Err(format!("csv: line {no}: low {l} is above open {o} / close {c} - not a valid bar"));
        }
        bars.push(Bar { stamp, ohlcv });
    }

    if bars.is_empty() {
        return Err("csv: header parsed but no data rows".to_string());
    }
    Ok(OhlcvSeries { bars })
}

/// Accepted timestamp shapes: `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM[:SS]` and the
/// same with a space separator. A trailing `Z` is accepted and ignored - this
/// parser has no timezone model, and pretending otherwise by silently shifting
/// hours would corrupt the `hour` calendar feature.
fn parse_stamp(s: &str) -> Result<Stamp, String> {
    let s = s.trim_end_matches('Z');
    let (date, time) = match s.split_once(['T', ' ']) {
        Some((d, t)) => (d, Some(t)),
        None => (s, None),
    };
    let d: Vec<&str> = date.split('-').collect();
    if d.len() != 3 {
        return Err(format!("{s:?} is not YYYY-MM-DD[THH:MM[:SS]]"));
    }
    let year: i32 = d[0].parse().map_err(|_| format!("{:?} is not a year", d[0]))?;
    let month: u32 = d[1].parse().map_err(|_| format!("{:?} is not a month", d[1]))?;
    let day: u32 = d[2].parse().map_err(|_| format!("{:?} is not a day", d[2]))?;
    if year < 1 {
        return Err(format!("year {year} is out of range"));
    }
    if !(1..=12).contains(&month) {
        return Err(format!("month {month} is out of range 1..=12"));
    }
    let dim = days_in_month(year, month);
    if day < 1 || day > dim {
        return Err(format!("day {day} is out of range 1..={dim} for {year}-{month:02}"));
    }

    let (hour, minute) = match time {
        None => (0, 0),
        Some(t) => {
            let p: Vec<&str> = t.split(':').collect();
            if p.len() < 2 || p.len() > 3 {
                return Err(format!("{t:?} is not HH:MM[:SS]"));
            }
            let hour: u32 = p[0].parse().map_err(|_| format!("{:?} is not an hour", p[0]))?;
            let minute: u32 = p[1].parse().map_err(|_| format!("{:?} is not a minute", p[1]))?;
            if let Some(sec) = p.get(2) {
                let sec: f64 = sec.parse().map_err(|_| format!("{sec:?} is not a second"))?;
                if !(0.0..60.0).contains(&sec) {
                    return Err(format!("second {sec} is out of range 0..60"));
                }
            }
            if hour > 23 {
                return Err(format!("hour {hour} is out of range 0..=23"));
            }
            if minute > 59 {
                return Err(format!("minute {minute} is out of range 0..=59"));
            }
            (hour, minute)
        }
    };
    // Monday = 0, from the crate's one days-from-civil implementation -- the
    // same function that builds the calendar stamps everywhere else.
    Ok(Stamp { year, month, day, hour, minute, weekday: crate::train_data::weekday(year, month, day) })
}

/// How a series was split into what the model sees and what it is scored
/// against - the shape a self-contained "is the forecast any good" run needs.
#[derive(Clone, Debug)]
pub struct Split {
    /// The bars fed to the model, at most `context` of them, in time order.
    pub context: Vec<Bar>,
    /// The held-out continuation, exactly `horizon` bars.
    pub actual: Vec<Bar>,
}

impl OhlcvSeries {
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// Hold out the last `horizon` bars as ground truth and keep at most
    /// `context` bars before them as the model's input.
    ///
    /// This is the semantic length check the module docs promise, and it is
    /// deliberately here rather than at parse time: how many rows are "enough"
    /// is a property of the REQUEST (context + horizon), not of the file.
    pub fn split(&self, context: usize, horizon: usize) -> Result<Split, String> {
        self.split_at_origin(context, horizon, 0)
    }

    /// [`split`](Self::split), rolled `back` bars earlier in the file: origin
    /// `0` is the end of the file, origin `back` holds out the `horizon` bars
    /// ending `back` rows from the end.
    ///
    /// One origin is one draw. A model that beats a baseline at one cut and
    /// loses at the next has demonstrated nothing, and a forecast scored at a
    /// single origin is an anecdote - which is why the CLI averages over
    /// several of these rather than quoting the last one.
    pub fn split_at_origin(&self, context: usize, horizon: usize, back: usize) -> Result<Split, String> {
        if horizon == 0 {
            return Err("csv: horizon must be >= 1".to_string());
        }
        if context == 0 {
            return Err("csv: context must be >= 1".to_string());
        }
        let need = context + horizon + back;
        if self.bars.len() < need {
            return Err(format!(
                "csv: {} rows is too few for a {context}-bar context plus a {horizon}-bar held-out horizon{} ({need} needed) - shorten --context/--horizon/--origins or supply a longer file",
                self.bars.len(),
                if back > 0 { format!(" at an origin {back} bars back") } else { String::new() },
            ));
        }
        let cut = self.bars.len() - horizon - back;
        Ok(Split { context: self.bars[cut - context..cut].to_vec(), actual: self.bars[cut..cut + horizon].to_vec() })
    }
}

/// Build the [`Panel`] a forecaster consumes from a [`Split`]: the five OHLCV
/// channels as variates (`close` the target, the rest past covariates) plus
/// the five calendar variates, each carrying the held-out rows' own stamps as
/// its known-future path.
pub fn panel(split: &Split, item_id: &str, freq: &str) -> Panel {
    let mut variates: Vec<Variate> = Vec::with_capacity(10);
    for (c, name) in OHLCV_HEADER.iter().enumerate() {
        variates.push(Variate {
            name: (*name).to_string(),
            role: if c == CLOSE { Role::Target } else { Role::PastCovariate },
            kind: Kind::Continuous,
            data: split.context.iter().map(|b| b.ohlcv[c]).collect(),
            future: None,
            observed: None,
            cardinality: None,
        });
    }
    for name in CAL_NAMES {
        variates.push(Variate {
            name: name.to_string(),
            role: Role::KnownFuture,
            kind: Kind::Categorical,
            data: split.context.iter().map(|b| b.stamp.cal(name)).collect(),
            future: Some(split.actual.iter().map(|b| b.stamp.cal(name)).collect()),
            observed: None,
            cardinality: None,
        });
    }
    let start = split.context.first().map(|b| {
        let s = b.stamp;
        format!("{:04}-{:02}-{:02}T{:02}:{:02}:00", s.year, s.month, s.day, s.hour, s.minute)
    });
    Panel { freq: freq.to_string(), start, items: vec![Item::new(item_id, variates)] }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal well-formed file: three hourly bars whose OHLC invariants
    /// hold. Every negative test below is this, with exactly one thing broken.
    const GOOD: &str = "timestamp,open,high,low,close,volume\n\
        2026-01-05T00:00:00,100.0,101.0,99.5,100.5,1000\n\
        2026-01-05T01:00:00,100.5,102.0,100.0,101.5,1200\n\
        2026-01-05T02:00:00,101.5,103.0,101.0,102.5,900\n";

    #[test]
    fn parses_a_well_formed_file() {
        let s = parse_ohlcv(GOOD).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.bars[0].ohlcv, [100.0, 101.0, 99.5, 100.5, 1000.0]);
        assert_eq!(s.bars[2].stamp.hour, 2);
        // 2026-01-05 is a Monday: weekday 0 in the Monday-first convention the
        // Kronos reference uses.
        assert_eq!(s.bars[0].stamp.weekday, 0);
    }

    #[test]
    fn accepts_the_timestamp_spellings_in_circulation() {
        for (first, rest) in [("date", "2026-01-05"), ("datetime", "2026-01-05 00:00"), ("time", "2026-01-05T00:00:00Z")] {
            let text = format!("{first},open,high,low,close,volume\n{rest},100.0,101.0,99.5,100.5,1000\n");
            assert!(parse_ohlcv(&text).is_ok(), "{first}/{rest} should parse");
        }
    }

    #[test]
    fn a_bom_and_blank_lines_do_not_hide_the_header() {
        let text = format!("\u{feff}\n\n{GOOD}");
        assert_eq!(parse_ohlcv(&text).unwrap().len(), 3);
    }

    /// Each of these is the ONE thing this parser exists to catch. The
    /// assertion is on the message, not just on `is_err`: an error that does
    /// not say which line and what was expected is the confusing failure this
    /// module replaces.
    #[test]
    fn rejects_every_structural_defect_with_a_located_message() {
        let cases: [(&str, &str); 6] = [
            ("", "no header"),
            ("timestamp,open,high,low,close\n", "header has 5 columns"),
            ("timestamp,open,high,low,volume,close\n2026-01-05,1,2,0.5,3,4\n", "column 5"),
            ("timestamp,open,high,low,close,volume\n2026-01-05,100,101,99.5,100.5\n", "line 2: 5 fields"),
            ("timestamp,open,high,low,close,volume\n2026-01-05,100,101,99.5,abc,4\n", "is not a number"),
            ("timestamp,open,high,low,close,volume\n", "no data rows"),
        ];
        for (text, want) in cases {
            let e = parse_ohlcv(text).unwrap_err();
            assert!(e.contains(want), "{text:?}\n  error {e:?} does not mention {want:?}");
        }
    }

    #[test]
    fn rejects_every_semantic_defect_with_a_located_message() {
        let row = |ts: &str, o: &str, h: &str, l: &str, c: &str, v: &str| {
            format!("timestamp,open,high,low,close,volume\n2026-01-05T00:00:00,100,101,99,100,10\n{ts},{o},{h},{l},{c},{v}\n")
        };
        let cases: [(String, &str); 8] = [
            // Not strictly increasing: equal, then backwards.
            (row("2026-01-05T00:00:00", "100", "101", "99", "100", "10"), "not after the previous"),
            (row("2026-01-04T23:00:00", "100", "101", "99", "100", "10"), "not after the previous"),
            // `"NaN"` and `"inf"` PARSE as f32 -- catching them is exactly why
            // the finiteness check is separate from the parse.
            (row("2026-01-05T01:00:00", "100", "NaN", "99", "100", "10"), "is not finite"),
            (row("2026-01-05T01:00:00", "100", "inf", "99", "100", "10"), "is not finite"),
            (row("2026-01-05T01:00:00", "0", "101", "99", "100", "10"), "expected a positive price"),
            (row("2026-01-05T01:00:00", "100", "101", "99", "100", "-1"), "expected >= 0"),
            // high below the close, and low above the open: the two OHLC
            // invariants a shape-only parser waves through.
            (row("2026-01-05T01:00:00", "100", "101", "99", "102", "10"), "is below open"),
            (row("2026-01-05T01:00:00", "100", "101", "99.5", "99.2", "10"), "is above open"),
        ];
        for (text, want) in cases {
            let e = parse_ohlcv(&text).unwrap_err();
            assert!(e.contains(want), "error {e:?} does not mention {want:?}");
        }
    }

    #[test]
    fn rejects_an_impossible_calendar_date() {
        for (ts, want) in [("2026-02-29", "out of range 1..=28"), ("2026-13-01", "out of range 1..=12"), ("2026-01-05T24:00", "out of range 0..=23")] {
            let text = format!("timestamp,open,high,low,close,volume\n{ts},100,101,99,100,10\n");
            let e = parse_ohlcv(&text).unwrap_err();
            assert!(e.contains(want), "{ts}: error {e:?} does not mention {want:?}");
        }
        // ... and accepts the leap day that 2026-02-29 is not.
        let text = "timestamp,open,high,low,close,volume\n2024-02-29,100,101,99,100,10\n";
        assert!(parse_ohlcv(text).is_ok());
    }

    #[test]
    fn split_refuses_a_file_too_short_for_the_request_and_says_by_how_much() {
        let s = parse_ohlcv(GOOD).unwrap();
        let e = s.split(4, 2).unwrap_err();
        assert!(e.contains("3 rows is too few") && e.contains("6 needed"), "{e}");
        assert!(s.split(1, 0).is_err(), "a zero horizon is not a forecast");
        assert!(s.split(0, 1).is_err(), "a zero context is not an input");
    }

    #[test]
    fn split_holds_out_the_tail_and_keeps_the_bars_immediately_before_it() {
        let s = parse_ohlcv(GOOD).unwrap();
        let sp = s.split(2, 1).unwrap();
        assert_eq!(sp.context.len(), 2);
        assert_eq!(sp.actual.len(), 1);
        assert_eq!(sp.context[0].stamp.hour, 0);
        assert_eq!(sp.context[1].stamp.hour, 1);
        assert_eq!(sp.actual[0].stamp.hour, 2);
        // The context is the bars ADJACENT to the held-out tail, never an
        // earlier window with a gap in between.
        assert_eq!(sp.context[1].ohlcv[CLOSE], 101.5);
    }

    #[test]
    fn rolling_the_origin_back_moves_both_windows_and_never_leaks() {
        let s = parse_ohlcv(GOOD).unwrap();
        let now = s.split_at_origin(1, 1, 0).unwrap();
        let prev = s.split_at_origin(1, 1, 1).unwrap();
        assert_eq!(now.actual[0].stamp.hour, 2);
        assert_eq!(prev.actual[0].stamp.hour, 1);
        // The earlier origin's context ends BEFORE its own held-out bar - the
        // window slides, it does not just re-label the same rows.
        assert_eq!(prev.context[0].stamp.hour, 0);
        // Rolling past the start of the file is refused, not silently clamped
        // to origin 0 (which would score the same window twice and call it two
        // independent draws).
        assert!(s.split_at_origin(1, 1, 2).is_err());
    }

    #[test]
    fn panel_carries_ten_variates_and_the_future_calendar() {
        let s = parse_ohlcv(GOOD).unwrap();
        let sp = s.split(2, 1).unwrap();
        let p = panel(&sp, "SYN", "1h");
        let item = &p.items[0];
        assert_eq!(item.item_id, "SYN");
        assert_eq!(item.variates.len(), 10);
        // close is the target, the other four bar channels are past covariates.
        assert_eq!(item.variate("close").unwrap().role, Role::Target);
        assert_eq!(item.variate("open").unwrap().role, Role::PastCovariate);
        assert_eq!(item.variate("close").unwrap().data, vec![100.5, 101.5]);
        // The calendar's `future` is the HELD-OUT bar's own stamp, which is
        // what makes it a known-future covariate rather than a guess.
        let hour = item.variate("hour").unwrap();
        assert_eq!(hour.data, vec![0.0, 1.0]);
        assert_eq!(hour.future.as_deref(), Some(&[2.0f32][..]));
        assert_eq!(item.variate("month").unwrap().future.as_deref(), Some(&[1.0f32][..]));
        assert_eq!(p.start.as_deref(), Some("2026-01-05T00:00:00"));
    }
}
