// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The patient-history update format: the stable external way to say "this
//! subject's record, as of now" and ask for a new prediction.
//!
//! Swedish Embedded AB implements record-to-risk interfaces whose inputs are
//! validated, deterministic and auditable, for its clients. If your team needs
//! expertise in turning longitudinal records into safe model inputs you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! ```json
//! {"id": "p1", "as_of": "2026-03-01", "birth": "1970-06-15",
//!  "static": {"sex": "female"},
//!  "events": [{"time": "2019-04-02", "code": "dx:diabetes"},
//!             {"time": "2026-02-20", "code": "ldl", "value": 3.1, "unit": "mmol/L"},
//!             {"time": 55.1, "code": "crp", "value": {"below": 0.2}}]}
//! ```
//!
//! Inference from a history is stateless: the whole history goes in, the
//! model's weights never change for a patient and nothing is kept between
//! calls. "Append a checkup" means send the history again with one more
//! record.
//!
//! # Time
//!
//! A `time` (and `as_of`) is either
//!
//! - a JSON **number**: the model's own clock, in years (attained age for a
//!   person), used as it is; or
//! - a **string**, an ISO-8601 / RFC 3339 date (`YYYY-MM-DD`, midnight UTC) or
//!   datetime (`YYYY-MM-DDTHH:MM:SS[.fraction](Z|+HH:MM|-HH:MM)`, normalised to
//!   UTC; leap seconds are refused), which needs `birth` (same syntax) and
//!   becomes `(instant - birth) / YEAR_SECONDS`, with the year a fixed
//!   [`YEAR_SECONDS`] (the Julian year, 365.25 days) so the conversion is
//!   exact integer arithmetic on nanoseconds followed by ONE floating-point
//!   division, identical on every machine. A date before `birth` is an error.
//!
//! The model also reads the calendar time at the prediction time
//! (`calendar_at_entry`). It is the decimal year of `as_of` when that is a
//! date or `birth` is given (`Y + (instant - Y-01-01T00:00Z) / length of
//! calendar year Y`, UTC); for a numeric `as_of` with no `birth` it cannot be
//! derived and the history must give `"calendar"` (the decimal year at
//! `as_of`). `calendar` given where it could be derived is an error: there
//! would be two sources of truth.
//!
//! # Records
//!
//! Each element of `events` is `{time, code, value?, unit?}`. With a `value`
//! it is a MEASUREMENT of variable `code`: a number, `{"below": x}`,
//! `{"above": x}` (a detection limit) or a category string, optionally with
//! the `unit` it is stated in (never for a category). Without one it is an
//! EVENT `code` that happened at `time`. `static` maps a variable to a value
//! (or `{"value": v, "unit": u}`) known at `as_of`, a measurement at `as_of`.
//!
//! # Rules, all deterministic
//!
//! - **Future**: a record dated after `as_of` is never fed to the model: it is
//!   dropped and reported ([`HistoryWarning::FutureIgnored`]). Measurements
//!   AT `as_of` are known; an event at `as_of` is not history (as in
//!   `timeline-v1`, events count strictly before the prediction time), so it
//!   is dropped and reported ([`HistoryWarning::EventAtAsOf`]).
//! - **Duplicates**: a record equal to another in time, code, value and unit is
//!   dropped and reported ([`HistoryWarning::DuplicateIgnored`]), so sending
//!   the same record twice changes nothing. Two measurements of one code at
//!   one time that differ in value or unit contradict each other: an error.
//! - **Order**: records are sorted by (time, code, value, unit) before they
//!   reach the model, so the order they are listed in never matters.
//! - **Unknown fields**, at any level, are never read; each is reported
//!   ([`HistoryWarning::UnknownField`]).
//! - **Units** are compared with the model's ([`PatientHistory::to_subject`]):
//!   a mismatch is an error, never a conversion.
//! - **Bounds**: at most [`MAX_RECORDS`] records, [`MAX_STATIC`] static
//!   values, [`MAX_TEXT_LEN`] bytes of text per input, [`MAX_NAME_LEN`]
//!   characters in a code or category and [`crate::timeline::MAX_UNIT_LEN`] in
//!   a unit.
//!
//! Every error names the record (`events[3]`, `static.sex`) and field.

use std::cmp::Ordering;

use serde::Serialize;
use serde_json::{Map, Value as Json};

use crate::timeline::{check_unit, Event, Observation, Subject, Value};
use crate::vocab::Vocab;

/// The length of a year on the model's clock when converting dates: 365.25
/// days of 86 400 seconds.
pub const YEAR_SECONDS: i64 = 31_557_600;
/// The most records (events plus measurements) one history may carry.
pub const MAX_RECORDS: usize = 10_000;
/// The most `static` values one history may carry.
pub const MAX_STATIC: usize = 1_000;
/// The longest text accepted in one call, in bytes.
pub const MAX_TEXT_LEN: usize = 8 * 1024 * 1024;
/// The most histories one call may carry.
pub const MAX_HISTORIES: usize = 1_024;
/// The longest code, variable name or category, in characters.
pub const MAX_NAME_LEN: usize = 128;
/// The longest time string, in characters.
const MAX_TIME_LEN: usize = 48;

/// Something about the input that was not used or was changed, reported and
/// never silent. None of them reaches the model.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistoryWarning {
    /// A field this format does not define.
    UnknownField {
        /// Where: `history`, `events[3]` or `static.name`.
        scope: String,
        /// The field.
        field: String,
    },
    /// A record dated after `as_of`: the future, dropped.
    FutureIgnored {
        /// `events[i]` or `static.name`.
        record: String,
        /// Its code.
        code: String,
        /// Its time on the model's clock.
        time: f64,
    },
    /// An event dated exactly `as_of`: not history, dropped.
    EventAtAsOf {
        /// `events[i]`.
        record: String,
        /// Its code.
        code: String,
    },
    /// A record equal to an earlier one, dropped.
    DuplicateIgnored {
        /// `events[i]` or `static.name`.
        record: String,
        /// Its code.
        code: String,
        /// Its time on the model's clock.
        time: f64,
    },
}

/// One record of the history, on the model's clock.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// When, on the model's clock (years).
    pub time: f64,
    /// The event code, or the measured variable when there is a value.
    pub code: String,
    /// The measured value; `None` for an event.
    pub value: Option<Value>,
    /// The unit of a numeric value, if stated.
    pub unit: Option<String>,
}

/// A validated, canonical history: everything known at `as_of`, sorted and
/// without duplicates, and the warnings about what was left out.
#[derive(Clone, Debug, PartialEq)]
pub struct PatientHistory {
    /// An identifier for errors and answers (`history` when none is given).
    pub id: String,
    /// The prediction time on the model's clock.
    pub as_of: f64,
    /// `as_of` as it was written.
    pub as_of_input: String,
    /// The calendar time (decimal year) at `as_of`.
    pub calendar_at_as_of: f64,
    /// Records at or before `as_of`, in canonical order.
    pub records: Vec<Record>,
    /// What was dropped or ignored, in input order.
    pub warnings: Vec<HistoryWarning>,
}

/// A point in time to the nanosecond, UTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Instant(i128);

const NANOS: i128 = 1_000_000_000;

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(y) => 29,
        _ => 28,
    }
}

/// `text[from..to]` as digits, exactly.
fn digits(text: &[u8], from: usize, to: usize) -> Result<i64, String> {
    let part = text
        .get(from..to)
        .filter(|p| p.iter().all(u8::is_ascii_digit))
        .ok_or("not a YYYY-MM-DD[THH:MM:SS[.f](Z|+HH:MM)] date")?;
    Ok(part.iter().fold(0i64, |a, d| a * 10 + i64::from(d - b'0')))
}

impl Instant {
    /// ISO-8601 / RFC 3339 date or datetime (see the module docs).
    fn parse(text: &str) -> Result<Instant, String> {
        if text.len() > MAX_TIME_LEN || !text.is_ascii() {
            return Err(format!("time {text:?} is not an ISO-8601 date or datetime"));
        }
        Self::parse_ascii(text.as_bytes()).map_err(|e| format!("time {text:?}: {e}"))
    }

    fn parse_ascii(b: &[u8]) -> Result<Instant, String> {
        if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
            return Err("not a YYYY-MM-DD[THH:MM:SS[.f](Z|+HH:MM)] date".into());
        }
        let (y, m, d) = (digits(b, 0, 4)?, digits(b, 5, 7)?, digits(b, 8, 10)?);
        if y < 1 || !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
            return Err("no such calendar date".into());
        }
        let mut secs = days_from_civil(y, m, d) * 86_400;
        let mut nanos = 0i64;
        if b.len() > 10 {
            if !matches!(b[10], b'T' | b't') || b.len() < 20 || b[13] != b':' || b[16] != b':' {
                return Err("a datetime is YYYY-MM-DDTHH:MM:SS followed by a UTC designator Z or an offset +HH:MM".into());
            }
            let (hh, mm, ss) = (digits(b, 11, 13)?, digits(b, 14, 16)?, digits(b, 17, 19)?);
            if hh > 23 || mm > 59 || ss > 59 {
                return Err("time of day out of range (leap seconds are refused)".into());
            }
            secs += hh * 3_600 + mm * 60 + ss;
            let mut at = 19;
            if b[at] == b'.' {
                let end = (at + 1..b.len()).find(|&i| !b[i].is_ascii_digit()).unwrap_or(b.len());
                if end == at + 1 || end - at - 1 > 9 {
                    return Err("a fraction of a second has 1 to 9 digits".into());
                }
                nanos = digits(b, at + 1, end)? * 10i64.pow(9 - (end - at - 1) as u32);
                at = end;
            }
            match b.get(at..) {
                Some(b"Z") | Some(b"z") => {}
                Some(off) if off.len() == 6 && matches!(off[0], b'+' | b'-') && off[3] == b':' => {
                    let (oh, om) = (digits(off, 1, 3)?, digits(off, 4, 6)?);
                    if oh > 23 || om > 59 {
                        return Err("UTC offset out of range".into());
                    }
                    let offset = oh * 3_600 + om * 60;
                    secs -= if off[0] == b'+' { offset } else { -offset };
                }
                _ => return Err("a datetime needs a UTC designator Z or an offset +HH:MM".into()),
            }
        }
        Ok(Instant(i128::from(secs) * NANOS + i128::from(nanos)))
    }

    /// Years from `origin` to `self` on the model's clock.
    fn years_since(self, origin: Instant) -> f64 {
        (self.0 - origin.0) as f64 / (YEAR_SECONDS as f64 * NANOS as f64)
    }

    fn plus_years(self, years: f64) -> Instant {
        Instant(self.0 + (years * YEAR_SECONDS as f64 * NANOS as f64).round() as i128)
    }

    /// The decimal year: `Y + (instant - Y-01-01T00:00Z) / length of Y`.
    fn decimal_year(self) -> f64 {
        let day = self.0.div_euclid(86_400 * NANOS) as i64;
        let mut y = 1970 + (day as f64 / 365.2425).floor() as i64;
        let start = |y: i64| i128::from(days_from_civil(y, 1, 1)) * 86_400 * NANOS;
        while start(y) > self.0 {
            y -= 1;
        }
        while start(y + 1) <= self.0 {
            y += 1;
        }
        y as f64 + (self.0 - start(y)) as f64 / (start(y + 1) - start(y)) as f64
    }
}

/// A time as written: the model's clock, or an instant to convert.
enum Time {
    Clock(f64),
    At(Instant),
}

struct Context<'a> {
    id: &'a str,
}

impl Context<'_> {
    fn err(&self, record: &str, field: &str, what: impl std::fmt::Display) -> String {
        format!("history {:?}: {record}: field {field:?}: {what}", self.id)
    }
}

fn parse_name(ctx: &Context, record: &str, field: &str, v: &Json) -> Result<String, String> {
    let s = v.as_str().ok_or_else(|| ctx.err(record, field, "must be a string"))?;
    if s.is_empty() || s.chars().count() > MAX_NAME_LEN || s.chars().any(char::is_control) {
        return Err(ctx.err(
            record,
            field,
            format!("must be 1 to {MAX_NAME_LEN} characters without control characters"),
        ));
    }
    Ok(s.to_string())
}

fn parse_time(ctx: &Context, record: &str, field: &str, v: &Json) -> Result<Time, String> {
    match v {
        Json::Number(n) => match n.as_f64() {
            Some(t) if t.is_finite() => Ok(Time::Clock(t)),
            _ => Err(ctx.err(record, field, "is not a finite number")),
        },
        Json::String(s) => Instant::parse(s).map(Time::At).map_err(|e| ctx.err(record, field, e)),
        _ => Err(ctx.err(
            record,
            field,
            "must be a number (the model's clock, in years) or an ISO-8601 date string",
        )),
    }
}

fn parse_value(ctx: &Context, record: &str, v: &Json) -> Result<Value, String> {
    let bad = || {
        ctx.err(
            record,
            "value",
            "must be a number, {\"below\": x}, {\"above\": x} or a category string",
        )
    };
    match v {
        Json::Number(n) => n.as_f64().filter(|x| x.is_finite()).map(Value::Number).ok_or_else(bad),
        Json::String(_) => Ok(Value::Category(parse_name(ctx, record, "value", v)?)),
        Json::Object(o) if o.len() == 1 => {
            let (key, limit) = o.iter().next().expect("one entry");
            let x = limit.as_f64().filter(|x| x.is_finite()).ok_or_else(bad)?;
            match key.as_str() {
                "below" => Ok(Value::Below { below: x }),
                "above" => Ok(Value::Above { above: x }),
                _ => Err(bad()),
            }
        }
        _ => Err(bad()),
    }
}

fn parse_unit(ctx: &Context, record: &str, v: &Json, value: Option<&Value>) -> Result<String, String> {
    let unit = v.as_str().ok_or_else(|| ctx.err(record, "unit", "must be a string"))?;
    check_unit(unit).map_err(|e| ctx.err(record, "unit", e))?;
    match value {
        None => Err(ctx.err(record, "unit", "a unit needs a value: an event has none")),
        Some(Value::Category(_)) => Err(ctx.err(record, "unit", "a category has no unit")),
        Some(_) => Ok(unit.to_string()),
    }
}

fn note_unknown(warnings: &mut Vec<HistoryWarning>, scope: &str, o: &Map<String, Json>, known: &[&str]) {
    for field in o.keys().filter(|k| !known.contains(&k.as_str())) {
        warnings.push(HistoryWarning::UnknownField { scope: scope.into(), field: field.clone() });
    }
}

/// A record before it is placed against `as_of`.
struct Raw {
    label: String,
    record: Record,
}

fn rank(v: &Option<Value>) -> u8 {
    match v {
        None => 0,
        Some(Value::Number(_)) => 1,
        Some(Value::Below { .. }) => 2,
        Some(Value::Above { .. }) => 3,
        Some(Value::Category(_)) => 4,
    }
}

fn compare_values(a: &Option<Value>, b: &Option<Value>) -> Ordering {
    use Value::*;
    rank(a).cmp(&rank(b)).then_with(|| match (a, b) {
        (Some(Number(x)), Some(Number(y)))
        | (Some(Below { below: x }), Some(Below { below: y }))
        | (Some(Above { above: x }), Some(Above { above: y })) => x.total_cmp(y),
        (Some(Category(x)), Some(Category(y))) => x.cmp(y),
        _ => Ordering::Equal,
    })
}

/// The canonical order of records: time, code, value, unit.
fn compare(a: &Record, b: &Record) -> Ordering {
    a.time
        .total_cmp(&b.time)
        .then_with(|| a.code.cmp(&b.code))
        .then_with(|| compare_values(&a.value, &b.value))
        .then_with(|| a.unit.cmp(&b.unit))
}

impl PatientHistory {
    /// Parse and validate one history from its JSON value.
    pub fn from_json(v: &Json) -> Result<PatientHistory, String> {
        let o = v.as_object().ok_or("history: a history is a JSON object")?;
        let id = match o.get("id") {
            None => "history".to_string(),
            Some(i) => {
                let ctx = Context { id: "history" };
                parse_name(&ctx, "history", "id", i)?
            }
        };
        let ctx = Context { id: &id };
        let mut warnings = Vec::new();
        note_unknown(&mut warnings, "history", o, &["id", "as_of", "birth", "calendar", "static", "events"]);

        let birth = match o.get("birth") {
            None | Some(Json::Null) => None,
            Some(Json::String(s)) => Some(Instant::parse(s).map_err(|e| ctx.err("history", "birth", e))?),
            Some(_) => return Err(ctx.err("history", "birth", "must be an ISO-8601 date string")),
        };
        let to_clock = |record: &str, field: &str, t: Time| -> Result<(f64, Option<Instant>), String> {
            match t {
                Time::Clock(c) => Ok((c, birth.map(|b| b.plus_years(c)))),
                Time::At(at) => {
                    let b = birth.ok_or_else(|| {
                        ctx.err(record, field, "a date needs \"birth\" to become the model's clock (or give the time as a number)")
                    })?;
                    let years = at.years_since(b);
                    if years < 0.0 {
                        return Err(ctx.err(record, field, "is before birth"));
                    }
                    Ok((years, Some(at)))
                }
            }
        };
        let as_of_json = o.get("as_of").ok_or_else(|| ctx.err("history", "as_of", "is required"))?;
        let as_of_time = parse_time(&ctx, "history", "as_of", as_of_json)?;
        let (as_of, as_of_instant) = to_clock("history", "as_of", as_of_time)?;
        let calendar_at_as_of = match (as_of_instant, o.get("calendar")) {
            (Some(at), None | Some(Json::Null)) => at.decimal_year(),
            (Some(_), Some(_)) => {
                return Err(ctx.err(
                    "history",
                    "calendar",
                    "is derived from as_of and birth: remove it (two sources of truth)",
                ))
            }
            (None, Some(c)) => c
                .as_f64()
                .filter(|c| c.is_finite())
                .ok_or_else(|| ctx.err("history", "calendar", "must be a finite number (the decimal year at as_of)"))?,
            (None, None) => {
                return Err(ctx.err(
                    "history",
                    "calendar",
                    "is required when as_of is a number and there is no birth: the model reads the calendar time",
                ))
            }
        };

        let mut raws: Vec<Raw> = Vec::new();
        let empty = Vec::new();
        let events = match o.get("events") {
            None | Some(Json::Null) => &empty,
            Some(Json::Array(a)) => a,
            Some(_) => return Err(ctx.err("history", "events", "must be an array")),
        };
        let statics = match o.get("static") {
            None | Some(Json::Null) => None,
            Some(Json::Object(m)) => Some(m),
            Some(_) => return Err(ctx.err("history", "static", "must be an object")),
        };
        if events.len() + statics.map_or(0, Map::len) > MAX_RECORDS
            || statics.is_some_and(|m| m.len() > MAX_STATIC)
        {
            return Err(format!(
                "history {id:?}: too many records (at most {MAX_RECORDS} events and measurements, {MAX_STATIC} static values)"
            ));
        }
        for (i, e) in events.iter().enumerate() {
            let label = format!("events[{i}]");
            let eo = e.as_object().ok_or_else(|| ctx.err(&label, "-", "an event is a JSON object"))?;
            note_unknown(&mut warnings, &label, eo, &["time", "code", "value", "unit"]);
            let time = parse_time(&ctx, &label, "time", eo.get("time").ok_or_else(|| ctx.err(&label, "time", "is required"))?)?;
            let code = parse_name(&ctx, &label, "code", eo.get("code").ok_or_else(|| ctx.err(&label, "code", "is required"))?)?;
            let value = match eo.get("value") {
                None | Some(Json::Null) => None,
                Some(v) => Some(parse_value(&ctx, &label, v)?),
            };
            let unit = match eo.get("unit") {
                None | Some(Json::Null) => None,
                Some(u) => Some(parse_unit(&ctx, &label, u, value.as_ref())?),
            };
            let (clock, _) = to_clock(&label, "time", time)?;
            raws.push(Raw { label, record: Record { time: clock, code, value, unit } });
        }
        for (name, v) in statics.into_iter().flatten() {
            let label = format!("static.{name}");
            let code = parse_name(&ctx, &label, "name", &Json::String(name.clone()))?;
            let (raw_value, raw_unit) = match v {
                Json::Object(w) if w.contains_key("value") => {
                    note_unknown(&mut warnings, &label, w, &["value", "unit"]);
                    (&w["value"], w.get("unit").filter(|u| !u.is_null()))
                }
                other => (other, None),
            };
            let value = parse_value(&ctx, &label, raw_value)?;
            let unit = raw_unit.map(|u| parse_unit(&ctx, &label, u, Some(&value))).transpose()?;
            raws.push(Raw { label, record: Record { time: as_of, code, value: Some(value), unit } });
        }

        // Place against as_of: the future and an event at as_of are dropped.
        let mut kept: Vec<(usize, Raw)> = Vec::new();
        for (n, raw) in raws.into_iter().enumerate() {
            let Raw { label, record, .. } = &raw;
            if record.time > as_of {
                warnings.push(HistoryWarning::FutureIgnored {
                    record: label.clone(),
                    code: record.code.clone(),
                    time: record.time,
                });
            } else if record.time == as_of && record.value.is_none() {
                warnings.push(HistoryWarning::EventAtAsOf { record: label.clone(), code: record.code.clone() });
            } else {
                kept.push((n, raw));
            }
        }
        // Canonical order; among equals the earliest listed is the one kept.
        kept.sort_by(|(na, a), (nb, b)| compare(&a.record, &b.record).then(na.cmp(nb)));
        let mut records: Vec<Record> = Vec::with_capacity(kept.len());
        let mut last_label = String::new();
        let mut dropped: Vec<(usize, HistoryWarning)> = Vec::new();
        for (n, raw) in kept {
            match records.last() {
                Some(prev) if compare(prev, &raw.record) == Ordering::Equal => dropped.push((
                    n,
                    HistoryWarning::DuplicateIgnored {
                        record: raw.label,
                        code: raw.record.code,
                        time: raw.record.time,
                    },
                )),
                Some(prev)
                    if prev.time == raw.record.time
                        && prev.code == raw.record.code
                        && prev.value.is_some()
                        && raw.record.value.is_some() =>
                {
                    return Err(ctx.err(
                        &raw.label,
                        "value",
                        format!(
                            "{} at {} contradicts {last_label}: the same code and time with a different value or unit",
                            raw.record.code, raw.record.time
                        ),
                    ))
                }
                _ => {
                    last_label = raw.label;
                    records.push(raw.record)
                }
            }
        }
        dropped.sort_by_key(|(n, _)| *n);
        warnings.extend(dropped.into_iter().map(|(_, w)| w));
        Ok(PatientHistory {
            id,
            as_of,
            as_of_input: as_of_json.to_string().trim_matches('"').to_string(),
            calendar_at_as_of,
            records,
            warnings,
        })
    }

    /// Parse the JSON text of one history, or several: the text is a JSON
    /// object, an array of objects, or a sequence of objects (one per line).
    pub fn parse_all(text: &str) -> Result<Vec<PatientHistory>, String> {
        if text.len() > MAX_TEXT_LEN {
            return Err(format!("history: input is larger than {MAX_TEXT_LEN} bytes"));
        }
        let mut out = Vec::new();
        let mut take = |v: &Json| -> Result<(), String> {
            if out.len() >= MAX_HISTORIES {
                return Err(format!("history: more than {MAX_HISTORIES} histories in one input"));
            }
            let n = out.len() + 1;
            out.push(PatientHistory::from_json(v).map_err(|e| format!("history #{n}: {e}"))?);
            Ok(())
        };
        for v in serde_json::Deserializer::from_str(text).into_iter::<Json>() {
            match v.map_err(|e| format!("history: {e}"))? {
                Json::Array(items) => items.iter().try_for_each(&mut take)?,
                one => take(&one)?,
            }
        }
        if out.is_empty() {
            return Err("history: no history in the input".into());
        }
        Ok(out)
    }

    /// The subject this history is for `vocab`'s model: entry and calendar
    /// time from `as_of`, the measurements at or before it as observations,
    /// the events before it as history, no outcome windows. A measurement in
    /// a unit other than the one the model was trained on is an error
    /// ([`Vocab::check_units`]).
    pub fn to_subject(&self, vocab: &Vocab) -> Result<Subject, String> {
        let mut s = Subject {
            subject_id: self.id.clone(),
            group_id: None,
            weight: 1.0,
            source: "history".into(),
            entry: self.as_of,
            calendar_at_entry: self.calendar_at_as_of,
            observations: Vec::new(),
            events: Vec::new(),
            at_risk: Vec::new(),
        };
        for r in &self.records {
            match &r.value {
                Some(value) => s.observations.push(Observation {
                    t: r.time,
                    var: r.code.clone(),
                    value: value.clone(),
                    unit: r.unit.clone(),
                }),
                None => s.events.push(Event { t: r.time, code: r.code.clone() }),
            }
        }
        s.validate()?;
        vocab.check_units(&s)?;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<PatientHistory, String> {
        PatientHistory::from_json(&serde_json::from_str(text).unwrap())
    }

    const BASE: &str = r#"{"as_of": 60.0, "calendar": 2020.5, "events": [
        {"time": 50, "code": "ldl", "value": 3.1, "unit": "mmol/L"},
        {"time": 40, "code": "dx:x"}]}"#;

    #[test]
    fn dates_become_years_on_a_fixed_year_and_the_calendar_is_the_decimal_year() {
        let h = parse(r#"{"as_of": "2026-03-01", "birth": "1970-03-01", "events": [
            {"time": "2000-03-01T00:00:00Z", "code": "a", "value": 1},
            {"time": "2000-03-01T01:00:00+01:00", "code": "b", "value": 1}]}"#)
        .unwrap();
        // 56 calendar years = 20454 days (14 leap days) -> 20454 / 365.25 years.
        assert_eq!(h.as_of, 20_454.0 * 86_400.0 / YEAR_SECONDS as f64);
        assert_eq!(h.records[0].time, h.records[1].time, "an offset is normalised to UTC");
        assert_eq!(h.records[0].time, 10_958.0 * 86_400.0 / YEAR_SECONDS as f64);
        // 2026-03-01 is day 59 of 365: 2026 + 59/365.
        assert_eq!(h.calendar_at_as_of, 2026.0 + 59.0 / 365.0);
        let leap = parse(r#"{"as_of": "2024-03-01T12:00:00Z", "birth": "2000-01-01"}"#).unwrap();
        assert_eq!(leap.calendar_at_as_of, 2024.0 + (60.5 / 366.0));
        let numeric = parse(r#"{"as_of": 30, "birth": "2000-01-01"}"#).unwrap();
        assert!((numeric.calendar_at_as_of - 2030.0).abs() < 0.01, "{}", numeric.calendar_at_as_of);
    }

    #[test]
    fn malformed_times_say_which_record_and_field() {
        for (bad, field) in [
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": "2020-13-01", "code": "a"}]}"#, "events[0]"),
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": "2020-02-30", "code": "a"}]}"#, "no such calendar date"),
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": "2020-02-03T10:00:00", "code": "a"}]}"#, "UTC"),
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": "2020-02-03T10:00:60Z", "code": "a"}]}"#, "leap seconds"),
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": "2020-02-03", "code": "a"}]}"#, "needs \"birth\""),
            (r#"{"as_of": "2020-01-01", "birth": "2021-01-01"}"#, "before birth"),
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": true, "code": "a"}]}"#, "\"time\""),
            (r#"{"as_of": 1, "calendar": 1, "events": [{"time": 1}]}"#, "\"code\""),
            (r#"{"calendar": 1}"#, "as_of"),
            (r#"{"as_of": 1}"#, "calendar"),
            (r#"{"as_of": "2020-01-01", "birth": "2000-01-01", "calendar": 2020}"#, "two sources of truth"),
        ] {
            let err = parse(bad).unwrap_err();
            assert!(err.contains(field), "{bad}: {err}");
        }
    }

    #[test]
    fn malformed_values_and_units_name_the_record() {
        let ev = |e: &str| format!(r#"{{"as_of": 60, "calendar": 2020, "events": [{e}]}}"#);
        for (bad, needle) in [
            (r#"{"time": 1, "code": "a", "value": 1, "unit": ""}"#, "unit"),
            (r#"{"time": 1, "code": "a", "value": 1, "unit": " kg"}"#, "unit"),
            (r#"{"time": 1, "code": "a", "value": 1, "unit": 5}"#, "unit"),
            (r#"{"time": 1, "code": "a", "unit": "kg"}"#, "an event has none"),
            (r#"{"time": 1, "code": "a", "value": "low", "unit": "kg"}"#, "no unit"),
            (r#"{"time": 1, "code": "a", "value": [1]}"#, "\"value\""),
            (r#"{"time": 1, "code": "a", "value": {"below": "x"}}"#, "\"value\""),
            (r#"{"time": 1, "code": "a", "value": {"under": 1}}"#, "\"value\""),
            (r#"{"time": 1, "code": "a", "value": ""}"#, "\"value\""),
        ] {
            let err = parse(&ev(bad)).unwrap_err();
            assert!(err.contains("events[0]") && err.contains(needle), "{bad}: {err}");
        }
        let err = parse(r#"{"as_of": 60, "calendar": 2020, "static": {"sex": {"value": 1, "unit": 3}}}"#).unwrap_err();
        assert!(err.contains("static.sex") && err.contains("unit"), "{err}");
    }

    #[test]
    fn the_future_and_an_event_at_as_of_are_dropped_with_a_warning() {
        let h = parse(
            r#"{"as_of": 60, "calendar": 2020, "events": [
            {"time": 61, "code": "ldl", "value": 9},
            {"time": 60, "code": "ldl", "value": 4},
            {"time": 60, "code": "dx:now"},
            {"time": 59.9, "code": "dx:before"}]}"#,
        )
        .unwrap();
        let codes: Vec<_> = h.records.iter().map(|r| (r.time, r.code.as_str())).collect();
        assert_eq!(codes, vec![(59.9, "dx:before"), (60.0, "ldl")]);
        assert!(h.warnings.contains(&HistoryWarning::FutureIgnored { record: "events[0]".into(), code: "ldl".into(), time: 61.0 }));
        assert!(h.warnings.contains(&HistoryWarning::EventAtAsOf { record: "events[2]".into(), code: "dx:now".into() }));
    }

    #[test]
    fn exact_duplicates_collapse_and_contradictions_are_errors() {
        let h = parse(
            r#"{"as_of": 60, "calendar": 2020, "events": [
            {"time": 50, "code": "ldl", "value": 3.1, "unit": "mmol/L"},
            {"time": 50, "code": "ldl", "value": 3.1, "unit": "mmol/L"},
            {"time": 40, "code": "dx:x"}, {"time": 40, "code": "dx:x"}]}"#,
        )
        .unwrap();
        assert_eq!(h.records.len(), 2);
        assert_eq!(h.warnings.len(), 2);
        for other in [r#""value": 3.2, "unit": "mmol/L""#, r#""value": 3.1, "unit": "mg/dL""#, r#""value": 3.1"#] {
            let err = parse(&format!(
                r#"{{"as_of": 60, "calendar": 2020, "events": [
                {{"time": 50, "code": "ldl", "value": 3.1, "unit": "mmol/L"}},
                {{"time": 50, "code": "ldl", {other}}}]}}"#
            ))
            .unwrap_err();
            assert!(
                err.contains("events[1]") && err.contains("events[0]") && err.contains("contradicts"),
                "{other}: {err}"
            );
        }
    }

    #[test]
    fn listing_order_never_matters_and_unknown_fields_are_only_warnings() {
        let a = parse(BASE).unwrap();
        let b = parse(
            r#"{"as_of": 60.0, "calendar": 2020.5, "events": [
            {"time": 40, "code": "dx:x"},
            {"time": 50, "code": "ldl", "value": 3.1, "unit": "mmol/L"}]}"#,
        )
        .unwrap();
        assert_eq!(a.records, b.records);
        let c = parse(
            r#"{"as_of": 60.0, "calendar": 2020.5, "note": "x", "events": [
            {"time": 40, "code": "dx:x", "source": "lab"}]}"#,
        )
        .unwrap();
        assert_eq!(
            c.warnings,
            vec![
                HistoryWarning::UnknownField { scope: "history".into(), field: "note".into() },
                HistoryWarning::UnknownField { scope: "events[0]".into(), field: "source".into() },
            ]
        );
        assert_eq!(c.records.len(), 1);
    }

    #[test]
    fn static_values_are_measurements_at_as_of_and_sizes_are_bounded() {
        let h = parse(r#"{"as_of": 60, "calendar": 2020, "static": {"sex": "female", "height": {"value": 1.7, "unit": "m"}}}"#)
            .unwrap();
        assert!(h.records.iter().all(|r| r.time == 60.0 && r.value.is_some()));
        assert_eq!(h.records.iter().find(|r| r.code == "height").unwrap().unit.as_deref(), Some("m"));
        let many: Vec<String> = (0..=MAX_RECORDS).map(|i| format!(r#"{{"time": {i}, "code": "a"}}"#)).collect();
        let err = parse(&format!(r#"{{"as_of": 1e9, "calendar": 1, "events": [{}]}}"#, many.join(","))).unwrap_err();
        assert!(err.contains("too many records"), "{err}");
        let long = "x".repeat(MAX_NAME_LEN + 1);
        let err = parse(&format!(r#"{{"as_of": 1, "calendar": 1, "events": [{{"time": 0, "code": "{long}"}}]}}"#)).unwrap_err();
        assert!(err.contains("events[0]") && err.contains("code"), "{err}");
    }

    #[test]
    fn one_object_an_array_or_lines_all_parse() {
        let one = r#"{"as_of": 1, "calendar": 2000}"#;
        assert_eq!(PatientHistory::parse_all(one).unwrap().len(), 1);
        assert_eq!(PatientHistory::parse_all(&format!("[{one},{one}]")).unwrap().len(), 2);
        assert_eq!(PatientHistory::parse_all(&format!("{one}\n{one}\n")).unwrap().len(), 2);
        assert!(PatientHistory::parse_all("  ").is_err());
        let err = PatientHistory::parse_all(&format!("{one}\n{{\"as_of\": 1}}")).unwrap_err();
        assert!(err.contains("history #2"), "{err}");
    }
}
