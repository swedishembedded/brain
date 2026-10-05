// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One subject, encoded once into what a batch is assembled from.
//!
//! **Only what is known at prediction time reaches the encoder**: observations
//! at or before `entry`, events strictly before it. Everything after entry is
//! either an outcome (an event inside its observation window) or ignored.
//! That rule is the whole defence against leakage and is tested here, not
//! left to the caller who builds the timelines.
//!
//! Exposure is exact: for every outcome code, the time the subject spent at
//! risk inside each hazard piece, from the start of its observation window
//! (left truncation) to its end, its own event, or the first absorbing event,
//! whichever comes first.

use model::hostmath::ndtri;

use crate::config::{HorizonConfig, BIN_ABOVE, BIN_BELOW, BIN_PRESENT, TIME_AGO_SPAN};
use crate::timeline::{Subject, Value};
use crate::vocab::Vocab;

/// Relative tolerance at an observation window's end (see [`encode`]).
const WINDOW_SLACK: f64 = 1e-9;

/// The measured state of a value target, as the loss kernel reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueState {
    /// Exactly observed.
    Exact = 1,
    /// At or below the recorded limit.
    Below = 2,
    /// At or above the recorded limit.
    Above = 3,
}

/// One encoded input token.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// Vocabulary id.
    pub id: u32,
    /// Value-bin weights `(bin, weight)`: up to two soft bins and a special bin.
    pub value_bins: Vec<(u32, f32)>,
    /// Time-ago bin weights `(bin, weight)`.
    pub time_bins: Vec<(u32, f32)>,
    /// The normal-score target and its state, for a numeric token.
    pub target: Option<(f32, ValueState)>,
}

/// One outcome code's observation of one subject, relative to entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Time at risk inside each hazard piece.
    pub exposure: Vec<f32>,
    /// The piece the event fell in, if it was observed.
    pub event_piece: Option<u32>,
}

/// A subject ready to batch.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// Input tokens (without the summary token), at most `max_tokens - 1`.
    pub tokens: Vec<Token>,
    /// Known tokens left out because the subject had more than fit.
    pub truncated: usize,
    /// Sampling weight.
    pub weight: f32,
    /// The subject's clock at entry.
    pub entry: f64,
    /// Calendar time at entry.
    pub calendar_at_entry: f64,
    /// Per outcome code.
    pub outcomes: Vec<Outcome>,
    /// Future measurements the forecast head is scored on (at most
    /// `HorizonConfig::forecasts`, the earliest first).
    pub forecasts: Vec<Forecast>,
}

/// A measurement at a time after entry: what the forecast head predicts.
/// Its value never reaches the encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    /// The variable's token id.
    pub token: u32,
    /// How long after entry, in the dataset's unit.
    pub ahead: f64,
    /// Soft bins over how long after entry.
    pub ahead_bins: Vec<(u32, f32)>,
    /// The normal-score target and its state; `None` for a query to predict.
    pub target: Option<(f32, ValueState)>,
}

/// Soft bins over a time difference on the log scale the time-ago axis uses.
fn span_bins(dt: f64, n: u32) -> Vec<(u32, f32)> {
    soft_bins((1.0 + dt.max(0.0)).ln() / (1.0 + TIME_AGO_SPAN).ln(), n)
}

/// A forecast query: the value of numeric `var` `ahead` after entry. `None`
/// for a variable the vocabulary has no knots for.
pub fn forecast_query(
    vocab: &Vocab,
    cfg: &HorizonConfig,
    var: &str,
    ahead: f64,
) -> Option<Forecast> {
    vocab.knots.get(var)?;
    Some(Forecast {
        token: vocab.numeric_token(var),
        ahead,
        ahead_bins: span_bins(ahead, cfg.time_bins),
        target: None,
    })
}

/// Soft weights over `n` evenly spaced bin centres for `x` in `[0, 1]`:
/// linear interpolation between the two nearest centres, clamped at the ends.
pub fn soft_bins(x: f64, n: u32) -> Vec<(u32, f32)> {
    let pos = (x.clamp(0.0, 1.0) * n as f64 - 0.5).clamp(0.0, (n - 1) as f64);
    let lo = pos.floor() as u32;
    let frac = (pos - lo as f64) as f32;
    if lo + 1 >= n || frac == 0.0 {
        vec![(lo, 1.0)]
    } else {
        vec![(lo, 1.0 - frac), (lo + 1, frac)]
    }
}

/// Map an empirical CDF to a normal score, kept off the infinite ends by half
/// a knot's width.
pub(crate) fn normal_score(u: f64, knots: usize) -> f32 {
    let eps = 0.5 / knots as f64;
    ndtri(u.clamp(eps, 1.0 - eps)) as f32
}

/// Encode one subject.
pub fn encode(s: &Subject, vocab: &Vocab, cfg: &HorizonConfig) -> Encoded {
    let nb = cfg.value_bins;
    let time_bins = |t: f64| -> Vec<(u32, f32)> {
        let ago = s.entry - t;
        if ago.abs() < 1e-9 {
            vec![(cfg.time_bins, 1.0)] // measured at entry: its own bin
        } else {
            soft_bins(
                (1.0 + ago.max(0.0)).ln() / (1.0 + TIME_AGO_SPAN).ln(),
                cfg.time_bins,
            )
        }
    };
    let mut tokens: Vec<(f64, Token)> = Vec::new();
    for o in s.known_observations() {
        let ago = s.entry - o.t;
        let tok = match &o.value {
            Value::Category(level) => Token {
                id: vocab.category_token(&o.var, level),
                value_bins: vec![(nb + BIN_PRESENT, 1.0)],
                time_bins: time_bins(o.t),
                target: None,
            },
            v => {
                let (x, special, state) = match v {
                    Value::Number(x) => (*x, None, ValueState::Exact),
                    Value::Below { below } => (*below, Some(BIN_BELOW), ValueState::Below),
                    Value::Above { above } => (*above, Some(BIN_ABOVE), ValueState::Above),
                    Value::Category(_) => unreachable!(),
                };
                match vocab.cdf(&o.var, x) {
                    Some(u) => {
                        let mut bins = soft_bins(u, nb);
                        bins.extend(special.map(|b| (nb + b, 1.0)));
                        let k = vocab.knots[&o.var].len();
                        Token {
                            id: vocab.numeric_token(&o.var),
                            value_bins: bins,
                            time_bins: time_bins(o.t),
                            target: Some((normal_score(u, k), state)),
                        }
                    }
                    // A numeric variable the vocabulary never saw: present, value unknown.
                    None => Token {
                        id: vocab.numeric_token(&o.var),
                        value_bins: vec![(nb + BIN_PRESENT, 1.0)],
                        time_bins: time_bins(o.t),
                        target: None,
                    },
                }
            }
        };
        tokens.push((ago, tok));
    }
    for e in s.known_events() {
        let tok = Token {
            id: vocab.event_token(&e.code),
            value_bins: vec![(nb + BIN_PRESENT, 1.0)],
            time_bins: time_bins(e.t),
            target: None,
        };
        tokens.push((s.entry - e.t, tok));
    }
    // Deterministic truncation: the entry visit first, then the most recent
    // history; ties by token id.
    tokens.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("finite")
            .then(a.1.id.cmp(&b.1.id))
    });
    let cap = cfg.max_tokens as usize - 1;
    let truncated = tokens.len().saturating_sub(cap);
    tokens.truncate(cap);

    Encoded {
        tokens: tokens.into_iter().map(|(_, t)| t).collect(),
        truncated,
        weight: s.weight as f32,
        entry: s.entry,
        calendar_at_entry: s.calendar_at_entry,
        outcomes: outcomes(s, vocab, &cfg.knots),
        forecasts: forecasts(s, vocab, cfg),
    }
}

/// The subject's future numeric measurements as forecast targets, the
/// earliest first, at most `cfg.forecasts`.
fn forecasts(s: &Subject, vocab: &Vocab, cfg: &HorizonConfig) -> Vec<Forecast> {
    if cfg.forecasts == 0 {
        return Vec::new();
    }
    let mut out: Vec<Forecast> = s
        .observations
        .iter()
        .filter(|o| o.t > s.entry)
        .filter_map(|o| {
            let (x, state) = match &o.value {
                Value::Number(x) => (*x, ValueState::Exact),
                Value::Below { below } => (*below, ValueState::Below),
                Value::Above { above } => (*above, ValueState::Above),
                Value::Category(_) => return None,
            };
            let u = vocab.cdf(&o.var, x)?;
            let mut f = forecast_query(vocab, cfg, &o.var, o.t - s.entry)?;
            f.target = Some((normal_score(u, vocab.knots[&o.var].len()), state));
            Some(f)
        })
        .collect();
    out.sort_by(|a, b| a.ahead.total_cmp(&b.ahead).then(a.token.cmp(&b.token)));
    out.truncate(cfg.forecasts as usize);
    out
}

/// Exposure and event piece per outcome code.
fn outcomes(s: &Subject, vocab: &Vocab, knots: &[f32]) -> Vec<Outcome> {
    let after = |code: &str| {
        s.events
            .iter()
            .filter(|e| e.code == code && e.t > s.entry)
            .map(|e| e.t)
            .fold(f64::INFINITY, f64::min)
    };
    let absorbed = vocab
        .absorbing
        .iter()
        .map(|c| after(c))
        .fold(f64::INFINITY, f64::min);
    vocab
        .codes
        .iter()
        .map(|code| {
            let pieces = knots.len() - 1;
            let Some(w) = s.window(code) else {
                return Outcome {
                    exposure: vec![0.0; pieces],
                    event_piece: None,
                };
            };
            let event = after(code);
            // An event within a rounding distance past its window's end is
            // inside it: window ends are typically computed from the event
            // time, and a one-ulp difference must not turn an event into a
            // censoring (it silently biases every hazard down).
            let to = w.to + WINDOW_SLACK * w.to.abs().max(1.0);
            let end = to.min(absorbed).min(event);
            let (a, b) = ((w.from - s.entry) as f32, (end - s.entry) as f32);
            let exposure: Vec<f32> = knots
                .windows(2)
                .map(|k| (b.min(k[1]) - a.max(k[0])).max(0.0))
                .collect();
            // The event counts only if observed inside the window and inside the knots.
            let rel = (event - s.entry) as f32;
            let event_piece = (event <= to && event.is_finite() && rel > a)
                .then(|| knots.windows(2).position(|k| rel > k[0] && rel <= k[1]))
                .flatten()
                .map(|p| p as u32);
            Outcome {
                exposure,
                event_piece,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::FitOptions;

    fn fixture() -> (Subject, Vocab, HorizonConfig) {
        let line = r#"{"subject_id":"a","source":"s","entry":50,"calendar_at_entry":2000,
            "observations":[{"t":50,"var":"sbp","value":120},{"t":50,"var":"crp","value":{"below":0.2}},
                            {"t":25,"var":"weight","value":70},{"t":52,"var":"sbp","value":180},
                            {"t":50,"var":"smoking","value":"never"}],
            "events":[{"t":44,"code":"dx"},{"t":53.5,"code":"death:heart"},{"t":51,"code":"dx"}],
            "at_risk":[{"code":"*","from":50,"to":60},{"code":"dx","from":50.5,"to":60}]}"#;
        let s = Subject::from_json_line(line).unwrap();
        let codes: Vec<String> = ["death:heart", "death:other", "dx"]
            .iter()
            .map(|c| c.to_string())
            .collect();
        let opts = FitOptions {
            knots: 5,
            min_count: 1,
        };
        let v = Vocab::fit(std::slice::from_ref(&s), &codes, &codes[..2], &opts).unwrap();
        let cfg = HorizonConfig::tiny(v.len(), 3);
        (s, v, cfg)
    }

    #[test]
    fn soft_bins_interpolate_and_clamp() {
        assert_eq!(soft_bins(0.0, 4), vec![(0, 1.0)]);
        assert_eq!(soft_bins(1.0, 4), vec![(3, 1.0)]);
        let mid = soft_bins(0.5, 4);
        assert_eq!(mid.len(), 2);
        assert_eq!((mid[0].0, mid[1].0), (1, 2));
        assert!((mid[0].1 - 0.5).abs() < 1e-6 && (mid[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn only_the_known_past_is_encoded() {
        let (s, v, cfg) = fixture();
        let e = encode(&s, &v, &cfg);
        // sbp@50, crp@50, smoking@50, weight@25, dx@44: the sbp at 52 and both
        // post-entry events are the future.
        assert_eq!(e.tokens.len(), 5);
        assert!(e
            .tokens
            .iter()
            .all(|t| t.id != v.event_token("death:heart")));
        let sbp: Vec<_> = e
            .tokens
            .iter()
            .filter(|t| t.id == v.numeric_token("sbp"))
            .collect();
        assert_eq!(sbp.len(), 1, "one sbp is known, the later one is not");
        // Entry-visit tokens come first and carry the "at entry" time bin.
        assert_eq!(e.tokens[0].time_bins, vec![(cfg.time_bins, 1.0)]);
        let crp = e
            .tokens
            .iter()
            .find(|t| t.id == v.numeric_token("crp"))
            .unwrap();
        assert_eq!(crp.target.unwrap().1, ValueState::Below);
        assert!(crp.value_bins.contains(&(cfg.value_bins + BIN_BELOW, 1.0)));
    }

    #[test]
    fn exposure_respects_windows_absorption_and_event_pieces() {
        let (s, v, cfg) = fixture(); // knots 0, 1, 2.5, 4
        let e = encode(&s, &v, &cfg);
        // death:heart at +3.5 ends follow-up for everything.
        let heart = &e.outcomes[0];
        assert_eq!(heart.exposure, vec![1.0, 1.5, 1.0]);
        assert_eq!(heart.event_piece, Some(2));
        let other = &e.outcomes[1];
        assert_eq!(other.exposure, vec![1.0, 1.5, 1.0]);
        assert_eq!(other.event_piece, None, "censored by the competing death");
        // dx: window opens at +0.5 (left truncation) and its own event at +1 ends it.
        let dx = &e.outcomes[2];
        assert_eq!(dx.exposure, vec![0.5, 0.0, 0.0]);
        assert_eq!(dx.event_piece, Some(0));
    }

    #[test]
    fn an_event_one_ulp_past_its_window_end_is_not_censored() {
        let (mut s, v, cfg) = fixture();
        let death = s
            .events
            .iter()
            .position(|e| e.code == "death:heart")
            .unwrap();
        s.at_risk[0].to = s.events[death].t;
        s.events[death].t = f64::from_bits(s.at_risk[0].to.to_bits() + 1);
        assert_eq!(encode(&s, &v, &cfg).outcomes[0].event_piece, Some(2));
        s.events[death].t += 1e-3;
        assert_eq!(
            encode(&s, &v, &cfg).outcomes[0].event_piece,
            None,
            "a real gap is still censoring"
        );
    }

    #[test]
    fn truncation_keeps_the_entry_visit_and_counts_what_it_drops() {
        let (s, v, mut cfg) = fixture();
        cfg.max_tokens = 4;
        let e = encode(&s, &v, &cfg);
        assert_eq!(e.tokens.len(), 3);
        assert_eq!(e.truncated, 2);
        assert!(
            e.tokens
                .iter()
                .all(|t| t.time_bins == vec![(cfg.time_bins, 1.0)]),
            "the entry visit is kept first"
        );
    }
}
