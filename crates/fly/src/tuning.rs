// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The parameters a connectome does not contain, written down.
//!
//! A search finds them and then the process exits. Without a way to carry the
//! answer out, every result in this crate is a number in a log that nobody can
//! watch, re-measure, or disagree with - and "the fly walks" is a claim about
//! a run that no longer exists.
//!
//! The format is one `name value` per line, because that is the format a
//! person can read, diff, edit by hand and paste into a bug report. It is NOT
//! a checkpoint: it holds tens of numbers, not millions, and the thing it
//! names is a cell CLASS rather than a synapse. Anything the file does not
//! name keeps the value the runtime would have used anyway, so an old tuning
//! against a newer runtime is a partial specification rather than a broken
//! one - and `Tuning::apply` reports what it recognised so that "partial" is
//! never silent.

use std::collections::BTreeMap;
use std::path::Path;

use crate::learn::GainSearch;
use crate::{Fly, Wiring};

/// Prefix marking a per-cell-class synaptic gain.
const GAIN: &str = "gain:";

/// A set of tuned parameters, by name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tuning {
    values: BTreeMap<String, f32>,
}

/// What [`Tuning::apply`] actually did.
///
/// Returned rather than logged, because the failure this guards against is
/// silent: a tuning naming gains for cell classes that this connectome does
/// not have - the bare cord's classes against the joined brain's, say -
/// applies cleanly, changes almost nothing, and leaves a caller believing the
/// animal is tuned.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Applied {
    /// Gains matched to a cell class in this connectome.
    pub gains: usize,
    /// Names in the file that nothing in this runtime recognised.
    pub unknown: Vec<String>,
}

impl Tuning {
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        self.values.get(name).copied()
    }

    pub fn set(&mut self, name: impl Into<String>, value: f32) {
        self.values.insert(name.into(), value);
    }

    /// The descending command this tuning was found at.
    ///
    /// Part of the tuning rather than of the experiment: a gain set that only
    /// works at one drive and is replayed at another is not the thing that was
    /// measured.
    pub fn command(&self) -> Option<f32> {
        self.get("command")
    }

    pub fn to_text(&self) -> String {
        let mut out = String::from("# brain fly tuning: one `name value` per line.\n");
        for (k, v) in &self.values {
            out.push_str(&format!("{k} {v}\n"));
        }
        out
    }

    pub fn parse(text: &str) -> Result<Tuning, String> {
        let mut t = Tuning::default();
        for (i, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some(name), Some(value), None) => {
                    let v: f32 = value
                        .parse()
                        .map_err(|_| format!("line {}: {value:?} is not a number", i + 1))?;
                    t.values.insert(name.to_string(), v);
                }
                _ => return Err(format!("line {}: expected `name value`, got {line:?}", i + 1)),
            }
        }
        Ok(t)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Tuning, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Tuning::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        std::fs::write(path, self.to_text()).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Apply to a creature: weights, dynamics, coupling.
    ///
    /// `wiring` has to be the one the creature was built with, because the
    /// gains multiply the network that creature is running and a different
    /// synapse floor is a different weight vector of a different length.
    /// `Fly::set_weights` rejects a mismatch rather than accepting it.
    pub fn apply(&self, fly: &mut Fly, c: &connectome::Connectome, wiring: Wiring) -> Result<Applied, String> {
        let mut report = Applied::default();
        let search = GainSearch::new(c, wiring);
        let mut gains = search.unit_gains();
        for (i, group) in search.groups().iter().enumerate() {
            if let Some(v) = self.get(&format!("{GAIN}{group}")) {
                gains[i] = v;
                report.gains += 1;
            }
        }
        fly.set_weights(&search.weights(&gains))?;

        let mut lif = fly.lif();
        let mut coupling = fly.coupling();
        for (name, v) in &self.values {
            let v = *v;
            match name.as_str() {
                "adapt_increment" => lif.adapt_increment = v,
                "adapt_decay" => lif.adapt_decay = v,
                "dt_over_tau_syn" => lif.dt_over_tau_syn = v,
                "dt_over_tau_inh" => lif.dt_over_tau_inh = v,
                "dt_over_tau" => lif.dt_over_tau = v,
                "v_th" => lif.v_th = v,
                "r" => lif.r = v,
                "activation_gain" => coupling.activation_gain = v,
                "activation_decay" => coupling.activation_decay = v,
                "angle_gain" => coupling.angle_gain = v,
                "load_gain" => coupling.load_gain = v,
                "odour_gain" => coupling.odour_gain = v,
                "wing_power_gain" => coupling.wing_power_gain = v,
                "wing_steer_gain" => coupling.wing_steer_gain = v,
                // The command is applied by the caller, which is the only
                // thing that knows whether this is an episode or a window
                // somebody is watching.
                "command" => {}
                other if other.starts_with(GAIN) => {
                    if !search.groups().iter().any(|g| format!("{GAIN}{g}") == other) {
                        report.unknown.push(other.to_string());
                    }
                }
                other => report.unknown.push(other.to_string()),
            }
        }
        fly.set_lif(lif)?;
        fly.set_coupling(coupling);
        Ok(report)
    }
}

/// Collect a tuning from a set of named knobs and the values a search found.
pub fn from_knobs(knobs: &[crate::search::Knob], values: &[f32]) -> Tuning {
    let mut t = Tuning::default();
    for (k, v) in knobs.iter().zip(values) {
        t.set(k.name.clone(), *v);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tuning_round_trips_through_its_own_text() {
        let mut t = Tuning::default();
        t.set("gain:descending", 1.25);
        t.set("adapt_increment", 0.0);
        t.set("command", 2.5);
        let back = Tuning::parse(&t.to_text()).expect("its own output parses");
        assert_eq!(back, t);
        assert_eq!(back.command(), Some(2.5));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored_and_junk_is_an_error() {
        let t = Tuning::parse("# a comment\n\n  v_th 0.5  # trailing\n").expect("parses");
        assert_eq!(t.get("v_th"), Some(0.5));
        let e = Tuning::parse("v_th nonsense\n").expect_err("not a number");
        assert!(e.contains("line 1"), "{e}");
        let e = Tuning::parse("one two three\n").expect_err("three fields");
        assert!(e.contains("line 1"), "{e}");
    }
}
