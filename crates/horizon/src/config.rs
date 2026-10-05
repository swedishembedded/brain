// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model configuration: sizes, the value and time-ago bins, and the hazard
//! pieces. Serialised into the checkpoint, so a saved model rebuilds itself.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Value bins beyond the `value_bins` soft bins: the value is hidden for the
/// masked-value objective.
pub const BIN_MASK: u32 = 0;
/// The value is at or below the recorded number (a detection limit).
pub const BIN_BELOW: u32 = 1;
/// The value is at or above the recorded number.
pub const BIN_ABOVE: u32 = 2;
/// The token carries no number (a category, an event): it is present.
pub const BIN_PRESENT: u32 = 3;
/// How many special value bins follow the soft bins.
pub const SPECIAL_VALUE_BINS: u32 = 4;
/// The time-ago axis spans this many units (for years: a century); older
/// history saturates the last bin.
pub const TIME_AGO_SPAN: f64 = 100.0;

/// Default hazard knots over time since prediction, in the dataset's unit
/// (years): finer early, where most follow-up is, to two decades.
pub const DEFAULT_KNOTS: [f32; 17] = [
    0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
];

/// Configuration of a [`crate::Horizon`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HorizonConfig {
    /// Token vocabulary size (from the fitted [`crate::vocab::Vocab`]).
    pub vocab: u32,
    /// Tokens per subject including the summary token; longer histories are
    /// truncated deterministically and the truncation is counted.
    pub max_tokens: u32,
    /// Width of the token representation.
    pub d_model: u32,
    /// Set-encoder layers.
    pub n_layers: u32,
    /// Attention heads.
    pub n_heads: u32,
    /// Feed-forward width.
    pub d_ff: u32,
    /// Soft bins over a value's empirical CDF.
    pub value_bins: u32,
    /// Soft bins over how long ago an observation was made.
    pub time_bins: u32,
    /// Rank of the hazard head (shared across outcome codes).
    pub rank: u32,
    /// Outcome codes.
    pub n_codes: u32,
    /// Hazard piece boundaries over time since prediction, starting at 0.
    pub knots: Vec<f32>,
    /// Weight of the masked-value objective relative to the event objective.
    pub value_weight: f32,
}

impl HorizonConfig {
    /// A small configuration for tests and gradient checks.
    pub fn tiny(vocab: u32, n_codes: u32) -> HorizonConfig {
        HorizonConfig {
            vocab,
            max_tokens: 8,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
            d_ff: 12,
            value_bins: 4,
            time_bins: 3,
            rank: 5,
            n_codes,
            knots: vec![0.0, 1.0, 2.5, 4.0],
            value_weight: 0.5,
        }
    }

    /// Hazard pieces.
    pub fn pieces(&self) -> u32 {
        self.knots.len() as u32 - 1
    }

    /// Value-bin table width (soft bins plus the special bins).
    pub fn value_table(&self) -> u32 {
        self.value_bins + SPECIAL_VALUE_BINS
    }

    /// Time-ago table width (soft bins plus "measured at entry").
    pub fn time_table(&self) -> u32 {
        self.time_bins + 1
    }

    /// Per-(subject, piece) time features: age and its square, calendar time,
    /// and a one-hot of the piece.
    pub fn time_features(&self) -> u32 {
        3 + self.pieces()
    }

    /// Check the configuration is buildable.
    pub fn validate(&self) -> Result<(), String> {
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(format!(
                "horizon: d_model {} is not a multiple of n_heads {}",
                self.d_model, self.n_heads
            ));
        }
        if self.knots.len() < 2
            || self.knots[0] != 0.0
            || self.knots.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(format!(
                "horizon: knots must start at 0 and increase, got {:?}",
                self.knots
            ));
        }
        if self.max_tokens < 2
            || self.vocab < 3
            || self.n_codes == 0
            || self.value_bins < 2
            || self.time_bins < 2
        {
            return Err("horizon: max_tokens >= 2, vocab >= 3, n_codes >= 1, value_bins >= 2, time_bins >= 2".into());
        }
        Ok(())
    }

    /// Parameter list, `(name, numel)`, weights in the `out = x @ W^T` layout.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let (d, ff, r) = (
            self.d_model as usize,
            self.d_ff as usize,
            self.rank as usize,
        );
        let mut v = vec![
            ("tok.gamma".to_string(), self.vocab as usize * d),
            ("tok.beta".to_string(), self.vocab as usize * d),
            (
                "value_bins.weight".to_string(),
                d * self.value_table() as usize,
            ),
            (
                "time_bins.weight".to_string(),
                d * self.time_table() as usize,
            ),
        ];
        for l in 0..self.n_layers {
            let p = |n: &str| format!("blocks.{l}.{n}");
            v.extend([
                (p("ln1.weight"), d),
                (p("ln1.bias"), d),
                (p("attn.qkv.weight"), 3 * d * d),
                (p("attn.qkv.bias"), 3 * d),
                (p("attn.out.weight"), d * d),
                (p("attn.out.bias"), d),
                (p("ln2.weight"), d),
                (p("ln2.bias"), d),
                (p("ffn.up.weight"), ff * d),
                (p("ffn.up.bias"), ff),
                (p("ffn.down.weight"), d * ff),
                (p("ffn.down.bias"), d),
            ]);
        }
        v.extend([
            ("ln_f.weight".to_string(), d),
            ("ln_f.bias".to_string(), d),
            ("value_head.weight".to_string(), 2 * d),
            ("value_head.bias".to_string(), 2),
            ("hazard.state.weight".to_string(), r * d),
            ("hazard.state.bias".to_string(), r),
            (
                "hazard.time.weight".to_string(),
                r * self.time_features() as usize,
            ),
            ("hazard.code.weight".to_string(), self.n_codes as usize * r),
            ("hazard.code.bias".to_string(), self.n_codes as usize),
        ]);
        v
    }

    /// Serialise (with a `model` tag) for the checkpoint header.
    pub fn to_json(&self) -> Value {
        let mut v = serde_json::to_value(self).expect("config serialises");
        v["model"] = Value::String("horizon".into());
        v
    }

    /// Deserialise from a checkpoint header.
    pub fn from_json(v: &Value) -> Result<HorizonConfig, String> {
        let mut v = v.clone();
        if let Some(o) = v.as_object_mut() {
            o.remove("model");
        }
        let c: HorizonConfig =
            serde_json::from_value(v).map_err(|e| format!("horizon config: {e}"))?;
        c.validate()?;
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_round_trips_and_validates() {
        let c = HorizonConfig::tiny(10, 3);
        assert_eq!(HorizonConfig::from_json(&c.to_json()).unwrap(), c);
        let mut bad = c.clone();
        bad.knots = vec![0.0, 2.0, 1.0];
        assert!(bad.validate().is_err());
        bad = c.clone();
        bad.n_heads = 3;
        assert!(bad.validate().is_err());
        assert_eq!(c.pieces(), 3);
        assert_eq!(c.time_features(), 6);
    }
}
