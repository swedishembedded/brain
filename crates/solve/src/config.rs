// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The policy's shape.

use serde_json::Value;

/// A residual MLP: one projection in, `blocks` residual blocks, one head out.
///
/// Deliberately not a transformer. The input is a fixed-width one-hot with no
/// sequence structure to attend over, and an attention stack would spend its
/// arithmetic re-deriving a position encoding that the feature layout already
/// states exactly. A wide MLP over a one-hot is a table lookup followed by
/// dense mixing, which is what this problem actually is.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Width of the one-hot the state space writes.
    pub in_dim: u32,
    /// Trunk width, carried by the residual stream.
    pub d_model: u32,
    /// Inner width of each residual block.
    pub d_ff: u32,
    pub blocks: u32,
    /// Output width: one logit per move.
    pub moves: u32,
}

impl Config {
    /// Every parameter and its shape, in one place, so the store, the
    /// initialiser and any importer cannot disagree about what exists.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (i, d, ff, m) = (
            self.in_dim as usize,
            self.d_model as usize,
            self.d_ff as usize,
            self.moves as usize,
        );
        let mut v = vec![("stem.weight".into(), vec![i, d]), ("stem.bias".into(), vec![d])];
        for b in 0..self.blocks {
            let p = format!("blocks.{b}");
            v.extend([
                (format!("{p}.up.weight"), vec![d, ff]),
                (format!("{p}.up.bias"), vec![ff]),
                (format!("{p}.down.weight"), vec![ff, d]),
                (format!("{p}.down.bias"), vec![d]),
            ]);
        }
        v.extend([("head.weight".into(), vec![d, m]), ("head.bias".into(), vec![m])]);
        // The cost-to-go head. One scalar, sharing the trunk with the policy:
        // "which move" and "how far from the goal" are the same question
        // asked two ways, and a representation good enough for one is good
        // enough for the other.
        v.extend([("value.weight".into(), vec![d, 1]), ("value.bias".into(), vec![1])]);
        v
    }

    pub fn params(&self) -> usize {
        self.tensor_manifest().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "architecture": "residual-mlp-policy",
            "in_dim": self.in_dim,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "blocks": self.blocks,
            "moves": self.moves,
        })
    }

    /// Read a config this crate wrote. Every field required: a checkpoint
    /// that lost `blocks` would load as a different network and mismatch its
    /// own weights somewhere far from the cause.
    pub fn from_json(text: &str) -> Result<Config, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("config: {e}"))?;
        let u = |k: &str| -> Result<u32, String> {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|x| x as u32)
                .ok_or_else(|| format!("config: missing or non-integer {k:?}"))
        };
        Ok(Config {
            in_dim: u("in_dim")?,
            d_model: u("d_model")?,
            d_ff: u("d_ff")?,
            blocks: u("blocks")?,
            moves: u("moves")?,
        })
    }
}
