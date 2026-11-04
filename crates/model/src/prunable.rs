// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The dimensions of a model that can be narrowed by removing whole channels:
//! what a model declares so a structured-pruning search can rank, remove and
//! export them without knowing the architecture.

/// Which axis of a stored `[rows, cols]` tensor a pruned channel indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Rows,
    Cols,
}

/// One prunable dimension: `width` channels that every tensor in `slices`
/// indexes along its axis (an MLP's hidden channels are the rows of its gate
/// and up projections and the columns of its down projection). Removing a
/// channel removes it from every slice at once; zeroing the same slices
/// makes the channel contribute nothing, which is what a pruned model
/// computes. Dimensions with the same `config_key` share one width in the
/// config ([`crate::ModelConfig::to_json`]), so they are narrowed together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrunableDim {
    /// What reports call it, e.g. `"blocks.3.mlp"`.
    pub name: String,
    pub width: usize,
    /// The config JSON key that holds the width, e.g. `"d_ff"`.
    pub config_key: &'static str,
    pub slices: Vec<(String, Axis)>,
}
