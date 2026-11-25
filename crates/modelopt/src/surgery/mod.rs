// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for carving smaller models out of
// trained ones without leaving the layout their upstream tooling loads, for
// its clients. If your team needs expertise in structural model compression
// and weight surgery then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Weight surgery: a checkpoint made physically smaller along the dimensions
//! its model declares ([`model::surgery::SurgeryGraph`]), as a checkpoint of
//! the same architecture whose config states every new width. No edit leaves
//! a factorised tensor or an index table behind.
//!
//! The edits, each on a [`Checkpoint`] in memory and each answering a
//! [`Change`] for the manifest:
//!
//! * [`select_channels`]: keep chosen channels of declared dimensions (an
//!   MLP's hidden channels, an attention layer's KV groups with their query
//!   heads); [`mask_channels`] zeroes the others instead, which computes the
//!   same function at the original shapes.
//! * [`remove_layers`]: drop layers of a stack and renumber the rest;
//!   [`mask_layers`] makes them the identity instead.
//! * [`rotate_stream`]: a residual stream rotated into an orthonormal basis
//!   and cut to its first `k` directions (norm gains folded into the linears
//!   that read the stream), every tensor coupled to the stream with it.
//! * [`fold_embedding`]: an embedding that feeds a square linear map narrowed
//!   to chosen output channels of that map, the map folded into the table.
//! * [`trim_vocabulary`]: the rows of kept token ids, renumbered densely, and
//!   every config key holding a token id rewritten; [`bpe`] works out which
//!   ids a byte-level BPE tokenizer must keep for a corpus and rewrites its
//!   files.
//!
//! The scores an edit is chosen by come from calibration ([`scores`]): any
//! [`model::Model`]'s curvature (the Fisher and group-Taylor estimates) and
//! activation statistics the model's own probe collects.

pub mod bpe;
pub mod calibration;
mod channels;
mod fold;
mod layers;
pub mod plan;
pub mod scores;
mod stream;
mod vocab;

use std::collections::BTreeMap;

use model::surgery::Axis;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use channels::{mask_channels, select_channels};
pub use fold::fold_embedding;
pub use layers::{mask_layers, remove_layers};
pub use stream::{basis_from_covariance, rotate_stream, Basis, BasisKind, SharedNormFit};
pub use vocab::trim_vocabulary;

/// One stored tensor, as f32, with its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor, String> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(format!("a tensor of shape {shape:?} holds {n} values, not {}", data.len()));
        }
        Ok(Tensor { shape, data })
    }

    /// The `[rows, cols]` view: `rows = shape[0]`, `cols` the rest (1 for a vector).
    pub fn rows_cols(&self) -> (usize, usize) {
        let rows = self.shape.first().copied().unwrap_or(1);
        (rows, self.shape.iter().skip(1).product())
    }
}

/// A checkpoint under surgery: its config JSON and every tensor under its
/// stored name.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub config: Value,
    pub tensors: BTreeMap<String, Tensor>,
}

impl Checkpoint {
    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors.get(name).ok_or_else(|| format!("the checkpoint has no tensor {name}"))
    }

    pub fn get_mut(&mut self, name: &str) -> Result<&mut Tensor, String> {
        self.tensors.get_mut(name).ok_or_else(|| format!("the checkpoint has no tensor {name}"))
    }

    /// The checkpoint at `dir`: its `config.json` and every tensor of its
    /// `model.safetensors`, as f32.
    pub fn read(dir: &std::path::Path) -> Result<Checkpoint, String> {
        let at = |what: &str, e: &dyn std::fmt::Display| format!("{}: {what}: {e}", dir.display());
        let config: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(|e| at("config.json", &e))?).map_err(|e| at("config.json", &e))?;
        let path = dir.join("model.safetensors");
        let reader = checkpoint::weightio::WeightReader::open(path.to_str().ok_or_else(|| at("path", &"not UTF-8"))?).map_err(|e| at("model.safetensors", &e))?;
        let mut tensors = BTreeMap::new();
        let names: Vec<String> = reader.names().map(str::to_string).collect();
        for name in names {
            let shape: Vec<usize> = reader.shape(&name).unwrap_or_default().iter().map(|&d| d as usize).collect();
            let data = reader.tensor(&name).ok_or_else(|| at("model.safetensors", &format!("{name} does not read as real numbers")))?;
            tensors.insert(name, Tensor::new(shape, data).map_err(|e| at("model.safetensors", &e))?);
        }
        Ok(Checkpoint { config, tensors })
    }

    /// Every tensor as `(name, shape, values)`, the form a checkpoint writer takes.
    pub fn to_list(&self) -> Vec<(String, Vec<u64>, Vec<f32>)> {
        self.tensors.iter().map(|(n, t)| (n.clone(), t.shape.iter().map(|&d| d as u64).collect(), t.data.clone())).collect()
    }

    /// Every parameter, counted.
    pub fn params(&self) -> u64 {
        self.tensors.values().map(|t| t.data.len() as u64).sum()
    }

    /// The config value at the dotted `path`.
    pub fn config_u64(&self, path: &str) -> Result<u64, String> {
        lookup(&self.config, path)?.as_u64().ok_or_else(|| format!("config key {path} is not an unsigned integer"))
    }

    /// Set the config value at the dotted `path`, which must already exist.
    pub fn set_config(&mut self, path: &str, value: Value) -> Result<(), String> {
        let (parent, leaf) = match path.rsplit_once('.') {
            Some((p, l)) => (lookup_mut(&mut self.config, p)?, l),
            None => (&mut self.config, path),
        };
        let slot = parent.get_mut(leaf).ok_or_else(|| format!("the config has no key {path} to rewrite"))?;
        *slot = value;
        Ok(())
    }

    fn shapes(&self) -> BTreeMap<String, Vec<usize>> {
        self.tensors.iter().map(|(n, t)| (n.clone(), t.shape.clone())).collect()
    }
}

fn lookup<'a>(v: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.').try_fold(v, |v, k| v.get(k).ok_or_else(|| format!("the config has no key {path}")))
}

fn lookup_mut<'a>(v: &'a mut Value, path: &str) -> Result<&'a mut Value, String> {
    path.split('.').try_fold(v, |v, k| v.get_mut(k).ok_or_else(|| format!("the config has no key {path}")))
}

/// What one edit did, for the manifest: what it was, which channels, layers
/// or directions it kept, every tensor whose shape changed (or that went),
/// every config value it rewrote, and the parameter count around it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Change {
    /// The edit, e.g. `select_channels`.
    pub edit: String,
    /// What it acted on: dimension, stack, stream, fold or vocabulary names.
    pub targets: Vec<String>,
    /// Per target, the kept indices (channels, layers, directions or token ids), ascending.
    pub kept: BTreeMap<String, Vec<usize>>,
    /// `tensor -> [before, after]`; `after` is empty for a removed tensor.
    pub shapes: BTreeMap<String, [Vec<usize>; 2]>,
    /// `config key -> [before, after]`.
    pub config: BTreeMap<String, [Value; 2]>,
    pub params_before: u64,
    pub params_after: u64,
}

impl Change {
    fn begin(edit: &str, ck: &Checkpoint) -> (Change, BTreeMap<String, Vec<usize>>) {
        (Change { edit: edit.to_string(), params_before: ck.params(), ..Change::default() }, ck.shapes())
    }

    fn finish(mut self, ck: &Checkpoint, before: BTreeMap<String, Vec<usize>>) -> Change {
        for (name, shape) in &before {
            match ck.tensors.get(name) {
                Some(t) if &t.shape == shape => {}
                Some(t) => {
                    self.shapes.insert(name.clone(), [shape.clone(), t.shape.clone()]);
                }
                None => {
                    self.shapes.insert(name.clone(), [shape.clone(), Vec::new()]);
                }
            }
        }
        self.params_after = ck.params();
        self
    }

    /// Rewrite config `key` to `value`, recording both values.
    fn set(&mut self, ck: &mut Checkpoint, key: &str, value: Value) -> Result<(), String> {
        let before = lookup(&ck.config, key)?.clone();
        ck.set_config(key, value.clone())?;
        self.config.entry(key.to_string()).or_insert([before, Value::Null])[1] = value;
        Ok(())
    }
}

/// The record of a whole surgery: what it started from, what calibrated it,
/// every edit in order and what was measured after each.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The source checkpoint's content digest.
    pub source: String,
    /// The calibration statistics' content digest, when an edit was scored by them.
    pub calibration: Option<String>,
    pub edits: Vec<EditRecord>,
}

/// One edit of a [`Manifest`]: the change, the scoring method that chose
/// what it kept, and the loss measured after it (before any recovery).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditRecord {
    pub change: Change,
    /// How what was kept was chosen (`wanda`, `ablation`, `pca`, `corpus`, `given`, ...).
    pub method: String,
    pub measured: Option<Measured>,
}

/// Losses of an edited model on held-out data.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    /// The model's training loss, teacher forced.
    pub loss: f64,
    /// The mean KL divergence from the unedited model's predictions to the edited one's.
    pub kl: f64,
}

/// Lines `block` long along `axis` of a `[rows, cols]` tensor: how many there are.
fn line_count(t: &Tensor, axis: Axis) -> usize {
    let (rows, cols) = t.rows_cols();
    match axis {
        Axis::Rows => rows,
        Axis::Cols => cols,
    }
}

/// The tensor keeping only `keep` (ascending line indices) along `axis`.
fn keep_lines(t: &Tensor, axis: Axis, keep: &[usize]) -> Result<Tensor, String> {
    let (rows, cols) = t.rows_cols();
    match axis {
        Axis::Rows => {
            let data = keep.iter().flat_map(|&r| t.data[r * cols..(r + 1) * cols].iter().copied()).collect();
            let mut shape = t.shape.clone();
            if shape.is_empty() {
                shape.push(keep.len());
            } else {
                shape[0] = keep.len();
            }
            Ok(Tensor { shape, data })
        }
        Axis::Cols => {
            if t.shape.len() != 2 {
                return Err(format!("a tensor of shape {:?} has no column axis to cut", t.shape));
            }
            let data = (0..rows).flat_map(|r| keep.iter().map(move |&c| t.data[r * cols + c])).collect();
            Ok(Tensor { shape: vec![rows, keep.len()], data })
        }
    }
}

/// Zero the lines of `t` along `axis` that `keep` (ascending) does not name.
fn zero_lines(t: &mut Tensor, axis: Axis, keep: &[usize]) {
    let (rows, cols) = t.rows_cols();
    let n = match axis {
        Axis::Rows => rows,
        Axis::Cols => cols,
    };
    for line in (0..n).filter(|l| keep.binary_search(l).is_err()) {
        match axis {
            Axis::Rows => t.data[line * cols..(line + 1) * cols].fill(0.0),
            Axis::Cols => (0..rows).for_each(|r| t.data[r * cols + line] = 0.0),
        }
    }
}

/// `keep`, checked to be strictly ascending and below `width`, naming `what`.
fn check_keep(what: &str, keep: &[usize], width: usize) -> Result<(), String> {
    if keep.is_empty() {
        return Err(format!("{what}: keeping nothing leaves no model"));
    }
    if keep.windows(2).any(|w| w[0] >= w[1]) {
        return Err(format!("{what}: kept indices must be strictly ascending"));
    }
    if let Some(&bad) = keep.iter().find(|&&k| k >= width) {
        return Err(format!("{what}: index {bad} is past its {width} channels"));
    }
    Ok(())
}

/// Indices `0..n` ordered by `scores` descending (ties by index), the first
/// `k` of them ascending: the `k` highest-scoring channels.
pub fn top_k(scores: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order.truncate(k);
    order.sort_unstable();
    order
}

#[cfg(test)]
mod tests;
