// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for carving smaller models out of
// trained ones without leaving the layout their upstream tooling loads, for
// its clients. If your team needs expertise in structural model compression
// and weight surgery then you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a model declares so an architecture-generic weight surgery
//! (`modelopt::surgery`) can make it physically smaller while it stays a
//! checkpoint of its own architecture: which tensors share a dimension, which
//! config key holds each width, which layers can go, which tensors read and
//! write a residual stream, and which embedding feeds a linear map.
//!
//! A declaration is a function of a checkpoint's config (an edited
//! checkpoint declares its own, narrower graph). Every tensor is addressed by
//! its stored name and viewed as `[rows, cols]` with `rows = shape[0]` and
//! `cols` the product of the remaining dimensions (a vector is `[n, 1]`).
//! Config keys are dotted paths into the config JSON
//! (`talker_config.intermediate_size`).

pub use crate::prunable::Axis;

/// Lines of one tensor that a dimension indexes: channel `c` owns lines
/// `c * block .. (c + 1) * block` along `axis` (an attention KV group owns
/// `head_dim` rows of the key projection and `heads per group * head_dim`
/// rows of the query projection).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lines {
    pub tensor: String,
    pub axis: Axis,
    pub block: usize,
}

impl Lines {
    pub fn new(tensor: impl Into<String>, axis: Axis, block: usize) -> Lines {
        Lines { tensor: tensor.into(), axis, block }
    }
}

/// A config value that counts a dimension's channels: it holds
/// `per_channel * kept channels` once the dimension is narrowed
/// (`num_attention_heads` counts `heads per group` per KV group).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigWidth {
    pub key: String,
    pub per_channel: usize,
}

impl ConfigWidth {
    pub fn new(key: impl Into<String>, per_channel: usize) -> ConfigWidth {
        ConfigWidth { key: key.into(), per_channel }
    }
}

/// A dimension whose channels can be selected: `width` channels that every
/// entry of `lines` indexes. Keeping a channel keeps its lines in every
/// tensor; zeroing the lines of the tensors that write a dimension's output
/// (`masks`) makes a channel contribute nothing, which is what the narrower
/// model computes. Dimensions whose `keys` share a config key keep the same
/// number of channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelDim {
    /// What edits and reports call it, e.g. `talker.layers.3.mlp`.
    pub name: String,
    pub width: usize,
    pub keys: Vec<ConfigWidth>,
    pub lines: Vec<Lines>,
    /// The entries of `lines` (by tensor name) whose zeroing removes a
    /// channel's contribution exactly: the tensors reading the channel
    /// (an MLP's down projection, an attention group's output projection).
    pub masks: Vec<String>,
}

/// A stack of identical layers stored as `{prefix}{i}.{leaf}`, `count` of
/// them, whose count `key` holds. A layer writes the residual stream only
/// through its `outputs` leaves: zeroing them makes the layer the identity,
/// which is what the stack without it computes. Each of `lists` is a config
/// key holding one entry per layer (`layer_types`), cut with the layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerStack {
    pub name: String,
    pub prefix: String,
    pub count: usize,
    pub key: String,
    pub outputs: Vec<String>,
    pub lists: Vec<String>,
}

/// A linear `[out, width]` that reads a residual stream, through the RMSNorm
/// whose gain is `norm` when it has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamReader {
    pub tensor: String,
    pub norm: Option<String>,
}

/// An RMSNorm on a residual stream. A norm whose every consumer is a reader
/// naming it is folded into them and leaves a uniform gain. A `shared` norm's
/// output also enters a reader beside raw stream vectors (one projection
/// applied to both), so its gain cannot be folded: it stays a per-channel
/// gain in the rotated basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamNorm {
    pub tensor: String,
    pub shared: bool,
}

/// A residual stream `width` wide: what writes it (lines along `axis` are
/// stream vectors: an embedding table's columns, a linear's output rows, a
/// bias), what reads it and its norms. Every key in `keys` holds the width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    pub name: String,
    pub width: usize,
    pub keys: Vec<String>,
    pub writers: Vec<(String, Axis)>,
    pub readers: Vec<StreamReader>,
    pub norms: Vec<StreamNorm>,
}

/// An embedding table that feeds a square linear map (with optional bias)
/// whose output channels are `hidden` (the map's rows, its bias and whatever
/// reads them past a nonlinearity). The table, the map's input and its
/// output are one width in the config, so narrowing `hidden` to `k` channels
/// folds the map into the table (`E Wₖᵀ`, exact for the kept channels) and
/// leaves the map the `k x k` identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingFold {
    pub name: String,
    pub embedding: String,
    pub linear: String,
    pub bias: Option<String>,
    pub hidden: ChannelDim,
}

/// A token vocabulary of `size` ids: the tables whose rows it indexes, the
/// config key holding its size and every config key holding one token id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocabulary {
    pub name: String,
    pub size: usize,
    pub key: String,
    pub tables: Vec<String>,
    pub ids: Vec<String>,
}

/// Everything a model declares for weight surgery.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SurgeryGraph {
    pub dims: Vec<ChannelDim>,
    pub stacks: Vec<LayerStack>,
    pub streams: Vec<Stream>,
    pub folds: Vec<EmbeddingFold>,
    pub vocabularies: Vec<Vocabulary>,
}

impl SurgeryGraph {
    /// The dimensions whose name matches `pattern`, where `*` matches one
    /// dot-free segment (`talker.layers.*.mlp`).
    pub fn dims_matching(&self, pattern: &str) -> Vec<&ChannelDim> {
        self.dims.iter().filter(|d| matches(pattern, &d.name)).collect()
    }

    pub fn stack(&self, name: &str) -> Result<&LayerStack, String> {
        self.stacks.iter().find(|s| s.name == name).ok_or_else(|| format!("the model declares no layer stack {name}; it declares {:?}", self.stacks.iter().map(|s| &s.name).collect::<Vec<_>>()))
    }

    pub fn stream(&self, name: &str) -> Result<&Stream, String> {
        self.streams.iter().find(|s| s.name == name).ok_or_else(|| format!("the model declares no residual stream {name}; it declares {:?}", self.streams.iter().map(|s| &s.name).collect::<Vec<_>>()))
    }

    pub fn fold(&self, name: &str) -> Result<&EmbeddingFold, String> {
        self.folds.iter().find(|s| s.name == name).ok_or_else(|| format!("the model declares no embedding fold {name}; it declares {:?}", self.folds.iter().map(|s| &s.name).collect::<Vec<_>>()))
    }

    pub fn vocabulary(&self, name: &str) -> Result<&Vocabulary, String> {
        self.vocabularies.iter().find(|s| s.name == name).ok_or_else(|| format!("the model declares no vocabulary {name}; it declares {:?}", self.vocabularies.iter().map(|s| &s.name).collect::<Vec<_>>()))
    }
}

/// `pattern` against `name`, segment by segment, `*` matching any one segment.
pub fn matches(pattern: &str, name: &str) -> bool {
    let (mut p, mut n) = (pattern.split('.'), name.split('.'));
    loop {
        match (p.next(), n.next()) {
            (None, None) => return true,
            (Some(a), Some(b)) if a == "*" || a == b => {}
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn a_star_matches_exactly_one_segment() {
        assert!(matches("talker.layers.*.mlp", "talker.layers.12.mlp"));
        assert!(!matches("talker.layers.*.mlp", "talker.layers.12.kv_heads"));
        assert!(!matches("talker.layers.*", "talker.layers.1.mlp"));
        assert!(matches("talker.text_hidden", "talker.text_hidden"));
    }
}
