// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The typed configuration of a Mirage checkpoint.
//!
//! [`MirageConfig`] is the checkpoint's `architecture` object, the same JSON
//! that `config.json` carries and that the safetensors header stores under
//! `__metadata__["config"]`. Every key of the released file has a field here
//! and unknown keys are refused, so a checkpoint from a different revision
//! fails at the door instead of loading with silently different geometry.
//! [`CheckpointConfig`] is the envelope around it.
//!
//! Swedish Embedded AB implements solutions for validating untrusted model
//! configuration at the boundary for its clients. If your team needs expertise
//! in hardening model-serving stacks then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Language tags the tokenizer appends after the text vocabulary
/// (`<|en|> <|es|> <|fr|> <|pt|> <|de|> <|it|> <|nl|> <|pl|> <|ru|> <|ar|> <|zh|>`).
/// The released embedding has `text + 11 + detections + regions` rows; the
/// count is a property of the tokenizer, not of `config.json`.
pub const LANGUAGE_TAGS: usize = 11;

/// Width of the box encoding appended to a region slot's centre feature:
/// sine and cosine of the four normalised box coordinates at eight frequencies.
pub const BOX_ENCODING_DIM: usize = 64;

/// Output channels of the dense head: per cell and kind a score, a width, a
/// height and two centre offsets, for the object and the text heatmap.
pub const DENSE_OUT_CHANNELS: usize = 10;

/// Channels of the mask prototype map and of the per-cell mask coefficients.
pub const MASK_RANK: usize = 32;

/// Residual blocks of the dense head's shared trunk.
pub const DENSE_BLOCKS: usize = 3;

/// Residual blocks of each of the mask-prototype and mask-coefficient branches.
pub const MASK_BRANCH_BLOCKS: usize = 2;

/// Channel widths of the reader's three convolutional stages before the
/// transformer width ([`MirageConfig::reader_dim`]).
pub const READER_STAGES: [usize; 3] = [32, 64, 128];

/// Depthwise kernel width of the reader's mixing blocks.
pub const READER_MIX_KERNEL: usize = 5;

/// Reader output classes: the 306-character charset plus the blank label 0.
pub const READER_CLASSES: usize = 307;

/// Rank of the Hadamard MLP's conditioning factorisation.
pub const HADAMARD_COND_RANK: usize = 8;

/// Taps of the engram value convolution.
pub const ENGRAM_TAPS: usize = 4;

/// Why a configuration was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The JSON is malformed, has an unknown or missing key, or a mistyped value.
    Parse(String),
    /// The JSON parses but a field violates an invariant.
    Invalid { field: String, reason: String },
}

impl ConfigError {
    fn invalid(field: &str, reason: impl Into<String>) -> ConfigError {
        ConfigError::Invalid { field: field.to_string(), reason: reason.into() }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Parse(m) => write!(f, "mirage config: {m}"),
            ConfigError::Invalid { field, reason } => write!(f, "mirage config: field `{field}`: {reason}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Deserialise `json`, naming the offending key path on a type error (plain
/// `serde_json` messages say what was wrong but not where).
fn parse<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, ConfigError> {
    let mut de = serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize(&mut de).map_err(|e| {
        let path = e.path().to_string();
        if path == "." {
            ConfigError::Parse(e.inner().to_string())
        } else {
            ConfigError::Parse(format!("`{path}`: {}", e.inner()))
        }
    })
}

/// The model geometry and training-time settings of one checkpoint.
///
/// Field order is the released file's, so [`MirageConfig::to_json`] reproduces
/// the same key order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirageConfig {
    /// Text vocabulary size, without language tags and reference tokens.
    pub text: usize,
    /// Decoder and encoder model width.
    pub dim: usize,
    /// Decoder key/value heads.
    pub kv_heads: usize,
    /// Decoder query/key head width.
    pub qk_dim: usize,
    /// Decoder value head width; `dim / v_dim` is the query head count.
    pub v_dim: usize,
    /// Decoder layers.
    pub dec_layers: usize,
    /// Causal convolution taps on the decoder's q, k and v projections.
    pub taps: usize,
    /// Hyper-connection residual lanes.
    pub lanes: usize,
    /// Decoder layers that read an engram table, ascending.
    pub sites: Vec<usize>,
    /// N-gram orders of the engram hashes, ascending.
    pub orders: Vec<usize>,
    /// Slots per engram table.
    pub slots: usize,
    /// Hash functions per n-gram order.
    pub hashes: usize,
    /// Decoder rotary base.
    pub rope: f32,
    /// Side of the square canvas the image is letterboxed into.
    pub canvas: usize,
    /// Output channels of each stride-2 stem stage; the last equals `dim`.
    pub stem: Vec<usize>,
    /// Gated-mixing + HConvGLU body blocks.
    pub body_layers: usize,
    /// Band width of the HConvGLU convolutions (odd).
    pub hconv_kernel: usize,
    /// Depthwise kernel of the body's gated mixing (odd).
    pub mix_kernel: usize,
    /// Attention blocks after the body.
    pub attn_layers: usize,
    /// Encoder query heads.
    pub attn_heads: usize,
    /// Encoder key/value heads.
    pub attn_kv_heads: usize,
    /// Encoder query/key head width.
    pub attn_qk_dim: usize,
    /// Encoder value head width.
    pub attn_v_dim: usize,
    /// Encoder rotary base.
    pub enc_rope: f32,
    /// Register tokens appended to the encoder sequence.
    pub registers: usize,
    /// Region slots are pooled box features (the only layout implemented).
    pub pooled_slots: bool,
    /// Body blocks end in global response normalisation (the only layout implemented).
    pub grn: bool,
    /// The dense head has a neck block (the only layout implemented).
    pub neck: bool,
    /// Earlier attention outputs are gated into the final grid (the only layout implemented).
    pub level_memory: bool,
    /// Convolution stages carry a channel norm (the only layout implemented).
    pub conv_norm: bool,
    /// Decoder sequence limit (image memory plus text).
    pub max_tokens: usize,
    /// Sequence-length buckets, ascending, the last equal to `max_tokens`.
    pub buckets: Vec<usize>,
    /// Training mix share of form tasks.
    pub form_share: f32,
    /// Training mix share of trace tasks.
    pub trace_share: f32,
    /// Weight of the name loss.
    pub name_loss: f32,
    /// Weight of the segmentation loss.
    pub seg_loss: f32,
    /// Object proposals per image (also the object reference tokens).
    pub detections: usize,
    /// Text-region proposals per image (also the region reference tokens).
    pub regions: usize,
    /// Reader windows decoded per forward call.
    pub reads: usize,
    /// Mixing weight of the heatmap target.
    pub heat_alpha: f32,
    /// Gradient scale the heads send back into the encoder.
    pub head_grad: f32,
    /// Stride of the dense head's working map.
    pub stride: usize,
    /// Dense head width.
    pub dense_dim: usize,
    /// Reader transformer width.
    pub reader_dim: usize,
    /// Reader transformer blocks.
    pub reader_blocks: usize,
    /// Reader window width in pixels.
    pub read_width: usize,
    /// Padding token id.
    pub pad: u32,
    /// End-of-sequence token id.
    pub eos: u32,
    /// Beginning-of-sequence token id.
    pub bos: u32,
    /// Training compute dtype.
    pub dtype: String,
}

impl MirageConfig {
    /// Parse and validate an `architecture` object.
    pub fn from_json(json: &str) -> Result<MirageConfig, ConfigError> {
        let config: MirageConfig = parse(json)?;
        config.validate()?;
        Ok(config)
    }

    /// Serialise as the `architecture` object (what the safetensors header stores).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a plain struct of numbers and strings serialises")
    }

    /// A tiny valid configuration whose every tensor family exists, for tests
    /// that need the whole structure without the released size.
    pub fn tiny() -> MirageConfig {
        MirageConfig {
            text: 24,
            dim: 32,
            kv_heads: 1,
            qk_dim: 4,
            v_dim: 8,
            dec_layers: 2,
            taps: 3,
            lanes: 2,
            sites: vec![1],
            orders: vec![2, 3],
            slots: 16,
            hashes: 1,
            rope: 10000.0,
            canvas: 64,
            stem: vec![4, 8, 16, 16, 32],
            body_layers: 2,
            hconv_kernel: 5,
            mix_kernel: 3,
            attn_layers: 2,
            attn_heads: 2,
            attn_kv_heads: 1,
            attn_qk_dim: 4,
            attn_v_dim: 8,
            enc_rope: 10000.0,
            registers: 2,
            pooled_slots: true,
            grn: true,
            neck: true,
            level_memory: true,
            conv_norm: true,
            max_tokens: 64,
            buckets: vec![32, 64],
            form_share: 0.1,
            trace_share: 0.2,
            name_loss: 0.5,
            seg_loss: 0.0,
            detections: 4,
            regions: 2,
            reads: 2,
            heat_alpha: 0.5,
            head_grad: 0.1,
            stride: 8,
            dense_dim: 8,
            reader_dim: 16,
            reader_blocks: 1,
            read_width: 32,
            pad: 0,
            eos: 1,
            bos: 2,
            dtype: "float32".to_string(),
        }
    }

    /// Decoder query heads.
    pub fn heads(&self) -> usize {
        self.dim / self.v_dim
    }

    /// Side of the encoder's final grid in cells.
    pub fn grid(&self) -> usize {
        self.canvas >> self.stem.len()
    }

    /// Rows of the tied embedding: text vocabulary, language tags, then one
    /// reference token per detection and per text region.
    pub fn embedding_rows(&self) -> usize {
        self.text + LANGUAGE_TAGS + self.detections + self.regions
    }

    /// Engram tables per site: one per (order, hash) pair.
    pub fn engram_tables(&self) -> usize {
        self.orders.len() * self.hashes
    }

    /// Width of one engram table row.
    pub fn engram_sub_dim(&self) -> usize {
        self.dim / self.engram_tables()
    }

    /// Width the Hadamard MLP works at: `dim` rounded up to a power of two.
    pub fn hadamard_width(&self) -> usize {
        self.dim.next_power_of_two()
    }

    /// The two Kronecker factors of the Hadamard MLP's Walsh rotations, the
    /// smaller first: `2^(q/2)` and `2^(q - q/2)` for a width of `2^q`.
    pub fn hadamard_factors(&self) -> (usize, usize) {
        let q = self.hadamard_width().trailing_zeros() as usize;
        (1 << (q / 2), 1 << (q - q / 2))
    }

    /// Stem stage indices feeding the dense head: the edge map (one stage
    /// finer than the working stride), the fine map (at the stride) and the
    /// mid map (one stage coarser).
    pub fn dense_levels(&self) -> (usize, usize, usize) {
        let fine = self.stride.trailing_zeros() as usize - 1;
        (fine - 1, fine, fine + 1)
    }

    /// Check every invariant the rest of the crate relies on, naming the field.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let positive = |field: &str, v: usize| if v == 0 { Err(ConfigError::invalid(field, "must be positive")) } else { Ok(()) };
        positive("text", self.text)?;
        positive("dim", self.dim)?;
        positive("kv_heads", self.kv_heads)?;
        positive("qk_dim", self.qk_dim)?;
        positive("v_dim", self.v_dim)?;
        positive("dec_layers", self.dec_layers)?;
        positive("taps", self.taps)?;
        positive("lanes", self.lanes)?;
        positive("slots", self.slots)?;
        positive("hashes", self.hashes)?;
        positive("canvas", self.canvas)?;
        positive("body_layers", self.body_layers)?;
        positive("attn_layers", self.attn_layers)?;
        positive("attn_heads", self.attn_heads)?;
        positive("attn_kv_heads", self.attn_kv_heads)?;
        positive("attn_qk_dim", self.attn_qk_dim)?;
        positive("attn_v_dim", self.attn_v_dim)?;
        positive("detections", self.detections)?;
        positive("regions", self.regions)?;
        positive("reads", self.reads)?;
        positive("dense_dim", self.dense_dim)?;
        positive("reader_dim", self.reader_dim)?;
        positive("reader_blocks", self.reader_blocks)?;
        positive("read_width", self.read_width)?;
        positive("max_tokens", self.max_tokens)?;

        // HConvGLU rotates the channel axis with a Walsh matrix, so the width
        // must be a power of two; the Hadamard MLP also needs two factors.
        if !self.dim.is_power_of_two() || self.dim < 4 {
            return Err(ConfigError::invalid("dim", format!("{} is not a power of two of at least 4 (Walsh rotation width)", self.dim)));
        }
        if !self.dim.is_multiple_of(self.v_dim) {
            return Err(ConfigError::invalid("v_dim", format!("dim {} is not divisible by v_dim {}", self.dim, self.v_dim)));
        }
        if !self.heads().is_multiple_of(self.kv_heads) {
            return Err(ConfigError::invalid("kv_heads", format!("{} query heads are not divisible by {} kv heads", self.heads(), self.kv_heads)));
        }
        if !self.qk_dim.is_multiple_of(2) {
            return Err(ConfigError::invalid("qk_dim", format!("{} is odd; rotary embedding rotates pairs", self.qk_dim)));
        }
        ascending("sites", &self.sites)?;
        if let Some(&last) = self.sites.last() {
            if last >= self.dec_layers {
                return Err(ConfigError::invalid("sites", format!("site {last} is beyond the {} decoder layers", self.dec_layers)));
            }
        }
        if self.orders.is_empty() {
            return Err(ConfigError::invalid("orders", "must name at least one n-gram order"));
        }
        ascending("orders", &self.orders)?;
        if self.orders[0] < 1 {
            return Err(ConfigError::invalid("orders", "n-gram orders start at 1"));
        }
        if !self.dim.is_multiple_of(self.engram_tables()) {
            return Err(ConfigError::invalid(
                "hashes",
                format!("{} engram tables (orders x hashes) do not divide dim {}", self.engram_tables(), self.dim),
            ));
        }
        if !self.rope.is_finite() || self.rope <= 1.0 {
            return Err(ConfigError::invalid("rope", format!("{} is not a rotary base above 1", self.rope)));
        }
        if !self.enc_rope.is_finite() || self.enc_rope <= 1.0 {
            return Err(ConfigError::invalid("enc_rope", format!("{} is not a rotary base above 1", self.enc_rope)));
        }

        if self.stem.is_empty() {
            return Err(ConfigError::invalid("stem", "must name at least one stage"));
        }
        if self.stem.contains(&0) {
            return Err(ConfigError::invalid("stem", "a stage has zero channels"));
        }
        if self.stem.last() != Some(&self.dim) {
            return Err(ConfigError::invalid("stem", format!("the last stage ({:?}) must equal dim {}", self.stem.last(), self.dim)));
        }
        if !self.canvas.is_multiple_of(1 << self.stem.len()) {
            return Err(ConfigError::invalid(
                "canvas",
                format!("{} is not divisible by 2^{} (one halving per stem stage)", self.canvas, self.stem.len()),
            ));
        }
        if !self.stride.is_power_of_two() || self.stride < 4 || (self.stride >> 1) >= (1 << self.stem.len()) {
            return Err(ConfigError::invalid(
                "stride",
                format!("{} must be a power of two from 4 up to the stem's total stride {}", self.stride, 1usize << self.stem.len()),
            ));
        }
        let (_, _, mid) = self.dense_levels();
        if mid >= self.stem.len() {
            return Err(ConfigError::invalid("stride", format!("the dense head needs a stem stage coarser than stride {}", self.stride)));
        }
        if self.hconv_kernel.is_multiple_of(2) {
            return Err(ConfigError::invalid("hconv_kernel", format!("{} is even; the band is centred", self.hconv_kernel)));
        }
        if self.mix_kernel.is_multiple_of(2) {
            return Err(ConfigError::invalid("mix_kernel", format!("{} is even; the window is centred", self.mix_kernel)));
        }
        if !self.attn_heads.is_multiple_of(self.attn_kv_heads) {
            return Err(ConfigError::invalid(
                "attn_kv_heads",
                format!("{} encoder heads are not divisible by {} kv heads", self.attn_heads, self.attn_kv_heads),
            ));
        }
        if !self.attn_qk_dim.is_multiple_of(2) {
            return Err(ConfigError::invalid("attn_qk_dim", format!("{} is odd; rotary embedding rotates pairs", self.attn_qk_dim)));
        }
        for (field, on) in [
            ("pooled_slots", self.pooled_slots),
            ("grn", self.grn),
            ("neck", self.neck),
            ("level_memory", self.level_memory),
            ("conv_norm", self.conv_norm),
        ] {
            if !on {
                return Err(ConfigError::invalid(field, "false selects a layout this crate does not implement; only true is supported"));
            }
        }

        if self.buckets.is_empty() {
            return Err(ConfigError::invalid("buckets", "must name at least one length"));
        }
        ascending("buckets", &self.buckets)?;
        if self.buckets.last() != Some(&self.max_tokens) {
            return Err(ConfigError::invalid(
                "buckets",
                format!("the last bucket ({:?}) must equal max_tokens {}", self.buckets.last(), self.max_tokens),
            ));
        }
        for (field, v) in [
            ("form_share", self.form_share),
            ("trace_share", self.trace_share),
            ("heat_alpha", self.heat_alpha),
            ("head_grad", self.head_grad),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(ConfigError::invalid(field, format!("{v} is outside [0, 1]")));
            }
        }
        for (field, v) in [("name_loss", self.name_loss), ("seg_loss", self.seg_loss)] {
            if !v.is_finite() || v < 0.0 {
                return Err(ConfigError::invalid(field, format!("{v} is not a finite non-negative weight")));
            }
        }
        let tokens = self.embedding_rows() as u64;
        let mut specials = [("pad", self.pad), ("eos", self.eos), ("bos", self.bos)];
        for (field, id) in specials {
            if u64::from(id) >= tokens {
                return Err(ConfigError::invalid(field, format!("token id {id} is outside the {tokens} embedding rows")));
            }
        }
        specials.sort_by_key(|&(_, id)| id);
        if let Some(pair) = specials.windows(2).find(|w| w[0].1 == w[1].1) {
            return Err(ConfigError::invalid(pair[1].0, format!("token id {} is shared with `{}`", pair[1].1, pair[0].0)));
        }
        match self.dtype.as_str() {
            "float32" | "bfloat16" | "float16" => {}
            other => return Err(ConfigError::invalid("dtype", format!("`{other}` is not float32, bfloat16 or float16"))),
        }
        Ok(())
    }
}

fn ascending(field: &str, values: &[usize]) -> Result<(), ConfigError> {
    match values.windows(2).find(|w| w[0] >= w[1]) {
        Some(w) => Err(ConfigError::invalid(field, format!("{values:?} is not strictly ascending ({} then {})", w[0], w[1]))),
        None => Ok(()),
    }
}

/// How many bits the released `.cact` gave each tensor family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CactBitsMap {
    /// Bits of the embedding table.
    pub embedding: u32,
    /// Bits of the hyper-connection routing matrices.
    pub mhc: u32,
    /// Bits of every other quantised matrix.
    pub default: u32,
}

/// The whole `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    /// Always `"mirage"`.
    pub model_type: String,
    /// The model geometry.
    pub architecture: MirageConfig,
    /// Optimiser steps the weights were trained for.
    pub training_step: u64,
    /// SHA-256 of the training checkpoint the release was exported from.
    pub source_checkpoint_sha256: String,
    /// Default quantisation width of the `.cact`.
    pub cact_bits: u32,
    /// Per-family quantisation widths of the `.cact`.
    pub cact_bits_map: CactBitsMap,
}

impl CheckpointConfig {
    /// Parse and validate a `config.json`.
    pub fn from_json(json: &str) -> Result<CheckpointConfig, ConfigError> {
        let config: CheckpointConfig = parse(json)?;
        if config.model_type != "mirage" {
            return Err(ConfigError::invalid("model_type", format!("`{}` is not `mirage`", config.model_type)));
        }
        config.architecture.validate()?;
        if config.source_checkpoint_sha256.len() != 64 || !config.source_checkpoint_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(ConfigError::invalid("source_checkpoint_sha256", "is not a 64-digit hexadecimal SHA-256"));
        }
        for (field, bits) in [
            ("cact_bits", config.cact_bits),
            ("cact_bits_map.embedding", config.cact_bits_map.embedding),
            ("cact_bits_map.mhc", config.cact_bits_map.mhc),
            ("cact_bits_map.default", config.cact_bits_map.default),
        ] {
            if !(2..=4).contains(&bits) {
                return Err(ConfigError::invalid(field, format!("{bits} bits has no codebook; supported widths are 2, 3 and 4")));
            }
        }
        Ok(config)
    }
}
