// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint-compatible configurations for looped mixture-of-experts decoders for its
// clients. If your team needs expertise in importing a model's exact architecture from its published files then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The typed `config.json` of an Inkling 2 checkpoint.
//!
//! Every key of the published files has a field and unknown keys are refused, at every level: a checkpoint from a
//! revision that adds a switch fails at the door instead of loading with different math. [`CheckpointConfig`] is the
//! file as published; [`Inkling2Config`] is the validated geometry the name map and the forward read.
//!
//! Two facts are derived rather than read:
//! - the layer kinds come from `local_layer_ids` (sliding-window layers; the rest attend over the whole history). The
//!   reference ignores that list and derives the kinds from `(i + 1) % 6`, or from an explicit `layer_types`; a config
//!   where those disagree with the list would run differently there, so it is refused rather than guessed;
//! - the loop: `inkling2_loop` declares `loop_config` (a core of whole depth-attention blocks run `num_passes` times),
//!   `inkling2` has none and is the same recipe with an empty loop.

use std::fmt;

use serde::Deserialize;

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
            ConfigError::Parse(m) => write!(f, "inkling2 config: {m}"),
            ConfigError::Invalid { field, reason } => write!(f, "inkling2 config: field `{field}`: {reason}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// The published `config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    pub dtype: Option<String>,
    pub architectures: Vec<String>,
    pub model_type: String,
    pub eos_token_id: Option<u32>,
    pub text_config: TextConfig,
    pub vision_config: Option<VisionConfig>,
    pub mtp_config: Option<MtpConfig>,
    /// The release's coarse global claim (`modelopt` / `NVFP4`). It is not accurate for most tensors: the per-tensor
    /// declaration in `hf_quant_config.json` is what `checkpoint::packed` reads.
    pub quantization_config: Option<QuantizationConfig>,
}

/// `text_config`: the decoder.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub model_type: String,
    pub model_max_length: u64,
    pub torch_dtype: Option<String>,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    pub rel_extent: u32,
    pub q_bias: bool,
    pub o_bias: bool,
    pub log_scaling_n_floor: u32,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    pub use_value_norm_no_gamma: bool,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub local_layer_ids: Vec<u32>,
    /// The reference's own spelling of the layer kinds (`hybrid_sliding` / `hybrid`); optional, checked against
    /// `local_layer_ids` when present.
    pub layer_types: Option<Vec<String>>,
    pub sconv_kernel_size: u32,
    pub unpadded_vocab_size: u32,
    pub final_logit_softcapping: f64,
    pub swa_head_dim: u32,
    pub swa_num_attention_heads: u32,
    pub swa_num_key_value_heads: u32,
    pub sliding_window_size: u32,
    pub n_layers_per_attn_res_block: u32,
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub n_shared_experts: u32,
    pub intermediate_size: u32,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub route_scale: f64,
    pub use_gate_bias: bool,
    pub scoring_func: Option<String>,
    pub eos_token_id: Option<u32>,
    pub loop_config: Option<LoopConfigJson>,
}

/// `text_config.loop_config`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoopConfigJson {
    pub core_start: u32,
    pub core_end: u32,
    pub num_passes: u32,
    pub gate_rank: u32,
    pub is_varloop: bool,
}

/// `vision_config`: the image tower (served by a later phase; parsed and checked here so the name map covers its
/// tensors).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisionConfig {
    pub model_type: String,
    pub text_hidden_size: u32,
    pub temporal_patch_size: u32,
    pub native_patch_size: u32,
    pub adapter_merge_size: u32,
    pub projection_hidden_dim: u32,
    pub projector_bias: bool,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub mlp_ratio: f64,
    pub posemb_grid_size: u32,
    pub layer_norm_eps: f64,
}

/// `mtp_config`: the optional block drafter in `mtp.safetensors` (a separate source; not part of the decoder's map).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtpConfig {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub max_position_embeddings: u64,
    pub torch_dtype: Option<String>,
    pub q_bias: bool,
    pub o_bias: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub local_layer_ids: Vec<u32>,
    pub sliding_window_size: u32,
    pub block_size: u32,
    pub markov_rank: u32,
    pub markov_head_type: String,
    pub target_layer_ids: Vec<u32>,
}

/// `quantization_config`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub quant_algo: String,
}

/// Which family member a config declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// `Inkling2LoopForConditionalGeneration` / `inkling2_loop`: a looped core with messages between passes.
    Looped,
    /// `Inkling2ForConditionalGeneration` / `inkling2`: every layer visited once.
    Plain,
}

/// The attention a layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Causal attention over the last `window` positions, the relative bias over the same extent.
    Sliding,
    /// Causal attention over the whole history; three depth-attention inputs (q, k, v), dynamic K/V convolutions,
    /// and the logarithmic position scale on queries and relative states.
    Full,
}

/// One attention geometry: query heads, KV heads and their width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl AttnShape {
    pub fn q_dim(&self) -> u32 {
        self.heads * self.head_dim
    }
    pub fn kv_dim(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// The looped core: layers `start..end` (whole depth-attention blocks) run `passes` times; passes after the first
/// restart from the prelude's states and read messages from the previous pass through gates of rank `gate_rank`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopSpec {
    pub start: u32,
    pub end: u32,
    pub passes: u32,
    pub gate_rank: u32,
}

impl LoopSpec {
    pub fn layers(&self) -> u32 {
        self.end - self.start
    }
}

/// The validated geometry of the decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct Inkling2Config {
    pub family: Family,
    pub hidden: u32,
    pub n_layers: u32,
    /// Rows of the stored embedding and unembedding.
    pub vocab: u32,
    /// Rows of the unembedding that are real tokens; the rest are padding and never scored.
    pub valid_vocab: u32,
    pub max_positions: u64,
    pub full: AttnShape,
    pub sliding: AttnShape,
    pub window: u32,
    pub d_rel: u32,
    /// Backward distance the full layers' relative bias covers (the sliding layers' covers `window`).
    pub rel_extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub conv_taps: u32,
    pub logit_cap: f32,
    /// Layers summed into one depth-attention block.
    pub block: u32,
    pub experts: u32,
    pub top_k: u32,
    pub moe_ff: u32,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub route_scale: f32,
    pub kinds: Vec<LayerKind>,
    pub looped: Option<LoopSpec>,
    pub eos: Option<u32>,
    pub vision: Option<VisionConfig>,
}

fn positive(field: &str, v: u32) -> Result<u32, ConfigError> {
    if v == 0 {
        return Err(ConfigError::invalid(field, "must be positive"));
    }
    Ok(v)
}

fn finite_positive(field: &str, v: f64) -> Result<f32, ConfigError> {
    if !(v.is_finite() && v > 0.0) {
        return Err(ConfigError::invalid(field, format!("{v} must be finite and positive")));
    }
    Ok(v as f32)
}

impl Inkling2Config {
    /// Parse and validate a `config.json`.
    pub fn from_json(text: &str) -> Result<Inkling2Config, ConfigError> {
        let raw: CheckpointConfig = serde_json::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        Inkling2Config::from_checkpoint(&raw)
    }

    /// Read `<dir>/config.json`.
    pub fn from_dir(dir: &std::path::Path) -> Result<Inkling2Config, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Inkling2Config::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Validate a parsed `config.json`.
    pub fn from_checkpoint(raw: &CheckpointConfig) -> Result<Inkling2Config, ConfigError> {
        let t = &raw.text_config;
        let family = match (raw.model_type.as_str(), t.model_type.as_str()) {
            ("inkling2_loop", "inkling2_loop_text") => Family::Looped,
            ("inkling2", "inkling2_text") => Family::Plain,
            (m, tm) => return Err(ConfigError::invalid("model_type", format!("{m:?} with text_config.model_type {tm:?} is not an Inkling 2 checkpoint"))),
        };
        let class = match family {
            Family::Looped => "Inkling2LoopForConditionalGeneration",
            Family::Plain => "Inkling2ForConditionalGeneration",
        };
        if raw.architectures.as_slice() != [class] {
            return Err(ConfigError::invalid("architectures", format!("{:?}, a {:?} checkpoint declares [{class:?}]", raw.architectures, raw.model_type)));
        }
        for (field, on, want) in [
            ("text_config.q_bias", t.q_bias, false),
            ("text_config.o_bias", t.o_bias, false),
            ("text_config.use_value_norm_no_gamma", t.use_value_norm_no_gamma, true),
            ("text_config.attn_res_normalize_values", t.attn_res_normalize_values, true),
            ("text_config.use_gate_bias", t.use_gate_bias, true),
        ] {
            if on != want {
                return Err(ConfigError::invalid(field, format!("{on} is not the published recipe ({want}); no reference defines that variant")));
            }
        }
        if t.n_shared_experts != 0 {
            return Err(ConfigError::invalid("text_config.n_shared_experts", format!("{}: the recipe has no shared expert", t.n_shared_experts)));
        }
        if let Some(f) = &t.scoring_func {
            if f != "sigmoid" {
                return Err(ConfigError::invalid("text_config.scoring_func", format!("{f:?}: the router scores with a sigmoid")));
            }
        }

        let hidden = positive("text_config.hidden_size", t.hidden_size)?;
        let n_layers = positive("text_config.num_hidden_layers", t.num_hidden_layers)?;
        let vocab = positive("text_config.vocab_size", t.vocab_size)?;
        let valid_vocab = positive("text_config.unpadded_vocab_size", t.unpadded_vocab_size)?;
        if valid_vocab > vocab {
            return Err(ConfigError::invalid("text_config.unpadded_vocab_size", format!("{valid_vocab} exceeds vocab_size {vocab}")));
        }
        let attn = |prefix: &str, heads: u32, kv: u32, hd: u32| -> Result<AttnShape, ConfigError> {
            let heads = positive(&format!("text_config.{prefix}num_attention_heads"), heads)?;
            let kv_heads = positive(&format!("text_config.{prefix}num_key_value_heads"), kv)?;
            let head_dim = positive(&format!("text_config.{prefix}head_dim"), hd)?;
            if !heads.is_multiple_of(kv_heads) {
                return Err(ConfigError::invalid(&format!("text_config.{prefix}num_key_value_heads"), format!("{kv_heads} does not divide {heads} query heads")));
            }
            Ok(AttnShape { heads, kv_heads, head_dim })
        };
        let full = attn("", t.num_attention_heads, t.num_key_value_heads, t.head_dim)?;
        let sliding = attn("swa_", t.swa_num_attention_heads, t.swa_num_key_value_heads, t.swa_head_dim)?;
        let conv_taps = positive("text_config.sconv_kernel_size", t.sconv_kernel_size)?;
        let block = positive("text_config.n_layers_per_attn_res_block", t.n_layers_per_attn_res_block)?;
        let experts = positive("text_config.n_routed_experts", t.n_routed_experts)?;
        let top_k = positive("text_config.num_experts_per_tok", t.num_experts_per_tok)?;
        if top_k > experts {
            return Err(ConfigError::invalid("text_config.num_experts_per_tok", format!("{top_k} exceeds the {experts} experts")));
        }

        let kinds = layer_kinds(t, n_layers)?;
        let looped = match (family, &t.loop_config) {
            (Family::Looped, Some(l)) => Some(loop_spec(l, n_layers, block)?),
            (Family::Looped, None) => return Err(ConfigError::invalid("text_config.loop_config", "an inkling2_loop checkpoint declares its loop")),
            (Family::Plain, Some(_)) => return Err(ConfigError::invalid("text_config.loop_config", "an inkling2 (not looped) checkpoint has no loop")),
            (Family::Plain, None) => None,
        };
        if let (Some(a), Some(b)) = (raw.eos_token_id, t.eos_token_id) {
            if a != b {
                return Err(ConfigError::invalid("text_config.eos_token_id", format!("{b} contradicts the top-level {a}")));
            }
        }
        let eos = raw.eos_token_id.or(t.eos_token_id);
        if let Some(e) = eos {
            if e >= valid_vocab {
                return Err(ConfigError::invalid("eos_token_id", format!("{e} is outside the {valid_vocab} valid tokens")));
            }
        }
        if let Some(v) = &raw.vision_config {
            validate_vision(v, hidden)?;
        }
        Ok(Inkling2Config {
            family,
            hidden,
            n_layers,
            vocab,
            valid_vocab,
            max_positions: t.model_max_length,
            full,
            sliding,
            window: positive("text_config.sliding_window_size", t.sliding_window_size)?,
            d_rel: positive("text_config.d_rel", t.d_rel)?,
            rel_extent: positive("text_config.rel_extent", t.rel_extent)?,
            log_floor: positive("text_config.log_scaling_n_floor", t.log_scaling_n_floor)?,
            log_alpha: {
                if !(t.log_scaling_alpha.is_finite() && t.log_scaling_alpha >= 0.0) {
                    return Err(ConfigError::invalid("text_config.log_scaling_alpha", format!("{} must be finite and not negative", t.log_scaling_alpha)));
                }
                t.log_scaling_alpha as f32
            },
            eps: finite_positive("text_config.rms_norm_eps", t.rms_norm_eps)?,
            qk_clip: finite_positive("text_config.qk_l2_clip_threshold", t.qk_l2_clip_threshold)?,
            rel_clip: finite_positive("text_config.rel_l2_clip_threshold", t.rel_l2_clip_threshold)?,
            conv_taps,
            logit_cap: finite_positive("text_config.final_logit_softcapping", t.final_logit_softcapping)?,
            block,
            experts,
            top_k,
            moe_ff: positive("text_config.intermediate_size", t.intermediate_size)?,
            gate_cap: finite_positive("text_config.mlp_gate_softcap", t.mlp_gate_softcap)?,
            up_cap: finite_positive("text_config.mlp_up_softcap", t.mlp_up_softcap)?,
            route_scale: finite_positive("text_config.route_scale", t.route_scale)?,
            kinds,
            looped,
            eos,
            vision: raw.vision_config.clone(),
        })
    }

    pub fn kind(&self, layer: u32) -> LayerKind {
        self.kinds[layer as usize]
    }

    /// The attention geometry of `layer`.
    pub fn attn(&self, layer: u32) -> AttnShape {
        match self.kind(layer) {
            LayerKind::Sliding => self.sliding,
            LayerKind::Full => self.full,
        }
    }

    /// Backward distance `layer`'s relative bias covers.
    pub fn bias_extent(&self, layer: u32) -> u32 {
        match self.kind(layer) {
            LayerKind::Sliding => self.window,
            LayerKind::Full => self.rel_extent,
        }
    }

    /// Whether `layer` opens a depth-attention block: its incoming partial sum is retained and its residual restarts.
    pub fn is_block_boundary(&self, layer: u32) -> bool {
        layer.is_multiple_of(self.block)
    }

    /// Whether `layer` belongs to the looped core (and so carries message gates).
    pub fn is_core(&self, layer: u32) -> bool {
        self.looped.is_some_and(|l| (l.start..l.end).contains(&layer))
    }

    /// Depth-attention queries of `layer`'s attention input: none at layer 0, three (q, k, v) on a full layer, else one.
    pub fn attn_queries(&self, layer: u32) -> u32 {
        match (layer, self.kind(layer)) {
            (0, _) => 0,
            (_, LayerKind::Full) => 3,
            (_, LayerKind::Sliding) => 1,
        }
    }

    /// Layers the forward visits per token (`n_layers` plus the extra passes over the core).
    pub fn visits(&self) -> u32 {
        self.n_layers + self.looped.map_or(0, |l| (l.passes - 1) * l.layers())
    }
}

fn layer_kinds(t: &TextConfig, n_layers: u32) -> Result<Vec<LayerKind>, ConfigError> {
    let mut kinds = vec![LayerKind::Full; n_layers as usize];
    for (i, &l) in t.local_layer_ids.iter().enumerate() {
        if l >= n_layers {
            return Err(ConfigError::invalid("text_config.local_layer_ids", format!("layer {l} does not exist ({n_layers} layers)")));
        }
        if t.local_layer_ids[..i].contains(&l) {
            return Err(ConfigError::invalid("text_config.local_layer_ids", format!("layer {l} is listed twice")));
        }
        kinds[l as usize] = LayerKind::Sliding;
    }
    match &t.layer_types {
        Some(types) => {
            if types.len() != n_layers as usize {
                return Err(ConfigError::invalid("text_config.layer_types", format!("{} entries for {n_layers} layers", types.len())));
            }
            for (i, ty) in types.iter().enumerate() {
                let declared = match ty.as_str() {
                    "hybrid_sliding" => LayerKind::Sliding,
                    "hybrid" => LayerKind::Full,
                    other => return Err(ConfigError::invalid("text_config.layer_types", format!("layer {i}: unknown kind {other:?}"))),
                };
                if declared != kinds[i] {
                    return Err(ConfigError::invalid("text_config.layer_types", format!("layer {i} is {ty:?} but local_layer_ids says {:?}", kinds[i])));
                }
            }
        }
        None => {
            // Without `layer_types` the reference places a full layer at every `(i + 1) % 6 == 0`, whatever
            // `local_layer_ids` says; the two must agree or the checkpoint means two different models.
            for (i, kind) in kinds.iter().enumerate() {
                let reference = if (i + 1) % 6 == 0 { LayerKind::Full } else { LayerKind::Sliding };
                if *kind != reference {
                    return Err(ConfigError::invalid(
                        "text_config.local_layer_ids",
                        format!("layer {i} is {kind:?} but the reference (no layer_types) makes it {reference:?}; declare layer_types"),
                    ));
                }
            }
        }
    }
    Ok(kinds)
}

fn loop_spec(l: &LoopConfigJson, n_layers: u32, block: u32) -> Result<LoopSpec, ConfigError> {
    let field = |f: &str| format!("text_config.loop_config.{f}");
    if l.is_varloop {
        return Err(ConfigError::invalid(&field("is_varloop"), "a variable pass count has no reference; the published loop is fixed"));
    }
    if l.num_passes == 0 {
        return Err(ConfigError::invalid(&field("num_passes"), "must be at least 1"));
    }
    if l.core_start == 0 || !l.core_start.is_multiple_of(block) {
        return Err(ConfigError::invalid(&field("core_start"), format!("{} must be a positive multiple of the block size {block}", l.core_start)));
    }
    if l.core_end <= l.core_start || l.core_end > n_layers || !l.core_end.is_multiple_of(block) {
        return Err(ConfigError::invalid(&field("core_end"), format!("{} must be a multiple of the block size {block} in {}..={n_layers}", l.core_end, l.core_start + 1)));
    }
    Ok(LoopSpec { start: l.core_start, end: l.core_end, passes: l.num_passes, gate_rank: positive(&field("gate_rank"), l.gate_rank)? })
}

fn validate_vision(v: &VisionConfig, hidden: u32) -> Result<(), ConfigError> {
    if v.model_type != "inkling2_vision" {
        return Err(ConfigError::invalid("vision_config.model_type", format!("{:?} is not inkling2_vision", v.model_type)));
    }
    if v.text_hidden_size != hidden {
        return Err(ConfigError::invalid("vision_config.text_hidden_size", format!("{} must equal the decoder's hidden size {hidden}", v.text_hidden_size)));
    }
    for (f, x) in [
        ("hidden_size", v.hidden_size),
        ("num_hidden_layers", v.num_hidden_layers),
        ("num_attention_heads", v.num_attention_heads),
        ("native_patch_size", v.native_patch_size),
        ("adapter_merge_size", v.adapter_merge_size),
        ("projection_hidden_dim", v.projection_hidden_dim),
        ("posemb_grid_size", v.posemb_grid_size),
    ] {
        positive(&format!("vision_config.{f}"), x)?;
    }
    if v.projector_bias {
        return Err(ConfigError::invalid("vision_config.projector_bias", "the published projector has no bias"));
    }
    if !v.hidden_size.is_multiple_of(v.num_attention_heads) {
        return Err(ConfigError::invalid("vision_config.num_attention_heads", format!("{} does not divide the width {}", v.num_attention_heads, v.hidden_size)));
    }
    let ff = v.hidden_size as f64 * v.mlp_ratio;
    if !(ff.is_finite() && ff >= 1.0 && ff.fract() == 0.0) {
        return Err(ConfigError::invalid("vision_config.mlp_ratio", format!("{} times the width {} is not a whole MLP width", v.mlp_ratio, v.hidden_size)));
    }
    finite_positive("vision_config.layer_norm_eps", v.layer_norm_eps)?;
    Ok(())
}

impl VisionConfig {
    /// The MLP width (`hidden_size * mlp_ratio`, validated whole).
    pub fn mlp_width(&self) -> u32 {
        (self.hidden_size as f64 * self.mlp_ratio) as u32
    }
}
