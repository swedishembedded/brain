// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TimesFM-3 configuration and parameter list.
//!
//! Param NAMES here are the reference checkpoint's own `state_dict` keys
//! verbatim (e.g. `transformer_stack.layers.0.seq_attn.query_proj.weight`),
//! not a brain-flavoured renaming - the same discipline `fincast`/`chronos2`
//! follow, so weight import is a 1:1 name match against
//! `google/timesfm-3.0-pytorch`'s `model.safetensors` and the T0 layout gate
//! is a mechanical diff, not a hand-maintained translation table.
//!
//! Architecture (extracted from `google-research/timesfm`'s `src/timesfm3/`
//! and verified against the real checkpoint header - 445 tensors, 330,710,976
//! params, native fp32):
//! - A **stacked mixing transformer**: patch the context into
//!   `input_patch_len`-wide windows, run through a `ResidualBlock`
//!   (`pre_transformer_resblock`) embedding `[values(32) | next-2-patches(64) |
//!   mask(32) | next-2-patches mask(64)] = 192` features, `num_layers` mixing
//!   blocks, then a biased `Linear` head (`output_head`) producing
//!   `output_patch_len * num_quantiles` outputs per patch (9 quantiles, no
//!   separate mean channel - q0.5 IS the point forecast).
//! - Each mixing block is **sandwich-normed**: `h = post_ln(sublayer(pre_ln(x)))
//!   + x`, three sublayers - **sequence attention** (causal, RoPE, per-head
//!   RMSNorm QK-norm, a `+sqrt(head_dim)` PerDimScale-folded score scale - see
//!   `model.rs`'s doc for why this is NOT the usual `1/sqrt(d)`), **variate
//!   attention** (bidirectional, no RoPE, re-strided over the variate axis
//!   instead of the sequence axis - `use_rope_var=false` in the shipped
//!   config), and a plain ReLU feedforward with hidden width EQUAL to
//!   `model_dims` (not the usual 4x).
//!
//! RMSNorm epsilon is `f32::EPSILON` (`1.1920929e-7`), not the usual `1e-6` -
//! confirmed empirically against `torch.nn.RMSNorm`'s no-eps-given default.

/// A named parameter with its shape (row-major, PyTorch `nn.Linear` `[out, in]`).
pub type Param = (String, Vec<usize>);

/// The 9 quantile levels TimesFM-3 emits natively. `QUANTILES[4] == 0.5` is
/// the point forecast - there is no separate mean channel.
pub const QUANTILES: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// `torch.nn.RMSNorm`'s no-eps-given default (`torch.finfo(f32).eps`), which
/// every RMSNorm in this checkpoint uses. Distinct from the `1e-6` most other
/// ported architectures use - reproduce this exact value, not the common one.
pub const RMS_NORM_EPS: f32 = f32::EPSILON;

/// TimesFM-3 core + task configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Timesfm3Config {
    // transformer core
    pub num_layers: usize,
    pub model_dims: usize,
    /// Feedforward hidden width. Equal to `model_dims` in the published
    /// config - NOT a 4x expansion.
    pub hidden_dims: usize,
    pub num_heads: usize,
    /// `model_dims / num_heads`. Kept as an explicit field (rather than
    /// derived) because a tiny test config must be free to pick a head_dim
    /// that shares no factor with the other dims - see [`Self::tiny`].
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    // patching
    pub input_patch_len: usize,
    pub output_patch_len: usize,
    pub num_quantiles: usize,
    pub max_variates: usize,
    pub max_context: usize,
    // preprocessing/postprocessing toggles - all on in the published config,
    // kept explicit because the tiny config exercises every one of them.
    pub use_variate_attention: bool,
    pub use_stitching: bool,
    pub use_linear_detrending: bool,
    pub linear_detrending_threshold: f32,
    pub use_iterative_cpm_revin: bool,
    pub use_frozen_running_stats: bool,
    pub value_clip: f32,
}

impl Default for Timesfm3Config {
    /// The published `google/timesfm-3.0-pytorch` checkpoint.
    ///
    /// Verified against the real checkpoint header: `num_layers=20`,
    /// `model_dims=1280`, `hidden_dims=1280` (equal, not 4x), `num_heads=16`
    /// (`head_dim = 1280/16 = 80`), `input_patch_len=32`,
    /// `output_patch_len=64`, 9 quantiles, `max_variates=32`,
    /// `max_context=15360` (`ceil(15360/32)*32 == 15360`, so the "round up to
    /// a patch boundary" rule is a no-op at this exact number).
    fn default() -> Self {
        Timesfm3Config {
            num_layers: 20,
            model_dims: 1280,
            hidden_dims: 1280,
            num_heads: 16,
            head_dim: 80,
            rms_norm_eps: RMS_NORM_EPS,
            input_patch_len: 32,
            output_patch_len: 64,
            num_quantiles: 9,
            max_variates: 32,
            max_context: 15360,
            use_variate_attention: true,
            use_stitching: true,
            use_linear_detrending: true,
            linear_detrending_threshold: 0.5,
            use_iterative_cpm_revin: true,
            use_frozen_running_stats: false,
            value_clip: 1e20,
        }
    }
}

impl Timesfm3Config {
    /// A tiny config for gradient-checking and fast unit tests. Every
    /// dimension is pairwise distinct AND pairwise coprime where the real
    /// checkpoint's aren't (`1280 = 16 * 80`, so `model_dims`/`num_heads`/
    /// `head_dim` share factors there) - chosen to match exactly the
    /// `tools/goldens/timesfm3_dump_reference.py` `dump_tiny()` config, so
    /// the two are gated against each other directly, not just individually
    /// plausible. `head_dim` is kept EVEN (6): RoPE's split-half rotation
    /// divides it in two, so an odd head_dim is a shape error, not a valid
    /// tiny choice.
    pub fn tiny() -> Self {
        Timesfm3Config {
            num_layers: 3,
            model_dims: 12,
            hidden_dims: 14,
            num_heads: 2,
            head_dim: 6,
            input_patch_len: 4,
            output_patch_len: 8,
            num_quantiles: 5,
            max_variates: 9,
            max_context: 16,
            ..Timesfm3Config::default()
        }
    }

    /// The pre-transformer `ResidualBlock`'s input width: the current patch's
    /// values+mask (`2*input_patch_len`) concatenated with the next `rolls`
    /// patches' values+mask (`2*output_patch_len`).
    pub fn resblock_in_dim(&self) -> usize {
        2 * (self.input_patch_len + self.output_patch_len)
    }

    /// `output_patch_len / input_patch_len` - how many input patches one
    /// decode step's stitched output spans.
    pub fn rolls(&self) -> usize {
        self.output_patch_len / self.input_patch_len
    }

    /// The output head's width: `output_patch_len * num_quantiles`.
    pub fn head_out_dim(&self) -> usize {
        self.output_patch_len * self.num_quantiles
    }

    /// The stitching extraction length: `min(2*input_patch_len,
    /// output_patch_len)`.
    pub fn stitch_extract_len(&self) -> usize {
        (2 * self.input_patch_len).min(self.output_patch_len)
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Timesfm3Config, String> {
        let u = |k: &str| -> Result<usize, String> {
            v[k].as_u64().map(|n| n as usize).ok_or_else(|| format!("missing/invalid {k}"))
        };
        let f = |k: &str, d: f32| v[k].as_f64().map(|n| n as f32).unwrap_or(d);
        let b = |k: &str, d: bool| v[k].as_bool().unwrap_or(d);
        Ok(Timesfm3Config {
            num_layers: u("num_layers")?,
            model_dims: u("model_dims")?,
            hidden_dims: u("hidden_dims")?,
            num_heads: u("num_heads")?,
            head_dim: u("head_dim")?,
            rms_norm_eps: f("rms_norm_eps", RMS_NORM_EPS),
            input_patch_len: u("input_patch_len")?,
            output_patch_len: u("output_patch_len")?,
            num_quantiles: u("num_quantiles")?,
            max_variates: u("max_variates").unwrap_or(32),
            max_context: u("max_context").unwrap_or(15360),
            use_variate_attention: b("use_variate_attention", true),
            use_stitching: b("use_stitching", true),
            use_linear_detrending: b("use_linear_detrending", true),
            linear_detrending_threshold: f("linear_detrending_threshold", 0.5),
            use_iterative_cpm_revin: b("use_iterative_cpm_revin", true),
            use_frozen_running_stats: b("use_frozen_running_stats", false),
            value_clip: f("value_clip", 1e20),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "num_layers": self.num_layers,
            "model_dims": self.model_dims,
            "hidden_dims": self.hidden_dims,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "input_patch_len": self.input_patch_len,
            "output_patch_len": self.output_patch_len,
            "num_quantiles": self.num_quantiles,
            "quantiles": QUANTILES.to_vec(),
            "max_variates": self.max_variates,
            "max_context": self.max_context,
            "use_variate_attention": self.use_variate_attention,
            "use_stitching": self.use_stitching,
            "use_linear_detrending": self.use_linear_detrending,
            "linear_detrending_threshold": self.linear_detrending_threshold,
            "use_iterative_cpm_revin": self.use_iterative_cpm_revin,
            "use_frozen_running_stats": self.use_frozen_running_stats,
            "value_clip": self.value_clip,
        })
    }

    /// The full learnable parameter list, in the reference checkpoint's own
    /// key names and shapes. Device-free - diffable against the real
    /// safetensors header before a single kernel runs (the T0 layout gate).
    /// The real checkpoint has NO non-persistent buffers (RoPE/PerDimScale
    /// are computed, not registered) - every tensor the header names is
    /// listed here, and vice versa.
    pub fn param_list(&self) -> Vec<Param> {
        let d = self.model_dims;
        let hd = self.head_dim;
        let mut p: Vec<Param> = Vec::new();

        p.push(("pre_transformer_resblock.hidden_layer.weight".into(), vec![d, self.resblock_in_dim()]));
        p.push(("pre_transformer_resblock.output_layer.weight".into(), vec![d, d]));
        p.push(("pre_transformer_resblock.residual_layer.weight".into(), vec![d, self.resblock_in_dim()]));

        for l in 0..self.num_layers {
            let pre = format!("transformer_stack.layers.{l}");
            attention_sublayer(&mut p, &pre, "seq_attn", d, hd);
            attention_sublayer(&mut p, &pre, "var_attn", d, hd);
            p.push((format!("{pre}.pre_ff_ln.weight"), vec![d]));
            p.push((format!("{pre}.ff0.weight"), vec![self.hidden_dims, d]));
            p.push((format!("{pre}.ff1.weight"), vec![d, self.hidden_dims]));
            p.push((format!("{pre}.post_ff_ln.weight"), vec![d]));
        }

        p.push(("output_head.weight".into(), vec![self.head_out_dim(), d]));
        p.push(("output_head.bias".into(), vec![self.head_out_dim()]));

        p
    }

    /// Total learnable parameter count.
    pub fn param_count(&self) -> usize {
        self.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

/// One sandwich-normed attention sublayer (`seq_attn` or `var_attn`): a
/// pre-norm, four bias-free square projections, per-head RMSNorm on Q/K, a
/// learned per-head-dim scale, and a post-norm. No bias anywhere in this
/// sublayer (`use_bias=false` in the published config) and no V-norm
/// (`v_norm="none"`).
fn attention_sublayer(p: &mut Vec<Param>, pre: &str, name: &str, d: usize, head_dim: usize) {
    p.push((format!("{pre}.pre_{name}_ln.weight"), vec![d]));
    p.push((format!("{pre}.{name}.query_proj.weight"), vec![d, d]));
    p.push((format!("{pre}.{name}.key_proj.weight"), vec![d, d]));
    p.push((format!("{pre}.{name}.value_proj.weight"), vec![d, d]));
    p.push((format!("{pre}.{name}.out_proj.weight"), vec![d, d]));
    p.push((format!("{pre}.{name}.query_ln.weight"), vec![head_dim]));
    p.push((format!("{pre}.{name}.key_ln.weight"), vec![head_dim]));
    p.push((format!("{pre}.{name}.per_dim_scale.per_dim_scale"), vec![head_dim]));
    p.push((format!("{pre}.post_{name}_ln.weight"), vec![d]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn default_dims_match_the_published_checkpoint() {
        let c = Timesfm3Config::default();
        assert_eq!((c.num_layers, c.model_dims, c.num_heads, c.head_dim), (20, 1280, 16, 80));
        assert_eq!(c.num_heads * c.head_dim, c.model_dims);
        assert_eq!(c.resblock_in_dim(), 192);
        assert_eq!(c.rolls(), 2);
        assert_eq!(c.head_out_dim(), 64 * 9);
        assert_eq!(c.stitch_extract_len(), 64);
        assert_eq!(c.rms_norm_eps, f32::EPSILON);
    }

    #[test]
    fn param_list_has_the_right_keys_and_no_dupes() {
        let c = Timesfm3Config::default();
        let pl = c.param_list();
        let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), pl.len(), "duplicate param keys");
        assert!(keys.contains("pre_transformer_resblock.hidden_layer.weight"));
        assert!(keys.contains("transformer_stack.layers.0.seq_attn.query_proj.weight"));
        assert!(keys.contains("transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale"));
        assert!(keys.contains("transformer_stack.layers.19.var_attn.out_proj.weight"));
        assert!(keys.contains("transformer_stack.layers.0.ff0.weight"));
        assert!(keys.contains("output_head.bias"));
        // No V-norm, no bias anywhere but the output head.
        assert!(!keys.contains("transformer_stack.layers.0.seq_attn.value_ln.weight"));
        assert!(!keys.contains("transformer_stack.layers.0.seq_attn.query_proj.bias"));
    }

    #[test]
    fn param_list_count_and_total_match_the_real_checkpoint() {
        // The real model.safetensors has exactly 445 tensors, all learnable
        // (no non-persistent buffers) - confirmed by reading its own header.
        let c = Timesfm3Config::default();
        let pl = c.param_list();
        // 3 (resblock) + 20 * (9*2 attention sublayers + 4 ff) + 2 (head) = 445
        assert_eq!(pl.len(), 3 + c.num_layers * (9 * 2 + 4) + 2);
        assert_eq!(pl.len(), 445);
        assert_eq!(c.param_count(), 330_710_976);
    }

    #[test]
    fn json_roundtrips() {
        let c = Timesfm3Config::default();
        let back = Timesfm3Config::from_json(&c.to_json()).unwrap();
        assert_eq!(c, back);
        let t = Timesfm3Config::tiny();
        assert_eq!(t, Timesfm3Config::from_json(&t.to_json()).unwrap());
    }

    #[test]
    fn tiny_dims_are_pairwise_distinct_and_head_dim_is_even() {
        let c = Timesfm3Config::tiny();
        let dims = [c.num_layers, c.model_dims, c.hidden_dims, c.num_heads, c.head_dim, c.input_patch_len, c.output_patch_len, c.num_quantiles, c.max_variates];
        let set: HashSet<usize> = dims.iter().copied().collect();
        assert_eq!(set.len(), dims.len(), "tiny() dims must be pairwise distinct: {dims:?}");
        assert_eq!(c.head_dim % 2, 0, "RoPE splits head_dim in half");
        assert_eq!(c.num_heads * c.head_dim, c.model_dims);
        assert_eq!(c.output_patch_len % c.input_patch_len, 0);
        assert!(c.output_patch_len > c.input_patch_len, "use_stitching requires this");
    }
}
