// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FinCast configuration and parameter list.
//!
//! The param NAMES here are the reference checkpoint's own `state_dict` keys
//! (e.g. `stacked_transformer.layers.0.self_attn.qkv_proj.weight`), not a
//! brain-flavoured renaming — deliberately, so weight import is a 1:1 name match
//! against `Vincent05R/FinCast` `v1.pth` (after the `torch.compile`/DDP prefix
//! strip in `tools/fincast_convert.py`) and the T0 layout gate is a mechanical
//! diff rather than a hand-maintained translation table.
//!
//! Architecture (extracted from the reference source + verified against the real
//! `v1.pth` header, 991.4M params, native fp32):
//! - A **TimesFM-style patched decoder** (`src/ffm/pytorch_patched_decoder_MOE.py`):
//!   patch the context into `patch_len`-wide windows, standardize, embed each
//!   `[values, mask]` (`2*patch_len`) window through a `ResidualBlock`
//!   (`input_ff_layer`), add a learned frequency embedding, run `num_layers`
//!   decoder blocks, then a `ResidualBlock` head (`horizon_ff_layer`) producing
//!   `horizon_len * (1 + num_quantiles)` outputs = mean + 9 quantiles per step.
//! - Each decoder block: pre-`RMSNorm` (`input_layernorm`) → **TimesFM attention**
//!   (fused `qkv_proj`, a learned per-dim `scaling` with softplus, causal mask,
//!   `o_proj`) with residual → a **SparseMoEBlock** (`st_moe_pytorch`): st-MoE
//!   `RMSNorm` prenorm (`moe_prenorm.gamma`) → top-2 gating over `num_experts`
//!   experts (each expert = `LayerNorm → gate_proj → ReLU → down_proj` + its own
//!   residual) → residual. No `ff_before`/`ff_after`.
//!
//! Parity trap: the reference's MoE uses **stochastic** threshold gating +
//! capacity dropping *even at eval* (`st_moe_pytorch.TopNGating.forward` draws
//! `uniform()` and compares to `gate/threshold`). brain implements the
//! deterministic top-2 expectation (always route to the top-2, gates
//! renormalized to sum 1, no capacity dropping) — the natural clean-room reading;
//! the reference is made deterministic in the parity dump by driving
//! `threshold_eval→0` and `capacity_factor_eval→∞`. See `docs/models/fincast/status.md`.

/// A named parameter with its shape (row-major, PyTorch `nn.Linear` `[out, in]`).
pub type Param = (String, Vec<usize>);

/// The 9 quantile levels FinCast emits natively (`create_quantiles()` in the
/// reference). The head emits `[mean, q0.1, …, q0.9]` per horizon step.
pub const QUANTILES: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// FinCast core + task configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct FincastConfig {
    // transformer core
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    // task
    pub patch_len: usize,
    pub horizon_len: usize,
    pub num_quantiles: usize,
    pub context_len: usize,
    pub pad_val: f32,
    pub tolerance: f32,
    pub use_positional_embedding: bool,
    // MoE
    pub num_experts: usize,
    pub gating_top_n: usize,
    pub threshold_eval: f32,
}

impl Default for FincastConfig {
    /// The published `Vincent05R/FinCast` `v1.pth` (~991.4M params).
    ///
    /// Verified against the real checkpoint header: `num_layers=50`,
    /// `hidden=1280`, `head_dim=80`, `num_heads=16` (`inner = 16*80 = 1280`),
    /// `qkv_proj` out `(16 + 2*16)*80 = 3840`, **`num_experts=4`** (the reference
    /// `FFMConfig` default of 3 is stale — the shipped weights carry
    /// `experts.0..3` and a `[4,1280]` gate), `gating_top_n=2`.
    fn default() -> Self {
        FincastConfig {
            num_layers: 50,
            num_heads: 16,
            num_kv_heads: 16,
            hidden_size: 1280,
            intermediate_size: 1280,
            head_dim: 80,
            rms_norm_eps: 1e-6,
            patch_len: 32,
            horizon_len: 128,
            num_quantiles: 9,
            context_len: 512,
            pad_val: 1123581321.0,
            tolerance: 1e-6,
            use_positional_embedding: false,
            num_experts: 4,
            gating_top_n: 2,
            threshold_eval: 0.2,
        }
    }
}

impl FincastConfig {
    /// A tiny config for gradient-checking and fast unit tests (same op
    /// structure, small dims). `inner_dim == hidden_size` is preserved and
    /// `num_quantiles` stays 9 (the head geometry the forecaster assumes). Keeps
    /// 3 experts with top-2 routing so the MoE routing path is exercised.
    pub fn tiny() -> Self {
        FincastConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            hidden_size: 16,
            intermediate_size: 16,
            head_dim: 4,
            patch_len: 4,
            horizon_len: 4,
            num_quantiles: 9,
            context_len: 32,
            num_experts: 3,
            gating_top_n: 2,
            ..FincastConfig::default()
        }
    }

    /// Inner attention dim = `num_heads * head_dim`.
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// The fused `qkv_proj` output width: `(num_heads + 2*num_kv_heads)*head_dim`.
    pub fn qkv_dim(&self) -> usize {
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
    }

    /// The per-patch input feature width: `[values, mask]` each `patch_len` wide.
    pub fn patch_feat_dim(&self) -> usize {
        2 * self.patch_len
    }

    /// Number of head outputs per horizon step: `1 (mean) + num_quantiles`.
    pub fn num_outputs(&self) -> usize {
        1 + self.num_quantiles
    }

    /// The horizon head output width: `horizon_len * (1 + num_quantiles)`.
    pub fn head_out_dim(&self) -> usize {
        self.horizon_len * self.num_outputs()
    }

    /// Parse from a brain container header (produced by [`Self::to_json`]).
    pub fn from_json(v: &serde_json::Value) -> Result<FincastConfig, String> {
        let u = |k: &str| -> Result<usize, String> {
            v[k].as_u64().map(|n| n as usize).ok_or_else(|| format!("missing/invalid {k}"))
        };
        let f = |k: &str, d: f32| v[k].as_f64().map(|n| n as f32).unwrap_or(d);
        let b = |k: &str, d: bool| v[k].as_bool().unwrap_or(d);
        Ok(FincastConfig {
            num_layers: u("num_layers")?,
            num_heads: u("num_heads")?,
            num_kv_heads: u("num_kv_heads")?,
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            head_dim: u("head_dim")?,
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            patch_len: u("patch_len")?,
            horizon_len: u("horizon_len")?,
            num_quantiles: u("num_quantiles")?,
            context_len: u("context_len").unwrap_or(512),
            pad_val: f("pad_val", 1123581321.0),
            tolerance: f("tolerance", 1e-6),
            use_positional_embedding: b("use_positional_embedding", false),
            num_experts: u("num_experts")?,
            gating_top_n: u("gating_top_n").unwrap_or(2),
            threshold_eval: f("threshold_eval", 0.2),
        })
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "patch_len": self.patch_len,
            "horizon_len": self.horizon_len,
            "num_quantiles": self.num_quantiles,
            "quantiles": QUANTILES.to_vec(),
            "context_len": self.context_len,
            "pad_val": self.pad_val,
            "tolerance": self.tolerance,
            "use_positional_embedding": self.use_positional_embedding,
            "num_experts": self.num_experts,
            "gating_top_n": self.gating_top_n,
            "threshold_eval": self.threshold_eval,
        })
    }

    /// The full learnable parameter list, in the reference checkpoint's own key
    /// names and shapes. Device-free — diffable against the real safetensors
    /// header before a single kernel runs (the T0 layout gate). Registered
    /// buffers (`gate.threshold_{train,eval}`) are recomputed in code and are
    /// NOT listed here (see [`is_non_persistent`]).
    pub fn param_list(&self) -> Vec<Param> {
        let d = self.hidden_size;
        let f = self.intermediate_size;
        let feat = self.patch_feat_dim();
        let head = self.head_out_dim();
        let mut p: Vec<Param> = Vec::new();

        // input patch embedding: ResidualBlock over the [values, mask] window.
        // NOTE: `hidden_layer` is a Sequential(Linear, SiLU) -> the Linear is at
        // index `.0.` in the state_dict.
        residual_block(&mut p, "input_ff_layer", feat, f, d, true);

        // learned frequency embedding (3 buckets: high/med/low).
        p.push(("freq_emb.weight".into(), vec![3, d]));

        // decoder blocks
        for b in 0..self.num_layers {
            let pre = format!("stacked_transformer.layers.{b}");
            // pre-attention RMSNorm
            p.push((format!("{pre}.input_layernorm.weight"), vec![d]));
            // TimesFM attention: fused qkv, per-dim scaling, o_proj (all biased
            // except scaling which is a bare parameter vector).
            p.push((format!("{pre}.self_attn.scaling"), vec![self.head_dim]));
            p.push((format!("{pre}.self_attn.qkv_proj.weight"), vec![self.qkv_dim(), d]));
            p.push((format!("{pre}.self_attn.qkv_proj.bias"), vec![self.qkv_dim()]));
            p.push((format!("{pre}.self_attn.o_proj.weight"), vec![d, self.inner_dim()]));
            p.push((format!("{pre}.self_attn.o_proj.bias"), vec![d]));
            // SparseMoEBlock
            p.push((format!("{pre}.moe.moe_prenorm.gamma"), vec![d]));
            p.push((format!("{pre}.moe.moe.gate.to_gates.weight"), vec![self.num_experts, d]));
            for e in 0..self.num_experts {
                let ep = format!("{pre}.moe.moe.experts.experts.{e}");
                p.push((format!("{ep}.gate_proj.weight"), vec![d, d]));
                p.push((format!("{ep}.gate_proj.bias"), vec![d]));
                p.push((format!("{ep}.down_proj.weight"), vec![d, d]));
                p.push((format!("{ep}.down_proj.bias"), vec![d]));
                p.push((format!("{ep}.layer_norm.weight"), vec![d]));
                p.push((format!("{ep}.layer_norm.bias"), vec![d]));
            }
        }

        // horizon head: ResidualBlock hidden -> horizon_len*(1+num_quantiles)
        residual_block(&mut p, "horizon_ff_layer", d, f, head, true);

        p
    }

    /// Total learnable parameter count.
    pub fn param_count(&self) -> usize {
        self.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

/// Tensors the checkpoint header carries that are NOT learnable params
/// (recomputed in code): the gating threshold buffers. (`gate.zero` and
/// `experts.dummy` are `persistent=False` and never reach the state_dict.)
pub fn is_non_persistent(name: &str) -> bool {
    name.ends_with("gate.threshold_train") || name.ends_with("gate.threshold_eval")
}

/// A `ResidualBlock`: `out = output_layer(silu(hidden_layer(x))) + residual_layer(x)`.
/// All three linears have bias. In the reference `hidden_layer` is a
/// `Sequential(Linear, SiLU)`, so its Linear weight/bias are keyed `.0.` when
/// `seq_hidden` is set (both FinCast ResidualBlocks use the Sequential form).
fn residual_block(p: &mut Vec<Param>, prefix: &str, in_dim: usize, h: usize, out: usize, seq_hidden: bool) {
    let hid = if seq_hidden { format!("{prefix}.hidden_layer.0") } else { format!("{prefix}.hidden_layer") };
    p.push((format!("{hid}.weight"), vec![h, in_dim]));
    p.push((format!("{hid}.bias"), vec![h]));
    p.push((format!("{prefix}.output_layer.weight"), vec![out, h]));
    p.push((format!("{prefix}.output_layer.bias"), vec![out]));
    p.push((format!("{prefix}.residual_layer.weight"), vec![out, in_dim]));
    p.push((format!("{prefix}.residual_layer.bias"), vec![out]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn default_dims_match_the_published_checkpoint() {
        let c = FincastConfig::default();
        assert_eq!((c.num_layers, c.num_heads, c.hidden_size, c.head_dim), (50, 16, 1280, 80));
        assert_eq!(c.inner_dim(), c.hidden_size, "H*Dk must equal hidden for this checkpoint");
        assert_eq!(c.qkv_dim(), 3840);
        assert_eq!(c.patch_feat_dim(), 64);
        assert_eq!(c.num_experts, 4);
        assert_eq!(c.head_out_dim(), 128 * 10);
    }

    #[test]
    fn param_list_has_the_right_keys_and_no_dupes() {
        let c = FincastConfig::default();
        let pl = c.param_list();
        let keys: HashSet<&str> = pl.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), pl.len(), "duplicate param keys");
        assert!(keys.contains("input_ff_layer.hidden_layer.0.weight"));
        assert!(keys.contains("freq_emb.weight"));
        assert!(keys.contains("stacked_transformer.layers.0.self_attn.qkv_proj.weight"));
        assert!(keys.contains("stacked_transformer.layers.49.self_attn.o_proj.bias"));
        assert!(keys.contains("stacked_transformer.layers.0.moe.moe_prenorm.gamma"));
        assert!(keys.contains("stacked_transformer.layers.0.moe.moe.gate.to_gates.weight"));
        assert!(keys.contains("stacked_transformer.layers.0.moe.moe.experts.experts.3.down_proj.weight"));
        assert!(keys.contains("horizon_ff_layer.output_layer.bias"));
    }

    #[test]
    fn param_list_count_matches_the_state_dict() {
        // The real v1.pth has 1713 tensors, of which 2 per layer are the
        // non-persistent gate threshold buffers -> 1713 - 2*50 = 1613 learnable.
        let c = FincastConfig::default();
        let pl = c.param_list();
        // per layer: input_layernorm(1) + attn(scaling,qkv w+b,o w+b = 5)
        //          + moe_prenorm(1) + to_gates(1) + 4 experts * 6 = 24  => 32
        // + input_ff(6) + freq_emb(1) + horizon_ff(6) = 13
        assert_eq!(pl.len(), 13 + c.num_layers * 32);
        assert_eq!(pl.len(), 1613);
    }

    #[test]
    fn total_param_count_is_about_991m() {
        let c = FincastConfig::default();
        let n = c.param_count();
        // 991_437_160 total in v1.pth minus the 100 non-persistent threshold
        // buffers (2 elems each, 2 per layer × 50 layers).
        assert_eq!(n, 991_436_960, "param count drifted: {n}");
    }

    #[test]
    fn json_roundtrips() {
        let c = FincastConfig::default();
        let back = FincastConfig::from_json(&c.to_json()).unwrap();
        assert_eq!(c, back);
        let t = FincastConfig::tiny();
        assert_eq!(t, FincastConfig::from_json(&t.to_json()).unwrap());
    }

    #[test]
    fn tiny_preserves_invariants() {
        let c = FincastConfig::tiny();
        assert_eq!(c.inner_dim(), c.hidden_size);
        let pl = c.param_list();
        assert_eq!(pl.len(), 13 + c.num_layers * (8 + c.num_experts * 6));
    }
}
