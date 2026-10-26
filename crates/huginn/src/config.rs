// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint-compatible layouts for
// recurrent-depth decoders for its clients. If your team needs expertise in
// importing a prelude/core/coda model's exact parameters then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Huginn (Raven) configuration and parameter layout.
//!
//! The released `huginn-0125` shape: 2 prelude blocks, 4 recurrent blocks,
//! 2 coda blocks, width 5280, 55 heads of width 96, SwiGLU width 17920,
//! RoPE base 50 000, tied embeddings, and a Q/K bias per block. The recurrent
//! core is applied `mean_recurrence` (32) times; the same 4 blocks' weights
//! are used on every step.

use serde_json::Value;

/// One of the three functional groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Prelude,
    Core,
    Coda,
}

impl Group {
    pub fn name(self) -> &'static str {
        match self {
            Group::Prelude => "prelude",
            Group::Core => "core",
            Group::Coda => "coda",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HuginnConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub d_ff: u32,
    pub n_prelude: u32,
    pub n_recurrent: u32,
    pub n_coda: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub tie_embeddings: bool,
    /// Every block adds separate Q and K bias vectors (`[2, 1, heads, dim]`).
    pub qk_bias: bool,
    /// Steps the core runs at inference when the caller does not choose.
    pub mean_recurrence: u32,
    /// Standard deviation of the reference's initial latent state
    /// (`init_values["std"]`); `like-init` draws from it.
    pub init_std: f32,
    /// The reference scales embeddings by this before the prelude. The
    /// reference derives it (`emb_scale` is `sqrt(n_embd)`) instead of reading
    /// a setting, so it is derived here too; `init_values["embed_scale"]` is
    /// descriptive and is not trusted.
    pub emb_scale: f32,
}

impl HuginnConfig {
    /// A tiny config that exercises all three groups, the adapter, the Q/K
    /// bias and the tied head.
    pub fn tiny() -> HuginnConfig {
        HuginnConfig {
            vocab: 31,
            block_size: 32,
            d_model: 16,
            n_heads: 4,
            head_dim: 4,
            d_ff: 32,
            n_prelude: 1,
            n_recurrent: 2,
            n_coda: 1,
            rope_theta: 50_000.0,
            norm_eps: 1e-6,
            tie_embeddings: true,
            qk_bias: true,
            mean_recurrence: 3,
            init_std: 0.02,
            emb_scale: 4.0,
        }
    }

    /// `tomg-group-umd/huginn-0125` as released.
    pub fn huginn_0125() -> HuginnConfig {
        HuginnConfig {
            vocab: 65536,
            block_size: 4096,
            d_model: 5280,
            n_heads: 55,
            head_dim: 96,
            d_ff: 17920,
            n_prelude: 2,
            n_recurrent: 4,
            n_coda: 2,
            rope_theta: 50_000.0,
            norm_eps: 1e-6,
            tie_embeddings: true,
            qk_bias: true,
            mean_recurrence: 32,
            init_std: 0.008703883,
            emb_scale: 72.663605,
        }
    }

    pub fn groups(&self) -> impl Iterator<Item = (Group, u32)> {
        [(Group::Prelude, self.n_prelude), (Group::Core, self.n_recurrent), (Group::Coda, self.n_coda)]
            .into_iter()
    }

    pub fn qkv_dim(&self) -> u32 {
        3 * self.d_model
    }

    /// HF `config.json` -> config. Every key a shape depends on is required.
    pub fn from_json(v: &Value) -> Result<HuginnConfig, String> {
        let u = |k: &str| v[k].as_u64().map(|x| x as u32).ok_or_else(|| format!("huginn config: {k}"));
        let cfg = HuginnConfig {
            vocab: u("vocab_size")?,
            block_size: u("block_size")?,
            d_model: u("n_embd")?,
            n_heads: u("n_heads")?,
            head_dim: u("head_dim").unwrap_or(u("n_embd")? / u("n_heads")?),
            d_ff: u("intermediate_size")?,
            n_prelude: u("n_layers_in_prelude")?,
            n_recurrent: u("n_layers_in_recurrent_block")?,
            n_coda: u("n_layers_in_coda")?,
            rope_theta: v["rope_base"].as_f64().ok_or("huginn config: rope_base")? as f32,
            norm_eps: v["norm_eps"].as_f64().ok_or("huginn config: norm_eps")? as f32,
            tie_embeddings: v["tie_embeddings"].as_bool().unwrap_or(true),
            qk_bias: v["qk_bias"].as_bool().unwrap_or(true),
            mean_recurrence: v["mean_recurrence"].as_u64().unwrap_or(32) as u32,
            init_std: v["init_values"]["std"].as_f64().unwrap_or(0.02) as f32,
            emb_scale: ((v["n_embd"].as_f64().unwrap_or(0.0)) as f32).sqrt(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_heads == 0 || self.n_heads * self.head_dim != self.d_model {
            return Err(format!("huginn config: n_heads*head_dim {} != n_embd {}", self.n_heads * self.head_dim, self.d_model));
        }
        for (name, v) in [
            ("vocab_size", self.vocab),
            ("block_size", self.block_size),
            ("n_layers_in_prelude", self.n_prelude),
            ("n_layers_in_recurrent_block", self.n_recurrent),
            ("n_layers_in_coda", self.n_coda),
        ] {
            if v == 0 {
                return Err(format!("huginn config: {name} is zero"));
            }
        }
        if !self.tie_embeddings {
            return Err("huginn config: an untied head is not part of the released layout".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "architectures": ["RavenForCausalLM"],
            "model_type": "huginn_raven",
            "vocab_size": self.vocab,
            "block_size": self.block_size,
            "n_embd": self.d_model,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.d_ff,
            "n_layers_in_prelude": self.n_prelude,
            "n_layers_in_recurrent_block": self.n_recurrent,
            "n_layers_in_coda": self.n_coda,
            "rope_base": self.rope_theta,
            "norm_eps": self.norm_eps,
            "tie_embeddings": self.tie_embeddings,
            "qk_bias": self.qk_bias,
            "mean_recurrence": self.mean_recurrence,
            "init_values": { "std": self.init_std, "embed_scale": self.emb_scale },
        })
    }

    /// `(name, numel)` in forward order. Every block carries four RMSNorms
    /// (sandwich normalization), a fused QKV projection, the output
    /// projection, the per-head Q/K bias, and a fused gate/up FC plus its
    /// projection. The head is tied to `tok.weight`.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let mut out = vec![("tok.weight".to_string(), v * d), ("ln_f.weight".to_string(), d), ("adapter.weight".to_string(), d * 2 * d)];
        for (group, n) in self.groups() {
            for i in 0..n {
                let p = |s: &str| format!("{}.{i}.{s}", group.name());
                out.push((p("ln1.weight"), d));
                out.push((p("ln2.weight"), d));
                out.push((p("attn.wqkv.weight"), self.qkv_dim() as usize * d));
                out.push((p("attn.wo.weight"), d * d));
                if self.qk_bias {
                    // The checkpoint stores one `[2, 1, heads, dim]` tensor;
                    // it is imported as the two halves the attention adds, so
                    // no dispatch has to bind a sub-range at a head-width
                    // offset (device buffer bindings require wider alignment).
                    out.push((p("attn.q_bias"), d));
                    out.push((p("attn.k_bias"), d));
                }
                out.push((p("ln3.weight"), d));
                out.push((p("ln4.weight"), d));
                out.push((p("mlp.fc.weight"), 2 * ff * d));
                out.push((p("mlp.proj.weight"), d * ff));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_round_trips() {
        let cfg = HuginnConfig::tiny();
        assert_eq!(HuginnConfig::from_json(&cfg.to_json()).unwrap(), cfg);
    }

    #[test]
    fn the_released_layout_matches_the_checkpoint_manifest() {
        // Facts from the checkpoint: 10 tensors per block (4 norms, fused QKV,
        // output projection, the two Q/K bias halves, fused FC, projection),
        // plus the embedding, the final norm and the injection adapter.
        // `lm_head` and `freqs_cis` are dropped: the head is tied and the
        // rotary table is rebuilt from `rope_base`.
        let cfg = HuginnConfig::huginn_0125();
        let list = cfg.param_list();
        assert_eq!(list.len(), 3 + 10 * 8);
        assert_eq!(list.iter().find(|(n, _)| n == "tok.weight").unwrap().1, 65536 * 5280);
        assert_eq!(list.iter().find(|(n, _)| n == "adapter.weight").unwrap().1, 5280 * 10560);
        assert_eq!(list.iter().find(|(n, _)| n == "prelude.0.attn.wqkv.weight").unwrap().1, 15840 * 5280);
        assert_eq!(list.iter().find(|(n, _)| n == "core.3.mlp.fc.weight").unwrap().1, 35840 * 5280);
        assert_eq!(list.iter().find(|(n, _)| n == "coda.1.attn.q_bias").unwrap().1, 5280);
        assert_eq!(list.iter().find(|(n, _)| n == "coda.1.attn.k_bias").unwrap().1, 5280);
    }

    #[test]
    fn a_width_that_is_not_heads_times_head_dim_is_refused() {
        let mut cfg = HuginnConfig::tiny();
        cfg.head_dim = 5;
        assert!(cfg.validate().unwrap_err().contains("n_heads*head_dim"));
    }
}
