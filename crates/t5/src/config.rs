// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T5 encoder topology + the canonical tensor manifest every import is checked
//! against.

/// One T5 encoder stack. The released FLUX.1 second text encoder is
/// [`T5Config::xxl`]; the fields are all the architecture needs, because T5 has
/// no RoPE, no absolute position embedding and no bias anywhere.
#[derive(Clone, Debug, PartialEq)]
pub struct T5Config {
    pub vocab: u32,
    pub d_model: u32,
    /// FFN width (`d_ff`). Both gate and up projections are `[d_ff, d_model]`.
    pub d_ff: u32,
    /// Per-head width (`d_kv`). NOTE `heads * d_kv` (the attention inner dim)
    /// is only *coincidentally* equal to `d_model` in T5-XXL; the code keeps
    /// them separate.
    pub d_kv: u32,
    pub layers: u32,
    pub heads: u32,
    /// `relative_attention_num_buckets` (32 for every released T5).
    pub rel_buckets: u32,
    /// `relative_attention_max_distance` (128 for every released T5).
    pub rel_max_distance: u32,
    /// `layer_norm_epsilon`. T5's norm is RMS (no mean subtraction, no bias).
    pub eps: f32,
}

impl T5Config {
    /// T5-XXL v1.1 encoder — `FLUX.1-*/text_encoder_2/config.json`
    /// (4.762 B parameters).
    pub fn xxl() -> T5Config {
        T5Config {
            vocab: 32128,
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            layers: 24,
            heads: 64,
            rel_buckets: 32,
            rel_max_distance: 128,
            eps: 1e-6,
        }
    }

    /// Attention inner dimension, `heads * d_kv`.
    pub fn inner(&self) -> u32 {
        self.heads * self.d_kv
    }

    /// Canonical parameter names + shapes. The single source of truth for
    /// [`crate::import`]'s two-way coverage check and for the `ParamStore`
    /// layout, so a name can only be wrong in one place.
    ///
    /// q/k/v are **fused** into one `[3*inner, d_model]` row-concatenation
    /// (q‖k‖v) at import time: the attention kernels read q, k and v from one
    /// buffer at `q_off`/`k_off`/`v_off` within a `3*inner` row, so fusing the
    /// weight makes the projection a single GEMM straight into that layout.
    /// The concatenation is bit-exact — each output row is the same dot
    /// product either way.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (d, ff, inner) = (self.d_model as usize, self.d_ff as usize, self.inner() as usize);
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("shared.weight".into(), vec![self.vocab as usize, d]),
            // The learned relative-position bias table: one row per bucket, one
            // column per head. It lives on block 0 in the checkpoint and is
            // shared by every block.
            ("rel_bias.weight".into(), vec![self.rel_buckets as usize, self.heads as usize]),
        ];
        for l in 0..self.layers {
            let p = format!("blocks.{l}");
            v.push((format!("{p}.attn_norm.weight"), vec![d]));
            v.push((format!("{p}.qkv.weight"), vec![3 * inner, d]));
            v.push((format!("{p}.o.weight"), vec![d, inner]));
            v.push((format!("{p}.ff_norm.weight"), vec![d]));
            v.push((format!("{p}.wi_0.weight"), vec![ff, d]));
            v.push((format!("{p}.wi_1.weight"), vec![ff, d]));
            v.push((format!("{p}.wo.weight"), vec![d, ff]));
        }
        v.push(("final_norm.weight".into(), vec![d]));
        v
    }

    /// Total parameter count of the manifest (fused q/k/v does not change it).
    pub fn param_count(&self) -> usize {
        self.tensor_manifest().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxl_manifest_shape_and_size() {
        let c = T5Config::xxl();
        let m = c.tensor_manifest();
        // 2 globals + 7 per block + final norm.
        assert_eq!(m.len(), 2 + 7 * 24 + 1);
        assert_eq!(c.inner(), 4096);
        // The checkpoint's 219 tensors fuse to 171 (24 blocks lose 2 each as
        // q/k/v become one qkv).
        assert_eq!(m.len(), 171);
        // 4.762 B — matches the reference dump ("loaded: 4.762 B params").
        let p = c.param_count();
        assert!((4.760e9..4.765e9).contains(&(p as f64)), "param count {p}");
    }
}
