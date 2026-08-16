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
    /// **Topology.** `true` = one `T5RelativeEmbedding` per block (umT5's
    /// `shared_pos=False`); `false` = one table on block 0, shared by every
    /// block (T5 v1.1's `shared_pos=True`). It changes
    /// [`T5Config::tensor_manifest`], the importer and the forward, and getting
    /// it wrong is SILENT - the wrong bias still produces plausible embeddings.
    pub per_block_rel_bias: bool,
    /// **Call contract**, not topology: `true` = the encoder is driven with a
    /// `[B, T]` key-padding mask, so right-pad positions are removed from every
    /// query's attention. FLUX passes no mask and needs `false`; Wan passes one
    /// and needs `true` (the reference dump measures 1.5 max|d| on the CONTENT
    /// rows between the two runs, so this is a real feature, not rounding).
    pub masked: bool,
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
            per_block_rel_bias: false,
            masked: false,
        }
    }

    /// **umT5-XXL encoder** - Wan2.1's text tower
    /// (`Wan2.1/wan/modules/t5.py::umt5_xxl`, 5.681 B parameters).
    ///
    /// Same block topology and the same widths as [`T5Config::xxl`], and it is
    /// tempting to treat it as "T5-XXL with a bigger vocabulary". It is three
    /// different models:
    ///
    /// * `vocab_size=256384` instead of 32128 - multilingual, and on its own
    ///   that is **+3.67 GB** of fp32 embedding table (0.53 -> 4.20 GB);
    /// * `shared_pos=False` - 24 independent relative-position tables;
    /// * it is called WITH an attention mask and a 512-token pad.
    pub fn umt5_xxl() -> T5Config {
        T5Config {
            vocab: 256384,
            per_block_rel_bias: true,
            masked: true,
            ..T5Config::xxl()
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
        let bias = vec![self.rel_buckets as usize, self.heads as usize];
        let mut v: Vec<(String, Vec<usize>)> =
            vec![("shared.weight".into(), vec![self.vocab as usize, d])];
        if !self.per_block_rel_bias {
            // The learned relative-position bias table: one row per bucket, one
            // column per head. It lives on block 0 in the checkpoint and is
            // shared by every block.
            v.push(("rel_bias.weight".into(), bias.clone()));
        }
        for l in 0..self.layers {
            let p = format!("blocks.{l}");
            if self.per_block_rel_bias {
                v.push((format!("{p}.rel_bias.weight"), bias.clone()));
            }
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

    /// The relative-position bias table block `l` reads: its own when
    /// `per_block_rel_bias`, otherwise the single shared one.
    pub fn rel_bias_name(&self, l: usize) -> String {
        if self.per_block_rel_bias {
            format!("blocks.{l}.rel_bias.weight")
        } else {
            "rel_bias.weight".to_string()
        }
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

    #[test]
    fn umt5_manifest_shape_and_size() {
        let c = T5Config::umt5_xxl();
        let m = c.tensor_manifest();
        // 1 global + 8 per block + final norm: the shared bias becomes 24
        // per-block ones, so the count goes UP by 23 against T5 v1.1.
        assert_eq!(m.len(), 1 + 8 * 24 + 1);
        assert_eq!(m.len(), 194);
        // The checkpoint's 242 tensors fuse to 194 (24 blocks lose 2 each as
        // q/k/v become one qkv).
        assert_eq!(m.len() + 2 * 24, 242);
        assert_eq!(c.rel_bias_name(7), "blocks.7.rel_bias.weight");
        assert_eq!(T5Config::xxl().rel_bias_name(7), "rel_bias.weight");
        // 5.681 B - matches the reference dump ("5.681 B params").
        let p = c.param_count();
        assert!((5.680e9..5.682e9).contains(&(p as f64)), "param count {p}");
        // The whole delta against T5-XXL is the embedding table: 224256 extra
        // rows of 4096 = 918 M parameters = 3.67 GB in fp32.
        let d = p - T5Config::xxl().param_count();
        assert_eq!(d, (256384 - 32128) * 4096 + 23 * 32 * 64);
        assert!((3.66e9..3.68e9).contains(&(d as f64 * 4.0)), "fp32 delta {} GB", d * 4);
    }
}
