// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PuLID-FLUX configuration, the canonical tensor manifest, and the **injection
//! schedule** — the three facts the port had to verify against the reference
//! rather than assume.
//!
//! Verified against `PuLID/pulid/encoders_transformer.py` (the two modules),
//! `PuLID/pulid/pipeline_flux.py` (how many cross-attentions are built and with
//! what intervals) and `PuLID/flux/model.py` (where they fire), at the
//! `pulid_flux_v0.9.1` checkpoint:
//!
//! 1. **Where the ID attention is inserted.** After double-stream block `i` when
//!    `i % 2 == 0`, and after single-stream block `i` when `i % 4 == 0` — so
//!    10 + 10 = 20 sites at FLUX.1's 19 + 38 depth, matching the 20
//!    `PerceiverAttentionCA` modules the checkpoint ships. `ca_idx` is ONE
//!    sequential counter shared by both loops (doubles take 0..10, singles
//!    10..20). See [`PulidConfig::schedule`].
//! 2. **Whether the ID embedding is projected/normalised before injection.**
//!    Projected, yes — `IDFormer` ends with `latents[:, :32] @ proj_out`
//!    (1024 → 2048). Normalised, no: there is no norm on the encoder output.
//!    Each cross-attention module applies *its own* `norm1` LayerNorm to the ID
//!    tokens and `norm2` to the image tokens, so the normalisation is per-site
//!    and part of the attention, not part of the embedding.
//!    (Two normalisations *do* happen upstream, inside the ID **condition**:
//!    the EVA-CLIP cls embedding is L2-normalised before the concat, the
//!    ArcFace embedding is not.)
//! 3. **How the ID contribution is scaled and combined.**
//!    `img = img + id_weight * ca(id, img)` — **added to the image residual
//!    stream**, never concatenated as tokens; the image rows are the attention
//!    QUERIES and the ID tokens the KEYS/VALUES. `id_weight` is a plain scalar
//!    (the reference's UI slider, default 1.0, range 0..3). The **start-step**
//!    schedule lives in the sampler, not the model: `flux/sampling.py` passes
//!    `id=id if i >= start_step else None`, i.e. before `start_step` the
//!    injection is simply absent. brain expresses that the same way — a caller
//!    that wants step `i` unconditioned calls the plain
//!    `Flux1Model::forward`. `crates/flux1` has no sampler loop, so no
//!    schedule is implemented here.

/// Which stream a cross-attention site sits after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Double,
    Single,
}

/// One cross-attention site: `(stream, block index, ca module index)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub stream: Stream,
    /// Block index within its stream.
    pub block: usize,
    /// Index into the 20 `PerceiverAttentionCA` modules.
    pub ca: usize,
}

/// PuLID-FLUX v0.9.1 hyperparameters.
///
/// Field defaults are the reference's constructor defaults; nothing here is
/// read from the checkpoint, so [`PulidConfig::tensor_manifest`] is an
/// independent statement of what the checkpoint must contain.
#[derive(Clone, Debug, PartialEq)]
pub struct PulidConfig {
    // ---- IDFormer (`pulid_encoder`) ----
    /// IDFormer working width.
    pub dim: usize,
    /// Total IDFormer layers; `depth / 5` of them run per ViT scale.
    pub depth: usize,
    pub dim_head: usize,
    pub heads: usize,
    /// ID tokens produced from the ArcFace+EVA condition vector.
    pub num_id_token: usize,
    /// Learned query latents; only these survive `proj_out`.
    pub num_queries: usize,
    /// `proj_out` output width = the cross-attention `kv_dim`.
    pub output_dim: usize,
    /// FeedForward inner width multiplier.
    pub ff_mult: usize,
    /// `id_cond` width: ArcFace 512 ‖ EVA-CLIP 768.
    pub id_cond_dim: usize,
    /// Tokens per tapped EVA-CLIP hidden state (cls + 24×24 patches).
    pub vit_tokens: usize,
    /// Tapped EVA-CLIP scales (= `depth / (depth/5)`).
    pub scales: usize,

    // ---- PerceiverAttentionCA (`pulid_ca`) ----
    /// Query width = the FLUX.1 hidden size.
    pub ca_dim: usize,
    pub ca_dim_head: usize,
    pub ca_heads: usize,

    // ---- injection schedule ----
    pub double_interval: usize,
    pub single_interval: usize,

    /// `nn.LayerNorm` default epsilon.
    pub eps: f32,
    /// `nn.LeakyReLU` default negative slope.
    pub leaky_slope: f32,
}

impl Default for PulidConfig {
    fn default() -> PulidConfig {
        PulidConfig::v0_9_1()
    }
}

impl PulidConfig {
    /// `pulid_flux_v0.9.1.safetensors`.
    pub fn v0_9_1() -> PulidConfig {
        PulidConfig {
            dim: 1024,
            depth: 10,
            dim_head: 64,
            heads: 16,
            num_id_token: 5,
            num_queries: 32,
            output_dim: 2048,
            ff_mult: 4,
            id_cond_dim: 1280,
            vit_tokens: 577,
            scales: 5,
            ca_dim: 3072,
            ca_dim_head: 128,
            ca_heads: 16,
            double_interval: 2,
            single_interval: 4,
            eps: 1e-5,
            leaky_slope: 0.01,
        }
    }

    /// IDFormer layers per tapped ViT scale (`depth // 5`).
    pub fn layers_per_scale(&self) -> usize {
        self.depth / self.scales
    }
    /// `heads * dim_head` — the IDFormer attention inner width.
    pub fn inner_dim(&self) -> usize {
        self.heads * self.dim_head
    }
    /// `ca_heads * ca_dim_head` — the cross-attention inner width. Equals
    /// [`Self::output_dim`], which is why `to_kv` reads the ID tokens directly.
    pub fn ca_inner_dim(&self) -> usize {
        self.ca_heads * self.ca_dim_head
    }
    /// FeedForward inner width.
    pub fn ff_hidden(&self) -> usize {
        self.dim * self.ff_mult
    }
    /// Query rows entering an IDFormer layer: `latents ‖ id_tokens`.
    pub fn latent_rows(&self) -> usize {
        self.num_queries + self.num_id_token
    }
    /// Context rows an IDFormer layer attends: `id_tokens ‖ mapped ViT scale`.
    pub fn ctx_rows(&self) -> usize {
        self.num_id_token + self.vit_tokens
    }
    /// Key/value rows: `cat(norm1(ctx), norm2(latents))` — the reference
    /// concatenates the *latents themselves* into the kv input, so an IDFormer
    /// layer is a Perceiver block, not a plain cross-attention.
    pub fn kv_rows(&self) -> usize {
        self.ctx_rows() + self.latent_rows()
    }

    /// The cross-attention sites for a backbone of `depth_double` +
    /// `depth_single` blocks, in dispatch order.
    ///
    /// Transcribed from `flux/model.py`: two loops, one shared `ca_idx`
    /// counter. `num_ca` in `pipeline_flux.py` is computed as
    /// `19/2 + 38/4 (+1 each if not divisible)` = 20, which is exactly the
    /// length of this list at the released depth — the assertion in
    /// [`Self::num_ca`].
    pub fn schedule(&self, depth_double: usize, depth_single: usize) -> Vec<Site> {
        let mut v = Vec::new();
        let mut ca = 0usize;
        for block in (0..depth_double).step_by(self.double_interval) {
            v.push(Site { stream: Stream::Double, block, ca });
            ca += 1;
        }
        for block in (0..depth_single).step_by(self.single_interval) {
            v.push(Site { stream: Stream::Single, block, ca });
            ca += 1;
        }
        v
    }

    /// Number of `PerceiverAttentionCA` modules for a backbone depth — the
    /// reference's `num_ca` formula, which `schedule().len()` must equal.
    pub fn num_ca(&self, depth_double: usize, depth_single: usize) -> usize {
        depth_double.div_ceil(self.double_interval) + depth_single.div_ceil(self.single_interval)
    }

    /// Canonical brain-side manifest for the ID encoder (`pulid_encoder.*`).
    ///
    /// `proj_out` is stored **transposed** relative to the checkpoint: the
    /// reference applies it as `latents @ proj_out` with `proj_out` a bare
    /// `[dim, output_dim]` Parameter, while every brain matmul computes
    /// `x @ Wᵀ` from a `[n, k]` weight. The transpose happens once at import.
    pub fn encoder_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (d, ff) = (self.dim, self.ff_hidden());
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("latents".into(), vec![self.num_queries, d]),
            ("proj_out".into(), vec![self.output_dim, d]),
        ];
        // `id_embedding_mapping`: Linear -> LN -> LeakyReLU -> Linear -> LN ->
        // LeakyReLU -> Linear, the last emitting `num_id_token * dim`.
        let mlp = |v: &mut Vec<(String, Vec<usize>)>, pfx: &str, k0: usize, n2: usize| {
            v.push((format!("{pfx}.lin0.weight"), vec![d, k0]));
            v.push((format!("{pfx}.lin0.bias"), vec![d]));
            v.push((format!("{pfx}.ln0.weight"), vec![d]));
            v.push((format!("{pfx}.ln0.bias"), vec![d]));
            v.push((format!("{pfx}.lin1.weight"), vec![d, d]));
            v.push((format!("{pfx}.lin1.bias"), vec![d]));
            v.push((format!("{pfx}.ln1.weight"), vec![d]));
            v.push((format!("{pfx}.ln1.bias"), vec![d]));
            v.push((format!("{pfx}.lin2.weight"), vec![n2, d]));
            v.push((format!("{pfx}.lin2.bias"), vec![n2]));
        };
        mlp(&mut v, "id_map", self.id_cond_dim, self.num_id_token * d);
        for i in 0..self.scales {
            mlp(&mut v, &format!("map{i}"), d, d);
        }
        let inner = self.inner_dim();
        for l in 0..self.depth {
            let b = format!("layers.{l}");
            v.push((format!("{b}.attn.norm1.weight"), vec![d]));
            v.push((format!("{b}.attn.norm1.bias"), vec![d]));
            v.push((format!("{b}.attn.norm2.weight"), vec![d]));
            v.push((format!("{b}.attn.norm2.bias"), vec![d]));
            v.push((format!("{b}.attn.to_q.weight"), vec![inner, d]));
            v.push((format!("{b}.attn.to_kv.weight"), vec![2 * inner, d]));
            v.push((format!("{b}.attn.to_out.weight"), vec![d, inner]));
            v.push((format!("{b}.ff.norm.weight"), vec![d]));
            v.push((format!("{b}.ff.norm.bias"), vec![d]));
            v.push((format!("{b}.ff.w1.weight"), vec![ff, d]));
            v.push((format!("{b}.ff.w2.weight"), vec![d, ff]));
        }
        v
    }

    /// Canonical manifest for the `n` injected cross-attentions (`pulid_ca.*`).
    pub fn ca_manifest(&self, n: usize) -> Vec<(String, Vec<usize>)> {
        let (dm, kv, inner) = (self.ca_dim, self.output_dim, self.ca_inner_dim());
        let mut v = Vec::new();
        for i in 0..n {
            let b = format!("ca.{i}");
            v.push((format!("{b}.norm1.weight"), vec![kv]));
            v.push((format!("{b}.norm1.bias"), vec![kv]));
            v.push((format!("{b}.norm2.weight"), vec![dm]));
            v.push((format!("{b}.norm2.bias"), vec![dm]));
            v.push((format!("{b}.to_q.weight"), vec![inner, dm]));
            v.push((format!("{b}.to_kv.weight"), vec![2 * inner, kv]));
            v.push((format!("{b}.to_out.weight"), vec![dm, inner]));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_reference_num_ca_formula() {
        let c = PulidConfig::v0_9_1();
        let s = c.schedule(19, 38);
        assert_eq!(s.len(), 20);
        assert_eq!(s.len(), c.num_ca(19, 38));
        // doubles 0,2,..,18 -> ca 0..9
        let d: Vec<usize> =
            s.iter().filter(|x| x.stream == Stream::Double).map(|x| x.block).collect();
        assert_eq!(d, (0..19).step_by(2).collect::<Vec<_>>());
        assert_eq!(d.len(), 10);
        // singles 0,4,..,36 -> ca 10..19, ONE shared counter
        let g: Vec<&Site> = s.iter().filter(|x| x.stream == Stream::Single).collect();
        assert_eq!(g.iter().map(|x| x.block).collect::<Vec<_>>(), (0..38).step_by(4).collect::<Vec<_>>());
        assert_eq!(g[0].ca, 10);
        assert_eq!(s.iter().map(|x| x.ca).collect::<Vec<_>>(), (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn reduced_depth_keeps_the_counter_sequential() {
        // what the golden dumps: 2 double + 2 single -> one site each, ca 0 then 1
        let c = PulidConfig::v0_9_1();
        assert_eq!(
            c.schedule(2, 2),
            vec![
                Site { stream: Stream::Double, block: 0, ca: 0 },
                Site { stream: Stream::Single, block: 0, ca: 1 },
            ]
        );
    }

    #[test]
    fn manifest_counts_match_the_checkpoint() {
        let c = PulidConfig::v0_9_1();
        assert_eq!(c.encoder_manifest().len(), 172);
        assert_eq!(c.ca_manifest(20).len(), 140);
        assert_eq!(c.kv_rows(), 619);
        assert_eq!(c.ctx_rows(), 582);
        assert_eq!(c.latent_rows(), 37);
        assert_eq!(c.ca_inner_dim(), c.output_dim);
    }
}
