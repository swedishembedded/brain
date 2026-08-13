// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a GenieRedux-G (CoinRun) tokenizer `.pt` checkpoint into
//! [`TokenizerWeights`], with full-coverage validation (every `model.*` tensor
//! is consumed; a missing or unexpected/leftover tensor is a hard error — never
//! silently skipped, matching `glm`/`wm-diamond` import discipline).
//!
//! Fixed config (the released 100M CoinRun tokenizer): dim 512, 8 heads × 64,
//! GEGLU inner 1365, VQ codebook 1024×32, patch 4×4×3, 8 encoder + 8 decoder
//! STBlocks. Two projections are split on load: the fused `to_kv` → `to_k|to_v`,
//! and the GEGLU in-proj `spatial_ff.1` → `w_x|w_gate` (`chunk(2)`: first half
//! is the value branch `x`, second is the `gate`).

use crate::bias::CpbLayer;
use crate::{
    AttnWeights, DynamicsWeights, FfWeights, PatchEmbedWeights, PegWeights, StBlockWeights,
    StTransformerWeights, ToPixelsWeights, TokenizerWeights, VqWeights,
};
use std::collections::HashMap;

/// Static geometry of the released GenieRedux-CoinRun tokenizer.
#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub ff_inner: u32,
    pub code_dim: u32,
    pub n_codes: u32,
    pub patch: u32,
    pub channels: u32,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub cpb_hidden: u32,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        TokenizerConfig {
            dim: 512, heads: 8, head_dim: 64, ff_inner: 1365,
            code_dim: 32, n_codes: 1024, patch: 4, channels: 3,
            enc_layers: 8, dec_layers: 8, cpb_hidden: 512,
        }
    }
}

/// Consumes tensors by name with shape validation, tracking leftovers.
struct Loader {
    map: HashMap<String, (Vec<usize>, Vec<f32>)>,
}
impl Loader {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (got, data) = self
            .map
            .remove(name)
            .ok_or_else(|| format!("missing tensor `{name}`"))?;
        if got != shape {
            return Err(format!("`{name}`: checkpoint shape {got:?}, expected {shape:?}"));
        }
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(format!("`{name}`: {} values for shape {shape:?}", data.len()));
        }
        Ok(data)
    }
    /// Take a buffer, assert it is (numerically) all-zero, and discard it — used
    /// for the custom LayerNorm's non-trainable `beta` buffers (always 0) and
    /// the unused self-attention `context_norm`, so full coverage still holds.
    fn take_zeros(&mut self, name: &str, shape: &[usize]) -> Result<(), String> {
        let v = self.take(name, shape)?;
        if v.iter().any(|x| x.abs() > 1e-6) {
            return Err(format!("`{name}` expected all-zero (unused/buffer) but was not"));
        }
        Ok(())
    }
    /// Take and discard (unused-but-present tensor, kept for full coverage).
    fn drop(&mut self, name: &str, shape: &[usize]) -> Result<(), String> {
        self.take(name, shape).map(|_| ())
    }
}

fn ln_bias(l: &mut Loader, prefix: &str, dim: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    // nn.LayerNorm: `.weight` = gamma, `.bias` = beta.
    Ok((l.take(&format!("{prefix}.weight"), &[dim])?, l.take(&format!("{prefix}.bias"), &[dim])?))
}

fn patch_embed(l: &mut Loader, prefix: &str, pf: usize, dim: usize) -> Result<PatchEmbedWeights, String> {
    // Sequential: [Rearrange, LayerNorm(pf)=.1, Linear(pf->dim)=.2, LayerNorm(dim)=.3]
    let (ln1_g, ln1_b) = ln_bias(l, &format!("{prefix}.1"), pf)?;
    let lin_w = l.take(&format!("{prefix}.2.weight"), &[dim, pf])?;
    let lin_b = l.take(&format!("{prefix}.2.bias"), &[dim])?;
    let (ln2_g, ln2_b) = ln_bias(l, &format!("{prefix}.3"), dim)?;
    Ok(PatchEmbedWeights { ln1_g, ln1_b, lin_w, lin_b, ln2_g, ln2_b })
}

fn attn(l: &mut Loader, prefix: &str, cfg: &TokenizerConfig) -> Result<AttnWeights, String> {
    let dim = cfg.dim as usize;
    let inner = (cfg.heads * cfg.head_dim) as usize;
    let hd = cfg.head_dim as usize;
    let q_scale = l.take(&format!("{prefix}.q_scale"), &[hd])?;
    let k_scale = l.take(&format!("{prefix}.k_scale"), &[hd])?;
    // custom LayerNorm: gamma trainable, beta a zero buffer.
    let norm_gamma = l.take(&format!("{prefix}.norm.gamma"), &[dim])?;
    l.take_zeros(&format!("{prefix}.norm.beta"), &[dim])?;
    // context_norm is unused for self-attention (context is None) but present.
    l.drop(&format!("{prefix}.context_norm.gamma"), &[dim])?;
    l.take_zeros(&format!("{prefix}.context_norm.beta"), &[dim])?;
    let to_q = l.take(&format!("{prefix}.to_q.weight"), &[inner, dim])?;
    // fused to_kv [2*inner, dim] -> to_k | to_v
    let kv = l.take(&format!("{prefix}.to_kv.weight"), &[2 * inner, dim])?;
    let to_k = kv[..inner * dim].to_vec();
    let to_v = kv[inner * dim..].to_vec();
    let to_out = l.take(&format!("{prefix}.to_out.weight"), &[dim, inner])?;
    Ok(AttnWeights { norm_gamma, to_q, to_k, to_v, to_out, q_scale, k_scale })
}

fn ff(l: &mut Loader, prefix: &str, cfg: &TokenizerConfig) -> Result<FfWeights, String> {
    let dim = cfg.dim as usize;
    let inner = cfg.ff_inner as usize;
    // Sequential: [LayerNorm=.0, Linear(dim->2*inner)=.1, GEGLU, Dropout, Linear(inner->dim)=.4]
    let (norm_gamma, norm_beta) = ln_bias(l, &format!("{prefix}.0"), dim)?;
    let w1 = l.take(&format!("{prefix}.1.weight"), &[2 * inner, dim])?;
    let w_x = w1[..inner * dim].to_vec();      // chunk(2) first half = value branch
    let w_gate = w1[inner * dim..].to_vec();   // second half = gate
    let w_out = l.take(&format!("{prefix}.4.weight"), &[dim, inner])?;
    Ok(FfWeights { norm_gamma, norm_beta, w_x, w_gate, w_out })
}

fn peg(l: &mut Loader, prefix: &str, dim: usize) -> Result<PegWeights, String> {
    let dsconv = l.take(&format!("{prefix}.dsconv.weight"), &[dim, 1, 3, 3, 3])?;
    let bias = l.take(&format!("{prefix}.dsconv.bias"), &[dim])?;
    Ok(PegWeights { dsconv, bias })
}

fn block(l: &mut Loader, prefix: &str, cfg: &TokenizerConfig) -> Result<StBlockWeights, String> {
    let dim = cfg.dim as usize;
    Ok(StBlockWeights {
        spatial_peg: peg(l, &format!("{prefix}.spatial_peg"), dim)?,
        spatial_attn: attn(l, &format!("{prefix}.spatial_attention"), cfg)?,
        spatial_ff: ff(l, &format!("{prefix}.spatial_ff"), cfg)?,
        temporal_peg: peg(l, &format!("{prefix}.temporal_peg"), dim)?,
        temporal_attn: attn(l, &format!("{prefix}.temporal_attention"), cfg)?,
        temporal_ff: ff(l, &format!("{prefix}.temporal_ff"), cfg)?,
    })
}

fn stack(l: &mut Loader, prefix: &str, n: usize, cfg: &TokenizerConfig) -> Result<StTransformerWeights, String> {
    let dim = cfg.dim as usize;
    let mut layers = Vec::with_capacity(n);
    for i in 0..n {
        layers.push(block(l, &format!("{prefix}.layers.{i}.0"), cfg)?);
    }
    let norm_out_gamma = l.take(&format!("{prefix}.norm_out.gamma"), &[dim])?;
    l.take_zeros(&format!("{prefix}.norm_out.beta"), &[dim])?;
    Ok(StTransformerWeights { layers, norm_out_gamma })
}

fn cpb(l: &mut Loader, prefix: &str, cfg: &TokenizerConfig) -> Result<Vec<CpbLayer>, String> {
    let d = cfg.cpb_hidden as usize;
    let heads = cfg.heads as usize;
    // net.0.0 (Linear 2->d), net.1.0 (Linear d->d), net.2 (Linear d->heads)
    let l0w = l.take(&format!("{prefix}.net.0.0.weight"), &[d, 2])?;
    let l0b = l.take(&format!("{prefix}.net.0.0.bias"), &[d])?;
    let l1w = l.take(&format!("{prefix}.net.1.0.weight"), &[d, d])?;
    let l1b = l.take(&format!("{prefix}.net.1.0.bias"), &[d])?;
    let l2w = l.take(&format!("{prefix}.net.2.weight"), &[heads, d])?;
    let l2b = l.take(&format!("{prefix}.net.2.bias"), &[heads])?;
    Ok(vec![
        CpbLayer { w: l0w, b: l0b, in_dim: 2, out_dim: d },
        CpbLayer { w: l1w, b: l1b, in_dim: d, out_dim: d },
        CpbLayer { w: l2w, b: l2b, in_dim: d, out_dim: heads },
    ])
}

/// Read `pt_path` and map its `model.*` tensors to [`TokenizerWeights`].
pub fn import_tokenizer(pt_path: &str) -> Result<(TokenizerWeights, TokenizerConfig), String> {
    let cfg = TokenizerConfig::default();
    let report = checkpoint::torchpt::read_report(pt_path)?;
    let mut map = HashMap::new();
    for t in report.tensors {
        if let Some(name) = t.name.strip_prefix("model.") {
            map.insert(name.to_string(), (t.shape, t.data));
        }
    }
    if map.is_empty() {
        return Err(format!("{pt_path}: no `model.*` tensors — not a GenieRedux tokenizer?"));
    }
    let mut l = Loader { map };
    let (dim, pf) = (cfg.dim as usize, (cfg.channels * cfg.patch * cfg.patch) as usize);

    let cpb_net = cpb(&mut l, "spatial_rel_pos_bias", &cfg)?;
    let patch_first = patch_embed(&mut l, "to_patch_emb_first_frame", pf, dim)?;
    let patch_rest = patch_embed(&mut l, "to_patch_emb", pf, dim)?;
    let encoder = stack(&mut l, "encoder", cfg.enc_layers, &cfg)?;
    let decoder = stack(&mut l, "decoder", cfg.dec_layers, &cfg)?;

    // VQ: project_in (dim->cd, bias), codebook [1,K,cd], project_out (cd->dim, bias).
    let (cd, kk) = (cfg.code_dim as usize, cfg.n_codes as usize);
    let vq = VqWeights {
        project_in_w: l.take("vq.project_in.weight", &[cd, dim])?,
        project_in_b: l.take("vq.project_in.bias", &[cd])?,
        codebook: l.take("vq._codebook.embed", &[1, kk, cd])?,
        project_out_w: l.take("vq.project_out.weight", &[dim, cd])?,
        project_out_b: l.take("vq.project_out.bias", &[dim])?,
    };
    // VQ EMA/bookkeeping buffers: present, not used at inference.
    l.drop("vq._codebook.initted", &[1])?;
    l.drop("vq._codebook.cluster_size", &[1, kk])?;
    l.drop("vq._codebook.embed_avg", &[1, kk, cd])?;

    let to_pixels_first = ToPixelsWeights {
        lin_w: l.take("to_pixels_first_frame.0.weight", &[pf, dim])?,
        lin_b: l.take("to_pixels_first_frame.0.bias", &[pf])?,
    };
    let to_pixels_rest = ToPixelsWeights {
        lin_w: l.take("to_pixels.0.weight", &[pf, dim])?,
        lin_b: l.take("to_pixels.0.bias", &[pf])?,
    };

    if !l.map.is_empty() {
        let mut extra: Vec<&String> = l.map.keys().collect();
        extra.sort();
        return Err(format!(
            "{} unexpected tensor(s) not mapped (never silently skipped), e.g. {:?}",
            l.map.len(),
            &extra[..extra.len().min(8)]
        ));
    }
    Ok((
        TokenizerWeights {
            patch_first, patch_rest, encoder, vq, decoder, to_pixels_first, to_pixels_rest, cpb_net,
        },
        cfg,
    ))
}

/// Static geometry of the released GenieRedux-CoinRun guided dynamics.
#[derive(Clone, Copy, Debug)]
pub struct DynamicsConfig {
    pub dim: u32,          // token/embedding dim (512)
    pub dim2: u32,         // transformer residual dim = dim + num_actions (519)
    pub heads: u32,
    pub head_dim: u32,
    pub ff_inner: u32,     // round(dim2 * 8/3) = 1384
    pub n_codes: u32,      // 1024
    pub code_dim: u32,     // 32
    pub num_actions: u32,  // 7
    pub layers: usize,     // 12
    pub cpb_hidden: u32,   // 64
    pub max_seq_len: u32,  // 4000
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        DynamicsConfig {
            dim: 512, dim2: 519, heads: 8, head_dim: 64, ff_inner: 1384,
            n_codes: 1024, code_dim: 32, num_actions: 7, layers: 12,
            cpb_hidden: 64, max_seq_len: 4000,
        }
    }
}

/// Read the dynamics `.pt` and map its `model.dynamics.maskgit.*` tensors to
/// [`DynamicsWeights`]. The tokenizer's `codebook`/`project_out` (from
/// [`import_tokenizer`]) are needed for the `use_token` embedding blend and are
/// copied in from `tok_vq`.
pub fn import_dynamics(pt_path: &str, tok_vq: &VqWeights) -> Result<(DynamicsWeights, DynamicsConfig), String> {
    let dc = DynamicsConfig::default();
    let report = checkpoint::torchpt::read_report(pt_path)?;
    let mut map = HashMap::new();
    for t in report.tensors {
        if let Some(name) = t.name.strip_prefix("model.dynamics.maskgit.") {
            map.insert(name.to_string(), (t.shape, t.data));
        }
    }
    if map.is_empty() {
        return Err(format!("{pt_path}: no `model.dynamics.maskgit.*` tensors — not a GenieRedux dynamics?"));
    }
    let mut l = Loader { map };

    // Reuse the STBlock helpers via a config with the transformer's dim2.
    let tcfg = TokenizerConfig {
        dim: dc.dim2, heads: dc.heads, head_dim: dc.head_dim, ff_inner: dc.ff_inner,
        code_dim: dc.code_dim, n_codes: dc.n_codes, patch: 4, channels: 3,
        enc_layers: dc.layers, dec_layers: 0, cpb_hidden: dc.cpb_hidden,
    };

    let (dim, dim2) = (dc.dim as usize, dc.dim2 as usize);
    let token_emb = l.take("token_emb.weight", &[dc.n_codes as usize + 1, dim])?;
    let pos_emb = l.take("pos_emb.weight", &[dc.max_seq_len as usize, dim])?;
    let cpb_net = cpb(&mut l, "continuous_pos_bias", &tcfg)?;
    let transformer = stack(&mut l, "transformer", dc.layers, &tcfg)?;
    let to_logits_w = l.take("to_logits.weight", &[dc.n_codes as usize, dim2])?;
    let to_logits_b = l.take("to_logits.bias", &[dc.n_codes as usize])?;

    if !l.map.is_empty() {
        let mut extra: Vec<&String> = l.map.keys().collect();
        extra.sort();
        return Err(format!(
            "{} unexpected dynamics tensor(s) not mapped, e.g. {:?}",
            l.map.len(),
            &extra[..extra.len().min(8)]
        ));
    }
    Ok((
        DynamicsWeights {
            token_emb, pos_emb,
            codebook: tok_vq.codebook.clone(),
            project_out_w: tok_vq.project_out_w.clone(),
            project_out_b: tok_vq.project_out_b.clone(),
            transformer, cpb_net, to_logits_w, to_logits_b,
        },
        dc,
    ))
}
