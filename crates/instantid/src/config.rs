// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! InstantID's shapes, read from the released checkpoint rather than assumed.
//!
//! InstantID conditions SDXL on a face in two places at once, and they are
//! independent mechanisms:
//!
//! * a **face-keypoint ControlNet** — already landed as `crates/controlnet`,
//!   whose SDXL implementation was imported from *this* release
//!   (`InstantX/InstantID/ControlNetModel`, 810 tensors / 5.00 GB);
//! * **IP-Adapter-FaceID decoupled cross-attention** — this crate. An ArcFace
//!   embedding becomes 16 ID tokens through a Perceiver `Resampler`, and every
//!   one of SDXL's 70 cross-attention sites gains its OWN bias-free `to_k_ip` /
//!   `to_v_ip` pair.
//!
//! "Decoupled" is load-bearing and easy to get wrong in a way that still runs:
//! the ID branch does **not** replace the text cross-attention and its tokens are
//! **not** concatenated onto the text ones. It is a second attention over the
//! same queries whose result is ADDED with its own scale —
//! `hidden = text_attn + scale * ip_attn` (upstream
//! `attention_processor.py::IPAttnProcessor`). Replacing or concatenating would
//! still produce a face, just not the conditioning the weights were trained for.

use std::collections::HashMap;

/// The Perceiver resampler that turns one ArcFace embedding into ID tokens.
///
/// Every field is read from the checkpoint by [`ResamplerConfig::from_tensors`];
/// the `released()` values are what `ip-adapter.bin` actually carries and exist
/// so a test can construct the config without the 1.7 GB file.
#[derive(Clone, Debug, PartialEq)]
pub struct ResamplerConfig {
    /// Working width of the resampler stack (1280).
    pub dim: usize,
    /// Number of `PerceiverAttention` + `FeedForward` layers (4).
    pub depth: usize,
    /// Per-head width (64).
    pub dim_head: usize,
    /// Attention heads (20).
    pub heads: usize,
    /// Learned latent queries, and therefore the ID-token count (16).
    pub num_queries: usize,
    /// Input width — the ArcFace embedding (512).
    pub embedding_dim: usize,
    /// Output width of each ID token (2048).
    pub output_dim: usize,
    /// Feed-forward expansion (4 -> inner 5120).
    pub ff_mult: usize,
}

impl ResamplerConfig {
    /// The shapes carried by `InstantX/InstantID/ip-adapter.bin`.
    pub fn released() -> ResamplerConfig {
        ResamplerConfig {
            dim: 1280,
            depth: 4,
            dim_head: 64,
            heads: 20,
            num_queries: 16,
            embedding_dim: 512,
            output_dim: 2048,
            ff_mult: 4,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.dim_head * self.heads
    }
    pub fn ff_inner(&self) -> usize {
        self.dim * self.ff_mult
    }

    /// The attention's key/value length: the reference builds `kv_input` as
    /// `cat(x, latents)`, and `x` is a SINGLE projected token, so k/v are one
    /// row longer than the query count. Getting this wrong is shape-legal in the
    /// `num_queries == 1` case and nowhere else.
    pub fn kv_rows(&self) -> usize {
        1 + self.num_queries
    }

    /// Derive the config from the checkpoint's own `image_proj` shapes.
    ///
    /// Nothing here is hardcoded: `proj_in` gives the ArcFace width and the
    /// model dim, `latents` the query count, `proj_out` the token width, the
    /// `layers.N.` keys the depth, and `to_q` divided by `dim_head` the head
    /// count. A future InstantID release with different shapes imports without
    /// a code change, and a checkpoint that disagrees with itself is an error.
    pub fn from_tensors(t: &HashMap<String, Vec<usize>>) -> Result<ResamplerConfig, String> {
        let get = |k: &str| t.get(k).ok_or_else(|| format!("instantid: image_proj is missing {k}"));
        let proj_in = get("proj_in.weight")?;
        let latents = get("latents")?;
        let proj_out = get("proj_out.weight")?;
        if proj_in.len() != 2 || proj_out.len() != 2 || latents.len() != 3 {
            return Err("instantid: unexpected rank in image_proj (proj_in/proj_out 2-D, latents 3-D)".into());
        }
        let dim = proj_in[0];
        let depth = t
            .keys()
            .filter_map(|k| k.strip_prefix("layers.").and_then(|r| r.split('.').next()))
            .filter_map(|n| n.parse::<usize>().ok())
            .max()
            .map(|m| m + 1)
            .ok_or("instantid: image_proj has no layers.N.* tensors")?;
        let dim_head = 64;
        let to_q = get("layers.0.0.to_q.weight")?;
        if to_q[0] % dim_head != 0 {
            return Err(format!("instantid: to_q out {} is not a multiple of dim_head {dim_head}", to_q[0]));
        }
        let cfg = ResamplerConfig {
            dim,
            depth,
            dim_head,
            heads: to_q[0] / dim_head,
            num_queries: latents[1],
            embedding_dim: proj_in[1],
            output_dim: proj_out[0],
            ff_mult: get("layers.0.1.1.weight")?[0] / dim,
        };
        if latents[2] != cfg.dim {
            return Err(format!("instantid: latents width {} != proj_in out {}", latents[2], cfg.dim));
        }
        if proj_out[1] != cfg.dim {
            return Err(format!("instantid: proj_out in {} != dim {}", proj_out[1], cfg.dim));
        }
        Ok(cfg)
    }

    /// Every tensor the resampler owns, with its shape — the manifest import
    /// validates two-way coverage against.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let (d, inner, ff) = (self.dim, self.inner_dim(), self.ff_inner());
        let mut v = vec![
            ("latents".into(), vec![1, self.num_queries, d]),
            ("proj_in.weight".into(), vec![d, self.embedding_dim]),
            ("proj_in.bias".into(), vec![d]),
            ("proj_out.weight".into(), vec![self.output_dim, d]),
            ("proj_out.bias".into(), vec![self.output_dim]),
            ("norm_out.weight".into(), vec![self.output_dim]),
            ("norm_out.bias".into(), vec![self.output_dim]),
        ];
        for l in 0..self.depth {
            // `layers.N.0` is the PerceiverAttention, `layers.N.1` the
            // nn.Sequential feed-forward — hence the positional 1.0 / 1.1 / 1.3
            // names (LayerNorm, Linear, GELU has no weights, Linear).
            v.push((format!("layers.{l}.0.norm1.weight"), vec![d]));
            v.push((format!("layers.{l}.0.norm1.bias"), vec![d]));
            v.push((format!("layers.{l}.0.norm2.weight"), vec![d]));
            v.push((format!("layers.{l}.0.norm2.bias"), vec![d]));
            v.push((format!("layers.{l}.0.to_q.weight"), vec![inner, d]));
            v.push((format!("layers.{l}.0.to_kv.weight"), vec![2 * inner, d]));
            v.push((format!("layers.{l}.0.to_out.weight"), vec![d, inner]));
            v.push((format!("layers.{l}.1.0.weight"), vec![d]));
            v.push((format!("layers.{l}.1.0.bias"), vec![d]));
            v.push((format!("layers.{l}.1.1.weight"), vec![ff, d]));
            v.push((format!("layers.{l}.1.3.weight"), vec![d, ff]));
        }
        v
    }
}

/// One SDXL cross-attention site's decoupled ID projections.
///
/// The released `ip_adapter` dict is keyed by the site's processor index, and
/// SDXL's 70 `attn2` modules come in two widths (640 and 1280) — so the site
/// list is data read from the checkpoint, not a hardcoded schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteConfig {
    /// The processor index the release names this site by.
    pub index: usize,
    /// The site's hidden width (`to_k_ip` output).
    pub hidden: usize,
    /// The ID-token width (`to_k_ip` input) — the resampler's `output_dim`.
    pub token_dim: usize,
}

impl SiteConfig {
    /// Read every site from the `ip_adapter` shapes, sorted by index.
    ///
    /// Both projections must be present and agree: a site with `to_k_ip` and no
    /// `to_v_ip` would otherwise import as a half-wired attention that runs.
    pub fn from_tensors(t: &HashMap<String, Vec<usize>>) -> Result<Vec<SiteConfig>, String> {
        let mut out: Vec<SiteConfig> = Vec::new();
        for (k, shape) in t {
            let Some(idx) = k.strip_suffix(".to_k_ip.weight") else { continue };
            let index: usize = idx.parse().map_err(|_| format!("instantid: site key '{k}' is not an index"))?;
            let vk = format!("{idx}.to_v_ip.weight");
            let vshape = t.get(&vk).ok_or_else(|| format!("instantid: site {index} has to_k_ip but no {vk}"))?;
            if shape != vshape {
                return Err(format!("instantid: site {index} k/v shapes disagree: {shape:?} vs {vshape:?}"));
            }
            if shape.len() != 2 {
                return Err(format!("instantid: site {index} to_k_ip should be 2-D, got {shape:?}"));
            }
            out.push(SiteConfig { index, hidden: shape[0], token_dim: shape[1] });
        }
        if out.is_empty() {
            return Err("instantid: ip_adapter carries no *.to_k_ip.weight".into());
        }
        out.sort_by_key(|s| s.index);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released_shapes() -> HashMap<String, Vec<usize>> {
        ResamplerConfig::released().tensor_manifest().into_iter().collect()
    }

    #[test]
    fn the_config_round_trips_through_its_own_manifest() {
        let want = ResamplerConfig::released();
        let got = ResamplerConfig::from_tensors(&released_shapes()).expect("derive");
        assert_eq!(got, want);
    }

    #[test]
    fn kv_is_one_row_longer_than_the_queries() {
        // `kv_input = cat(x, latents)` with x a single projected token. Sizing
        // k/v at num_queries instead would be shape-legal only at 1 query.
        let c = ResamplerConfig::released();
        assert_eq!(c.kv_rows(), c.num_queries + 1);
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let mut s = released_shapes();
        s.remove("latents");
        let e = ResamplerConfig::from_tensors(&s).unwrap_err();
        assert!(e.contains("latents"), "error should name the tensor, got: {e}");
    }

    #[test]
    fn a_half_wired_site_is_rejected() {
        // A site with to_k_ip and no to_v_ip would import as an attention that
        // runs with an uninitialised value projection.
        let mut t: HashMap<String, Vec<usize>> = HashMap::new();
        t.insert("1.to_k_ip.weight".into(), vec![640, 2048]);
        let e = SiteConfig::from_tensors(&t).unwrap_err();
        assert!(e.contains("to_v_ip"), "error should name the missing projection, got: {e}");
    }

    #[test]
    fn sites_come_back_in_index_order() {
        let mut t: HashMap<String, Vec<usize>> = HashMap::new();
        for i in [5usize, 1, 3] {
            t.insert(format!("{i}.to_k_ip.weight"), vec![1280, 2048]);
            t.insert(format!("{i}.to_v_ip.weight"), vec![1280, 2048]);
        }
        let s = SiteConfig::from_tensors(&t).expect("sites");
        assert_eq!(s.iter().map(|x| x.index).collect::<Vec<_>>(), vec![1, 3, 5]);
    }
}
