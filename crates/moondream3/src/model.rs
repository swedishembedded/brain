// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream 3 composite: SigLIP ViT → connector → spliced MoE decoder.
//!
//! Mirrors `qwen3vl::Qwen3Vl`/`fastvlm::FastVlm`: the vision tower runs on its own
//! `Gpu`, image tokens cross host-side into the decoder's positional prefix.
//!
//! # The stack is built ONCE and owned
//!
//! This composite used to rebuild the ViT, the connector and all 24 decoder
//! blocks from host `Vec<f32>` weights on **every forward**, because those types
//! borrowed the `Gpu` (`SiglipEncoder<'g>`, `MoondreamBlock<'g>`, …) and a struct
//! cannot hold both a device and something borrowing it. That is fine for a
//! research forward and fatal for a served one: at the real preview config the
//! decoder is 8.8 B parameters, so a per-call rebuild re-uploads ~33 GB before
//! answering each request.
//!
//! Those five types now own their `DeviceBuffer`s and take `&Gpu` as a method
//! argument - the same shape `sam1::SamEncoder` and `sam2::Sam2` already use, and
//! the reason they can be resident while this could not. [`MoondreamModel`]
//! therefore owns its two devices AND the built stack: weights are uploaded once,
//! in [`MoondreamModel::new`], and dropping the model frees them.
//!
//! NB: the single-crop path is what [`MoondreamModel::forward`] runs; the global‖
//! local overlap-multi-crop concat that widens the connector input to `2·dim` is
//! [`MoondreamModel::forward_multicrop`].

use std::collections::HashMap;

use gpu_core::Gpu;

use crate::config::MoondreamConfig;
use crate::decoder::{MoeFfn, MoeFfn8, MoondreamBlock, MoondreamDecoder};
use crate::vision::{Connector, SiglipEncoder};

/// How the decoder's expert weights are stored, and therefore whether the built
/// model can be differentiated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Precision {
    /// fp32 experts, per-block activation scratch. Differentiable; this is what
    /// every gradcheck builds.
    #[default]
    Fp32,
    /// int8 experts ([`MoeFfn8`]) and ONE activation set shared by all blocks.
    /// Inference only - `backward` on such a block refuses by name.
    ///
    /// This is not a tuning knob, it is what makes the released config
    /// loadable: fp32 experts plus per-block scratch is ~43 GiB, int8 plus
    /// shared scratch is ~9 GiB.
    Int8,
}

/// An assembled Moondream 3 (forward path). Image tokens occupy rows `[1, 1+n_img)`.
///
/// Owns both devices and every device buffer the stack needs, so a request runs
/// the graph rather than rebuilding it.
pub struct MoondreamModel {
    vgpu: Gpu,
    dgpu: Gpu,
    cfg: MoondreamConfig,
    enc: SiglipEncoder,
    conn: Connector,
    dec: MoondreamDecoder,
    conn_in: u32,
    seq_len: u32,
    precision: Precision,
}

impl MoondreamModel {
    /// Build the whole stack on the two given devices, uploading every weight once.
    ///
    /// `vgpu` must carry [`crate::vision::vision_pipelines`] and `dgpu`
    /// [`crate::decoder::pipelines`]. Two devices rather than one because the
    /// towers have disjoint kernel sets and (at real scale) very different
    /// footprints - the same split `deepseek2ocr` and `fastvlm` use.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vgpu: Gpu,
        dgpu: Gpu,
        cfg: MoondreamConfig,
        vweights: HashMap<String, Vec<f32>>,
        conn_weights: HashMap<String, Vec<f32>>,
        dweights: HashMap<String, Vec<f32>>,
        conn_in: u32,
        seq_len: u32,
    ) -> MoondreamModel {
        Self::new_with(vgpu, dgpu, cfg, vweights, conn_weights, dweights, conn_in, seq_len, Precision::Fp32)
    }

    /// [`MoondreamModel::new`] at a chosen [`Precision`].
    ///
    /// `Precision::Int8` quantizes the MoE experts on the way in and puts every
    /// block on one shared activation set - together the difference between
    /// ~43 GiB and ~9 GiB at the released config, and therefore between "does
    /// not load" and "loads". The resulting model cannot be differentiated.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with(
        vgpu: Gpu,
        dgpu: Gpu,
        cfg: MoondreamConfig,
        vweights: HashMap<String, Vec<f32>>,
        conn_weights: HashMap<String, Vec<f32>>,
        dweights: HashMap<String, Vec<f32>>,
        conn_in: u32,
        seq_len: u32,
        precision: Precision,
    ) -> MoondreamModel {
        let enc = SiglipEncoder::new(&vgpu, cfg.vision.clone(), &vweights);
        let conn = Connector::new(&vgpu, &conn_weights, conn_in, cfg.proj_inner, cfg.proj_out);
        let blocks = build_blocks(&dgpu, &cfg, &dweights, seq_len, precision);
        let ppc = cfg.vision.patches_per_crop();
        let mut dec = MoondreamDecoder::new(&dgpu, &dweights, blocks, seq_len, cfg.dim, cfg.vocab, ppc);
        if precision == Precision::Int8 {
            dec = dec.share_scratch(&dgpu, cfg.n_heads, cfg.ff_dim);
        }
        MoondreamModel { vgpu, dgpu, cfg, enc, conn, dec, conn_in, seq_len, precision }
    }

    /// Build on the CPU backend for both towers - the shape every checkpoint-free
    /// test uses, and the honest default on a box with no usable accelerator.
    #[allow(clippy::too_many_arguments)]
    pub fn new_cpu(
        cfg: MoondreamConfig,
        vweights: HashMap<String, Vec<f32>>,
        conn_weights: HashMap<String, Vec<f32>>,
        dweights: HashMap<String, Vec<f32>>,
        conn_in: u32,
        seq_len: u32,
    ) -> MoondreamModel {
        Self::new_cpu_with(cfg, vweights, conn_weights, dweights, conn_in, seq_len, Precision::Fp32)
    }

    /// [`MoondreamModel::new_cpu`] at a chosen [`Precision`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_cpu_with(
        cfg: MoondreamConfig,
        vweights: HashMap<String, Vec<f32>>,
        conn_weights: HashMap<String, Vec<f32>>,
        dweights: HashMap<String, Vec<f32>>,
        conn_in: u32,
        seq_len: u32,
        precision: Precision,
    ) -> MoondreamModel {
        Self::new_with(
            Gpu::new_cpu(crate::vision::vision_pipelines()),
            Gpu::new_cpu(crate::decoder::pipelines()),
            cfg,
            vweights,
            conn_weights,
            dweights,
            conn_in,
            seq_len,
            precision,
        )
    }

    /// Which precision this model was built at.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn config(&self) -> &MoondreamConfig {
        &self.cfg
    }

    /// The built context length. The image block alone is `1 + patches_per_crop`
    /// rows, so anything shorter than that cannot hold a prompt at all.
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Encode one crop and project it to `[patches_per_crop, proj_out]` image
    /// embeddings - the value that crosses from the vision device to the decoder's
    /// device as a host `Vec<f32>` (never a raw device buffer, which is what lets
    /// the two towers sit on different backends).
    pub fn image_embeds(&self, packed: &[f32]) -> Vec<f32> {
        let feats = self.enc.encode(&self.vgpu, 1, packed);
        self.conn.forward(&self.vgpu, self.cfg.vision.patches_per_crop(), &feats)
    }

    /// [`Self::image_embeds`] for the faithful overlap multi-crop input: a global
    /// crop plus `h_tiles·w_tiles` local crops, reconstructed and adaptive-pooled
    /// into the `[patches_per_crop, 2·dim]` connector input. Requires
    /// `conn_in == 2·vision.dim`.
    pub fn image_embeds_multicrop(&self, global_packed: &[f32], locals_packed: &[f32], h_tiles: u32, w_tiles: u32) -> Vec<f32> {
        let (dim, grid, margin) = (self.cfg.vision.dim, self.cfg.vision.grid(), self.cfg.vision.overlap_margin);
        assert_eq!(self.conn_in, 2 * dim, "multi-crop connector input must be 2·vision.dim");
        let global = self.enc.encode(&self.vgpu, 1, global_packed);
        let locals = self.enc.encode(&self.vgpu, h_tiles * w_tiles, locals_packed);
        let concat = crate::preprocess::build_connector_input(&self.vgpu, &global, &locals, h_tiles, w_tiles, grid, dim, margin);
        self.conn.forward(&self.vgpu, self.cfg.vision.patches_per_crop(), &concat)
    }

    /// Image embeddings straight from RAW HWC PIXELS - the full reference
    /// front end: overlap multi-crop, ViT, feature-space stitch, connector.
    ///
    /// This is the entry point a served request uses, and the one that makes
    /// the multi-crop path reachable without the caller doing its own
    /// cropping. Falls back to the single-crop path when the model was built
    /// with a `conn_in` of one `vision.dim` rather than two, since the
    /// global‖local concat is what widens the connector input.
    pub fn image_embeds_from_pixels(&self, hwc: &[f32], w: u32, h: u32) -> Vec<f32> {
        let v = &self.cfg.vision;
        if self.conn_in != 2 * v.dim {
            // Single-crop connector: just the whole image at one crop's size.
            let side = v.crop_size;
            let resized = imaging::host::resize_bilinear_hwc(hwc, 3, w, h, side, side);
            let packed = crate::preprocess::patchify_crop(&resized, side, v.patch);
            return self.image_embeds(&packed);
        }
        let (global, locals, plan) = crate::preprocess::overlap_crop_image(hwc, w, h, v);
        self.image_embeds_multicrop(&global, &locals, plan.h_tiles, plan.w_tiles)
    }

    /// [`Self::image_embeds_from_pixels`] for SEVERAL images in ONE ViT pass.
    ///
    /// # This is the model's real batch axis
    ///
    /// The decoder has none worth using: each request has its own prompt, its
    /// own image embeddings and its own KV cache, and the block forward has no
    /// batch dimension. The VISION tower does. `SiglipEncoder::encode` already
    /// takes a crop count and attends within each crop as its own span, so N
    /// requests' crops concatenate into one call - and at the released config
    /// each request is 1 global + up to 12 local crops of 729 patches, which is
    /// the dominant per-request cost.
    ///
    /// Crops-per-request VARIES (the tiling follows each image's aspect ratio),
    /// so this returns per-request embeddings sliced back out by each request's
    /// own tile count rather than assuming a uniform stride.
    pub fn image_embeds_from_pixels_batch(&self, images: &[(&[f32], u32, u32)]) -> Vec<Vec<f32>> {
        let v = &self.cfg.vision;
        let ppc = v.patches_per_crop() as usize;
        let pv = v.patch_vec() as usize;
        if self.conn_in != 2 * v.dim {
            // Single-crop connector: one crop per request, still one ViT pass.
            let mut packed = Vec::with_capacity(images.len() * ppc * pv);
            for &(hwc, w, h) in images {
                let r = imaging::host::resize_bilinear_hwc(hwc, 3, w, h, v.crop_size, v.crop_size);
                packed.extend(crate::preprocess::patchify_crop(&r, v.crop_size, v.patch));
            }
            let feats = self.enc.encode(&self.vgpu, images.len() as u32, &packed);
            let dim = v.dim as usize;
            return (0..images.len())
                .map(|i| self.conn.forward(&self.vgpu, v.patches_per_crop(), &feats[i * ppc * dim..(i + 1) * ppc * dim]))
                .collect();
        }

        // Multi-crop: lay every request's global crop and local crops end to
        // end, encode once, then stitch each request back separately.
        let mut packed: Vec<f32> = Vec::new();
        let mut plans = Vec::with_capacity(images.len());
        for &(hwc, w, h) in images {
            let (global, locals, plan) = crate::preprocess::overlap_crop_image(hwc, w, h, v);
            packed.extend_from_slice(&global);
            packed.extend_from_slice(&locals);
            plans.push(plan);
        }
        let total_crops: u32 = plans.iter().map(|p| 1 + p.h_tiles * p.w_tiles).sum();
        let feats = self.enc.encode(&self.vgpu, total_crops, &packed);

        let dim = v.dim as usize;
        let (grid, margin) = (v.grid(), v.overlap_margin);
        let mut out = Vec::with_capacity(images.len());
        let mut crop0 = 0usize;
        for plan in &plans {
            let n_local = (plan.h_tiles * plan.w_tiles) as usize;
            let g0 = crop0 * ppc * dim;
            let global = &feats[g0..g0 + ppc * dim];
            let locals = &feats[g0 + ppc * dim..g0 + (1 + n_local) * ppc * dim];
            let concat = crate::preprocess::build_connector_input(&self.vgpu, global, locals, plan.h_tiles, plan.w_tiles, grid, v.dim, margin);
            out.push(self.conn.forward(&self.vgpu, v.patches_per_crop(), &concat));
            crop0 += 1 + n_local;
        }
        out
    }

    /// End-to-end forward: encode one crop, project, splice, decode → loss.
    /// `tokens`/`targets` length `seq_len`; `packed` is `[patches, patch_vec]`.
    pub fn forward(&self, tokens: &[u32], targets: &[u32], packed: &[f32]) -> f32 {
        let img_embeds = self.image_embeds(packed);
        self.dec.forward(&self.dgpu, tokens, targets, &img_embeds)
    }

    /// Faithful overlap multi-crop forward → loss. `global_packed` is one crop's
    /// `[ppc, patch_vec]`; `locals_packed` is `[h·w·ppc, patch_vec]` (tile order).
    pub fn forward_multicrop(&self, tokens: &[u32], targets: &[u32], global_packed: &[f32], locals_packed: &[f32], h_tiles: u32, w_tiles: u32) -> f32 {
        let img_embeds = self.image_embeds_multicrop(global_packed, locals_packed, h_tiles, w_tiles);
        self.dec.forward(&self.dgpu, tokens, targets, &img_embeds)
    }

    /// Greedy autoregressive decode: `prompt` (already including the bos and the
    /// `n_img` image-placeholder rows) in, generated ids out.
    ///
    /// Stops at `eos` or after `max_new` tokens, whichever comes first.
    ///
    /// # Why padding the sequence is exact here, not an approximation
    ///
    /// [`Self::logits`] runs a graph built for a FIXED `seq_len`, so each step
    /// pads `tokens` out to that length and reads row `pos - 1`. That is not a
    /// shortcut: this decoder's mask is
    /// `allow(i, j) = (i < P && j < P) || (j <= i)` (`attn_prefix_mask`), i.e.
    /// the `P = prefix_attn` bos+image rows attend to each other bidirectionally
    /// and EVERYTHING ELSE IS CAUSAL. Row `pos - 1` therefore reads no position
    /// past itself, so whatever sits in the padding slots cannot reach it. A
    /// model with a bidirectional tail would need real masking instead, and this
    /// loop would be wrong for it.
    ///
    /// # This is `O(T²)` per token, and that is the honest cost
    ///
    /// There is no KV cache: every step re-runs the whole grown sequence through
    /// all `n_layers`. `crates/gpt2`, `crates/qwen3` and `crates/deepseek2` each
    /// keep an `O(1)`-per-token incremental twin alongside their recompute tier;
    /// this decoder does not have one yet. At the preview config that is 24
    /// layers over a 730-row image prefix per generated token, so a long caption
    /// is minutes, not milliseconds.
    pub fn generate(&self, prompt: &[u32], img_embeds: &[f32], max_new: usize, eos: Option<u32>) -> Result<Vec<u32>, String> {
        let t = self.seq_len as usize;
        if prompt.len() >= t {
            return Err(format!("moondream3: prompt is {} tokens but the graph was built for seq_len {t}", prompt.len()));
        }
        let pad = prompt.last().copied().unwrap_or(0);
        let mut ids = prompt.to_vec();
        let mut out = Vec::new();
        let vocab = self.cfg.vocab as usize;
        while out.len() < max_new && ids.len() < t {
            let pos = ids.len();
            let mut padded = ids.clone();
            padded.resize(t, pad);
            let logits = self.logits(&padded, img_embeds);
            let row = &logits[(pos - 1) * vocab..pos * vocab];
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or("moondream3: empty logits row")?;
            if Some(next) == eos {
                break;
            }
            out.push(next);
            ids.push(next);
        }
        Ok(out)
    }

    /// [`Self::generate`] with a KV CACHE: one batched prefill over the prompt,
    /// then `O(pos)` incremental steps instead of a full `O(T²)` recompute per
    /// token.
    ///
    /// Produces the SAME ids as [`Self::generate`] - pinned by
    /// `kv_decode_matches_the_recompute_path_token_for_token`, which is the only
    /// thing that makes an incremental path safe to prefer: the two share no
    /// code, so agreement is evidence rather than tautology.
    ///
    /// The prefill is what keeps the image prefix BIDIRECTIONAL. Decode steps
    /// are causal-only by construction (they read cache rows `0..=pos`), which
    /// is correct for generated tokens and would be wrong for the prefix - so
    /// the prefix's K/V come from the masked batched pass, once.
    pub fn generate_kv(&self, prompt: &[u32], img_embeds: &[f32], max_new: usize, eos: Option<u32>) -> Result<Vec<u32>, String> {
        let t = self.seq_len as usize;
        if prompt.len() >= t {
            return Err(format!("moondream3: prompt is {} tokens but the graph was built for seq_len {t}", prompt.len()));
        }
        let c = &self.cfg;
        let caches = self.dec.new_kv_caches(&self.dgpu, c.n_heads, c.head_dim, c.ff_dim);
        let mut padded = prompt.to_vec();
        let pad = prompt.last().copied().unwrap_or(0);
        padded.resize(t, pad);

        let mut hidden = self.dec.prefill(&self.dgpu, &padded, img_embeds, &caches, prompt.len() as u32);
        let mut out = Vec::new();
        let mut pos = prompt.len() as u32;
        while out.len() < max_new && (pos as usize) < t {
            let logits = self.dec.head(&self.dgpu, &hidden);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or("moondream3: empty logits row")?;
            if Some(next) == eos {
                break;
            }
            out.push(next);
            hidden = self.dec.decode_step(&self.dgpu, &caches, next, pos);
            pos += 1;
        }
        Ok(out)
    }

    /// The `[seq_len, vocab]` logits for `tokens` with `img_embeds` spliced at rows
    /// `[1, 1+n_img)`. `tokens` must be exactly `seq_len` long (pad past the real
    /// content; see [`Self::generate`] for why the padding cannot affect the row
    /// that is read).
    pub fn logits(&self, tokens: &[u32], img_embeds: &[f32]) -> Vec<f32> {
        self.dec.logits_all(&self.dgpu, tokens, img_embeds)
    }
}

/// Build the decoder block stack (dense below `moe.start_layer`, MoE at and above)
/// from the prefixed decoder weights.
fn build_blocks(gpu: &Gpu, cfg: &MoondreamConfig, dweights: &HashMap<String, Vec<f32>>, t: u32, precision: Precision) -> Vec<MoondreamBlock> {
    (0..cfg.n_layers)
        .map(|l| {
            let bw = strip_prefix(dweights, &format!("blocks.{l}."));
            let blk = MoondreamBlock::new(gpu, &bw, t, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.ff_dim, cfg.prefix_attn, cfg.rot_dim, cfg.rope_theta);
            if !cfg.is_moe_layer(l) {
                return blk;
            }
            let mw = strip_prefix(dweights, &format!("blocks.{l}.moe."));
            let (d, inner, e, k) = (cfg.dim, cfg.moe.inner_dim, cfg.moe.num_experts, cfg.moe.top_k);
            match precision {
                Precision::Fp32 => blk.with_moe(MoeFfn::new(gpu, &mw, t, d, inner, e, k)),
                Precision::Int8 => blk.with_moe8(MoeFfn8::new(gpu, &mw, t, d, inner, e, k)),
            }
        })
        .collect()
}

/// Extract the entries of `w` whose key starts with `prefix`, with the prefix
/// stripped (and dropping any deeper `moe.` sub-keys for the block-level map).
fn strip_prefix(w: &HashMap<String, Vec<f32>>, prefix: &str) -> HashMap<String, Vec<f32>> {
    w.iter()
        .filter_map(|(k, v)| k.strip_prefix(prefix).map(|s| (s.to_string(), v.clone())))
        .filter(|(k, _)| !k.starts_with("moe.")) // block map excludes the MoE sub-tree
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MoeConfig, VisionConfig};
    use crate::decoder::IGNORE;
    use data::rng::Rng;

    /// A named tensor map, as the model's constructors take them.
    type Weights = HashMap<String, Vec<f32>>;
    /// `(config, vision weights, connector weights, decoder weights)`.
    type TinyFixture = (MoondreamConfig, Weights, Weights, Weights);

    /// The tiny config the tests below build, plus its weights.
    fn tiny(vision_dim: u32, conn_in: u32, tau: bool, seed: u64) -> TinyFixture {
        let vision = VisionConfig { dim: vision_dim, patch: 2, n_layers: 2, ff_dim: 2 * vision_dim, n_heads: 2, crop_size: 8, max_crops: 4, overlap_margin: 1 };
        let cfg = MoondreamConfig {
            dim: 24,
            ff_dim: 48,
            n_layers: 2,
            vocab: 23,
            n_heads: 3,
            head_dim: 8,
            prefix_attn: 17, // 1 bos + 16 image
            rot_dim: 4,
            rope_theta: 1.5e6,
            proj_inner: 48,
            proj_out: 24,
            vision: vision.clone(),
            moe: MoeConfig { num_experts: 3, start_layer: 1, top_k: 2, inner_dim: 8 },
        };
        let ppc = vision.patches_per_crop();
        let c = vision.dim as usize;
        let pv = vision.patch_vec() as usize;
        let mut rng = Rng::new(seed);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();

        let mut vw = HashMap::new();
        vw.insert("patch_emb.weight".into(), r(c * pv));
        vw.insert("patch_emb.bias".into(), r(c));
        vw.insert("pos_emb".into(), r(ppc as usize * c));
        vw.insert("post_ln.weight".into(), vec![1.0; c]);
        vw.insert("post_ln.bias".into(), r(c));
        for b in 0..vision.n_layers {
            for (leaf, sz) in [
                ("ln1.weight", c), ("ln1.bias", c), ("attn.qkv.weight", 3 * c * c), ("attn.qkv.bias", 3 * c),
                ("attn.proj.weight", c * c), ("attn.proj.bias", c), ("ln2.weight", c), ("ln2.bias", c),
                ("mlp.fc1.weight", vision.ff_dim as usize * c), ("mlp.fc1.bias", vision.ff_dim as usize),
                ("mlp.fc2.weight", c * vision.ff_dim as usize), ("mlp.fc2.bias", c),
            ] {
                let v = if leaf.ends_with("ln1.weight") || leaf.ends_with("ln2.weight") { vec![1.0; sz] } else { r(sz) };
                vw.insert(format!("blocks.{b}.{leaf}"), v);
            }
        }

        let mut cw = HashMap::new();
        cw.insert("fc1.weight".into(), r((cfg.proj_inner * conn_in) as usize));
        cw.insert("fc1.bias".into(), r(cfg.proj_inner as usize));
        cw.insert("fc2.weight".into(), r((cfg.proj_out * cfg.proj_inner) as usize));
        cw.insert("fc2.bias".into(), r(cfg.proj_out as usize));

        let (d, ff, vocab) = (cfg.dim, cfg.ff_dim, cfg.vocab);
        let mut dw = HashMap::new();
        dw.insert("tok.weight".into(), r((vocab * d) as usize));
        dw.insert("post_ln.weight".into(), vec![1.0; d as usize]);
        dw.insert("post_ln.bias".into(), r(d as usize));
        dw.insert("lm_head.weight".into(), r((vocab * d) as usize));
        dw.insert("lm_head.bias".into(), r(vocab as usize));
        for l in 0..cfg.n_layers {
            let mut leaves: Vec<(&str, usize)> = vec![
                ("ln.weight", d as usize), ("ln.bias", d as usize), ("attn.qkv.weight", (3 * d * d) as usize),
                ("attn.proj.weight", (d * d) as usize), ("attn.proj.bias", d as usize),
                ("mlp.fc1.weight", (ff * d) as usize), ("mlp.fc1.bias", ff as usize),
                ("mlp.fc2.weight", (d * ff) as usize), ("mlp.fc2.bias", d as usize),
            ];
            if tau {
                leaves.extend([
                    ("attn.tau.wq", (cfg.n_heads * 3 * d) as usize),
                    ("attn.tau.wv", (cfg.n_heads * 3 * d) as usize),
                    ("attn.tau.alpha", cfg.n_heads as usize),
                ]);
            }
            for (leaf, sz) in leaves {
                let v = if leaf.ends_with("ln.weight") { vec![1.0; sz] } else { r(sz) };
                dw.insert(format!("blocks.{l}.{leaf}"), v);
            }
            if cfg.is_moe_layer(l) {
                let (inner, e) = (cfg.moe.inner_dim, cfg.moe.num_experts);
                dw.insert(format!("blocks.{l}.moe.router.weight"), r((e * d) as usize));
                for ei in 0..e {
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_h.weight"), r((inner * d) as usize));
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_g.weight"), r((inner * d) as usize));
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_down.weight"), r((d * inner) as usize));
                }
            }
        }
        (cfg, vw, cw, dw)
    }

    /// `tokens`/`targets` for a `1 + ppc + 3` sequence with the bos + image rows
    /// unsupervised.
    fn seq(ppc: u32) -> (Vec<u32>, Vec<u32>) {
        let mut tokens = vec![0u32];
        tokens.extend(std::iter::repeat_n(5u32, ppc as usize));
        tokens.extend([7u32, 9, 11]);
        let mut targets = tokens[1..].to_vec();
        targets.push(13);
        for tg in targets.iter_mut().take(1 + ppc as usize) {
            *tg = IGNORE;
        }
        (tokens, targets)
    }

    #[test]
    fn end_to_end_image_to_loss() {
        // Tiny everything: ViT 2×2-patch dim 32 → connector 32→48→24 → decoder d24.
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 2);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 3;
        let mut rng = Rng::new(21);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();

        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let (tokens, targets) = seq(ppc);
        let loss = model.forward(&tokens, &targets, &packed);
        assert!(loss.is_finite() && loss > 0.0, "moondream end-to-end loss must be finite+positive, got {loss}");
    }

    #[test]
    fn multicrop_forward_with_tau_is_finite() {
        // Faithful path: global + 2×2 local crops → [ppc,2·dim] concat → connector.
        let (cfg, vw, cw, dw) = tiny(16, 32, true, 4);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 3;
        let (ht, wt) = (2u32, 2u32);
        let mut rng = Rng::new(41);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let global = r((ppc * vision.patch_vec()) as usize);
        let locals = r((ht * wt * ppc * vision.patch_vec()) as usize);

        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, 2 * vision.dim, seq_len);
        let (tokens, targets) = seq(ppc);
        let loss = model.forward_multicrop(&tokens, &targets, &global, &locals, ht, wt);
        assert!(loss.is_finite() && loss > 0.0, "multi-crop loss must be finite+positive, got {loss}");
    }

    /// Greedy decode runs, is deterministic, and stops at `eos`.
    ///
    /// The padding argument this loop rests on is checked here too: the same
    /// prompt decoded twice must give the same ids (nothing in the padding
    /// slots leaks into the read row), and a `max_new` cap must be honoured.
    #[test]
    fn generate_is_deterministic_and_honours_max_new() {
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 13);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let vocab = cfg.vocab;
        // Room for the bos + image block + a few prompt tokens + generation.
        let seq_len = 1 + ppc + 8;
        let mut rng = Rng::new(130);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();

        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let embeds = model.image_embeds(&packed);
        let mut prompt = vec![0u32];
        prompt.extend(std::iter::repeat_n(5u32, ppc as usize));
        prompt.push(7);

        let a = model.generate(&prompt, &embeds, 4, None).expect("generate runs");
        let b = model.generate(&prompt, &embeds, 4, None).expect("generate runs");
        assert_eq!(a, b, "greedy decode must be deterministic");
        assert_eq!(a.len(), 4, "max_new must cap the run");
        assert!(a.iter().all(|&id| id < vocab), "every id must be in vocab: {a:?}");
    }

    /// `eos` ends the run. Fed its own first output as the stop id, the loop
    /// must return nothing at all rather than one token then stopping.
    #[test]
    fn generate_stops_at_eos() {
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 17);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 8;
        let mut rng = Rng::new(170);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();

        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let embeds = model.image_embeds(&packed);
        let mut prompt = vec![0u32];
        prompt.extend(std::iter::repeat_n(5u32, ppc as usize));
        prompt.push(7);

        let free = model.generate(&prompt, &embeds, 3, None).expect("generate runs");
        let stopped = model.generate(&prompt, &embeds, 3, Some(free[0])).expect("generate runs");
        assert!(stopped.is_empty(), "eos on the first sampled id must yield no output, got {stopped:?}");
    }

    /// THE GATE FOR THE KV PATH: incremental decode must produce exactly what
    /// the `O(T²)` recompute produces.
    ///
    /// This is the only check that makes preferring the fast path safe. The two
    /// share no code below `generate`: the recompute path re-runs the whole
    /// masked batched graph per token, while the KV path prefills once and then
    /// takes a single-row step with `rope_partial_at`, a cache-attending
    /// `gqa_decode_step` and NO mask kernel. Every one of those substitutions
    /// runs and returns plausible ids when wrong - rotating the new row at
    /// position 0 instead of `pos`, folding `tau_pos` at the wrong index,
    /// seeding the cache from the pre-RoPE `qkv` instead of the post-RoPE one.
    /// Token-for-token agreement catches all of them; nothing weaker does.
    #[test]
    fn kv_decode_matches_the_recompute_path_token_for_token() {
        let (cfg, vw, cw, dw) = tiny(32, 32, true, 41);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 10;
        let mut rng = Rng::new(410);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();

        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let embeds = model.image_embeds(&packed);
        let mut prompt = vec![0u32];
        prompt.extend(std::iter::repeat_n(5u32, ppc as usize));
        prompt.extend([7u32, 9]);

        let slow = model.generate(&prompt, &embeds, 5, None).expect("recompute path");
        let fast = model.generate_kv(&prompt, &embeds, 5, None).expect("kv path");
        assert_eq!(slow, fast, "the KV path must reproduce the recompute path exactly");
        assert_eq!(slow.len(), 5, "the comparison would be vacuous on an empty run");
    }

    /// The same agreement with tau OFF, so a defect in the tau fold cannot be
    /// what makes the two paths agree.
    #[test]
    fn kv_decode_matches_the_recompute_path_without_tau() {
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 43);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 8;
        let mut rng = Rng::new(430);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();
        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let embeds = model.image_embeds(&packed);
        let mut prompt = vec![0u32];
        prompt.extend(std::iter::repeat_n(5u32, ppc as usize));
        prompt.push(7);
        assert_eq!(
            model.generate(&prompt, &embeds, 4, None).unwrap(),
            model.generate_kv(&prompt, &embeds, 4, None).unwrap()
        );
    }

    /// THE GATE FOR BATCHING: N images through one ViT pass must give exactly
    /// what N separate passes give.
    ///
    /// The failure this catches is a slicing bug, and it is invisible to a
    /// shape check: crops-per-request VARIES with each image's aspect ratio, so
    /// an offset that assumes a uniform stride hands request 2 some of request
    /// 1's crops. Same length, finite values, plausible embeddings, wrong
    /// picture. Two DELIBERATELY different aspect ratios here for that reason.
    #[test]
    fn batched_vision_matches_one_image_at_a_time() {
        let (cfg, vw, cw, dw) = tiny(16, 32, false, 47);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 3;
        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, 2 * vision.dim, seq_len);

        let mut rng = Rng::new(470);
        let mut img = |w: u32, h: u32| -> Vec<f32> { (0..(w * h * 3) as usize).map(|_| rng.next_f32()).collect() };
        // Different aspect ratios -> different tile counts -> different crop
        // counts, which is exactly what a uniform-stride slice gets wrong.
        let (a, b) = (img(13, 9), img(9, 21));
        let batch = model.image_embeds_from_pixels_batch(&[(&a, 13, 9), (&b, 9, 21)]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], model.image_embeds_from_pixels(&a, 13, 9), "request 0 differs when batched");
        assert_eq!(batch[1], model.image_embeds_from_pixels(&b, 9, 21), "request 1 differs when batched");
        assert_ne!(batch[0], batch[1], "two different images must not produce the same embedding");
    }

    /// A prompt that does not fit the built graph is a named error, not a panic
    /// deep in a buffer write.
    #[test]
    fn generate_refuses_a_prompt_longer_than_the_built_context() {
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 19);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 2;
        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let too_long = vec![1u32; seq_len as usize + 1];
        let err = model.generate(&too_long, &[], 1, None).unwrap_err();
        assert!(err.contains("seq_len"), "{err}");
    }

    /// THE POINT OF `Precision::Int8`: the whole composite still computes the
    /// same function, end to end, with quantized experts AND one activation set
    /// shared by every block.
    ///
    /// Two independent things could break here and both would still produce
    /// finite, plausible logits. The int8 experts could mis-scale (checked more
    /// tightly in `decoder`'s own `int8_experts_agree_with_the_fp32_tier`), and
    /// the shared scratch could let one block read the previous block's
    /// activations - which is a real risk precisely because the buffers are the
    /// right SHAPE, so nothing errors. Comparing full logits against the fp32,
    /// per-block-scratch build is what catches the second.
    #[test]
    fn int8_with_shared_scratch_matches_the_fp32_build_end_to_end() {
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 23);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 3;
        let vocab = cfg.vocab as usize;
        let mut rng = Rng::new(230);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();
        let (tokens, _) = seq(ppc);

        let logits_at = |p: Precision| {
            let m = MoondreamModel::new_cpu_with(cfg.clone(), vw.clone(), cw.clone(), dw.clone(), vision.dim, seq_len, p);
            assert_eq!(m.precision(), p);
            let embeds = m.image_embeds(&packed);
            m.logits(&tokens, &embeds)
        };
        let a = logits_at(Precision::Fp32);
        let b = logits_at(Precision::Int8);
        assert_eq!(a.len(), seq_len as usize * vocab);
        assert_eq!(a.len(), b.len());
        assert!(b.iter().all(|v| v.is_finite()), "the int8 build produced a non-finite logit");

        let dot: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb).max(1e-30);
        // A lossy tier, so this floor catches a BROKEN build (cross-block
        // scratch bleed, a mis-scaled expert), not a claim about accuracy.
        assert!(cos > 0.99, "int8 + shared scratch diverges from fp32 end to end: cosine {cos:.6}");
    }

    /// A block on shared scratch cannot be differentiated, and says so rather
    /// than quietly producing a gradient against another block's activations.
    #[test]
    fn a_shared_scratch_build_refuses_to_run_backward() {
        use crate::decoder::{pipelines, MoondreamBlock};
        let gpu = gpu_core::Gpu::new_cpu(pipelines());
        let (t, d, nh, hd, ff) = (4u32, 8u32, 2u32, 4u32, 16u32);
        let mut rng = Rng::new(31);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        for (leaf, sz) in [
            ("ln.weight", d as usize), ("ln.bias", d as usize), ("attn.qkv.weight", (3 * d * d) as usize),
            ("attn.proj.weight", (d * d) as usize), ("attn.proj.bias", d as usize),
            ("mlp.fc1.weight", (ff * d) as usize), ("mlp.fc1.bias", ff as usize),
            ("mlp.fc2.weight", (d * ff) as usize), ("mlp.fc2.bias", d as usize),
        ] {
            w.insert(leaf.to_string(), if leaf.ends_with("ln.weight") { vec![1.0; sz] } else { r(sz) });
        }
        let blk = MoondreamBlock::new(&gpu, &w, t, d, nh, hd, ff, 1, 4, 1.5e6).without_scratch();
        let x = gpu.storage_init("x", &r((t * d) as usize));
        let gr = crate::decoder::MoondreamBlockGrads::new(&gpu, d, ff);
        let d_out = gpu.storage((t * d) as u64);
        let d_x = gpu.storage((t * d) as u64);
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| blk.backward(&gpu, &x, &d_out, &gr, &d_x)));
        assert!(err.is_err(), "backward on a shared-scratch block must refuse");
    }

    /// THE POINT OF OWNING THE STACK: two forwards on ONE model, with no rebuild
    /// between them. Before this refactor the composite re-uploaded every weight
    /// per call, so a resident was impossible; a second call that agrees with the
    /// first is what says the built graph is reusable.
    #[test]
    fn a_second_forward_reuses_the_built_stack_and_agrees() {
        let (cfg, vw, cw, dw) = tiny(32, 32, false, 7);
        let vision = cfg.vision.clone();
        let ppc = vision.patches_per_crop();
        let seq_len = 1 + ppc + 3;
        let mut rng = Rng::new(70);
        let packed: Vec<f32> = (0..(ppc * vision.patch_vec()) as usize).map(|_| (rng.next_f32() - 0.5) * 0.2).collect();

        let model = MoondreamModel::new_cpu(cfg, vw, cw, dw, vision.dim, seq_len);
        let (tokens, targets) = seq(ppc);
        let a = model.forward(&tokens, &targets, &packed);
        let b = model.forward(&tokens, &targets, &packed);
        assert_eq!(a.to_bits(), b.to_bits(), "a reused stack must be bit-identical, got {a} then {b}");
    }
}
