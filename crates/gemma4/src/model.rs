// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The full tiny Gemma-4 text tower forward, host-orchestrated over
//! [`crate::block::Gemma4Layer`] - `Gemma4UnifiedTextModel.forward`'s
//! text-only path (`transformers.models.gemma4_unified.
//! modeling_gemma4_unified`).
//!
//! ## What runs where, and why
//!
//! The token embedding gather+scale and the final `norm` are plain host math
//! (a single `[T, hidden]`-scale pass each - not worth a device round trip),
//! mirroring `ltxv::dit`'s own split ("everything outside the block stack
//! runs as plain host math"). Only each decoder layer's internals go through
//! the GPU dispatch graph in [`crate::block`].
//!
//! ## The `hidden_states` convention (pinned against source, not assumed -
//! ## see `tools/goldens/gemma4_dump_reference.py`'s module doc for the
//! ## empirical verification this mirrors)
//!
//! [`Gemma4Output::hidden_states`] has exactly `num_hidden_layers + 1`
//! entries: entry `0` is the embedding output, entries `1..num_hidden_layers`
//! are each layer's RAW output (layer `0`'s output at index `1`, ..., layer
//! `N-2`'s output at index `N-1`), and the LAST entry (index `N`) is
//! `norm(layer N-1's raw output)` - i.e. bit-identical to
//! [`Gemma4Output::last_hidden_state`], NOT layer `N-1`'s raw pre-norm
//! output. This is what the real LTX-2.5 `text_embedding_projection.
//! {video,audio}_aggregate_embed` (`Linear(hidden*49 -> ...)`) consumes for
//! the real 48-layer config, and is why `188160 = 3840*49` lines up exactly:
//! 1 embedding + 47 raw intermediate outputs + 1 post-final-norm output,
//! never a 48th raw layer output.

use gpu_core::Gpu;

use crate::block::{Gemma4Layer, Precision, Tensors, KERNELS};
use crate::config::{Gemma4Config, LayerType};
use crate::rope::{full_table, sliding_table, upload_rope};

/// Load a flat `name -> (shape, data)` weight map from a safetensors file -
/// no renaming (the golden's own tensor names ARE this crate's canonical
/// name space, see `crate::block`'s `tget` calls). Mirrors `ltxv::dit::
/// load_tiny_weights` exactly.
pub fn load_tiny_weights(path: &str) -> Tensors {
    let raw = checkpoint::safetensors::read(path).unwrap_or_else(|e| panic!("gemma4: {e}"));
    raw.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

/// Plain `nn.Embedding` gather, scaled by `sqrt(hidden_size)` -
/// `Gemma4UnifiedTextScaledWordEmbedding.forward`. Host math (see this
/// module's doc): a single `[T, hidden]` pass, not worth a device round trip.
fn embed(input_ids: &[u32], table: &[f32], hidden: usize) -> Vec<f32> {
    let scale = (hidden as f32).sqrt();
    let mut out = vec![0f32; input_ids.len() * hidden];
    for (i, &tok) in input_ids.iter().enumerate() {
        let row = &table[tok as usize * hidden..tok as usize * hidden + hidden];
        for d in 0..hidden {
            out[i * hidden + d] = row[d] * scale;
        }
    }
    out
}

/// `Gemma4RMSNorm(hidden_size, eps)` (`with_scale=True`) - the model's final
/// `norm`. Host math for the same reason [`embed`] is.
fn rmsnorm_host(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..r * dim + dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = w[d] * row[d] * inv;
        }
    }
    out
}

/// Every tap a parity test bisects with - the golden's own tensor names.
pub struct Gemma4Output {
    /// `num_hidden_layers + 1` entries - see this module's doc for the
    /// EXACT (not "all raw") convention.
    pub hidden_states: Vec<Vec<f32>>,
    pub last_hidden_state: Vec<f32>,
    pub layer0_self_attn_out: Vec<f32>,
    pub layer_last_self_attn_out: Vec<f32>,
    pub rope_sliding_cos: Vec<f32>,
    pub rope_sliding_sin: Vec<f32>,
    pub rope_full_cos: Vec<f32>,
    pub rope_full_sin: Vec<f32>,
}

pub struct Gemma4Model {
    cfg: Gemma4Config,
    w: Tensors,
    device: Option<String>,
}

impl Gemma4Model {
    pub fn new(cfg: Gemma4Config, weights: Tensors, device: Option<&str>) -> Gemma4Model {
        Gemma4Model { cfg, w: weights, device: device.map(str::to_string) }
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.cfg
    }

    /// One forward pass over `input_ids` (`[T]`, real token ids - NOT
    /// pre-embedded, unlike `ltxv::LtxDit::forward`'s `latent` input, since
    /// this crate owns the embedding table too).
    ///
    /// Weights come from the eager whole-model map this was constructed
    /// with, in fp32. [`forward_streamed`] is the same forward over a
    /// checkpoint read one layer at a time.
    pub fn forward(&self, input_ids: &[u32]) -> Gemma4Output {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size as usize;
        let embed_table = &self.w.get("embed_tokens.weight").unwrap_or_else(|| panic!("gemma4: missing embed_tokens.weight")).1;
        let embed_out = embed(input_ids, embed_table, hidden);
        let norm_w = self.w.get("norm.weight").unwrap_or_else(|| panic!("gemma4: missing norm.weight")).1.clone();
        forward_core(cfg, self.device.as_deref(), input_ids.len() as u32, embed_out, &norm_w, |gpu, l| {
            Gemma4Layer::on(gpu.share(), cfg, &self.w, l, Precision::Fp32)
        })
    }
}

/// The layer stack, shared by every way of getting weights into it.
///
/// There is exactly ONE of these on purpose. A model with two forward
/// implementations for the same architecture - a reference one and a fast
/// one - reliably grows a feature in one that is silently missing from the
/// other, and nothing type-checks the gap. Here the only thing that varies
/// is `build_layer`; the embedding, both RoPE tables, the `hidden_states`
/// convention and the final norm are literally the same code for the eager
/// fp32 path and the streamed int8 one.
fn forward_core(
    cfg: &Gemma4Config,
    device: Option<&str>,
    t: u32,
    embed_out: Vec<f32>,
    norm_w: &[f32],
    mut build_layer: impl FnMut(&Gpu, u32) -> Gemma4Layer,
) -> Gemma4Output {
    let hidden = cfg.hidden_size as usize;
    let n = cfg.num_hidden_layers;

    // Both RoPE tables, built once and shared by every layer of that
    // type - see `crate::rope`'s doc for why BOTH fit the existing
    // `rope2d` kernel unchanged (the `full_attention` table carries its
    // own zero-padded identity columns; `rope2d_partial` is refuted).
    let sliding_tbl = sliding_table(cfg.head_dim, cfg.rope_theta_sliding, t as usize);
    let full_tbl = full_table(cfg.global_head_dim, cfg.rope_theta_full, cfg.partial_rotary_factor, t as usize);

    let gpu: Gpu = Gpu::open(device, &KERNELS);
    let sliding_rope = upload_rope(&gpu, &sliding_tbl);
    let full_rope = upload_rope(&gpu, &full_tbl);

    let mut x = embed_out.clone();
    let mut raw_outputs: Vec<Vec<f32>> = Vec::with_capacity(n as usize);
    let mut layer0_self_attn_out = Vec::new();
    let mut layer_last_self_attn_out = Vec::new();

    for l in 0..n {
        let t_layer = std::time::Instant::now();
        let layer = build_layer(&gpu, l);
        let build_ms = t_layer.elapsed().as_secs_f32() * 1000.0;
        let lt = layer.layer_type();
        let rope = match lt {
            LayerType::Sliding => &sliding_rope,
            LayerType::Full => &full_rope,
        };
        let t_fwd = std::time::Instant::now();
        let (out, attn_out) = layer.forward(&x, rope, t);
        tracing::trace!(layer = l, kind = ?lt, build_ms, gpu_ms = t_fwd.elapsed().as_secs_f32() * 1000.0, "gemma4 layer done");
        if l == 0 {
            layer0_self_attn_out = attn_out.clone();
        }
        if l == n - 1 {
            layer_last_self_attn_out = attn_out;
        }
        x = out;
        raw_outputs.push(x.clone());
    }

    // `hidden_states` per this module's doc: embed, then layers
    // 0..N-2's raw outputs, then the FINAL layer's raw output run
    // through the model's own `norm` - not raw.
    let last_hidden_state = rmsnorm_host(&raw_outputs[n as usize - 1], norm_w, t as usize, hidden, cfg.rms_norm_eps);

    let mut hidden_states = Vec::with_capacity(n as usize + 1);
    hidden_states.push(embed_out);
    for raw in raw_outputs.iter().take(n as usize - 1) {
        hidden_states.push(raw.clone());
    }
    hidden_states.push(last_hidden_state.clone());

    Gemma4Output {
        hidden_states,
        last_hidden_state,
        layer0_self_attn_out,
        layer_last_self_attn_out,
        rope_sliding_cos: sliding_tbl.cos,
        rope_sliding_sin: sliding_tbl.sin,
        rope_full_cos: full_tbl.cos,
        rope_full_sin: full_tbl.sin,
    }
}

/// Every tensor name layer `l` owns, from the manifest itself rather than a
/// second hand-written list - so a layer-shape change (a `full_attention`
/// layer's absent `v_proj`, say) cannot be right in one place and wrong here.
pub fn layer_tensor_names(cfg: &Gemma4Config, l: u32) -> Vec<(String, Vec<usize>)> {
    let prefix = format!("layers.{l}.");
    crate::import::gemma4_tensor_manifest(cfg).into_iter().filter(|(n, _)| n.starts_with(&prefix)).collect()
}

/// Pull one layer's tensors out of `src`. Errors by name on anything absent
/// or the wrong length - never a zero fill, matching the importer's own
/// contract for the eager path.
pub fn load_layer_tensors(src: &dyn checkpoint::TensorSource, cfg: &Gemma4Config, l: u32) -> Result<Tensors, String> {
    let mut out = Tensors::new();
    for (name, shape) in layer_tensor_names(cfg, l) {
        let want: usize = shape.iter().product();
        let mut got: Option<Vec<f32>> = None;
        let found = src.with_tensor(&name, &mut |d| got = Some(d.to_vec()));
        let data = match (found, got) {
            (true, Some(d)) => d,
            _ => return Err(format!("gemma4: source has no tensor {name}")),
        };
        if data.len() != want {
            return Err(format!("gemma4: {name} has {} values, expected {want}", data.len()));
        }
        out.insert(name, (shape, data));
    }
    Ok(out)
}

/// [`Gemma4Model::forward`] over a checkpoint read on demand, with the
/// projections resident at `requested` precision.
///
/// This is what makes the real 12B tower runnable without first expanding it:
/// only the embedding table, the final norm and ONE layer are host-resident
/// at a time, instead of the whole model as f32. `requested` is a request -
/// [`Precision::for_device`] resolves it against what the device can actually
/// execute, so asking for int8 on a device with no packed-dot path runs fp32
/// and says so rather than quantizing for nothing.
pub fn forward_streamed(
    cfg: &Gemma4Config,
    src: &dyn checkpoint::TensorSource,
    device: Option<&str>,
    requested: Precision,
    input_ids: &[u32],
) -> Result<Gemma4Output, String> {
    let hidden = cfg.hidden_size as usize;

    let mut embed_out = None;
    if !src.with_tensor("embed_tokens.weight", &mut |table| embed_out = Some(embed(input_ids, table, hidden))) {
        return Err("gemma4: source has no tensor embed_tokens.weight".to_string());
    }
    let embed_out = embed_out.expect("with_tensor reported found, so the callback ran");

    let mut norm_w = None;
    if !src.with_tensor("norm.weight", &mut |d| norm_w = Some(d.to_vec())) {
        return Err("gemma4: source has no tensor norm.weight".to_string());
    }
    let norm_w = norm_w.expect("with_tensor reported found, so the callback ran");

    tracing::info!(?requested, layers = cfg.num_hidden_layers, tokens = input_ids.len(), "gemma4 streamed forward");

    // Resolved ONCE, on the device the forward is actually running on -
    // never on a throwaway handle opened just to ask. Opening a device is
    // not free (it compiles this crate's whole pipeline set) and device
    // lifecycle in this workspace has its own history, so the capability
    // query rides the handle `forward_core` already made rather than
    // creating a second one. `Cell` because `build_layer` is `FnMut` and
    // this is set on its first call.
    let resolved: std::cell::Cell<Option<Precision>> = std::cell::Cell::new(None);
    let mut err: Option<String> = None;
    let out = forward_core(cfg, device, input_ids.len() as u32, embed_out, &norm_w, |gpu, l| {
        let precision = match resolved.get() {
            Some(p) => p,
            None => {
                let p = Precision::for_device(gpu, requested);
                tracing::info!(?requested, resolved = ?p, "gemma4 precision resolved against the device");
                resolved.set(Some(p));
                p
            }
        };
        let t = load_layer_tensors(src, cfg, l).unwrap_or_else(|e| {
            err = Some(e);
            Tensors::new()
        });
        Gemma4Layer::on(gpu.share(), cfg, &t, l, precision)
    });
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// LTX's own `text_embedding_projection.{video,audio}_aggregate_embed` -
/// `Linear(hidden*(num_hidden_layers+1) -> out_dim)` over the token-wise
/// concatenation of the FULL `hidden_states` tuple (see this module's doc for
/// its exact per-entry semantics), with the reference's per-token, per-layer
/// RMS normalization and rescale applied to that concatenation FIRST - see
/// [`AggregateEmbed::forward`]. Not an HF class: LTX's own addition on top of
/// the Gemma4Unified text tower, implemented in
/// `ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2`
/// (`resources/ltxv/source/packages/ltx-core/src/ltx_core/text_encoders/
/// gemma/feature_extractor.py`).
pub struct AggregateEmbed {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl AggregateEmbed {
    /// `weight`: `[out_dim, in_dim]` row-major (`nn.Linear` layout), `bias`:
    /// `[out_dim]`.
    pub fn new(weight: Vec<f32>, bias: Vec<f32>, in_dim: usize, out_dim: usize) -> AggregateEmbed {
        assert_eq!(weight.len(), in_dim * out_dim);
        assert_eq!(bias.len(), out_dim);
        AggregateEmbed { weight, bias, in_dim, out_dim }
    }

    /// Loads `text_embedding_projection.{prefix}_aggregate_embed.{weight,
    /// bias}` from a [`Tensors`] map - `hidden`/`n_states` derive `in_dim`.
    /// Shared by [`Self::from_weights`] (`prefix="video"`, out_dim 4096) and
    /// [`Self::from_weights_audio`] (`prefix="audio"`, out_dim 2048) - the
    /// two heads are identical apart from which checkpoint tensors they
    /// read (`crate::import::VIDEO_AGGREGATE_OUT_DIM`/
    /// `AUDIO_AGGREGATE_OUT_DIM`), so this generalizes the original
    /// video-only function rather than duplicating its body.
    fn from_weights_prefixed(w: &Tensors, hidden: usize, n_states: usize, prefix: &str) -> AggregateEmbed {
        let weight = w
            .get(&format!("text_embedding_projection.{prefix}_aggregate_embed.weight"))
            .unwrap_or_else(|| panic!("gemma4: missing {prefix} aggregate-embed weight"))
            .1
            .clone();
        let bias = w
            .get(&format!("text_embedding_projection.{prefix}_aggregate_embed.bias"))
            .unwrap_or_else(|| panic!("gemma4: missing {prefix} aggregate-embed bias"))
            .1
            .clone();
        let out_dim = bias.len();
        AggregateEmbed::new(weight, bias, hidden * n_states, out_dim)
    }

    /// Loads `text_embedding_projection.video_aggregate_embed.{weight,bias}`
    /// from a [`Tensors`] map - `hidden`/`n_states` derive `in_dim`.
    pub fn from_weights(w: &Tensors, hidden: usize, n_states: usize) -> AggregateEmbed {
        Self::from_weights_prefixed(w, hidden, n_states, "video")
    }

    /// Loads `text_embedding_projection.audio_aggregate_embed.{weight,bias}`.
    /// [`Self::from_weights`]'s audio twin (out_dim 2048 vs video's 4096,
    /// everything else - including [`Self::forward`]'s `hidden_states`
    /// concatenation convention - identical, see `crate::model`'s doc for
    /// the exact per-entry `hidden_states` semantics both heads consume).
    pub fn from_weights_audio(w: &Tensors, hidden: usize, n_states: usize) -> AggregateEmbed {
        Self::from_weights_prefixed(w, hidden, n_states, "audio")
    }

    /// `hidden_states`: the FULL tuple (length `n_states`, each `[t,
    /// hidden]`) - see this module's doc for the exact per-entry semantics
    /// this concatenation assumes. Plain host math (T is tiny; see this
    /// module's doc for why the outer stage stays on the host).
    /// At the real config this is `Linear(188160 -> 4096)`: 770 million
    /// multiply-accumulates PER TOKEN. A scalar triple loop over that is not
    /// "host glue that is too small to be worth a device round trip" - it is
    /// one of the largest single matrix products in the whole text-encode
    /// stage, and it ran on one core. `hostmath::matvec_par` is the
    /// workspace's shared parallel matrix-vector product (the same one
    /// `ltxv::dit`'s host GEMMs use) and is called directly rather than
    /// wrapped, so the row-splitting exists once.
    ///
    /// # The normalization and the column order are part of the projection
    ///
    /// The linear is fed `norm_and_concat_per_token_rms` followed by
    /// `_rescale_norm` - `ltx_core.text_encoders.gemma.feature_extractor.
    /// FeatureExtractorV2.forward`, the module the real 22B checkpoint's
    /// `text_embedding_projection` lives inside:
    ///
    /// 1. every `(token, state)` slice of the `[t, hidden, n_states]` stack
    ///    is scaled to unit RMS over the `hidden` axis,
    ///    `x * rsqrt(mean(x^2) + 1e-6)` - no learned weight, no mean
    ///    subtraction, and INDEPENDENTLY per state, which is the whole point;
    /// 2. that `[t, hidden, n_states]` stack is flattened as-is
    ///    (`torch.stack(hidden_states, dim=-1)` then `.reshape(B, T, D*L)`),
    ///    so input column `d*n_states + k` is state `k`'s coordinate `d` -
    ///    the states are INTERLEAVED per hidden dimension, NOT laid out as
    ///    `n_states` contiguous `hidden`-wide blocks;
    /// 3. the result is multiplied by `sqrt(out_dim / hidden)`
    ///    (`_rescale_norm(x, out_features, embedding_dim)`, where
    ///    `embedding_dim` is the text tower's own `hidden_size` - the
    ///    `hidden` argument here).
    ///
    /// All three were wrong for as long as this function existed, on the
    /// documented (and wrong) premise that a checkpoint header's tensor
    /// shape was all there was to derive the module from. Neither error is
    /// one a downstream norm absorbs. The wrong column order permutes the
    /// weight matrix outright, and the 49 raw Gemma states differ in
    /// magnitude by orders of magnitude, so an un-normalized flatten - and
    /// therefore every row of the DiT's text context - was dominated by the
    /// largest states' near-constant component, leaving the caption's own
    /// content as a tiny residual. Two entirely different prompts produced
    /// context matrices whose mean rows sat at cosine 0.9998 of each other,
    /// and a text-to-video generation reproduced the same scene whatever it
    /// was asked for.
    ///
    /// The reference additionally zeroes padded positions here; this function
    /// is only ever handed a prompt's REAL tokens (`crate::pipeline` in
    /// `ltxv` pads afterwards, and the embeddings connector overwrites the
    /// padded tail with its learnable registers anyway), so there is no mask
    /// argument to honor.
    pub fn forward(&self, hidden_states: &[Vec<f32>], t: usize, hidden: usize) -> Vec<f32> {
        let n_states = hidden_states.len();
        assert_eq!(self.in_dim, hidden * n_states, "AggregateEmbed: in_dim {} != hidden*n_states {}", self.in_dim, hidden * n_states);
        // `_rescale_norm(x, target_dim=out_features, source_dim=embedding_dim)`.
        let rescale = (self.out_dim as f32 / hidden as f32).sqrt();
        let mut concat_row = vec![0f32; self.in_dim];
        let mut out = vec![0f32; t * self.out_dim];
        for ti in 0..t {
            for (k, hs) in hidden_states.iter().enumerate() {
                let slice = &hs[ti * hidden..ti * hidden + hidden];
                let ms = slice.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
                let inv = (1.0 / (ms + 1e-6).sqrt()) as f32 * rescale;
                for (d, &v) in slice.iter().enumerate() {
                    concat_row[d * n_states + k] = v * inv;
                }
            }
            let row = model::hostmath::matvec_par(&self.weight, &concat_row, self.out_dim, self.in_dim);
            for (o, v) in row.iter().enumerate() {
                out[ti * self.out_dim + o] = v + self.bias[o];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The audio head loads its OWN `text_embedding_projection.
    /// audio_aggregate_embed.*` tensors (never the video ones, even when
    /// both are present in the same weight map), keeps `in_dim` identical
    /// to the video head's (same `hidden*n_states` concatenation), and
    /// produces the real checkpoint's `out_dim=2048` (vs video's 4096) -
    /// `crate::import::AUDIO_AGGREGATE_OUT_DIM`/`VIDEO_AGGREGATE_OUT_DIM`.
    #[test]
    fn audio_aggregate_embed_loads_its_own_weights_independent_of_video() {
        let hidden = 4usize;
        let n_states = 3usize;
        let in_dim = hidden * n_states;
        let video_out = 5usize;
        let audio_out = 2usize;

        let mut w: Tensors = Tensors::new();
        w.insert("text_embedding_projection.video_aggregate_embed.weight".into(), (vec![video_out, in_dim], vec![1.0; video_out * in_dim]));
        w.insert("text_embedding_projection.video_aggregate_embed.bias".into(), (vec![video_out], vec![0.0; video_out]));
        w.insert("text_embedding_projection.audio_aggregate_embed.weight".into(), (vec![audio_out, in_dim], vec![2.0; audio_out * in_dim]));
        w.insert("text_embedding_projection.audio_aggregate_embed.bias".into(), (vec![audio_out], vec![7.0; audio_out]));

        let video = AggregateEmbed::from_weights(&w, hidden, n_states);
        let audio = AggregateEmbed::from_weights_audio(&w, hidden, n_states);
        assert_eq!(video.out_dim, video_out);
        assert_eq!(audio.out_dim, audio_out);
        assert_eq!(video.in_dim, audio.in_dim);

        let hidden_states: Vec<Vec<f32>> = (0..n_states).map(|_| vec![1.0f32; hidden]).collect();
        let audio_out_vals = audio.forward(&hidden_states, 1, hidden);
        // An all-ones slice is already unit-RMS, so the normalization is the
        // identity here and only the reference's `_rescale_norm` factor
        // `sqrt(out_dim/hidden)` remains: weight=2.0 everywhere, in_dim=12
        // ones summed -> 24.0, scaled by sqrt(2/4), + bias 7.0.
        let want = 24.0 * (audio_out as f32 / hidden as f32).sqrt() + 7.0;
        for v in &audio_out_vals {
            assert!((v - want).abs() < 1e-4, "audio head: {v} != {want}");
        }
        let video_out_vals = video.forward(&hidden_states, 1, hidden);
        assert_ne!(video_out_vals.len(), audio_out_vals.len());
    }

    /// The projection's INPUT is per-token, per-layer RMS-normalized and then
    /// rescaled - `ltx_core.text_encoders.gemma.feature_extractor.
    /// FeatureExtractorV2.forward`'s `norm_and_concat_per_token_rms` followed
    /// by `_rescale_norm(x, out_features, embedding_dim)`. Skipping it is not
    /// a scale detail: raw Gemma hidden states differ in magnitude by orders
    /// of magnitude across the 49 states, so a plain concatenate-then-project
    /// is dominated by whichever state happens to be largest and the
    /// per-token caption content survives only as a few-percent residual on a
    /// near-constant vector - which is what a text-to-video prompt then fails
    /// to steer: two unrelated prompts decoded to the same picture.
    ///
    /// Two properties pin the whole formula:
    ///
    /// * scaling ONE token's ONE layer slice by any positive factor must not
    ///   change that token's output at all (this is what the norm buys, and
    ///   exactly what a plain `Linear` over the raw concatenation lacks);
    /// * the surviving scale is the reference's `sqrt(out_dim/hidden)`, not 1.
    #[test]
    fn the_projection_rms_normalizes_every_layer_slice_and_rescales() {
        let (hidden, n_states, out_dim) = (4usize, 3usize, 2usize);
        let in_dim = hidden * n_states;
        // Row `o` reads exactly one input coordinate: row 0 -> state 0's dim
        // 0, row 1 -> state 1's dim 0. That makes the expected value a closed
        // form rather than a re-implementation of the function under test -
        // and it pins the reference's INTERLEAVED column order at the same
        // time (state `k`, dim `d` lives at `d*n_states + k`, so state 1's
        // dim 0 is column 1, not column `hidden`).
        let mut weight = vec![0f32; out_dim * in_dim];
        weight[0] = 1.0;
        weight[in_dim + 1] = 1.0;
        let agg = AggregateEmbed::new(weight, vec![0.0; out_dim], in_dim, out_dim);

        // Token 0 and token 1 carry the SAME directions; token 1's slices are
        // scaled by 10x (layer 0) and 7x (layer 1) - the spread across states
        // that the real 49-state stack has and that the norm exists to remove.
        let hidden_states = vec![
            vec![3.0, 1.0, 1.0, 1.0, /* token 1 */ 30.0, 10.0, 10.0, 10.0],
            vec![1.0, 1.0, 1.0, 1.0, /* token 1 */ 7.0, 7.0, 7.0, 7.0],
            vec![0.0, 0.0, 0.0, 0.0, /* token 1 */ 0.0, 0.0, 0.0, 0.0],
        ];
        let out = agg.forward(&hidden_states, 2, hidden);

        let rescale = (out_dim as f32 / hidden as f32).sqrt();
        // layer 0, token 0: mean square = (9+1+1+1)/4 = 3, so dim 0 normalizes
        // to 3/sqrt(3) = sqrt(3).
        let want0 = 3.0f32 / 3.0f32.sqrt() * rescale;
        // layer 1, token 0: an all-ones slice is already unit RMS.
        let want1 = 1.0f32 * rescale;
        assert!((out[0] - want0).abs() < 1e-5, "token 0 row 0: {} != {want0}", out[0]);
        assert!((out[1] - want1).abs() < 1e-5, "token 0 row 1: {} != {want1}", out[1]);
        assert!((out[out_dim] - want0).abs() < 1e-5, "a 10x-larger layer slice changed the output: {} != {want0}", out[out_dim]);
        assert!((out[out_dim + 1] - want1).abs() < 1e-5, "a 7x-larger layer slice changed the output: {} != {want1}", out[out_dim + 1]);
    }

    /// An all-zero layer slice has no direction to normalize, and the
    /// reference's `+1e-6` inside the `rsqrt` is what keeps it finite rather
    /// than producing NaNs that would poison the whole 49-state
    /// concatenation.
    #[test]
    fn an_all_zero_layer_slice_stays_finite() {
        let (hidden, n_states, out_dim) = (4usize, 2usize, 2usize);
        let in_dim = hidden * n_states;
        let agg = AggregateEmbed::new(vec![1.0; out_dim * in_dim], vec![0.0; out_dim], in_dim, out_dim);
        let hidden_states = vec![vec![0.0f32; hidden], vec![1.0f32; hidden]];
        let out = agg.forward(&hidden_states, 1, hidden);
        assert!(out.iter().all(|v| v.is_finite()), "an all-zero slice produced non-finite output: {out:?}");
    }
}
