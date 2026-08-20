// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The condition encoder: projects the autoregressive stage's per-frame
//! hidden states onto the Flow-VAE latent timeline.
//!
//! Pure host math, not a device (WGSL) forward. Every op here runs once per
//! ~200-frame denoise chunk, not per token or per layer, and the whole
//! tensor at real dims is a few MB - a device round trip would be pure
//! overhead with nothing to parallelize across. The conv reuses
//! `audio::conv::conv1d_ref`, the exact reference oracle the WGSL `conv1d`
//! kernel is gradient-checked against elsewhere, so this stays consistent
//! with the device engine's own math even though it never dispatches to it.
//!
//! Forward, matching `MiniMaxMusic3ConditionEncoder.forward`:
//! `hidden_states[B, frames, layers*hidden]` -> reshape to per-layer rows ->
//! softmax-weighted sum over the `layers` axis (`layer_weight_logits`,
//! `layer_scale`) -> `Conv1d(hidden, out_dim, k=3, pad=1)` + bias ->
//! nearest-neighbor resample from the 25 Hz frame rate to the Flow-VAE
//! latent rate -> `[B, latent_length, out_dim]`.

use audio::conv::{conv1d_ref, Conv1d};
use checkpoint::safetensors;
use model::hostmath::softmax;
use std::path::Path;

use crate::config::ConditionEncoderConfig;

/// The condition encoder's four weight tensors.
pub struct ConditionEncoderWeights {
    /// `[num_condition_layers]` - pre-softmax layer-mix logits.
    pub layer_weight_logits: Vec<f32>,
    /// Scalar (stored as a 1-element tensor upstream).
    pub layer_scale: f32,
    /// `[out_dim, condition_hidden_dim, 3]`.
    pub proj_weight: Vec<f32>,
    /// `[out_dim]`.
    pub proj_bias: Vec<f32>,
}

/// Read `layer_weight_logits`, `layer_scale`, `proj.weight`, `proj.bias`
/// from a `condition_encoder/` checkpoint directory (a single
/// `diffusion_pytorch_model.safetensors`, per the released layout).
pub fn import(dir: &str) -> Result<ConditionEncoderWeights, String> {
    from_tensors(safetensors::read_model_dir(Path::new(dir))?, dir)
}

/// [`import`], from tensors already read (e.g. a golden fixture's own
/// `state_dict.safetensors`, which is not laid out as a checkpoint
/// directory). `label` is only used in error messages.
pub fn from_tensors(tensors: Vec<safetensors::StTensor>, label: &str) -> Result<ConditionEncoderWeights, String> {
    let mut layer_weight_logits = None;
    let mut layer_scale = None;
    let mut proj_weight = None;
    let mut proj_bias = None;
    for t in tensors {
        match t.name.as_str() {
            "layer_weight_logits" => layer_weight_logits = Some(t.data),
            "layer_scale" => layer_scale = Some(t.data),
            "proj.weight" => proj_weight = Some(t.data),
            "proj.bias" => proj_bias = Some(t.data),
            other => return Err(format!("condition_encoder: unexpected tensor {other:?} in {label}")),
        }
    }
    let layer_scale = layer_scale.ok_or_else(|| format!("condition_encoder: missing layer_scale in {label}"))?;
    if layer_scale.len() != 1 {
        return Err(format!("condition_encoder: layer_scale has {} elements, expected 1", layer_scale.len()));
    }
    Ok(ConditionEncoderWeights {
        layer_weight_logits: layer_weight_logits
            .ok_or_else(|| format!("condition_encoder: missing layer_weight_logits in {label}"))?,
        layer_scale: layer_scale[0],
        proj_weight: proj_weight.ok_or_else(|| format!("condition_encoder: missing proj.weight in {label}"))?,
        proj_bias: proj_bias.ok_or_else(|| format!("condition_encoder: missing proj.bias in {label}"))?,
    })
}

/// `F.interpolate(mode="nearest")` along the length axis of an `[C, L_in]`
/// row-major buffer, matching PyTorch's index formula
/// `floor(out_idx * L_in / L_out)`.
fn nearest_resample(x: &[f32], c: usize, l_in: usize, l_out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; c * l_out];
    for lo in 0..l_out {
        let li = (lo * l_in / l_out).min(l_in.saturating_sub(1));
        for ch in 0..c {
            y[ch * l_out + lo] = x[ch * l_in + li];
        }
    }
    y
}

/// The latent length `F.interpolate` targets for `num_frames` autoregressive
/// frames, matching the reference's own formula exactly: Python's `/` is
/// FLOAT division at every step (`int()` truncates only once, at the very
/// end), not integer division chained left to right - those give different
/// answers (e.g. real dims at 5 frames: 17 the reference's way, 16 if each
/// `/` truncated).
pub fn latent_length(cfg: &ConditionEncoderConfig, num_frames: usize) -> usize {
    let scaled = num_frames as f64 * cfg.output_sampling_rate as f64 / cfg.input_sampling_rate as f64
        * cfg.input_hop_length as f64
        / cfg.output_hop_length as f64;
    (scaled as usize).max(1)
}

/// Forward. `hidden_states` is `[batch, frames, num_condition_layers *
/// condition_hidden_dim]` row-major; returns `(out, latent_length)` where
/// `out` is `[batch, latent_length, out_dim]` row-major.
pub fn forward(
    cfg: &ConditionEncoderConfig,
    w: &ConditionEncoderWeights,
    hidden_states: &[f32],
    batch: usize,
    frames: usize,
) -> (Vec<f32>, usize) {
    let (layers, hidden) = (cfg.num_condition_layers as usize, cfg.condition_hidden_dim as usize);
    assert_eq!(
        hidden_states.len(),
        batch * frames * layers * hidden,
        "condition_encoder::forward: hidden_states has {} elements, expected batch*frames*layers*hidden={}",
        hidden_states.len(),
        batch * frames * layers * hidden
    );

    let mut weights = w.layer_weight_logits.clone();
    softmax(&mut weights);

    let lo = latent_length(cfg, frames);
    let out_dim = cfg.out_dim as usize;
    let mut out = vec![0.0f32; batch * out_dim * lo];

    for b in 0..batch {
        // Weighted sum over `layers`, transposed from [frames, layers, hidden]
        // (the input's own layout, `.transpose(1,2).reshape(B,layers,hidden,frames)`
        // in the reference) directly into NCL [hidden, frames] - no separate
        // permute buffer, since the destination index is a pure function of
        // (channel, frame, layer).
        let mut mixed = vec![0.0f32; hidden * frames];
        for f in 0..frames {
            let frame_base = (b * frames + f) * layers * hidden;
            for (l, &weight) in weights.iter().enumerate() {
                let s = weight * w.layer_scale;
                let layer_base = frame_base + l * hidden;
                for ch in 0..hidden {
                    mixed[ch * frames + f] += s * hidden_states[layer_base + ch];
                }
            }
        }

        let conv = Conv1d { n: 1, cin: hidden as u32, l: frames as u32, cout: out_dim as u32, k: 3, stride: 1, pad: 1, dilation: 1, groups: 1, lo: frames as u32 };
        let mut proj = conv1d_ref(&conv, &mixed, &w.proj_weight);
        for ch in 0..out_dim {
            for f in 0..frames {
                proj[ch * frames + f] += w.proj_bias[ch];
            }
        }

        let resampled = nearest_resample(&proj, out_dim, frames, lo);
        // Transpose [out_dim, lo] -> [lo, out_dim] for this batch row.
        let out_base = b * lo * out_dim;
        for t in 0..lo {
            for ch in 0..out_dim {
                out[out_base + t * out_dim + ch] = resampled[ch * lo + t];
            }
        }
    }
    (out, lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_length_matches_the_reference_formula_by_hand() {
        // 5 frames at 25 Hz (24000/960) -> 44100 Hz / 512-hop latents:
        // 5 * 44100/24000 * 960/512 = 17.226 -> floor via integer math order.
        let cfg = crate::config::ConditionEncoderConfig::real();
        assert_eq!(latent_length(&cfg, 5), 17);
    }

    #[test]
    fn nearest_resample_matches_pytorch_index_formula() {
        // l_in=3 -> l_out=6: out index i maps to floor(i*3/6) = floor(i/2).
        let x = [10.0f32, 20.0, 30.0];
        let y = nearest_resample(&x, 1, 3, 6);
        assert_eq!(y, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    }

    #[test]
    fn forward_shape_matches_tiny_config() {
        let cfg = crate::config::ConditionEncoderConfig::tiny();
        let (layers, hidden, out_dim) = (cfg.num_condition_layers as usize, cfg.condition_hidden_dim as usize, cfg.out_dim as usize);
        let w = ConditionEncoderWeights {
            layer_weight_logits: vec![0.1; layers],
            layer_scale: 1.0,
            proj_weight: vec![0.01; out_dim * hidden * 3],
            proj_bias: vec![0.0; out_dim],
        };
        let frames = 5;
        let hidden_states = vec![0.5f32; frames * layers * hidden];
        let (out, lo) = forward(&cfg, &w, &hidden_states, 1, frames);
        assert_eq!(lo, latent_length(&cfg, frames));
        assert_eq!(out.len(), lo * out_dim);
    }
}
