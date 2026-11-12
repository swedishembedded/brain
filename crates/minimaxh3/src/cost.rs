// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DiT's forward-pass cost at a config, without running it.
//!
//! The cost of a pass is a function of shapes alone, so this builds the model
//! over ZERO weights (every name and shape of [`crate::model::dit_tensor_manifest`])
//! and records one forward under `gpu_core::cost::Recording::dry()`: the graph
//! is built and tallied, nothing reaches the device.
//!
//! The pass priced is one small fixed packed request ([`PRICED_ROWS`] rows:
//! text, audio and video, two timesteps), because a DiT has no context length
//! of its own to take from the config. The number is therefore "this
//! architecture at this width and depth over a nine-row sequence", comparable
//! between configs, not a prediction for a particular generation: scale the
//! per-row work by the real row count for that.
//!
//! The weights are materialised at fp32 on the CPU backend, so this is for the
//! tiny and mid-size configs a registry probe names; the real 33B config would
//! need 130 GB of host memory and is priced from its manifest instead.
//!
//! Swedish Embedded AB implements solutions for predicting what a diffusion
//! model costs before it is run. If your team needs expertise in cost
//! accounting for large generative models then you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::cost::{CostReport, Recording};
use gpu_core::select::Dtype;
use model::StreamPlan;

use crate::block::{self, ActLevels};
use crate::config::{H3TransformerConfig, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
use crate::model::{dit_tensor_manifest, H3Transformer, PackedInputs};

const TEXT_ROWS: usize = 3;
const AUDIO_ROWS: usize = 2;
const VIDEO_ROWS: usize = 4;

/// Rows in the packed request [`forward_cost`] prices.
pub const PRICED_ROWS: usize = TEXT_ROWS + AUDIO_ROWS + VIDEO_ROWS;

/// One dry-recorded forward of the whole DiT at `cfg`, over zero weights.
pub fn forward_cost(cfg: &H3TransformerConfig) -> Result<CostReport, String> {
    let tensors: vae::blocks::Tensors = dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product();
            (name, (shape, vec![0.0f32; n]))
        })
        .collect();
    let model = H3Transformer::load_on(&tensors, *cfg, StreamPlan::single(cfg.num_layers as usize), Some("cpu"), Dtype::F32, None)?;

    let text_indices: Vec<u32> = (0..TEXT_ROWS as u32).collect();
    let audio_indices: Vec<u32> = (TEXT_ROWS as u32..(TEXT_ROWS + AUDIO_ROWS) as u32).collect();
    let video_indices: Vec<u32> = ((TEXT_ROWS + AUDIO_ROWS) as u32..PRICED_ROWS as u32).collect();
    let mut token_tags = vec![TAG_VIDEO; PRICED_ROWS];
    let mut timestep_indices = vec![0u32; PRICED_ROWS];
    for &i in &text_indices {
        token_tags[i as usize] = TAG_TEXT;
    }
    for (n, &i) in audio_indices.iter().enumerate() {
        token_tags[i as usize] = TAG_AUDIO;
        timestep_indices[i as usize] = (n % 2) as u32;
    }
    for (n, &i) in video_indices.iter().enumerate() {
        timestep_indices[i as usize] = (n % 2) as u32;
    }
    let position_ids: Vec<f32> = (0..PRICED_ROWS).flat_map(|r| [r as f32, (r % 3) as f32, (r % 2) as f32]).collect();
    let timestep = [0.2f32, 0.8];
    let video = vec![0.0f32; VIDEO_ROWS * cfg.video_patch_dim() as usize];
    let audio = vec![0.0f32; AUDIO_ROWS * cfg.audio_in_channels as usize];
    let text = vec![0.0f32; TEXT_ROWS * cfg.text_dim as usize];

    // Preparing the AdaLN table is host math over the checkpoint, not device
    // work, so it stays outside the recording.
    let _plan = model.prepare(&tensors, &timestep)?;
    let inp = PackedInputs {
        hidden_states: &video,
        audio_hidden_states: &audio,
        encoder_hidden_states: &text,
        timestep: &timestep,
        timestep_indices: &timestep_indices,
        token_tags: &token_tags,
        position_ids: &position_ids,
        video_indices: &video_indices,
        audio_indices: &audio_indices,
        text_indices: &text_indices,
        act_levels: ActLevels::One,
        vsa: None,
        row_chunk: block::ROW_CHUNK,
    };
    let rec = Recording::dry();
    let _ = model.forward(&inp);
    Ok(rec.take())
}
