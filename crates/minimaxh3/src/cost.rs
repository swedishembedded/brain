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

use std::time::Instant;

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

/// Zero weights for every tensor of the DiT at `cfg`: cost is a function of shapes, never of values.
fn zero_tensors(cfg: &H3TransformerConfig) -> vae::blocks::Tensors {
    dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n = shape.iter().product();
            (name, (shape, vec![0.0f32; n]))
        })
        .collect()
}

/// The packed request both [`forward_cost`] and [`measure_forward`] run: [`PRICED_ROWS`] rows of text, audio and video
/// over two timesteps, zero-valued.
struct PricedRequest {
    video: Vec<f32>,
    audio: Vec<f32>,
    text: Vec<f32>,
    timestep: [f32; 2],
    timestep_indices: Vec<u32>,
    token_tags: Vec<u32>,
    position_ids: Vec<f32>,
    video_indices: Vec<u32>,
    audio_indices: Vec<u32>,
    text_indices: Vec<u32>,
}

impl PricedRequest {
    fn new(cfg: &H3TransformerConfig) -> PricedRequest {
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
        PricedRequest {
            video: vec![0.0f32; VIDEO_ROWS * cfg.video_patch_dim() as usize],
            audio: vec![0.0f32; AUDIO_ROWS * cfg.audio_in_channels as usize],
            text: vec![0.0f32; TEXT_ROWS * cfg.text_dim as usize],
            timestep: [0.2f32, 0.8],
            timestep_indices,
            token_tags,
            position_ids: (0..PRICED_ROWS).flat_map(|r| [r as f32, (r % 3) as f32, (r % 2) as f32]).collect(),
            video_indices,
            audio_indices,
            text_indices,
        }
    }

    fn inputs(&self) -> PackedInputs<'_> {
        PackedInputs {
            hidden_states: &self.video,
            audio_hidden_states: &self.audio,
            encoder_hidden_states: &self.text,
            timestep: &self.timestep,
            timestep_indices: &self.timestep_indices,
            token_tags: &self.token_tags,
            position_ids: &self.position_ids,
            video_indices: &self.video_indices,
            audio_indices: &self.audio_indices,
            text_indices: &self.text_indices,
            act_levels: ActLevels::One,
            vsa: None,
            row_chunk: block::ROW_CHUNK,
        }
    }
}

/// One dry-recorded forward of the whole DiT at `cfg`, over zero weights.
pub fn forward_cost(cfg: &H3TransformerConfig) -> Result<CostReport, String> {
    let tensors = zero_tensors(cfg);
    let model = H3Transformer::load_on(&tensors, *cfg, StreamPlan::single(cfg.num_layers as usize), Some("cpu"), Dtype::F32, None)?;
    let request = PricedRequest::new(cfg);
    // Preparing the AdaLN table is host math over the checkpoint, not device
    // work, so it stays outside the recording.
    let _plan = model.prepare(&tensors, &request.timestep)?;
    let rec = Recording::dry();
    let _ = model.forward(&request.inputs());
    Ok(rec.take())
}

/// What [`measure_forward`] observed of a real forward on the default device.
#[derive(Clone, Copy, Debug)]
pub struct ForwardMeasurement {
    /// Building the model: uploading every weight and compiling the pipelines.
    pub load_seconds: f64,
    /// The first forward after the build.
    pub cold_seconds: f64,
    /// The best of the later forwards.
    pub hot_seconds: f64,
    /// Device bytes live once the model is built (weights plus the buffers its blocks keep), `None` where the backend
    /// does not account for its allocations.
    pub held_bytes: Option<u64>,
    /// The most device bytes live during the timed forwards, counted from the held figure.
    pub peak_bytes: Option<u64>,
}

/// Build the DiT at `cfg` resident on the default device (one stage, fp32, zero weights: the values never change the
/// cost) and time `reps` hot forwards of the [`PRICED_ROWS`]-row request after one cold one.
pub fn measure_forward(cfg: &H3TransformerConfig, reps: usize) -> Result<ForwardMeasurement, String> {
    let tensors = zero_tensors(cfg);
    let request = PricedRequest::new(cfg);

    let load_start = Instant::now();
    let model = H3Transformer::load_on(&tensors, *cfg, StreamPlan::single(cfg.num_layers as usize), None, Dtype::F32, None)?;
    let _plan = model.prepare(&tensors, &request.timestep)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let gpu = model.stage_gpu(0).ok_or("minimaxh3: the resident model has no stage 0")?;
    let held_bytes = gpu.memory_usage().map(|u| u.live);
    gpu.reset_memory_peak();

    let forward = || {
        let t = Instant::now();
        let _ = model.forward(&request.inputs());
        gpu.poll_wait();
        t.elapsed().as_secs_f64()
    };
    let cold_seconds = forward();
    let hot_seconds = (0..reps.max(1)).map(|_| forward()).fold(f64::INFINITY, f64::min);
    Ok(ForwardMeasurement { load_seconds, cold_seconds, hot_seconds, held_bytes, peak_bytes: gpu.memory_usage().map(|u| u.peak_live) })
}
