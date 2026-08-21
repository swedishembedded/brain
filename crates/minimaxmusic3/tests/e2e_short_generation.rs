// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real, short, single-chunk end-to-end generation: lyrics + caption in,
//! a playable WAV out, using every one of the five real checkpoint
//! components chained together (`pipeline::generate_frames` ->
//! `denoise::denoise_chunk` -> `stitch::Stitcher`).
//!
//! Gated behind all six `BRAIN_MINIMAXMUSIC3_{LM,DEPTH,CONDITION,DIT,
//! VOCODER,TOKENIZER}` env vars - skips cleanly when any is unset or
//! missing (the combined checkpoint is ~28 GB, never committed).
//!
//! Kept deliberately SHORT: a handful of AR frames, a single denoise
//! chunk (well under the 200-frame chunk size, so `denoise::chunk_starts`
//! would also only ever emit one chunk here), and fewer Euler steps than
//! the reference's own default of 30. A full multi-minute, multi-chunk
//! generation is a recorded, hardware-bound gap, not attempted here.
//!
//! Sequential-stage RAM discipline: the AR stage (two int8 Global LLM
//! instances + the depth decoder) is dropped (its own block scope ends)
//! before the denoise stage (the DiT, fp32) loads, which is dropped
//! before the vocoder stage loads - at no point are all five components
//! resident at once.

use std::env;
use std::path::Path;

use data::qwen_tokenizer::QwenBpe;
use gpu_core::Gpu;
use minimaxmusic3::config::{ConditionEncoderConfig, DepthDecoderConfig, DitConfig, VocoderConfig};
use minimaxmusic3::{condition_encoder, denoise, depth_decoder, dit, global_llm, pipeline, stitch, vocoder};

fn env_dir(name: &str) -> Option<String> {
    let dir = env::var(name).ok()?;
    if !Path::new(&dir).exists() {
        return None;
    }
    Some(dir)
}

#[test]
fn real_short_generation_produces_a_playable_wav() {
    let (Some(lm_dir), Some(depth_dir), Some(cond_dir), Some(dit_dir), Some(vocoder_dir), Some(tok_dir)) = (
        env_dir("BRAIN_MINIMAXMUSIC3_LM"),
        env_dir("BRAIN_MINIMAXMUSIC3_DEPTH"),
        env_dir("BRAIN_MINIMAXMUSIC3_CONDITION"),
        env_dir("BRAIN_MINIMAXMUSIC3_DIT"),
        env_dir("BRAIN_MINIMAXMUSIC3_VOCODER"),
        env_dir("BRAIN_MINIMAXMUSIC3_TOKENIZER"),
    )
    else {
        brain_testutil::skip("one or more BRAIN_MINIMAXMUSIC3_{LM,DEPTH,CONDITION,DIT,VOCODER,TOKENIZER} env vars unset");
        return;
    };

    let max_frames = 12usize;
    let num_inference_steps = 8usize;

    // ---- AR stage: two int8 Global LLM instances + the depth decoder. ----
    // Everything this block allocates is dropped at its closing brace,
    // before the DiT (the next stage's own multi-GB weights) loads.
    let frame_hiddens = {
        let tokenizer = QwenBpe::from_dir(&tok_dir).expect("load tokenizer");
        let (conditional_ids, unconditional_ids) = global_llm::assemble_prompt(
            &tokenizer,
            "warm acoustic ballad, gentle fingerpicked guitar, soft female vocals, 80 BPM",
            "[verse]\nquiet morning light\nfading into you\n[chorus]\nhold on to this feeling\n",
        );
        let cap = (conditional_ids.len() + max_frames + 8) as u32;
        let (cfg, lm_cond) = global_llm::import(&lm_dir, 1, cap).expect("import Global LLM (conditional)");
        let (_, lm_uncond) = global_llm::import(&lm_dir, 1, cap).expect("import Global LLM (unconditional)");
        let head = lm_cond.read_weight(cfg.head_weight());

        let dd_cfg = DepthDecoderConfig::real();
        let dd_w = depth_decoder::import(&depth_dir, &dd_cfg).expect("import depth decoder");

        pipeline::generate_frames(
            &lm_cond,
            &lm_uncond,
            &dd_w,
            &dd_cfg,
            &head,
            cfg.vocab as usize,
            cfg.d_model as usize,
            &conditional_ids,
            &unconditional_ids,
            max_frames,
            1234,
        )
    };

    let cond_cfg = ConditionEncoderConfig::real();
    let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
    assert!(!frame_hiddens.is_empty(), "AR stage produced zero frames - the model sampled AUDIO_END on frame 0");
    assert_eq!(frame_hiddens.len() % per_frame, 0, "frame_hiddens length isn't a whole number of frames");
    let num_frames = frame_hiddens.len() / per_frame;
    println!("e2e: AR stage produced {num_frames} frames");

    // ---- Denoise stage: condition encoder + the DiT. ----
    let latents = {
        let cond_w = condition_encoder::import(&cond_dir).expect("import condition encoder");
        let dit_cfg = DitConfig::real();
        let dit_w = dit::import(&dit_dir, &dit_cfg).expect("import DiT");
        let gpu = Gpu::new_cpu(dit::PIPELINES);
        let mut state = denoise::ChunkState::default();
        denoise::denoise_chunk(&gpu, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 0, &mut state, num_inference_steps, 5678)
    };
    let latent_length = condition_encoder::latent_length(&cond_cfg, num_frames);
    assert_eq!(latents.len(), DitConfig::real().in_channels as usize * latent_length);
    println!("e2e: denoise stage produced {latent_length} latent steps");

    // ---- Vocoder stage: crop-and-stitch, write the WAV. ----
    let out_path = env::temp_dir().join("minimaxmusic3_e2e_short.wav");
    let (left, right) = {
        let vocoder_cfg = VocoderConfig::real();
        let vocoder_w = vocoder::import(&vocoder_dir, &vocoder_cfg).expect("import vocoder");
        let gpu = Gpu::new_cpu(vocoder::PIPELINES);
        let mut stitcher = stitch::Stitcher::new();
        stitcher.push_chunk(&gpu, &vocoder_cfg, &vocoder_w, &latents, latent_length, true, true);
        stitcher.finish()
    };
    assert!(!left.is_empty() && left.len() == right.len(), "stitched waveform is empty or channel-length-mismatched");
    audio::wav::write_multi(&out_path, &[&left, &right], VocoderConfig::real().sampling_rate).expect("write wav");
    let samples = left.len();
    let seconds = samples as f32 / VocoderConfig::real().sampling_rate as f32;
    println!("e2e: wrote {samples} stereo samples ({seconds:.2}s) to {}", out_path.display());
    assert!(seconds > 0.5, "generated clip is implausibly short: {seconds:.2}s");
}
