// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-ASR composite: audio encoder → projector → spliced Qwen3 decoder.
//!
//! The audio tower produces `[n_audio, d_model]` embeddings that are spliced into
//! the decoder's residual stream at the contiguous run of `audio_token_id`
//! placeholders, exactly like `qwenvl` splices visual tokens. Transcription is
//! greedy argmax decoding. The audio encoder runs on its own `Gpu`; audio tokens
//! cross to the decoder host-side via `write_img_embeds`.

use std::collections::HashMap;
use std::path::Path;

use gpu_core::Gpu;
use qwen::Qwen;

use crate::config::QwenAsrConfig;
use crate::encoder::{audio_pipelines, AudioEncoder};

/// An assembled Qwen3-ASR model for one fixed audio placement (`audio_row0`,
/// `n_audio`) and sequence budget.
pub struct Qwen3Asr {
    agpu: Gpu,
    cfg: QwenAsrConfig,
    aweights: HashMap<String, Vec<f32>>,
    decoder: Qwen,
    audio_row0: u32,
    n_audio: u32,
}

impl Qwen3Asr {
    /// Assemble from already-loaded HF tensors. `audio_row0`/`n_audio` are the
    /// placeholder run in the prompt; `seq_budget` sizes the decoder buffers
    /// (must be ≥ prompt length + generated tokens).
    pub fn from_tensors(
        tensors: Vec<checkpoint::safetensors::StTensor>,
        cfg: QwenAsrConfig,
        seq_budget: u32,
        audio_row0: u32,
        n_audio: u32,
    ) -> Qwen3Asr {
        let src: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
        let aweights = crate::import::map_audio_encoder(&src, &cfg.audio);
        let dweights = crate::import::map_decoder_weights(&src);
        drop(src); // release the ~7 GB source map before uploading the decoder
        // Inference-only decoder (weights only, no grad/moment state) so the 1.7B
        // model fits in RAM; full-training use goes through a trainable builder.
        let shard = qwen::Shard::whole(cfg.text.n_layers as usize);
        let mut decoder = Qwen::new_shard(cfg.text.clone(), 1, seq_budget, &dweights, false, shard);
        decoder.enable_mm_splice(audio_row0, n_audio);
        Qwen3Asr { agpu: Gpu::new_cpu(audio_pipelines()), cfg, aweights, decoder, audio_row0, n_audio }
    }

    /// Load a Hugging Face Qwen3-ASR checkpoint directory (bf16 → f32).
    pub fn from_hf(dir: &str, cfg: QwenAsrConfig, seq_budget: u32, audio_row0: u32, n_audio: u32) -> Result<Qwen3Asr, String> {
        let tensors = checkpoint::safetensors::read_model_dir(Path::new(dir))?;
        Ok(Self::from_tensors(tensors, cfg, seq_budget, audio_row0, n_audio))
    }

    /// Encode a `[num_mel, T]` log-mel spectrogram (first `valid_frames` columns
    /// real audio) into `[n_audio, output_dim]` decoder-space audio embeddings.
    pub fn encode_audio(&self, mel: &[f32], valid_frames: u32) -> Vec<f32> {
        let enc = AudioEncoder::new(&self.agpu, self.cfg.audio, &self.aweights);
        enc.encode(mel, valid_frames).1
    }

    /// Greedy transcription: splice `audio_embeds` at the placeholder run, then
    /// argmax-decode from `input_ids` until an EOS token or `max_new` tokens.
    /// Returns the generated token ids (excluding the prompt). Cache-free
    /// (recompute) — correct but O(n²); the KV-cache path is a Phase-5 optimisation.
    pub fn transcribe(&self, input_ids: &[u32], audio_embeds: &[f32], eos: &[u32], max_new: usize) -> Vec<u32> {
        assert_eq!(audio_embeds.len(), (self.n_audio * self.cfg.text.d_model) as usize, "audio_embeds shape");
        self.decoder.write_img_embeds(audio_embeds);
        let vocab = self.cfg.text.vocab as usize;
        let prompt_len = input_ids.len();
        let mut seq = input_ids.to_vec();
        let mut out = Vec::new();
        while out.len() < max_new {
            let logits = self.decoder.logits_all(&seq);
            let last = &logits[(seq.len() - 1) * vocab..seq.len() * vocab];
            let next = argmax(last);
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            seq.push(next);
        }
        let _ = (self.audio_row0, prompt_len);
        out
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i as u32;
        }
    }
    best
}
