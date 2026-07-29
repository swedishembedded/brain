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

    /// The number of audio placeholder tokens this model is assembled for.
    pub fn n_audio(&self) -> u32 {
        self.n_audio
    }

    /// Load a checkpoint **for a fixed audio window**. Qwen3-ASR is offline and its
    /// decoder is assembled for a fixed `n_audio` placement, so a served instance is
    /// built once for a window: the audio encoder is **probed** with a
    /// `window_samples`-long clip to discover the exact token count (the chunked
    /// packing makes it non-analytic), then the decoder is assembled for that count.
    /// Clips are later padded/truncated to the window (see `caps::pad_to_window`).
    /// Reads the checkpoint once. Returns the model + its `n_audio`.
    pub fn from_hf_windowed(dir: &str, cfg: QwenAsrConfig, window_samples: usize, audio_row0: u32, max_new: u32) -> Result<(Qwen3Asr, u32), String> {
        let tensors = checkpoint::safetensors::read_model_dir(Path::new(dir))?;
        let src: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
        let aweights = crate::import::map_audio_encoder(&src, &cfg.audio);
        let agpu = Gpu::new_cpu(audio_pipelines());
        // Probe: encode a full window of silence to get the actual audio-token count.
        let silence = vec![0.0f32; window_samples];
        let (mel, valid, _n) = audio::asr_frontend::qwen_logmel(&silence, window_samples);
        let enc = AudioEncoder::new(&agpu, cfg.audio, &aweights);
        let embeds = enc.encode(&mel, valid as u32).1;
        let n_audio = (embeds.len() / cfg.audio.output_dim as usize) as u32;
        drop(enc);
        // Assemble the decoder for that fixed placement.
        let dweights = crate::import::map_decoder_weights(&src);
        drop(src);
        let prompt_len = crate::caps::PROMPT_PREFIX.len() as u32 + n_audio + crate::caps::PROMPT_SUFFIX.len() as u32;
        let seq_budget = prompt_len + max_new + 4;
        let shard = qwen::Shard::whole(cfg.text.n_layers as usize);
        let mut decoder = Qwen::new_shard(cfg.text.clone(), 1, seq_budget, &dweights, false, shard);
        decoder.enable_mm_splice(audio_row0, n_audio);
        Ok((Qwen3Asr { agpu, cfg, aweights, decoder, audio_row0, n_audio }, n_audio))
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
    /// Greedy transcription with a **KV-cache** prefill + incremental decode.
    /// The prompt is prefilled token-by-token into the cache; at the contiguous
    /// audio-placeholder run the projected audio embeddings are injected via
    /// `step_embed` (no re-embedding of the placeholder id). O(n) per token
    /// instead of the cache-free O(n²) recompute.
    pub fn transcribe(&self, input_ids: &[u32], audio_embeds: &[f32], eos: &[u32], max_new: usize) -> Vec<u32> {
        let d = self.cfg.text.d_model as usize;
        assert_eq!(audio_embeds.len(), self.n_audio as usize * d, "audio_embeds shape");
        let vocab = self.cfg.text.vocab as usize;
        let head = self.decoder.read_weight(self.cfg.text.head_weight()); // [vocab, d]
        let logits_of = |h: &[f32]| -> Vec<f32> {
            (0..vocab).map(|o| head[o * d..o * d + d].iter().zip(h).map(|(a, b)| a * b).sum()).collect()
        };

        self.decoder.reset_cache();
        let (row0, n) = (self.audio_row0 as usize, self.n_audio as usize);
        // prefill: audio embeds at the placeholder run, token embeddings elsewhere
        let mut hidden = Vec::new();
        for (pos, &tok) in input_ids.iter().enumerate() {
            hidden = if pos >= row0 && pos < row0 + n {
                self.decoder.step_embed(&audio_embeds[(pos - row0) * d..(pos - row0 + 1) * d])
            } else {
                self.decoder.step(tok)
            };
        }
        // incremental greedy decode
        let mut out = Vec::new();
        while out.len() < max_new {
            let next = argmax(&logits_of(&hidden));
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            hidden = self.decoder.step(next);
        }
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
