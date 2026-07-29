// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Nemotron 3.5 ASR end-to-end: log-mel front end → device FastConformer encoder
//! → RNN-T greedy transducer decode. Returns emitted (non-blank) token ids.
//!
//! The heavy lifting lives on [`Encoder`], which owns its `Gpu` and uploads the
//! weights **once** — so a served/resident instance is just a held [`NemotronAsr`]
//! and every call reuses the built encoder (no per-call rebuild/re-upload).

use std::path::Path;

use gpu_core::Gpu;

use crate::config::NemotronConfig;
use crate::encoder::{encoder_pipelines, Encoder};

pub struct NemotronAsr {
    enc: Encoder,
}

impl NemotronAsr {
    /// Load an HF checkpoint and build the encoder on a **device-resolved** `Gpu`
    /// (honours `BRAIN_DEVICE`: cpu / vulkan / gpu). Weights are uploaded once.
    pub fn from_hf(dir: &str, cfg: NemotronConfig) -> Result<NemotronAsr, String> {
        let weights = crate::import::load_tensors(Path::new(dir))?;
        Ok(NemotronAsr { enc: Encoder::new(Gpu::new(encoder_pipelines()), cfg, &weights) })
    }

    /// Load an HF checkpoint building the encoder on an explicitly-provided `Gpu`
    /// (e.g. a CPU device for parity tests, or a shared device for serving).
    pub fn from_hf_on(dir: &str, cfg: NemotronConfig, gpu: Gpu) -> Result<NemotronAsr, String> {
        let weights = crate::import::load_tensors(Path::new(dir))?;
        Ok(NemotronAsr { enc: Encoder::new(gpu, cfg, &weights) })
    }

    pub fn config(&self) -> &NemotronConfig {
        self.enc.config()
    }

    /// The built encoder (shared by the resident adapter for batched forwards).
    pub fn encoder(&self) -> &Encoder {
        &self.enc
    }

    /// Transcribe a 16 kHz mono waveform → emitted RNN-T token ids (non-blank).
    /// `prompt_id` selects the language prompt.
    pub fn transcribe(&self, wav: &[f32], prompt_id: usize) -> Vec<u32> {
        self.enc.transcribe(wav, prompt_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "loads the 0.6B checkpoint + full pipeline (~seconds)"]
    fn transcribe_end_to_end_matches_reference() {
        use std::io::Read;
        let ckpt = crate::testdata("asr/nemotron/hf");
        let wav_path = crate::testdata("asr/audio/librispeech_mr_quilter.wav");
        let gold = crate::testdata("asr/golden/nemotron");
        if !Path::new(&wav_path).exists() || !Path::new(&format!("{ckpt}/model.safetensors")).exists() {
            eprintln!("skipping: assets absent (run `make fetch/testdata`)");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let wav = audio::wav::read(&wav_path).expect("wav");
        let model = NemotronAsr::from_hf_on(&ckpt, cfg, Gpu::new_cpu(encoder_pipelines())).expect("load");

        let t0 = std::time::Instant::now();
        let toks = model.transcribe(&wav.samples, 0); // prompt 0 = en
        let dt = t0.elapsed();

        let mut f = std::fs::File::open(format!("{gold}/output_ids.f32")).unwrap();
        let mut b = Vec::new();
        f.read_to_end(&mut b).unwrap();
        let ref_ids: Vec<u32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
        let ref_nonblank: Vec<u32> = ref_ids.into_iter().filter(|&x| x != cfg.blank_token_id).collect();

        let audio_s = wav.samples.len() as f32 / 16000.0;
        eprintln!("transcribe: {} tokens in {:?} (audio {audio_s:.2}s, RTF {:.3})", toks.len(), dt, dt.as_secs_f32() / audio_s);
        eprintln!("brain: {:?}", &toks[..toks.len().min(15)]);
        assert_eq!(toks, ref_nonblank, "end-to-end token sequence must match HF");
    }
}
