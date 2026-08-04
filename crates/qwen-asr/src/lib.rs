// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-ASR: a Whisper-style audio encoder + multi-modal projector feeding a
//! spliced Qwen3-1.7B decoder (reused from `crates/qwen`). Forward parity-gated
//! against the HuggingFace `Qwen3ASRForConditionalGeneration`.

pub mod caps;
pub mod config;
pub mod encoder;
pub mod import;
pub mod model;

pub use config::{AudioEncoderConfig, QwenAsrConfig};
pub use encoder::AudioEncoder;
pub use model::Qwen3Asr;

/// Resolve a test-fixture path under the gitignored `testdata/` tree — never a
/// hardcoded absolute path (enforced in `AGENTS.md`). Root is `$BRAIN_TESTDATA`,
/// defaulting to `<repo>/testdata` (populated by `make fetch/testdata`).
#[cfg(test)]
use brain_testutil::testdata;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // GOLD/CKPT test-path locals (see AGENTS.md: no absolute paths)
    use super::*;
    use std::io::Read;
    use std::path::Path;


    fn read_f32(path: &str) -> Vec<f32> {
        let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("missing {path}"));
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn have(p: &str) -> bool {
        Path::new(p).exists()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn read_u32_from_f32(path: &str) -> Vec<u32> {
        read_f32(path).iter().map(|&v| v as u32).collect()
    }

    /// End-to-end transcription on a real LibriSpeech clip: brain must reproduce
    /// the HF model's greedy token sequence exactly, and thus the transcription.
    #[test]
    #[ignore = "slow: loads the 1.7B checkpoint + KV-cache decode (~minutes; bandwidth-bound)"]
    fn qwen_transcribe_matches_reference() {
        let dg = crate::testdata("asr/golden/qwen_decode");
        let CKPT = crate::testdata("asr/qwen-asr/hf");
        if !have(&format!("{dg}/output_ids.f32")) || !have(&format!("{CKPT}/model.safetensors")) {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = QwenAsrConfig::qwen3_asr_1_7b();
        let input_ids = read_u32_from_f32(&format!("{dg}/input_ids.f32"));
        let mel = read_f32(&format!("{dg}/input_features.f32"));
        let mask = read_f32(&format!("{dg}/input_features_mask.f32"));
        let valid = mask.iter().filter(|&&v| v > 0.5).count() as u32;
        let ref_out = read_u32_from_f32(&format!("{dg}/output_ids.f32"));
        let ref_emb = read_f32(&format!("{dg}/audio_embeds.f32"));

        // audio placement from the prompt: contiguous run of audio_token_id.
        let atid = cfg.audio_token_id;
        let row0 = input_ids.iter().position(|&t| t == atid).unwrap() as u32;
        let n_audio = input_ids.iter().filter(|&&t| t == atid).count() as u32;
        let seq_budget = input_ids.len() as u32 + ref_out.len() as u32 + 4;

        let model = Qwen3Asr::from_hf(&CKPT, cfg, seq_budget, row0, n_audio).expect("load");
        let audio_embeds = model.encode_audio(&mel, valid);
        // encoder parity on this clip too (isolates encoder from decoder)
        let de = maxdiff(&audio_embeds, &ref_emb);
        eprintln!("this-clip audio_embeds maxdiff {de} (n_audio={n_audio})");
        assert!(de < 3e-2, "audio_embeds maxdiff {de}");

        let audio_s = mel.len() as f32 / 128.0 * 0.01; // ~frames*10ms
        let t0 = std::time::Instant::now();
        let out = model.transcribe(&input_ids, &audio_embeds, &[151643, 151645], 64);
        let dt = t0.elapsed();
        eprintln!("transcribe (KV): {} tokens in {:?} (prompt {} + gen)", out.len(), dt, input_ids.len());
        let _ = audio_s;
        eprintln!("brain out ({} tok): {:?}", out.len(), out);
        eprintln!("ref   out ({} tok): {:?}", ref_out.len(), ref_out);
        assert_eq!(out, ref_out, "greedy token sequence must match HF exactly");
    }

    /// Full audio encoder + projector parity against the dumped HF activations.
    #[test]
    fn qwen_encoder_matches_reference() {
        let GOLD = crate::testdata("asr/golden/qwen_encoder");
        let CKPT = crate::testdata("asr/qwen-asr/hf");
        if !have(&format!("{GOLD}/encoder_out.f32")) || !have(&format!("{CKPT}/model.safetensors")) {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = AudioEncoderConfig::qwen3_asr();
        // input mel [128, 3000] and mask [3000]
        let mel = read_f32(&format!("{GOLD}/input_features.f32"));
        let mask = read_f32(&format!("{GOLD}/input_features_mask.f32"));
        let valid_frames = mask.iter().filter(|&&v| v > 0.5).count() as u32;

        let weights = import::load_audio_encoder(Path::new(&CKPT), &cfg).expect("load");
        let gpu = gpu_core::Gpu::new_cpu(encoder::audio_pipelines());
        let enc = AudioEncoder::new(&gpu, cfg, &weights);
        let (encoder_out, audio_embeds) = enc.encode(&mel, valid_frames);

        let ref_enc = read_f32(&format!("{GOLD}/encoder_out.f32"));
        let ref_emb = read_f32(&format!("{GOLD}/audio_embeds.f32"));
        let de = maxdiff(&encoder_out, &ref_enc);
        let db = maxdiff(&audio_embeds, &ref_emb);
        eprintln!("encoder_out maxdiff {de}  audio_embeds maxdiff {db}  (n_audio={})", encoder_out.len() / 1024);
        assert!(de < 2e-2, "encoder_out maxdiff {de}");
        assert!(db < 3e-2, "audio_embeds maxdiff {db}");
    }
}
