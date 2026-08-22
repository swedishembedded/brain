// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The full non-streaming CosyVoice 2 pipeline: text + a reference audio clip
//! (zero-shot voice cloning) in, one real 24 kHz waveform out. ONE
//! implementation of "run the whole five-component chain", composing the
//! already individually-parity-proven pieces (`crate::llm`, `crate::flow`,
//! `crate::hift`, `campplus`, `s3tokenizer`) the way
//! `minimaxmusic3::generate::generate` composes ITS five components - a
//! `Paths`/`GenOpts` pair and one `generate()` entry point, gated on the same
//! kind of `BRAIN_*` env vars `crates/arch`'s registry names.
//!
//! ```text
//! ref_wav (any rate) ---resample--> 16 kHz ---kaldi fbank---> CAM++       -> 192-d x-vector
//!                    \--resample--> 16 kHz ---whisper mel---> S3Tokenizer -> prompt speech tokens (FSQ ids, 25 Hz)
//!                    \--resample--> 24 kHz ---cosyvoice mel-> prompt mel  -> prompt_feat
//! ref_text, text -----Qwen BPE----> text_ids = concat(prompt_text_ids, target_text_ids)
//!
//! text_ids, prompt speech tokens -> CosyVoiceLm -> generated speech tokens (own RNG - see below)
//! prompt speech tokens ++ generated tokens, x-vector, prompt mel -> flow -> target mel
//! target mel -> HiFT -> waveform (own RNG - see below)
//! ```
//!
//! # Sequential-stage RAM discipline
//!
//! This box has 30 GB RAM and no discrete GPU; the five checkpoints
//! (`llm.pt` 2 GB, `flow.pt` 451 MB, `hift.pt` 83 MB, `speech_tokenizer_v2.onnx`
//! 496 MB, `campplus.onnx` 28 MB) are individually fine but not all worth
//! holding resident at once. Exactly like `minimaxmusic3::generate`'s own
//! module doc, each stage's weights live in their own block scope in
//! [`generate`] and are dropped before the next stage's import runs: CAM++
//! and S3Tokenizer (the two reference-audio analysis stages) never overlap
//! the LM, which never overlaps the flow decoder, which never overlaps HiFT.
//!
//! # Truncation matches the reference's own zero-shot frontend
//!
//! `CosyVoiceFrontEnd._extract_speech_token`/`_extract_speech_feat` produce
//! the prompt speech tokens (S3Tokenizer, one call over the whole clip) and
//! the prompt mel (this crate's own `cosyvoice_24k` mel front end)
//! independently, then `frontend_zero_shot` truncates BOTH to
//! `token_len = min(mel_frames / 2, num_tokens)` before handing them to
//! either the LM or the flow decoder (`token_mel_ratio = 2`) - reproduced
//! verbatim here, not assumed. The SAME truncated token sequence feeds both
//! `CosyVoiceLm::prefill`'s `prompt_speech_tokens` and `flow::forward`'s
//! `prompt_tokens` - one truncation, two consumers, matching the reference's
//! own single `speech_token` variable.
//!
//! # RNG-crossing gaps this pipeline inherits, not introduces
//!
//! Two components this pipeline calls already document, and this module does
//! not paper over: [`crate::llm::CosyVoiceLm::generate`]'s `ras_sampling`
//! draws from `data::rng::Rng`, not torch's Mersenne-Twister stream, so the
//! SAME seed reproduces the SAME token sequence on this port but not the
//! reference's; [`crate::hift::forward_seeded`]'s NSF source noise is the
//! same honest gap. End to end, this means [`generate`] is deterministic
//! given a fixed seed (verified in this module's own tests) and produces
//! real, playable, structurally-correct speech - but is NOT expected to
//! reproduce a reference run bit-for-bit, exactly as `porting.md`'s parity
//! ladder rung 5 ("real run: own RNG, own text encoder - statistically
//! equivalent, not bit-identical") describes.
//!
//! # Kaldi fbank front end: a genuine, honestly-recorded gap
//!
//! CAM++'s reference input is `torchaudio.compliance.kaldi.fbank`
//! (`audio::kaldi_fbank`, ported from that library's own source line for
//! line). No golden capture of a REAL `torchaudio.compliance.kaldi.fbank`
//! run exists in this workspace to check it against bit-for-bit - CAM++'s own
//! real-weight parity test reads its fbank input from a captured golden
//! rather than computing it - so this pipeline's x-vector is structurally,
//! not numerically, verified against the reference. See
//! `audio::kaldi_fbank`'s own module doc for the exact scope of that gap.
//!
//! # Streaming
//!
//! Chunked `token2wav` (growing-prefix flow re-run, Hamming cross-fade,
//! `token_hop_len`/`token_overlap_len`/`mel_cache_len`/`source_cache_len`) is
//! a deliberate, NOT-yet-implemented follow-up - this module always runs
//! `flow.inference(..., finalize=True)`'s non-streaming path (a single Euler
//! solve over the whole generated span, a single HiFT forward with no
//! `cache_source`), matching `crate::flow`/`crate::hift`'s own already-recorded
//! "streaming is a documented gap" scope.

use std::path::Path;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;

use crate::flow;
use crate::flow_config::FlowConfig;
use crate::flow_import::import_flow_pt;
use crate::hift;
use crate::hift_config::HiftConfig;
use crate::hift_import::import_hift_pt;
use crate::llm::CosyVoiceLm;
use crate::profile;

/// One role per checkpoint/resource directory, resolved from the SAME six
/// env vars `crates/arch`'s `cosyvoice`/`s3tokenizer`/`campplus` registry
/// rows name in their own `weights_env` tables.
#[derive(Clone, Debug)]
pub struct CosyVoicePaths {
    /// Directory containing `llm.pt`.
    pub llm: String,
    /// Directory containing `flow.pt`.
    pub flow: String,
    /// Directory containing `hift.pt`.
    pub hift: String,
    /// Directory containing `speech_tokenizer_v2.onnx`.
    pub s3tokenizer: String,
    /// Directory containing `campplus.onnx`.
    pub campplus: String,
    /// Directory containing the `CosyVoice-BlankEN` Qwen BPE identity
    /// (`vocab.json`/`merges.txt` or `tokenizer.json`).
    pub tokenizer: String,
}

/// `(variable, human role name)`, in the order [`CosyVoicePaths`] declares
/// them - one table so the env reader and its own error messages cannot
/// disagree about the spelling (matches `minimaxmusic3::generate::PATH_VARS`'s
/// own shape).
pub const PATH_VARS: [(&str, &str); 6] = [
    ("BRAIN_COSYVOICE_LLM", "speech-token LM"),
    ("BRAIN_COSYVOICE_FLOW", "flow decoder"),
    ("BRAIN_COSYVOICE_HIFT", "HiFT vocoder"),
    ("BRAIN_S3TOKENIZER_V2", "S3Tokenizer FSQ speech tokenizer"),
    ("BRAIN_CAMPPLUS_DIR", "CAM++ speaker encoder"),
    ("BRAIN_COSYVOICE_TOKENIZER", "text BPE tokenizer (CosyVoice-BlankEN)"),
];

impl CosyVoicePaths {
    /// Every role from its environment variable; `Err` names the first
    /// missing one.
    pub fn from_env() -> Result<CosyVoicePaths, String> {
        let get = |i: usize| -> Result<String, String> {
            let (var, role) = PATH_VARS[i];
            std::env::var(var).ok().filter(|v| !v.is_empty()).ok_or_else(|| format!("no {role} weights: set {var}"))
        };
        Ok(CosyVoicePaths { llm: get(0)?, flow: get(1)?, hift: get(2)?, s3tokenizer: get(3)?, campplus: get(4)?, tokenizer: get(5)? })
    }
}

/// Generation knobs. `max_token_text_ratio`/`min_token_text_ratio` are the
/// reference's own `Qwen2LM.inference` defaults (`20`/`2`) - the AR decode
/// cap and the eos-ignore floor are BOTH sized off the TARGET text's own
/// token count (excluding the prompt text), matching
/// `min_len = int((text_len - prompt_text_len) * min_token_text_ratio)` read
/// directly from `cosyvoice/llm/llm.py`.
#[derive(Clone, Debug)]
pub struct GenOpts {
    pub seed: u64,
    pub max_token_text_ratio: f32,
    pub min_token_text_ratio: f32,
    /// Euler steps the flow decoder's CFM solver takes (`n_timesteps`, `10`
    /// in the reference).
    pub n_timesteps: usize,
}

impl Default for GenOpts {
    fn default() -> GenOpts {
        GenOpts { seed: 0, max_token_text_ratio: 20.0, min_token_text_ratio: 2.0, n_timesteps: FlowConfig::cosyvoice2().n_timesteps as usize }
    }
}

/// A finished utterance: 24 kHz mono samples.
pub struct GeneratedSpeech {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// `wav.samples` resampled to `target_rate` via the accurate Kaiser-windowed-sinc
/// resampler (identity when the rates already match - `audio::resample::rational`'s
/// own `l == m` fast path).
fn resample_to(wav: &audio::wav::Wav, target_rate: u32) -> Vec<f32> {
    if wav.sample_rate == target_rate {
        wav.samples.clone()
    } else {
        audio::resample::rational(&wav.samples, target_rate, wav.sample_rate)
    }
}

/// `kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)`
/// then `feat - feat.mean(dim=0, keepdim=True)` - CAM++'s own
/// `_extract_spk_embedding`, reproduced verbatim (the per-mel-bin time-mean
/// subtraction, not a normalization this crate invented). Returns `([t, 80]`
/// row-major, `t)`.
fn campplus_fbank(samples_16k: &[f32]) -> (Vec<f32>, u32) {
    let cfg = audio::kaldi_fbank::KaldiFbankConfig::cosyvoice();
    let (mut feat, t) = audio::kaldi_fbank::fbank(samples_16k, &cfg);
    let d = cfg.num_mel_bins;
    if t > 0 {
        for c in 0..d {
            let mean: f32 = (0..t).map(|ti| feat[ti * d + c]).sum::<f32>() / t as f32;
            for ti in 0..t {
                feat[ti * d + c] -= mean;
            }
        }
    }
    (feat, t as u32)
}

/// `matcha.utils.audio.mel_spectrogram` (CosyVoice's own `feat_extractor`):
/// magnitude (not power) spectrogram -> Slaney mel filter -> `log(clamp(x,
/// 1e-5))`. Returns `([n_frames, 80]` row-major TIME-major - the layout
/// `flow::assemble_conditions`'s own `prompt_feat_tc` parameter expects,
/// matching `_extract_speech_feat`'s `.transpose(0, 1)` - and `n_frames)`.
pub fn extract_prompt_mel(samples_24k: &[f32]) -> (Vec<f32>, usize) {
    let cfg = audio::mel::MelConfig::cosyvoice_24k();
    let (power, n_frames, bins) = audio::mel::power_spectrogram(samples_24k, &cfg);
    let fb = audio::mel::mel_filterbank(&cfg);
    let n_mels = cfg.n_mels;
    let mut out = vec![0.0f32; n_frames * n_mels];
    for fr in 0..n_frames {
        for m in 0..n_mels {
            let row = &fb[m * bins..(m + 1) * bins];
            let mut acc = 0.0f32;
            for b in 0..bins {
                let magnitude = (power[fr * bins + b] + 1e-9).sqrt();
                acc += row[b] * magnitude;
            }
            out[fr * n_mels + m] = acc.max(1e-5).ln();
        }
    }
    (out, n_frames)
}

/// `(max_tokens, min_len)` sized off the TARGET text's own token count - see
/// [`GenOpts`]'s doc for the exact reference formula this reproduces.
fn token_budget(target_text_len: usize, opts: &GenOpts) -> (usize, usize) {
    let n = target_text_len.max(1) as f32;
    ((n * opts.max_token_text_ratio).round() as usize, (n * opts.min_token_text_ratio).round() as usize)
}

/// Run the full non-streaming pipeline: reference-audio analysis (CAM++ +
/// S3Tokenizer + the 24 kHz prompt mel) -> LM speech-token generation ->
/// flow decoder -> HiFT vocoder. See this module's own doc for the RAM
/// discipline and the honest RNG-crossing/kaldi-fbank gaps.
pub fn generate(paths: &CosyVoicePaths, opts: &GenOpts, text: &str, ref_wav_path: &str, ref_text: &str) -> Result<GeneratedSpeech, String> {
    // Opt-in per-stage wall-clock profiling: a per-kernel-kind table should
    // exist before any NPU-export optimization work touches code, so this
    // is the seam that produces it. Off by default so a normal call pays
    // only the cost of one env lookup
    // and a handful of `Instant::now()` calls it never reads.
    let profile = std::env::var("BRAIN_COSYVOICE_PROFILE").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
    let mut stage_times: Vec<(&'static str, std::time::Duration)> = Vec::new();

    let wav = audio::wav::read(ref_wav_path).map_err(|e| format!("cosyvoice::generate: read {ref_wav_path}: {e}"))?;
    let samples_16k = resample_to(&wav, 16000);
    let samples_24k = resample_to(&wav, 24000);

    let tokenizer = QwenBpe::from_dir(&paths.tokenizer)?;
    let prompt_text_ids = tokenizer.encode(ref_text);
    let target_text_ids = tokenizer.encode(text);
    let mut text_ids = prompt_text_ids;
    text_ids.extend(target_text_ids.iter().copied());

    // ---- Reference-audio analysis stage: CAM++ + S3Tokenizer, own scope. ----
    let t_ref_audio = std::time::Instant::now();
    let (xvec, prompt_tokens_full) = {
        let (fbank, t) = campplus_fbank(&samples_16k);
        let campplus_weights = campplus::import::import_dir(Path::new(&paths.campplus))
            .map_err(|e| format!("cosyvoice::generate: campplus import: {e}"))?;
        let campplus_cfg = campplus::config::CampplusConfig::campplus_v2();
        let gpu = Gpu::new_cpu(campplus::model::PIPELINES);
        let campplus_model = campplus::model::Campplus::new(gpu, campplus_cfg, &campplus_weights);
        let xvec = campplus_model.forward(&fbank, t);

        let (mel, n_mels, n_frames) = audio::asr_frontend::qwen_logmel(&samples_16k, samples_16k.len());
        let s3_cfg = s3tokenizer::config::S3TokenizerConfig::v2();
        if n_mels != s3_cfg.n_mels as usize {
            return Err(format!("cosyvoice::generate: whisper mel produced {n_mels} mels, s3tokenizer wants {}", s3_cfg.n_mels));
        }
        let s3_weights = s3tokenizer::import::import_dir(Path::new(&paths.s3tokenizer))
            .map_err(|e| format!("cosyvoice::generate: s3tokenizer import: {e}"))?;
        let s3_w = s3tokenizer::model::S3TokenizerWeights::from_tensors(&s3_weights, &s3_cfg);
        let tokens: Vec<u32> = s3tokenizer::model::forward(&s3_cfg, &s3_w, &mel, n_frames).into_iter().map(|t| t as u32).collect();
        (xvec, tokens)
    };
    if profile {
        let d = t_ref_audio.elapsed();
        eprintln!("[profile] ref_audio_analysis (campplus+s3tokenizer): {:.3} ms", d.as_secs_f64() * 1000.0);
        stage_times.push(("ref_audio_analysis (campplus+s3tokenizer)", d));
    }

    // ---- Prompt mel (host-only math, no weights) + the reference's own truncation. ----
    let (prompt_feat_full_tc, feat_frames) = extract_prompt_mel(&samples_24k);
    let mel_dim = FlowConfig::cosyvoice2().output_size as usize;
    let token_len = (feat_frames / 2).min(prompt_tokens_full.len());
    if token_len == 0 {
        return Err("cosyvoice::generate: the reference clip produced zero usable prompt tokens/frames".to_string());
    }
    let prompt_tokens = prompt_tokens_full[..token_len].to_vec();
    let mel_len1 = 2 * token_len;
    let prompt_feat_tc = prompt_feat_full_tc[..mel_len1 * mel_dim].to_vec();

    // ---- LM stage: text -> generated speech tokens, own scope. ----
    let (max_tokens, min_len) = token_budget(target_text_ids.len(), opts);
    let t_lm_load = std::time::Instant::now();
    let ctx = (1 + text_ids.len() + 1 + prompt_tokens.len() + max_tokens + 8) as u32;
    let lm = CosyVoiceLm::load(&format!("{}/llm.pt", paths.llm), ctx)?;
    if profile {
        let d = t_lm_load.elapsed();
        eprintln!("[profile] lm_load_import: {:.3} ms", d.as_secs_f64() * 1000.0);
        stage_times.push(("lm_load_import", d));
    }
    let t_lm_prefill = std::time::Instant::now();
    let d = lm.cfg.llm_input_size as usize;
    let hidden = lm.prefill(&text_ids, &prompt_tokens);
    if profile {
        let dt = t_lm_prefill.elapsed();
        eprintln!("[profile] lm_prefill ({} prefix rows): {:.3} ms", 1 + text_ids.len() + 1 + prompt_tokens.len(), dt.as_secs_f64() * 1000.0);
        stage_times.push(("lm_prefill", dt));
    }
    let last_hidden = &hidden[hidden.len() - d..];
    let t_lm_generate = std::time::Instant::now();
    let gen_tokens = lm.generate(last_hidden, max_tokens, min_len, opts.seed);
    if profile {
        let dt = t_lm_generate.elapsed();
        eprintln!("[profile] lm_generate (autoregressive decode, {} tokens): {:.3} ms", gen_tokens.len(), dt.as_secs_f64() * 1000.0);
        stage_times.push(("lm_generate (autoregressive decode)", dt));
    }
    drop(lm);
    if gen_tokens.is_empty() {
        return Err("cosyvoice::generate: the LM sampled an immediate stop token (zero speech tokens) - try a different seed or prompt".to_string());
    }

    // ---- Flow decoder stage: tokens + x-vector + prompt mel -> target mel, own scope. ----
    profile::reset_flow_self_attn();
    let t_flow = std::time::Instant::now();
    let mel_out = {
        let flow_cfg = FlowConfig::cosyvoice2();
        let flow_w = import_flow_pt(&format!("{}/flow.pt", paths.flow), &flow_cfg)?;
        let noise = flow::rand_noise();
        flow::forward(&flow_w, &flow_cfg, &prompt_tokens, &gen_tokens, &xvec, &prompt_feat_tc, mel_len1, &noise, opts.n_timesteps).mel
    };
    if profile {
        let total = t_flow.elapsed();
        let attn = std::time::Duration::from_nanos(profile::flow_self_attn_ns());
        let rest = total.saturating_sub(attn);
        eprintln!(
            "[profile] flow_forward_total (t={} frames): {:.3} ms  [self_attn: {:.3} ms ({:.1}%), rest: {:.3} ms]",
            mel_len1 + 2 * gen_tokens.len(),
            total.as_secs_f64() * 1000.0,
            attn.as_secs_f64() * 1000.0,
            100.0 * attn.as_secs_f64() / total.as_secs_f64().max(1e-9),
            rest.as_secs_f64() * 1000.0
        );
        stage_times.push(("flow_forward_total (encoder + 10-step Euler CFM loop)", total));
        stage_times.push(("  of which: self_attn scalar loops (conformer + UNet transformer)", attn));
        stage_times.push(("  of which: everything else (convs, resnets, linears, ISTFT-free)", rest));
    }
    let mel_len2 = mel_out.len() / mel_dim;
    if mel_len2 == 0 {
        return Err("cosyvoice::generate: the flow decoder produced an empty target mel".to_string());
    }

    // ---- HiFT vocoder stage: target mel -> waveform, own scope. Broken into
    // its three named sub-stages (mirroring `hift::forward_seeded`'s own call
    // order) only when profiling; behaviorally identical to calling
    // `hift::forward_seeded` directly. ----
    let t_hift = std::time::Instant::now();
    let hift_cfg = HiftConfig::cosyvoice2();
    let hift_w = import_hift_pt(&format!("{}/hift.pt", paths.hift), &hift_cfg)?;
    let t_f0 = std::time::Instant::now();
    let f0 = hift::f0_predictor_forward(&hift_w.f0_predictor, &mel_out, hift_cfg.in_channels as usize, hift_cfg.f0_cond_channels as usize, mel_len2);
    if profile {
        let d = t_f0.elapsed();
        eprintln!("[profile] hift_f0_predictor: {:.3} ms", d.as_secs_f64() * 1000.0);
        stage_times.push(("hift_f0_predictor", d));
    }
    let n_noise = mel_len2 * hift_cfg.nsf_upsample_scale() as usize * hift_cfg.harmonics() as usize;
    let mut rng = data::rng::Rng::new(opts.seed);
    let randn: Vec<f32> = (0..n_noise).map(|_| rng.next_gaussian() as f32).collect();
    let t_nsf = std::time::Instant::now();
    let excitation = hift::nsf_source_forward(&f0, &hift_cfg, &hift_w, &randn);
    if profile {
        let d = t_nsf.elapsed();
        eprintln!("[profile] hift_nsf_source: {:.3} ms", d.as_secs_f64() * 1000.0);
        stage_times.push(("hift_nsf_source", d));
    }
    let t_conv = std::time::Instant::now();
    let out = hift::decode(&mel_out, mel_len2, &excitation, &hift_cfg, &hift_w);
    if profile {
        let d = t_conv.elapsed();
        let total = t_hift.elapsed();
        eprintln!("[profile] hift_conv_trunk_and_istft (decode): {:.3} ms", d.as_secs_f64() * 1000.0);
        eprintln!("[profile] hift_total: {:.3} ms", total.as_secs_f64() * 1000.0);
        stage_times.push(("hift_conv_trunk_and_istft (decode)", d));
        stage_times.push(("hift_total", total));
    }
    let (waveform, sample_rate) = (out.waveform, hift_cfg.sampling_rate);

    if profile {
        eprintln!("=== cosyvoice::pipeline::generate per-stage profile ===");
        let total: std::time::Duration = stage_times.iter().filter(|(n, _)| !n.starts_with("  of which")).map(|(_, d)| *d).sum();
        for (name, d) in &stage_times {
            eprintln!("  {name:<70} {:>10.3} ms", d.as_secs_f64() * 1000.0);
        }
        eprintln!("  {:<70} {:>10.3} ms", "TOTAL (sum of top-level stages)", total.as_secs_f64() * 1000.0);
    }

    Ok(GeneratedSpeech { samples: waveform, sample_rate })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_vars_names_match_the_arch_registrys_own_weights_env_tables() {
        // `crates/arch`'s "cosyvoice"/"s3tokenizer"/"campplus" rows'
        // `weights_env` (checked by hand against `crates/arch/src/lib.rs`,
        // not re-imported here to avoid a dependency cycle).
        let vars: Vec<&str> = PATH_VARS.iter().map(|(v, _)| *v).collect();
        assert_eq!(
            vars,
            ["BRAIN_COSYVOICE_LLM", "BRAIN_COSYVOICE_FLOW", "BRAIN_COSYVOICE_HIFT", "BRAIN_S3TOKENIZER_V2", "BRAIN_CAMPPLUS_DIR", "BRAIN_COSYVOICE_TOKENIZER"]
        );
    }

    #[test]
    fn from_env_names_the_first_missing_var() {
        // Serial by construction, same caveat as `minimaxmusic3::generate`'s
        // own version of this test: env vars are process-global, but this
        // test only ever asserts on the `Err` path.
        for (var, _) in PATH_VARS {
            std::env::remove_var(var);
        }
        let err = CosyVoicePaths::from_env().unwrap_err();
        assert!(err.contains("BRAIN_COSYVOICE_LLM"), "unexpected error: {err}");
    }

    #[test]
    fn token_budget_matches_the_reference_ratio_formula() {
        let opts = GenOpts::default();
        assert_eq!(token_budget(15, &opts), (300, 30));
        assert_eq!(token_budget(0, &opts), (20, 2), "zero-length target text still gets a floor of one token's worth of budget");
    }

    #[test]
    fn resample_to_is_identity_when_the_rate_already_matches() {
        let wav = audio::wav::Wav { sample_rate: 24000, samples: vec![0.1, -0.2, 0.3] };
        assert_eq!(resample_to(&wav, 24000), wav.samples);
    }

    #[test]
    fn campplus_fbank_output_is_zero_mean_per_bin_and_finite() {
        let sr = 16000.0f32;
        let n = 16000usize; // 1 second
        let samples: Vec<f32> = (0..n).map(|i| 0.1 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin()).collect();
        let (feat, t) = campplus_fbank(&samples);
        assert!(t > 0);
        let d = 80usize;
        assert_eq!(feat.len(), t as usize * d);
        assert!(feat.iter().all(|v| v.is_finite()));
        for c in 0..d {
            let mean: f32 = (0..t as usize).map(|ti| feat[ti * d + c]).sum::<f32>() / t as f32;
            assert!(mean.abs() < 1e-3, "bin {c} time-mean {mean} should be ~0 after subtraction");
        }
    }

    #[test]
    fn extract_prompt_mel_produces_the_right_shape_and_is_finite() {
        let sr = 24000.0f32;
        let n = 24000usize; // 1 second
        let samples: Vec<f32> = (0..n).map(|i| 0.2 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr).sin()).collect();
        let (mel, t) = extract_prompt_mel(&samples);
        assert!(t > 0);
        assert_eq!(mel.len(), t * 80);
        assert!(mel.iter().all(|v| v.is_finite()));
    }
}
