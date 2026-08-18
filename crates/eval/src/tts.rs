// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Text-to-speech (Qwen3-TTS) audio evaluation metrics — fp32, dependency-light.
//!
//! Three metrics, all reusing the existing audio/speaker front-ends so the
//! evaluator shares the inference stack's windowing and never drifts from it:
//!
//! - [`mel_cepstral_distortion`] (**MCD**) — the standard objective measure of
//!   spectral distance between two utterances. Each waveform is turned into a
//!   log-mel spectrogram (reusing [`audio::mel::log_mel`]); each frame is run
//!   through a DCT-II into mel-cepstral coefficients; the per-frame Euclidean
//!   distance (over coefficients `1..=N_CEPSTRA`, excluding the `c0` energy term)
//!   is averaged and scaled by the conventional `10/ln(10) * sqrt(2)` constant.
//!   See [`mel_cepstral_distortion`] for the **frame-alignment** choice.
//! - [`speaker_similarity`] — cosine similarity of the two utterances'
//!   ECAPA x-vectors (reusing [`ecapatdnn::SpeakerEncoder::embed_wav`]). The honest
//!   "is it the same voice" number for voice-clone evaluation. Requires a loaded
//!   speaker checkpoint (GPU), so it is the only metric that is not model-free.
//! - [`log_mel_l1`] — a simple structural distance: mean absolute difference of
//!   the two log-mel spectrograms (length-clipped). Cheap, model-free, and a good
//!   smoke signal that something changed.
//!
//! [`TtsMetrics`] bundles all three and [`tts_eval`] computes them in one call
//! (the speaker term is skipped — left `NaN` — when no checkpoint is supplied).

use audio::mel::{log_mel, MelConfig};

/// Number of mel-cepstral coefficients used by [`mel_cepstral_distortion`]
/// (excluding the `c0` energy coefficient): the classic MCD uses ~13–25.
pub const N_CEPSTRA: usize = 24;

/// The codec/speaker front-ends operate at 24 kHz; both inputs are resampled to
/// this rate before any spectral comparison so frame grids line up.
const EVAL_SR: u32 = 24000;

/// Bundle of TTS evaluation metrics for one `(prediction, reference)` pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TtsMetrics {
    /// Mel-cepstral distortion (dB). Lower is better; `0` for identical audio.
    pub mcd: f32,
    /// Speaker-embedding cosine similarity in `[-1, 1]`; `1` for identical audio.
    /// `NaN` when no speaker checkpoint was supplied to [`tts_eval`].
    pub speaker_sim: f32,
    /// Mean absolute log-mel difference. Lower is better; `0` for identical audio.
    pub log_mel_l1: f32,
}

/// 24 kHz mel config shared by every metric here (128 mels, matches the speaker
/// encoder / codec front-ends).
fn eval_mel_cfg() -> MelConfig {
    MelConfig::default_24k()
}

/// Resample to the common 24 kHz evaluation rate (no-op when already 24 kHz).
fn to_eval_sr(wav: &[f32], sr: u32) -> Vec<f32> {
    audio::resample_linear(wav, sr, EVAL_SR)
}

/// DCT-II of one log-mel frame `m[0..M]` into the first `n_cepstra+1`
/// mel-cepstral coefficients (index `0` is the energy term `c0`):
/// `c_k = Σ_m m[m] · cos(π/M · (m + 0.5) · k)`.
fn dct_ii(frame: &[f32], n_cepstra: usize) -> Vec<f32> {
    let m = frame.len();
    let out_len = (n_cepstra + 1).min(m);
    let mut out = vec![0.0f32; out_len];
    for (k, c) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (i, &v) in frame.iter().enumerate() {
            acc += v * (std::f32::consts::PI / m as f32 * (i as f32 + 0.5) * k as f32).cos();
        }
        *c = acc;
    }
    out
}

/// Per-frame mel-cepstra `[n_frames, n_cepstra+1]` (row-major) for a waveform.
fn cepstra(wav: &[f32], sr: u32) -> (Vec<f32>, usize, usize) {
    let wav = to_eval_sr(wav, sr);
    let cfg = eval_mel_cfg();
    let (mel, n_frames) = log_mel(&wav, &cfg);
    let n_mels = cfg.n_mels;
    let width = (N_CEPSTRA + 1).min(n_mels);
    let mut out = vec![0.0f32; n_frames * width];
    for fr in 0..n_frames {
        let frame = &mel[fr * n_mels..(fr + 1) * n_mels];
        let c = dct_ii(frame, N_CEPSTRA);
        out[fr * width..fr * width + c.len()].copy_from_slice(&c);
    }
    (out, n_frames, width)
}

/// **Mel-cepstral distortion** (MCD, dB) between a predicted and a reference
/// waveform.
///
/// Both signals are resampled to 24 kHz, converted to a 128-bin log-mel
/// spectrogram, and DCT-II'd per frame into mel-cepstra. The distortion is the
/// per-frame Euclidean distance over coefficients `1..=N_CEPSTRA` (the `c0`
/// energy term is excluded, as is conventional), averaged over frames and scaled
/// by `10/ln(10) · √2`.
///
/// **Frame alignment.** This uses the simple **length-clip** alignment: frames
/// are compared index-for-index up to `min(n_pred, n_ref)` frames. It is exact
/// for the intended use (compare a synthesis against its own ground truth at the
/// same nominal rate/length) and fully deterministic. A DTW alignment would be
/// needed for utterances with differing speaking rates / durations; it is left
/// out here to keep the evaluator dependency-light and reproducible. Returns `0`
/// when either side has no frames.
pub fn mel_cepstral_distortion(pred_wav: &[f32], ref_wav: &[f32], sr: u32) -> f32 {
    let (cp, np, w) = cepstra(pred_wav, sr);
    let (cr, nr, _) = cepstra(ref_wav, sr);
    let frames = np.min(nr);
    if frames == 0 || w <= 1 {
        return 0.0;
    }
    let scale = 10.0 / std::f32::consts::LN_10 * std::f32::consts::SQRT_2;
    let mut total = 0.0f32;
    for fr in 0..frames {
        let a = &cp[fr * w..(fr + 1) * w];
        let b = &cr[fr * w..(fr + 1) * w];
        // Euclidean distance over coefficients 1..w (skip c0 energy term).
        let mut ss = 0.0f32;
        for k in 1..w {
            let d = a[k] - b[k];
            ss += d * d;
        }
        total += ss.sqrt();
    }
    scale * total / frames as f32
}

/// **Log-mel L1** — mean absolute difference of the two log-mel spectrograms,
/// length-clipped to `min(n_pred, n_ref)` frames. Model-free structural distance;
/// `0` for identical audio. Returns `0` when either side has no frames.
pub fn log_mel_l1(pred_wav: &[f32], ref_wav: &[f32], sr: u32) -> f32 {
    let cfg = eval_mel_cfg();
    let (mp, np) = log_mel(&to_eval_sr(pred_wav, sr), &cfg);
    let (mr, nr) = log_mel(&to_eval_sr(ref_wav, sr), &cfg);
    let frames = np.min(nr);
    let n_mels = cfg.n_mels;
    if frames == 0 {
        return 0.0;
    }
    let n = frames * n_mels;
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += (mp[i] - mr[i]).abs();
    }
    acc / n as f32
}

/// Cosine similarity of two equal-length vectors in `[-1, 1]`. Returns `0` when
/// either vector has zero norm or the lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// **Speaker similarity** — cosine of the two utterances' ECAPA x-vectors.
///
/// Loads the inference speaker encoder from `speaker_weights` (a brain
/// checkpoint, GPU), embeds both waveforms via
/// [`ecapatdnn::SpeakerEncoder::embed_wav`] (which resamples internally), and
/// returns their cosine similarity. `1.0` for identical audio. This is the one
/// metric requiring a checkpoint; gate any test of it on GPU availability.
pub fn speaker_similarity(pred_wav: &[f32], ref_wav: &[f32], speaker_weights: &str, sr: u32) -> f32 {
    let enc = ecapatdnn::SpeakerEncoder::load_inference(speaker_weights);
    let ep = enc.embed_wav(pred_wav, sr);
    let er = enc.embed_wav(ref_wav, sr);
    cosine_similarity(&ep, &er)
}

/// Compute all [`TtsMetrics`] for one `(prediction, reference)` pair.
///
/// `mcd` and `log_mel_l1` are always computed (model-free). `speaker_sim` is
/// computed only when `speaker_weights` is `Some` (it needs a checkpoint + GPU);
/// otherwise it is left `NaN`.
pub fn tts_eval(pred_wav: &[f32], ref_wav: &[f32], sr: u32, speaker_weights: Option<&str>) -> TtsMetrics {
    let mcd = mel_cepstral_distortion(pred_wav, ref_wav, sr);
    let log_mel_l1 = log_mel_l1(pred_wav, ref_wav, sr);
    let speaker_sim = match speaker_weights {
        Some(w) => speaker_similarity(pred_wav, ref_wav, w, sr),
        None => f32::NAN,
    };
    TtsMetrics { mcd, speaker_sim, log_mel_l1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short deterministic test tone (a few cycles of a sine) at 24 kHz.
    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / EVAL_SR as f32).sin())
            .collect()
    }

    #[test]
    fn identical_audio_is_zero_distance() {
        let w = tone(220.0, 6000);
        // MCD and L1 are exactly 0 for identical inputs (length-clip alignment).
        assert!(mel_cepstral_distortion(&w, &w, EVAL_SR).abs() < 1e-4, "MCD≈0");
        assert!(log_mel_l1(&w, &w, EVAL_SR).abs() < 1e-6, "L1≈0");
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        // Stands in for speaker_sim≈1 on identical audio without a checkpoint:
        // identical waveforms produce identical embeddings ⇒ cosine 1.0.
        let v: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
        // Orthogonal-ish / opposite vectors behave sensibly.
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        assert!((cosine_similarity(&v, &neg) + 1.0).abs() < 1e-5);
        assert_eq!(cosine_similarity(&v, &[0.0; 1024]), 0.0);
        assert_eq!(cosine_similarity(&v, &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn perturbed_audio_increases_distance() {
        let n = 6000;
        let w = tone(220.0, n);
        // A different pitch + additive noise must read as a larger distortion.
        let mut other = tone(330.0, n);
        let mut seed: u32 = 12345;
        for s in other.iter_mut() {
            // cheap LCG noise in [-0.1, 0.1]
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *s += ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 0.2;
        }
        let mcd_same = mel_cepstral_distortion(&w, &w, EVAL_SR);
        let mcd_diff = mel_cepstral_distortion(&w, &other, EVAL_SR);
        let l1_same = log_mel_l1(&w, &w, EVAL_SR);
        let l1_diff = log_mel_l1(&w, &other, EVAL_SR);
        assert!(mcd_diff > mcd_same + 1.0, "MCD should grow: {mcd_same} -> {mcd_diff}");
        assert!(l1_diff > l1_same + 0.1, "L1 should grow: {l1_same} -> {l1_diff}");
    }

    #[test]
    fn tts_eval_skips_speaker_without_checkpoint() {
        let w = tone(220.0, 4000);
        let m = tts_eval(&w, &w, EVAL_SR, None);
        assert!(m.mcd.abs() < 1e-3);
        assert!(m.log_mel_l1.abs() < 1e-5);
        assert!(m.speaker_sim.is_nan(), "speaker_sim is NaN without a checkpoint");
    }

    /// Checkpoint-dependent: only runs with a real speaker checkpoint path in
    /// `BRAIN_QWEN3TTS_SPEAKER` and GPU tests enabled.
    #[test]
    fn speaker_similarity_identical_is_one_gated() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        }
        let weights = match std::env::var("BRAIN_QWEN3TTS_SPEAKER") {
            Ok(p) => p,
            Err(_) => {
                return brain_testutil::skip(
                    "set BRAIN_QWEN3TTS_SPEAKER to a real speaker checkpoint to run this",
                )
            }
        };
        let w = tone(220.0, 24000);
        let sim = speaker_similarity(&w, &w, &weights, EVAL_SR);
        assert!((sim - 1.0).abs() < 1e-3, "identical audio -> cosine 1.0, got {sim}");
    }
}
