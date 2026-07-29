// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Log-mel front ends for the ASR models, computed in f64 internally and
//! parity-gated (byte-for-byte inputs) against the HuggingFace feature
//! extractors — see `tools/asr_dump_reference.py` and the tests below.
//!
//! This is deliberately separate from [`crate::mel`] (which is f32, radix-2 only
//! and tuned for the TTS/speaker front end). The ASR extractors need: arbitrary
//! `n_fft` (Qwen uses 400, not a power of two → Bluestein), constant *and* reflect
//! centre padding, pre-emphasis, a centred sub-window (Nemotron: a 400-sample Hann
//! inside a 512 FFT), and two different log compressions. A unit test asserts the
//! shared slaney filterbank matches the reference `mel_filters` buffers exactly.
//!
//! Two entry points reproduce the two reference extractors step for step:
//!   * [`nemotron_logmel`] ← `NemotronAsrStreamingFeatureExtractor`
//!   * [`qwen_logmel`]     ← `Qwen3ASRFeatureExtractor`

use std::f64::consts::PI;

// ─────────────────────────────── FFT ───────────────────────────────

/// In-place iterative radix-2 Cooley–Tukey FFT (f64). `n` must be a power of two.
fn fft_radix2(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = cr * re[b] - ci * im[b];
                let ti = cr * im[b] + ci * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Complex FFT for arbitrary `n`: radix-2 when possible, else Bluestein (chirp-z),
/// which reduces an arbitrary-length DFT to a power-of-two convolution.
fn fft_any(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n.is_power_of_two() {
        fft_radix2(re, im);
        return;
    }
    // Bluestein: a_k = x_k * w^{k^2/2}, convolve with b_k = w^{-k^2/2} (w = exp(-2πi/n)).
    let m = (2 * n - 1).next_power_of_two();
    // chirp w^{k^2/2} with w = exp(-2πi/n): angle = -π k^2 / n
    let mut wr = vec![0.0f64; n];
    let mut wi = vec![0.0f64; n];
    for k in 0..n {
        // k^2 mod 2n keeps the angle accurate for large k
        let kk = (k as u128 * k as u128 % (2 * n as u128)) as f64;
        let ang = -PI * kk / n as f64;
        wr[k] = ang.cos();
        wi[k] = ang.sin();
    }
    // a_k = x_k * chirp_k
    let mut ar = vec![0.0f64; m];
    let mut ai = vec![0.0f64; m];
    for k in 0..n {
        ar[k] = re[k] * wr[k] - im[k] * wi[k];
        ai[k] = re[k] * wi[k] + im[k] * wr[k];
    }
    // b_k = conj(chirp) = w^{-k^2/2}, symmetric b_{m-k}=b_k for k in 1..n
    let mut br = vec![0.0f64; m];
    let mut bi = vec![0.0f64; m];
    br[0] = wr[0];
    bi[0] = -wi[0];
    for k in 1..n {
        br[k] = wr[k];
        bi[k] = -wi[k];
        br[m - k] = wr[k];
        bi[m - k] = -wi[k];
    }
    // convolution via FFT
    fft_radix2(&mut ar, &mut ai);
    fft_radix2(&mut br, &mut bi);
    for k in 0..m {
        let tr = ar[k] * br[k] - ai[k] * bi[k];
        let ti = ar[k] * bi[k] + ai[k] * br[k];
        ar[k] = tr;
        ai[k] = ti;
    }
    // inverse FFT (conjugate trick) with 1/m scale
    for v in ai.iter_mut() {
        *v = -*v;
    }
    fft_radix2(&mut ar, &mut ai);
    let inv = 1.0 / m as f64;
    // y_k = conv_k * chirp_k, then /m
    for k in 0..n {
        let cr = ar[k] * inv;
        let ci = -ai[k] * inv; // undo the conjugate
        re[k] = cr * wr[k] - ci * wi[k];
        im[k] = cr * wi[k] + ci * wr[k];
    }
}

// ──────────────────────────── mel scale ────────────────────────────

/// Slaney mel scale (HTK=false), matching librosa / transformers `mel_scale="slaney"`.
fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Slaney-normalised triangular mel filterbank, row-major `[n_mels, n_fft/2+1]`.
/// Matches `librosa.filters.mel(norm="slaney")` / transformers `mel_filter_bank`.
pub fn mel_filterbank_slaney(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    // n_mels + 2 band edges, evenly spaced in mel, mapped back to Hz
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    // FFT bin centre frequencies
    let fft_freqs: Vec<f64> = (0..bins).map(|b| b as f64 * sr as f64 / n_fft as f64).collect();
    let mut fb = vec![0.0f64; n_mels * bins];
    for m in 0..n_mels {
        let (lo, ce, hi) = (pts[m], pts[m + 1], pts[m + 2]);
        // slaney area normalisation: 2 / (hi - lo)
        let enorm = 2.0 / (hi - lo);
        for b in 0..bins {
            let f = fft_freqs[b];
            let lower = (f - lo) / (ce - lo);
            let upper = (hi - f) / (hi - ce);
            let w = lower.min(upper).max(0.0);
            fb[m * bins + b] = w * enorm;
        }
    }
    fb
}

fn hann(win: usize, periodic: bool) -> Vec<f64> {
    // periodic=true → divide by N (torch default); periodic=false → divide by N-1 (symmetric)
    let denom = if periodic { win as f64 } else { (win - 1) as f64 };
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / denom).cos()).collect()
}

// ───────────────────────── framing / spectrum ─────────────────────────

#[derive(Clone, Copy)]
enum Pad {
    Constant,
    Reflect,
}

/// Centre-pad a signal by `pad` samples each side (matching `torch.stft(center=True)`).
fn center_pad(x: &[f64], pad: usize, mode: Pad) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0f64; n + 2 * pad];
    out[pad..pad + n].copy_from_slice(x);
    match mode {
        Pad::Constant => {}
        Pad::Reflect => {
            // reflect without repeating the edge sample: [pad-1-i] = x[i+1]
            for i in 0..pad {
                out[pad - 1 - i] = x.get(i + 1).copied().unwrap_or(0.0);
                out[pad + n + i] = x.get(n.wrapping_sub(2 + i)).copied().unwrap_or(0.0);
            }
        }
    }
    out
}

/// Power spectrogram of `signal` given an `n_fft`-length window buffer.
/// Returns `[n_frames, bins]` row-major with `bins = n_fft/2 + 1`.
fn power_frames(signal: &[f64], n_fft: usize, hop: usize, window: &[f64]) -> (Vec<f64>, usize, usize) {
    let bins = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft { 1 + (signal.len() - n_fft) / hop } else { 0 };
    let mut spec = vec![0.0f64; n_frames * bins];
    let mut re = vec![0.0f64; n_fft];
    let mut im = vec![0.0f64; n_fft];
    for fr in 0..n_frames {
        let start = fr * hop;
        for i in 0..n_fft {
            re[i] = signal[start + i] * window[i];
            im[i] = 0.0;
        }
        fft_any(&mut re, &mut im);
        for b in 0..bins {
            spec[fr * bins + b] = re[b] * re[b] + im[b] * im[b];
        }
    }
    (spec, n_frames, bins)
}

// ─────────────────────────── Nemotron ───────────────────────────

const NEM_N_FFT: usize = 512;
const NEM_HOP: usize = 160;
const NEM_WIN: usize = 400;
const NEM_SR: u32 = 16000;
const NEM_N_MELS: usize = 128;
const NEM_PREEMPH: f64 = 0.97;
const NEM_LOG_GUARD: f64 = 5.960_464_477_539_063e-8; // 2^-24

/// The centred 400-Hann inside a 512 buffer (torch pads the window symmetrically).
fn nemotron_window() -> Vec<f64> {
    let h = hann(NEM_WIN, false);
    let off = (NEM_N_FFT - NEM_WIN) / 2; // 56
    let mut window = vec![0.0f64; NEM_N_FFT];
    window[off..off + NEM_WIN].copy_from_slice(&h);
    window
}

/// One Nemotron mel frame from a 512-sample pre-emphasised segment: windowed FFT →
/// power → slaney mel dot → `ln(x + 2^-24)`. The single implementation both the
/// offline extractor and [`NemotronMelStream`] compute frames with, so the two are
/// bit-identical by construction.
fn nemotron_frame(seg: &[f64], window: &[f64], fb: &[f64], out: &mut [f32]) {
    let bins = NEM_N_FFT / 2 + 1;
    let mut re = vec![0.0f64; NEM_N_FFT];
    let mut im = vec![0.0f64; NEM_N_FFT];
    for i in 0..NEM_N_FFT {
        re[i] = seg[i] * window[i];
    }
    fft_any(&mut re, &mut im);
    let mut sp = vec![0.0f64; bins];
    for b in 0..bins {
        sp[b] = re[b] * re[b] + im[b] * im[b];
    }
    for m in 0..NEM_N_MELS {
        let mut acc = 0.0f64;
        let row = &fb[m * bins..m * bins + bins];
        for b in 0..bins {
            acc += row[b] * sp[b];
        }
        out[m] = (acc + NEM_LOG_GUARD).ln() as f32;
    }
}

/// `NemotronAsrStreamingFeatureExtractor`: pre-emphasis(0.97) → STFT(n_fft 512,
/// hop 160, 400-sample Hann centred in the 512 window, constant pad, center=True)
/// → power → slaney mel(128, fmax 8000) → ln(x + 2^-24), no normalisation, valid
/// frames beyond `floor((L)/hop)` zeroed. Output `[n_frames, 128]` row-major.
pub fn nemotron_logmel(samples: &[f32]) -> (Vec<f32>, usize, usize) {
    // pre-emphasis: y[0]=x[0]; y[t]=x[t]-0.97*x[t-1]
    let n = samples.len();
    let mut x = vec![0.0f64; n];
    if n > 0 {
        x[0] = samples[0] as f64;
        for t in 1..n {
            x[t] = samples[t] as f64 - NEM_PREEMPH * samples[t - 1] as f64;
        }
    }

    let window = nemotron_window();
    let padded = center_pad(&x, NEM_N_FFT / 2, Pad::Constant);
    let n_frames = if padded.len() >= NEM_N_FFT { 1 + (padded.len() - NEM_N_FFT) / NEM_HOP } else { 0 };
    let fb = mel_filterbank_slaney(NEM_SR, NEM_N_FFT, NEM_N_MELS, 0.0, NEM_SR as f64 / 2.0);

    // valid length = floor((L + 2*(n_fft//2) - n_fft) / hop) = floor(L/hop)
    let valid = n / NEM_HOP;

    let mut out = vec![0.0f32; n_frames * NEM_N_MELS];
    for fr in 0..n_frames.min(valid) {
        // frames >= valid stay zero (zeroed by the attention mask in the reference)
        nemotron_frame(&padded[fr * NEM_HOP..fr * NEM_HOP + NEM_N_FFT], &window, &fb, &mut out[fr * NEM_N_MELS..(fr + 1) * NEM_N_MELS]);
    }
    (out, n_frames, NEM_N_MELS)
}

/// Frame-synchronous Nemotron mel front end: push 16 kHz samples as they arrive,
/// get complete mel frames back the moment their 512-sample window is fully
/// covered by real samples. `finish` flushes the tail using the same right
/// zero-padding the offline extractor applies, so the concatenation of every
/// `push`/`finish` output is **bit-identical** to `nemotron_logmel(all_samples)`
/// restricted to its `floor(L/hop)` valid frames (the frames the encoder consumes;
/// the offline extractor zeroes everything past them).
pub struct NemotronMelStream {
    /// Pre-emphasised samples from absolute index `base` on (earlier ones consumed).
    buf: Vec<f64>,
    base: usize,
    /// Last raw sample (pre-emphasis carry across pushes).
    prev: Option<f32>,
    /// Total raw samples received.
    total: usize,
    /// Next mel frame index to emit.
    next: usize,
    window: Vec<f64>,
    fb: Vec<f64>,
}

impl NemotronMelStream {
    #[allow(clippy::new_without_default)]
    pub fn new() -> NemotronMelStream {
        NemotronMelStream {
            buf: Vec::new(),
            base: 0,
            prev: None,
            total: 0,
            next: 0,
            window: nemotron_window(),
            fb: mel_filterbank_slaney(NEM_SR, NEM_N_FFT, NEM_N_MELS, 0.0, NEM_SR as f64 / 2.0),
        }
    }

    /// Mel frames emitted so far.
    pub fn frames(&self) -> usize {
        self.next
    }

    /// Valid frame count if the stream ended now (`floor(total/hop)`).
    pub fn valid(&self) -> usize {
        self.total / NEM_HOP
    }

    /// Extract frame `fr` (needs pre-emphasised indices `fr*hop-256 .. fr*hop+255`;
    /// out-of-range indices read the centre-pad zeros).
    fn frame(&self, fr: usize, out: &mut [f32]) {
        let mut seg = [0.0f64; NEM_N_FFT];
        let start = fr as i64 * NEM_HOP as i64 - (NEM_N_FFT / 2) as i64;
        for (i, s) in seg.iter_mut().enumerate() {
            let idx = start + i as i64;
            if idx >= 0 && (idx as usize) < self.total {
                *s = self.buf[idx as usize - self.base];
            }
        }
        nemotron_frame(&seg, &self.window, &self.fb, out);
    }

    /// Push raw samples; returns every newly complete mel frame, `[n, 128]` row-major.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        for &s in samples {
            let y = match self.prev {
                None => s as f64,
                Some(p) => s as f64 - NEM_PREEMPH * p as f64,
            };
            self.buf.push(y);
            self.prev = Some(s);
        }
        self.total += samples.len();
        // frame fr is fully real-sample-covered once index fr*hop + 256 - 1 exists
        let mut out = Vec::new();
        while self.next * NEM_HOP + NEM_N_FFT / 2 <= self.total {
            let mut row = [0.0f32; NEM_N_MELS];
            self.frame(self.next, &mut row);
            out.extend_from_slice(&row);
            self.next += 1;
        }
        // drop samples no future frame reaches (next frame reads from next*hop - 256)
        let need = (self.next * NEM_HOP).saturating_sub(NEM_N_FFT / 2);
        if need > self.base {
            self.buf.drain(..need - self.base);
            self.base = need;
        }
        out
    }

    /// Flush: emit the remaining frames up to `floor(total/hop)` using the offline
    /// extractor's right zero-padding. Returns `(frames, total_valid)`.
    pub fn finish(&mut self) -> (Vec<f32>, usize) {
        let valid = self.valid();
        let mut out = Vec::new();
        while self.next < valid {
            let mut row = [0.0f32; NEM_N_MELS];
            self.frame(self.next, &mut row);
            out.extend_from_slice(&row);
            self.next += 1;
        }
        (out, valid)
    }
}

// ─────────────────────────── Qwen3-ASR ───────────────────────────

/// `Qwen3ASRFeatureExtractor`: pad raw audio to a whole number of seconds
/// (`target_samples`), STFT(n_fft 400, hop 160, 400-Hann periodic, reflect pad,
/// center=True), drop the last time frame, power → slaney mel(128, fmax 8000) →
/// log10(clamp 1e-10) → dynamic-range compress (max−8) → (x+4)/4.
/// Output `[128, n_frames]` (channels-first) row-major.
pub fn qwen_logmel(samples: &[f32], target_samples: usize) -> (Vec<f32>, usize, usize) {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    const N_MELS: usize = 128;
    const SR: u32 = 16000;

    let mut x = vec![0.0f64; target_samples.max(samples.len())];
    for (i, &s) in samples.iter().enumerate() {
        x[i] = s as f64;
    }
    x.truncate(target_samples.max(samples.len()));

    let window = hann(N_FFT, true);
    let padded = center_pad(&x, N_FFT / 2, Pad::Reflect);
    let (spec, n_frames_full, bins) = power_frames(&padded, N_FFT, HOP, &window);
    let n_frames = n_frames_full.saturating_sub(1); // drop last time frame

    let fb = mel_filterbank_slaney(SR, N_FFT, N_MELS, 0.0, 8000.0);

    // mel energies → log10(clamp 1e-10)
    let mut log_spec = vec![0.0f64; n_frames * N_MELS];
    let mut gmax = f64::NEG_INFINITY;
    for fr in 0..n_frames {
        let sp = &spec[fr * bins..fr * bins + bins];
        for m in 0..N_MELS {
            let mut acc = 0.0f64;
            let row = &fb[m * bins..m * bins + bins];
            for b in 0..bins {
                acc += row[b] * sp[b];
            }
            let v = acc.max(1e-10).log10();
            log_spec[fr * N_MELS + m] = v;
            if v > gmax {
                gmax = v;
            }
        }
    }
    // dynamic-range compression + affine, output channels-first [n_mels, n_frames]
    let floor = gmax - 8.0;
    let mut out = vec![0.0f32; N_MELS * n_frames];
    for fr in 0..n_frames {
        for m in 0..N_MELS {
            let v = log_spec[fr * N_MELS + m].max(floor);
            out[m * n_frames + fr] = ((v + 4.0) / 4.0) as f32;
        }
    }
    (out, N_MELS, n_frames)
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)

#[allow(dead_code)]
fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
    use super::*;
    use std::io::Read;

    fn read_f32(path: &str) -> Vec<f32> {
        let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("missing golden {path}"));
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }


    fn have_goldens() -> bool {
        let GOLD = testdata("asr/golden/frontend");
        std::path::Path::new(&GOLD).join("waveform.f32").exists()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn nemotron_mel_filters_match_reference() {
        let GOLD = testdata("asr/golden/frontend");
        if !have_goldens() {
            return;
        }
        // reference [128, 257]
        let refm = read_f32(&format!("{GOLD}/nemotron_mel_filters.f32"));
        let fb = mel_filterbank_slaney(16000, 512, 128, 0.0, 8000.0);
        let fb32: Vec<f32> = fb.iter().map(|&v| v as f32).collect();
        let d = max_abs_diff(&fb32, &refm);
        assert!(d < 1e-6, "nemotron mel filterbank maxdiff {d}");
    }

    #[test]
    fn qwen_mel_filters_match_reference() {
        let GOLD = testdata("asr/golden/frontend");
        if !have_goldens() {
            return;
        }
        // reference [201, 128] (freq-major) — transpose our [128, 201]
        let refm = read_f32(&format!("{GOLD}/qwen_mel_filters.f32"));
        let fb = mel_filterbank_slaney(16000, 400, 128, 0.0, 8000.0); // [128, 201]
        let bins = 201;
        let mut ours = vec![0.0f32; 201 * 128];
        for m in 0..128 {
            for b in 0..bins {
                ours[b * 128 + m] = fb[m * bins + b] as f32;
            }
        }
        let d = max_abs_diff(&ours, &refm);
        assert!(d < 1e-6, "qwen mel filterbank maxdiff {d}");
    }

    #[test]
    fn nemotron_logmel_matches_reference() {
        let GOLD = testdata("asr/golden/frontend");
        if !have_goldens() {
            return;
        }
        let wav = read_f32(&format!("{GOLD}/waveform.f32"));
        let refm = read_f32(&format!("{GOLD}/nemotron_mel.f32")); // [201, 128]
        let (mel, t, mels) = nemotron_logmel(&wav);
        assert_eq!(t * mels, refm.len(), "shape {t}x{mels} vs golden {}", refm.len());
        let d = max_abs_diff(&mel, &refm);
        assert!(d < 2e-3, "nemotron log-mel maxdiff {d}");
    }

    /// The streaming front end, fed in ragged pushes, must reproduce the offline
    /// extractor's valid frames bit-for-bit (pure math — no fixtures needed).
    #[test]
    fn nemotron_mel_stream_matches_offline() {
        // deterministic pseudo-random signal, awkward length (not a hop multiple)
        let n = 16000 + 137;
        let mut state = 0x2545F4914F6CDD1Du64;
        let wav: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        let (mel, _t, mels) = nemotron_logmel(&wav);
        let valid = n / 160;

        let mut st = NemotronMelStream::new();
        let mut got = Vec::new();
        // ragged push sizes crossing every boundary kind
        let mut i = 0;
        for (k, &sz) in [1usize, 159, 160, 161, 512, 7, 4096, 333].iter().cycle().enumerate() {
            if i >= n {
                break;
            }
            let end = (i + sz + k % 3).min(n);
            got.extend(st.push(&wav[i..end]));
            i = end;
        }
        let (tail, v) = st.finish();
        got.extend(tail);
        assert_eq!(v, valid, "valid frame count");
        assert_eq!(got.len(), valid * mels, "streamed frame count");
        for (i, (a, b)) in got.iter().zip(&mel[..valid * mels]).enumerate() {
            assert!(a == b, "frame {} bin {}: stream {a} != offline {b}", i / mels, i % mels);
        }
    }

    #[test]
    fn qwen_logmel_matches_reference() {
        let GOLD = testdata("asr/golden/frontend");
        if !have_goldens() {
            return;
        }
        let wav = read_f32(&format!("{GOLD}/waveform.f32"));
        let refm = read_f32(&format!("{GOLD}/qwen_mel.f32")); // [128, 3000]
        // reference pads raw audio to 30 s (480000 samples)
        let (mel, mels, t) = qwen_logmel(&wav, 480000);
        assert_eq!(mels * t, refm.len(), "shape {mels}x{t} vs golden {}", refm.len());
        let d = max_abs_diff(&mel, &refm);
        assert!(d < 2e-3, "qwen log-mel maxdiff {d}");
    }
}
