// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements efficient neural-network architectures and
// the evidence that they work for its clients. If your team needs expertise
// in structured and spectral model operators, you can procure our services
// by sending an email to info@swedishembedded.com.

//! SpectralLM configuration: the decoder shape, the feed-forward mixer, and
//! the spectral representation the residual stream is stored in.

use serde_json::{json, Value};

/// The feed-forward sublayer of every block.
#[derive(Clone, Debug, PartialEq)]
pub enum Mixer {
    /// The Qwen3 SwiGLU, `W_down(SiLU(W_gate h) * W_up h)`, hidden width
    /// `d_ff`. With this mixer the model is a Qwen3 decoder of width
    /// `d_model`: the conventional control the spectral mixer is measured
    /// against, sharing every other line of this crate.
    SwiGlu { d_ff: u32 },
    /// The spectral mixer: `sum_c W_c * S^T [SiLU(S(G_c * h)) * S(U_c * h)]`
    /// with per-mode filters `G, U, W: [channels, d_model]` and `S` the
    /// `[signal_len, d_model]` truncated real Fourier basis
    /// (`model::hostmath::real_fourier_basis`).
    Spectral { signal_len: u32, channels: u32 },
}

/// How the spectral mixer computes `S z` and `S^T x`. Both compute the same
/// product; this is an execution choice and is never written to a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TransformPath {
    /// FFT where the device runs workgroup barriers and the signal length is
    /// a power of two the kernels take, the dense product otherwise.
    #[default]
    Auto,
    /// A matmul against the stored basis: the reference, runs everywhere.
    Dense,
    /// `rdft_synth`/`rdft_analysis`; refused where they cannot run.
    Fft,
}

/// The largest signal length the FFT kernels hold in workgroup memory.
pub const FFT_MAX_SIGNAL: u32 = 2048;

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    /// Residual width: the number of retained real Fourier coefficients `m`.
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub tie_embeddings: bool,
    pub mixer: Mixer,
    /// Execution only; see [`TransformPath`].
    pub transform: TransformPath,
}

impl SpectralConfig {
    /// A tiny spectral config for tests and gradient checks: every width
    /// distinct (`d_model` 13, `q_dim` 16, `kv_dim` 8, signal 32, channels 3)
    /// so a transposed index cannot hide, and `d_model` odd so the retained
    /// band ends on a cosine without its sine partner.
    pub fn tiny() -> SpectralConfig {
        SpectralConfig {
            vocab: 23,
            block_size: 12,
            n_layers: 2,
            d_model: 13,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            rope_theta: 1.0e4,
            rms_eps: 1e-6,
            tie_embeddings: true,
            mixer: Mixer::Spectral { signal_len: 32, channels: 3 },
            transform: TransformPath::Auto,
        }
    }

    /// [`Self::tiny`] with the SwiGLU mixer.
    pub fn tiny_swiglu() -> SpectralConfig {
        SpectralConfig { mixer: Mixer::SwiGlu { d_ff: 20 }, ..SpectralConfig::tiny() }
    }

    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// Reject a shape the kernels cannot run, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        let pos = |name: &str, v: u32| if v == 0 { Err(format!("spectrallm config: {name} must be positive")) } else { Ok(()) };
        pos("vocab", self.vocab)?;
        pos("block_size", self.block_size)?;
        pos("n_layers", self.n_layers)?;
        pos("d_model", self.d_model)?;
        pos("n_heads", self.n_heads)?;
        pos("n_kv_heads", self.n_kv_heads)?;
        pos("head_dim", self.head_dim)?;
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!("spectrallm config: n_heads {} is not a multiple of n_kv_heads {}", self.n_heads, self.n_kv_heads));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(format!("spectrallm config: head_dim {} must be even for RoPE", self.head_dim));
        }
        match self.mixer {
            Mixer::SwiGlu { d_ff } => pos("mixer.d_ff", d_ff)?,
            Mixer::Spectral { signal_len, channels } => {
                pos("mixer.channels", channels)?;
                if signal_len < 2 || !signal_len.is_multiple_of(2) {
                    return Err(format!("spectrallm config: mixer.signal_len {signal_len} must be even and at least 2"));
                }
                if self.d_model > signal_len {
                    return Err(format!("spectrallm config: d_model {} retained modes exceed mixer.signal_len {signal_len}", self.d_model));
                }
                if self.transform == TransformPath::Fft && !fft_supports(signal_len) {
                    return Err(format!("spectrallm config: transform fft needs a power-of-two signal_len <= {FFT_MAX_SIGNAL}, got {signal_len}"));
                }
            }
        }
        Ok(())
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let (d, v) = (self.d_model as usize, self.vocab as usize);
        let (hq, hkv, hd) = (self.q_dim() as usize, self.kv_dim() as usize, self.head_dim as usize);
        let mut out = vec![("tok.weight".to_string(), v * d)];
        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("attn.wq.weight"), hq * d));
            out.push((p("attn.wk.weight"), hkv * d));
            out.push((p("attn.wv.weight"), hkv * d));
            out.push((p("attn.q_norm.weight"), hd));
            out.push((p("attn.k_norm.weight"), hd));
            out.push((p("attn.wo.weight"), d * hq));
            out.push((p("ln2.weight"), d));
            match self.mixer {
                Mixer::SwiGlu { d_ff } => {
                    let ff = d_ff as usize;
                    out.push((p("mlp.gate.weight"), ff * d));
                    out.push((p("mlp.up.weight"), ff * d));
                    out.push((p("mlp.down.weight"), d * ff));
                }
                Mixer::Spectral { channels, .. } => {
                    let c = channels as usize;
                    out.push((p("mlp.gate_filter"), c * d));
                    out.push((p("mlp.up_filter"), c * d));
                    out.push((p("mlp.out_filter"), c * d));
                }
            }
        }
        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }
        out
    }

    /// Parameters outside the token embedding and the output head.
    pub fn non_embedding_params(&self) -> usize {
        self.param_list().into_iter().filter(|(n, _)| n != "tok.weight" && n != "lm_head.weight").map(|(_, c)| c).sum()
    }

    /// Analytic forward floating-point operations per token at context
    /// length `t`: every matmul and transform, the attention products at the
    /// mean causal length `(t+1)/2`, and the head. Normalizations and
    /// elementwise ops are left out, the convention of the 6N-style counts.
    pub fn forward_flops_per_token(&self, t: u32, transform: TransformPath) -> u64 {
        let (d, v) = (self.d_model as u64, self.vocab as u64);
        let (hq, hkv) = (self.q_dim() as u64, self.kv_dim() as u64);
        let attn_proj = 2 * d * (2 * hq + 2 * hkv);
        let attn_mix = 2 * 2 * hq * (t as u64 + 1) / 2;
        let mixer = match self.mixer {
            Mixer::SwiGlu { d_ff } => 6 * d * d_ff as u64,
            Mixer::Spectral { signal_len, channels } => {
                let (n, c) = (signal_len as u64, channels as u64);
                let filters = 4 * c * d;
                let product = 5 * c * n;
                let transforms = match transform {
                    TransformPath::Dense => 3 * c * 2 * n * d,
                    _ => 3 * c * 5 * n * (n.max(2).ilog2() as u64),
                };
                filters + product + transforms
            }
        };
        self.n_layers as u64 * (attn_proj + attn_mix + mixer) + 2 * d * v
    }

    pub fn to_json(&self) -> Value {
        let mixer = match self.mixer {
            Mixer::SwiGlu { d_ff } => json!({ "kind": "swiglu", "d_ff": d_ff }),
            Mixer::Spectral { signal_len, channels } => json!({
                "kind": "spectral",
                "signal_len": signal_len,
                "channels": channels,
                // The coefficient convention a loader must interpret the
                // residual stream in; see `model::hostmath::real_fourier_basis`.
                "basis": "real-fourier-orthonormal-frequency-ordered",
            }),
        };
        json!({
            "model": "spectrallm",
            "vocab_size": self.vocab,
            "block_size": self.block_size,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_eps,
            "tie_embeddings": self.tie_embeddings,
            "mixer": mixer,
        })
    }

    /// Parse [`Self::to_json`]'s output, refusing a missing or unknown field
    /// rather than defaulting it.
    pub fn from_json(v: &Value) -> Result<SpectralConfig, String> {
        let obj = v.as_object().ok_or("spectrallm config: not a JSON object")?;
        let allowed = [
            "model", "vocab_size", "block_size", "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim",
            "rope_theta", "rms_norm_eps", "tie_embeddings", "mixer",
        ];
        if let Some(k) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(format!("spectrallm config: unknown field {k:?}"));
        }
        if v["model"] != "spectrallm" {
            return Err(format!("spectrallm config: model is {}, not \"spectrallm\"", v["model"]));
        }
        let u = |k: &str| v[k].as_u64().map(|x| x as u32).ok_or(format!("spectrallm config: {k} missing or not an integer"));
        let fl = |k: &str| v[k].as_f64().map(|x| x as f32).ok_or(format!("spectrallm config: {k} missing or not a number"));
        let m = &v["mixer"];
        let mixer = match m["kind"].as_str() {
            Some("swiglu") => Mixer::SwiGlu { d_ff: m["d_ff"].as_u64().ok_or("spectrallm config: mixer.d_ff missing")? as u32 },
            Some("spectral") => {
                if m["basis"] != "real-fourier-orthonormal-frequency-ordered" {
                    return Err(format!("spectrallm config: mixer.basis {} is not the basis this build implements", m["basis"]));
                }
                Mixer::Spectral {
                    signal_len: m["signal_len"].as_u64().ok_or("spectrallm config: mixer.signal_len missing")? as u32,
                    channels: m["channels"].as_u64().ok_or("spectrallm config: mixer.channels missing")? as u32,
                }
            }
            other => return Err(format!("spectrallm config: mixer.kind {other:?} is not swiglu or spectral")),
        };
        let cfg = SpectralConfig {
            vocab: u("vocab_size")?,
            block_size: u("block_size")?,
            n_layers: u("n_layers")?,
            d_model: u("d_model")?,
            n_heads: u("n_heads")?,
            n_kv_heads: u("n_kv_heads")?,
            head_dim: u("head_dim")?,
            rope_theta: fl("rope_theta")?,
            rms_eps: fl("rms_norm_eps")?,
            tie_embeddings: v["tie_embeddings"].as_bool().ok_or("spectrallm config: tie_embeddings missing")?,
            mixer,
            transform: TransformPath::Auto,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Whether the FFT kernels take a signal of length `n`.
pub fn fft_supports(n: u32) -> bool {
    n.is_power_of_two() && (2..=FFT_MAX_SIGNAL).contains(&n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trips_and_refuses_unknown_or_missing_fields() {
        for cfg in [SpectralConfig::tiny(), SpectralConfig::tiny_swiglu()] {
            assert_eq!(SpectralConfig::from_json(&cfg.to_json()).unwrap(), cfg);
        }
        let mut v = SpectralConfig::tiny().to_json();
        v["extra"] = json!(1);
        assert!(SpectralConfig::from_json(&v).unwrap_err().contains("unknown field"));
        let mut v = SpectralConfig::tiny().to_json();
        v.as_object_mut().unwrap().remove("head_dim");
        assert!(SpectralConfig::from_json(&v).unwrap_err().contains("head_dim"));
        let mut v = SpectralConfig::tiny().to_json();
        v["mixer"]["basis"] = json!("dct");
        assert!(SpectralConfig::from_json(&v).unwrap_err().contains("basis"));
    }

    #[test]
    fn validate_refuses_more_modes_than_samples() {
        let cfg = SpectralConfig { d_model: 40, ..SpectralConfig::tiny() };
        assert!(cfg.validate().unwrap_err().contains("exceed"));
    }

    /// The mixer's parameter count is `3 C m`, independent of the signal
    /// length; the SwiGLU's is `3 m f`.
    #[test]
    fn mixer_parameter_counts() {
        let s = SpectralConfig::tiny();
        let w = SpectralConfig::tiny_swiglu();
        let shared = w.non_embedding_params() - 2 * 3 * 13 * 20;
        assert_eq!(s.non_embedding_params(), shared + 2 * 3 * 3 * 13);
    }
}
