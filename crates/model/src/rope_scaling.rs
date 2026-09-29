// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every RoPE frequency scaling a checkpoint can declare, parsed once and
//! turned into one per-channel `inv_freq` table.
//!
//! Each scheme is a different table of rotary inverse frequencies (plus, for
//! YaRN, an attention-magnitude factor); the rotation that consumes the table
//! is the same, so a consumer reads [`RopeScaling::inv_freq`] and dispatches
//! one table-driven RoPE kernel, whatever the scheme.
//!
//! | scheme | declared as | table |
//! |---|---|---|
//! | none | no `rope_scaling`, `null`, or type `default` | `1 / theta^(2i/dim)` |
//! | linear | `{"type"/"rope_type": "linear", "factor"}` | base / factor |
//! | llama3 | `{"rope_type": "llama3", "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}` | base, divided by `factor` below the low-frequency wavelength and smoothly blended in between |
//! | yarn | `{"type"/"rope_type": "yarn", ...}` | [`crate::yarn`] |
//! | factors | llama.cpp's `rope_freqs.weight` tensor | base / factor_i per channel |
//!
//! The formulas are transformers' `modeling_rope_utils` (`_compute_linear_...`,
//! `_compute_llama3_parameters`, `_compute_yarn_parameters`), the reference
//! the checkpoints were trained against. A type this module does not
//! implement is an error, never "no scaling": a model run past its trained
//! window at the wrong frequencies degrades without a single failing check.
//!
//! Swedish Embedded AB implements long-context inference for its clients'
//! transformer stacks. If your team needs a checkpoint's declared context
//! window to behave as it was trained, you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::yarn::{scaled_inv_freq, YarnConfig};
use serde_json::{json, Value};

/// A checkpoint's RoPE frequency scaling. `Option::None` (never a variant) is
/// "unscaled".
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    /// Position interpolation: every frequency divided by `factor`.
    Linear { factor: f32 },
    /// Llama 3.1's piecewise scaling.
    Llama3 { factor: f32, low_freq_factor: f32, high_freq_factor: f32, original_max_position_embeddings: u32 },
    /// YaRN - see [`crate::yarn`].
    Yarn(YarnConfig),
    /// Explicit per-channel divisors (`dim / 2` of them), the form llama.cpp
    /// stores a llama3 scaling in (`rope_freqs.weight`) - the table is the
    /// declaration.
    Factors(Vec<f32>),
}

impl RopeScaling {
    /// Parse a `config.json` `rope_scaling` value (the object, `null`, or an
    /// absent key's `Value::Null`). Reads `rope_type`, falling back to the
    /// older `type` spelling. `Ok(None)` for no scaling; an error naming the
    /// type for anything not implemented here.
    pub fn from_config(v: &Value) -> Result<Option<RopeScaling>, String> {
        if v.is_null() {
            return Ok(None);
        }
        let o = v.as_object().ok_or_else(|| format!("rope_scaling: expected an object, got {v}"))?;
        let kind = o.get("rope_type").or_else(|| o.get("type")).and_then(Value::as_str).ok_or("rope_scaling: no `rope_type`/`type`")?;
        let f = |k: &str| o.get(k).and_then(Value::as_f64).map(|x| x as f32).ok_or_else(|| format!("rope_scaling {kind}: missing `{k}`"));
        let u = |k: &str| o.get(k).and_then(Value::as_u64).map(|x| x as u32).ok_or_else(|| format!("rope_scaling {kind}: missing `{k}`"));
        Ok(Some(match kind {
            "default" => return Ok(None),
            "linear" => RopeScaling::Linear { factor: f("factor")? },
            "llama3" => RopeScaling::Llama3 {
                factor: f("factor")?,
                low_freq_factor: f("low_freq_factor")?,
                high_freq_factor: f("high_freq_factor")?,
                original_max_position_embeddings: u("original_max_position_embeddings")?,
            },
            "yarn" => RopeScaling::Yarn(YarnConfig {
                factor: f("factor")?,
                original_max_position_embeddings: u("original_max_position_embeddings")?,
                beta_fast: o.get("beta_fast").and_then(Value::as_f64).map_or(32.0, |x| x as f32),
                beta_slow: o.get("beta_slow").and_then(Value::as_f64).map_or(1.0, |x| x as f32),
                attention_factor: o.get("attention_factor").and_then(Value::as_f64).map(|x| x as f32),
            }),
            "factors" => RopeScaling::Factors(
                o.get("factors").and_then(Value::as_array).ok_or("rope_scaling factors: missing `factors`")?.iter().map(|x| x.as_f64().map(|x| x as f32).ok_or("rope_scaling factors: non-numeric entry")).collect::<Result<_, _>>()?,
            ),
            other => return Err(format!("rope_scaling type {other:?} is not implemented")),
        }))
    }

    /// The config value [`Self::from_config`] reads back to `self`.
    pub fn to_config(&self) -> Value {
        match self {
            RopeScaling::Linear { factor } => json!({"rope_type": "linear", "factor": factor}),
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_position_embeddings } => json!({
                "rope_type": "llama3", "factor": factor, "low_freq_factor": low_freq_factor,
                "high_freq_factor": high_freq_factor, "original_max_position_embeddings": original_max_position_embeddings,
            }),
            RopeScaling::Yarn(y) => {
                let mut v = json!({"rope_type": "yarn", "factor": y.factor, "original_max_position_embeddings": y.original_max_position_embeddings,
                    "beta_fast": y.beta_fast, "beta_slow": y.beta_slow});
                if let Some(a) = y.attention_factor {
                    v["attention_factor"] = json!(a);
                }
                v
            }
            RopeScaling::Factors(f) => json!({"rope_type": "factors", "factors": f}),
        }
    }

    /// `(inv_freq, attention_factor)` for `dim` rotated channels at base
    /// `theta`: `dim / 2` inverse frequencies, and the factor every cos/sin
    /// entry is scaled by (1.0 for everything but YaRN).
    pub fn inv_freq(&self, dim: u32, theta: f32) -> (Vec<f32>, f32) {
        match self {
            RopeScaling::Yarn(y) => scaled_inv_freq(dim, theta, y),
            RopeScaling::Linear { factor } => (base_inv_freq(dim, theta).into_iter().map(|f| f / factor).collect(), 1.0),
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_position_embeddings } => {
                let orig = *original_max_position_embeddings as f32;
                let (low_wavelen, high_wavelen) = (orig / low_freq_factor, orig / high_freq_factor);
                let table = base_inv_freq(dim, theta)
                    .into_iter()
                    .map(|f| {
                        let wavelen = 2.0 * std::f32::consts::PI / f;
                        if wavelen < high_wavelen {
                            f
                        } else if wavelen > low_wavelen {
                            f / factor
                        } else {
                            let smooth = (orig / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                            (1.0 - smooth) * f / factor + smooth * f
                        }
                    })
                    .collect();
                (table, 1.0)
            }
            RopeScaling::Factors(div) => {
                assert_eq!(div.len(), (dim / 2) as usize, "rope_freqs has {} divisors for {} rotated channels", div.len(), dim);
                (base_inv_freq(dim, theta).into_iter().zip(div).map(|(f, d)| f / d).collect(), 1.0)
            }
        }
    }
}

/// The unscaled table, `1 / theta^(2i/dim)`, computed the way transformers'
/// `_compute_default_rope_parameters` does (a reciprocal of a power, in f32).
pub fn base_inv_freq(dim: u32, theta: f32) -> Vec<f32> {
    (0..dim / 2).map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32)).collect()
}

/// `(inv_freq, attention_factor)` for an optional scaling: the one table
/// every RoPE path is fed.
pub fn inv_freq(dim: u32, theta: f32, scaling: Option<&RopeScaling>) -> (Vec<f32>, f32) {
    match scaling {
        Some(s) => s.inv_freq(dim, theta),
        None => (base_inv_freq(dim, theta), 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], rel: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= rel * y.abs().max(f32::MIN_POSITIVE), "channel {i}: {x} vs {y}");
        }
    }

    /// DeepSeek-R1-Distill-Llama-8B's own declaration (Llama-3.1).
    fn r1_llama() -> Value {
        json!({"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"})
    }

    #[test]
    fn llama3_matches_an_independent_f64_recomputation() {
        let s = RopeScaling::from_config(&r1_llama()).unwrap().unwrap();
        let (got, af) = s.inv_freq(128, 500_000.0);
        assert_eq!(af, 1.0);
        let want: Vec<f32> = (0..64)
            .map(|i| {
                let f = 1.0 / 500_000f64.powf(2.0 * i as f64 / 128.0);
                let wl = 2.0 * std::f64::consts::PI / f;
                let (lo, hi) = (8192.0 / 1.0, 8192.0 / 4.0);
                (if wl < hi {
                    f
                } else if wl > lo {
                    f / 8.0
                } else {
                    let s = (8192.0 / wl - 1.0) / (4.0 - 1.0);
                    (1.0 - s) * f / 8.0 + s * f
                }) as f32
            })
            .collect();
        close(&got, &want, 1e-5);
        // All three regimes are exercised at these parameters.
        let base = base_inv_freq(128, 500_000.0);
        assert!(got.iter().zip(&base).any(|(g, b)| g == b), "some channel is left unscaled");
        assert!(got.iter().zip(&base).any(|(g, b)| (g * 8.0 - b).abs() <= 1e-6 * b), "some channel is divided by the full factor");
        assert!(got.iter().zip(&base).any(|(g, b)| g < b && g * 8.0 > b * 1.0001), "some channel is blended");
    }

    #[test]
    fn linear_divides_every_frequency_by_the_factor() {
        let s = RopeScaling::from_config(&json!({"factor": 4.0, "type": "linear"})).unwrap().unwrap();
        let (got, _) = s.inv_freq(128, 100_000.0);
        let want: Vec<f32> = base_inv_freq(128, 100_000.0).iter().map(|f| f / 4.0).collect();
        assert_eq!(got, want);
    }

    /// llama.cpp stores a llama3 scaling as the divisors `base / scaled`;
    /// reading them back must reproduce the llama3 table.
    #[test]
    fn explicit_factors_reproduce_the_scheme_they_were_derived_from() {
        let l3 = RopeScaling::from_config(&r1_llama()).unwrap().unwrap();
        let (scaled, _) = l3.inv_freq(128, 500_000.0);
        let divisors: Vec<f32> = base_inv_freq(128, 500_000.0).iter().zip(&scaled).map(|(b, s)| b / s).collect();
        let (from_factors, _) = RopeScaling::Factors(divisors).inv_freq(128, 500_000.0);
        close(&from_factors, &scaled, 1e-6);
    }

    #[test]
    fn yarn_is_the_yarn_module() {
        let v = json!({"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768});
        let s = RopeScaling::from_config(&v).unwrap().unwrap();
        assert_eq!(s.inv_freq(128, 1e6), scaled_inv_freq(128, 1e6, &YarnConfig::new(4.0, 32768)));
    }

    #[test]
    fn no_scaling_and_unknown_scaling_are_told_apart() {
        assert_eq!(RopeScaling::from_config(&Value::Null).unwrap(), None);
        assert_eq!(RopeScaling::from_config(&json!({"rope_type": "default"})).unwrap(), None);
        let e = RopeScaling::from_config(&json!({"rope_type": "dynamic", "factor": 2.0})).unwrap_err();
        assert!(e.contains("dynamic"), "{e}");
        assert!(RopeScaling::from_config(&json!({"rope_type": "llama3", "factor": 8.0})).unwrap_err().contains("low_freq_factor"));
    }

    #[test]
    fn every_scheme_round_trips_through_its_config() {
        for v in [
            json!({"type": "linear", "factor": 4.0}),
            r1_llama(),
            json!({"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "attention_factor": 1.2}),
            json!({"rope_type": "factors", "factors": [1.0, 2.0, 8.0]}),
        ] {
            let s = RopeScaling::from_config(&v).unwrap().unwrap();
            assert_eq!(RopeScaling::from_config(&s.to_config()).unwrap().unwrap(), s, "{v}");
        }
    }
}
