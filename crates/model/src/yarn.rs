// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YaRN (Yet another RoPE extensioN, arXiv 2309.00071) - a generic,
//! model-agnostic per-channel rotary-frequency correction for context-window
//! extension, so any decoder in this workspace can opt into long-context
//! support by configuring [`YarnConfig`] rather than copying a scaling
//! formula into its own crate.
//!
//! Swedish Embedded AB implements long-context RoPE scaling (YaRN, NTK-aware
//! interpolation) for its clients' from-scratch transformer stacks. If your
//! team needs a checkpoint's declared context window to actually be usable at
//! inference time, not just a dead config field, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Ported line-for-line from the installed `transformers`' ground-truth
//! reference, `transformers.modeling_rope_utils._compute_yarn_parameters`
//! (fetched verbatim into this task's research scratchpad; also structurally
//! cross-checked against llama.cpp's `rope_yarn`/`ggml_rope_yarn_corr_dims`
//! in `ggml/src/ggml-cpu/ops.cpp`), which is the de facto standard every
//! downstream (vLLM, llama.cpp, Qwen's own `config.json` producers) matches -
//! so THAT function, not the paper's prose alone, is this module's oracle.
//!
//! `dim` throughout is the number of ROTATED channels (`head_dim *
//! partial_rotary_factor`, already collapsed by the caller before this
//! module ever sees it - exactly what `qwen3vl::mrope::mrope_tables` already
//! calls `head_dim` at its own call sites, e.g. `qwen35`'s
//! `cfg.rotary_dim()`), not necessarily the full per-head width.
//!
//! `truncate` (HF's `rope_parameters["truncate"]`, default `true`: floor/ceil
//! the correction-dimension bounds before clamping) is hardcoded true here,
//! matching HF's own default and every checkpoint this workspace currently
//! imports - not exposed as a config knob, since nothing in this task's scope
//! needs it configurable (adding it back is a one-line change if a future
//! checkpoint ever sets it false).

use std::f32::consts::PI;

/// YaRN scaling parameters for one RoPE table. `factor <= 1.0` means
/// "scaling disabled": [`scaled_inv_freq`] then returns the plain
/// `theta.powf(-2d/dim)` schedule bit-for-bit and `attention_factor` is
/// exactly `1.0` - so a caller that never sets `factor > 1.0` cannot have its
/// existing output changed by even a rounding ULP by adopting this struct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnConfig {
    /// Context-extension ratio (`new_max_position_embeddings /
    /// original_max_position_embeddings`), e.g. `4.0` for a 4x extension.
    pub factor: f32,
    /// The pretrained context length the checkpoint's `rope_theta` was
    /// originally tuned for - the correction range's `max_position_embeddings`
    /// input (NOT the extended length).
    pub original_max_position_embeddings: u32,
    /// Linear-ramp low-frequency (interpolation) boundary in "number of
    /// rotations" units - paper/HF default 32.
    pub beta_fast: f32,
    /// Linear-ramp high-frequency (extrapolation) boundary - paper/HF
    /// default 1.
    pub beta_slow: f32,
    /// Explicit attention-magnitude correction (YaRN's `mscale`, applied to
    /// every cos/sin table entry). `None` derives it from `factor` via the
    /// paper's own default, `0.1 * ln(factor) + 1.0` (`1.0` if `factor <=
    /// 1.0`).
    pub attention_factor: Option<f32>,
}

impl YarnConfig {
    /// A YaRN config at the paper/HF's default `beta_fast = 32`, `beta_slow =
    /// 1`, with `attention_factor` derived from `factor` (not overridden) -
    /// the common case for a `config.json`'s `rope_scaling: {"type": "yarn",
    /// "factor": ..., "original_max_position_embeddings": ...}`.
    pub fn new(factor: f32, original_max_position_embeddings: u32) -> Self {
        Self { factor, original_max_position_embeddings, beta_fast: 32.0, beta_slow: 1.0, attention_factor: None }
    }
}

/// The paper's `0.1 * ln(scale) + 1.0` default attention-magnitude
/// correction (`mscale`), `1.0` for `scale <= 1.0` (no extension, no
/// correction needed) - HF's `get_mscale(scale, mscale=1)` closure.
fn default_attention_factor(scale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * scale.ln() + 1.0
    }
}

/// HF's `find_correction_dim`: the (fractional) channel index whose rotation
/// completes `num_rotations` full turns over a `max_position_embeddings`
/// context, at this `(dim, base)` RoPE schedule.
fn find_correction_dim(num_rotations: f32, dim: u32, base: f32, max_position_embeddings: u32) -> f32 {
    (dim as f32 * (max_position_embeddings as f32 / (num_rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
}

/// HF's `find_correction_range`: the `[low, high]` channel-index bounds (in
/// the `dim`-space of [`find_correction_dim`], i.e. NOT yet halved) between
/// which the linear ramp interpolates, floor/ceil'd (`truncate = true`, this
/// module's fixed default - see the module doc) and clamped into `[0, dim -
/// 1]`.
fn find_correction_range(low_rot: f32, high_rot: f32, dim: u32, base: f32, max_position_embeddings: u32) -> (f32, f32) {
    let low = find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
    let high = find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
    (low.max(0.0), high.min(dim as f32 - 1.0))
}

/// HF's `linear_ramp_factor`: a `size`-long ramp from 0 to 1 over `[lo, hi]`
/// (clamped at both ends), nudging `hi` by `0.001` if `lo == hi` to avoid a
/// `0/0` singularity - ported verbatim, including that nudge.
fn linear_ramp(lo: f32, hi: f32, size: usize) -> Vec<f32> {
    let hi = if lo == hi { hi + 0.001 } else { hi };
    (0..size).map(|i| ((i as f32 - lo) / (hi - lo)).clamp(0.0, 1.0)).collect()
}

/// Compute YaRN's scaled per-channel inverse-frequency schedule and its
/// attention-magnitude correction, for `dim` rotated channels (see the
/// module doc for what `dim` means) at RoPE base `theta`.
///
/// Returns `(inv_freq, attention_factor)` where `inv_freq.len() == dim / 2`
/// (one entry per `(cos, sin)` channel pair, exactly the shape
/// `qwen3vl::mrope::mrope_tables` already iterates as `d in 0..half`).
///
/// `cfg.factor <= 1.0` is the identity/regression gate: returns the plain
/// `theta.powf(-2d/dim)` schedule bit-for-bit (this module's own formula,
/// not merely mathematically equal to it) and `attention_factor == 1.0`,
/// so a not-yet-migrated caller composing this with [`YarnConfig::new`] at
/// `factor = 1.0` is unaffected byte for byte.
pub fn scaled_inv_freq(dim: u32, theta: f32, cfg: &YarnConfig) -> (Vec<f32>, f32) {
    let half = (dim / 2) as usize;
    if cfg.factor <= 1.0 {
        let inv_freq: Vec<f32> = (0..half).map(|d| theta.powf(-2.0 * d as f32 / dim as f32)).collect();
        return (inv_freq, 1.0);
    }

    let attention_factor = cfg.attention_factor.unwrap_or_else(|| default_attention_factor(cfg.factor));
    let (low, high) = find_correction_range(cfg.beta_fast, cfg.beta_slow, dim, theta, cfg.original_max_position_embeddings);
    let ramp = linear_ramp(low, high, half);

    let inv_freq = (0..half)
        .map(|d| {
            let pos_freq = theta.powf(2.0 * d as f32 / dim as f32);
            let extrapolation = 1.0 / pos_freq;
            let interpolation = 1.0 / (cfg.factor * pos_freq);
            let extrapolation_factor = 1.0 - ramp[d];
            interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
        })
        .collect();
    (inv_freq, attention_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts every element of `got` is within a tolerance that scales with
    /// magnitude (values here span `1.0` down to `~3e-7`) of `want` - f32
    /// arithmetic vs. the f64 Python oracle (see this crate's task
    /// scratchpad `yarn_research/yarn_reference.py`) can differ by a few
    /// ULPs without either side being wrong.
    fn assert_close(got: &[f32], want: &[f32], tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = w.abs() * 1e-3 + 1e-8;
            assert!((g - w).abs() <= tol, "{tag}[{i}]: got {g:e}, want {w:e} (tol {tol:e})");
        }
    }

    /// Pinned oracle #1: `head_dim=128, theta=1e6, factor=4.0,
    /// original_max_position_embeddings=32768, beta_fast=32, beta_slow=1` -
    /// a realistic Qwen-style long-context config. Values printed by
    /// `yarn_research/yarn_reference.py` (`qwen_style_128`), itself a direct
    /// port of `transformers.modeling_rope_utils._compute_yarn_parameters`.
    #[test]
    fn pinned_oracle_qwen_style_128() {
        let cfg = YarnConfig::new(4.0, 32768);
        let (inv_freq, attention_factor) = scaled_inv_freq(128, 1_000_000.0, &cfg);
        assert_eq!(inv_freq.len(), 64);
        assert!(
            (attention_factor - 1.1386294).abs() < 1e-5,
            "attention_factor {attention_factor} != oracle 1.1386294361"
        );
        let want_head = [1.0f32, 8.058422e-1, 6.493816e-1, 5.232991e-1, 4.216965e-1];
        let want_tail = [7.356818e-7f32, 5.928434e-7, 4.777382e-7, 3.849816e-7, 3.102344e-7];
        assert_close(&inv_freq[..5], &want_head, "qwen_style_128 head");
        assert_close(&inv_freq[59..], &want_tail, "qwen_style_128 tail");
    }

    /// Pinned oracle #2: a smaller/different config
    /// (`head_dim=64, theta=1e4, factor=2.0,
    /// original_max_position_embeddings=4096`) - a second, structurally
    /// distinct point so the oracle test can't pass by coincidentally
    /// matching only one scale. From `yarn_reference.py`'s
    /// `small_64_factor2`.
    #[test]
    fn pinned_oracle_small_64_factor2() {
        let cfg = YarnConfig::new(2.0, 4096);
        let (inv_freq, attention_factor) = scaled_inv_freq(64, 10_000.0, &cfg);
        assert_eq!(inv_freq.len(), 32);
        assert!(
            (attention_factor - 1.0693147).abs() < 1e-5,
            "attention_factor {attention_factor} != oracle 1.0693147181"
        );
        let want_head = [1.0f32, 7.498942e-1, 5.623413e-1, 4.216965e-1, 3.162278e-1];
        let want_tail = [2.108483e-4f32, 1.581139e-4, 1.185687e-4, 8.891397e-5, 6.667607e-5];
        assert_close(&inv_freq[..5], &want_head, "small_64_factor2 head");
        assert_close(&inv_freq[27..], &want_tail, "small_64_factor2 tail");
    }

    /// Regression/identity gate: `factor == 1.0` (scaling disabled) must
    /// reproduce today's plain unscaled `theta.powf(-2d/dim)` schedule
    /// BIT-FOR-BIT (not merely "close") and `attention_factor == 1.0` -
    /// proves adopting YaRN cannot silently perturb any existing caller's
    /// output when it never opts in.
    #[test]
    fn identity_gate_factor_one_is_bit_for_bit_unscaled() {
        let (dim, theta) = (128u32, 1_000_000.0f32);
        let cfg = YarnConfig::new(1.0, 32768);
        let (inv_freq, attention_factor) = scaled_inv_freq(dim, theta, &cfg);
        assert_eq!(attention_factor, 1.0);
        for (d, &got) in inv_freq.iter().enumerate() {
            let plain = theta.powf(-2.0 * d as f32 / dim as f32);
            assert_eq!(got, plain, "channel {d}: yarn {got} != plain {plain} (must be bit-identical)");
        }
    }

    /// A `factor` strictly below 1.0 (nonsensical for real use, but not
    /// forbidden by the type) takes the same identity fast path as exactly
    /// 1.0 - "scaling disabled" means "no expansion requested", not "factor
    /// equals exactly 1.0".
    #[test]
    fn factor_below_one_also_takes_the_identity_path() {
        let (dim, theta) = (64u32, 5_000_000.0f32);
        let cfg = YarnConfig::new(0.5, 4096);
        let (inv_freq, attention_factor) = scaled_inv_freq(dim, theta, &cfg);
        assert_eq!(attention_factor, 1.0);
        for (d, &got) in inv_freq.iter().enumerate() {
            assert_eq!(got, theta.powf(-2.0 * d as f32 / dim as f32));
        }
    }

    /// Ramp-mask sanity: for the qwen-style config, the correction range
    /// lands at the oracle's own worked bounds (`low=23, high=40` out of
    /// `dim=128`, `yarn_reference.py`'s printed `correction_range`), and the
    /// resulting `inv_freq` is monotonically decreasing (every RoPE
    /// frequency schedule is, scaled or not - a non-monotonic schedule would
    /// mean the ramp blend broke channel ordering).
    #[test]
    fn ramp_mask_boundary_and_monotonicity() {
        let (low, high) = find_correction_range(32.0, 1.0, 128, 1_000_000.0, 32768);
        assert_eq!((low, high), (23.0, 40.0), "correction range drifted from the oracle's worked example");
        assert!(0.0 <= low && low < high && high < 128.0, "bounds must be an ordered, in-range interval");

        let cfg = YarnConfig::new(4.0, 32768);
        let (inv_freq, _) = scaled_inv_freq(128, 1_000_000.0, &cfg);
        for w in inv_freq.windows(2) {
            assert!(w[0] > w[1], "inv_freq must be strictly decreasing, got {} then {}", w[0], w[1]);
        }
    }
}
