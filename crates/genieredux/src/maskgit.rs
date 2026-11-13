// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for faithful ports of iterative
// token samplers for its clients. If your team needs expertise in generative
// world models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The guided-MaskGIT sampler, ported line for line in behaviour from the
//! reference `Dynamics.sample` in GenieRedux `models/dynamics.py` (with its
//! helpers `mask_ratio_schedule`, `gumbel_sample`, `gumbel_noise` and
//! `Dynamics.compute_confidence_scores` in the same file).
//!
//! The rules, each as the reference has them:
//!
//! * Every pass starts from the previous pass's predictions with the
//!   `int(mask_ratio * num_tokens)` LEAST confident positions replaced by the
//!   mask token (pass 0: everything masked). `mask_ratio` is
//!   [`MaskSchedule::mask_ratio`] at `ratio = (pass + 1) / inference_steps`,
//!   clamped to `[1e-6, 1]`; the last pass therefore re-masks nothing.
//! * Every pass overwrites EVERY position with its own prediction. The
//!   reference's `torch.where(mask, pred, video_token_ids)` is commented out in
//!   the source, so positions kept by the schedule are re-predicted too, not
//!   frozen.
//! * Confidence is `1 - softmax(logits)[pred]`, kept only for the positions that
//!   were masked in that pass (`-1e4` elsewhere); it is computed on the raw
//!   logits, not the temperature-perturbed ones.
//! * Predictions are `argmax(logits + max(t, 1e-10) * gumbel)` with
//!   `t = sample_temperature * (steps_til_x0 / inference_steps)`, so the last
//!   pass is greedy; `gumbel = -ln(-ln(u + 1e-10) + 1e-10)`, `u ~ U[0,1)`.
//! * The model input is `cat(prime, video)[:, :-1]`: the last frame is dropped
//!   and the logits kept are the final `num_tokens` rows of what remains, so a
//!   frame is predicted from the frame before it.
//!
//! One deliberate choice where the reference is unspecified: `torch.topk`
//! leaves ties in the confidence ordering unspecified; here equal confidences
//! re-mask the lowest position first, which makes the sampler deterministic.
//! The reference reads its schedule, step count and temperature from
//! constructor attributes that its source stores as one-element tuples, so its
//! callers always pass them explicitly; [`MaskgitSettings`] is that explicit
//! set.

use crate::import::DynamicsConfig;
use crate::{dynamics_forward, DynamicsWeights};
use gpu_core::Gpu;

/// The reference `mask_ratio_schedule` methods.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskSchedule {
    Cosine,
    Uniform,
    /// `1 - ratio^exponent` (reference method `"pow<exponent>"`).
    Pow(f32),
    Log,
    Exp,
}

impl MaskSchedule {
    /// Parse the reference's method string: `cosine`, `uniform`, `log`, `exp`,
    /// `pow<exponent>`.
    pub fn parse(method: &str) -> Result<MaskSchedule, String> {
        match method {
            "cosine" => Ok(MaskSchedule::Cosine),
            "uniform" => Ok(MaskSchedule::Uniform),
            "log" => Ok(MaskSchedule::Log),
            "exp" => Ok(MaskSchedule::Exp),
            m => m
                .strip_prefix("pow")
                .and_then(|e| e.parse::<f32>().ok())
                .map(MaskSchedule::Pow)
                .ok_or_else(|| format!("invalid mask schedule `{method}` (cosine, uniform, log, exp, pow<exponent>)")),
        }
    }

    /// Fraction of the unknown tokens still masked after `ratio` of the passes,
    /// in `f32` like the reference's tensors, clamped to `[1e-6, 1]`.
    pub fn mask_ratio(self, ratio: f32, total_unknown: usize) -> f32 {
        let total = total_unknown as f32;
        let r = match self {
            MaskSchedule::Uniform => 1.0 - ratio,
            MaskSchedule::Pow(e) => 1.0 - ratio.powf(e),
            MaskSchedule::Cosine => (ratio * std::f32::consts::PI / 2.0).cos(),
            MaskSchedule::Log => -ratio.log2() / total.log2(),
            MaskSchedule::Exp => 1.0 - (-total.log2() * (1.0 - ratio)).exp2(),
        };
        r.clamp(1e-6, 1.0)
    }
}

/// Sampler settings the reference takes per call (checkpoint defaults: one
/// pass, cosine, temperature 1.0, which makes that single pass greedy).
#[derive(Clone, Copy, Debug)]
pub struct MaskgitSettings {
    pub inference_steps: usize,
    pub schedule: MaskSchedule,
    pub sample_temperature: f32,
}

/// SplitMix64: the sampler's noise stream and the session's seed stream.
pub(crate) struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)` with 24 bits, exact in `f32`.
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Reference `gumbel_sample` of one row.
fn gumbel_argmax(row: &[f32], temperature: f32, rng: &mut SplitMix64) -> u32 {
    let temp = temperature.max(1e-10);
    let log = |x: f32| (x + 1e-10).ln();
    let (mut best, mut best_v) = (0u32, f32::NEG_INFINITY);
    for (i, &l) in row.iter().enumerate() {
        let v = l + temp * -log(-log(rng.uniform()));
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

/// The MaskGIT loop over an arbitrary dynamics forward.
///
/// `forward` receives the current video tokens (`num_tokens`, the mask token
/// `n_codes` where re-masked) and returns the logits `[num_tokens, n_codes]` of
/// those positions. Returns the final tokens. `seed` drives the Gumbel noise.
pub fn maskgit_sample_with<F>(
    num_tokens: usize, n_codes: u32, settings: &MaskgitSettings, seed: u64, mut forward: F,
) -> Vec<u32>
where
    F: FnMut(&[u32]) -> Vec<f32>,
{
    let steps = settings.inference_steps;
    assert!(steps >= 1, "inference_steps must be at least 1");
    let nc = n_codes as usize;
    let mask_id = n_codes;
    let mut vids = vec![mask_id; num_tokens];
    let mut scores = vec![0.0f32; num_tokens];
    let mut mask = vec![true; num_tokens];
    let mut rng = SplitMix64(seed);

    for step in 0..steps {
        let steps_til_x0 = steps - (step + 1);
        if step != 0 {
            // The ratio is a Python float turned into an f32 tensor.
            let ratio = ((step + 1) as f64 / steps as f64) as f32;
            let num_mask = (settings.schedule.mask_ratio(ratio, num_tokens) * num_tokens as f32) as usize;
            // Highest score = least confident; stable, so ties keep position order.
            let mut order: Vec<usize> = (0..num_tokens).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            mask = vec![false; num_tokens];
            for &i in &order[..num_mask.min(num_tokens)] {
                mask[i] = true;
            }
            for (v, &m) in vids.iter_mut().zip(&mask) {
                if m {
                    *v = mask_id;
                }
            }
        }
        let logits = forward(&vids);
        assert_eq!(logits.len(), num_tokens * nc, "forward must return [num_tokens, n_codes] logits");
        let temperature = settings.sample_temperature * (steps_til_x0 as f32 / steps as f32);
        for (t, v) in vids.iter_mut().enumerate() {
            *v = gumbel_argmax(&logits[t * nc..(t + 1) * nc], temperature, &mut rng);
        }
        if step + 1 < steps {
            for t in 0..num_tokens {
                let row = &logits[t * nc..(t + 1) * nc];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = row.iter().map(|&l| (l - mx).exp()).sum();
                let prob = (row[vids[t] as usize] - mx).exp() / denom;
                scores[t] = if mask[t] { 1.0 - prob } else { -1e4 };
            }
        }
    }
    vids
}

/// Guided-MaskGIT sampling over the full dynamics forward: generates
/// `num_tokens` tokens (whole frames) conditioned on `prime_ids` and the
/// per-frame one-hot `actions` of the `t - 1` frames the model sees (`t` =
/// frames in `prime ++ generated`). Every pass recomputes the whole sequence;
/// see [`crate::StCache`] for the incremental path.
#[allow(clippy::too_many_arguments)]
pub fn maskgit_sample(
    gpu: &Gpu, prime_ids: &[u32], actions: &[f32], num_tokens: usize, h: u32, wd: u32,
    dw: &DynamicsWeights, dc: &DynamicsConfig, settings: &MaskgitSettings, seed: u64,
) -> Vec<u32> {
    let hw = (h * wd) as usize;
    let total = prime_ids.len() + num_tokens;
    assert!(total % hw == 0 && total / hw >= 2, "prime + generated must be at least two whole frames");
    let t_fwd = total / hw - 1; // the reference drops the last frame
    let nc = dc.n_codes as usize;
    maskgit_sample_with(num_tokens, dc.n_codes, settings, seed, |vids| {
        let mut input = Vec::with_capacity(total);
        input.extend_from_slice(prime_ids);
        input.extend_from_slice(vids);
        let logits = dynamics_forward(
            gpu, &input[..t_fwd * hw], actions, dw, 1, t_fwd as u32, h, wd, dc.dim, dc.heads,
            dc.head_dim, dc.n_codes, dc.code_dim, dc.num_actions,
        );
        logits[(t_fwd * hw - num_tokens) * nc..].to_vec()
    })
}
