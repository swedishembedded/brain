// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for steering audio-visual
// diffusion models toward their prompts for its clients. If your team needs
// expertise in diffusion guidance then you can procure our services by
// sending an email to info@swedishembedded.com.

//! LTX-2.5's multimodal guidance (`ltx_core.components.guiders.MultiModalGuider`).
//!
//! One step runs up to four forwards, each a guidance PASS
//! (`ltx_pipelines.utils.denoisers._guided_denoise`): the conditional one; a
//! negative-prompt one when either stream's CFG scale is not 1; an STG pass
//! that bypasses self-attention in the chosen blocks of each stream whose
//! STG scale is not 0; and an isolated-modality pass that cuts both
//! audio<->video directions when either stream's modality scale is not 1.
//! Each stream then combines its own share of those passes ON THE x0
//! ESTIMATE ([`GuiderParams::combine`]):
//!
//! ```text
//! pred = cond + (cfg - 1)(cond - uncond) + stg (cond - perturbed) + (modality - 1)(cond - isolated)
//! ```
//!
//! and, with `rescale` set, scales `pred` toward `cond`'s standard deviation -
//! measured over the tokens being denoised only, so frozen conditioning and
//! reference tokens do not skew it. CFG alone is linear, so it could equally
//! act on the velocity; the rescale is not, which is why every term acts on
//! x0.

use crate::block::Perturbations;

/// One stream's guidance (`MultiModalGuiderParams`). The neutral value of
/// every term is the [`Default`]: a single conditional forward.
#[derive(Clone, Debug, PartialEq)]
pub struct GuiderParams {
    /// Classifier-free guidance against the negative prompt; 1 is off.
    pub cfg: f32,
    /// Spatio-temporal guidance against the self-attention-bypassed pass; 0
    /// is off.
    pub stg: f32,
    /// The blocks whose self-attention the STG pass bypasses.
    pub stg_blocks: Vec<usize>,
    /// How far `pred` is scaled back toward `cond`'s standard deviation, in
    /// `[0, 1]`; 0 is off.
    pub rescale: f32,
    /// Guidance against the pass with the audio<->video cross-attention cut
    /// (a2v for video, v2a for audio); 1 is off.
    pub modality: f32,
    /// Reuse the last x0 on all but every `skip_step + 1`-th step; 0 guides
    /// every step.
    pub skip_step: usize,
}

impl Default for GuiderParams {
    fn default() -> GuiderParams {
        GuiderParams { cfg: 1.0, stg: 0.0, stg_blocks: Vec::new(), rescale: 0.0, modality: 1.0, skip_step: 0 }
    }
}

impl GuiderParams {
    /// Classifier-free guidance alone.
    pub fn cfg(scale: f32) -> GuiderParams {
        GuiderParams { cfg: scale, ..GuiderParams::default() }
    }

    /// The reference pipeline's video guidance for a non-distilled LTX-2.3+
    /// checkpoint (`LTX_2_3_PARAMS.video_guider_params`).
    pub fn reference_video() -> GuiderParams {
        GuiderParams { cfg: 3.0, stg: 1.0, stg_blocks: vec![28], rescale: 0.7, modality: 3.0, skip_step: 0 }
    }

    /// The reference pipeline's audio guidance (`LTX_2_3_PARAMS.audio_guider_params`).
    pub fn reference_audio() -> GuiderParams {
        GuiderParams { cfg: 7.0, ..GuiderParams::reference_video() }
    }

    pub fn needs_uncond(&self) -> bool {
        self.cfg != 1.0
    }

    pub fn needs_perturbed(&self) -> bool {
        self.stg != 0.0
    }

    pub fn needs_isolated(&self) -> bool {
        self.modality != 1.0
    }

    /// Whether step `step` reuses the previous x0 instead of guiding.
    pub fn skips(&self, step: usize) -> bool {
        self.skip_step != 0 && !step.is_multiple_of(self.skip_step + 1)
    }

    /// Combine one stream's passes into its guided x0. `cond` is the
    /// conditional pass; each other pass is given exactly when this stream's
    /// term needs it. `denoise_mask` is one weight per token (`channels`
    /// values each), non-zero where the token is being denoised; `None`
    /// measures the rescale over every token.
    pub fn combine(&self, cond: &[f32], uncond: Option<&[f32]>, perturbed: Option<&[f32]>, isolated: Option<&[f32]>, denoise_mask: Option<&[f32]>, channels: usize) -> Vec<f32> {
        fn term<'a>(scale: f32, other: Option<&'a [f32]>, what: &str) -> Option<(f32, &'a [f32])> {
            (scale != 0.0).then(|| (scale, other.unwrap_or_else(|| panic!("guidance: the {what} pass was not run"))))
        }
        let terms: Vec<(f32, &[f32])> = [term(self.cfg - 1.0, uncond, "negative-prompt"), term(self.stg, perturbed, "STG"), term(self.modality - 1.0, isolated, "isolated-modality")].into_iter().flatten().collect();
        let mut pred: Vec<f32> = cond.to_vec();
        for (scale, other) in &terms {
            assert_eq!(other.len(), cond.len(), "guidance: every pass predicts the same tokens");
            for ((p, &c), &o) in pred.iter_mut().zip(cond).zip(other.iter()) {
                *p += scale * (c - o);
            }
        }
        if self.rescale != 0.0 {
            let active = |i: usize| denoise_mask.is_none_or(|m| m[i / channels] != 0.0);
            // A constant prediction has no spread to rescale; the reference
            // would divide by zero there.
            let factor = match (unbiased_std(cond, &active), unbiased_std(&pred, &active)) {
                (Some(c), Some(p)) if p > 0.0 => c / p,
                _ => 1.0,
            };
            let factor = self.rescale * factor + (1.0 - self.rescale);
            pred.iter_mut().for_each(|p| *p *= factor);
        }
        pred
    }
}

/// The unbiased standard deviation of `x` over its `active` elements, in
/// f64; `None` with fewer than two (`_masked_std`'s `count > 1` guard).
fn unbiased_std(x: &[f32], active: &dyn Fn(usize) -> bool) -> Option<f32> {
    let values = || x.iter().enumerate().filter(|(i, _)| active(*i)).map(|(_, &v)| v as f64);
    let n = values().count();
    if n < 2 {
        return None;
    }
    let mean = values().sum::<f64>() / n as f64;
    let ss: f64 = values().map(|v| (v - mean).powi(2)).sum();
    Some((ss / (n - 1) as f64).sqrt() as f32)
}

/// Both streams' guidance for one generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Guidance {
    pub video: GuiderParams,
    pub audio: GuiderParams,
}

impl Guidance {
    /// The same classifier-free scale on both streams, nothing else.
    pub fn cfg(scale: f32) -> Guidance {
        Guidance { video: GuiderParams::cfg(scale), audio: GuiderParams::cfg(scale) }
    }

    /// The reference pipeline's guidance for a non-distilled checkpoint.
    pub fn reference() -> Guidance {
        Guidance { video: GuiderParams::reference_video(), audio: GuiderParams::reference_audio() }
    }

    /// The reference's high-quality preset (`LTX_2_3_HQ_PARAMS`, which its
    /// res_2s pipeline runs): CFG 3 video / 7 audio and modality guidance 3
    /// as [`Self::reference`], no STG, rescale 0.45 video / 1.0 audio.
    pub fn hq() -> Guidance {
        let base = GuiderParams { stg: 0.0, stg_blocks: Vec::new(), ..GuiderParams::reference_video() };
        Guidance { video: GuiderParams { rescale: 0.45, ..base.clone() }, audio: GuiderParams { cfg: 7.0, rescale: 1.0, ..base } }
    }

    /// The passes one step runs, in order, beyond the conditional one -
    /// `audio` when the audio stream is generated alongside.
    pub fn passes(&self, audio: bool) -> Vec<Pass> {
        let streams: Vec<&GuiderParams> = if audio { vec![&self.video, &self.audio] } else { vec![&self.video] };
        let mut passes = Vec::new();
        if streams.iter().any(|g| g.needs_uncond()) {
            passes.push(Pass::Uncond);
        }
        if streams.iter().any(|g| g.needs_perturbed()) {
            let blocks = |g: &GuiderParams| if g.needs_perturbed() { g.stg_blocks.clone() } else { Vec::new() };
            passes.push(Pass::Perturbed(Perturbations { video_self_attn_blocks: blocks(&self.video), audio_self_attn_blocks: if audio { blocks(&self.audio) } else { Vec::new() }, cross_modal: false }));
        }
        if audio && streams.iter().any(|g| g.needs_isolated()) {
            passes.push(Pass::Isolated(Perturbations { cross_modal: true, ..Default::default() }));
        }
        passes
    }

    /// A named starting point: `none` (one conditional forward per step - the
    /// distilled checkpoint), `reference` ([`Self::reference`]) or `hq`
    /// ([`Self::hq`]).
    pub fn preset(name: &str) -> Result<Guidance, String> {
        match name {
            "none" => Ok(Guidance::default()),
            "reference" => Ok(Guidance::reference()),
            "hq" => Ok(Guidance::hq()),
            other => Err(format!("unknown guidance preset '{other}' (none, reference, hq)")),
        }
    }

    /// Set one named term from its textual value - the one parser behind the
    /// CLI flags and the capability parameters, which share these names
    /// (`guidance`, `audio_guidance`, `stg`, `audio_stg`, `stg_blocks`,
    /// `rescale`, `audio_rescale`, `modality_guidance`,
    /// `audio_modality_guidance`). An unprefixed term sets both streams
    /// except `modality_guidance`, which is the video stream's (a2v). Returns
    /// `false` for a name that is not a guidance term.
    pub fn set(&mut self, name: &str, value: &str) -> Result<bool, String> {
        let num = || value.trim().parse::<f32>().map_err(|e| format!("{name}: '{value}': {e}"));
        match name {
            "guidance" => (self.video.cfg, self.audio.cfg) = (num()?, num()?),
            "audio_guidance" => self.audio.cfg = num()?,
            "stg" => (self.video.stg, self.audio.stg) = (num()?, num()?),
            "audio_stg" => self.audio.stg = num()?,
            "rescale" => (self.video.rescale, self.audio.rescale) = (num()?, num()?),
            "audio_rescale" => self.audio.rescale = num()?,
            "modality_guidance" => self.video.modality = num()?,
            "audio_modality_guidance" => self.audio.modality = num()?,
            "stg_blocks" => {
                let blocks: Vec<usize> = value.split(',').map(|b| b.trim().parse::<usize>().map_err(|e| format!("stg_blocks: '{b}': {e}"))).collect::<Result<_, _>>()?;
                (self.video.stg_blocks, self.audio.stg_blocks) = (blocks.clone(), blocks);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Refuse a combination no pass can carry out.
    pub fn validate(&self) -> Result<(), String> {
        for (name, p) in [("video", &self.video), ("audio", &self.audio)] {
            if !(0.0..=1.0).contains(&p.rescale) {
                return Err(format!("{name} rescale must lie in 0..1, got {}", p.rescale));
            }
            if p.needs_perturbed() && p.stg_blocks.is_empty() {
                return Err(format!("{name} STG is on but stg_blocks names no block"));
            }
        }
        Ok(())
    }

    /// Whether some stream guides against the negative prompt, which must
    /// then be encoded too.
    pub fn needs_uncond(&self, audio: bool) -> bool {
        self.video.needs_uncond() || (audio && self.audio.needs_uncond())
    }

    /// The forwards one guided step runs: the conditional one plus
    /// [`Self::passes`].
    pub fn forwards_per_step(&self, audio: bool) -> usize {
        1 + self.passes(audio).len()
    }
}

/// One guidance pass beyond the conditional one.
#[derive(Clone, Debug, PartialEq)]
pub enum Pass {
    /// The negative prompt, unperturbed.
    Uncond,
    /// The prompt, with STG's self-attention bypass.
    Perturbed(Perturbations),
    /// The prompt, with both audio<->video directions cut.
    Isolated(Perturbations),
}

#[cfg(test)]
mod tests {
    use super::*;

    // `MultiModalGuider(MultiModalGuiderParams(cfg_scale=3, stg_scale=1,
    // rescale_scale=0.7, modality_scale=3)).calculate(cond, uncond, ptb, mod,
    // mask)` over 4 tokens of 3 channels, the second token frozen - computed
    // by ltx_core itself.
    const COND: [f32; 12] = [-0.43, -0.049999997, 0.33, -0.4, -0.02, 0.36, -0.37, 0.010000002, 0.39, -0.34, 0.039999995, 0.42000002];
    const UNCOND: [f32; 12] = [-0.06, 0.33, 0.72, 0.0, 0.39, 0.78, 0.06, 0.45, 0.84000003, 0.11999999, 0.51, 0.9];
    const PTB: [f32; 12] = [0.31, 0.71, 1.1099999, 0.4, 0.8, 1.1999999, 0.49, 0.89, 1.29, 0.58, 0.98, 1.3799999];
    const MOD: [f32; 12] = [1.05, -0.43, -0.010000005, 1.1999999, -0.28, 0.13999999, 1.3499999, -0.13, 0.28999996, 1.5, 0.01999998, 0.44];
    const OUT_MASKED: [f32; 12] = [-1.9883058, -0.33070382, -0.22455193, -2.1230369, -0.46543503, -0.3592831, -2.2577682, -0.60016626, -0.4940144, -2.3924992, -0.7348974, -0.62874544];
    const OUT: [f32; 12] = [-1.9897224, -0.3309394, -0.22471191, -2.1245494, -0.46576664, -0.35953906, -2.2593765, -0.6005938, -0.49436635, -2.3942037, -0.73542094, -0.6291934];

    fn params() -> GuiderParams {
        GuiderParams { cfg: 3.0, stg: 1.0, stg_blocks: vec![28], rescale: 0.7, modality: 3.0, skip_step: 0 }
    }

    #[test]
    fn every_term_and_the_masked_rescale_match_the_reference() {
        let close = |got: &[f32], want: &[f32]| got.iter().zip(want).all(|(a, b)| (a - b).abs() <= 2e-6 * b.abs().max(1.0));
        let masked = params().combine(&COND, Some(&UNCOND), Some(&PTB), Some(&MOD), Some(&[1.0, 0.0, 1.0, 1.0]), 3);
        assert!(close(&masked, &OUT_MASKED), "{masked:?}");
        let whole = params().combine(&COND, Some(&UNCOND), Some(&PTB), Some(&MOD), None, 3);
        assert!(close(&whole, &OUT), "{whole:?}");
    }

    /// `LTX_2_3_HQ_PARAMS`: no STG pass, so an hq step runs the negative
    /// prompt and the isolated-modality pass only.
    #[test]
    fn the_hq_preset_is_the_references() {
        let hq = Guidance::preset("hq").unwrap();
        assert_eq!((hq.video.cfg, hq.video.stg, hq.video.rescale, hq.video.modality), (3.0, 0.0, 0.45, 3.0));
        assert_eq!((hq.audio.cfg, hq.audio.stg, hq.audio.rescale, hq.audio.modality), (7.0, 0.0, 1.0, 3.0));
        assert_eq!(hq.passes(true).len(), 2);
    }

    #[test]
    fn neutral_guidance_is_the_conditional_pass() {
        assert_eq!(GuiderParams::default().combine(&COND, None, None, None, None, 3), COND);
        assert!(Guidance::default().passes(true).is_empty());
    }

    /// A pass runs once for both streams whenever either needs it; the
    /// isolated-modality pass exists only with an audio stream; STG perturbs
    /// only the streams that ask for it.
    #[test]
    fn passes_are_shared_across_streams() {
        let g = Guidance { video: GuiderParams { stg: 1.0, stg_blocks: vec![28], ..Default::default() }, audio: GuiderParams { cfg: 7.0, modality: 3.0, ..Default::default() } };
        let passes = g.passes(true);
        assert_eq!(passes.len(), 3);
        assert_eq!(passes[0], Pass::Uncond);
        assert_eq!(passes[1], Pass::Perturbed(Perturbations { video_self_attn_blocks: vec![28], audio_self_attn_blocks: vec![], cross_modal: false }));
        assert!(matches!(passes[2], Pass::Isolated(Perturbations { cross_modal: true, .. })));
        assert_eq!(Guidance::reference().passes(false).len(), 2, "video alone: negative prompt and STG, no isolated pass");
    }
}
