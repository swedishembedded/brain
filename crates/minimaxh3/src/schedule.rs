// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's dual rectified-flow schedule: **two independent** shifted-
//! sigma Euler schedules - `shift=12.0` for video, `shift=3.0` for audio
//! (`model_index.json`'s own `sigma_shift_scales`, real values, not a guess)
//! - driven from **one** transformer forward that predicts both a video and
//! an audio velocity per step. This is never two separate `denoise()` loops:
//! [`DualSchedule::step`] takes both predictions and advances both cursors
//! together, structurally preventing the two schedules from drifting out of
//! lockstep (a real correctness property, not just a convenience - a caller
//! that only ever gets one prediction per forward has no way to accidentally
//! step one modality twice for the other's one).
//!
//! Reuses `diffusion::scheduler::{FlowMatchConfig, FlowMatchEulerScheduler}`
//! directly rather than re-deriving the shift transform (`σ' = shift·σ /
//! (1 + (shift-1)·σ)`) - the exact formula every other flow-matching model in
//! this workspace (Z-Image, FLUX.2, Wan) already uses via that module, at
//! different shift constants. Nothing here duplicates that math.
//!
//! **Open convention questions** (recorded here AND in the roadmap ledger,
//! not assumed from the Z-Image/FLUX.2 precedent - MiniMax-H3's own
//! `diffusers` reference has to confirm each of these before real-weight
//! parity can be claimed):
//! - the base (pre-shift) sigma spacing (`linspace(1, 1/n, n)`, matching
//!   [`diffusion::scheduler::default_z_image_sigmas`], is assumed here as a
//!   documented DEFAULT, not a verified fact);
//! - `num_train_timesteps` (assumed `1000`, the Z-Image/FLUX.2/Wan value -
//!   H3's own config has not been read yet);
//! - `invert_sigmas` (assumed `false` - MiniMax Music 3 is the one model in
//!   this workspace where that assumption was wrong, so it is not free to
//!   skip checking);
//! - whether the two modalities' schedules ever need DIFFERENT base sigma
//!   spacings or step counts (assumed here to share both - only the shift
//!   differs), or only different shifts as this module currently assumes.

use diffusion::scheduler::{FlowMatchConfig, FlowMatchEulerScheduler};

/// `model_index.json`'s `sigma_shift_scales.video` - real, not assumed.
pub const H3_VIDEO_SHIFT: f32 = 12.0;
/// `model_index.json`'s `sigma_shift_scales.audio` - real, not assumed.
pub const H3_AUDIO_SHIFT: f32 = 3.0;

/// The two flow-matching schedules H3's denoise loop steps together, one
/// video prediction and one audio prediction per forward.
pub struct DualSchedule {
    video: FlowMatchEulerScheduler,
    audio: FlowMatchEulerScheduler,
}

impl DualSchedule {
    /// `num_train_timesteps`/`invert_sigmas` shared by both modalities (only
    /// the shift differs, per this module's own doc) - see the doc's "open
    /// convention questions" for what is still an assumption here.
    pub fn new(num_train_timesteps: u32, invert_sigmas: bool) -> DualSchedule {
        let video_cfg = FlowMatchConfig { num_train_timesteps, shift: H3_VIDEO_SHIFT, invert_sigmas };
        let audio_cfg = FlowMatchConfig { num_train_timesteps, shift: H3_AUDIO_SHIFT, invert_sigmas };
        DualSchedule { video: FlowMatchEulerScheduler::new(video_cfg), audio: FlowMatchEulerScheduler::new(audio_cfg) }
    }

    /// Build both schedules from the SAME base (pre-shift) sigma spacing -
    /// each applies its own shift internally
    /// ([`FlowMatchEulerScheduler::set_timesteps`]), so the two end up with
    /// different sigma/timestep sequences despite sharing `sigmas_in`.
    pub fn set_timesteps(&mut self, sigmas_in: &[f32]) {
        self.video.set_timesteps(sigmas_in);
        self.audio.set_timesteps(sigmas_in);
    }

    /// The number of Euler steps this schedule will take (`sigmas().len() - 1`
    /// on either half - both always agree, since [`set_timesteps`](Self::set_timesteps)
    /// builds them from the same input length).
    pub fn num_steps(&self) -> usize {
        self.video.sigmas().len().saturating_sub(1)
    }

    /// One lockstep Euler step for BOTH modalities, from the two velocity
    /// predictions one transformer forward produced. Returns `(video_next,
    /// audio_next)`.
    pub fn step(&mut self, video_pred: &[f32], video_sample: &[f32], audio_pred: &[f32], audio_sample: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let v = self.video.step(video_pred, video_sample);
        let a = self.audio.step(audio_pred, audio_sample);
        (v, a)
    }

    /// The video half's `[timestep_video, timestep_audio]` pair at the
    /// CURRENT (next-to-take) step - what the transformer's per-modality
    /// timestep embedding reads before calling [`Self::step`]. Panics if the
    /// schedule is exhausted or [`Self::set_timesteps`] was never called.
    pub fn current_timesteps(&self, step_index: usize) -> (f32, f32) {
        (self.video.timesteps()[step_index], self.audio.timesteps()[step_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-form shift formula, evaluated by hand at a few points, as
    /// the independent oracle this module's `DualSchedule` is checked
    /// against - never re-deriving `diffusion::scheduler`'s own formula, only
    /// confirming H3's two shift CONSTANTS produce the values that formula
    /// predicts.
    fn shift_ref(shift: f64, sigma: f64) -> f64 {
        shift * sigma / (1.0 + (shift - 1.0) * sigma)
    }

    #[test]
    fn video_and_audio_shifts_match_the_real_config() {
        assert_eq!(H3_VIDEO_SHIFT, 12.0);
        assert_eq!(H3_AUDIO_SHIFT, 3.0);
    }

    #[test]
    fn the_two_schedules_diverge_from_the_same_base_sigmas() {
        let mut sched = DualSchedule::new(1000, false);
        let base = diffusion::scheduler::default_z_image_sigmas(4);
        sched.set_timesteps(&base);

        assert_eq!(sched.num_steps(), 4, "N base sigmas -> N steps (+1 terminal sigma)");

        // Same base sigma, different shift -> different shifted sigma, at
        // every non-degenerate point (sigma=1 and the terminal 0 both fix to
        // the same value under ANY shift, so mid-schedule points are the
        // real check).
        let (t_v0, t_a0) = sched.current_timesteps(1);
        assert_ne!(t_v0, t_a0, "video (shift=12) and audio (shift=3) must diverge at the same base sigma");

        // Cross-check against the closed-form formula directly, not just
        // "the two differ from each other" (which a bug swapping both shifts
        // by the same wrong constant could still pass).
        let s1 = base[1] as f64;
        let expect_v = (shift_ref(H3_VIDEO_SHIFT as f64, s1) * 1000.0) as f32;
        let expect_a = (shift_ref(H3_AUDIO_SHIFT as f64, s1) * 1000.0) as f32;
        assert!((t_v0 - expect_v).abs() < 1e-3, "video timestep {t_v0} vs closed-form {expect_v}");
        assert!((t_a0 - expect_a).abs() < 1e-3, "audio timestep {t_a0} vs closed-form {expect_a}");
    }

    #[test]
    fn step_advances_both_cursors_in_lockstep() {
        let mut sched = DualSchedule::new(1000, false);
        sched.set_timesteps(&diffusion::scheduler::default_z_image_sigmas(3));

        let (mut vs, mut as_) = (vec![1.0f32; 4], vec![1.0f32; 4]);
        for _ in 0..sched.num_steps() {
            let (v_next, a_next) = sched.step(&vec![0.1; 4], &vs, &vec![0.2; 4], &as_);
            vs = v_next;
            as_ = a_next;
        }
        assert!(vs.iter().all(|v| v.is_finite()));
        assert!(as_.iter().all(|v| v.is_finite()));
        // Different velocities (0.1 vs 0.2) over the SAME dt sequence (shared
        // base sigmas, though shifted differently) must land somewhere
        // different - a copy-paste bug feeding one modality's prediction to
        // both would make these equal.
        assert_ne!(vs, as_);
    }

    #[test]
    #[should_panic(expected = "step() called")]
    fn stepping_past_the_schedule_panics() {
        let mut sched = DualSchedule::new(1000, false);
        sched.set_timesteps(&diffusion::scheduler::default_z_image_sigmas(1));
        let (v, a) = (vec![0.0f32; 2], vec![0.0f32; 2]);
        let _ = sched.step(&v, &v, &a, &a);
        let _ = sched.step(&v, &v, &a, &a); // one step configured; the second must panic
    }
}
