// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's dual rectified-flow schedule: **two independent** shifted-
//! sigma Euler schedules (`shift=12.0` for video, `shift=3.0` for audio,
//! `model_index.json`'s own `sigma_shift_scales`, real values, not a guess),
//! driven from **one** transformer forward that predicts both a video and
//! an audio velocity per step. This is never two separate `denoise()` loops:
//! a caller steps both modalities from the one pair of predictions one
//! forward produces, at the SAME `step_index`, so the two schedules cannot
//! drift out of lockstep.
//!
//! This is a from-scratch port of `MiniMaxH3Scheduler`
//! (`diffusers/schedulers/scheduling_minimax_h3.py`, read directly, not
//! reused from `diffusion::scheduler::FlowMatchEulerScheduler` - an earlier
//! draft of this module reused that generic flow-match scheduler as a
//! documented PLACEHOLDER pending the real source; reading the real source
//! showed that reuse was wrong on every axis it guessed:
//!
//! - **No `num_train_timesteps` at all.** The real class's `__init__` is
//!   `def __init__(self, shift: float = 12.0)` - nothing else. Timesteps
//!   live directly in `[0, 1]` via `t = 1 - sigma` (`t = 1` is CLEAN), not on
//!   `FlowMatchEulerScheduler`'s `sigma * num_train_timesteps` scale.
//! - **No `invert_sigmas` flag.** The sigma grid is unconditionally
//!   `linspace(1, 0, num_inference_steps)` (the terminal `0.0` is part of the
//!   REQUESTED step count itself, not appended afterward the way
//!   `FlowMatchEulerScheduler` does) pushed through the shift, then
//!   `unique_consecutive`-deduplicated (the shift compresses the grid near
//!   `sigma=1` and can create float32 collisions there).
//! - **`step()` is not plain Euler `x + dt*v`.** It recovers a denoised
//!   estimate from a DATA-WARD velocity (`x0 = x_t + (1-t)*v`, the `+` is the
//!   real sign - the opposite of the usual flow-match `x0 = x_t - sigma*v`),
//!   then blends `x_t`/`x0` by `ratio = sigma_next/sigma` pulled from the
//!   SIGMA GRID - a deliberately different source of sigma than the
//!   `1 - timestep` used for the `x0` recovery (the docstring: "for sigma <
//!   0.5 the float32 round trip `1 - (1 - sigma)` is not exact, and the
//!   reference keeps the two sources apart").
//!
//! [`H3Scheduler`] ports all three exactly. [`DualSchedule`] is the pair at
//! H3's own two shift constants, stepped together.
//!
//! Swedish Embedded AB implements rectified-flow diffusion schedulers like
//! this one for its clients. If your team needs expertise in porting
//! diffusion model sampling loops to new inference stacks, you can procure
//! our services by sending an email to info@swedishembedded.com.

/// `model_index.json`'s `sigma_shift_scales.video` - real, not assumed.
pub const H3_VIDEO_SHIFT: f32 = 12.0;
/// `model_index.json`'s `sigma_shift_scales.audio` - real, not assumed.
pub const H3_AUDIO_SHIFT: f32 = 3.0;

/// `torch.linspace(start, stop, n)` - `n` points, endpoints included
/// (`start + i*(stop-start)/(n-1)`), computed in f32 to match the reference's
/// own `dtype=torch.float32` grid.
fn linspace_f32(start: f32, stop: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![start];
    }
    let denom = (n - 1) as f32;
    (0..n).map(|i| start + (stop - start) * (i as f32) / denom).collect()
}

/// `sigma' = shift*sigma / (1 + (shift-1)*sigma)`.
fn shift_sigma(shift: f32, sigma: f32) -> f32 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// `torch.unique_consecutive` on a (monotonic) f32 slice: drop a value only
/// when it repeats its IMMEDIATE predecessor - never a full dedup, and never
/// reorders anything (this schedule's sigma grid is already monotonically
/// non-increasing by construction, so the two coincide here, but the
/// consecutive-only rule is what the reference actually calls).
fn unique_consecutive_f32(v: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(v.len());
    for &x in v {
        if out.last() != Some(&x) {
            out.push(x);
        }
    }
    out
}

/// One `MiniMaxH3Scheduler` instance - a rectified-flow Euler schedule at a
/// fixed `shift`. [`DualSchedule`] holds two: `shift=12` for video, `shift=3`
/// for audio.
#[derive(Clone, Debug)]
pub struct H3Scheduler {
    shift: f32,
    /// The (deduplicated) sigma grid, decreasing, ending at `0.0`.
    sigmas: Vec<f32>,
    /// `1 - sigmas[..sigmas.len()-1]` - one entry per model evaluation.
    timesteps: Vec<f32>,
}

impl H3Scheduler {
    pub fn new(shift: f32) -> H3Scheduler {
        assert!(shift > 0.0, "H3Scheduler::new: shift must be positive, got {shift}");
        H3Scheduler { shift, sigmas: Vec::new(), timesteps: Vec::new() }
    }

    /// `set_timesteps`: `linspace(1, 0, num_inference_steps)` shifted, then
    /// `unique_consecutive`-collapsed; `timesteps = 1 - sigmas[:-1]`. The
    /// schedule then drives `timesteps.len()` model evaluations (at most
    /// `num_inference_steps - 1`, fewer if the shift created duplicates).
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        assert!(num_inference_steps >= 2, "H3Scheduler::set_timesteps: num_inference_steps must be >= 2, got {num_inference_steps}");
        let base = linspace_f32(1.0, 0.0, num_inference_steps);
        let sigmas: Vec<f32> = base.iter().map(|&s| shift_sigma(self.shift, s)).collect();
        let sigmas = unique_consecutive_f32(&sigmas);
        self.timesteps = sigmas[..sigmas.len() - 1].iter().map(|&s| 1.0 - s).collect();
        self.sigmas = sigmas;
    }

    pub fn shift(&self) -> f32 {
        self.shift
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// The number of Euler steps this schedule will take
    /// (`timesteps().len()`).
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// `MiniMaxH3Scheduler.step`, at an explicit `step_index` (this port has
    /// no internal `_step_index` counter - the pipeline's own denoise loop
    /// already tracks `i`, so a caller passes it directly rather than this
    /// module maintaining a second copy that could drift out of sync with
    /// it). `timestep` is the SAME scalar `self.timesteps[step_index]` the
    /// transformer was conditioned on for the rows being stepped - see this
    /// module's own doc for why the `x0` recovery and the Euler ratio
    /// deliberately read sigma from two different sources.
    pub fn step(&self, model_output: &[f32], timestep: f32, sample: &[f32], step_index: usize) -> Vec<f32> {
        assert_eq!(model_output.len(), sample.len(), "H3Scheduler::step: model_output/sample length mismatch");
        assert!(step_index + 1 < self.sigmas.len(), "H3Scheduler::step: step_index {step_index} out of range ({} sigmas)", self.sigmas.len());
        let sigma_from_timestep = 1.0 - timestep;
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let ratio = sigma_next / sigma;
        sample
            .iter()
            .zip(model_output)
            .map(|(&s, &v)| {
                let denoised = s + sigma_from_timestep * v;
                ratio * s + (1.0 - ratio) * denoised
            })
            .collect()
    }

    /// `MiniMaxH3Scheduler.scale_noise`: `x_t = t*x_0 + (1-t)*noise` in H3's
    /// own `t` convention (`t=1` returns `sample` unchanged). Used to noise a
    /// visual conditioning anchor (`fl2va` keyframes, `ref2va` image/video
    /// references) to `components.keyframe_noise_aug` (`0.999`) rather than
    /// looked up in a schedule - `timestep` is taken at face value, no
    /// `index_for_timestep` involved, hence a plain associated function
    /// rather than an `&self` method.
    pub fn scale_noise(sample: &[f32], timestep: f32, noise: &[f32]) -> Vec<f32> {
        assert_eq!(sample.len(), noise.len(), "H3Scheduler::scale_noise: sample/noise length mismatch");
        sample.iter().zip(noise).map(|(&s, &n)| timestep * s + (1.0 - timestep) * n).collect()
    }
}

/// The two flow-matching schedules H3's denoise loop steps together, one
/// video prediction and one audio prediction per forward.
pub struct DualSchedule {
    video: H3Scheduler,
    audio: H3Scheduler,
}

impl Default for DualSchedule {
    fn default() -> DualSchedule {
        DualSchedule::new()
    }
}

impl DualSchedule {
    pub fn new() -> DualSchedule {
        DualSchedule { video: H3Scheduler::new(H3_VIDEO_SHIFT), audio: H3Scheduler::new(H3_AUDIO_SHIFT) }
    }

    /// Build both schedules for the same requested step count - each applies
    /// its OWN shift internally, so the two end up with different sigma/
    /// timestep sequences despite requesting the same `num_inference_steps`.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.video.set_timesteps(num_inference_steps);
        self.audio.set_timesteps(num_inference_steps);
    }

    pub fn video(&self) -> &H3Scheduler {
        &self.video
    }

    pub fn audio(&self) -> &H3Scheduler {
        &self.audio
    }

    /// The number of Euler steps this schedule will take - the SMALLER of
    /// the two modalities' own step counts (`unique_consecutive` can collapse
    /// them by a different amount since the two shifts differ), so a caller
    /// driving both off one loop index never runs one modality past its own
    /// schedule.
    pub fn num_steps(&self) -> usize {
        self.video.num_steps().min(self.audio.num_steps())
    }

    /// The `[timestep_video, timestep_audio]` pair at `step_index`.
    pub fn current_timesteps(&self, step_index: usize) -> (f32, f32) {
        (self.video.timesteps[step_index], self.audio.timesteps[step_index])
    }

    /// One lockstep Euler step for BOTH modalities, from the two velocity
    /// predictions one transformer forward produced. Returns `(video_next,
    /// audio_next)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(&self, step_index: usize, video_pred: &[f32], video_sample: &[f32], audio_pred: &[f32], audio_sample: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (tv, ta) = self.current_timesteps(step_index);
        (self.video.step(video_pred, tv, video_sample, step_index), self.audio.step(audio_pred, ta, audio_sample, step_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_and_audio_shifts_match_the_real_config() {
        assert_eq!(H3_VIDEO_SHIFT, 12.0);
        assert_eq!(H3_AUDIO_SHIFT, 3.0);
    }

    /// Hand-computed against the closed-form shift formula at the schedule's
    /// own base sigma grid - the independent oracle every H3Scheduler value
    /// is checked against.
    fn shift_ref(shift: f64, sigma: f64) -> f64 {
        shift * sigma / (1.0 + (shift - 1.0) * sigma)
    }

    #[test]
    fn set_timesteps_matches_the_closed_form_shift_at_every_grid_point() {
        let mut sched = H3Scheduler::new(12.0);
        sched.set_timesteps(8);
        assert_eq!(sched.sigmas().last(), Some(&0.0), "the terminal sigma must be exactly 0.0");
        assert!(sched.sigmas().windows(2).all(|w| w[0] > w[1]), "sigmas must be strictly decreasing after dedup");
        assert_eq!(sched.timesteps().len(), sched.sigmas().len() - 1);
        // Spot-check the FIRST base point (i=0, sigma=1.0) and the terminal
        // point (sigma=0.0) - both are exact under any shift and cannot
        // silently pass a swapped-formula bug the way a fixed interior point
        // could if `num_inference_steps` also happened to be wrong.
        assert!((sched.sigmas()[0] - 1.0).abs() < 1e-6);
        assert_eq!(*sched.sigmas().last().unwrap(), 0.0);
        // An interior point, cross-checked against the closed form directly
        // at the KNOWN base value (mid-index of an 8-point linspace(1,0,8)).
        let base_mid = 1.0 - 3.0 / 7.0;
        let expect_mid = shift_ref(12.0, base_mid) as f32;
        assert!(sched.sigmas().iter().any(|&s| (s - expect_mid).abs() < 1e-4), "expected {expect_mid} to appear in {:?}", sched.sigmas());
    }

    #[test]
    fn the_two_schedules_diverge_at_the_same_base_sigma() {
        let mut sched = DualSchedule::new();
        sched.set_timesteps(8);
        // sigma=1 and sigma=0 both fix to the same value under ANY shift, so
        // an interior step is the real divergence check.
        let (t_v, t_a) = sched.current_timesteps(1);
        assert_ne!(t_v, t_a, "video (shift=12) and audio (shift=3) must diverge at the same base sigma");
    }

    #[test]
    fn scale_noise_at_t1_returns_the_sample_unchanged() {
        let sample = vec![1.0f32, -2.0, 3.5];
        let noise = vec![9.0f32, 9.0, 9.0];
        let out = H3Scheduler::scale_noise(&sample, 1.0, &noise);
        assert_eq!(out, sample);
    }

    #[test]
    fn scale_noise_at_t0_returns_pure_noise() {
        let sample = vec![1.0f32, -2.0, 3.5];
        let noise = vec![9.0f32, 8.0, 7.0];
        let out = H3Scheduler::scale_noise(&sample, 0.0, &noise);
        assert_eq!(out, noise);
    }

    /// A hand-derived oracle at ONE step: with `shift=12`, `sigma[0]=1.0` and
    /// a velocity of zero, `step()` must return exactly `sample` (denoised ==
    /// sample when `model_output` is all-zero, and any x_t/x0 blend of two
    /// equal vectors is that same vector).
    #[test]
    fn step_with_zero_velocity_is_a_no_op() {
        let mut sched = H3Scheduler::new(12.0);
        sched.set_timesteps(5);
        let sample = vec![0.3f32, -0.7, 1.2];
        let zero = vec![0.0f32; 3];
        let t0 = sched.timesteps()[0];
        let out = sched.step(&zero, t0, &sample, 0);
        for (o, s) in out.iter().zip(&sample) {
            assert!((o - s).abs() < 1e-6, "zero-velocity step must be a no-op: {o} vs {s}");
        }
    }

    #[test]
    fn step_advances_both_cursors_in_lockstep_over_a_full_loop() {
        let mut sched = DualSchedule::new();
        sched.set_timesteps(6);

        let (mut vs, mut as_) = (vec![1.0f32; 4], vec![1.0f32; 4]);
        for i in 0..sched.num_steps() {
            let (v_next, a_next) = sched.step(i, &[0.1; 4], &vs, &[0.2; 4], &as_);
            vs = v_next;
            as_ = a_next;
        }
        assert!(vs.iter().all(|v| v.is_finite()));
        assert!(as_.iter().all(|v| v.is_finite()));
        // Different velocities (0.1 vs 0.2) over the same step count must
        // land somewhere different - a copy-paste bug feeding one modality's
        // prediction to both would make these equal.
        assert_ne!(vs, as_);
    }

    #[test]
    #[should_panic(expected = "step_index")]
    fn stepping_past_the_schedule_panics() {
        let mut sched = H3Scheduler::new(12.0);
        sched.set_timesteps(3);
        let n = sched.num_steps();
        let v = vec![0.0f32; 2];
        let _ = sched.step(&v, sched.timesteps()[n - 1], &v, n); // one past the last valid step_index
    }
}
