// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From noisy per-frame eye measurements to the eye position a frame should
//! be rendered for.
//!
//! Three jobs, each with a failure mode a viewer notices immediately:
//!
//! * **Smoothing without lag.** A One Euro filter (Casiez, Roussel and Vogel,
//!   CHI 2012): the cutoff rises with speed, so a still head does not shimmer
//!   and a moving one is not dragged behind. A fixed moving average cannot do
//!   both: an N-tap one trails by (N - 1) / 2 samples at every speed.
//! * **Prediction to presentation time.** A measurement describes the moment
//!   the camera exposed it; the frame is seen later, and the filter's output
//!   trails its input by its own group delay on top. The filtered velocity
//!   extrapolates across both, capped at [`EstimatorConfig::max_lead_s`] so a
//!   head that stops does not overshoot by more than a few millimetres, and
//!   gated by speed ([`EstimatorConfig::prediction_onset_mps`]) so a still
//!   head's velocity NOISE is not extrapolated into shimmer.
//! * **Losing and finding the viewer without a jump.** Tracking state is
//!   derived from the age of the last good measurement: fresh is
//!   [`TrackingStatus::Tracking`], briefly stale is `Coasting` (held, no
//!   extrapolation past the cap), long stale is `Lost` (eases toward a rest
//!   position). A measurement too far from the prediction to be the same
//!   motion is held back until the next one confirms it ([`Rejected::Unconfirmed`]),
//!   so a one-frame misdetection never reaches the screen. EVERY accepted
//!   discontinuity - reacquiring after loss, or a confirmed relocation - is
//!   crossfaded from what was on screen over [`EstimatorConfig::blend_s`].
//!   There is no code path that moves the rendered eye by more than the blend
//!   allows in one frame.
//!
//! The estimator is a pure value: [`ViewerEstimator::predict`] takes `&self`
//! and a time, so a render thread may query it at its own rate while a
//! tracking thread feeds [`ViewerEstimator::observe`] at the camera's.

use sfm::linalg::{add, norm, scale, sub};

use crate::V3;

/// Tuning, in physical units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimatorConfig {
    /// One Euro cutoff at rest, Hz. Lower is steadier and laggier.
    pub min_cutoff_hz: f64,
    /// Cutoff increase per m/s of eye speed, Hz/(m/s).
    pub beta_hz_per_mps: f64,
    /// Cutoff of the velocity estimate, Hz.
    pub derivative_cutoff_hz: f64,
    /// Longest extrapolation from the last measurement, seconds.
    pub max_lead_s: f64,
    /// Eye speed at which prediction reaches half strength, m/s. Below it
    /// the velocity estimate is mostly noise and is barely extrapolated.
    pub prediction_onset_mps: f64,
    /// Measurements older than this make the state `Coasting`, seconds.
    pub stale_s: f64,
    /// Measurements older than this make the state `Lost`, seconds.
    pub lost_s: f64,
    /// Where the eye rests when nobody is tracked, screen frame.
    pub rest_eye: V3,
    /// Time constant of the ease toward [`EstimatorConfig::rest_eye`], seconds.
    pub return_tau_s: f64,
    /// Length of every crossfade, seconds.
    pub blend_s: f64,
    /// A measurement further than this from the prediction is a
    /// discontinuity (a different face, a recovered track), metres.
    pub jump_m: f64,
    /// Measurements below this confidence are dropped.
    pub min_confidence: f64,
}

impl Default for EstimatorConfig {
    fn default() -> EstimatorConfig {
        EstimatorConfig {
            // Chosen by sweeping static jitter, ramp error and stop overshoot
            // together, in the scenario the estimator tests replay
            // perf-number: that scenario's parameters, not a measurement
            // (30 Hz camera, 30 ms capture-to-photon): at these
            // values 3 mm of measurement noise shows as about 1.4 mm, a
            // 0.3 m/s head is followed within 2 mm, and an instant stop from
            // it overshoots about 7 mm - under the 9 mm an ideal extrapolator
            // over the same lead would.
            min_cutoff_hz: 1.0,
            beta_hz_per_mps: 20.0,
            derivative_cutoff_hz: 2.0,
            max_lead_s: 0.03,
            prediction_onset_mps: 0.15,
            stale_s: 0.12,
            lost_s: 0.5,
            rest_eye: [0.0, 0.0, 0.6],
            return_tau_s: 0.8,
            blend_s: 0.2,
            jump_m: 0.08,
            min_confidence: 0.5,
        }
    }
}

/// How current the estimate is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingStatus {
    /// No measurement has ever been accepted.
    Searching,
    /// The last measurement is fresh.
    Tracking,
    /// The last measurement is stale but recent; the eye is held.
    Coasting,
    /// Nobody has been measured for a while; the eye eases to rest.
    Lost,
}

/// The eye a frame is rendered for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackedViewer {
    /// Capture time of the last accepted measurement (NaN when none).
    pub sample_time_s: f64,
    /// The subject's right pupil, screen frame, predicted to the query time.
    pub right_eye: V3,
    /// The subject's left pupil, screen frame, predicted to the query time.
    pub left_eye: V3,
    /// Confidence and reprojection error of the last accepted measurement.
    pub confidence: f64,
    pub rmse_px: f64,
    pub status: TrackingStatus,
    /// Query time minus [`TrackedViewer::sample_time_s`].
    pub sample_age_s: f64,
    /// How far the position was extrapolated, seconds.
    pub lead_s: f64,
}

impl TrackedViewer {
    /// The midpoint of the pupils: the viewpoint for a single (non-stereo)
    /// image.
    pub fn cyclopean(&self) -> V3 {
        scale(add(self.left_eye, self.right_eye), 0.5)
    }
}

/// Why a measurement was not used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rejected {
    /// Not after the last accepted measurement.
    OutOfOrder,
    /// Below [`EstimatorConfig::min_confidence`].
    LowConfidence,
    /// A coordinate is not finite.
    NotFinite,
    /// Too far from the prediction to be the same motion; held until the
    /// next measurement agrees with it.
    Unconfirmed,
}

/// A One Euro filter over a 3-vector: one cutoff from the vector's speed, so
/// smoothing is the same in every direction.
#[derive(Clone, Copy, Debug)]
struct OneEuro3 {
    x: V3,
    dx: V3,
    /// Group delay of the position smoothing at the last update: how far
    /// behind a ramp the output sits, seconds.
    delay: f64,
}

fn alpha(cutoff_hz: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
    1.0 / (1.0 + tau / dt)
}

fn lerp(a: V3, b: V3, w: f64) -> V3 {
    add(a, scale(sub(b, a), w))
}

impl OneEuro3 {
    fn at(x: V3) -> OneEuro3 {
        OneEuro3 { x, dx: [0.0; 3], delay: 0.0 }
    }

    fn update(&mut self, x: V3, dt: f64, c: &EstimatorConfig) {
        let raw = scale(sub(x, self.x), 1.0 / dt);
        self.dx = lerp(self.dx, raw, alpha(c.derivative_cutoff_hz, dt));
        let cutoff = c.min_cutoff_hz + c.beta_hz_per_mps * norm(self.dx);
        let a = alpha(cutoff, dt);
        self.x = lerp(self.x, x, a);
        // an exponential smoother's steady-state lag behind a ramp
        self.delay = dt * (1.0 - a) / a;
    }

    /// Extrapolate `age` seconds past the last measurement, plus the
    /// smoother's own delay, scaled down at noise-level speeds.
    fn predict(&self, age: f64, c: &EstimatorConfig) -> (V3, f64) {
        let speed2 = norm(self.dx).powi(2);
        let gain = speed2 / (speed2 + c.prediction_onset_mps.powi(2));
        let lead = (age.max(0.0) + self.delay).min(c.max_lead_s) * gain;
        (add(self.x, scale(self.dx, lead)), lead)
    }
}

#[derive(Clone, Copy, Debug)]
struct Track {
    right: OneEuro3,
    left: OneEuro3,
    time: f64,
    confidence: f64,
    rmse_px: f64,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    time: f64,
    mid: V3,
}

#[derive(Clone, Copy, Debug)]
struct Blend {
    right: V3,
    left: V3,
    start: f64,
}

/// The estimator. See the module docs for its states.
#[derive(Clone, Debug)]
pub struct ViewerEstimator {
    config: EstimatorConfig,
    track: Option<Track>,
    blend: Option<Blend>,
    pending: Option<Pending>,
}

/// Half the canonical interpupillary distance, for the rest pose's eyes.
const HALF_IPD_M: f64 = crate::face::CANONICAL_IPD_M / 2.0;

impl ViewerEstimator {
    pub fn new(config: EstimatorConfig) -> ViewerEstimator {
        ViewerEstimator { config, track: None, blend: None, pending: None }
    }

    pub fn config(&self) -> &EstimatorConfig {
        &self.config
    }

    /// Feed one measurement taken at capture time `t`.
    pub fn observe(&mut self, t: f64, right_eye: V3, left_eye: V3, confidence: f64, rmse_px: f64) -> Result<(), Rejected> {
        if !(t.is_finite() && right_eye.iter().chain(&left_eye).all(|v| v.is_finite())) {
            return Err(Rejected::NotFinite);
        }
        if confidence.is_nan() || confidence < self.config.min_confidence {
            return Err(Rejected::LowConfidence);
        }
        let mid = scale(add(right_eye, left_eye), 0.5);
        if let Some(tr) = self.track.as_mut() {
            if t <= tr.time {
                return Err(Rejected::OutOfOrder);
            }
            let age = t - tr.time;
            if age <= self.config.lost_s {
                // compare against the FILTER's own prediction, never the
                // blended picture: mid-crossfade the screen is deliberately
                // somewhere the viewer is not
                let (pr, _) = tr.right.predict(age, &self.config);
                let (pl, _) = tr.left.predict(age, &self.config);
                let expected = scale(add(pr, pl), 0.5);
                if norm(sub(mid, expected)) <= self.config.jump_m {
                    tr.right.update(right_eye, age, &self.config);
                    tr.left.update(left_eye, age, &self.config);
                    tr.time = t;
                    tr.confidence = confidence;
                    tr.rmse_px = rmse_px;
                    self.pending = None;
                    return Ok(());
                }
                // a discontinuity while tracking: only a second agreeing
                // measurement makes it real
                let confirmed = self.pending.is_some_and(|p| t > p.time && norm(sub(mid, p.mid)) <= self.config.jump_m);
                if !confirmed {
                    self.pending = Some(Pending { time: t, mid });
                    return Err(Rejected::Unconfirmed);
                }
            }
        }
        // a (re)acquisition: start the filters on the measurement, and
        // crossfade from whatever is on screen
        let shown = self.predict(t);
        self.blend = Some(Blend { right: shown.right_eye, left: shown.left_eye, start: t });
        self.track = Some(Track { right: OneEuro3::at(right_eye), left: OneEuro3::at(left_eye), time: t, confidence, rmse_px });
        self.pending = None;
        Ok(())
    }

    /// The eye to render a frame presented at time `t` for.
    pub fn predict(&self, t: f64) -> TrackedViewer {
        let c = &self.config;
        let rest_r = add(c.rest_eye, [-HALF_IPD_M, 0.0, 0.0]);
        let rest_l = add(c.rest_eye, [HALF_IPD_M, 0.0, 0.0]);
        let Some(tr) = self.track else {
            return TrackedViewer {
                sample_time_s: f64::NAN,
                right_eye: rest_r,
                left_eye: rest_l,
                confidence: 0.0,
                rmse_px: f64::NAN,
                status: TrackingStatus::Searching,
                sample_age_s: f64::INFINITY,
                lead_s: 0.0,
            };
        };
        let age = t - tr.time;
        let (mut right, lead) = tr.right.predict(age, c);
        let (mut left, _) = tr.left.predict(age, c);
        let status = if age > c.lost_s {
            // ease from the held position toward rest
            let k = (-(age - c.lost_s) / c.return_tau_s).exp();
            right = lerp(rest_r, right, k);
            left = lerp(rest_l, left, k);
            TrackingStatus::Lost
        } else if age > c.stale_s {
            TrackingStatus::Coasting
        } else {
            TrackingStatus::Tracking
        };
        if let Some(b) = self.blend {
            let s = ((t - b.start) / c.blend_s).clamp(0.0, 1.0);
            let w = s * s * (3.0 - 2.0 * s);
            right = lerp(b.right, right, w);
            left = lerp(b.left, left, w);
        }
        TrackedViewer { sample_time_s: tr.time, right_eye: right, left_eye: left, confidence: tr.confidence, rmse_px: tr.rmse_px, status, sample_age_s: age, lead_s: lead }
    }
}
