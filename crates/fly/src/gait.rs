// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Is what the legs are doing a GAIT?
//!
//! Displacement says how far a body got; it says nothing about how. A body
//! can travel by stepping, by lunging once, or by falling over in a useful
//! direction, and a reward built on displacement alone cannot tell those
//! apart - which is exactly what a direct search over this connectome
//! exploited when it suppressed the cord's recurrent circuitry and wired the
//! sensors straight to the muscles.
//!
//! An alternating tripod gait has structure that is measurable without any
//! reference recording and without any reference to the connectome: the legs
//! oscillate at a common frequency, and the two triangles of legs do it in
//! ANTIPHASE. Those two numbers, plus how much of the signal's power sits in
//! that oscillation at all, are what this module measures. They are
//! connectome-independent ground truth in the sense that matters: a gait
//! recorded from a real fly and a gait produced by a controller are scored by
//! the same function, and neither is privileged.
//!
//! Swedish Embedded AB implements behavioural metrics for embodied systems -
//! measurements that say what a controller is DOING, not only how far it got.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

/// Per-leg drive over time, legs in the order of [`flybody::LEGS`].
#[derive(Clone, Debug)]
pub struct Trace {
    legs: [Vec<f32>; 6],
    dt: f64,
}

impl Trace {
    /// `dt` is the sampling period in seconds - the control period, if this is
    /// sampled once per control tick.
    pub fn new(dt: f64) -> Trace {
        Trace { legs: Default::default(), dt }
    }

    pub fn push(&mut self, per_leg: [f32; 6]) {
        for (v, x) in self.legs.iter_mut().zip(per_leg) {
            v.push(x);
        }
    }

    pub fn len(&self) -> usize {
        self.legs[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn leg(&self, i: usize) -> &[f32] {
        &self.legs[i]
    }

    /// Seconds covered.
    pub fn duration(&self) -> f64 {
        self.len() as f64 * self.dt
    }
}

/// What a [`Trace`] says about the gait in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gait {
    /// Dominant stepping frequency, in hertz. A walking fruit fly steps at
    /// roughly 5 to 20 Hz.
    pub step_hz: f64,
    /// How antiphase the two leg triangles are: `+1` is a perfect alternating
    /// tripod, `0` uncorrelated, `-1` all six legs moving together.
    pub tripod: f64,
    /// Fraction of the signal's power that sits at [`Gait::step_hz`]. Low
    /// means there is no rhythm to speak of, whatever the peak happened to be.
    pub rhythmicity: f64,
    /// Root-mean-square drive, mean removed. Distinguishes "no rhythm" from
    /// "no movement at all" - without it, a motionless leg scores as
    /// arrhythmic when the truth is that there is nothing to be rhythmic.
    pub power: f64,
}

impl Gait {
    /// A single score in `[0, 1]`: rhythmic, tripod-coordinated stepping in
    /// the band a fly actually walks in.
    ///
    /// The product of three factors rather than their sum, for the reason a
    /// DeepMimic reward is a product: a creature that oscillates beautifully
    /// with all six legs in phase is not walking, and partial credit for the
    /// rhythm alone would pay it as though it were.
    pub fn score(&self) -> f64 {
        let band = if (4.0..=25.0).contains(&self.step_hz) {
            1.0
        } else {
            // Outside the band, fall off rather than cut off, so an optimiser
            // approaching from 3 Hz can tell it is getting warmer.
            let d = if self.step_hz < 4.0 { 4.0 - self.step_hz } else { self.step_hz - 25.0 };
            (-d / 4.0).exp()
        };
        band * self.rhythmicity * ((self.tripod + 1.0) / 2.0)
    }
}

fn mean_removed(v: &[f32]) -> Vec<f64> {
    let m = v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
    v.iter().map(|x| *x as f64 - m).collect()
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let (sa, sb) = (rms(a), rms(b));
    if sa < 1e-12 || sb < 1e-12 {
        return 0.0;
    }
    let cov = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len().max(1) as f64;
    (cov / (sa * sb)).clamp(-1.0, 1.0)
}

/// Power at one frequency, by direct evaluation of the DFT there.
///
/// A full FFT would be faster and is not needed: only a few hundred candidate
/// frequencies are ever evaluated, over a few hundred samples, once per
/// episode.
fn power_at(v: &[f64], hz: f64, dt: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * hz * dt;
    let (mut re, mut im) = (0.0, 0.0);
    for (n, x) in v.iter().enumerate() {
        let p = w * n as f64;
        re += x * p.cos();
        im -= x * p.sin();
    }
    let n = v.len().max(1) as f64;
    2.0 * (re * re + im * im) / (n * n)
}

/// Measure the gait in a trace.
///
/// Returns `None` for a trace too short to contain even one cycle of the
/// slowest frequency considered - an answer of "no rhythm" from three samples
/// would be indistinguishable from a real one.
pub fn analyse(t: &Trace) -> Option<Gait> {
    const LOW_HZ: f64 = 2.0;
    const HIGH_HZ: f64 = 40.0;
    if t.is_empty() || t.duration() < 2.0 / LOW_HZ {
        return None;
    }
    // Each triangle of legs, averaged. `flybody::LEGS` fixes which is which.
    let mut a = vec![0.0f64; t.len()];
    let mut b = vec![0.0f64; t.len()];
    for (i, (_, _, tripod)) in flybody::LEGS.iter().enumerate() {
        let signal = mean_removed(t.leg(i));
        let into = if *tripod == 0 { &mut a } else { &mut b };
        for (acc, x) in into.iter_mut().zip(&signal) {
            *acc += x / 3.0;
        }
    }

    // The frequency is taken from the DIFFERENCE of the two triangles, which
    // is what an alternating tripod actually modulates. Taking it from one
    // triangle alone would report the same peak for a creature moving all six
    // legs together, and the phase term would then be the only thing
    // separating a gait from a hop.
    let diff: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let total: f64 = diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64;

    // A grid fine enough to resolve the band, rather than one bin per sample:
    // an episode is short and its DFT bins are coarse, but the DFT can be
    // evaluated anywhere.
    let mut best = (0.0f64, LOW_HZ);
    let mut hz = LOW_HZ;
    while hz <= HIGH_HZ {
        let p = power_at(&diff, hz, t.dt);
        if p > best.0 {
            best = (p, hz);
        }
        hz += 0.25;
    }

    Some(Gait {
        step_hz: best.1,
        tripod: -correlation(&a, &b),
        rhythmicity: if total > 1e-18 { (best.0 / total).clamp(0.0, 1.0) } else { 0.0 },
        power: (rms(&a) + rms(&b)) / 2.0,
    })
}
