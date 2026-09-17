// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Is this population oscillating, and how fast?
//!
//! The question matters because of a specific claim about the nerve cord: that
//! the rhythm a fly walks to is already in the wiring. Drive one descending
//! cell type tonically - no oscillating input, no pattern generator written by
//! hand, just a constant command - and the recurrent circuitry downstream is
//! reported to produce rhythmic motor-neuron activity on its own, at a
//! frequency that rises with the strength of the drive.
//!
//! That is a prediction rather than an objective, which is what makes it worth
//! more than anything a search here has produced. Nothing is optimised against
//! it. It either happens in the measured connectome or it does not, and the
//! controls are cheap: a constant input cannot itself contain a rhythm, so if
//! one appears at the motor neurons it was made by the graph.
//!
//! Autocorrelation rather than a Fourier transform, for a reason that is about
//! the signal and not about convenience. A population spike count over a few
//! hundred ticks is short, non-sinusoidal and burst-like; a periodogram of it
//! is dominated by the DC term and by the window, and reading a peak off one
//! invites finding structure in noise. The normalised autocorrelation at the
//! best lag answers the narrower question actually being asked - does this
//! signal resemble itself one period later - and it is bounded by 1, so the
//! answer is comparable across conditions with different firing rates.
//!
//! Swedish Embedded AB implements signal analysis for neural and sensor data
//! for its clients, including the discipline of choosing an estimator that
//! matches what the data can support. If your team needs this, you can procure
//! our services by sending an email to info@swedishembedded.com.

/// What a population rate signal looks like in time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rhythm {
    /// Dominant frequency in hertz, `0.0` when nothing periodic was found.
    pub hz: f64,
    /// Normalised autocorrelation at that lag, in `[-1, 1]`. How strongly the
    /// signal resembles itself one period later.
    pub strength: f64,
    /// Mean of the signal, in spikes per tick. A rhythm in a population that
    /// barely fires is a rhythm in nothing.
    pub rate: f64,
    /// Standard deviation, in the same units. A perfectly flat signal has no
    /// period and this is how that is told apart from a strong one.
    pub deviation: f64,
    /// Whether [`Self::hz`] fell inside the band the caller asked about.
    ///
    /// Separate from the frequency because the interesting failure is a signal
    /// whose real period is far outside the band: a population firing on
    /// alternate ticks is a 250 Hz square wave, it correlates perfectly with
    /// itself at every even lag, and an estimator that only ever looks inside
    /// 3 to 30 Hz finds one of those lags and reports a beautiful walking
    /// rhythm. This module did exactly that, and called refractory ringing an
    /// 11 Hz gait, until the control that deleted the inhibitory neuron came
    /// back MORE rhythmic rather than less.
    pub in_band: bool,
}

impl Rhythm {
    /// Whether this is a rhythm worth reporting: periodic, in band, and in
    /// something that is actually active.
    pub fn is_rhythmic(&self, min_strength: f64) -> bool {
        self.in_band && self.strength >= min_strength && self.deviation > 0.0 && self.rate > 0.0
    }
}

/// Analyse a population rate for periodicity within a frequency band.
///
/// `dt` is the tick length in seconds. `band` is `(low_hz, high_hz)` and is
/// required rather than optional: an unconstrained search over lags will
/// always find its best match at the longest lag it is allowed, where the
/// overlap is shortest and the estimate is noisiest. A walking fly steps at
/// something like 5 to 20 hertz, and asking the question inside that band is
/// the difference between a measurement and a fishing expedition.
pub fn analyse(rate: &[f64], dt: f64, band: (f64, f64)) -> Rhythm {
    let n = rate.len();
    if n < 8 || dt <= 0.0 {
        return Rhythm::default();
    }
    let mean = rate.iter().sum::<f64>() / n as f64;

    // Remove the linear trend, not just the mean. A signal that merely ramps
    // resembles itself at every lag - its first half predicts its second - so
    // an estimator that subtracts only the mean reports a pure ramp as 0.99
    // periodic, which this module's own test caught. A cord whose activity is
    // climbing towards saturation is the most likely thing to be mistaken for
    // a walking rhythm, so this is the confound that had to go.
    let (mut sx, mut sxx, mut sy, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (t, y) in rate.iter().enumerate() {
        let t = t as f64;
        sx += t;
        sxx += t * t;
        sy += y;
        sxy += t * y;
    }
    let nf = n as f64;
    let denom = nf * sxx - sx * sx;
    let slope = if denom.abs() > 0.0 { (nf * sxy - sx * sy) / denom } else { 0.0 };
    let intercept = (sy - slope * sx) / nf;
    let dev: Vec<f64> = rate.iter().enumerate().map(|(t, y)| y - (intercept + slope * t as f64)).collect();

    let power: f64 = dev.iter().map(|x| x * x).sum();
    let deviation = (power / nf).sqrt();
    // A residual at the level of floating-point rounding is not a signal. The
    // scale is taken from the data rather than fixed, so this means "flat
    // compared with itself" rather than "small".
    let scale: f64 = rate.iter().map(|x| x * x).sum::<f64>() / nf;
    if power <= 0.0 || deviation * deviation <= 1e-18 * (1.0 + scale) {
        return Rhythm { hz: 0.0, strength: 0.0, rate: mean, deviation, in_band: false };
    }

    // The search starts at the shortest measurable lag, NOT at the top of the
    // caller's band. Restricting the search to the band does not exclude
    // out-of-band signals, it disguises them: a period of two ticks repeats at
    // four, six, eight and every other multiple, so a band that admits any one
    // of those multiples reports a slow rhythm that is not there. Find the
    // fundamental wherever it is, then say whether it landed in the band.
    let lo = 2usize;
    // Half the record, so every lag is estimated from at least half the
    // samples. Beyond that the autocorrelation is mostly an artefact of how
    // few pairs are left to average.
    let hi = ((1.0 / (band.0 * dt)).ceil() as usize).max(lo + 1).min(n / 2);
    if lo >= hi {
        return Rhythm { hz: 0.0, strength: 0.0, rate: mean, deviation, in_band: false };
    }

    let corr: Vec<f64> = (lo..=hi)
        .map(|lag| {
            let c: f64 = (0..n - lag).map(|t| dev[t] * dev[t + lag]).sum();
            // Normalised over the overlapping window only, so a long lag is
            // not penalised merely for overlapping less.
            let a: f64 = (0..n - lag).map(|t| dev[t] * dev[t]).sum();
            let b: f64 = (0..n - lag).map(|t| dev[t + lag] * dev[t + lag]).sum();
            let norm = (a * b).sqrt();
            if norm > 0.0 {
                c / norm
            } else {
                0.0
            }
        })
        .collect();

    // The period is the first peak AFTER the correlation has fallen away.
    //
    // Not simply the largest peak: a periodic signal correlates with itself at
    // every multiple of its period, and which multiple wins is decided by
    // noise, so taking the maximum reports an 18 Hz oscillation as 4.5 Hz
    // whenever the fourth peak wins by a hair.
    //
    // Not simply the first peak either, which is the mistake this made next.
    // Autocorrelation is near 1 at the shortest lags for any smooth signal,
    // so "first lag within 90% of the best" always fires at lag 2 and reports
    // every rhythm as 250 Hz. The correlation has to be allowed to descend
    // first; the fundamental is the first peak after it does.
    let best = corr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if best <= 0.0 {
        return Rhythm { hz: 0.0, strength: best.max(0.0), rate: mean, deviation, in_band: false };
    }
    let dipped = corr.iter().position(|&r| r <= 0.0);
    let (mut lag, mut strength) = match dipped {
        Some(d) => {
            // The first local maximum after the dip, which for a signal with
            // one period is that period.
            let mut found = None;
            for i in d + 1..corr.len() {
                let rising = corr[i] >= corr[i - 1];
                let falling = i + 1 == corr.len() || corr[i] >= corr[i + 1];
                if rising && falling && corr[i] > 0.0 {
                    found = Some((lo + i, corr[i]));
                    break;
                }
            }
            // A signal that dips and never comes back up has no second
            // period within the record, so there is nothing to report.
            found.unwrap_or((0, 0.0))
        }
        // Never negative anywhere: a slow drift the detrender did not remove,
        // or a record shorter than one period. Either way the best available
        // answer is the largest peak, and it will usually fall out of band.
        None => {
            let i = corr.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
            (lo + i, corr[i])
        }
    };
    if lag == 0 {
        return Rhythm { hz: 0.0, strength: 0.0, rate: mean, deviation, in_band: false };
    }
    // Prefer an earlier peak of equal quality: harmonics again, now on the
    // other side. A period of 2P is also a period, and the smaller one is the
    // fundamental.
    for (i, &r) in corr.iter().enumerate() {
        let at = lo + i;
        if at >= lag {
            break;
        }
        let rising = i == 0 || r >= corr[i - 1];
        let falling = i + 1 == corr.len() || r >= corr[i + 1];
        if rising && falling && r >= 0.98 * strength && dipped.is_some_and(|d| i > d) {
            lag = at;
            strength = r;
            break;
        }
    }
    let hz = 1.0 / (lag as f64 * dt);
    Rhythm { hz, strength, rate: mean, deviation, in_band: hz >= band.0 && hz <= band.1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.002;
    const BAND: (f64, f64) = (4.0, 30.0);

    #[test]
    fn a_known_oscillation_is_found_at_its_own_frequency() {
        for want in [6.0, 10.0, 18.0] {
            let n = 600;
            let x: Vec<f64> = (0..n)
                .map(|t| 5.0 + 3.0 * (2.0 * std::f64::consts::PI * want * t as f64 * DT).sin())
                .collect();
            let r = analyse(&x, DT, BAND);
            assert!(r.strength > 0.9, "{want} Hz: a clean sinusoid should be strongly periodic, got {r:?}");
            // The lag is an integer number of ticks, so the recoverable
            // resolution is coarser at high frequency: at 18 Hz one tick of
            // lag is already 0.7 Hz.
            assert!((r.hz - want).abs() / want < 0.06, "{want} Hz: recovered {:.2} Hz", r.hz);
            assert!((r.rate - 5.0).abs() < 0.05, "the mean should survive the analysis, got {}", r.rate);
            assert!(r.in_band, "{want} Hz is inside {BAND:?}");
        }
    }

    #[test]
    fn a_constant_signal_has_no_rhythm_and_neither_does_noise() {
        let flat = vec![3.0; 400];
        let r = analyse(&flat, DT, BAND);
        assert_eq!(r.strength, 0.0, "a flat signal cannot be periodic");
        assert!(!r.is_rhythmic(0.3));

        let mut rng = crate::learn::Lcg::new(0x1234);
        let noise: Vec<f64> = (0..600).map(|_| rng.normal() as f64).collect();
        let r = analyse(&noise, DT, BAND);
        assert!(r.strength < 0.3, "white noise should not look periodic, got {r:?}");
    }

    /// The control that matters for the claim being tested: a signal that
    /// simply ramps has no period, but a naive estimator reports one because
    /// the two halves of a trend correlate.
    #[test]
    fn a_trend_is_not_a_rhythm() {
        let ramp: Vec<f64> = (0..600).map(|t| t as f64 * 0.01).collect();
        let r = analyse(&ramp, DT, BAND);
        assert_eq!(r.strength, 0.0, "a pure ramp is not an oscillation");
        assert!(!r.is_rhythmic(0.3));

        // And a rhythm riding ON a ramp is still a rhythm: detrending must
        // remove the confound without removing the signal.
        let both: Vec<f64> = (0..600)
            .map(|t| t as f64 * 0.01 + 2.0 * (2.0 * std::f64::consts::PI * 10.0 * t as f64 * DT).sin())
            .collect();
        let r = analyse(&both, DT, BAND);
        assert!(r.strength > 0.9, "a 10 Hz oscillation on a ramp is still periodic, got {r:?}");
        assert!((r.hz - 10.0).abs() < 0.5, "recovered {:.2} Hz", r.hz);
    }

    /// The failure that made `in_band` exist.
    ///
    /// A population firing on alternate ticks is a 250 Hz square wave. It
    /// correlates perfectly with itself at every even lag, so an estimator
    /// that only searches 3 to 30 Hz finds a lag in there, reports strength
    /// 1.0, and calls refractory ringing a walking rhythm.
    #[test]
    fn ringing_faster_than_the_band_is_not_reported_as_a_slow_rhythm() {
        let x: Vec<f64> = (0..600).map(|t| if t % 2 == 0 { 6.0 } else { 0.0 }).collect();
        let r = analyse(&x, DT, BAND);
        assert!(r.strength > 0.9, "it IS periodic, just not in band: {r:?}");
        // Reported as 125 Hz rather than 250: the correlation of a period-two
        // square wave alternates between +1 and -1, so the first peak after
        // the dip is at lag 4, not lag 2. A factor of two on a signal that is
        // eight times above the band does not change the verdict, which is
        // all this has to get right.
        assert!(r.hz > 100.0, "the fundamental is far above the band, got {:.1} Hz", r.hz);
        assert!(!r.in_band, "250 Hz is not a walking rhythm");
        assert!(!r.is_rhythmic(0.3), "an out-of-band oscillation must not pass the gate");
    }

    #[test]
    fn a_burst_train_is_found_even_though_it_is_not_a_sinusoid() {
        // What a motor pool actually does: silent, then a burst, repeating.
        let period = 50; // 10 Hz at 2 ms
        let x: Vec<f64> = (0..600).map(|t| if t % period < 12 { 9.0 } else { 0.0 }).collect();
        let r = analyse(&x, DT, BAND);
        assert!(r.strength > 0.7, "a burst train is periodic, got {r:?}");
        assert!((r.hz - 10.0).abs() < 0.5, "recovered {:.2} Hz", r.hz);
        assert!(r.is_rhythmic(0.3));
    }
}
