// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The codebook-0 logit filter chain, as one reusable unit.
//!
//! Every autoregressive decode loop in this crate - the device-agnostic
//! [`crate::pipeline::generate_codes`], the CPU-cached mirror, the interleaved
//! batch scheduler and the three NPU loops - draws its next codebook-0 token
//! through exactly the same six stages:
//!
//! 1. **suppress-tokens mask**: the reference blanks the top-1024 vocab entries
//!    (`[vocab-1024, vocab)`), including the codec EOS unless `min_new_tokens`
//!    has been satisfied;
//! 2. **repetition penalty**, once per DISTINCT token in the history;
//! 3. **temperature** scaling (or an argmax short-circuit when greedy);
//! 4. **top-k** truncation;
//! 5. **top-p** (nucleus) truncation;
//! 6. a softmax + inverse-CDF **categorical draw**.
//!
//! Those stages used to live as private helpers inside one decode loop, which
//! is how a defect in stage 2 (penalty applied once per OCCURRENCE, compounding
//! to `penalty^count`) survived: there was nowhere to test the chain that was
//! not "run the whole model". They are public here so the chain can be driven
//! from synthetic logits in the unit lane, and so the MTP/subtalker sampler can
//! reuse it rather than growing a second, drifting copy.
//!
//! The configuration this consumes is [`SamplerCfg`] - a FULLY resolved set of
//! knobs, never an `Option`. Deciding what those values should be (caller
//! override > the checkpoint's `generation_config.json` > the reference's hard
//! defaults) is [`crate::genconfig`]'s job, not this module's.
//!
//! Swedish Embedded AB implements solutions for correct, testable token-sampling
//! pipelines in from-scratch inference engines for its clients. If your team
//! needs expertise in autoregressive decoding and logit processors then you can
//! procure our services by sending an email to info@swedishembedded.com.

use data::rng::Rng;

/// Width of the reference's `suppress_tokens` window: the top 1024 vocab
/// entries are special/reserved ids, never legal acoustic codes.
pub const SUPPRESS_WINDOW: usize = 1024;

/// A fully resolved filter chain for ONE codebook stream.
///
/// Codebook-0 and the MTP's residual codebooks get their own instance (the
/// reference configures them separately, as `temperature`/`top_k`/`top_p` and
/// `subtalker_temperature`/`subtalker_top_k`/`subtalker_top_p`), which is why
/// this is a standalone value rather than fields on the generation options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerCfg {
    /// The reference's `do_sample`. `false` decodes greedily (argmax) and
    /// ignores every other knob here except the suppress mask and the
    /// repetition penalty.
    pub do_sample: bool,
    pub temperature: f32,
    /// `0` disables top-k.
    pub top_k: usize,
    /// Nucleus cutoff, applied after `top_k`. `<= 0.0` or `>= 1.0` disables it
    /// (the reference ships `1.0`, i.e. off).
    pub top_p: f32,
    /// HF's repetition penalty. `1.0` disables it; the reference ships `1.05`
    /// and this model genuinely needs it - see [`apply_repetition_penalty`].
    pub repetition_penalty: f32,
}

impl SamplerCfg {
    /// Whether this draw is deterministic.
    ///
    /// Two independent ways to ask for greedy decoding, both honoured: the
    /// reference's own `do_sample=False`, and a non-positive `temperature`
    /// (this crate's CLI has always used `--temp 0` for the deterministic
    /// parity path, and dividing logits by zero is not a distribution).
    pub fn is_greedy(&self) -> bool {
        !self.do_sample || self.temperature <= 0.0
    }
}

/// One token drawn from the chain, with the evidence needed to tell a healthy
/// draw from a degenerating one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draw {
    pub token: u32,
    /// The drawn token's probability under the **post-filter** distribution -
    /// after masking, penalty, temperature, top-k and top-p, i.e. the actual
    /// distribution the draw came from. A greedy draw reports `1.0`, which is
    /// exactly what an argmax's degenerate distribution assigns it.
    ///
    /// This is the number that made the silent-collapse bug legible: a
    /// codebook-0 repetition loop is a run of frames whose top-1 probability
    /// climbs past ~0.99, and it is free to surface here because the chain has
    /// already normalised the distribution to draw from it.
    pub prob: f32,
}

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

/// The reference's `suppress_tokens` mask, in place: blank the top
/// [`SUPPRESS_WINDOW`] vocab entries, restoring the codec EOS only when
/// `allow_eos` (the `min_new_tokens` guard) permits it.
pub fn suppress_specials(logits: &mut [f32], eos: u32, allow_eos: bool) {
    let lo = logits.len().saturating_sub(SUPPRESS_WINDOW);
    let eos_logit = logits[eos as usize];
    for x in logits[lo..].iter_mut() {
        *x = f32::NEG_INFINITY;
    }
    if allow_eos {
        logits[eos as usize] = eos_logit;
    }
}

/// Apply HF's standard repetition penalty in place: a previously-seen logit is
/// divided by `penalty` if positive, multiplied if negative (so either way it
/// moves toward zero, discouraging but never fully forbidding a repeat).
/// `penalty <= 1.0` is a no-op (`1.0` = disabled).
///
/// **Once per distinct token, not once per occurrence.** HF's
/// `RepetitionPenaltyLogitsProcessor` is a `gather`/`where`/`scatter` over
/// `input_ids`, so a token that appears fifty times in the history is penalized
/// exactly as hard as one that appears once. Applying it per occurrence instead
/// would compound to `penalty^count` - at the reference's 1.05 that is a 131x
/// logit shrink after a hundred repeats, which is a different (and, on a
/// long-form clip, destructive) processor than the one the reference calibrated
/// its value against.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::with_capacity(history.len());
    for &t in history {
        let idx = t as usize;
        if !seen.insert(idx) {
            continue;
        }
        if idx < logits.len() && logits[idx].is_finite() {
            logits[idx] = if logits[idx] > 0.0 { logits[idx] / penalty } else { logits[idx] * penalty };
        }
    }
}

/// Top-k truncation in place on temperature-scaled logits: keep the `top_k`
/// highest entries, mask the rest to `-inf`. `top_k == 0` (or `>= len`) is a
/// no-op.
pub fn apply_top_k(scaled: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= scaled.len() {
        return;
    }
    let mut idx: Vec<usize> = (0..scaled.len()).collect();
    idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
    let threshold = scaled[idx[top_k - 1]];
    for x in scaled.iter_mut() {
        if *x < threshold {
            *x = f32::NEG_INFINITY;
        }
    }
}

/// Nucleus (top-p) filter in place: keep the smallest prefix of `scaled`
/// (already temperature-scaled and, if requested, top-k-masked) whose softmax
/// probability mass is `>= top_p`, mask everything else to `-inf`. Always
/// keeps at least one token. `top_p <= 0.0` or `>= 1.0` is a no-op.
pub fn apply_top_p(scaled: &mut [f32], top_p: f32) {
    if top_p <= 0.0 || top_p >= 1.0 {
        return;
    }
    let max0 = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut ranked: Vec<(usize, f32)> =
        scaled.iter().enumerate().filter(|&(_, &x)| x.is_finite()).map(|(i, &x)| (i, (x - max0).exp())).collect();
    let z: f32 = ranked.iter().map(|&(_, p)| p).sum();
    if z <= 0.0 {
        return;
    }
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cum = 0.0f32;
    let mut cutoff = ranked.len();
    for (rank, &(_, p)) in ranked.iter().enumerate() {
        cum += p / z;
        if cum >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    let keep: std::collections::HashSet<usize> = ranked[..cutoff].iter().map(|&(i, _)| i).collect();
    for (i, x) in scaled.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *x = f32::NEG_INFINITY;
        }
    }
}

/// Run the whole chain and draw one codebook-0 token.
///
/// `history` is the sequence of already-generated codebook-0 ids for this clip,
/// consulted only when `cfg.repetition_penalty > 1.0`.
pub fn sample_cb0(mut logits: Vec<f32>, eos: u32, allow_eos: bool, cfg: &SamplerCfg, history: &[u32], rng: &mut Rng) -> Draw {
    suppress_specials(&mut logits, eos, allow_eos);
    apply_repetition_penalty(&mut logits, history, cfg.repetition_penalty);
    if cfg.is_greedy() {
        return Draw { token: argmax(&logits) as u32, prob: 1.0 };
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / cfg.temperature).collect();
    apply_top_k(&mut scaled, cfg.top_k);
    apply_top_p(&mut scaled, cfg.top_p);
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    // Inverse-CDF draw. Only a token that survived the masks can be returned:
    // `p > 0.0` skips every `-inf` entry (whose `exp` is exactly 0 and so never
    // advances `acc`), which matters at the two boundaries - `r == 0.0` would
    // otherwise return index 0, and a `r == sum` rounding would fall through to
    // the last index, both of which are suppressed specials far more often than
    // they are real codec tokens. `last` is the final surviving candidate, the
    // only correct fall-through.
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    let mut last = 0usize;
    for (i, &p) in scaled.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        acc += p;
        last = i;
        if acc >= r {
            return Draw { token: i as u32, prob: p / sum };
        }
    }
    Draw { token: last as u32, prob: if sum > 0.0 { scaled[last] / sum } else { 1.0 } }
}

/// Run length at which a codebook-0 repeat starts to look like a locked loop
/// rather than a legitimately sustained acoustic token.
///
/// Calibrated against the measured collapse. A HEALTHY clip of the repro
/// sentence emits 34 distinct codebook-0 tokens over 38 frames, so its runs are
/// two or three frames long; the longest run that still escaped was `1657 x10`;
/// the runs inside the locked clip were `706 x20`, `1318 x41`, `617 x52` and
/// `706 x80`. 20 therefore sits an order of magnitude above normal decoding and
/// twice the longest recoverable run, at the bottom of the range where clips
/// stopped coming back. Erring high is deliberate: this reports rather than
/// intervenes, and a false alarm on a legitimately sustained acoustic token
/// would teach people to ignore the line.
pub const DEGENERATE_RUN: usize = 20;

/// Post-filter top-1 probability at which the same collapse becomes
/// unescapable. Measured: a healthy varied prefix draws at 0.74-0.92; the
/// locked loop climbed 0.964 -> 0.978 -> 0.989 -> 0.996 -> 0.9998, and past
/// ~0.99 no temperature-0.9 / top-k-50 draw got out.
pub const DEGENERATE_PROB: f32 = 0.99;

/// Per-clip watcher for the codebook-0 repetition death-spiral.
///
/// **A diagnostic, never a mitigation.** It observes draws and reports; it does
/// not reseed, retune temperature, or force a different token. The sanctioned
/// countermeasure is the reference's `repetition_penalty = 1.05`, which the
/// resolved plan now carries by default. This exists because the reference
/// itself reports that roughly 0.5% of generations can still fail to find EOS
/// even with correct settings, and when that happens the failure should be
/// legible from the run's own stderr instead of costing another multi-hour
/// root-cause session.
#[derive(Clone, Debug, Default)]
pub struct DegenerationWatch {
    last: Option<u32>,
    run: usize,
    reported: bool,
}

impl DegenerationWatch {
    pub fn new() -> DegenerationWatch {
        DegenerationWatch::default()
    }

    /// Current same-token run length (`0` before the first observation).
    pub fn run_len(&self) -> usize {
        self.run
    }

    /// Record one frame's draw. Returns the report line the FIRST time a
    /// contiguous run crosses both thresholds - once per run, not once per
    /// frame, so a 200-frame collapse costs one line rather than 180.
    pub fn observe(&mut self, frame: usize, draw: Draw) -> Option<String> {
        if self.last == Some(draw.token) {
            self.run += 1;
        } else {
            self.last = Some(draw.token);
            self.run = 1;
            self.reported = false;
        }
        if self.reported || self.run <= DEGENERATE_RUN || draw.prob <= DEGENERATE_PROB {
            return None;
        }
        self.reported = true;
        Some(format!(
            "qwen3tts: degenerate autoregressive loop detected at frame {frame}: codebook-0 token {} \
             repeated {} times, post-filter p={:.4} (thresholds: run>{} and p>{}). The clip is very \
             likely to run to its frame cap and decode to near-silence. Sampling was NOT altered - \
             report this run's text, seed and resolved plan.",
            draw.token, self.run, draw.prob, DEGENERATE_RUN, DEGENERATE_PROB
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampled(temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32) -> SamplerCfg {
        SamplerCfg { do_sample: true, temperature, top_k, top_p, repetition_penalty }
    }

    /// Two live candidates at `a` (high) and `b` (lower), everything else far
    /// below, in a vocab wide enough for the suppress window.
    fn logits_two_hot(v: usize, a: usize, b: usize) -> Vec<f32> {
        let mut l = vec![-10.0f32; v];
        l[a] = 5.0;
        l[b] = 4.0;
        l
    }

    #[test]
    fn top_p_excludes_the_long_tail_once_the_top_candidates_cover_it() {
        let (v, a, b, eos) = (2048usize, 10usize, 11usize, 1500u32);
        let mut rng = Rng::new(7);
        for _ in 0..64 {
            let got = sample_cb0(logits_two_hot(v, a, b), eos, false, &sampled(1.0, 0, 0.9, 1.0), &[], &mut rng);
            assert!(got.token == a as u32 || got.token == b as u32, "top_p let a tail token through: {}", got.token);
        }
    }

    #[test]
    fn top_p_disabled_can_sample_outside_the_top_two() {
        let (v, a, b, eos) = (2048usize, 10usize, 11usize, 1500u32);
        let mut rng = Rng::new(3);
        let mut outside = 0;
        for _ in 0..2000 {
            let got = sample_cb0(logits_two_hot(v, a, b), eos, false, &sampled(4.0, 0, 0.0, 1.0), &[], &mut rng);
            if got.token != a as u32 && got.token != b as u32 {
                outside += 1;
            }
        }
        assert!(outside > 0, "top_p=0 (disabled) never sampled outside the top two");
    }

    #[test]
    fn repetition_penalty_demotes_a_token_dominating_greedy_history() {
        let (v, a, b, eos) = (2048usize, 10usize, 11usize, 1500u32);
        let history: Vec<u32> = vec![a as u32; 8];
        let mut rng = Rng::new(1);
        // Greedy (temperature 0) isolates the penalty's effect from the draw.
        let got = sample_cb0(logits_two_hot(v, a, b), eos, false, &sampled(0.0, 0, 0.0, 3.0), &history, &mut rng);
        assert_eq!(got.token, b as u32, "repetition penalty did not demote the repeated token");
        let got_unpenalized = sample_cb0(logits_two_hot(v, a, b), eos, false, &sampled(0.0, 0, 0.0, 1.0), &history, &mut rng);
        assert_eq!(got_unpenalized.token, a as u32, "penalty=1.0 should be a no-op");
    }

    /// The bug this guards: the penalty applied once per OCCURRENCE compounds
    /// to `penalty^count`. Fifty repeats at 1.05 is 11.5x, not 1.05x, which
    /// flips the argmax that the correct processor leaves alone.
    #[test]
    fn repetition_penalty_counts_each_token_once_not_once_per_occurrence() {
        let (v, a, b, eos) = (2048usize, 10usize, 11usize, 1500u32);
        let fifty: Vec<u32> = vec![a as u32; 50];
        let mut rng = Rng::new(1);
        // Correct (once): 5.0/1.05 = 4.76 > 4.0, so `a` still wins.
        // Per-occurrence: 5.0/1.05^50 = 0.44 < 4.0, so `b` would win.
        let got = sample_cb0(logits_two_hot(v, a, b), eos, false, &sampled(0.0, 0, 0.0, 1.05), &fifty, &mut rng);
        assert_eq!(got.token, a as u32, "repetition penalty compounded per occurrence instead of once per distinct token");
    }

    /// Exact logit-level parity with HF's `RepetitionPenaltyLogitsProcessor`,
    /// on both arithmetic branches.
    ///
    /// This is the assertion that would have caught the once-per-occurrence
    /// defect the moment it was written, without a checkpoint, a waveform or an
    /// RMS threshold: token `5` occurs three times in the history and must come
    /// out adjusted EXACTLY once - `1.05` and not `1.05^3` or `1.05^4`. The
    /// positive and negative branches are separate cases because HF divides a
    /// positive logit and MULTIPLIES a negative one (both move it toward zero),
    /// so a fix that only handled one branch would still be wrong.
    #[test]
    fn repetition_penalty_adjusts_a_repeated_logit_exactly_once_on_both_branches() {
        const P: f32 = 1.05;
        let history = [5u32, 5, 5, 8];

        // Positive branch: divide, exactly once.
        let mut pos = vec![0.0f32; 16];
        pos[5] = 2.0;
        pos[8] = 3.0;
        pos[9] = 4.0; // untouched control: not in the history
        apply_repetition_penalty(&mut pos, &history, P);
        assert_eq!(pos[5], 2.0 / P, "positive logit was not divided exactly once (got {}, want {})", pos[5], 2.0 / P);
        assert_eq!(pos[8], 3.0 / P, "a single-occurrence token must take the same single adjustment");
        assert_eq!(pos[9], 4.0, "a token absent from the history must be untouched");

        // Negative branch: multiply, exactly once.
        let mut neg = vec![0.0f32; 16];
        neg[5] = -2.0;
        neg[8] = -3.0;
        neg[9] = -4.0;
        apply_repetition_penalty(&mut neg, &history, P);
        assert_eq!(neg[5], -2.0 * P, "negative logit was not multiplied exactly once (got {}, want {})", neg[5], -2.0 * P);
        assert_eq!(neg[8], -3.0 * P, "a single-occurrence token must take the same single adjustment");
        assert_eq!(neg[9], -4.0, "a token absent from the history must be untouched");

        // And the compounding the old code produced is genuinely a different
        // number, so the assertions above are not vacuously true at this
        // penalty: 1.05^3 differs from 1.05 by ~10%.
        assert!((2.0 / P - 2.0 / P.powi(3)).abs() > 1e-3);
        assert!(((-2.0 * P) - (-2.0 * P.powi(3))).abs() > 1e-3);
    }

    /// `do_sample=false` is greedy even at a positive temperature - the
    /// reference's own switch, independent of this crate's `--temp 0`.
    #[test]
    fn do_sample_false_is_greedy_regardless_of_temperature() {
        let (v, a, b, eos) = (2048usize, 10usize, 11usize, 1500u32);
        let cfg = SamplerCfg { do_sample: false, temperature: 0.9, top_k: 50, top_p: 1.0, repetition_penalty: 1.0 };
        assert!(cfg.is_greedy());
        let mut rng = Rng::new(5);
        for _ in 0..32 {
            let got = sample_cb0(logits_two_hot(v, a, b), eos, false, &cfg, &[], &mut rng);
            assert_eq!(got.token, a as u32, "do_sample=false drew something other than the argmax");
            assert_eq!(got.prob, 1.0);
        }
    }

    /// A near-one-hot distribution must report a near-one probability - this is
    /// the signal the degeneration watch keys on, so it has to be real.
    #[test]
    fn a_near_one_hot_distribution_reports_a_near_one_probability() {
        let v = 2048usize;
        let mut l = vec![-30.0f32; v];
        l[10] = 20.0;
        let got = sample_cb0(l, 1500, false, &sampled(0.9, 50, 1.0, 1.0), &[], &mut Rng::new(2));
        assert_eq!(got.token, 10);
        assert!(got.prob > 0.99, "post-filter probability of a one-hot draw was {}", got.prob);
    }

    #[test]
    fn the_watch_fires_once_per_run_and_only_past_both_thresholds() {
        let mut w = DegenerationWatch::new();
        // A long run at a LOW probability is varied decoding, not a collapse.
        for f in 0..60 {
            assert!(w.observe(f, Draw { token: 7, prob: 0.5 }).is_none());
        }
        // A high-probability draw that has not repeated is fine too.
        let mut w2 = DegenerationWatch::new();
        for f in 0..60 {
            assert!(w2.observe(f, Draw { token: f as u32, prob: 0.9999 }).is_none());
        }
        // Both together fire, exactly once for the run.
        let mut w3 = DegenerationWatch::new();
        let mut fired = 0;
        for f in 0..60 {
            if let Some(line) = w3.observe(f, Draw { token: 706, prob: 0.9998 }) {
                fired += 1;
                assert!(line.contains("706"), "report must name the token: {line}");
                // run length crosses 20 on the 21st observation, index 20.
                assert!(line.contains("frame 20"), "report must name the frame it tripped on: {line}");
                assert!(line.contains("repeated 21 times"), "report must name the run length: {line}");
            }
        }
        assert_eq!(fired, 1, "the watch must report once per run, not once per frame");
        // A different token resets the run, and a fresh collapse reports again.
        assert!(w3.observe(60, Draw { token: 12, prob: 0.5 }).is_none());
        assert_eq!(w3.run_len(), 1);
        let mut fired2 = 0;
        for f in 61..100 {
            if w3.observe(f, Draw { token: 617, prob: 0.999 }).is_some() {
                fired2 += 1;
            }
        }
        assert_eq!(fired2, 1, "a NEW run must be reportable after the first one ended");
    }
}
