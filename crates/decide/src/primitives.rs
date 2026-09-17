// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The three question types a caller may ask, and the answers they return.
//!
//! One mechanism underneath all three: a score per option, softmaxed within
//! the question. `Noul` is that over `{yes, no}` and `Score` is that over the
//! supplied levels, so nothing here is a special case in the model - only in
//! what the caller is handed back.
//!
//! **A `Score`'s levels are scored independently.** No level sees its own
//! index or its neighbours; the ordinal reading is applied afterwards, when
//! the expectation is taken. That falls out of per-option scoring rather than
//! being arranged, and it is why a level's text has to stand on its own -
//! "worse than the previous one" describes nothing the model can see.

use crate::loss::softmax;

/// A `Choice` may carry this many options.
pub const MAX_OPTIONS: usize = 255;
/// A `Score` needs at least two levels and takes at most this many.
pub const MIN_LEVELS: usize = 2;
pub const MAX_LEVELS: usize = 10;

/// One option of a `Choice`: the name the caller gets back, and an optional
/// description that only the model sees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opt {
    pub name: String,
    pub description: Option<String>,
}

impl Opt {
    pub fn new(name: impl Into<String>) -> Opt {
        Opt { name: name.into(), description: None }
    }

    pub fn described(name: impl Into<String>, description: impl Into<String>) -> Opt {
        Opt { name: name.into(), description: Some(description.into()) }
    }
}

#[derive(Clone, Debug)]
pub enum Question {
    /// One of the supplied options.
    Choice { instructions: String, options: Vec<Opt> },
    /// A position along the supplied levels, in order.
    Score { instructions: String, levels: Vec<String> },
    /// The probability that a proposition holds.
    Noul { instructions: String, yes: Option<String>, no: Option<String> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Choice { choice: String, probabilities: Vec<(String, f32)>, confidence: f32 },
    Score { score: f32, legend: Vec<String>, probabilities: Vec<f32>, confidence: f32 },
    /// No `confidence`: a two-outcome probability already IS its own
    /// confidence, and publishing both would invite a caller to threshold the
    /// derived number instead of the one that means something.
    Noul { noul: f32 },
}

/// The logistic, which is what turns a `Noul`'s single score into a
/// probability.
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        let e = (-z as f64).exp();
        (1.0 / (1.0 + e)) as f32
    } else {
        // Written both ways round so neither tail overflows.
        let e = (z as f64).exp();
        (e / (1.0 + e)) as f32
    }
}

/// How concentrated a distribution is, on `[0, 1]`.
///
/// `1 - H/ln(k)`: exactly 1 on a one-hot, exactly 0 on a uniform, and - the
/// reason for this form rather than the maximum probability - **independent of
/// how many options there are**. `p_max` has a floor of `1/k`, so it reads 0.5
/// on a two-option question and 0.004 on a 255-option one for the same
/// completely-undecided answer. Since the option set is supplied per request,
/// a threshold written against `p_max` would mean something different on every
/// call, which defeats the purpose of publishing one.
///
/// The full distribution is always returned as well, so a caller who wants
/// `p_max`, the margin, or anything else computes it from that.
pub fn confidence(probs: &[f32]) -> f32 {
    let k = probs.len();
    if k <= 1 {
        // One option is not a decision; there is nothing to be unsure between.
        return 1.0;
    }
    let h: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    (1.0 - h / (k as f64).ln()).clamp(0.0, 1.0) as f32
}

impl Question {
    /// The option count this question scores.
    pub fn arity(&self) -> usize {
        match self {
            Question::Choice { options, .. } => options.len(),
            Question::Score { levels, .. } => levels.len(),
            // ONE slot, not two - see `slots`.
            Question::Noul { .. } => 1,
        }
    }

    /// Reject a question the model cannot answer, by name. The limits mirror
    /// the published contract so a request that is valid there is valid here.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Question::Choice { options, .. } => {
                if options.is_empty() {
                    return Err("a choice needs at least one option".into());
                }
                if options.len() > MAX_OPTIONS {
                    return Err(format!("a choice takes at most {MAX_OPTIONS} options, got {}", options.len()));
                }
                Ok(())
            }
            Question::Score { levels, .. } => {
                if !(MIN_LEVELS..=MAX_LEVELS).contains(&levels.len()) {
                    return Err(format!(
                        "a score takes {MIN_LEVELS} to {MAX_LEVELS} levels, got {}",
                        levels.len()
                    ));
                }
                Ok(())
            }
            Question::Noul { .. } => Ok(()),
        }
    }

    /// The slot text scored for each option: the question's instructions, then
    /// the option, joined by a real `[SEP]`.
    ///
    /// Instructions travel with the OPTION and never with the state. That is
    /// what lets one state encoding serve every question in a request, and it
    /// is the reason question independence is structural here rather than a
    /// promise.
    ///
    /// **A `Noul` emits ONE slot, and this is load-bearing.** The obvious
    /// design is two - "yes" and "no" - scored against each other by softmax,
    /// and it does not work. The head uses a slot's representation as its
    /// attention QUERY, and two slots that differ by a word or two are nearly
    /// parallel: measured on the released checkpoint, the cosine between a
    /// yes-slot and its no-slot is **0.989**. Two nearly-parallel queries
    /// attend to the state almost identically, so both scores carry the same
    /// state information and a softmax - which sees only their DIFFERENCE -
    /// subtracts it right back out.
    ///
    /// The numbers, on three deliberately extreme conversations:
    ///
    /// ```text
    ///                     yes      no    yes-no
    /// obvious close    0.5740  0.6161   -0.0421
    /// obvious loss     0.6482  0.6969   -0.0487
    /// undecided        0.3587  0.4008   -0.0421
    ///                  \_____________/  \______/
    ///                  spread 0.2895    spread 0.0066
    /// ```
    ///
    /// The state signal is plainly there in the absolute scores and gone from
    /// their difference. So a `Noul` scores ONE slot and reads a logistic off
    /// it, which keeps the absolute score; the `no` text is carried for the
    /// answer's legend and never encoded.
    ///
    /// A `Choice` is not affected because its options are genuinely different
    /// text, which is what gives a softmax something to work with.
    pub fn slots(&self) -> Vec<String> {
        let join = |instructions: &str, tail: &str| format!("{instructions} [SEP] {tail}");
        match self {
            Question::Choice { instructions, options } => options
                .iter()
                .map(|o| match &o.description {
                    Some(d) => join(instructions, &format!("{} [SEP] {d}", o.name)),
                    None => join(instructions, &o.name),
                })
                .collect(),
            Question::Score { instructions, levels } => {
                levels.iter().map(|l| join(instructions, l)).collect()
            }
            // ONE slot: the proposition itself. `no` never becomes a slot -
            // see this method's own note on why two of them cannot work.
            Question::Noul { instructions, yes, .. } => match yes {
                Some(y) => vec![join(instructions, y)],
                None => vec![instructions.clone()],
            },
        }
    }

    /// Turn this question's option scores into its answer.
    pub fn answer(&self, scores: &[f32]) -> Answer {
        assert_eq!(scores.len(), self.arity(), "one score per option");
        let p = softmax(scores);
        match self {
            Question::Choice { options, .. } => {
                let best = argmax(&p);
                Answer::Choice {
                    choice: options[best].name.clone(),
                    probabilities: options.iter().map(|o| o.name.clone()).zip(p.iter().copied()).collect(),
                    confidence: confidence(&p),
                }
            }
            Question::Score { levels, .. } => {
                // The 0-BASED expectation over level indices: a distribution
                // sitting entirely on the first level scores 0, not 1.
                let score = p.iter().enumerate().map(|(i, &pi)| i as f32 * pi).sum();
                Answer::Score {
                    score,
                    legend: levels.clone(),
                    confidence: confidence(&p),
                    probabilities: p,
                }
            }
            // A logistic on the single score, NOT a softmax over two. The
            // softmax sees only the DIFFERENCE between its inputs, and for a
            // yes/no pair that difference is where the answer is not.
            Question::Noul { .. } => Answer::Noul { noul: sigmoid(scores[0]) },
        }
    }
}

/// First maximal index, so ties resolve to the option the caller listed first
/// rather than to whichever the float comparison happened to visit last.
fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(n: usize) -> Question {
        Question::Choice {
            instructions: "which team".into(),
            options: (0..n).map(|i| Opt::new(format!("opt{i}"))).collect(),
        }
    }

    #[test]
    fn confidence_is_one_on_a_one_hot_and_zero_on_a_uniform() {
        assert!((confidence(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(confidence(&[1.0 / 3.0; 3]).abs() < 1e-6);
    }

    /// The property `p_max` does not have, and the reason this form was
    /// chosen: a completely undecided answer reads 0 whether there are two
    /// options or two hundred, so ONE published threshold means one thing.
    #[test]
    fn confidence_does_not_move_with_the_option_count() {
        for k in [2usize, 5, 77, 255] {
            let uniform = vec![1.0 / k as f32; k];
            assert!(confidence(&uniform).abs() < 1e-5, "k={k} uniform read {}", confidence(&uniform));
            let mut one_hot = vec![0.0; k];
            one_hot[0] = 1.0;
            assert!((confidence(&one_hot) - 1.0).abs() < 1e-5, "k={k} one-hot");
        }
    }

    #[test]
    fn the_limits_match_the_published_contract() {
        assert!(choice(255).validate().is_ok());
        assert!(choice(256).validate().unwrap_err().contains("255"));
        assert!(choice(0).validate().is_err());
        let lv = |n: usize| Question::Score {
            instructions: "how bad".into(),
            levels: (0..n).map(|i| format!("l{i}")).collect(),
        };
        assert!(lv(2).validate().is_ok());
        assert!(lv(10).validate().is_ok());
        assert!(lv(1).validate().is_err());
        assert!(lv(11).validate().is_err());
    }

    /// The expectation is 0-based: all the mass on the first level is 0.
    #[test]
    fn a_score_is_the_zero_based_expectation_over_levels() {
        let q = Question::Score {
            instructions: "frustration".into(),
            levels: vec!["calm".into(), "annoyed".into(), "angry".into()],
        };
        // Scores chosen so the softmax is ~[0, 0.7, 0.3]: expected 1.3.
        let p = [(0.0f32).ln_1p() - 50.0, (0.7f32 / 0.3).ln(), 0.0];
        let Answer::Score { score, .. } = q.answer(&p) else { panic!("wrong variant") };
        assert!((score - 1.3).abs() < 0.01, "score {score}");
        let all_first = q.answer(&[50.0, 0.0, 0.0]);
        let Answer::Score { score, .. } = all_first else { panic!("wrong variant") };
        assert!(score.abs() < 1e-3, "all mass on level 0 must score 0, got {score}");
    }

    /// Instructions belong to the slot, never to the state - the property that
    /// makes one state encoding serve every question.
    #[test]
    fn every_slot_carries_the_instructions() {
        let q = choice(3);
        let slots = q.slots();
        assert_eq!(slots.len(), 3);
        assert!(slots.iter().all(|s| s.starts_with("which team [SEP] ")));
        assert!(slots[2].ends_with("opt2"));
    }

    #[test]
    fn a_noul_scores_one_slot_and_reports_a_logistic_of_it() {
        let q = Question::Noul { instructions: "is it urgent".into(), yes: None, no: None };
        assert_eq!(q.arity(), 1, "a noul scores one slot");
        assert_eq!(q.slots(), vec!["is it urgent".to_string()]);
        // A logistic, so the ABSOLUTE score is what moves the answer - which
        // is the whole reason this is not a two-slot softmax.
        let p = |z: f32| match q.answer(&[z]) {
            Answer::Noul { noul } => noul,
            other => panic!("a noul answered {other:?}"),
        };
        assert!((p(0.0) - 0.5).abs() < 1e-6, "a zero score is 0.5, got {}", p(0.0));
        assert!(p(2.0) > 0.85, "a positive score should lean yes, got {}", p(2.0));
        assert!(p(-2.0) < 0.15, "a negative score should lean no, got {}", p(-2.0));
        assert!(p(2.0) > p(0.5) && p(0.5) > p(-0.5), "not monotone in the score");
    }

    /// A supplied positive label enters the slot; the negative one never does.
    #[test]
    fn a_nouls_negative_label_is_legend_only() {
        let q = Question::Noul {
            instructions: "does this convert".into(),
            yes: Some("the deal closes".into()),
            no: Some("the deal is lost".into()),
        };
        let slots = q.slots();
        assert_eq!(slots.len(), 1);
        assert!(slots[0].contains("the deal closes"));
        assert!(!slots[0].contains("lost"), "the negative label was encoded: {:?}", slots[0]);
    }

    /// The logistic must not overflow at either tail - a saturated score is a
    /// normal thing for a trained model to produce.
    #[test]
    fn the_logistic_survives_both_tails() {
        for z in [-120.0f32, -40.0, 0.0, 40.0, 120.0] {
            let p = sigmoid(z);
            assert!(p.is_finite() && (0.0..=1.0).contains(&p), "sigmoid({z}) = {p}");
        }
        assert!(sigmoid(-120.0) < 1e-6 && sigmoid(120.0) > 1.0 - 1e-6);
    }
}
