// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What an episode will be checked on, frozen before it is ever trained on.
//!
//! A probe here is **held-out line continuation**: some lines of the episode
//! are withheld from training and become the questions, with the lines before
//! them as the prompt. That keeps verification programmatic and exact - a
//! decoded answer either is the withheld line or is not - with no second
//! model judging the first.
//!
//! ## The three families, and which of them this crate can build
//!
//! - [`ProbeFamily::Literal`] - the withheld line itself. Built here.
//! - [`ProbeFamily::Counterfactual`] - two withheld lines sharing a long
//!   prefix and diverging after it, each asked from that shared prefix. A
//!   model that learned the mapping answers each with its own ending; one
//!   that memorised the surface answers both with whichever it saw more.
//!   Built here, when the episode contains such a pair.
//! - [`ProbeFamily::Paraphrase`] - the same question asked in different
//!   words. **This crate cannot build one**, and says so rather than
//!   pretending: paraphrasing needs to know what the text MEANS, which is
//!   either a model or a corpus that generated itself and kept the semantics.
//!   A caller that has one supplies it; [`ProbeSet::coverage`] reports
//!   honestly when nobody did, so a run cannot quietly claim a control it
//!   never ran.
//!
//! ## Two separations that make the numbers mean something
//!
//! **Selected against blind.** A selector decides which lines are worth
//! probing, and a selector that prefers easy lines makes everything pass. So
//! a second, independent draw is taken uniformly from the lines the selector
//! did NOT want, and both are withheld from training. If the selected probes
//! pass at a materially higher rate than the blind ones, the selection rule
//! is the result rather than the model.
//!
//! One property of that first separation is worth stating before it confuses
//! someone: a refusal is about THIS DRAW, not about the episode in the
//! abstract. A line restated elsewhere in the same episode leaks only when
//! one copy is withheld and the other trained. If both land in training there
//! is no probe to leak into, and if both are withheld there is nothing
//! trained to leak from - so the same file can build cleanly at one seed and
//! be refused at another, and that is correct rather than flaky.
//!
//! **A zero-shot baseline, frozen at ingest.** Real documents are not
//! difficulty-invariant, so an absolute score is not comparable between two
//! episodes and a trend across a stream of them means nothing. Every probe
//! carries what the UNTRAINED model scored on it, frozen before any training,
//! and [`ProbeSet::delta`] refuses to report anything until that column
//! exists.

use std::collections::BTreeSet;

use promote::document::normalize;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::stream::{Episode, EpisodeId};

/// Which control a probe belongs to. See this module's doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProbeFamily {
    Literal,
    Counterfactual,
    Paraphrase,
}

/// Stable identity of one probe, derived from its episode, family and the
/// line it was frozen from - never from its position in a vector, which
/// changes whenever the config does.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProbeId(String);

impl ProbeId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One frozen question over an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    pub id: ProbeId,
    pub family: ProbeFamily,
    /// What the model is shown.
    pub prompt: String,
    /// The one string that counts as correct, after [`normalize`].
    pub expected: String,
    /// Which line of the episode it was frozen from.
    pub line: usize,
    /// Drawn by the independent audit rule rather than by the selector.
    pub blind: bool,
    /// What the UNTRAINED model scored here, frozen at ingest. `None` until
    /// [`ProbeSet::freeze_baseline`] has been called.
    pub baseline: Option<f64>,
}

/// How an episode is cut into questions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Lines of preceding context a prompt carries.
    pub context_lines: usize,
    /// Share of eligible lines the selector withholds, in parts per thousand.
    pub probe_permille: u32,
    /// Share withheld for the independent blind draw, in parts per thousand.
    pub blind_permille: u32,
    /// A line shorter than this is not worth asking about.
    pub min_answer_chars: usize,
    /// How long a shared prefix must be before two lines count as a
    /// counterfactual pair.
    pub counterfactual_prefix: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        ProbeConfig { context_lines: 4, probe_permille: 200, blind_permille: 100, min_answer_chars: 8, counterfactual_prefix: 8 }
    }
}

/// Decides which lines are worth probing.
///
/// This exists to be AUDITED, not trusted: whatever it prefers, the blind
/// draw is taken from what it rejected, and [`ProbeSet::selection_bias`]
/// compares the two.
pub trait SpanSelector {
    /// Higher means more worth asking about. Must be a pure function of the
    /// line: a selector that adapts to results is a selector that can be
    /// tuned until everything passes.
    fn score(&self, line: &str) -> f64;
}

/// Every eligible line is equally worth probing.
#[derive(Debug, Default, Clone, Copy)]
pub struct UniformSelector;

impl SpanSelector for UniformSelector {
    fn score(&self, _line: &str) -> f64 {
        0.0
    }
}

/// Which controls a probe set can actually support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coverage {
    pub literal: usize,
    pub counterfactual: usize,
    pub paraphrase: usize,
    pub blind: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum BankError {
    #[error("probe answer frozen from line {line} also appears in a row that WILL be trained on ({answer:?}) - the probe would be measuring memorisation of the training half, not retention")]
    ProbeAnswerInTrainedRow { line: usize, answer: String },
    #[error("episode {0} has no line long enough to probe (min_answer_chars)")]
    NoEligibleLines(String),
    #[error("expected {expected} scores, one per probe in order, got {got}")]
    ScoreArityMismatch { expected: usize, got: usize },
    #[error("no zero-shot baseline has been frozen for this probe set - an absolute score is not comparable across episodes, so a delta cannot be reported")]
    BaselineNotFrozen,
}

type Result<T> = std::result::Result<T, BankError>;

/// Every question frozen over one episode, plus the rows that may be trained.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeSet {
    episode: EpisodeId,
    probes: Vec<Probe>,
    trained: Vec<String>,
    pairs: Vec<(ProbeId, ProbeId)>,
}

impl ProbeSet {
    /// Freeze the questions for `ep`.
    ///
    /// Fails rather than warns when a probe's answer also appears in a row
    /// that will be trained on: that probe would measure memorisation of the
    /// training half rather than anything retained, and a set containing one
    /// is not a set whose numbers can be reported.
    pub fn build(ep: &Episode, cfg: &ProbeConfig, selector: &dyn SpanSelector) -> Result<ProbeSet> {
        let lines: Vec<&str> = ep.text.lines().collect();
        let eligible: Vec<usize> =
            (0..lines.len()).filter(|&i| lines[i].trim().chars().count() >= cfg.min_answer_chars).collect();
        if eligible.is_empty() {
            return Err(BankError::NoEligibleLines(ep.id.as_str().to_string()));
        }

        // Ranked by the selector, ties broken by a digest of (episode, line)
        // rather than by position: an ordering that fell out of iteration
        // order would change whenever the config did, and a probe set that
        // moves is a probe set nothing can be frozen against.
        let mut ranked = eligible.clone();
        ranked.sort_by(|&a, &b| {
            selector
                .score(lines[b])
                .partial_cmp(&selector.score(lines[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| tiebreak(&ep.id, a).cmp(&tiebreak(&ep.id, b)))
        });

        let take = |permille: u32| (eligible.len() as u64 * permille as u64 / 1000) as usize;
        let n_selected = take(cfg.probe_permille).min(ranked.len());
        let selected: BTreeSet<usize> = ranked[..n_selected].iter().copied().collect();

        // The blind draw comes from what the selector did NOT want, which is
        // what makes the comparison in `selection_bias` mean anything. It is
        // ordered by the same tiebreak digest, so it is a draw the selector
        // has no influence over rather than simply the next ones down its
        // own ranking.
        let mut rest: Vec<usize> = ranked[n_selected..].to_vec();
        rest.sort_by_key(|&i| tiebreak(&ep.id, i));
        let n_blind = take(cfg.blind_permille).min(rest.len());
        let blind: BTreeSet<usize> = rest[..n_blind].iter().copied().collect();

        let held: BTreeSet<usize> = selected.union(&blind).copied().collect();
        let trained: Vec<String> = (0..lines.len()).filter(|i| !held.contains(i)).map(|i| lines[i].to_string()).collect();

        // Containment, not equality: a probe answer reproduced ANYWHERE
        // inside a trained row is answerable from training, and real text
        // repeats itself in ways synthetic curricula never do.
        let trained_norm: Vec<String> = trained.iter().map(|r| normalize(r)).collect();
        for &i in &held {
            let answer = normalize(lines[i]);
            if answer.is_empty() {
                continue;
            }
            if trained_norm.iter().any(|r| r.contains(&answer)) {
                return Err(BankError::ProbeAnswerInTrainedRow { line: i, answer: lines[i].to_string() });
            }
        }

        let prompt_for = |i: usize| -> String {
            let lo = i.saturating_sub(cfg.context_lines);
            let mut p = lines[lo..i].join("\n");
            if !p.is_empty() {
                p.push('\n');
            }
            p
        };

        let mut probes: Vec<Probe> = held
            .iter()
            .map(|&i| Probe {
                id: probe_id(&ep.id, ProbeFamily::Literal, i),
                family: ProbeFamily::Literal,
                prompt: prompt_for(i),
                expected: lines[i].to_string(),
                line: i,
                blind: blind.contains(&i),
                baseline: None,
            })
            .collect();

        // Counterfactuals: two withheld lines sharing a long prefix and
        // diverging after it. Each is asked FROM the shared prefix, so the
        // only thing separating the two questions is what the model learned
        // about the part that differs. Paired in line order and never reused,
        // so the pairing is a function of the episode alone.
        let mut pairs: Vec<(ProbeId, ProbeId)> = Vec::new();
        let mut taken: BTreeSet<usize> = BTreeSet::new();
        let held_v: Vec<usize> = held.iter().copied().collect();
        for (a_pos, &a) in held_v.iter().enumerate() {
            if taken.contains(&a) {
                continue;
            }
            for &b in &held_v[a_pos + 1..] {
                if taken.contains(&b) {
                    continue;
                }
                let n = shared_prefix(lines[a], lines[b]);
                if n >= cfg.counterfactual_prefix && lines[a] != lines[b] {
                    let shared = &lines[a][..n];
                    for &m in &[a, b] {
                        probes.push(Probe {
                            id: probe_id(&ep.id, ProbeFamily::Counterfactual, m),
                            family: ProbeFamily::Counterfactual,
                            prompt: format!("{}{}", prompt_for(m), shared),
                            expected: lines[m][n..].to_string(),
                            line: m,
                            blind: blind.contains(&m),
                            baseline: None,
                        });
                    }
                    pairs.push((probe_id(&ep.id, ProbeFamily::Counterfactual, a), probe_id(&ep.id, ProbeFamily::Counterfactual, b)));
                    taken.insert(a);
                    taken.insert(b);
                    break;
                }
            }
        }

        probes.sort_by(|x, y| x.line.cmp(&y.line).then_with(|| x.family.cmp(&y.family)));
        Ok(ProbeSet { episode: ep.id.clone(), probes, trained, pairs })
    }

    pub fn episode(&self) -> &EpisodeId {
        &self.episode
    }

    pub fn probes(&self) -> &[Probe] {
        &self.probes
    }

    /// The rows of this episode that may be trained on. Disjoint from every
    /// probe answer by construction.
    pub fn trained_rows(&self) -> &[String] {
        &self.trained
    }

    /// Counterfactual probes that are each other's minimal edit.
    pub fn counterfactual_pairs(&self) -> &[(ProbeId, ProbeId)] {
        &self.pairs
    }

    pub fn coverage(&self) -> Coverage {
        let n = |f: ProbeFamily| self.probes.iter().filter(|p| p.family == f).count();
        Coverage {
            literal: n(ProbeFamily::Literal),
            counterfactual: n(ProbeFamily::Counterfactual),
            paraphrase: n(ProbeFamily::Paraphrase),
            blind: self.probes.iter().filter(|p| p.blind).count(),
        }
    }

    /// Record what the untrained model scored, in probe order. See this
    /// module's doc on why an absolute score is not enough.
    pub fn freeze_baseline(&mut self, scores: &[f64]) -> Result<()> {
        if scores.len() != self.probes.len() {
            return Err(BankError::ScoreArityMismatch { expected: self.probes.len(), got: scores.len() });
        }
        for (p, s) in self.probes.iter_mut().zip(scores) {
            p.baseline = Some(*s);
        }
        Ok(())
    }

    /// Mean improvement over this episode's own frozen baseline.
    pub fn delta(&self, scores: &[f64]) -> Result<f64> {
        if scores.len() != self.probes.len() {
            return Err(BankError::ScoreArityMismatch { expected: self.probes.len(), got: scores.len() });
        }
        let mut base = 0.0;
        for p in &self.probes {
            base += p.baseline.ok_or(BankError::BaselineNotFrozen)?;
        }
        let n = self.probes.len() as f64;
        Ok(scores.iter().sum::<f64>() / n - base / n)
    }

    /// Selected pass rate minus blind pass rate. `None` when either draw is
    /// empty, since a difference needs both sides.
    ///
    /// A materially positive value means the selector is picking easier lines
    /// than a uniform draw would, so the probe set is reporting the selection
    /// rule rather than the model.
    pub fn selection_bias(&self, scores: &[f64]) -> Result<Option<f64>> {
        if scores.len() != self.probes.len() {
            return Err(BankError::ScoreArityMismatch { expected: self.probes.len(), got: scores.len() });
        }
        let mean = |blind: bool| -> Option<f64> {
            let v: Vec<f64> = self.probes.iter().zip(scores).filter(|(p, _)| p.blind == blind).map(|(_, s)| *s).collect();
            (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
        };
        Ok(match (mean(false), mean(true)) {
            (Some(sel), Some(bl)) => Some(sel - bl),
            _ => None,
        })
    }
}

/// Deterministic ordering key for a line of an episode, so a tie between two
/// equally-scored lines is broken by the content rather than by the order a
/// vector happened to be in.
fn tiebreak(ep: &EpisodeId, line: usize) -> String {
    let mut h = Sha256::new();
    h.update(ep.as_str().as_bytes());
    h.update(line.to_le_bytes());
    format!("{:x}", h.finalize())
}

fn probe_id(ep: &EpisodeId, family: ProbeFamily, line: usize) -> ProbeId {
    let mut h = Sha256::new();
    h.update(ep.as_str().as_bytes());
    h.update(format!("{family:?}").as_bytes());
    h.update(line.to_le_bytes());
    ProbeId(format!("{:x}", h.finalize()))
}

/// Length in BYTES of the longest common prefix of `a` and `b` that ends on a
/// character boundary, so slicing at it cannot panic on multi-byte text.
fn shared_prefix(a: &str, b: &str) -> usize {
    let mut n = 0;
    for ((ia, ca), (_, cb)) in a.char_indices().zip(b.char_indices()) {
        if ca != cb {
            return ia;
        }
        n = ia + ca.len_utf8();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn episode(text: &str) -> Episode {
        Episode { id: EpisodeId::of(text), source: PathBuf::from("manual.txt"), ordinal: 0, text: text.to_string() }
    }

    /// A manual page shaped corpus: enough distinct long lines to withhold
    /// from, and no repeats.
    fn manual(n: usize) -> String {
        (0..n).map(|i| format!("--flag{i:03} VALUE   set the {i:03} option to VALUE\n")).collect()
    }

    fn cfg() -> ProbeConfig {
        ProbeConfig { probe_permille: 300, blind_permille: 200, ..ProbeConfig::default() }
    }

    /// Scores of 1.0 for every probe, so a control that needs scores can be
    /// exercised without a model.
    fn all_pass(s: &ProbeSet) -> Vec<f64> {
        vec![1.0; s.probes().len()]
    }

    /// V13's first half. Selection must be reproducible from the content
    /// alone - not from wall-clock, not from iteration order, and not from
    /// anything that could be retried until the probes came out easy.
    #[test]
    fn probe_selection_is_a_pure_function_of_the_episode_content() {
        let ep = episode(&manual(60));
        let a = ProbeSet::build(&ep, &cfg(), &UniformSelector).expect("build");
        let b = ProbeSet::build(&ep, &cfg(), &UniformSelector).expect("build");
        let ids = |s: &ProbeSet| s.probes().iter().map(|p| (p.id.clone(), p.line)).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b), "the same episode must freeze the same probes");
        assert!(!a.probes().is_empty(), "a 60 line manual must yield probes");

        let other = ProbeSet::build(&episode(&manual(61)), &cfg(), &UniformSelector).expect("build");
        assert_ne!(ids(&a), ids(&other), "different content must not freeze the identical probe set");
    }

    /// V1, the half that fires. A line that appears twice cannot be both
    /// withheld and trained on: whichever copy is probed would be answerable
    /// from the copy that was trained.
    #[test]
    fn a_probe_answer_that_also_appears_in_a_trained_row_is_refused_at_ingest() {
        // One line is restated inside a longer one further down. Which side
        // of the split each lands on is otherwise up to the tiebreak, so the
        // selector pins it: the restated line is ranked first and therefore
        // withheld, the line quoting it is ranked last and therefore trained.
        struct Pin;
        impl SpanSelector for Pin {
            fn score(&self, line: &str) -> f64 {
                if line.starts_with("--flag007") {
                    1.0
                } else if line.starts_with("alias") {
                    -1.0
                } else {
                    0.0
                }
            }
        }

        let mut text = manual(40);
        text.push_str("alias for the above: --flag007 VALUE   set the 007 option to VALUE\n");
        // No blind draw here: it is taken from what the selector rejected,
        // which is exactly where the quoting line was put.
        let c = ProbeConfig { blind_permille: 0, ..cfg() };
        match ProbeSet::build(&episode(&text), &c, &Pin) {
            Err(BankError::ProbeAnswerInTrainedRow { answer, .. }) => {
                assert!(answer.contains("007"), "the refusal must name the offending answer, got {answer:?}");
            }
            other => panic!("expected ProbeAnswerInTrainedRow, got {:?}", other.map(|s| s.probes().len())),
        }
    }

    /// V1, the half that stays silent. A bar that refuses everything is not a
    /// bar.
    #[test]
    fn an_episode_with_no_repeated_lines_builds_a_clean_probe_set() {
        let s = ProbeSet::build(&episode(&manual(40)), &cfg(), &UniformSelector).expect("a clean episode must build");
        let answers: BTreeSet<String> = s.probes().iter().map(|p| normalize(&p.expected)).collect();
        for row in s.trained_rows() {
            assert!(!answers.contains(&normalize(row)), "a trained row reproduced a probe answer: {row:?}");
        }
    }

    /// The blind draw only means something if it is genuinely separate from
    /// the selected one AND equally withheld from training.
    #[test]
    fn the_blind_draw_is_disjoint_from_the_selected_draw_and_both_are_held_out() {
        let s = ProbeSet::build(&episode(&manual(60)), &cfg(), &UniformSelector).expect("build");
        let selected: BTreeSet<usize> = s.probes().iter().filter(|p| !p.blind).map(|p| p.line).collect();
        let blind: BTreeSet<usize> = s.probes().iter().filter(|p| p.blind).map(|p| p.line).collect();
        assert!(!selected.is_empty() && !blind.is_empty(), "both draws must be non-empty at this config");
        assert!(selected.is_disjoint(&blind), "a line cannot be in both draws");

        let trained: BTreeSet<String> = s.trained_rows().iter().map(|r| normalize(r)).collect();
        for p in s.probes() {
            assert!(!trained.contains(&normalize(&p.expected)), "probe line {} was also handed to training", p.line);
        }
    }

    /// V13's second half, as a pair. A selector that prefers lines it knows
    /// are easy shows up as bias; one that does not, does not.
    #[test]
    fn a_biased_selector_shows_up_as_selection_bias_and_an_unbiased_one_does_not() {
        struct PrefersShort;
        impl SpanSelector for PrefersShort {
            fn score(&self, line: &str) -> f64 {
                -(line.len() as f64)
            }
        }

        let ep = episode(&manual(60));
        let biased = ProbeSet::build(&ep, &cfg(), &PrefersShort).expect("build");
        let fair = ProbeSet::build(&ep, &cfg(), &UniformSelector).expect("build");

        // Synthetic scores standing in for a model: the selector's own
        // favourites pass, everything else fails.
        let scores = |s: &ProbeSet| -> Vec<f64> { s.probes().iter().map(|p| if p.blind { 0.0 } else { 1.0 }).collect() };
        let b = biased.selection_bias(&scores(&biased)).expect("scored").expect("both draws present");
        assert!(b > 0.5, "a selector whose picks all pass while the blind draw all fail must read as biased, got {b}");

        let even: Vec<f64> = vec![1.0; fair.probes().len()];
        let f = fair.selection_bias(&even).expect("scored").expect("both draws present");
        assert!(f.abs() < 1e-9, "equal pass rates on both draws must read as no bias, got {f}");
    }

    /// V3. Two lines that share a long prefix and diverge are exactly the
    /// surface-memorisation trap, and the pair is what detects it.
    #[test]
    fn counterfactual_pairs_are_found_for_lines_sharing_a_long_prefix() {
        struct PrefersRetention;
        impl SpanSelector for PrefersRetention {
            fn score(&self, line: &str) -> f64 {
                if line.contains("retention") {
                    1.0
                } else {
                    0.0
                }
            }
        }

        let mut text = manual(30);
        text.push_str("--retention-window TIME   drop entries older than TIME\n");
        text.push_str("--retention-window SIZE   drop entries beyond SIZE bytes\n");
        let s = ProbeSet::build(&episode(&text), &ProbeConfig { counterfactual_prefix: 18, ..cfg() }, &PrefersRetention).expect("build");

        assert!(!s.counterfactual_pairs().is_empty(), "two lines sharing 19 characters must pair");
        let cf: Vec<&Probe> = s
            .probes()
            .iter()
            .filter(|p| p.family == ProbeFamily::Counterfactual && p.prompt.contains("--retention-window"))
            .collect();
        assert!(cf.len() >= 2, "a pair must produce a probe for each member, got {}", cf.len());
        for p in &cf {
            assert!(!p.expected.is_empty(), "a counterfactual is answered with what follows the shared prefix");
            assert!(!p.expected.contains("--retention-window"), "the shared prefix belongs to the PROMPT, not the answer");
        }
        let endings: BTreeSet<String> = cf.iter().map(|p| normalize(&p.expected)).collect();
        assert!(endings.len() >= 2, "the two members must have different answers, or the pair proves nothing");
    }

    /// The other half. An episode with nothing near-duplicate in it has no
    /// counterfactual coverage, and must SAY so rather than report a control
    /// it did not run.
    #[test]
    fn an_episode_with_no_near_duplicate_lines_reports_no_counterfactual_coverage() {
        // `--flag000 ...` and `--flag001 ...` share only eight characters, so
        // a forty character requirement finds nothing to pair.
        let s = ProbeSet::build(&episode(&manual(30)), &ProbeConfig { counterfactual_prefix: 40, ..cfg() }, &UniformSelector).expect("build");
        assert!(s.counterfactual_pairs().is_empty());
        assert_eq!(s.coverage().counterfactual, 0);
        assert_eq!(s.coverage().paraphrase, 0, "this crate cannot build paraphrases and must not claim any");
        assert!(s.coverage().literal > 0);
    }

    /// V10. A raw score is not comparable between two episodes of real text,
    /// so a delta without a frozen baseline is refused rather than computed
    /// against an implied zero.
    #[test]
    fn a_delta_cannot_be_reported_before_the_zero_shot_baseline_is_frozen() {
        let mut s = ProbeSet::build(&episode(&manual(40)), &cfg(), &UniformSelector).expect("build");
        let scores = all_pass(&s);
        assert!(matches!(s.delta(&scores), Err(BankError::BaselineNotFrozen)));

        let base = vec![0.25; s.probes().len()];
        s.freeze_baseline(&base).expect("freeze");
        let d = s.delta(&scores).expect("a frozen baseline makes a delta reportable");
        assert!((d - 0.75).abs() < 1e-9, "delta must be measured against this episode's own baseline, got {d}");

        assert!(matches!(s.freeze_baseline(&[0.1]), Err(BankError::ScoreArityMismatch { .. })));
    }
}
