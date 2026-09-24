// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Turning source material into question/answer pairs a model can be trained
//! to answer.
//!
//! ## Why this stage has to exist
//!
//! Training a model on raw text teaches it to CONTINUE that text. Asking it a
//! question afterwards is a different task in a different format, and a model
//! that learned the document perfectly has been given no reason to answer a
//! question about it. Measured on this repository's own reader sample: a
//! corpus of manual pages, trained as next-token continuation, moved a
//! question-answering battery by nothing at all - which is the expected
//! result of that design and says nothing about whether the model learned.
//!
//! The missing step is the one here. The source material is turned into
//! questions and answers FIRST, and the model is fine-tuned on those, so
//! training and use are the same shape.
//!
//! ## Nothing generated is trusted
//!
//! A model writing questions about a document also writes wrong answers to
//! them, confidently and in the right format. Fine-tuning on those teaches
//! the model its own mistakes and launders them into weights nobody can
//! grep. So every candidate goes through a [`Check`] the CALLER supplies,
//! and this module has no opinion about what a right answer looks like - it
//! cannot have one, because that is the caller's domain.
//!
//! A checker that can EXECUTE the answer is worth far more than one that
//! compares it to a string. Measured on the reader sample's invented
//! command-line tool, where the checker runs the command:
//!
//! ```text
//! hask0 graft --cutoff                 correct
//! hask0 graft --cutoff --depth 7d      valid but different   <- a string check passes this
//! hask0 graft --cutoff --depth banana  BadValue { wants: "TIME" }
//! haskell0 --splice budget-store       NotThisTool
//! ```
//!
//! ## Every stage writes what it produced
//!
//! Each step takes typed input, returns typed output, and can be written to
//! and read back from a run directory through [`crate::artifact`]. A stage
//! that only ever hands its result to the next one can be checked by running
//! the whole pipeline again with a print statement in it; one that writes it
//! down can be read.
//!
//! Swedish Embedded AB builds training-data pipelines whose every
//! intermediate is inspectable and whose accepted data is verified rather
//! than assumed. If your team needs a model fine-tuned on your own material
//! with an audit trail from source line to training row, you can procure our
//! services by sending an email to info@swedishembedded.com.

use serde::{Deserialize, Serialize};

use crate::text::{TextGenerationOptions, TextGenerationPipeline};
use crate::Result;

/// A unit of source material small enough to ask questions about.
///
/// How source becomes passages is the caller's: a manual page splits by
/// command, a specification by clause, a transcript by turn. The SDK does not
/// guess, because guessing wrong produces passages that no question can be
/// answered from and nothing downstream can tell that apart from a weak
/// model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Passage {
    /// Stable identity, so a later stage can point back at what a training
    /// row came from.
    pub id: String,
    /// Where it came from, for a person reading the artefact.
    pub source: String,
    pub text: String,
}

/// One question and answer a model proposed about a passage, before anything
/// has checked it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// [`Passage::id`] this was drawn from.
    pub passage: String,
    pub question: String,
    pub answer: String,
    /// Everything the model wrote, kept whole. A candidate that parsed
    /// wrongly and a model that answered wrongly look identical once the
    /// text has been thrown away.
    pub raw: String,
}

/// What a caller's checker decided about one candidate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// The answer is right, and here is the evidence - what running it
    /// produced, what it matched. Recorded so an accepted row can be
    /// audited as easily as a refused one.
    Accept { evidence: String },
    /// Why not. Read by a person deciding whether the generator or the
    /// checker is at fault.
    Reject { reason: String },
}

impl Verdict {
    pub fn accepted(&self) -> bool {
        matches!(self, Verdict::Accept { .. })
    }
}

/// A candidate and what the checker said about it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checked {
    pub candidate: Candidate,
    pub verdict: Verdict,
}

/// The caller's domain oracle: does this answer actually answer the question?
///
/// The SDK cannot implement this. Whether `hask0 graft --cutoff` is right
/// depends on what `hask0` is, and the strongest possible implementation runs
/// the answer and looks at what happened.
pub trait Check {
    fn check(&self, candidate: &Candidate) -> Verdict;
}

impl<F: Fn(&Candidate) -> Verdict> Check for F {
    fn check(&self, candidate: &Candidate) -> Verdict {
        self(candidate)
    }
}

/// What a filtering pass did, in numbers a report can carry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Yield {
    pub offered: usize,
    pub accepted: usize,
}

impl Yield {
    /// Share of candidates the checker accepted. Worth watching: a rate near
    /// 1.0 usually means the checker is not checking, and a rate near 0.0
    /// that the generator is being asked for something it cannot produce.
    pub fn rate(&self) -> f64 {
        if self.offered == 0 {
            0.0
        } else {
            self.accepted as f64 / self.offered as f64
        }
    }
}

/// Run every candidate past `check`, keeping the verdicts.
///
/// Returns all of them, accepted and refused, because the refused ones are
/// the evidence that the checker is doing anything - a pipeline that reports
/// only what it kept cannot be told apart from one that keeps everything.
pub fn check_all(candidates: Vec<Candidate>, check: &dyn Check) -> Vec<Checked> {
    candidates
        .into_iter()
        .map(|candidate| {
            let verdict = check.check(&candidate);
            Checked { candidate, verdict }
        })
        .collect()
}

/// The accepted candidates, and how many there were of each.
pub fn accepted(checked: &[Checked]) -> (Vec<Candidate>, Yield) {
    let kept: Vec<Candidate> = checked.iter().filter(|c| c.verdict.accepted()).map(|c| c.candidate.clone()).collect();
    let tally = Yield { offered: checked.len(), accepted: kept.len() };
    (kept, tally)
}

/// The default instruction. A caller with a domain of its own should say so
/// in its own words - this one knows nothing about what the passages are.
pub const DEFAULT_INSTRUCTION: &str = "\
Read the reference material above. Write questions a user might ask about it, \
and the exact answer to each, taken only from the material. Reply with one \
question and one answer per pair, in exactly this form and nothing else:
Q: <question>
A: <answer>";

/// Generates question/answer candidates from passages, using a model.
pub struct Distil {
    model: TextGenerationPipeline,
    instruction: String,
    per_passage: usize,
    max_new: u32,
    seed: u64,
}

impl Distil {
    /// Drive `model`. Chat templating, the model's own stop token and
    /// sampling all come from the text pipeline, so a model that requires a
    /// template gets one without this stage knowing it happened.
    pub fn with(model: TextGenerationPipeline) -> Distil {
        Distil { model, instruction: DEFAULT_INSTRUCTION.to_string(), per_passage: 4, max_new: 320, seed: 0 }
    }

    /// How to ask. Replace it with something that names the domain: a
    /// generator told what the material IS writes better questions about it.
    pub fn instruction(mut self, text: impl Into<String>) -> Self {
        self.instruction = text.into();
        self
    }

    /// How many pairs to ask for per passage. An upper bound, not a promise:
    /// the model writes what it writes and the parser keeps what it can read.
    pub fn per_passage(mut self, n: usize) -> Self {
        self.per_passage = n.max(1);
        self
    }

    /// Token budget for one passage's worth of pairs.
    pub fn max_new_tokens(mut self, n: u32) -> Self {
        self.max_new = n.max(1);
        self
    }

    /// Seeds generation, so the same passages give the same candidates.
    /// Everything downstream is reproducible from a seed and this stage must
    /// not be the one that is not.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The prompt a passage is turned into. Exposed so a caller can see
    /// exactly what the model was asked, which is the first thing to look at
    /// when the candidates are bad.
    pub fn prompt_for(&self, passage: &Passage) -> String {
        format!("{}\n\n---\n{}\n---\n\n{}\n\nWrite up to {} pairs.", "Reference material:", passage.text, self.instruction, self.per_passage)
    }

    /// Ask the model about every passage and parse what it wrote.
    ///
    /// A passage the model answers unusably yields no candidates rather than
    /// an error: one bad passage in a corpus is not a reason to fail the run,
    /// and the count of passages that produced nothing is itself a signal
    /// worth reading off the artefact.
    pub fn generate(&self, passages: &[Passage]) -> Result<Vec<Candidate>> {
        let mut out = Vec::new();
        for (i, p) in passages.iter().enumerate() {
            let opts = TextGenerationOptions::new()
                .max_new_tokens(self.max_new)
                .temperature(0.0)
                .seed(self.seed.wrapping_add(i as u64));
            let raw = self.model.generate_with(&self.prompt_for(p), opts)?.text;
            out.extend(parse_pairs(&p.id, &raw));
        }
        Ok(out)
    }
}

/// Read `Q:`/`A:` pairs out of a model's reply.
///
/// Tolerant about what surrounds them and strict about the pairing: a `Q:`
/// with no `A:` after it is dropped, because a question with no answer is not
/// a training row and guessing one would be inventing data.
///
/// Separate from [`Distil`] so a caller whose model replies in another shape
/// can parse `raw` itself and build [`Candidate`]s directly.
pub fn parse_pairs(passage_id: &str, raw: &str) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut question: Option<String> = None;
    for line in raw.lines() {
        let line = line.trim();
        if let Some(q) = strip_marker(line, "Q:") {
            question = (!q.is_empty()).then(|| q.to_string());
        } else if let Some(a) = strip_marker(line, "A:") {
            if let (Some(q), false) = (question.take(), a.is_empty()) {
                out.push(Candidate {
                    passage: passage_id.to_string(),
                    question: q,
                    answer: a.to_string(),
                    raw: raw.to_string(),
                });
            }
        }
    }
    out
}

/// `Q:` at the start of a line, however the model bulleted or emphasised it.
fn strip_marker<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    let stripped = line.trim_start_matches(['-', '*', '#', '>', ' ', '`']).trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')');
    let rest = stripped.trim_start().strip_prefix(marker)?;
    // The emphasis a model wraps its marker in (`**Q:**`) lands on this side
    // of the colon. Trimmed from the ends only, so an answer that genuinely
    // contains one keeps it.
    Some(rest.trim().trim_matches(|c| c == '*' || c == '_' || c == '`').trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(q: &str, a: &str) -> Candidate {
        Candidate { passage: "p1".into(), question: q.into(), answer: a.into(), raw: String::new() }
    }

    /// The model writes prose around its pairs, bullets them, fences them.
    /// What matters is that the pairs come out and nothing is invented.
    #[test]
    fn pairs_are_read_out_of_whatever_the_model_wrapped_them_in() {
        let raw = "Here are some pairs:\n\n\
                   1. Q: How do I graft?\n   A: `hask0 graft --cutoff`\n\n\
                   - **Q:** What sets the depth?\n- A: hask0 graft --cutoff --depth 7d\n";
        let got = parse_pairs("p1", raw);
        assert_eq!(got.len(), 2, "{got:#?}");
        assert_eq!(got[0].question, "How do I graft?");
        assert_eq!(got[0].answer, "hask0 graft --cutoff");
        assert_eq!(got[1].question, "What sets the depth?");
        assert_eq!(got[0].passage, "p1");
        assert!(got[0].raw.contains("Here are some pairs"), "the whole reply is kept");
    }

    /// A question with no answer is dropped. Inventing the missing half is
    /// how a pipeline manufactures its own training data.
    #[test]
    fn a_question_with_no_answer_is_dropped_rather_than_completed() {
        assert!(parse_pairs("p1", "Q: What sets the depth?\nQ: And the weight?\n").is_empty());
        assert!(parse_pairs("p1", "A: hask0 graft --cutoff\n").is_empty());
        let paired = parse_pairs("p1", "Q: one\nQ: two\nA: answer to two\n");
        assert_eq!(paired.len(), 1, "the answer belongs to the question it followed");
        assert_eq!(paired[0].question, "two");
    }

    /// The checker decides, the SDK does not - and BOTH outcomes are kept,
    /// because a pipeline that reports only what it accepted cannot be told
    /// apart from one that accepts everything.
    #[test]
    fn checking_keeps_the_refusals_and_counts_the_yield() {
        let cands = vec![candidate("q1", "good"), candidate("q2", "bad"), candidate("q3", "good")];
        let check = |c: &Candidate| {
            if c.answer == "good" {
                Verdict::Accept { evidence: "ran clean".into() }
            } else {
                Verdict::Reject { reason: "did not run".into() }
            }
        };
        let checked = check_all(cands, &check);
        assert_eq!(checked.len(), 3, "every candidate is accounted for");
        let (kept, tally) = accepted(&checked);
        assert_eq!(kept.len(), 2);
        assert_eq!(tally, Yield { offered: 3, accepted: 2 });
        assert!((tally.rate() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(
            checked[1].verdict,
            Verdict::Reject { reason: "did not run".into() },
            "and the refusal says why, for whoever has to decide if the generator or the checker is wrong"
        );
    }

    /// An empty pass reports a rate rather than dividing by zero.
    #[test]
    fn an_empty_pass_has_a_rate_of_zero() {
        assert_eq!(Yield::default().rate(), 0.0);
    }
}
