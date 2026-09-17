// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SaaS sales conversations: the sequential-decision dataset.
//!
//! 100,000 synthetic business-to-business sales dialogues, each with a binary
//! conversion outcome and a conversion probability recorded **at every turn**.
//! Published by DeepMost Innovations (Apache-2.0) alongside *SalesRLAgent: A
//! Reinforcement Learning Approach for Real-Time Sales Conversion Prediction
//! and Optimization* (Nandakishor M, 2025), whose method this feeds.
//!
//! `samples/decision/salesagent/fetch-dataset.sh` puts it in place, projecting
//! the eight columns below out of the published 3088 - the other 3072 are
//! Azure OpenAI embeddings, and the point of running this here is that the
//! encoder is local.
//!
//! **What the labels are, and are not.** Both the conversation and its
//! trajectory were generated together by GPT-4o. So the trajectory is a
//! plausible, self-consistent narration of the dialogue it accompanies - not a
//! measurement of anything that happened to a real buyer. A model that matches
//! it has learned to read sales conversations the way the generator wrote
//! them. That is a real and checkable task, and it is not evidence about
//! real-world conversion; the two should never be quoted as if they were the
//! same number.
//!
//! Swedish Embedded AB builds conversation-scoring models that run on the
//! customer's own hardware. If your team needs judgment over a live dialogue
//! without shipping the dialogue to a third party, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::Path;

use data::rng::Rng;

/// One utterance.
#[derive(Clone, Debug)]
pub struct Message {
    /// `customer`, `sales_rep`, or occasionally `system`.
    pub speaker: String,
    pub text: String,
}

impl Message {
    /// How a turn is written into the state the encoder reads.
    ///
    /// The speaker is part of the text rather than a separate field because
    /// the encoder has one token stream: who said a thing is most of what it
    /// means in a sales dialogue, and a bare concatenation loses it.
    pub fn render(&self) -> String {
        let who = match self.speaker.as_str() {
            "customer" => "customer",
            "sales_rep" => "rep",
            other => other,
        };
        format!("{who}: {}", self.text)
    }
}

/// One conversation, with its outcome and its per-turn labels.
#[derive(Clone, Debug)]
pub struct Conversation {
    pub turns: Vec<Message>,
    /// Did it convert.
    pub outcome: bool,
    /// The generator's conversion probability after each turn. Same length as
    /// `turns` - the fetcher drops rows where it is not.
    pub trajectory: Vec<f32>,
    pub engagement: f32,
    pub effectiveness: f32,
    pub industry: String,
}

impl Conversation {
    /// The state the model reads at turn `t`: every turn up to and including
    /// it, and nothing after.
    ///
    /// The truncation is the whole point. A model shown the closing turn can
    /// read the outcome off it, so a trajectory built from full transcripts
    /// would score well and predict nothing.
    pub fn prefix(&self, t: usize) -> String {
        self.turns[..=t.min(self.turns.len() - 1)]
            .iter()
            .map(Message::render)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Draw a turn to train on, weighted toward the END of the conversation.
    ///
    /// Turn `k` steps from the close is drawn with probability proportional to
    /// `gamma^k`; `gamma = 1.0` is uniform.
    ///
    /// Uniform sampling is the obvious choice and it trains the wrong model.
    /// A conversation's middle is mostly ambiguous - over this dataset the
    /// mean label across all turns is 0.43, against 0.55 over the last two -
    /// so a uniform sampler spends almost every step on states whose best
    /// answer really is "about the base rate". The model duly learns to emit
    /// the base rate and stops there, which is exactly what a first run did:
    /// every turn of every conversation pinned to 0.50, accuracy 0.502 against
    /// a bag-of-words baseline of 0.72.
    ///
    /// The weighting is the same `gamma` the policy phase discounts by, for
    /// the same reason: a probability committed near the close is the one
    /// being asked for.
    pub fn sample_turn(&self, gamma: f32, rng: &mut Rng) -> usize {
        let n = self.turns.len();
        if n <= 1 || gamma >= 1.0 {
            return (rng.next_u64() % n.max(1) as u64) as usize;
        }
        let w: Vec<f32> = (0..n).map(|t| gamma.powi((n - 1 - t) as i32)).collect();
        let total: f32 = w.iter().sum();
        let mut pick = rng.next_f32() * total;
        for (t, &wi) in w.iter().enumerate() {
            pick -= wi;
            if pick <= 0.0 {
                return t;
            }
        }
        n - 1
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

pub struct SalesConversations {
    pub train: Vec<Conversation>,
    pub test: Vec<Conversation>,
}

impl SalesConversations {
    /// Read `train.jsonl` and `test.jsonl` from `dir`.
    pub fn load(dir: &Path) -> Result<SalesConversations, String> {
        Ok(SalesConversations {
            train: read_jsonl(&dir.join("train.jsonl"))?,
            test: read_jsonl(&dir.join("test.jsonl"))?,
        })
    }

    /// Fraction of training conversations that converted - the number any
    /// accuracy here has to be read against.
    pub fn base_rate(&self) -> f32 {
        if self.train.is_empty() {
            return 0.0;
        }
        self.train.iter().filter(|c| c.outcome).count() as f32 / self.train.len() as f32
    }
}

fn read_jsonl(path: &Path) -> Result<Vec<Conversation>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "{}: {e}\n  run samples/decision/salesagent/fetch-dataset.sh to put the dataset in place",
            path.display()
        )
    })?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        let turns: Vec<Message> = v["turns"]
            .as_array()
            .ok_or_else(|| format!("{}:{}: no turns array", path.display(), i + 1))?
            .iter()
            .map(|m| Message {
                speaker: m["speaker"].as_str().unwrap_or("system").to_string(),
                text: m["message"].as_str().unwrap_or("").to_string(),
            })
            .collect();
        let trajectory: Vec<f32> =
            v["trajectory"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect())
                .unwrap_or_default();
        // The fetcher already enforces this; re-checked because a
        // hand-assembled or truncated file is the likelier cause of a
        // misalignment than the fetcher is, and the failure it produces
        // otherwise is a silently off-by-one label.
        if trajectory.len() != turns.len() {
            return Err(format!(
                "{}:{}: {} turns but {} trajectory points",
                path.display(),
                i + 1,
                turns.len(),
                trajectory.len()
            ));
        }
        out.push(Conversation {
            turns,
            outcome: v["outcome"].as_i64().unwrap_or(0) != 0,
            trajectory,
            engagement: v["engagement"].as_f64().unwrap_or(0.0) as f32,
            effectiveness: v["effectiveness"].as_f64().unwrap_or(0.0) as f32,
            industry: v["industry"].as_str().unwrap_or("").to_string(),
        });
    }
    Ok(out)
}

/// Draws conversations in a **curriculum**, shortest first, and keeps each
/// draw **outcome-balanced** - two of the six training measures the paper
/// lists, and the two that change what the model sees rather than how it is
/// updated.
///
/// Why balanced: a policy rewarded for matching the outcome on a skewed set
/// can collect most of its reward by learning the base rate, and the gradient
/// that teaches it to READ the conversation is the smaller part. Alternating
/// the outcome removes that shortcut entirely rather than down-weighting it.
///
/// Why shortest first: a short conversation's turns are nearly all decisive,
/// so the reward is closely tied to the text. A 30-turn dialogue spends its
/// first ten turns on small talk the outcome does not depend on, and starting
/// there teaches the base rate first. `progress` moves from 0 to 1 over the
/// run and opens the pool as it goes.
pub struct Curriculum {
    by_outcome: [Vec<usize>; 2],
    next_outcome: usize,
}

impl Curriculum {
    /// Orders each outcome's conversations by length.
    pub fn new(convs: &[Conversation]) -> Curriculum {
        let mut by_outcome: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for (i, c) in convs.iter().enumerate() {
            by_outcome[usize::from(c.outcome)].push(i);
        }
        for v in &mut by_outcome {
            v.sort_by_key(|&i| convs[i].len());
        }
        Curriculum { by_outcome, next_outcome: 0 }
    }

    /// Whether either outcome has any conversations at all.
    pub fn is_empty(&self) -> bool {
        self.by_outcome.iter().all(|v| v.is_empty())
    }

    /// One conversation index: the outcome that is this draw's turn, drawn
    /// from the shortest `progress` share of that outcome's pool.
    ///
    /// `progress` is clamped into `[0.1, 1.0]`, so the first draws still have
    /// a tenth of the set to choose from rather than returning the single
    /// shortest conversation over and over.
    pub fn draw(&mut self, progress: f32, rng: &mut Rng) -> usize {
        let want = self.next_outcome;
        self.next_outcome ^= 1;
        // Fall back to the other outcome rather than failing: a caller may
        // legitimately hand over a single-outcome set.
        let pool = if self.by_outcome[want].is_empty() { &self.by_outcome[want ^ 1] } else { &self.by_outcome[want] };
        let open = ((pool.len() as f32 * progress.clamp(0.1, 1.0)).ceil() as usize).clamp(1, pool.len());
        pool[(rng.next_u64() % open as u64) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(outcome: bool, n: usize) -> Conversation {
        Conversation {
            turns: (0..n)
                .map(|i| Message {
                    speaker: if i % 2 == 0 { "customer".into() } else { "sales_rep".into() },
                    text: format!("turn {i}"),
                })
                .collect(),
            outcome,
            trajectory: (0..n).map(|i| i as f32 / n as f32).collect(),
            engagement: 0.5,
            effectiveness: 0.5,
            industry: "test".into(),
        }
    }

    /// A prefix must stop where it is asked to. Leaking one later turn is the
    /// difference between predicting an outcome and reading it.
    #[test]
    fn a_prefix_holds_only_the_turns_up_to_it() {
        let c = conv(true, 5);
        assert_eq!(c.prefix(0), "customer: turn 0");
        assert_eq!(c.prefix(1), "customer: turn 0\nrep: turn 1");
        assert!(!c.prefix(2).contains("turn 3"), "a later turn leaked into the prefix");
        // Out of range clamps rather than panicking: callers iterate turns and
        // the trajectory together, and a truncated row must not crash a run.
        assert_eq!(c.prefix(99), c.prefix(4));
    }

    /// The speaker has to survive into the text, or the state cannot tell who
    /// raised an objection from who answered it.
    #[test]
    fn the_speaker_is_part_of_the_rendered_turn() {
        let c = conv(true, 2);
        assert!(c.prefix(1).starts_with("customer:"));
        assert!(c.prefix(1).contains("rep:"));
    }

    /// Successive draws alternate outcomes, whatever the pool's own skew.
    #[test]
    fn the_curriculum_alternates_outcomes() {
        let convs: Vec<Conversation> =
            (0..40).map(|i| conv(i < 35, 4 + i % 7)).collect(); // 35 positive, 5 negative
        let mut cur = Curriculum::new(&convs);
        let mut rng = Rng::new(1);
        let drawn: Vec<bool> = (0..20).map(|_| convs[cur.draw(1.0, &mut rng)].outcome).collect();
        let pos = drawn.iter().filter(|&&o| o).count();
        assert_eq!(pos, 10, "expected an even split, got {pos}/20 positive from a 7:1 pool");
    }

    /// Early draws come from the SHORT end, and the pool really does open up.
    #[test]
    fn the_curriculum_starts_short_and_widens() {
        let convs: Vec<Conversation> = (0..40).map(|i| conv(i % 2 == 0, 2 + i)).collect();
        let mut cur = Curriculum::new(&convs);
        let mut rng = Rng::new(7);
        let early: usize =
            (0..40).map(|_| convs[cur.draw(0.1, &mut rng)].len()).max().unwrap();
        let late: usize = (0..40).map(|_| convs[cur.draw(1.0, &mut rng)].len()).max().unwrap();
        assert!(early < late, "the curriculum never widened: early max {early}, late max {late}");
    }

    /// The turn sampler has to stay in range, and `gamma = 1` has to be
    /// genuinely uniform - a caller turning the bias off should get the plain
    /// behaviour back exactly.
    #[test]
    fn uniform_turn_sampling_is_uniform_and_in_range() {
        let c = conv(true, 8);
        let mut rng = Rng::new(2);
        let mut seen = [0usize; 8];
        for _ in 0..8000 {
            let t = c.sample_turn(1.0, &mut rng);
            assert!(t < 8, "turn {t} is outside an 8-turn conversation");
            seen[t] += 1;
        }
        for (t, &n) in seen.iter().enumerate() {
            assert!((n as f32 / 8000.0 - 0.125).abs() < 0.02, "turn {t} drawn {n}/8000");
        }
    }

    /// ...and a gamma below 1 has to actually move the mass to the end, or the
    /// fix that motivated it does nothing.
    #[test]
    fn biased_turn_sampling_favours_the_close() {
        let c = conv(true, 12);
        let mut rng = Rng::new(3);
        let (mut early, mut late) = (0usize, 0usize);
        for _ in 0..8000 {
            let t = c.sample_turn(0.85, &mut rng);
            assert!(t < 12);
            if t < 6 { early += 1 } else { late += 1 }
        }
        assert!(late > early * 2, "late {late} vs early {early}: the bias is not biting");
        // Every turn must still be reachable - the early conversation is not
        // to be abandoned, only de-emphasized.
        let mut seen = vec![false; 12];
        for _ in 0..20000 {
            seen[c.sample_turn(0.85, &mut rng)] = true;
        }
        assert!(seen.iter().all(|&s| s), "some turn became unreachable: {seen:?}");
    }

    /// A single-outcome set must still be drawable - a caller may hand one
    /// over, and refusing would be a crash in the middle of a run.
    #[test]
    fn a_single_outcome_pool_still_draws() {
        let convs: Vec<Conversation> = (0..4).map(|i| conv(true, 3 + i)).collect();
        let mut cur = Curriculum::new(&convs);
        let mut rng = Rng::new(3);
        for _ in 0..8 {
            assert!(cur.draw(1.0, &mut rng) < convs.len());
        }
    }
}
