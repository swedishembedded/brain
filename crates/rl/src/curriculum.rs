// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The **position-copy** task family: a verifiable-reward `Environment` whose
//! rules are genuinely different from one another and yet of *identical
//! structural difficulty by construction* - the property a continual-learning
//! study needs before any of its numbers mean anything.
//!
//! ## What a rule is
//!
//! A prompt is `[cue, c0, c1, c2, c3, c4, SEP]`: one cue token followed by
//! five content tokens. A rule is a `(cue, picks)` pair where `picks` is one
//! of the 60 ORDERED 3-of-5 selections; the completion is the three content
//! tokens at those positions, in that order. Every rule is three positional
//! copies - only *which* positions differ - so no rule is harder than
//! another, and a "fresh task" plasticity probe is just another rule rather
//! than a differently-shaped problem whose score would not be comparable.
//!
//! ## Why the split is a property of the CONTENT, not of the seed
//!
//! [`ContentSplit`] partitions the 16^5 content tuples by a hash of the
//! tuple itself: `hash % 4 == 0` is the evaluation space, everything else is
//! the exploration space. A seed-range split (draw explore tasks from seeds
//! 0..N, held-out tasks from seeds 10_000..) only holds as long as no two
//! seeds land on the same content - and at the scale a 12-cycle study draws
//! (~1400 explore tuples against ~192 evaluation tuples) at least one
//! collision is *likely*, not negligible. A structural partition makes the
//! leak impossible instead of improbable: an explore tuple can never also be
//! an evaluation tuple, so a held-out score cannot be a memorized training
//! instance even when two seeds collide. [`Task::id`] is derived from the
//! content and the cue, never from the seed, so
//! [`crate::improve::explore_anchor_split`]'s id-hash disjointness assertion
//! is a real second check on top of that rather than a tautology.
//!
//! ## What this family is NOT
//!
//! Synthetic and difficulty-invariant *by construction*: that control was
//! bought by removing exactly the properties that break real systems -
//! distribution shift, ambiguity, label noise, adversarial content. A result
//! measured here is a result about a controlled task family, not about live
//! data.
//!
//! Swedish Embedded AB builds verifiable-reward task families whose controls
//! are structural rather than assumed - difficulty-invariant rule sets,
//! content-space train/eval partitions, and rewards a stored run artifact can
//! be re-scored against byte-for-byte. If your team needs expertise in
//! designing evaluation environments that cannot silently leak their own
//! answers, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use data::rng::Rng;

use crate::env::{Environment, Reward, Step, Task, Verifier};

/// Padding filler (never predicted from a real prompt).
pub const PAD: u32 = 0;
/// Prompt terminator: the position the first completion token is decoded from.
pub const SEP: u32 = 1;
/// First cue token. Study cue ids are `CUE_BASE .. CUE_BASE + MAX_CUES`.
pub const CUE_BASE: u32 = 2;
/// How many distinct study cues exist (cue tokens `2..=15`).
pub const MAX_CUES: usize = 14;
/// First content token. Content ids are `CONTENT_BASE .. CONTENT_BASE + CONTENT_N`.
pub const CONTENT_BASE: u32 = 16;
/// Size of the content alphabet.
pub const CONTENT_N: u32 = 16;
/// Total vocabulary: `PAD`, `SEP`, 14 cues, 16 content tokens.
pub const VOCAB: u32 = 32;
/// Content slots in the prompt.
pub const SLOTS: usize = 5;
/// Completion length (how many positions a rule copies).
pub const OUT_LEN: usize = 3;
/// `[cue, c0..c4, SEP]`.
pub const PROMPT_LEN: usize = 1 + SLOTS + 1;
/// Packed row / model `block_size`. `PROMPT_LEN + OUT_LEN = 10` fits with room.
pub const SEQ_LEN: usize = 12;

/// First PRETRAINING cue token. Deliberately past the study cue range, so a
/// pretrained base has never seen a study rule's cue: these ids overlap the
/// CONTENT id range numerically, which is harmless because slot 0 of a prompt
/// is always the cue and slots 1..=5 are always content - position, not id,
/// says which is which. The property that matters is that
/// `PRETRAIN_CUE_BASE .. VOCAB` is disjoint from `CUE_BASE .. CUE_BASE +
/// MAX_CUES`, and it is.
pub const PRETRAIN_CUE_BASE: u32 = CUE_BASE + MAX_CUES as u32;
/// How many distinct pretraining cues exist (`16..=31`).
pub const MAX_PRETRAIN_CUES: usize = (VOCAB - PRETRAIN_CUE_BASE) as usize;

/// Which content split a drawn tuple belongs to. The split is a property of
/// the CONTENT, not of the seed: an explore tuple can never also be an eval
/// tuple, so a held-out score cannot be a memorized training instance even
/// when two seeds collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentSplit {
    Explore,
    Eval,
}

/// One rule: a cue token plus the ordered positions its completion copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rule {
    pub cue: u32,
    pub picks: [usize; OUT_LEN],
}

/// Stride used to spread the study rules across [`Rule::all_picks`]'s
/// enumeration. 13 is coprime with 60, so `(k * STUDY_PICK_STRIDE) % 60` is a
/// bijection on `0..60` - and, unlike taking `all_picks()[k]` directly (whose
/// first 14 entries ALL begin `picks[0] == 0`, i.e. every study cycle would
/// share its first output position), it gives 12 rules whose first copied
/// position cycles through all five slots. Rules that share structure are
/// rules that interfere less; a study that quietly chose the low-interference
/// rules would bias its own retention result upward, which is exactly the
/// kind of self-flattering control this file exists to avoid.
const STUDY_PICK_STRIDE: usize = 13;

impl Rule {
    /// The 60 ordered 3-of-5 selections, in one fixed enumeration order
    /// (lexicographic on `(picks[0], picks[1], picks[2])` after removing
    /// already-used positions).
    pub fn all_picks() -> Vec<[usize; OUT_LEN]> {
        let mut out = Vec::with_capacity(60);
        for a in 0..SLOTS {
            for b in 0..SLOTS {
                if b == a {
                    continue;
                }
                for c in 0..SLOTS {
                    if c == a || c == b {
                        continue;
                    }
                    out.push([a, b, c]);
                }
            }
        }
        out
    }

    /// Rule for study cycle `k` (0-based): `cue = CUE_BASE + k`, picks drawn
    /// from [`Rule::all_picks`] at the spread stride (see
    /// [`STUDY_PICK_STRIDE`]). Panics for `k >= MAX_CUES`.
    pub fn for_cycle(k: usize) -> Rule {
        assert!(k < MAX_CUES, "curriculum::Rule::for_cycle: cycle {k} exceeds the {MAX_CUES} distinct cue tokens this vocabulary has");
        let picks = Rule::all_picks();
        Rule { cue: CUE_BASE + k as u32, picks: picks[(k * STUDY_PICK_STRIDE) % picks.len()] }
    }

    /// Rules reserved for PRETRAINING only - cues that no study cycle ever
    /// uses (`PRETRAIN_CUE_BASE..`), and pick-triples disjoint from every
    /// study rule's, so the pretrained base learns the format and the copy
    /// skill without ever seeing a study rule (neither its cue nor its
    /// positional selection). Panics for `n > MAX_PRETRAIN_CUES`.
    pub fn pretrain_rules(n: usize) -> Vec<Rule> {
        assert!(n <= MAX_PRETRAIN_CUES, "curriculum::Rule::pretrain_rules: only {MAX_PRETRAIN_CUES} pretraining cues exist, asked for {n}");
        let all = Rule::all_picks();
        let study: std::collections::HashSet<usize> = (0..MAX_CUES).map(|k| (k * STUDY_PICK_STRIDE) % all.len()).collect();
        let free: Vec<[usize; OUT_LEN]> = (0..all.len()).filter(|i| !study.contains(i)).map(|i| all[i]).collect();
        (0..n).map(|i| Rule { cue: PRETRAIN_CUE_BASE + i as u32, picks: free[(i * 3) % free.len()] }).collect()
    }
}

/// SplitMix64 finalizer over the packed content tuple - the same mixer
/// [`data::rng::Rng`] uses, applied directly to the tuple so the split is a
/// pure function of the CONTENT and nothing else (no seed, no cue, no draw
/// order).
fn content_hash(content: &[u32; SLOTS]) -> u64 {
    let mut z = content.iter().fold(0x9E37_79B9_7F4A_7C15u64, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Which split a content tuple structurally belongs to: one tuple in four is
/// evaluation space, the rest is exploration space.
pub fn split_of(content: &[u32; SLOTS]) -> ContentSplit {
    if content_hash(content).is_multiple_of(4) {
        ContentSplit::Eval
    } else {
        ContentSplit::Explore
    }
}

/// One task family instance: a fixed rule, restricted to one content split.
pub struct PositionCopyEnv {
    pub rule: Rule,
    pub split: ContentSplit,
}

impl PositionCopyEnv {
    pub fn new(rule: Rule, split: ContentSplit) -> PositionCopyEnv {
        PositionCopyEnv { rule, split }
    }

    /// Rejection-sample one content tuple in this env's split. Terminates
    /// with probability 1 (both splits are non-empty and dense: ~25 % / ~75 %
    /// of a 16^5 space), and the bounded loop below turns a hypothetical
    /// pathological hash into a loud failure rather than a hang.
    fn draw_content(&self, seed: u64) -> [u32; SLOTS] {
        let mut rng = Rng::new(seed);
        for _ in 0..10_000 {
            let mut c = [0u32; SLOTS];
            for slot in c.iter_mut() {
                *slot = CONTENT_BASE + (rng.next_u64() % CONTENT_N as u64) as u32;
            }
            if split_of(&c) == self.split {
                return c;
            }
        }
        panic!("curriculum::PositionCopyEnv: 10000 rejection draws found no {:?} content tuple - the content hash is degenerate", self.split);
    }
}

impl Environment for PositionCopyEnv {
    fn name(&self) -> &str {
        "position-copy"
    }

    /// Exactly one task, rejection-sampled into this env's content split.
    /// `Task::answer` carries the RULE (`{"picks": [i0, i1, i2]}`), never the
    /// target: the target is recomputed by reading the content back out of
    /// `Task::prompt` (see [`target_of`]), so nothing that looks like a label
    /// exists anywhere in a `Task` to accidentally reach training.
    fn tasks(&self, seed: u64) -> Vec<Task> {
        let c = self.draw_content(seed);
        let mut prompt = Vec::with_capacity(PROMPT_LEN);
        prompt.push(self.rule.cue);
        prompt.extend_from_slice(&c);
        prompt.push(SEP);
        let cue = self.rule.cue;
        let id = format!(
            "poscopy-c{cue}-{:x}{:x}{:x}{:x}{:x}",
            c[0] - CONTENT_BASE,
            c[1] - CONTENT_BASE,
            c[2] - CONTENT_BASE,
            c[3] - CONTENT_BASE,
            c[4] - CONTENT_BASE
        );
        vec![Task { id, prompt, answer: serde_json::json!({ "picks": self.rule.picks }) }]
    }

    // `legal_actions` stays defaulted (`None`). It exists to thicken a
    // cold-start policy's hit rate when exact-match rejection sampling would
    // otherwise find nothing to train on - and `PositionCopyVerifier`'s dense
    // partial credit already solves that thinness directly, by giving even a
    // one-of-three-correct completion a reward that differs from its group
    // mates'. Constraining the action space on top would be a second control
    // buying nothing, while making the reported scores less comparable to an
    // unconstrained policy's.
}

/// Recompute the target from the task's own prompt + rule. Public because the
/// demos print it next to `Task::answer` to show visibly that the answer is
/// recomputed and never stored; [`PositionCopyVerifier`] uses it internally.
pub fn target_of(task: &Task) -> Vec<u32> {
    let picks: Vec<usize> = task.answer["picks"]
        .as_array()
        .unwrap_or_else(|| panic!("curriculum::target_of: task {} has no `picks` in its answer", task.id))
        .iter()
        .map(|v| v.as_u64().expect("curriculum::target_of: a pick is a slot index") as usize)
        .collect();
    assert_eq!(task.prompt.len(), PROMPT_LEN, "curriculum::target_of: task {} has a malformed prompt", task.id);
    picks.iter().map(|&p| task.prompt[1 + p]).collect()
}

/// Fraction of the `OUT_LEN` output positions matched positionally.
///
/// Dense partial credit is what gives a cold-start GRPO group reward
/// VARIANCE to learn from: an exact-match reward makes every early group
/// all-wrong, and [`crate::objective::grpo::group_advantages`] drops a
/// zero-variance group entirely, so an exact-match reward would train on
/// nothing at all until the policy stumbled onto a fully correct completion
/// by chance (probability `1/32^3`).
pub struct PositionCopyVerifier;

impl Verifier for PositionCopyVerifier {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let target = target_of(task);
        let matches = completion.iter().zip(target.iter()).filter(|(a, b)| a == b).count();
        let value = matches as f32 / OUT_LEN as f32;
        Reward { value, parts: std::collections::BTreeMap::from([("frac_match".to_string(), value)]) }
    }
}

/// Uniform-token analytic reference (`1 / VOCAB`).
///
/// NOT used as the chance baseline in any assertion: a real model is not
/// uniform over its vocabulary, so the study measures the untrained base's
/// own score on the same probe instead (`continual::StudyReport::b_base`) and
/// asserts against that. This function exists so the two numbers can be
/// printed side by side and the difference seen.
pub fn uniform_chance() -> f64 {
    1.0 / VOCAB as f64
}

/// Stream length of one pretraining record: `[cue, c0..c4, SEP, t0, t1, t2,
/// PAD]`. The trailing `PAD` is the RECORD SEPARATOR - see
/// [`write_pretrain_dataset`] on why that matters more than it looks.
pub const RECORD_LEN: usize = PROMPT_LEN + OUT_LEN + 1;

/// Write a `model::load_dataset`-shaped directory (`train.u32.bin`,
/// `val.u32.bin`, `meta.json`) of `n` flat `[cue, c0..c4, SEP, t0, t1, t2,
/// PAD]` records over `rules`, drawn from the Explore content split only.
///
/// **The caller MUST train with `FitOpts::align_to_lines = true` and
/// `block_size >= RECORD_LEN + 1`.** The `meta.json` written here maps token
/// id `PAD` to `'\n'` precisely so `data::loader`'s line-start alignment
/// treats every record boundary as a line start; without that flag the loader
/// draws windows at uniformly random offsets, and a window starting in the
/// middle of a record supervises completion tokens whose own prompt is not
/// inside the window - an impossible prediction, i.e. pure label noise. This
/// was not a theoretical concern: an unaligned first attempt at this dataset
/// trained to a 0.15 held-out score on rules it had been trained on (barely
/// above the 0.03 uniform-token reference) and the fixture sanity check
/// caught it.
///
/// Generic: emits token ids and knows nothing about any model.
pub fn write_pretrain_dataset(rules: &[Rule], n: usize, seed: u64, out_dir: &Path) -> std::io::Result<()> {
    assert!(!rules.is_empty(), "curriculum::write_pretrain_dataset: needs at least one rule");
    std::fs::create_dir_all(out_dir)?;
    let mut rng = Rng::new(seed);
    let emit = |count: usize, rng: &mut Rng| -> Vec<u32> {
        let mut out = Vec::with_capacity(count * RECORD_LEN);
        for _ in 0..count {
            let rule = rules[(rng.next_u64() % rules.len() as u64) as usize];
            let env = PositionCopyEnv::new(rule, ContentSplit::Explore);
            let task = env.tasks(rng.next_u64()).into_iter().next().expect("PositionCopyEnv yields one task");
            out.extend_from_slice(&task.prompt);
            out.extend_from_slice(&target_of(&task));
            out.push(PAD);
        }
        out
    };
    let train = emit(n, &mut rng);
    let val = emit((n / 10).max(SEQ_LEN), &mut rng);
    data::binio::write_u32_bin(&out_dir.join("train.u32.bin"), &train)?;
    data::binio::write_u32_bin(&out_dir.join("val.u32.bin"), &val)?;
    std::fs::write(out_dir.join("meta.json"), line_aligned_meta())?;
    Ok(())
}

/// The [`model::FitOpts::mask_before`] char for a
/// [`write_pretrain_dataset`] directory: [`SEP`]'s own `itos` char, since
/// [`line_aligned_meta`] maps token id `i` to `'a' + i` (and `PAD -> '\n'`).
/// Masking up to and including it supervises `[t0, t1, t2]` - the completion -
/// and nothing else.
///
/// Exported so no caller re-derives `'b'` by hand. The difference this makes
/// is measured, not stylistic: on this exact dataset, the same budget reaches
/// 0.509 with the loss spread over the whole record and 0.910 with it masked
/// to the completion, because most of an unmasked record's gradient is spent
/// trivially re-predicting a prompt that is already in the window.
pub fn mask_before_char() -> char {
    char::from_u32('a' as u32 + SEP).expect("ascii")
}

/// `meta.json` whose `itos` maps `PAD -> '\n'`, which is how
/// `model::load_dataset` learns which token id delimits a record. Every other
/// id gets a distinct placeholder char (the loader only ever looks up `'\n'`,
/// but a table with duplicates would make `Meta::stoi` ambiguous).
fn line_aligned_meta() -> String {
    let itos: Vec<char> = (0..VOCAB).map(|i| if i == PAD { '\n' } else { char::from_u32('a' as u32 + i).expect("ascii") }).collect();
    data::binio::Meta { vocab_size: VOCAB as usize, itos }.to_json()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_picks_enumerates_the_sixty_ordered_three_of_five_selections() {
        let picks = Rule::all_picks();
        assert_eq!(picks.len(), 60);
        let unique: std::collections::HashSet<[usize; OUT_LEN]> = picks.iter().copied().collect();
        assert_eq!(unique.len(), 60, "every ordered selection must appear exactly once");
        for p in &picks {
            assert!(p[0] != p[1] && p[1] != p[2] && p[0] != p[2], "a selection may not repeat a slot: {p:?}");
            assert!(p.iter().all(|&i| i < SLOTS));
        }
    }

    #[test]
    fn for_cycle_yields_twelve_distinct_cues_and_twelve_distinct_pick_triples() {
        let rules: Vec<Rule> = (0..12).map(Rule::for_cycle).collect();
        let cues: std::collections::HashSet<u32> = rules.iter().map(|r| r.cue).collect();
        assert_eq!(cues.len(), 12, "each study cycle needs its own cue token");
        let picks: std::collections::HashSet<[usize; OUT_LEN]> = rules.iter().map(|r| r.picks).collect();
        assert_eq!(picks.len(), 12, "each study cycle needs its own positional rule");
        // The spread stride's whole purpose: the study's rules must not all
        // share their first copied position (which the raw enumeration order
        // would give), or the tasks would interfere far less than they should.
        let first: std::collections::HashSet<usize> = rules.iter().map(|r| r.picks[0]).collect();
        assert_eq!(first.len(), SLOTS, "the 12 study rules must span all 5 first-copy positions, got {first:?}");
    }

    #[test]
    #[should_panic(expected = "exceeds the 14 distinct cue tokens")]
    fn for_cycle_panics_past_the_cue_budget() {
        let _ = Rule::for_cycle(MAX_CUES);
    }

    #[test]
    fn pretrain_rules_share_no_cue_and_no_pick_triple_with_any_study_rule() {
        let pre = Rule::pretrain_rules(MAX_PRETRAIN_CUES);
        assert_eq!(pre.len(), MAX_PRETRAIN_CUES);
        let study: Vec<Rule> = (0..MAX_CUES).map(Rule::for_cycle).collect();
        for p in &pre {
            assert!(p.cue >= PRETRAIN_CUE_BASE, "a pretraining cue must be outside the study cue range: {p:?}");
            for s in &study {
                assert_ne!(p.cue, s.cue, "pretraining must never see a study cue");
                assert_ne!(p.picks, s.picks, "pretraining must never see a study rule's positional selection");
            }
        }
    }

    /// The claim [`ContentSplit`] rests on: the two splits partition the same
    /// tuple space (never the same tuple twice) AND are the same
    /// DISTRIBUTION over that space (so an eval score is not a distribution-
    /// shift measurement wearing a held-out label).
    #[test]
    fn explore_and_eval_splits_are_a_partition_not_a_distribution_shift() {
        let explore_env = PositionCopyEnv::new(Rule::for_cycle(0), ContentSplit::Explore);
        let eval_env = PositionCopyEnv::new(Rule::for_cycle(0), ContentSplit::Eval);
        let mut explore = std::collections::HashSet::new();
        let mut eval = std::collections::HashSet::new();
        let mut explore_marg = [[0usize; CONTENT_N as usize]; SLOTS];
        let mut eval_marg = [[0usize; CONTENT_N as usize]; SLOTS];
        const N: u64 = 20_000;
        for s in 0..N {
            let e = explore_env.tasks(s).into_iter().next().unwrap();
            let v = eval_env.tasks(s + 1_000_000).into_iter().next().unwrap();
            for slot in 0..SLOTS {
                explore_marg[slot][(e.prompt[1 + slot] - CONTENT_BASE) as usize] += 1;
                eval_marg[slot][(v.prompt[1 + slot] - CONTENT_BASE) as usize] += 1;
            }
            explore.insert(e.prompt[1..=SLOTS].to_vec());
            eval.insert(v.prompt[1..=SLOTS].to_vec());
        }
        let overlap: Vec<_> = explore.intersection(&eval).collect();
        assert!(overlap.is_empty(), "the splits must never yield the same tuple: {} collisions", overlap.len());
        for slot in 0..SLOTS {
            for tok in 0..CONTENT_N as usize {
                let pe = explore_marg[slot][tok] as f64 / N as f64;
                let pv = eval_marg[slot][tok] as f64 / N as f64;
                assert!(
                    (pe - pv).abs() < 0.02,
                    "slot {slot} token {tok}: explore marginal {pe:.4} vs eval marginal {pv:.4} - the eval split must be a partition of the SAME distribution, not a shifted one"
                );
            }
        }
    }

    /// The "answer is not a label" invariant, checked rather than claimed.
    #[test]
    fn task_answer_carries_only_the_rule_never_the_content_or_the_target() {
        let env = PositionCopyEnv::new(Rule::for_cycle(3), ContentSplit::Explore);
        let task = env.tasks(42).into_iter().next().unwrap();
        let obj = task.answer.as_object().expect("answer is an object");
        assert_eq!(obj.keys().collect::<Vec<_>>(), vec!["picks"], "the answer must carry the rule and nothing else");
        // Structural: no content token and no target token may appear
        // ANYWHERE in the serialized answer. `picks` are slot indices 0..5,
        // and content/target token ids are all >= CONTENT_BASE (16), so a
        // target leaking in would show up as an out-of-range number.
        let numbers: Vec<u64> = obj["picks"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert!(numbers.iter().all(|&n| n < SLOTS as u64), "the answer contains something that is not a slot index: {numbers:?}");
        let target = target_of(&task);
        let text = serde_json::to_string(&task.answer).unwrap();
        for t in target.iter().chain(task.prompt[1..=SLOTS].iter()) {
            assert!(!text.contains(&t.to_string()), "token {t} (content or target) leaked into the answer JSON {text}");
        }
    }

    #[test]
    fn target_of_recomputes_the_completion_from_the_prompt_alone() {
        let env = PositionCopyEnv::new(Rule { cue: CUE_BASE, picks: [3, 0, 4] }, ContentSplit::Explore);
        let task = env.tasks(7).into_iter().next().unwrap();
        let content = &task.prompt[1..=SLOTS];
        assert_eq!(target_of(&task), vec![content[3], content[0], content[4]]);
        assert_eq!(task.prompt[0], CUE_BASE);
        assert_eq!(task.prompt[PROMPT_LEN - 1], SEP);
    }

    #[test]
    fn verifier_gives_dense_partial_credit_in_thirds() {
        let env = PositionCopyEnv::new(Rule { cue: CUE_BASE, picks: [0, 1, 2] }, ContentSplit::Explore);
        let task = env.tasks(11).into_iter().next().unwrap();
        let target = target_of(&task);
        let v = PositionCopyVerifier;
        assert_eq!(v.verify(&task, &[], &target).value, 1.0);
        assert_eq!(v.verify(&task, &[], &[target[0], target[1], PAD]).value, 2.0 / 3.0);
        assert_eq!(v.verify(&task, &[], &[target[0], PAD, PAD]).value, 1.0 / 3.0);
        assert_eq!(v.verify(&task, &[], &[PAD, PAD, PAD]).value, 0.0);
        // A short completion is scored over the same OUT_LEN denominator -
        // decoding fewer tokens is not a way to a higher fraction.
        assert_eq!(v.verify(&task, &[], &[target[0]]).value, 1.0 / 3.0);
    }

    #[test]
    fn uniform_chance_is_one_over_vocab() {
        assert!((uniform_chance() - 1.0 / 32.0).abs() < 1e-12);
    }

    /// The masking char is derived, not guessed: it must round-trip through
    /// the dataset's OWN metadata back to [`SEP`], or the loss would be masked
    /// at the wrong position and supervise the prompt instead of - or as well
    /// as - the completion.
    #[test]
    fn mask_before_char_round_trips_through_the_datasets_own_metadata_to_sep() {
        let meta = data::binio::Meta::from_json(&line_aligned_meta()).unwrap();
        assert_eq!(meta.stoi().get(&mask_before_char()), Some(&SEP));
        assert_ne!(mask_before_char(), '\n', "masking before the record separator would supervise the whole record, not the completion");
    }

    #[test]
    fn write_pretrain_dataset_emits_separator_delimited_records_over_pretrain_rules_only() {
        let dir = std::env::temp_dir().join(format!("brain-rl-curriculum-pretrain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rules = Rule::pretrain_rules(4);
        write_pretrain_dataset(&rules, 50, 3, &dir).unwrap();
        let train = data::binio::read_tokens_u32(&dir.join("train")).unwrap();
        assert_eq!(train.len(), 50 * RECORD_LEN);
        let cues: std::collections::HashSet<u32> = rules.iter().map(|r| r.cue).collect();
        for rec in train.chunks(RECORD_LEN) {
            assert!(cues.contains(&rec[0]), "record cue {} is not a pretraining cue", rec[0]);
            assert_eq!(rec[PROMPT_LEN - 1], SEP);
            assert_eq!(rec[RECORD_LEN - 1], PAD, "every record must end with the separator the loader aligns on");
            assert!(rec[1..=SLOTS].iter().all(|&t| (CONTENT_BASE..CONTENT_BASE + CONTENT_N).contains(&t)));
            let rule = rules.iter().find(|r| r.cue == rec[0]).unwrap();
            let expect: Vec<u32> = rule.picks.iter().map(|&p| rec[1 + p]).collect();
            assert_eq!(&rec[PROMPT_LEN..PROMPT_LEN + OUT_LEN], &expect[..], "the record's completion must be its rule's positional copy");
        }
        // PAD (the separator) must resolve to '\n' in the metadata, or the
        // loader has no way to know where a record begins and every window it
        // draws is a random, mostly-misaligned slice.
        let meta = data::binio::Meta::from_json(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta.vocab_size, VOCAB as usize);
        assert_eq!(meta.stoi().get(&'\n'), Some(&PAD));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The alignment claim, checked against the real loader rather than
    /// argued: every window it draws must start at a record boundary.
    #[test]
    fn the_loader_draws_only_record_aligned_windows_from_this_dataset() {
        let dir = std::env::temp_dir().join(format!("brain-rl-curriculum-align-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rules = Rule::pretrain_rules(4);
        write_pretrain_dataset(&rules, 500, 5, &dir).unwrap();
        let opts = model::FitOpts { block_size: SEQ_LEN as u32, batch_size: 16, align_to_lines: true, ..Default::default() };
        let (train, _val, batch_cfg, vocab, _itos) = model::load_dataset_with_itos(&dir, &opts).unwrap();
        assert_eq!(vocab, VOCAB);
        let mut rng = Rng::new(9);
        for _ in 0..20 {
            let (x, _y) = train.get_batch(&batch_cfg, &mut rng);
            for row in x.chunks(SEQ_LEN) {
                assert!(
                    rules.iter().any(|r| r.cue == row[0]),
                    "a training window started at token {} instead of a record's cue - the loader is drawing misaligned windows and every supervised completion token in them is label noise",
                    row[0]
                );
                assert_eq!(row[PROMPT_LEN - 1], SEP);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
