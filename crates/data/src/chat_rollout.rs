// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a chat record says about the rollout it came from, and how a prepared dataset keeps it: version 2 of the
//! training-record contract.
//!
//! An orchestrator that collects agent rollouts outside brain writes one `generic-messages-v2` record per model call
//! (the assistant turn of the call is the action) and may add
//!
//! * `"reward": <number>` and `"group": "<key>"` on the record: the scalar reward the verifier gave the rollout the
//!   call belongs to, and the group of rollouts it is compared with (the attempts at one prompt by one policy). The
//!   trainer turns the rewards of a group into each rollout's advantage, by the run's advantage normalisation; the
//!   advantage is the weight of every supervised token of the record in a policy-gradient run. A group needs at least
//!   two rollouts that reached an outcome. The two fields come together.
//! * `"rollout": "<key>"` on the record (with `group`): which rollout of its group the record is one call of. The
//!   records of one rollout share its reward (and its `truncated`), and the rollout counts once in its group however
//!   many calls it made. Without it the record is a rollout of its own.
//! * `"behavior_logprobs": [<number>, ...]` on a supervised assistant message: the log-probability the sampler gave
//!   each token of that answer, one per token of the answer as the chat template renders it after the prompt (the
//!   answer's text and its closing end-of-turn marker; the whitespace the template writes after the marker, such as the newline after Qwen's
//!   `<|im_end|>`, is context: the sampler stops at the marker and never chose it). A value above zero is not a log-probability.
//! * `"token_ids": [<id>, ...]` on a supervised assistant message (beside `behavior_logprobs`): the ids the sampler
//!   generated for the answer, end-of-turn marker included. The trainer trains these ids, never the ids the answer's
//!   text renders to again: a model writes a call's arguments, its whitespace and its think block its own way, and the
//!   parsed message the record holds renders them the template's way, so a re-rendering would supervise tokens the
//!   policy did not sample (and their behaviour log-probabilities would be another token's). The ids are admitted when
//!   they are what the template renders for the answer, or when they read back, by the template's own reply format
//!   (its think block and `<tool_call>` calls), as the same answer: the same reasoning and content (up to the
//!   whitespace around them) and the same calls (names, and arguments as JSON values). Otherwise the record is refused
//!   (rule `token_ids_mismatch`), naming the first position that differs: ids that do not belong to the message.
//!   `behavior_logprobs` then holds one value per id.
//! * `"prompt_token_ids": [<id>, ...]` on a supervised assistant message (with `token_ids`): the ids the sampler was
//!   given before the answer. The trainer checks that the context it renders before the answer encodes to exactly
//!   these, and refuses the record otherwise (rule `prompt_token_ids_mismatch`, naming the first position that
//!   differs): a log-probability is the token's given everything before it, so an importance ratio is exact only in
//!   the context the sampler saw (a server that adds an instruction to the prompt, a producer that rewrites the
//!   history, a template that writes an earlier answer otherwise than it was generated all change it).
//! * `"truncated": true` on the record: the rollout did not reach an outcome (the token budget cut it, a step timed
//!   out), so its reward says nothing about the action. A policy-gradient run never trains on such a record and its
//!   reward takes no part in its group's advantages; it is counted, not drawn. A truncated rollout is masked as a
//!   whole, because the sampler's budget, not the policy, decided where it ended.
//! * `"baseline_only": true` on the record (with `reward` and `group`): a rollout of the group the run does not train
//!   on (another policy's attempt, a reference answer, a failure kept for the statistics). Its reward counts among its
//!   group's (and towards the group's two rollouts) and nothing of it is trained: no message has `"train": true`, and
//!   it carries no `behavior_logprobs`, `token_ids`, `teacher_context` or `truncated` (rule `baseline_trains`).
//! * `"tokenizer_digest": "sha256:<hex>"` on the record: the tokenizer the sampler encoded the rollout with (a served
//!   reply's `policy_trace` names it). The records of one job that name one must all name the same.
//! * `"teacher_context": {"at": <index>, "messages": [{"role": "system"|"user", "content": "..."}, ...]}` on the
//!   record: messages the teacher of an on-policy self-distillation run is shown and the student never saw, inserted
//!   before the record's message `at` (at or before its first supervised message). The teacher's rendering of the
//!   conversation is the student's with those messages in it; its supervised tokens must be exactly the student's.
//!   A record may carry a teacher context instead of a reward (the scalar advantage is then 0) or with one.
//!
//! Version 1's precomputed `"advantage"` is refused, naming the field and its replacement: the trainer computes the
//! advantage from the rewards of a group, so one implementation decides what a reward is worth.
//!
//! A prepared dataset keeps all of it beside its token files, per split: `<split>.rollouts.json` (the groups' keys and,
//! per example, whether it is a rollout and its group, rollout and reward; absent when no example is a rollout),
//! `<split>.blp.bin` (`f32` per token of the stream, NaN where no behaviour log-probability was recorded) and
//! `<split>.trn.bin` (a flag per example, written only when some example is truncated). Preparing a split removes a
//! stale sidecar. The teacher's renderings are kept as `<split>.tch.u32.bin`, `<split>.tch.mask.bin` and
//! `<split>.tch.ex.bin` (an example without a teacher context holds none of its own: an empty range).

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::binio;
use crate::chat::ChatSample;

/// The rule of the training-record contract a record breaks, when that is why it is refused: what a producer acts on
/// without reading the message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordRule {
    /// `behavior_logprobs` does not hold one value per token of its answer as the template renders it.
    BehaviorLogprobCount,
    /// `token_ids` are neither the tokens the template renders for the answer nor ids that read back as that answer.
    TokenIdsMismatch,
    /// `prompt_token_ids` are not the tokens of the context the trainer renders before the answer.
    PromptTokenIdsMismatch,
    /// A rollout field (`behavior_logprobs`, `token_ids`, `truncated`, `rollout`, `tokenizer_digest`) on a record with
    /// no `reward` (and no `teacher_context`), or a `group` without a `reward`: nothing says what its tokens earned.
    MissingReward,
    /// A `reward` (or a `rollout`) without the `group` it is compared within.
    MissingGroup,
    /// The record carries version 1's precomputed `advantage`; version 2 takes `reward` and `group` instead.
    AdvantageRemoved,
    /// Two records of one rollout (`group` and `rollout`) disagree on its `reward` or on `truncated`.
    RolloutMismatch,
    /// The record's group holds one rollout that reached an outcome: there is nothing to compare its reward with.
    GroupOfOne,
    /// The record's `tokenizer_digest` differs from the one the job's other records name.
    MixedTokenizerDigest,
    /// A record marked `baseline_only` supervises a message, carries the per-token fields of a trained answer
    /// (`behavior_logprobs`, `token_ids`), a `teacher_context`, or is `truncated`: a baseline only sets its group's
    /// baseline.
    BaselineTrains,
    /// The record's `tokenizer_digest` is not the digest of the tokenizer of the base the run trains.
    TokenizerMismatch,
    /// A message's `loss_weight` is not a finite number above 0, sits on a message that is not supervised
    /// (`"train": false`), or on a rollout record, whose tokens a policy gradient weighs by their advantage.
    LossWeight,
}

impl RecordRule {
    /// The rule's name on the wire (`check_dataset`'s refusals).
    pub fn name(self) -> &'static str {
        match self {
            RecordRule::BehaviorLogprobCount => "behavior_logprob_count",
            RecordRule::TokenIdsMismatch => "token_ids_mismatch",
            RecordRule::PromptTokenIdsMismatch => "prompt_token_ids_mismatch",
            RecordRule::MissingReward => "missing_reward",
            RecordRule::MissingGroup => "missing_group",
            RecordRule::AdvantageRemoved => "advantage_removed",
            RecordRule::RolloutMismatch => "rollout_mismatch",
            RecordRule::GroupOfOne => "group_of_one",
            RecordRule::MixedTokenizerDigest => "mixed_tokenizer_digest",
            RecordRule::BaselineTrains => "baseline_trains",
            RecordRule::TokenizerMismatch => "tokenizer_mismatch",
            RecordRule::LossWeight => "loss_weight",
        }
    }
}

/// A refusal of the training-record contract: the rule broken and the record field that broke it, as a JSON path
/// (`messages[1].behavior_logprobs`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordFault {
    pub rule: RecordRule,
    pub field: String,
}

impl RecordFault {
    pub fn new(rule: RecordRule, field: &str) -> RecordFault {
        RecordFault { rule, field: field.to_string() }
    }
}

/// Why a record was refused: the message, and the contract rule when that is the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordError {
    message: String,
    fault: Option<RecordFault>,
}

impl RecordError {
    pub(crate) fn of(fault: RecordFault, message: String) -> RecordError {
        RecordError { message, fault: Some(fault) }
    }

    /// The contract rule the record broke, when that is why it was refused.
    pub fn fault(&self) -> Option<&RecordFault> {
        self.fault.as_ref()
    }
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RecordError {}

impl From<String> for RecordError {
    fn from(message: String) -> RecordError {
        RecordError { message, fault: None }
    }
}

impl From<RecordError> for String {
    fn from(e: RecordError) -> String {
        e.message
    }
}

/// A log-probability can be at most zero; this much above it is rounding of the collector's own arithmetic.
const LOGPROB_SLACK: f32 = 1e-3;

/// The longest `group` or `rollout` key, in bytes.
pub const MAX_KEY_BYTES: usize = 512;

/// Which rollout of its group a record is one call of.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Member {
    /// The record's `"rollout"` key: every record naming it in the group is a call of the same rollout.
    Named(String),
    /// A record without a `"rollout"` key is a rollout of its own; every sample made from the record (one per answer,
    /// a repeat in a replay mix) carries the same identity.
    Record(u64),
}

impl Member {
    /// A rollout identity no other record has.
    pub(crate) fn of_record() -> Member {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Member::Record(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// The graded outcome of the rollout a record is one call of.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The verifier's reward for the rollout.
    pub reward: f64,
    /// The group of rollouts the reward is compared within.
    pub group: String,
    /// Which rollout of the group.
    pub member: Member,
}

/// The rollout data of one record: its outcome and, per message of the record, the behaviour log-probabilities of
/// that message's answer when the collector recorded them.
#[derive(Clone, Debug)]
pub struct Rollout {
    /// The rollout's reward and group; `None` for a record that trains on a teacher context alone.
    pub outcome: Option<Outcome>,
    /// The rollout did not reach an outcome; no policy-gradient run trains on it (see the module docs).
    pub truncated: bool,
    /// One entry per message of the record; `Some` only on a supervised assistant message.
    pub behavior_logprobs: Vec<Option<Vec<f32>>>,
    /// One entry per message of the record: the ids the sampler generated for that answer and, when recorded, the ids
    /// it was given before it (`"token_ids"`, `"prompt_token_ids"`). The answer trains exactly these ids.
    pub token_ids: Vec<Option<SampledIds>>,
    /// What the teacher is shown instead of the record's own conversation, when the record carries a
    /// `"teacher_context"`.
    pub teacher: Option<Teacher>,
    /// The tokenizer the sampler encoded the rollout with (`sha256:<hex>`), when the record names it.
    pub tokenizer_digest: Option<String>,
    /// The record only sets its group's baseline (`"baseline_only": true`): its reward counts among the group's, and
    /// nothing of it is trained.
    pub baseline_only: bool,
}

/// What the sampler generated for one answer and what it was given before it, as the collector recorded them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledIds {
    /// The ids generated for the answer (`"token_ids"`), end-of-turn marker included.
    pub answer: Vec<u32>,
    /// The ids of the prompt the answer was generated after (`"prompt_token_ids"`).
    pub prompt: Option<Vec<u32>>,
}

/// The teacher's view of a record: the record's conversation with the privileged messages inserted. It goes through
/// every transformation the student's sample goes through (one example per answer, packing, no-think), so the two stay
/// renderings of the same answers.
#[derive(Clone, Debug)]
pub struct Teacher {
    pub sample: Box<ChatSample>,
    /// How many messages the teacher's conversation holds beyond the student's, all before the first supervised one.
    pub inserted: usize,
}

impl Rollout {
    /// Whether any message of the record carries behaviour log-probabilities.
    pub fn has_behavior(&self) -> bool {
        self.behavior_logprobs.iter().any(Option::is_some)
    }

    /// The rollout of a sample derived from this record's (a prefix, one answer, a packing): the same outcome, with
    /// the derived sample's log-probabilities and teacher view.
    pub(crate) fn derived(&self, behavior_logprobs: Vec<Option<Vec<f32>>>, token_ids: Vec<Option<SampledIds>>, teacher: Option<Teacher>) -> Rollout {
        Rollout { outcome: self.outcome.clone(), truncated: self.truncated, behavior_logprobs, token_ids, teacher, tokenizer_digest: self.tokenizer_digest.clone(), baseline_only: self.baseline_only }
    }
}

/// A record's reward as the wire gives it: a finite number.
pub(crate) fn check_reward(reward: &serde_json::Value) -> Result<f64, String> {
    let number = reward.as_f64().ok_or_else(|| format!("\"reward\" is {reward}, which is not a number"))?;
    if !number.is_finite() {
        return Err(format!("\"reward\" is {number}, which is not a finite number"));
    }
    Ok(number)
}

/// A `group` or `rollout` key: not empty, at most [`MAX_KEY_BYTES`] bytes, no control characters.
pub(crate) fn check_key(name: &str, key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err(format!("\"{name}\" is empty"));
    }
    if key.len() > MAX_KEY_BYTES {
        return Err(format!("\"{name}\" is {} bytes long; a key is at most {MAX_KEY_BYTES}", key.len()));
    }
    if key.chars().any(char::is_control) {
        return Err(format!("\"{name}\" holds a control character"));
    }
    Ok(())
}

/// A tokenizer digest as a served reply names one: `sha256:` and 64 lowercase hex digits.
pub(crate) fn check_tokenizer_digest(digest: &str) -> Result<(), String> {
    let hex = digest.strip_prefix("sha256:").unwrap_or("");
    if hex.len() != 64 || !hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(format!("\"tokenizer_digest\" is {digest:?}; it is \"sha256:\" and 64 lowercase hex digits, as a policy trace names it"));
    }
    Ok(())
}

/// One message's behaviour log-probabilities: a non-empty list of finite numbers, none above zero.
pub(crate) fn check_behavior_logprobs(values: &serde_json::Value) -> Result<Vec<f32>, String> {
    let list = values.as_array().ok_or_else(|| "\"behavior_logprobs\" is not a list of numbers".to_string())?;
    if list.is_empty() {
        return Err("\"behavior_logprobs\" is empty; omit it when the sampler did not record any".to_string());
    }
    list.iter()
        .enumerate()
        .map(|(k, v)| {
            let number = v.as_f64().ok_or_else(|| format!("\"behavior_logprobs\"[{k}] is {v}, which is not a number"))?;
            let narrowed = number as f32;
            if !number.is_finite() || !narrowed.is_finite() {
                Err(format!("\"behavior_logprobs\"[{k}] is {number}, which is not a finite number"))
            } else if narrowed > LOGPROB_SLACK {
                Err(format!("\"behavior_logprobs\"[{k}] is {number}; a log-probability is at most 0"))
            } else {
                Ok(narrowed.min(0.0))
            }
        })
        .collect()
}

/// What a run does with a rollout marked `"truncated": true`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncatedRollouts {
    /// Never drawn, scored or trained, and its reward takes no part in its group's advantages (DAPO's overlong
    /// filtering): the sampler's budget, not the policy, decided where it ended.
    #[default]
    Mask,
    /// Trained like any rollout, its reward (the collector's verdict on a cut-off rollout, a penalty for instance)
    /// counted in its group.
    Keep,
}

impl TruncatedRollouts {
    /// Both, in the order the wire lists them.
    pub const ALL: [TruncatedRollouts; 2] = [TruncatedRollouts::Mask, TruncatedRollouts::Keep];

    /// The name on the wire (`truncated`).
    pub fn name(self) -> &'static str {
        match self {
            TruncatedRollouts::Mask => "mask",
            TruncatedRollouts::Keep => "keep",
        }
    }

    /// The handling a wire name names.
    ///
    /// # Errors
    /// The name is neither `mask` nor `keep`.
    pub fn parse(name: &str) -> Result<TruncatedRollouts, String> {
        TruncatedRollouts::ALL.into_iter().find(|t| t.name() == name).ok_or_else(|| format!("truncated {name:?} is neither \"mask\" nor \"keep\""))
    }

    /// Whether a rollout with this `truncated` flag takes part in its group and trains.
    pub fn counts(self, truncated: bool) -> bool {
        !truncated || self == TruncatedRollouts::Keep
    }
}

/// One graded call of a rollout, as the group check reads it: its group, which rollout, the reward and whether the
/// rollout was truncated.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupEntry<G, M> {
    pub group: G,
    pub member: M,
    pub reward: f64,
    pub truncated: bool,
}

/// What breaks the group rules of the contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupFault<G> {
    /// Entry `entry` names a rollout an earlier entry named with another `reward` (`field` `"reward"`) or another
    /// `truncated` (`field` `"truncated"`).
    Mismatch { entry: usize, field: &'static str },
    /// `group` holds one rollout that reached an outcome.
    OfOne { group: G },
}

/// Per group, its rollouts that reached an outcome, each once as `(rollout, reward)` in the order first seen.
pub type GroupRewards<G, M> = BTreeMap<G, Vec<(M, f64)>>;

/// The rollouts of each group that take part in it ([`GroupRewards`]: those that reached an outcome, and under
/// [`TruncatedRollouts::Keep`] the truncated ones too) and what breaks the group rules: the one place the contract's
/// groups are read, for a record file and for a prepared split alike. A group none of whose rollouts takes part trains
/// on nothing and is not in the map.
pub fn group_members<G: Ord + Clone, M: Ord + Clone>(entries: &[GroupEntry<G, M>], truncated: TruncatedRollouts) -> (GroupRewards<G, M>, Vec<GroupFault<G>>) {
    let mut seen: BTreeMap<(G, M), (f64, bool)> = BTreeMap::new();
    let mut groups: GroupRewards<G, M> = BTreeMap::new();
    let mut faults = Vec::new();
    for (at, entry) in entries.iter().enumerate() {
        match seen.get(&(entry.group.clone(), entry.member.clone())) {
            Some(&(reward, truncated)) => {
                // The same value twice is the same outcome; a bitwise comparison would tell 0.0 from -0.0.
                #[allow(clippy::float_cmp)]
                let fault = if reward != entry.reward {
                    Some("reward")
                } else if truncated != entry.truncated {
                    Some("truncated")
                } else {
                    None
                };
                if let Some(field) = fault {
                    faults.push(GroupFault::Mismatch { entry: at, field });
                }
            }
            None => {
                seen.insert((entry.group.clone(), entry.member.clone()), (entry.reward, entry.truncated));
                if truncated.counts(entry.truncated) {
                    groups.entry(entry.group.clone()).or_default().push((entry.member.clone(), entry.reward));
                }
            }
        }
    }
    faults.extend(groups.iter().filter(|(_, members)| members.len() == 1).map(|(group, _)| GroupFault::OfOne { group: group.clone() }));
    (groups, faults)
}

/// The rules of the contract that hold between records: the records of one rollout agree on its outcome, a group
/// holds at least two rollouts that take part in it (under `truncated`), and the records that name a tokenizer name
/// one, which is `base_tokenizer` (the digest of the trained base's own tokenizer, when it is known) where it is given.
/// `records` are the rollouts of the records read together (one file, or every set one run trains on), `None` for a
/// record that is not a rollout. The refusals, at most one per record, by index into `records`.
pub fn check_records(records: &[Option<&Rollout>], truncated: TruncatedRollouts, base_tokenizer: Option<&str>) -> Vec<(usize, RecordError)> {
    let mut refusals: BTreeMap<usize, RecordError> = BTreeMap::new();
    let graded: Vec<(usize, &Rollout, &Outcome)> = records.iter().enumerate().filter_map(|(at, r)| r.and_then(|r| r.outcome.as_ref().map(|o| (at, r, o)))).collect();
    let entries: Vec<GroupEntry<&str, &Member>> = graded.iter().map(|(_, r, o)| GroupEntry { group: o.group.as_str(), member: &o.member, reward: o.reward, truncated: r.truncated }).collect();
    let (_, faults) = group_members(&entries, truncated);
    for fault in faults {
        match fault {
            GroupFault::Mismatch { entry, field } => {
                let (at, rollout, outcome) = graded[entry];
                let rollout_key = match &outcome.member {
                    Member::Named(key) => key.as_str(),
                    Member::Record(_) => "",
                };
                let value = if field == "reward" { outcome.reward.to_string() } else { rollout.truncated.to_string() };
                let message = format!("{field}: rollout {rollout_key:?} of group {:?} is {value} on this record and differs on an earlier one; the records of one rollout share its reward and whether it was truncated", outcome.group);
                refusals.entry(at).or_insert_with(|| RecordError::of(RecordFault::new(RecordRule::RolloutMismatch, field), message));
            }
            GroupFault::OfOne { group } => {
                for (at, rollout, outcome) in &graded {
                    if outcome.group == group && truncated.counts(rollout.truncated) {
                        let message = format!("group: group {group:?} holds one rollout that takes part in it; a group-relative advantage compares a rollout's reward with the others of its group, so a group needs at least two (sample more rollouts of the prompt, or leave the group out)");
                        refusals.entry(*at).or_insert_with(|| RecordError::of(RecordFault::new(RecordRule::GroupOfOne, "group"), message));
                    }
                }
            }
        }
    }
    let mut first: Option<&str> = None;
    for (at, rollout) in records.iter().enumerate() {
        let Some(digest) = rollout.and_then(|r| r.tokenizer_digest.as_deref()) else { continue };
        if let Some(base) = base_tokenizer.filter(|base| *base != digest) {
            let message = format!("tokenizer_digest: the rollout was encoded by the tokenizer {digest}, and the base this run trains reads {base}; its token ids and log-probabilities belong to another vocabulary");
            refusals.entry(at).or_insert_with(|| RecordError::of(RecordFault::new(RecordRule::TokenizerMismatch, "tokenizer_digest"), message));
            continue;
        }
        match first {
            None => first = Some(digest),
            Some(expected) if expected != digest => {
                let message = format!("tokenizer_digest: {digest} is not {expected}, the tokenizer an earlier record names; the rollouts one run trains on were all encoded by one tokenizer");
                refusals.entry(at).or_insert_with(|| RecordError::of(RecordFault::new(RecordRule::MixedTokenizerDigest, "tokenizer_digest"), message));
            }
            Some(_) => {}
        }
    }
    refusals.into_iter().collect()
}

/// The graded outcome of a prepared example: its group and rollout (indices into the split's tables) and the reward.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graded {
    /// Index into [`SplitRollouts::groups`].
    pub group: u32,
    /// The rollout within the split: examples with one `member` are calls (or answers) of one rollout.
    pub member: u32,
    pub reward: f64,
}

/// What a prepared example says about its rollout; an imitation example has none.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExampleRollout {
    /// `None` for a rollout that trains on a teacher context alone.
    pub graded: Option<Graded>,
}

/// The `<split>.rollouts.json` sidecar.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutsFile {
    groups: Vec<String>,
    examples: Vec<Option<ExampleRollout>>,
    /// The outcomes of baseline-only records: counted in their groups, never trained.
    #[serde(default)]
    baselines: Vec<Graded>,
}

fn names(dir: &Path, split: &str) -> (PathBuf, PathBuf, PathBuf) {
    (dir.join(format!("{split}.rollouts.json")), dir.join(format!("{split}.blp.bin")), dir.join(format!("{split}.trn.bin")))
}

/// Whether `split` of the prepared dataset in `dir` holds rollouts (its rollout sidecar is there).
pub fn holds_rollouts(dir: &Path, split: &str) -> bool {
    names(dir, split).0.exists()
}

/// Write a split's sidecars: `rollouts[i]` is example `i`'s rollout (`None`: an imitation example; `Some(None)`: a
/// rollout without a reward of its own), `behavior[t]` token `t`'s log-probability (NaN for none), `truncated[i]`
/// whether example `i`'s rollout did not reach an outcome. Groups and rollouts are numbered in the order first seen.
/// Nothing is written, and a stale sidecar is removed, when no example is a rollout.
pub fn write(dir: &Path, split: &str, rollouts: &[Option<Option<Outcome>>], baselines: &[Outcome], behavior: &[f32], truncated: &[bool]) -> io::Result<()> {
    let (outcomes, blp, trn) = names(dir, split);
    for stale in [&outcomes, &blp, &trn] {
        match std::fs::remove_file(stale) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    if rollouts.iter().all(Option::is_none) && baselines.is_empty() {
        return Ok(());
    }
    if let Some(o) = rollouts.iter().flatten().flatten().chain(baselines).find(|o| !o.reward.is_finite()) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("group {:?}: reward {} is not a finite number", o.group, o.reward)));
    }
    let mut groups: Vec<String> = Vec::new();
    let mut group_index: BTreeMap<String, u32> = BTreeMap::new();
    let mut member_index: BTreeMap<(u32, Member), u32> = BTreeMap::new();
    let mut grade = |o: &'_ Outcome| -> Graded {
        let group = *group_index.entry(o.group.clone()).or_insert_with(|| {
            groups.push(o.group.clone());
            (groups.len() - 1) as u32
        });
        let next = member_index.len() as u32;
        let member = *member_index.entry((group, o.member.clone())).or_insert(next);
        Graded { group, member, reward: o.reward }
    };
    let examples: Vec<Option<ExampleRollout>> = rollouts.iter().map(|rollout| rollout.as_ref().map(|outcome| ExampleRollout { graded: outcome.as_ref().map(&mut grade) })).collect();
    let baselines: Vec<Graded> = baselines.iter().map(&mut grade).collect();
    let file = RolloutsFile { groups, examples, baselines };
    std::fs::write(&outcomes, serde_json::to_vec(&file).map_err(io::Error::other)?)?;
    if behavior.iter().any(|b| !b.is_nan()) {
        binio::write_f32_bin(&blp, behavior)?;
    }
    if truncated.iter().any(|t| *t) {
        binio::write_mask_bin(&trn, truncated)?;
    }
    Ok(())
}

/// The rollout data of one split of a prepared dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitRollouts {
    /// The groups' keys, by the index [`Graded::group`] names.
    pub groups: Vec<String>,
    /// Per example; `None` for an imitation example.
    pub examples: Vec<Option<ExampleRollout>>,
    /// Per token of the split's stream, aligned with it; NaN where none was recorded.
    pub behavior: Vec<f32>,
    /// Per example: the rollout did not reach an outcome and is not trained on.
    pub truncated: Vec<bool>,
    /// The outcomes of baseline-only records, numbered in the examples' group and rollout spaces: counted in their
    /// groups, never trained.
    pub baselines: Vec<Graded>,
}

/// The sidecars of `split` in `dir`, checked against the `examples` and `tokens` the split holds; `None` when the
/// split has no rollout data.
pub fn read(dir: &Path, split: &str, examples: usize, tokens: usize) -> io::Result<Option<SplitRollouts>> {
    let (outcomes, blp, trn) = names(dir, split);
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    if !outcomes.exists() {
        return Ok(None);
    }
    let file: RolloutsFile = serde_json::from_slice(&std::fs::read(&outcomes)?).map_err(|e| bad(format!("{}: {e}", outcomes.display())))?;
    if file.examples.len() != examples {
        return Err(bad(format!("{}: rollouts for {} examples, the split holds {examples}", outcomes.display(), file.examples.len())));
    }
    for (e, graded) in file.examples.iter().enumerate().filter_map(|(e, r)| r.and_then(|r| r.graded).map(|g| (e, g))).chain(file.baselines.iter().map(|g| (usize::MAX, *g))) {
        if graded.group as usize >= file.groups.len() || !graded.reward.is_finite() {
            return Err(bad(format!("{}: example {e} names group {} of {} with reward {}", outcomes.display(), graded.group, file.groups.len(), graded.reward)));
        }
    }
    let behavior = if blp.exists() {
        let behavior = binio::read_f32_bin(&blp)?;
        if behavior.len() != tokens {
            return Err(bad(format!("{}: log-probabilities for {} tokens, the split holds {tokens}", blp.display(), behavior.len())));
        }
        behavior
    } else {
        vec![f32::NAN; tokens]
    };
    let truncated = if trn.exists() {
        let flags = binio::read_mask_bin(&trn)?;
        if flags.len() != examples {
            return Err(bad(format!("{}: truncation flags for {} examples, the split holds {examples}", trn.display(), flags.len())));
        }
        flags
    } else {
        vec![false; examples]
    };
    Ok(Some(SplitRollouts { groups: file.groups, examples: file.examples, behavior, truncated, baselines: file.baselines }))
}

/// The teacher renderings of a split: every example's tokens and supervision mask, back to back.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherStream {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
    /// Where each example's rendering starts; an example without a teacher context holds an empty range.
    pub starts: Vec<usize>,
}

impl TeacherStream {
    /// The teacher's tokens and mask of example `e`; `None` for an example without a teacher context.
    pub fn example(&self, e: usize) -> Option<(&[u32], &[bool])> {
        let (a, b) = (self.starts[e], self.starts.get(e + 1).copied().unwrap_or(self.ids.len()));
        (a < b).then(|| (&self.ids[a..b], &self.mask[a..b]))
    }
}

fn teacher_names(dir: &Path, split: &str) -> [PathBuf; 3] {
    ["tch.u32.bin", "tch.mask.bin", "tch.ex.bin"].map(|suffix| dir.join(format!("{split}.{suffix}")))
}

/// Write a split's teacher renderings (`None` for an example without one). Nothing is written, and a stale sidecar is
/// removed, when no example has a teacher.
pub fn write_teacher(dir: &Path, split: &str, teachers: &[Option<(Vec<u32>, Vec<bool>)>]) -> io::Result<()> {
    let [ids_file, mask_file, starts_file] = teacher_names(dir, split);
    for stale in [&ids_file, &mask_file, &starts_file] {
        match std::fs::remove_file(stale) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    if teachers.iter().all(Option::is_none) {
        return Ok(());
    }
    let (mut ids, mut mask, mut starts) = (Vec::new(), Vec::new(), Vec::new());
    for teacher in teachers {
        starts.push(ids.len() as u64);
        if let Some((t_ids, t_mask)) = teacher {
            ids.extend_from_slice(t_ids);
            mask.extend_from_slice(t_mask);
        }
    }
    binio::write_u32_bin(&ids_file, &ids)?;
    binio::write_mask_bin(&mask_file, &mask)?;
    binio::write_u64_bin(&starts_file, &starts)
}

/// The teacher renderings of `split` in `dir`, checked against the `examples` the split holds; `None` when no example
/// has a teacher.
pub fn read_teacher(dir: &Path, split: &str, examples: usize) -> io::Result<Option<TeacherStream>> {
    let [ids_file, mask_file, starts_file] = teacher_names(dir, split);
    if !ids_file.exists() {
        return Ok(None);
    }
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    let ids = binio::read_u32_bin(&ids_file)?;
    let mask = binio::read_mask_bin(&mask_file)?;
    let starts: Vec<usize> = binio::read_u64_bin(&starts_file)?.into_iter().map(|s| s as usize).collect();
    if mask.len() != ids.len() {
        return Err(bad(format!("{}: {} supervision flags for {} teacher tokens", mask_file.display(), mask.len(), ids.len())));
    }
    if starts.len() != examples {
        return Err(bad(format!("{}: teacher renderings for {} examples, the split holds {examples}", starts_file.display(), starts.len())));
    }
    if starts.windows(2).any(|w| w[0] > w[1]) || starts.last().is_some_and(|&s| s > ids.len()) {
        return Err(bad(format!("{}: example starts are not ordered within the stream", starts_file.display())));
    }
    Ok(Some(TeacherStream { ids, mask, starts }))
}
