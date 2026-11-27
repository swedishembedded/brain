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
//! * `"behavior_top_ids": [[<id>, ...K], ...]` and `"behavior_top_logprobs": [[<number>, ...K], ...]` on a supervised
//!   assistant message, beside `behavior_logprobs` (which they require; the two come together): the sampler's `K`
//!   most likely tokens at each step and their log-probabilities, one inner list per token, exactly as many as
//!   `behavior_logprobs` holds (rule `behavior_top_count`, naming `messages[i].behavior_top_ids`, otherwise). `K >= 1`
//!   is the same for every token of the record; the ids of one token are distinct; the log-probabilities are finite,
//!   at most 0, and whole-row log-probabilities of the sampler's distribution at its temperature, so the
//!   probabilities of one token sum to at most one (rule `behavior_top`, naming the field). The sampled token need
//!   not be among them. A `baseline_only` record carries none (`baseline_trains`) and neither does a record without a
//!   reward or teacher (`missing_reward`). The data crate does not know the vocabulary of the model a dataset is
//!   trained on, so an id that is not in it is refused by the trainer, not here.
//! * `"behavior_support_offsets": [<u64>, ...]`, `"behavior_support_ids": [<id>, ...]` and
//!   `"behavior_support_logprobs": [<number>, ...]` on a supervised assistant message, beside `behavior_logprobs` (which
//!   they require; the three come together): the support a cut sampler (`top_k`, `top_p`, `min_p`) drew each token
//!   from, and the token's log-probability within it, the figures a trainer needs to replay the draw without bias (a
//!   served reply's `policy_trace` with `support` carries them). Token `t`'s support is
//!   `behavior_support_ids[offsets[t]..offsets[t + 1]]`: the offsets are one more than `behavior_logprobs` holds
//!   values, start at 0, rise at every token (no token has an empty support) and end at the number of ids
//!   (rule `behavior_support_count` for a length that does not number the answer's tokens). The ids of one token are
//!   distinct, at most [`MAX_BEHAVIOR_SUPPORT`] of them, never [`NO_TOP_ID`], and hold the token's own id where the
//!   message carries `token_ids`; each log-probability is finite, at most 0 and at least the token's
//!   `behavior_logprobs` value (renormalising over a part of the row never lowers it) (rule `behavior_support`). As for
//!   the top-k, an id past the vocabulary, and a token outside its support where no `token_ids` are given, are refused
//!   by the trainer, which knows the vocabulary and the tokens.
//! * `"policy_version": <u64>` on the record: the oldest policy version among the requests behind it (a non-negative
//!   integer). Kept per example in the prepared split.
//! * `"sampler": {"temperature": <number>?, "top_k": <u32>?, "top_p": <number>?, "min_p": <number>?}` on the record:
//!   the sampling controls the decoder applied (`temperature` finite and above 0, `top_k` at least 1, `top_p` in
//!   (0, 1], `min_p` in [0, 1)). Kept per example in the prepared split ([`SamplerDecl::truncates`] says whether the
//!   sampled distribution is a truncation of the model's).
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
//! * A **session-merged record**: one rollout record of a whole agent session whose every sampled answer is supervised
//!   (`"train": true`), each with its `token_ids`, `prompt_token_ids` and `behavior_logprobs`, trained as one sequence
//!   by a run that packs answers (`pack_answers`; [`crate::chat::ChatSample::answers_packed`]): the shared context is
//!   encoded once, and the loss covers the sampled tokens only. It is exact when the session is append-only: every
//!   later answer's `prompt_token_ids` are the session so far (the earlier requests, the answers' sampled ids and the
//!   turns between, as the template renders them). A record that names `prompt_token_ids` on any answer is held to that
//!   on every later answer that carries `behavior_logprobs` (they must name `token_ids` and `prompt_token_ids`, and
//!   those must extend the session), and is refused otherwise (rule `session_not_append_only`, naming the first answer):
//!   a template that renders an earlier answer otherwise than it was sampled (an earlier turn's reasoning dropped), or
//!   a compacted history, is not append-only, and its producer sends the session as one record per call. Without
//!   packing, the same record trains as one example per answer, each in its own request.
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
//! * A teacher model's scores of the record's sampled answers, for on-policy distillation from another model
//!   (`rl::objective::onpolicy_distill`), in one of two forms, on every supervised assistant message of the record:
//!   * `"teacher_logprobs": [<number>, ...]`: the teacher's log-probability of each token of the answer, one per token
//!     exactly as `behavior_logprobs` counts them (the `score` action's per-token reply for a token record of the
//!     sampled `prompt_token_ids` and `token_ids`). Token-level distillation: only for a teacher whose tokenizer is the
//!     student's, which the run checks; with it the record needs no `reward` (the scalar advantage is then 0).
//!   * `"teacher_logprob": <number>`: the teacher's log-probability of the answer's text, rendered with the teacher's
//!     own chat template and tokenized its own way (the `score` action's `logprob` of the answer for the same chat
//!     record). Sequence-level distillation, for any teacher: the record's signal is the sum of its answers' scores
//!     less the sum of their `behavior_logprobs` (which they must carry), compared with the other rollouts of its
//!     `group` (so a `reward` and `group` are needed; a reward of 0 on every rollout when there is no verifier), and the
//!     record is the whole rollout (no `rollout` key).
//!
//!   Both forms in one record, a score on some answers and not others, a score beside a `teacher_context` or a
//!   per-answer score without what it needs is refused (rule `teacher_scores`), as is a `teacher_logprobs` that does
//!   not number its answer's tokens (rule `teacher_logprob_count`).
//!
//! Version 1's precomputed `"advantage"` is refused, naming the field and its replacement: the trainer computes the
//! advantage from the rewards of a group, so one implementation decides what a reward is worth.
//!
//! A prepared dataset keeps all of it beside its token files, per split: `<split>.rollouts.json` (the groups' keys and,
//! per example, whether it is a rollout and its group, rollout and reward; absent when no example is a rollout),
//! `<split>.blp.bin` (`f32` per token of the stream, NaN where no behaviour log-probability was recorded) and
//! `<split>.trn.bin` (a flag per example, written only when some example is truncated). A sampler's top-k is kept as
//! `<split>.btk.bin` (header-free `u32` ids, `[tokens * K]`, [`NO_TOP_ID`] where none was recorded) and
//! `<split>.btl.bin` (`f32` log-probabilities, `[tokens * K]`, NaN where none), `K` as `behavior_top_k` in
//! `<split>.rollouts.json` ([`read_behavior_top`]); one split holds one `K`. A sampler's support is kept as
//! `<split>.bso.bin` (`u64` offsets, `[tokens + 1]`, an empty range at a token without one), `<split>.bsi.bin` (`u32`
//! ids) and `<split>.bsl.bin` (`f32` per token, NaN where none), the widest support as `behavior_support_max` in
//! `<split>.rollouts.json` ([`read_behavior_support`]). Preparing a split removes a stale sidecar. The teacher's renderings are kept as `<split>.tch.u32.bin`, `<split>.tch.mask.bin` and
//! `<split>.tch.ex.bin` (an example without a teacher context holds none of its own: an empty range). A teacher model's
//! per-token scores are kept as `<split>.tlp.bin` (`f32` per token, NaN where none was recorded), a rollout's
//! sequence-level signal with its outcome in `<split>.rollouts.json`.

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
    /// no `reward` (and no `teacher_context` or `teacher_logprobs`), or a `group` without a `reward`: nothing says what
    /// its tokens earned.
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
    /// (`behavior_logprobs`, `behavior_top_ids`, `token_ids`), a `teacher_context`, or is `truncated`: a baseline only sets its group's
    /// baseline.
    BaselineTrains,
    /// The record's `tokenizer_digest` is not the digest of the tokenizer of the base the run trains.
    TokenizerMismatch,
    /// A message's `loss_weight` is not a finite number above 0, sits on a message that is not supervised
    /// (`"train": false`), or on a rollout record, whose tokens a policy gradient weighs by their advantage.
    LossWeight,
    /// `teacher_logprobs` does not hold one value per token of its answer as the template renders it.
    TeacherLogprobCount,
    /// `behavior_top_ids` or `behavior_top_logprobs` does not hold one list per value of the answer's
    /// `behavior_logprobs`.
    BehaviorTopCount,
    /// `behavior_top_ids` and `behavior_top_logprobs` stated so that they are not a sampler's top-k: one without the
    /// other or without `behavior_logprobs`, an empty or ragged list (a `K` that differs between tokens, answers of a
    /// record, or the examples of a split), a repeated id, an id that is [`NO_TOP_ID`], or log-probabilities that are
    /// not finite, above 0 or sum (as probabilities) to more than one.
    BehaviorTop,
    /// `behavior_support_offsets` or `behavior_support_logprobs` does not number the tokens of the answer's
    /// `behavior_logprobs` (one more offset, one log-probability per token).
    BehaviorSupportCount,
    /// `behavior_support_offsets`, `behavior_support_ids` and `behavior_support_logprobs` stated so that they are not a
    /// sampler's support: one without the others or without `behavior_logprobs`, offsets that do not start at 0, rise at
    /// every token and end at the ids, a repeated id, an id that is [`NO_TOP_ID`], more than [`MAX_BEHAVIOR_SUPPORT`]
    /// ids at a token, a token outside its support, or a log-probability that is not finite, above 0 or below the
    /// token's whole-row figure.
    BehaviorSupport,
    /// A teacher model's scores (`teacher_logprobs`, `teacher_logprob`) stated so that no distillation estimator can
    /// use them: both forms in one record, a score on some supervised answers and not on others, beside a
    /// `teacher_context`, a value that is not a log-probability, or a per-answer score without what the
    /// sequence-level estimator needs (`behavior_logprobs` on the answer, a `reward` and `group`, and the record being
    /// its whole rollout: no `rollout` key).
    TeacherScores,
    /// A session-merged record (several supervised answers of one session trained as one sequence) whose session is
    /// not append-only: a later answer's `prompt_token_ids` are not the session so far (the earlier requests, the
    /// answers as they were sampled and the turns between), or a later answer with `behavior_logprobs` names no
    /// `token_ids` and `prompt_token_ids` to show it. Its producer sends the session as one record per call instead.
    SessionNotAppendOnly,
}

/// The training-record contract this build of brain holds records to: its name, as the example set's `manifest.json`
/// states it.
pub const CONTRACT: &str = "training-records-v2";
/// The contract's version number.
pub const CONTRACT_VERSION: u32 = 2;
/// The digest of the contract's example set (`crates/data/tests/fixtures/training-records/v2`, [`contract_digest`]):
/// what a producer that keeps a copy of the set compares with the digest of its copy to know it writes the records
/// this build checks. Any change to what the set accepts or refuses changes it; a test holds it to the set.
pub const CONTRACT_DIGEST: &str = "sha256:4fb13e2f610fb2d3d4ede4731c5da509d2ccfae1071143c796015e824a4bc20b";

/// The digest of a contract example set in `dir`: `sha256:<hex>` of the lines `<path>\0<sha256 hex of the file>\n`,
/// one per file under `dir` but its `README.md` (the prose), `path` relative to `dir` with `/` separators, sorted by
/// path. A producer computes the same over its copy.
pub fn contract_digest(dir: &Path) -> io::Result<String> {
    use sha2::{Digest, Sha256};
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, out)?;
                continue;
            }
            let relative = path.strip_prefix(root).map_err(io::Error::other)?;
            let name = relative.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
            if name != "README.md" {
                let hex: String = Sha256::digest(std::fs::read(&path)?).iter().map(|b| format!("{b:02x}")).collect();
                out.push((name, hex));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files)?;
    files.sort();
    let mut listing = Sha256::new();
    for (name, hex) in &files {
        listing.update(format!("{name}\0{hex}\n").as_bytes());
    }
    Ok(format!("sha256:{}", listing.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()))
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
            RecordRule::TeacherLogprobCount => "teacher_logprob_count",
            RecordRule::BehaviorTopCount => "behavior_top_count",
            RecordRule::BehaviorTop => "behavior_top",
            RecordRule::BehaviorSupportCount => "behavior_support_count",
            RecordRule::BehaviorSupport => "behavior_support",
            RecordRule::TeacherScores => "teacher_scores",
            RecordRule::SessionNotAppendOnly => "session_not_append_only",
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

/// The id a `<split>.btk.bin` holds where no top-k was recorded for a token (no vocabulary has that many ids).
pub const NO_TOP_ID: u32 = u32::MAX;

/// The sampler's `k` most likely tokens at each step and their log-probabilities (`"behavior_top_ids"`,
/// `"behavior_top_logprobs"`): of one answer, one example, or a split's whole token stream, row-major `[tokens * k]`,
/// with [`NO_TOP_ID`] and NaN at a token that has none.
#[derive(Clone, Debug, PartialEq)]
pub struct BehaviorTop {
    pub k: usize,
    pub ids: Vec<u32>,
    pub logprobs: Vec<f32>,
}

impl BehaviorTop {
    /// How many tokens the table holds rows for.
    pub fn tokens(&self) -> usize {
        self.ids.len().checked_div(self.k).unwrap_or(0)
    }

    /// `tokens` rows with nothing recorded.
    fn none(k: usize, tokens: usize) -> BehaviorTop {
        BehaviorTop { k, ids: vec![NO_TOP_ID; tokens * k], logprobs: vec![f32::NAN; tokens * k] }
    }

    /// Append `tokens` rows with nothing recorded.
    pub(crate) fn extend_none(&mut self, tokens: usize) {
        self.ids.resize(self.ids.len() + tokens * self.k, NO_TOP_ID);
        self.logprobs.resize(self.logprobs.len() + tokens * self.k, f32::NAN);
    }

    /// Append the rows of `other`, which has the same `k`.
    pub(crate) fn extend(&mut self, other: &BehaviorTop) {
        assert_eq!(self.k, other.k, "BehaviorTop::extend: tables of different k");
        self.ids.extend_from_slice(&other.ids);
        self.logprobs.extend_from_slice(&other.logprobs);
    }

    /// The rows of consecutive stretches of tokens, `None` for a stretch with no table: `parts` are `(tokens, table)`.
    /// `None` when no stretch has one.
    pub(crate) fn assemble(parts: &[(usize, Option<&BehaviorTop>)]) -> Result<Option<BehaviorTop>, String> {
        let Some(k) = parts.iter().find_map(|(_, top)| top.map(|t| t.k)) else { return Ok(None) };
        let mut out = BehaviorTop::none(k, 0);
        for (tokens, top) in parts {
            match top {
                Some(top) if top.k != k => return Err(format!("the answers of one example carry top-{k} and top-{} tables; one example holds one K", top.k)),
                Some(top) => out.extend(top),
                None => out.extend_none(*tokens),
            }
        }
        Ok(Some(out))
    }
}

/// The most ids a token's support may hold (`"behavior_support_ids"`): a cut sampler's survivors, bounded so a record
/// cannot ask the trainer for an unbounded number of extra targets per position.
pub const MAX_BEHAVIOR_SUPPORT: usize = 1024;

/// The support a cut sampler drew each token from and the token's log-probability within it
/// (`"behavior_support_offsets"`, `"behavior_support_ids"`, `"behavior_support_logprobs"`): of one answer, one example,
/// or a split's whole token stream. Token `t`'s support is `ids[offsets[t]..offsets[t + 1]]`, empty (and its
/// log-probability NaN) at a token that has none.
#[derive(Clone, Debug, PartialEq)]
pub struct BehaviorSupport {
    pub offsets: Vec<u64>,
    pub ids: Vec<u32>,
    pub logprobs: Vec<f32>,
    /// The widest support of any token.
    pub max: usize,
}

impl BehaviorSupport {
    /// How many tokens the table holds rows for.
    pub fn tokens(&self) -> usize {
        self.logprobs.len()
    }

    /// Token `t`'s support (empty: none recorded).
    pub fn of(&self, t: usize) -> &[u32] {
        &self.ids[self.offsets[t] as usize..self.offsets[t + 1] as usize]
    }

    /// `tokens` rows with nothing recorded.
    pub fn none(tokens: usize) -> BehaviorSupport {
        BehaviorSupport { offsets: vec![0; tokens + 1], ids: Vec::new(), logprobs: vec![f32::NAN; tokens], max: 0 }
    }

    /// Append `tokens` rows with nothing recorded.
    pub fn extend_none(&mut self, tokens: usize) {
        let end = self.ids.len() as u64;
        self.offsets.extend(std::iter::repeat_n(end, tokens));
        self.logprobs.extend(std::iter::repeat_n(f32::NAN, tokens));
    }

    /// Append the rows of `other`.
    pub fn extend(&mut self, other: &BehaviorSupport) {
        let base = self.ids.len() as u64;
        self.offsets.extend(other.offsets[1..].iter().map(|o| base + o));
        self.ids.extend_from_slice(&other.ids);
        self.logprobs.extend_from_slice(&other.logprobs);
        self.max = self.max.max(other.max);
    }

    /// The rows of consecutive stretches of tokens, `None` for a stretch with no table: `parts` are `(tokens, table)`.
    /// `None` when no stretch has one.
    pub fn assemble(parts: &[(usize, Option<&BehaviorSupport>)]) -> Option<BehaviorSupport> {
        if parts.iter().all(|(_, s)| s.is_none()) {
            return None;
        }
        let mut out = BehaviorSupport::none(0);
        for (tokens, support) in parts {
            match support {
                Some(support) => out.extend(support),
                None => out.extend_none(*tokens),
            }
        }
        Some(out)
    }
}

/// One message's support as the wire gives it: the three fields, any of which may be absent (a fault unless all are).
pub(crate) struct WireSupport {
    pub(crate) offsets: Option<serde_json::Value>,
    pub(crate) ids: Option<serde_json::Value>,
    pub(crate) logprobs: Option<serde_json::Value>,
}

/// The supports of a record's messages, checked against the `behavior_logprobs` they sit beside (`supervised[i]`:
/// message `i` is a supervised assistant message) and, where the message carries them, its sampled `token_ids`.
pub(crate) fn check_behavior_support(wire: &[WireSupport], supervised: &[bool], behavior: &[Option<Vec<f32>>], token_ids: &[Option<SampledIds>]) -> Result<Vec<Option<BehaviorSupport>>, RecordError> {
    let refuse = |rule: RecordRule, i: usize, name: &str, why: String| {
        let field = format!("messages[{i}].{name}");
        RecordError::of(RecordFault::new(rule, &field), format!("{field}: {why}"))
    };
    let (offsets_name, ids_name, lps_name) = ("behavior_support_offsets", "behavior_support_ids", "behavior_support_logprobs");
    let mut out = Vec::with_capacity(wire.len());
    for (i, support) in wire.iter().enumerate() {
        let (offsets, ids, lps) = match (&support.offsets, &support.ids, &support.logprobs) {
            (None, None, None) => {
                out.push(None);
                continue;
            }
            (Some(o), Some(ids), Some(l)) => (o, ids, l),
            (o, ids, _) => {
                let missing = if o.is_none() { offsets_name } else if ids.is_none() { ids_name } else { lps_name };
                return Err(refuse(RecordRule::BehaviorSupport, i, missing, format!("\"{offsets_name}\", \"{ids_name}\" and \"{lps_name}\" come together, and \"{missing}\" is missing")));
            }
        };
        if !supervised[i] {
            return Err(refuse(RecordRule::BehaviorSupport, i, offsets_name, "only a supervised \"assistant\" message carries a sampler's support (its \"train\" is true)".to_string()));
        }
        let Some(behavior) = &behavior[i] else {
            return Err(refuse(RecordRule::BehaviorSupport, i, offsets_name, "a support sits beside the \"behavior_logprobs\" of the same answer, which this message does not carry".to_string()));
        };
        let tokens = behavior.len();
        let offsets: Vec<u64> = offsets
            .as_array()
            .and_then(|a| a.iter().map(serde_json::Value::as_u64).collect::<Option<Vec<u64>>>())
            .ok_or_else(|| refuse(RecordRule::BehaviorSupport, i, offsets_name, format!("\"{offsets_name}\" is not a list of non-negative integers")))?;
        if offsets.len() != tokens + 1 {
            return Err(refuse(RecordRule::BehaviorSupportCount, i, offsets_name, format!("\"{offsets_name}\" holds {} offsets, and \"behavior_logprobs\" {tokens} values; the offsets frame one support per token, so they are one more", offsets.len())));
        }
        let lp_list = lps.as_array().ok_or_else(|| refuse(RecordRule::BehaviorSupport, i, lps_name, format!("\"{lps_name}\" is not a list of numbers")))?;
        if lp_list.len() != tokens {
            return Err(refuse(RecordRule::BehaviorSupportCount, i, lps_name, format!("\"{lps_name}\" holds {} values, and \"behavior_logprobs\" {tokens}; one per token", lp_list.len())));
        }
        let id_list = ids.as_array().ok_or_else(|| refuse(RecordRule::BehaviorSupport, i, ids_name, format!("\"{ids_name}\" is not a list of ids")))?;
        if offsets[0] != 0 || offsets[tokens] != id_list.len() as u64 {
            return Err(refuse(RecordRule::BehaviorSupport, i, offsets_name, format!("the offsets run from {} to {}, and they frame the {} ids of \"{ids_name}\" from 0", offsets[0], offsets[tokens], id_list.len())));
        }
        let mut table = BehaviorSupport { offsets: offsets.clone(), ids: Vec::with_capacity(id_list.len()), logprobs: Vec::with_capacity(tokens), max: 0 };
        for t in 0..tokens {
            let bad = |name: &str, why: String| refuse(RecordRule::BehaviorSupport, i, name, format!("token {t}: {why}"));
            let (a, b) = (offsets[t], offsets[t + 1]);
            if b <= a {
                return Err(bad(offsets_name, format!("offsets {a} then {b}; every token's support holds at least the token itself")));
            }
            let width = (b - a) as usize;
            if width > MAX_BEHAVIOR_SUPPORT {
                return Err(bad(ids_name, format!("{width} ids; a support holds at most {MAX_BEHAVIOR_SUPPORT}")));
            }
            let start = table.ids.len();
            for v in &id_list[a as usize..b as usize] {
                let id = v.as_u64().and_then(|n| u32::try_from(n).ok()).filter(|&n| n != NO_TOP_ID).ok_or_else(|| bad(ids_name, format!("{v} is not a token id (a non-negative integer below {NO_TOP_ID})")))?;
                if table.ids[start..].contains(&id) {
                    return Err(bad(ids_name, format!("id {id} is listed twice")));
                }
                table.ids.push(id);
            }
            if let Some(token) = token_ids[i].as_ref().and_then(|s| s.answer.get(t)) {
                if !table.ids[start..].contains(token) {
                    return Err(bad(ids_name, format!("the sampled token {token} is not in its support; a sampler draws from its support")));
                }
            }
            let v = &lp_list[t];
            let number = v.as_f64().filter(|n| n.is_finite() && (*n as f32).is_finite()).ok_or_else(|| bad(lps_name, format!("{v} is not a finite number")))?;
            if number as f32 > LOGPROB_SLACK {
                return Err(bad(lps_name, format!("{number} is above 0; a log-probability is at most 0")));
            }
            let figure = (number as f32).min(0.0);
            if figure < behavior[t] - LOGPROB_SLACK {
                return Err(bad(lps_name, format!("{figure} is below the token's whole-row figure {} in \"behavior_logprobs\"; renormalising over a part of the row never lowers it", behavior[t])));
            }
            table.logprobs.push(figure);
            table.max = table.max.max(width);
        }
        out.push(Some(table));
    }
    Ok(out)
}

/// The sampling controls a decoder applied (`"sampler"`): what the record's tokens were drawn from.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerDecl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
}

impl SamplerDecl {
    /// A sampler as the wire gives it: `temperature` finite and above 0, `top_k` at least 1, `top_p` in (0, 1],
    /// `min_p` in [0, 1).
    pub(crate) fn from_wire(value: &serde_json::Value) -> Result<SamplerDecl, String> {
        let decl: SamplerDecl = serde_json::from_value(value.clone()).map_err(|e| format!("\"sampler\": {e}"))?;
        if decl.temperature.is_some_and(|t| !t.is_finite() || t <= 0.0) {
            return Err("\"sampler\": \"temperature\" is not a finite number above 0".to_string());
        }
        if decl.top_k == Some(0) {
            return Err("\"sampler\": \"top_k\" is 0; it is at least 1".to_string());
        }
        if decl.top_p.is_some_and(|p| !(p > 0.0 && p <= 1.0)) {
            return Err("\"sampler\": \"top_p\" is not in (0, 1]".to_string());
        }
        if decl.min_p.is_some_and(|p| !(0.0..1.0).contains(&p)) {
            return Err("\"sampler\": \"min_p\" is not in [0, 1)".to_string());
        }
        Ok(decl)
    }

    /// Whether the decoder drew from a truncation of the model's distribution (`top_k`, `top_p` below 1 or `min_p`
    /// above 0) rather than from its whole row at the temperature.
    pub fn truncates(&self) -> bool {
        self.top_k.is_some() || self.top_p.is_some_and(|p| p < 1.0) || self.min_p.is_some_and(|p| p > 0.0)
    }
}

/// Where a rollout's tokens came from: the oldest policy version among the requests behind them and the sampling
/// controls the decoder applied, as far as the record states them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Provenance {
    pub policy_version: Option<u64>,
    pub sampler: Option<SamplerDecl>,
}

/// A record's `policy_version` as the wire gives it: a non-negative integer that fits `u64`.
pub(crate) fn check_policy_version(value: &serde_json::Value) -> Result<u64, String> {
    value.as_u64().ok_or_else(|| format!("\"policy_version\" is {value}, which is not a non-negative integer that fits in 64 bits"))
}

/// One message's top-k as the wire gives it: the two lists, either of which may be absent (a fault when only one is).
pub(crate) struct WireTop {
    pub(crate) ids: Option<serde_json::Value>,
    pub(crate) logprobs: Option<serde_json::Value>,
}

/// Slack on the mass of one token's top-k probabilities, which are one row of a distribution: rounding of the
/// collector's own arithmetic.
const TOP_MASS_SLACK: f64 = 1e-3;

/// The top-k tables of a record's messages, checked against the `behavior_logprobs` they sit beside (`supervised[i]`:
/// message `i` is a supervised assistant message). One `K` for the whole record.
pub(crate) fn check_behavior_top(wire: &[WireTop], supervised: &[bool], behavior: &[Option<Vec<f32>>]) -> Result<Vec<Option<BehaviorTop>>, RecordError> {
    let refuse = |rule: RecordRule, i: usize, name: &str, why: String| {
        let field = format!("messages[{i}].{name}");
        RecordError::of(RecordFault::new(rule, &field), format!("{field}: {why}"))
    };
    let mut record_k: Option<usize> = None;
    let mut out = Vec::with_capacity(wire.len());
    for (i, top) in wire.iter().enumerate() {
        let (ids, lps) = match (&top.ids, &top.logprobs) {
            (None, None) => {
                out.push(None);
                continue;
            }
            (Some(ids), Some(lps)) => (ids, lps),
            (Some(_), None) => return Err(refuse(RecordRule::BehaviorTop, i, "behavior_top_logprobs", "\"behavior_top_ids\" comes with \"behavior_top_logprobs\"".to_string())),
            (None, Some(_)) => return Err(refuse(RecordRule::BehaviorTop, i, "behavior_top_ids", "\"behavior_top_logprobs\" comes with \"behavior_top_ids\"".to_string())),
        };
        if !supervised[i] {
            return Err(refuse(RecordRule::BehaviorTop, i, "behavior_top_ids", "only a supervised \"assistant\" message carries a top-k (its \"train\" is true)".to_string()));
        }
        let Some(behavior) = &behavior[i] else {
            return Err(refuse(RecordRule::BehaviorTop, i, "behavior_top_ids", "a top-k sits beside the \"behavior_logprobs\" of the same answer, which this message does not carry".to_string()));
        };
        let rows = |name: &str, value: &serde_json::Value| -> Result<Vec<serde_json::Value>, RecordError> {
            let list = value.as_array().ok_or_else(|| refuse(RecordRule::BehaviorTop, i, name, format!("\"{name}\" is not a list of lists")))?;
            if list.len() != behavior.len() {
                return Err(refuse(RecordRule::BehaviorTopCount, i, name, format!("\"{name}\" holds {} lists, and \"behavior_logprobs\" {} values; a top-k is one list per token", list.len(), behavior.len())));
            }
            Ok(list.clone())
        };
        let (id_rows, lp_rows) = (rows("behavior_top_ids", ids)?, rows("behavior_top_logprobs", lps)?);
        let mut table: Option<BehaviorTop> = None;
        for (t, (id_row, lp_row)) in id_rows.iter().zip(&lp_rows).enumerate() {
            let bad = |name: &str, why: String| refuse(RecordRule::BehaviorTop, i, name, format!("token {t}: {why}"));
            let id_list = id_row.as_array().ok_or_else(|| bad("behavior_top_ids", "not a list of ids".to_string()))?;
            let lp_list = lp_row.as_array().ok_or_else(|| bad("behavior_top_logprobs", "not a list of numbers".to_string()))?;
            if id_list.is_empty() {
                return Err(bad("behavior_top_ids", "no ids; a top-k holds at least one".to_string()));
            }
            let k = *record_k.get_or_insert(id_list.len());
            if id_list.len() != k {
                return Err(bad("behavior_top_ids", format!("{} ids, and the record's top-k holds {k} per token", id_list.len())));
            }
            if lp_list.len() != k {
                return Err(bad("behavior_top_logprobs", format!("{} values for {k} ids", lp_list.len())));
            }
            let table = table.get_or_insert_with(|| BehaviorTop::none(k, 0));
            let mut row_ids = Vec::with_capacity(k);
            for v in id_list {
                let id = v.as_u64().and_then(|n| u32::try_from(n).ok()).filter(|&n| n != NO_TOP_ID).ok_or_else(|| bad("behavior_top_ids", format!("{v} is not a token id (a non-negative integer below {NO_TOP_ID})")))?;
                if row_ids.contains(&id) {
                    return Err(bad("behavior_top_ids", format!("id {id} is listed twice")));
                }
                row_ids.push(id);
            }
            let mut mass = 0.0f64;
            let mut row_lps = Vec::with_capacity(k);
            for v in lp_list {
                let number = v.as_f64().filter(|n| n.is_finite() && (*n as f32).is_finite()).ok_or_else(|| bad("behavior_top_logprobs", format!("{v} is not a finite number")))?;
                if number as f32 > LOGPROB_SLACK {
                    return Err(bad("behavior_top_logprobs", format!("{number} is above 0; a log-probability is at most 0")));
                }
                let clamped = (number as f32).min(0.0);
                mass += f64::from(clamped).exp();
                row_lps.push(clamped);
            }
            if mass > 1.0 + TOP_MASS_SLACK {
                return Err(bad("behavior_top_logprobs", format!("the probabilities of its {k} entries sum to {mass:.4}; they are entries of one distribution, so at most 1")));
            }
            table.ids.extend(row_ids);
            table.logprobs.extend(row_lps);
        }
        out.push(table);
    }
    Ok(out)
}

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
    /// The sequence-level distillation signal of the rollout, when its answers carry a teacher model's
    /// `teacher_logprob`: the sum of those less the sum of the answers' `behavior_logprobs`, `log q(text) - log p(text)`.
    pub teacher_log_ratio: Option<f64>,
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
    /// One entry per message of the record: a teacher model's log-probability of each token of that answer
    /// (`"teacher_logprobs"`), on a supervised assistant message whose record carries them.
    pub teacher_logprobs: Vec<Option<Vec<f32>>>,
    /// The tokenizer the sampler encoded the rollout with (`sha256:<hex>`), when the record names it.
    pub tokenizer_digest: Option<String>,
    /// The record only sets its group's baseline (`"baseline_only": true`): its reward counts among the group's, and
    /// nothing of it is trained.
    pub baseline_only: bool,
    /// One entry per message of the record: the sampler's top-k at each token of that answer
    /// (`"behavior_top_ids"`, `"behavior_top_logprobs"`), on a supervised assistant message that carries them.
    pub behavior_top: Vec<Option<BehaviorTop>>,
    /// One entry per message of the record: the support a cut sampler drew each token of that answer from
    /// (`"behavior_support_*"`), on a supervised assistant message that carries one.
    pub behavior_support: Vec<Option<BehaviorSupport>>,
    /// The oldest policy version among the requests behind the record (`"policy_version"`).
    pub policy_version: Option<u64>,
    /// The sampling controls the decoder applied (`"sampler"`).
    pub sampler: Option<SamplerDecl>,
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

    /// Whether any message of the record carries a teacher model's per-token log-probabilities.
    pub fn has_teacher_logprobs(&self) -> bool {
        self.teacher_logprobs.iter().any(Option::is_some)
    }

    /// The rollout of a sample derived from this record's (a prefix, one answer, a packing): the same outcome, with
    /// the derived sample's per-message data (`sampled`) and teacher view.
    pub(crate) fn derived(&self, sampled: PerMessage, teacher: Option<Teacher>) -> Rollout {
        let PerMessage { behavior_logprobs, behavior_top, behavior_support, token_ids, teacher_logprobs } = sampled;
        Rollout {
            outcome: self.outcome.clone(),
            truncated: self.truncated,
            behavior_logprobs,
            token_ids,
            teacher,
            teacher_logprobs,
            tokenizer_digest: self.tokenizer_digest.clone(),
            baseline_only: self.baseline_only,
            behavior_top,
            behavior_support,
            policy_version: self.policy_version,
            sampler: self.sampler,
        }
    }

    /// Where the rollout's tokens came from.
    pub fn provenance(&self) -> Provenance {
        Provenance { policy_version: self.policy_version, sampler: self.sampler }
    }

    /// The per-message data of the first `len` messages.
    pub(crate) fn first_messages(&self, len: usize) -> PerMessage {
        PerMessage { behavior_logprobs: self.behavior_logprobs[..len].to_vec(), behavior_top: self.behavior_top[..len].to_vec(), behavior_support: self.behavior_support[..len].to_vec(), token_ids: self.token_ids[..len].to_vec(), teacher_logprobs: self.teacher_logprobs[..len].to_vec() }
    }
}

/// What a rollout records per message of its record, `Some` only on a supervised assistant message: the sampler's
/// log-probabilities and ids, and a teacher model's per-token log-probabilities. Empty for a sample that no longer
/// holds its messages (one rendered as asked or packed carries them on its segments).
#[derive(Clone, Debug, Default)]
pub(crate) struct PerMessage {
    pub(crate) behavior_logprobs: Vec<Option<Vec<f32>>>,
    pub(crate) behavior_top: Vec<Option<BehaviorTop>>,
    pub(crate) behavior_support: Vec<Option<BehaviorSupport>>,
    pub(crate) token_ids: Vec<Option<SampledIds>>,
    pub(crate) teacher_logprobs: Vec<Option<Vec<f32>>>,
}

impl PerMessage {
    /// Nothing per message.
    pub(crate) fn none() -> PerMessage {
        PerMessage::default()
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
pub(crate) fn check_logprobs(name: &str, values: &serde_json::Value) -> Result<Vec<f32>, String> {
    let list = values.as_array().ok_or_else(|| format!("\"{name}\" is not a list of numbers"))?;
    if list.is_empty() {
        return Err(format!("\"{name}\" is empty; omit it when none were recorded"));
    }
    list.iter()
        .enumerate()
        .map(|(k, v)| {
            let number = v.as_f64().ok_or_else(|| format!("\"{name}\"[{k}] is {v}, which is not a number"))?;
            let narrowed = number as f32;
            if !number.is_finite() || !narrowed.is_finite() {
                Err(format!("\"{name}\"[{k}] is {number}, which is not a finite number"))
            } else if narrowed > LOGPROB_SLACK {
                Err(format!("\"{name}\"[{k}] is {number}; a log-probability is at most 0"))
            } else {
                Ok(narrowed.min(0.0))
            }
        })
        .collect()
}

/// One log-probability as the wire gives it (`name` the field): a finite number at most zero.
pub(crate) fn check_logprob(name: &str, value: &serde_json::Value) -> Result<f64, String> {
    let number = value.as_f64().ok_or_else(|| format!("\"{name}\" is {value}, which is not a number"))?;
    if !number.is_finite() {
        return Err(format!("\"{name}\" is {number}, which is not a finite number"));
    }
    if number > f64::from(LOGPROB_SLACK) {
        return Err(format!("\"{name}\" is {number}; a log-probability is at most 0"));
    }
    Ok(number.min(0.0))
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
    /// The rollout's sequence-level distillation signal ([`Outcome::teacher_log_ratio`]), when it carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_log_ratio: Option<f64>,
}

/// What a prepared example says about its rollout; an imitation example has none.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExampleRollout {
    /// `None` for a rollout that trains on a teacher context alone.
    pub graded: Option<Graded>,
    /// The oldest policy version among the requests behind the example's record, when the record named it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_version: Option<u64>,
    /// The sampling controls the record's decoder applied, when the record stated them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<SamplerDecl>,
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
    /// `K` of the split's top-k sidecars (`<split>.btk.bin`, `<split>.btl.bin`), when it has them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    behavior_top_k: Option<usize>,
    /// The widest support of the split's support sidecars (`<split>.bso.bin`, `.bsi.bin`, `.bsl.bin`), when it has them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    behavior_support_max: Option<usize>,
}

fn names(dir: &Path, split: &str) -> [PathBuf; 8] {
    ["rollouts.json", "blp.bin", "trn.bin", "btk.bin", "btl.bin", "bso.bin", "bsi.bin", "bsl.bin"].map(|suffix| dir.join(format!("{split}.{suffix}")))
}

/// Whether `split` of the prepared dataset in `dir` holds rollouts (its rollout sidecar is there).
pub fn holds_rollouts(dir: &Path, split: &str) -> bool {
    names(dir, split)[0].exists()
}

/// Write a split's sidecars: `rollouts[i]` is example `i`'s rollout (`None`: an imitation example; `Some(None)`: a
/// rollout without a reward of its own), `behavior[t]` token `t`'s log-probability (NaN for none), `truncated[i]`
/// whether example `i`'s rollout did not reach an outcome, `provenance[i]` where it came from, `top` the sampler's top-k
/// per token of the stream and `support` the supports of a cut sampler (`None`: none recorded). Groups and rollouts are numbered in the order first seen.
/// Nothing is written, and a stale sidecar is removed, when no example is a rollout.
#[allow(clippy::too_many_arguments)]
pub fn write(dir: &Path, split: &str, rollouts: &[Option<Option<Outcome>>], baselines: &[Outcome], behavior: &[f32], truncated: &[bool], provenance: &[Provenance], top: Option<&BehaviorTop>, support: Option<&BehaviorSupport>) -> io::Result<()> {
    let [outcomes, blp, trn, btk, btl, bso, bsi, bsl] = names(dir, split);
    for stale in [&outcomes, &blp, &trn, &btk, &btl, &bso, &bsi, &bsl] {
        match std::fs::remove_file(stale) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    if rollouts.iter().all(Option::is_none) && baselines.is_empty() {
        return Ok(());
    }
    let (groups, examples, baselines) = number(rollouts, baselines, provenance).map_err(|why| io::Error::new(io::ErrorKind::InvalidInput, why))?;
    let top = top.filter(|t| t.logprobs.iter().any(|l| !l.is_nan()));
    let support = support.filter(|s| !s.ids.is_empty());
    let file = RolloutsFile { groups, examples, baselines, behavior_top_k: top.map(|t| t.k), behavior_support_max: support.map(|s| s.max) };
    std::fs::write(&outcomes, serde_json::to_vec(&file).map_err(io::Error::other)?)?;
    if behavior.iter().any(|b| !b.is_nan()) {
        binio::write_f32_bin(&blp, behavior)?;
    }
    if truncated.iter().any(|t| *t) {
        binio::write_mask_bin(&trn, truncated)?;
    }
    if let Some(top) = top {
        binio::write_u32_bin(&btk, &top.ids)?;
        binio::write_f32_bin(&btl, &top.logprobs)?;
    }
    if let Some(support) = support {
        binio::write_u64_bin(&bso, &support.offsets)?;
        binio::write_u32_bin(&bsi, &support.ids)?;
        binio::write_f32_bin(&bsl, &support.logprobs)?;
    }
    Ok(())
}

/// Groups and rollouts numbered in the order first seen: the group keys, each example's rollout and each baseline's
/// outcome in those numbers. Refuses a reward or a teacher log-ratio that is not finite.
type Numbered = (Vec<String>, Vec<Option<ExampleRollout>>, Vec<Graded>);

fn number(rollouts: &[Option<Option<Outcome>>], baselines: &[Outcome], provenance: &[Provenance]) -> Result<Numbered, String> {
    if let Some(o) = rollouts.iter().flatten().flatten().chain(baselines).find(|o| !o.reward.is_finite() || o.teacher_log_ratio.is_some_and(|r| !r.is_finite())) {
        return Err(format!("group {:?}: reward {} or teacher log-ratio {:?} is not a finite number", o.group, o.reward, o.teacher_log_ratio));
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
        Graded { group, member, reward: o.reward, teacher_log_ratio: o.teacher_log_ratio }
    };
    assert_eq!(rollouts.len(), provenance.len(), "chat_rollout: one provenance per example");
    let examples: Vec<Option<ExampleRollout>> = rollouts
        .iter()
        .zip(provenance)
        .map(|(rollout, from)| rollout.as_ref().map(|outcome| ExampleRollout { graded: outcome.as_ref().map(&mut grade), policy_version: from.policy_version, sampler: from.sampler }))
        .collect();
    let baselines: Vec<Graded> = baselines.iter().map(&mut grade).collect();
    Ok((groups, examples, baselines))
}

impl SplitRollouts {
    /// The rollout data of an encoded split held in memory, numbered as [`write`] numbers it: what [`read`] gives back
    /// for the split once written.
    pub fn of_split(split: &crate::chat::EncodedSplit) -> Result<SplitRollouts, String> {
        let (groups, examples, baselines) = number(&split.rollouts, &split.baselines, &split.provenance)?;
        Ok(SplitRollouts { groups, examples, behavior: split.behavior.clone(), truncated: split.truncated.clone(), baselines })
    }
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
    let [outcomes, blp, trn, ..] = names(dir, split);
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    if !outcomes.exists() {
        return Ok(None);
    }
    let file: RolloutsFile = serde_json::from_slice(&std::fs::read(&outcomes)?).map_err(|e| bad(format!("{}: {e}", outcomes.display())))?;
    if file.examples.len() != examples {
        return Err(bad(format!("{}: rollouts for {} examples, the split holds {examples}", outcomes.display(), file.examples.len())));
    }
    for (e, graded) in file.examples.iter().enumerate().filter_map(|(e, r)| r.and_then(|r| r.graded).map(|g| (e, g))).chain(file.baselines.iter().map(|g| (usize::MAX, *g))) {
        if graded.group as usize >= file.groups.len() || !graded.reward.is_finite() || graded.teacher_log_ratio.is_some_and(|r| !r.is_finite()) {
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

/// The sampler's top-k of each token of `split` in `dir`, checked against the `tokens` the split holds; `None` when the
/// split has none (an older dataset, or no record carried a top-k).
pub fn read_behavior_top(dir: &Path, split: &str, tokens: usize) -> io::Result<Option<BehaviorTop>> {
    let [outcomes, _, _, btk, btl, ..] = names(dir, split);
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    if !outcomes.exists() {
        return Ok(None);
    }
    let file: RolloutsFile = serde_json::from_slice(&std::fs::read(&outcomes)?).map_err(|e| bad(format!("{}: {e}", outcomes.display())))?;
    let Some(k) = file.behavior_top_k else {
        return Ok(None);
    };
    if k == 0 || !btk.exists() || !btl.exists() {
        return Err(bad(format!("{}: top-{k} recorded, and {} or {} is missing", outcomes.display(), btk.display(), btl.display())));
    }
    let (ids, logprobs) = (binio::read_u32_bin(&btk)?, binio::read_f32_bin(&btl)?);
    for (file, len) in [(&btk, ids.len()), (&btl, logprobs.len())] {
        if len != tokens * k {
            return Err(bad(format!("{}: {len} values, the split holds {tokens} tokens of top-{k}", file.display())));
        }
    }
    if let Some(t) = logprobs.iter().position(|v| v.is_infinite() || *v > LOGPROB_SLACK) {
        return Err(bad(format!("{}: entry {t} is {}, which is not a log-probability", btl.display(), logprobs[t])));
    }
    Ok(Some(BehaviorTop { k, ids, logprobs }))
}

/// A split's support sidecars ([`write`]), checked against the `tokens` the split holds: offsets that frame the ids,
/// no support wider than the split's recorded widest, a log-probability at most 0 where a support is and NaN where none
/// is. `None` when the split has none.
pub fn read_behavior_support(dir: &Path, split: &str, tokens: usize) -> io::Result<Option<BehaviorSupport>> {
    let [outcomes, _, _, _, _, bso, bsi, bsl] = names(dir, split);
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    if !outcomes.exists() {
        return Ok(None);
    }
    let file: RolloutsFile = serde_json::from_slice(&std::fs::read(&outcomes)?).map_err(|e| bad(format!("{}: {e}", outcomes.display())))?;
    let Some(max) = file.behavior_support_max else {
        return Ok(None);
    };
    if !(bso.exists() && bsi.exists() && bsl.exists()) {
        return Err(bad(format!("{}: a support recorded, and {}, {} or {} is missing", outcomes.display(), bso.display(), bsi.display(), bsl.display())));
    }
    let (offsets, ids, logprobs) = (binio::read_u64_bin(&bso)?, binio::read_u32_bin(&bsi)?, binio::read_f32_bin(&bsl)?);
    if offsets.len() != tokens + 1 || logprobs.len() != tokens {
        return Err(bad(format!("{} / {}: {} offsets and {} values, the split holds {tokens} tokens", bso.display(), bsl.display(), offsets.len(), logprobs.len())));
    }
    if offsets[0] != 0 || offsets[tokens] != ids.len() as u64 {
        return Err(bad(format!("{}: the offsets run from {} to {}, and {} holds {} ids", bso.display(), offsets[0], offsets[tokens], bsi.display(), ids.len())));
    }
    for t in 0..tokens {
        let (a, b) = (offsets[t], offsets[t + 1]);
        let lp = logprobs[t];
        let fine = if b > a { b - a <= max as u64 && lp.is_finite() && lp <= LOGPROB_SLACK } else { b == a && lp.is_nan() };
        if !fine {
            return Err(bad(format!("{}: token {t} has offsets {a}..{b} and log-probability {lp}; the split records supports of at most {max} ids", bso.display())));
        }
    }
    Ok(Some(BehaviorSupport { offsets, ids, logprobs, max }))
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

fn teacher_logprobs_name(dir: &Path, split: &str) -> PathBuf {
    dir.join(format!("{split}.tlp.bin"))
}

/// Write a split's per-token teacher log-probabilities (`teacher_logprobs[t]` token `t`'s, NaN where the record
/// carried none). Nothing is written, and a stale sidecar is removed, when no token carries one.
pub fn write_teacher_logprobs(dir: &Path, split: &str, teacher_logprobs: &[f32]) -> io::Result<()> {
    let file = teacher_logprobs_name(dir, split);
    match std::fs::remove_file(&file) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    if teacher_logprobs.iter().all(|t| t.is_nan()) {
        return Ok(());
    }
    binio::write_f32_bin(&file, teacher_logprobs)
}

/// The per-token teacher log-probabilities of `split` in `dir`, checked against the `tokens` the split holds (NaN
/// where none was recorded); `None` when no token carries one.
pub fn read_teacher_logprobs(dir: &Path, split: &str, tokens: usize) -> io::Result<Option<Vec<f32>>> {
    let file = teacher_logprobs_name(dir, split);
    if !file.exists() {
        return Ok(None);
    }
    let values = binio::read_f32_bin(&file)?;
    if values.len() != tokens {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: teacher log-probabilities for {} tokens, the split holds {tokens}", file.display(), values.len())));
    }
    if let Some(t) = values.iter().position(|v| v.is_infinite() || *v > LOGPROB_SLACK) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: token {t} has teacher log-probability {}, which is not a log-probability", file.display(), values[t])));
    }
    Ok(Some(values))
}
