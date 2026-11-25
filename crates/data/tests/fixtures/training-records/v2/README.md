<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!-- Swedish Embedded AB implements solutions for the data contracts between the systems that collect agent
     experience and the trainers that learn from it, for its clients. If your team needs expertise in training-data
     pipelines for language models then you can procure our services by sending an email to
     info@swedishembedded.com. -->

# Training records, contract version 2 (`training-records-v2`)

These files are the contract brain's trainer holds a producer of training data to, as examples: what it accepts and
what it refuses, and why. brain's test `crates/sdk/tests/it/training_records.rs` checks every file against
`check_dataset` (the `check_dataset` action of `brain/train` runs the same check), so a change to what brain accepts
changes these files and, when an accepted record stops being accepted, the version. A producer copies the directory
and checks its own output against the same rules. The set's digest (`data::chat_rollout::contract_digest`: SHA-256 of
`<path>\0<sha256 of the file>\n` per file but this README, sorted by path) is `CONTRACT_DIGEST` and is what a server
reports as `server.training_records.digest` (`GET /v1/capabilities`), so a producer compares it with its copy's.

Version 2 replaces version 1's precomputed `advantage` with the rollout's `reward` and its `group`: the trainer
computes each rollout's advantage from the rewards of its group, by the run's `advantage_norm` (`mean-centred`,
`z-score` or `leave-one-out`). A version 1 record that carries `advantage` is refused, naming the field.

The record formats themselves are documented in brain's SDK reference (policy-gradient fine-tuning on collected
rollouts) and in its guide to training as a served job.

## Files

| file | format | what it shows |
|---|---|---|
| `model/` | tokenizer | the tokenizer (letters `a` to `w` as ids 0 to 22, `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645) and the chat template every `token_ids` here is computed with; `check_dataset` reads nothing else of a model |
| `chat.jsonl` | `generic-messages-v2` | supervised records: per-message `train` masks, a record with two supervised answers, an assistant turn that is context |
| `chat_rollout.jsonl` | `generic-messages-v2` with rollout fields | group `cab`: two rollouts named by `rollout`, one of them two records (two calls) that share its reward, line 1's answer carrying the `prompt_token_ids` it was sampled after, each record naming the `tokenizer_digest`; line 8 a third rollout of group `cab` with the sampler's top-2 per token, a `policy_version` and a `sampler` (the sha256 of `model/tokenizer.json`); line 9 a fourth rollout sampled with `top_k` 2, carrying each token's support (the two ids the sampler kept, the token among them) and its log-probability within it; group `fed`: three records that are a rollout each, one of them `truncated` (it takes no part in the group); an imitation record (no `reward`) in the same file |
| `preference.jsonl` | `generic-preference-v1` | a prompt and its `chosen` and `rejected` answers; line 3's `chosen` is a continuation (a tool call, its result, the answer) against a single `rejected` turn |
| `feedback.jsonl` | `generic-feedback-v1` | a desirable and an undesirable judged `completion`, and a desirable continuation |
| `invalid/behavior_logprob_count.jsonl` | line 1 refused | `behavior_logprobs` holds three values for an answer of four tokens |
| `invalid/token_ids_mismatch.jsonl` | line 1 refused | `token_ids` that are not the tokens the answer's text renders to (this template's replies are not read back, so only exact ids belong to the answer) |
| `invalid/prompt_token_ids_mismatch.jsonl` | line 1 refused | `prompt_token_ids` that are not the context the template renders before the answer (they differ at position 5) |
| `invalid/behavior_top_count.jsonl` | line 1 refused | `behavior_top_ids` holds three lists for an answer of four tokens |
| `invalid/behavior_top.jsonl` | line 1 refused | a token whose top-k lists one id twice |
| `invalid/behavior_support_count.jsonl` | line 1 refused | `behavior_support_offsets` frames three supports for an answer of four tokens |
| `invalid/behavior_support.jsonl` | line 1 refused | a sampled token (`token_ids`) that is not in its support |
| `invalid/missing_reward.jsonl` | refused | `behavior_logprobs` on a record with no `reward` |
| `invalid/missing_group.jsonl` | refused | a `reward` without a `group` |
| `invalid/advantage_removed.jsonl` | refused | version 1's `advantage` |
| `invalid/rollout_mismatch.jsonl` | line 2 refused | a second record of rollout `cab-0` with another reward |
| `invalid/group_of_one.jsonl` | lines 1 and 4 refused | group `cab` holds one rollout; group `ace` holds one that reached an outcome beside a truncated one |
| `invalid/baseline_trains.jsonl` | line 3 refused | lines 1 and 2: a trained rollout and a `baseline_only` record of its group (a pair: the baseline counts); line 3 marks a supervised answer `baseline_only` |
| `invalid/tokenizer_mismatch.jsonl` | line 2 refused | a record naming another tokenizer than the base's own (line 1 names it) |
| `manifest.json` | expected verdicts | per file: the format, the record count, the valid count and each refusal as `{line, rule, field}` |

## The rules of version 2

- A supervised answer is the answer as the chat template renders it after its prompt: its text and the end-of-turn
  marker that closes it. The whitespace a template writes after the marker is context.
- `reward` (a finite number) and `group` (a key of 1 to 512 bytes, no control characters) come together. A record
  without them is imitated, unless it carries a `teacher_context` (self-distillation; its scalar advantage is 0).
- `rollout` (a key, with `group`) names which rollout of the group the record is one call of: the records of one
  rollout carry the same `reward` and `truncated`, and the rollout counts once in its group. Without `rollout` the
  record is a rollout of its own.
- A group holds at least two rollouts that did not end `truncated`: a group-relative advantage compares a rollout with
  the others of its group. A truncated rollout is never trained on and takes no part in its group's advantages.
- `behavior_logprobs` holds one value per token of that answer: the natural log of the probability of the token given
  everything before it, under the model's full, untruncated distribution at the sampling temperature (the figures a
  served reply's `policy_trace` returns). Not the probability a sampler reports after top-k, top-p or min-p, and not
  the `0` a greedy sampler reports. Values are finite and at most 0. The training run is given the sampling
  temperature (`temperature`).
- `token_ids`, when present, are the ids the sampler generated for the answer, end-of-turn marker included, and they
  are what the trainer trains (with `behavior_logprobs` one value per id), never a re-encoding of the answer's text.
  They are admitted when the template renders the answer to exactly them, or when they read back by the template's
  reply format (a think block, `<tool_call>` calls) as the same answer: same reasoning and content up to surrounding
  whitespace, same calls with arguments equal as JSON values. Otherwise the record is refused (`token_ids_mismatch`).
- `prompt_token_ids`, only beside `token_ids`: the ids the sampler was given before the answer. The context the trainer
  renders before the answer must encode to exactly these, or the record is refused (`prompt_token_ids_mismatch`).
- `tokenizer_digest` (`sha256:` and 64 lowercase hex digits, as a `policy_trace` names it), when present, is the
  digest of the trained base's own `tokenizer.json` (`tokenizer_mismatch`, naming both), and where the base's is not
  known, the same on every record of one run that names one (`mixed_tokenizer_digest`).
- `behavior_top_ids` and `behavior_top_logprobs` (a supervised message, together, beside `behavior_logprobs`): the
  sampler's top-K per token, one inner list each per token of the answer (`behavior_top_count` when the count differs
  from `behavior_logprobs`). `K` is at least 1 and the same on every token of the record, the ids of a token are
  distinct, the log-probabilities finite and at most 0 and the probabilities of one token sum to at most 1 (they are
  whole-row log-probabilities at the sampling temperature); the sampled token need not be listed (`behavior_top`).
  A prepared split holds one `K` and keeps them as `btk.bin`/`btl.bin`.
- `behavior_support_offsets`, `behavior_support_ids` and `behavior_support_logprobs` (a supervised message, all three,
  beside `behavior_logprobs`): the support a cut sampler (`top_k`, `top_p`, `min_p`) drew each token from, and the
  token's log-probability within it, so a trainer can renormalise its own figures over the same support. Token `t`'s
  support is `behavior_support_ids[offsets[t]..offsets[t + 1]]`; the offsets are one more than the answer's tokens,
  start at 0, rise at every token and end at the number of ids (`behavior_support_count` when a count differs). A
  token's ids are distinct, at most 1024, and hold the token where `token_ids` are given; each log-probability is
  finite, at most 0 and at least the token's `behavior_logprobs` value (`behavior_support`). A prepared split keeps
  them as `bso.bin`/`bsi.bin`/`bsl.bin`.
- `policy_version` (a non-negative integer) and `sampler` (`temperature` above 0, `top_k` at least 1, `top_p` in
  (0, 1], `min_p` in [0, 1)) belong to a rollout; the prepared split keeps them per example.
- `baseline_only: true` (with `reward` and `group`) marks a rollout that sets its group's baseline and is never
  trained: no message `"train": true`, no `behavior_logprobs`, `token_ids`, `teacher_context` or `truncated`
  (`baseline_trains`).
- `loss_weight` (message, optional, default 1.0): a finite number above 0 on a supervised message (`"train": true`),
  the weight of its tokens in the loss; on any other message, out of range, or on a rollout record it is refused
  (`loss_weight`).
- `behavior_logprobs`, `behavior_top_ids`, `token_ids`, `truncated`, `rollout`, `tokenizer_digest`, `policy_version` and `sampler` belong to a rollout: a record
  carrying any of them carries a `reward` and `group` (or a `teacher_context`).
- A preference candidate (`chosen`, `rejected`) and a feedback `completion` is one assistant turn or a continuation: a
  non-empty array of turns that starts and ends with an assistant turn. Its assistant turns are supervised, every
  other turn (a tool result, a user turn) is context, and tool calls pair with their results over the prompt and the
  continuation together. Two candidates that are the same turns in order are refused.
- Every other field and message rule (roles, tool calls, `deny_unknown_fields`) is that of `generic-messages-v2`.

A refusal of one of these rules is reported typed: `rule` and `field` (a JSON path into the record, such as
`messages[1].behavior_logprobs`), beside the message. The rules: `behavior_logprob_count`, `behavior_top_count`, `behavior_top`, `token_ids_mismatch`,
`missing_reward`, `missing_group`, `advantage_removed`, `baseline_trains`, `loss_weight`, `rollout_mismatch`, `group_of_one`,
`mixed_tokenizer_digest`, `tokenizer_mismatch`. A record that breaks a rule of its own is refused for that; the last
four hold between records, over the whole file
(`check_dataset`) or every set a run trains on together (the dataset and its replay sets). Other refusals carry the
message alone.
