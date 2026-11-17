<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!-- Swedish Embedded AB implements solutions for the data contracts between the systems that collect agent
     experience and the trainers that learn from it, for its clients. If your team needs expertise in training-data
     pipelines for language models then you can procure our services by sending an email to
     info@swedishembedded.com. -->

# Training records, contract version 1 (`training-records-v1`)

These files are the contract brain's trainer holds a producer of training data to, as examples: what it accepts and
what it refuses, and why. brain's test `crates/sdk/tests/it/training_records.rs` checks every file against
`check_dataset` (the `check_dataset` action of `brain/train` runs the same check), so a change to what brain accepts
changes these files and, when an accepted record stops being accepted, the version. A producer copies the directory
and checks its own output against the same rules.

The record formats themselves are documented in brain's SDK reference (policy-gradient fine-tuning on collected
rollouts) and in its guide to training as a served job.

## Files

| file | format | what it shows |
|---|---|---|
| `model/` | tokenizer | the tokenizer (letters `a` to `w` as ids 0 to 22, `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645) and the chat template every `token_ids` here is computed with; `check_dataset` reads nothing else of a model |
| `chat.jsonl` | `generic-messages-v2` | supervised records: per-message `train` masks, a record with two supervised answers, an assistant turn that is context |
| `chat_rollout.jsonl` | `generic-messages-v2` with rollout fields | `advantage` on the record, `token_ids` and `behavior_logprobs` on each supervised answer, a `truncated` rollout, and an imitation record (no `advantage`) in the same file |
| `preference.jsonl` | `generic-preference-v1` | a prompt and its `chosen` and `rejected` answers |
| `invalid/behavior_logprob_count.jsonl` | refused | `behavior_logprobs` holds three values for an answer of four tokens |
| `invalid/token_ids_mismatch.jsonl` | refused | `token_ids` that are not the tokens the answer's text renders to |
| `invalid/missing_advantage.jsonl` | refused | `behavior_logprobs` on a record with no `advantage` |
| `manifest.json` | expected verdicts | per file: the format, the record count, the valid count and each refusal as `{line, rule, field}` |

## The rules of version 1

- A supervised answer is the answer as the chat template renders it after its prompt: its text and the end-of-turn
  marker that closes it. The whitespace a template writes after the marker is context.
- `behavior_logprobs` holds one value per token of that answer: the natural log of the probability of the token given
  everything before it, under the model's full, untruncated distribution at the sampling temperature (the figures a
  served reply's `policy_trace` returns). Not the probability a sampler reports after top-k, top-p or min-p, and not
  the `0` a greedy sampler reports. Values are finite and at most 0. The training run is given the sampling
  temperature (`temperature`).
- `token_ids`, when present, are the ids the sampler generated for the answer, end-of-turn marker included. The trainer
  renders the answer's text again and refuses the record when the ids differ: the text must round-trip to them.
- `behavior_logprobs`, `token_ids` and `truncated` belong to a rollout: a record carrying any of them carries an
  `advantage` (or a `teacher_context`).
- Every other field and message rule (roles, tool calls, `deny_unknown_fields`) is that of `generic-messages-v2`.

A refusal of one of these rules is reported typed: `rule` (`behavior_logprob_count`, `token_ids_mismatch`,
`missing_advantage`) and `field` (a JSON path into the record, such as `messages[1].behavior_logprobs`), beside the
message. Other refusals carry the message alone.
