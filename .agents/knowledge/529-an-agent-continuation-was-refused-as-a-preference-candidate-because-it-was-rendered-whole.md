<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 529. An agent continuation was refused as a preference candidate because it was rendered whole

**Lesson:** A trainer's records are rendered the way packed SFT renders a dialogue whenever they hold more than one
answer: each answer as the model was asked for it, through the same function. A second rendering path for the same
shape gives the same template a different chance to break.

## What happened

Preference and feedback candidates became multi-turn continuations (15f6f67ee) so that agent rollouts could train DPO
and KTO. The first real datasets were 175 DPO pairs and 745 KTO examples from Qwen3-4B G=8 practice rollouts. Brain's
dataset check accepted 4 of the pairs and 34 of the examples. Every other record was refused: "message N is not
prefix-stable under this template".

Two things caused it:

- `PreferenceSample::encode_with` and `FeedbackSample::encode_with` rendered each candidate whole, through
  `ChatSample::encode_with`'s message boundaries.
- Qwen3's template drops the reasoning of every assistant turn before the last user turn, and an agent run carries
  user turns mid-run, such as an agent runtime's time-left notice. So an earlier answer rendered in isolation, with its
  reasoning, differs from the same answer rendered in full context, without it.

Packed SFT never met this: `Asked::examples_of` packs any record of several answers (`ChatSample::answers_packed`).

## Fix

`ChatSample::encode_candidate` is the one rendering for a candidate:

- A continuation is packed and trained the way packed SFT trains it. A continuation means more than one supervised
  turn, or assistant turns in its context.
- A single answer after a plain prompt keeps the whole-record rendering, so `keep_reasoning` still applies to it.

`PreferenceSample::encode_with`, `FeedbackSample::encode_with` and the KTO trainer's mismatched-pair render
(`sdk/src/feedback.rs`) all use it.

On the same real datasets, every DPO pair that parses (168) and all 745 KTO examples now encode. The other 7 pairs
opened with a user turn: a defect of the data producer's preference view, fixed on its side.

Test: `data` `chat_dataset_generic::an_agent_continuation_with_reasoning_trains_each_answer_as_asked_in_preference_and_feedback`
uses the real Qwen3-4B template. It checks a two-answer continuation with reasoning and a mid-run user turn: both
sides of a pair and a feedback example encode, and they supervise exactly packed SFT's tokens.
