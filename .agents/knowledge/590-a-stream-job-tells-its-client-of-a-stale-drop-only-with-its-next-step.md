<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 590. A stream job tells its client of a stale drop only with its next step, and of its policy only after it loads

**Lesson:** two of the streaming job's frames report facts later than they happen, and a real model on a card made
both visible. (1) `RolloutFeed::take_step` drops stale groups from the head of its queue while it waits for
records, but their `dropped` frame goes out only with the next step (frames are emitted from the step callback).
A client that counts held groups against a one-step token allowance (a synchronous run, `max_staleness 0`)
therefore waits for the drop while the job waits for records: a deadlock. (2) The `stream` frame comes before the
base is loaded and the run's first version is published, so a rollout drawn right after it is refused "model not
found". On the CPU toy the load is instant; on a card it took seconds. Both are fixed on the client side (the orchestrator:
the controller stops counting groups a publication makes stale, and the client polls `/v1/models` for the
publish id). A brain-side fix would emit `dropped` when the drop happens and send the first `published` frame as
soon as that version is routable, which needs a frame path outside the step callback. It is open in the roadmap.

## Evidence (2026-11-25, lane agrpo-real, brain 760c04d3b)

Qwen3-0.6B, `max_staleness 0`, `step_tokens 1024`: after step 2's publication the job held four groups drawn at
version 2 (1216 tokens) and stalled for 6 minutes; gdb showed the trainer in `RolloutFeed::take_step` and the
input thread idle in `serve_inputs`. Before the client waited for its policy, the first run on a card had 16 of 16
rollouts refused `model_not_found` within its first second.
