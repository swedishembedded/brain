<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 531. Served Qwen3 prompts carry a reasoning directive the trainer never renders, so no recorded prompt matches

**Lesson:** an importance ratio is exact only in the context the behaviour log-probabilities were taken in, and the
served prompt of a Qwen3-flavor model is not the checkpoint template's rendering: `qwen3::chat::render_prompt` defaults
`reasoning_effort` to `xhigh` whatever the flavor, and `qwen_chat::render` injects the xhigh instruction into the system
turn even under the Qwen3 flavor, whose upstream template (Qwen3-4B's own) has no such text. The trainer renders the
checkpoint's Jinja template and never writes it. Check the served prompt ids against the trainer's rendering before
trusting recorded log-probabilities; brain's contract now does (`prompt_token_ids`, rule `prompt_token_ids_mismatch`).

## What happened

2026-11-19, on real practice rollouts of an agent (Qwen3-4B + a pilot adapter, served by brain e9044bbeb through an
examiner's gateway, the agent runtime recording each step's policy trace: prompt ids, generated ids and policy
log-probabilities):

- Rendering the request the agent runtime sent through the Qwen3-4B template reproduced the recorded prompt ids for 0 of 415 steps
  (48 trials, support and compiled); with the xhigh instruction prepended to the system turn and the tool schemas'
  keys sorted (brain's `json_py::dumps`, which the trainer's rendering of the orchestrator's records matches) for 364 of 415.
- Through the trainer itself (`answers_as_asked` + `encode_rollout` on the orchestrator's policy-gradient records with the
  new `prompt_token_ids`), every traced record was refused at position 3 of its prompt: the sampler was given
  `Reasoning ...` (id 25139) where the template renders `You ...` (id 2610), 37 ids of instruction before the
  system prompt.

## What changed

- Contract v2 (b4ffeb71e): a supervised answer trains its sampled `token_ids` (admitted when they read back, by the
  template's reply format, as the message: 122 of 122 traced answers of those records; 119 of them also round-trip
  through a re-rendering), and `prompt_token_ids` must equal the trainer's context or the record is refused.
- Not changed: the serving default. Making `reasoning_effort` default to no instruction under the Qwen3 flavor (only
  Qwen3.8's template reads it) changes the served prompt of every Qwen3 and Qwen3.5-MoE model, so it is a campaign
  decision taken with a fresh baseline, not a silent fix; until then a collector that wants exact ratios has
  to request `reasoning_effort: medium` (no instruction), where its client can pass it.
