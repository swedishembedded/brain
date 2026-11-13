# GRPO corrections - roadmap

Scope: Remaining corrections to the GRPO objective (`crates/rl`).

The objective in `crates/rl/src/objective/grpo.rs` now trains the distribution
that sampled, never steps on an empty batch and normalises per token (knowledge
290). What is left:

- [ ] Batched, KV-cached rollouts. The model's own rollout still re-prefills the whole
      sequence per token through `Model::logits_all` (O(T^2), a different
      forward from training). The serving path exists: `model::rollout::SchedulerRollout` submits a group to the
      generic scheduler over any paged engine (shared prompt prefill, batched decode, per-token log-probabilities and
      the policy version on each completion; gated against the dense model on a tiny `qwen3`), and `qwen3`'s
      `EngineRollout` builds on it and refits by publishing the trainer's LoRA factors to the engine's adapter bank
      as the next policy version (no engine rebuild, the previous version's cached prompts purged; a full fine-tune
      still rebuilds). Not yet the default of the SDK's improve cycle, and not yet measured on real
      weights. The trainer's behaviour log-probabilities are recomputed by the training forward either way, so only
      speed is at stake.
- [ ] An entropy term (or an entropy floor) next to the KL term; neither is in
      the loss, and the KL to the adapter-disabled reference is logged only.
- [ ] The documented 12-cycle study (`crates/rl/tests/continual_study.rs`) was
      recorded with the legacy objective; its numbers say nothing about the new
      one until it is rerun. A reduced A/B on tiny models is recorded in
      knowledge 292 when it exists.
- [ ] `Grpo` trains one row per forward (`b = 1`). Rows of one group could share
      a batch once `logits_all` is no longer the sampling bottleneck.

## Policy gradient on external rollouts

`ChatFineTune::policy_gradient` (`crates/rl/src/objective/external.rs`) trains the clipped GRPO/PPO objective on chat
records an orchestrator collected outside brain (a record per model call, the advantage of its rollout in
`"advantage"`, the sampler's log-probabilities in the assistant message's `"behavior_logprobs"`). It runs through the
same `pg_batch::backward_weighted` step `Grpo` uses, so the clipped surrogate and the k3 KL exist once. Format, objective
and call: `docs/reference/sdk.md`. Gated on the CPU with a tiny `qwen35` LoRA model: finite differences of the adapters
on text rows (positive, negative, zero and no advantage, KL on) and on rows holding an image
(`crates/rl/tests/it/external_policy_gradient_gradcheck.rs`); the advantage's direction, the clip per side, the KL against
its definition, the on-policy first step and the supervised parity of a record without an advantage
(`external_policy_gradient.rs`); the SDK run end to end on a qwen35 checkpoint, continuing an adapter
(`crates/sdk/tests/it/chat_finetune_policy_gradient.rs`).

- [x] On-policy self-distillation with privileged information, sampled-token version: a record's `"teacher_context"`
      (privileged messages inserted for the teacher only) gives every supervised token
      `A_t = log q_teacher(y_t | privileged context, y_<t) - old_t`, the frozen base or start-of-run adapter as teacher,
      through the same `pg_batch::backward_weighted` step. Named guards: `advantage_bound` (clip `A_t`),
      `sequence_advantage_cap` (scale a sequence down when its summed `|A_t|` is over it), the k3 KL stays; per step the
      mean, largest magnitude and clipped fraction of the token advantages are reported. Gated on the CPU: spec tests
      (a preferred token rises, a rejected one falls, an identical view gives zero, guards, refusals by name) in
      `crates/rl/tests/it/self_distill.rs`, a per-token-advantage gradcheck in `external_policy_gradient_gradcheck.rs`,
      the record format and the teacher sidecar in `crates/data/tests/it/chat_teacher_context.rs`, and an SDK run in
      `chat_finetune_policy_gradient.rs`. Not run on real weights.
- [ ] Full-distribution reverse KL for self-distillation (follow-up). Instead of `log q - log p` at the sampled token,
      take the teacher's top-K tokens at each position and minimise `sum_v p(v) (log p(v) - log q(v))` over the
      student's own top-K, reusing the `DistillTopK` weighted-CE trick with the direction reversed (the weight of
      token `v` is `p(v) (log q(v) - log p(v) + KL-baseline)`, so it still lands in `backward_weighted`). It needs the
      teacher's top-K log-probabilities stored per position (K ids and values) in the sidecar instead of one scalar.
- [ ] Self-distillation on a real 27B (bf16 base, rank-16 adapter): confirm teacher advantages are sane (mean near 0,
      few tokens at the bound), that the sequence cap rarely binds, and that held-out pass rates do not collapse.
- [ ] A real-weight check, on a card, which has not run: Qwen3.8-27B-FP8 as a bf16 base with a rank-16 adapter, the
      records of a few rollout groups with and without images and with the sampler's log-probabilities. It should
      (1) confirm that the first step's ratios are 1 within bf16 noise when no log-probabilities were recorded and that
      recorded ones from the serving engine give ratios near 1 at step 0 (the size of the serve/train log-probability gap
      is what decides whether `clip_eps` 0.2 is loose enough for them), (2) show the mean log-probability of positive-
      and negative-advantage actions moving apart over 20-40 steps on `score_tokens_qwen35`, and (3) record the
      peak device memory and the seconds of the start-of-run pass per 1000 tokens.
- [ ] `brain qwen35 finetune` (the CLI) reads no advantage sidecar; the SDK is the one entry point.
- [ ] Only `qwen35` bases; `qwen3` is refused by name. Held-out and monitoring selection are refused for these runs.
- [x] Lazy, cached preparation: the start-of-run pass, the reference (KL) pass and the teacher pass used to be one
      forward per example over the whole dataset before step 1 (a 5890-record, 265 MB agent-call set sat over an hour
      there while 60 steps x 4 records read 240 of them; the earlier CAD run measured ~47 min for 1.19M tokens). Now an
      example is scored the first time a step draws it and kept for the rest of the run in a file beside the prepared
      split (`score_cache.rs`: a slot per example in memory, the vectors on disk, removed with the run); an example no
      step draws is never scored, one drawn again is read back. On-policy is preserved: behaviour log-probabilities
      are the START policy's, measured under a frozen copy of the start adapter's tensors put in while scoring (the
      adapters switched off for a fresh adapter, which is exactly the base), the live adapter written back
      afterwards; no swap is made while the adapter is still the start one, and a resumed run (`begin` with draws done)
      swaps from its first step because `prepare` captured the start before the checkpoint restored. The reference and
      the teacher are measured under the policies they name, one pass serving both when they are the start policy.
      Gated on the CPU in `crates/rl/tests/it/lazy_scoring.rs`: never-drawn never scored, redrawn scored once (counts
      and bytes written), behaviour/reference/teacher advantage equal an independent pass under the start policy bit
      for bit after the adapter moved (continued and fresh adapter, resumed run), the live adapter restored exactly,
      memory a slot per example. The existing spec tests and gradchecks run unchanged. Not yet measured on real weights:
      seconds of the first-draw scoring per step on the 27B (reported per step as `scoring_seconds`).
- [ ] The score file is run-local. Reusing it across rounds that continue the same adapter would need the base's and
      the adapter's identity in its name.
- [ ] Per-sequence normalisation (`LossAgg::SeqMean`, the default) needs a batch of one sequence; long agent calls
      weigh less per token than short ones under it (`TokenSum` is the alternative, see knowledge 290).


## External policy gradient on a toy: what the measurements say (2026-11-12)

On a toy `qwen3` LoRA model (`brain::toy`, a three-letter cipher, held-out exact match) the objective behaves as
specified: the first step's importance ratio is exactly 1 under the start-of-run behaviour, and the finite-difference
gradchecks hold. What decided whether the two-sided (default, standardised) advantage improved the held-out score
was the data, not the loss:

- negative advantages alone destroy the held-out score (loss 2.93 to 5.9) and positive ones alone improve it, so a
  run that revisits a fixed set of prompts lets the negatives memorise those prompts and the score falls after four
  rounds; a fresh slice of prompts each round improves it every round with the same objective and settings (0.39 to
  0.53 over six rounds);
- larger groups on the same prompts help little; a higher learning rate or a wider clip band makes it worse.

Open here: whether the same holds on a real model (it was not run), and an optional negative-advantage weight if a
reused-prompt regime must be supported.
