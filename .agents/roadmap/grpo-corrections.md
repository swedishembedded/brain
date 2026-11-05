# GRPO corrections - roadmap

Scope: Remaining corrections to the GRPO objective (`crates/rl`).

The objective in `crates/rl/src/objective/grpo.rs` now trains the distribution
that sampled, never steps on an empty batch and normalises per token (knowledge
290). What is left:

- [ ] Batched, KV-cached rollouts. A rollout still re-prefills the whole
      sequence per token through `Model::logits_all` (O(T^2), a different
      forward from training). A paged rollout engine exists for `qwen3` serving;
      the trainer's behaviour log-probabilities are recomputed by the training
      forward either way, so only speed is at stake.
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
- [ ] The start-of-run and reference passes are one forward per example before the first step; they could be skipped
      for the examples whose log-probabilities are all recorded and whose KL is not wanted (already done) or cached on
      disk across rounds that continue the same adapter.
- [ ] Per-sequence normalisation (`LossAgg::SeqMean`, the default) needs a batch of one sequence; long agent calls
      weigh less per token than short ones under it (`TokenSum` is the alternative, see knowledge 290).

