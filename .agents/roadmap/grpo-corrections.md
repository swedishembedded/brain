# GRPO corrections - roadmap

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
