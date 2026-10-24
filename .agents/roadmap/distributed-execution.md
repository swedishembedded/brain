<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# distributed-execution - roadmap

Scope: Running one training job as ranks across machines - the job plan every rank agrees on, the process group it joins, the transport's failure semantics, and the tests that start a cluster as real processes (`crates/model`: `job`, `collective`, `netcollective`, `distributed`, `grid`).

## Delivered

- [x] **A failed collective is an error, not a panic.** `FlatAdam::step`,
      `DdpOptimizer::step`, `fed_avg_flat` and `federated_average` return
      `CollectiveResult`; a lost peer or a rejected payload leaves the
      replica's weights and optimiser state untouched, so a driver can stop at
      the last committed step instead of losing the process. Tests:
      `crates/model/tests/collective_errors.rs`.
- [x] **A job plan every rank validates.** `model::job::JobPlan` names the job,
      the world size, the tp/pp/dp grid and the coordinator; it is a value the
      launcher writes, never something brain reads from a scheduler. Every
      rank announces the plan's exact bytes when it joins, and a rank started
      with a different plan is refused with the mismatch rather than joining
      someone else's collective. `JobContext` carries a rank and derives its
      grid coordinate. Tests: the unit tests in `crates/model/src/job.rs`.
- [x] **The group fails together.** The TCP transport has an abort frame: a
      coordinator that rejects a payload or loses a peer sends every remaining
      rank the reason, and each op carries a timeout from the plan, so no rank
      waits forever for a payload that will not come. Join announcements and
      aborts are length-bounded.
- [x] **A cluster can be simulated without a scheduler.** `crates/model/tests/distributed_job.rs`
      starts one OS process per rank on loopback, with the same plan file, and
      covers: a three-rank reduction across processes, a rank that exits after
      joining (its peers get an error inside the plan's bound instead of
      hanging), and a rank started with another plan (refused before any
      tensor moves).

- [x] **A committed checkpoint generation.** `CheckpointCommit` names the
      job, the plan identity, the global step, the world and every file with
      its length and `sha256`. Rank 0 writes it once, atomically, after the
      files are durable; a crashed commit leaves the previous generation the
      newest, a reader verifies every file against the manifest before
      resuming, and a manifest from another plan or world size is refused.
      Tests: the `job::tests` commit cases (a generation, a crashed commit, a
      tampered file).

- [x] **Gradients are reduced across the group before the model's own
      optimizer runs.** `Model::write_grad` exposes the replacement (Gpt's
      paramstore does), `reduce_grads` averages every parameter's gradient over
      the ranks, and `fit_controlled_sync` applies it before `adamw_step` so
      the model's schedule, clipping, moments and save path are unchanged. A
      one-rank reduction is the identity; exact resume with a sync is refused
      until the world size is part of the state's identity. Tests: the mean a
      reduction writes, the one-rank identity, and a fit run through the hook.

- [x] **A distributed driver with a real data-parallel data split.**
      `model::train::fit_distributed` builds the model and dataset exactly as
      `fit` does and reduces every step's gradients over the caller's
      `GradSync` group. `Objective::shard` gives each rank its own shard of the
      deterministic batch stream: rank `r` of `world` trains on batches `r`,
      `r + world`, ...; the shards are disjoint and their interleaving is the
      unsharded stream, so averaging their gradients is data parallelism rather
      than the same batch on every rank. `fit_controlled_sharded` applies the
      shard and refuses one that disagrees with the gradient group's rank.
      `gpt2::train::train_distributed` is the GPT binding. Only rank 0 writes
      the checkpoint (the replicas are bit-identical); every rank reads it to
      resume. Tests: the disjoint-shard/union and one-rank-identity cases in
      `crates/model/src/train.rs`, and
      `crates/gpt2/tests/distributed_train.rs`, which starts two OS processes
      over loopback and asserts their trained replicas are bit-identical.

- [x] **A train verb takes a job plan.** `brain gpt2 train <data_dir>
      --job-plan FILE --job-rank N` joins the plan's collective, announces the
      job and rank, and runs the distributed driver; giving only one of the
      pair is a usage error. The flag is `--job-rank` because `--rank` is LoRA
      rank throughout the CLI.

- [x] **The LoRA chat fine-tune takes a job plan.** `FitControl::grad_sync`
      turns the shared `fit_controlled` loop into a data-parallel run: every
      rank reduces each step's gradients over the group and trains on its own
      shard of the training-record stream, derived from the group's rank and
      world size. `brain qwen3 finetune --lora ... --job-plan FILE
      --job-rank N` joins the plan's collective and sets it; half the pair is a
      usage error, and a rank without `--seed` derives one from the plan bytes.
      The record strategy is index-based and explicit: record `n` belongs to
      rank `n % world`, every rank consumes the whole stream, and evaluation is
      never sharded (every rank scores the same held-out set). An objective
      that cannot shard, or a model that cannot write reduced gradients, fails
      by name instead of falling back to a single process. Only rank 0 writes
      the shared checkpoint and evaluation snapshots. Tests: the record
      disjointness/union and one-rank-identity cases in
      `crates/model/src/train.rs`, the half-pair usage case in
      `crates/cli/src/qwen_cli.rs`, and a two-process Qwen3-0.6B LoRA smoke
      that must end with byte-identical adapters.

## Decisions

- A cluster job's authority is its durable job record and its checkpoint
  commit, not a per-worker session store. The durable session engine
  (`crates/session`, content-addressed artifacts) is a single-writer local
  control surface: a worker never opens it, and a multi-node run does not
  create sessions on its workers.

- Training-stream sharding is index-based: every rank consumes the whole
  stream and keeps record `n` only when `n % world == rank`. On a dataset with
  example boundaries (the prepared JSONL a fine-tune trains on) a record is one
  example; a plain token stream has no record boundaries, so there one drawn
  batch is one record. That keeps every rank on one byte-for-byte identical
  schedule (the epoch order and the token generator advance the same way
  everywhere) at the cost of `world` times the host-side sampling work, which
  is not model work. Evaluation is never sharded: every rank scores the same
  held-out set, so replicas agree on the held-out loss and on an early-stop
  decision.

## Open

- [ ] Only the GPT trainer and the Qwen3 LoRA fine-tune take a job plan. The
      GLM trainer, the full-parameter fine-tunes and the other model families
      still run single-process.
- [ ] Exact resume is still refused for a distributed run: the training state
      names one rank's step and options but not the world size, so a resumed
      job could change its group without being refused.
- [ ] `DdpOptimizer`, `DataParallel` and `federated_average` have no
      production caller; the shipped driver reduces through `reduce_grads` so
      each model keeps its own optimizer and schedule.
- [ ] Device-resident collectives and an NCCL provider (the `Collective` seam
      and the CUDA provider table are where they belong); today every payload
      is staged through host fp32.
- [ ] Wire the commit into a trainer: rank 0 writes the weights and the
      exact-resume state, calls `CheckpointCommit::build`/`write`, and the
      driver resumes from `CheckpointCommit::latest` after verifying it.
      Replicas are bit-identical, so one writer is enough until tensor
      parallelism shards the weights.
- [ ] Ring/tree all-reduce for worlds where the coordinator star's bandwidth
      matters, and cross-rank validation of every op on the worker side (the
      coordinator validates today; a worker's own rank/length is checked
      locally).
- [ ] Sharded optimizer state, and the tensor- and expert-parallel dimensions
      whose groups `Grid` already addresses.
