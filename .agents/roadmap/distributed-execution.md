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

## Decisions

- A cluster job's authority is its durable job record and its checkpoint
  commit, not a per-worker session store. The durable session engine
  (`crates/session`, content-addressed artifacts) is a single-writer local
  control surface: a worker never opens it, and a multi-node run does not
  create sessions on its workers.

## Open

- [ ] No training verb takes the plan yet. A launcher starts
      `brain <arch> train ...` with `--job-plan FILE --job-rank N`; the flag
      names must not be `--rank`, which every trainer already uses for LoRA
      rank. The driver builds `JobContext::collective`, passes
      `fit_controlled_sync` the group, and gives each rank its own shard of the
      batch stream (the reduction exists; the data split does not).
- [ ] The distributed driver is reachable as a library only: `DdpOptimizer`,
      `DataParallel` and `federated_average` have no production caller. The
      first one should be the LoRA chat fine-tune, which today plans pipeline
      stages across local cards inside one process.
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
