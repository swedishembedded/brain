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

## Open

- [ ] No training verb takes the plan yet. A launcher starts
      `brain <arch> train ...` with `--job-plan FILE --job-rank N`; the flag
      names must not be `--rank`, which every trainer already uses for LoRA
      rank. The driver then builds `JobContext::collective` and runs
      `DdpOptimizer` over it.
- [ ] The distributed driver is reachable as a library only: `DdpOptimizer`,
      `DataParallel` and `federated_average` have no production caller. The
      first one should be the LoRA chat fine-tune, which today plans pipeline
      stages across local cards inside one process.
- [ ] Device-resident collectives and an NCCL provider (the `Collective` seam
      and the CUDA provider table are where they belong); today every payload
      is staged through host fp32.
- [ ] A distributed checkpoint commit: rank 0 writes the weights and the
      exact-resume state with a manifest naming the global step and the plan;
      replicas are bit-identical, so one writer is enough until tensor
      parallelism shards the weights.
- [ ] Ring/tree all-reduce for worlds where the coordinator star's bandwidth
      matters, and cross-rank validation of every op on the worker side (the
      coordinator validates today; a worker's own rank/length is checked
      locally).
- [ ] Sharded optimizer state, and the tensor- and expert-parallel dimensions
      whose groups `Grid` already addresses.
