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
      never sharded (rank 0 scores it and broadcasts the result). An objective
      that cannot shard, or a model that cannot write reduced gradients, fails
      by name instead of falling back to a single process. Only rank 0 writes
      the shared checkpoint and evaluation snapshots. Tests: the record
      disjointness/union and one-rank-identity cases in
      `crates/model/src/train.rs`, the half-pair usage case in
      `crates/cli/src/qwen_cli.rs`, and a two-process Qwen3-0.6B LoRA smoke
      that must end with byte-identical adapters.

- [x] **A distributed run is measurable on one machine.**
      `collective::TimedCollective` wraps any transport and times each op
      inside its async block, accounting per op kind for call count, staged
      payload bytes (elements times the dtype's bit width, rounded up, so the
      sub-byte tiers count) and total wall time, with p50/p95/p99 computed
      from the raw per-call samples. A run that does not construct one pays
      nothing. `collective::ShapedCollective` charges a deterministic link
      model before the wrapped op - two one-way propagation delays plus
      payload/bandwidth plus jitter drawn from `seed ^ op_index` through
      SplitMix64, with no mutable RNG and no environment - and can fail at a
      configured op index with a transport error without touching the inner
      collective. `crates/bench/src/distributed.rs` trains the tiny GPT of the
      distributed tests as world 1/2/4 over the host, loopback TCP and shaped
      transports through the public fit seam, and emits one JSON document:
      per-run step times, tokens/s, communication bytes/time/share, throughput
      speedup and parallel efficiency against the same transport's world-1
      run, and whether the replicas ended bit-identical. `make
      bench/distributed` runs it, and it selects the CPU backend by name so a
      run needs no device and no environment variable. Tests: the collective
      unit tests (accounting, dtype widths, percentile ranks, the link's
      latency/bandwidth/seeded-jitter properties, and the failure firing
      point), and `crates/bench/tests/distributed_profile.rs`, which runs the
      profile small and parses its report.

- [x] **A distributed run reports its own collective cost.**
      `brain gpt2 train` and `brain qwen3 finetune --lora` take
      `--profile-collectives FILE` alongside `--job-plan`/`--job-rank`. Each
      rank wraps its joined transport in `collective::TimedCollective` for the
      run and writes `FILE.rank<rank>.json`: the plan identity, rank, world,
      the training call's wall time and the collective's share of it, plus per
      op kind the call count, staged bytes, total and p50/p95/p99. The
      reported share is of the whole call (a checkpoint load and dataset
      render included); a per-step share is measured by differencing two step
      counts so the fixed part cancels. The wrapper lives
      only in the verb's own scope, so a run without the flag constructs no
      wrapper and writes no file (the `FitControl`/`GradSync` seam is
      unchanged); a flag without a job plan is a usage error, since a
      one-process run has no collective to measure and `distributed_profile`
      already covers world 1. The document is `Snapshot::to_json`, the same
      accounting shape the profile bench emits. Tests: the flag's refusal on
      both verbs, its acceptance with a job plan, and the two-process
      distributed training test, which writes one document per rank and
      asserts they agree on world and plan identity with a non-zero op count
      and byte count, and that the run without the flag writes none.

- [x] **A base is hashed once, not once per rank and per save.**
      `fetch::file_digest_sidecar` records `sha256:<hex>` with the file's length
      and nanosecond modification time in a `<file>.sha256` sibling and reuses
      it only while both still match; a missing, corrupt or stale sidecar is
      ignored and rewritten, and a cache hit answers exactly the digest the
      file's bytes give. Writes go through a per-call temporary and a rename,
      so ranks hashing the same base concurrently leave one complete sidecar.
      `fetch::base_digest_sidecar` applies it to every file of a
      `transformers` directory base. The fine-tune save paths use it (the Qwen3
      CLI and capability action, and the Qwen3.5 streamed fine-tune), so the
      first process pays the hash and the rest stat the file.

- [x] **Held-out evaluation runs once per job, not once per rank.**
      Rank 0 scores a due evaluation and broadcasts the scalar over the
      gradient group; every rank drives selection and early stopping from the
      same value. A step with no evaluation broadcasts that fact as its own
      value, so no rank waits on a root that skipped the call, and a failed
      broadcast fails the run with the transport error. The replicas hold
      identical weights, so the replicated evaluation measured the same number
      world-size times: 2.6-3.0 s each at block 128 over 20 batches, about 83%
      of the default 2-step wall, with no gain from a second rank. What
      remains: the evaluation stays on rank 0's critical path, so sharding it
      across ranks and reducing the partial losses would shorten it - a
      distinct change from removing the duplication. Tests:
      `only_rank_zero_evaluates_and_every_rank_selects_the_same` and
      `every_rank_agrees_when_the_objective_reports_no_evaluation` over a
      two-rank `HostCollective`, plus the unchanged single-process case, in
      `crates/model/src/train.rs`.

- [x] **The trainer commits what it saves, and a resume verifies it.**
      Every periodic and final save publishes `CheckpointCommit` beside the
      checkpoint; a resume refuses a generation that is missing, fails
      verification, or names another job, plan or world, and a distributed
      resume must name its world size in the run identity so a changed group is
      refused. The Qwen CLI commits its adapter (written into the model store,
      not the run directory) from rank 0 after saving. Tests: the published
      commit and its three refusals, the world-identity refusal, and the
      two-rank run below.

## Decisions

- A cluster job's authority is its durable job record and its checkpoint
  commit, not a per-worker session store. The durable session engine
  (`crates/session`, content-addressed artifacts) is a single-writer local
  control surface: a worker never opens it, and a multi-node run does not
  create sessions on its workers.

- The one-machine profile's speedup is a tokens-per-second ratio against the
  same transport's world-1 run, so parallel efficiency (speedup / world) is
  the step-time ratio for a fixed amount of work per rank. Its communication
  figures are the payload one rank staged and the time its collective wrapper
  measured, not bytes on a NIC; the host transport moves those bytes through
  host RAM.

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

## Measured

- **Single-machine profile** (`make bench/distributed ARGS="--steps 3"`): a
  1-layer GPT (d_model 32, block 16, batch 2) on the CPU backend, 3 steps,
  world 1/2/4, all replicas bit-identical. Shaped link = 100 Gb/s, 100 us
  one-way, no jitter.
  - host (threads, shared RAM): 1880/2294/3112 tok/s at world 1/2/4,
    speedup 1.00/1.22/1.66, efficiency 1.00/0.61/0.41, comm share
    1.1/5.9/5.5%.
  - loopback TCP: 1655/2136/2387 tok/s, speedup 1.00/1.29/1.44,
    efficiency 1.00/0.65/0.36, comm share 0.0/12.6/23.7%.
  - shaped: 1550/1406/2075 tok/s, speedup 1.00/**0.91**/1.34,
    efficiency 1.00/0.45/0.34, comm share 0.9/**42.5**/42.6%.
  The shaped world-2 row is the surprise the method exists to catch: at
  realistic latency the reduction costs more than the second rank adds, for a
  model this small. The tiny model exaggerates the share; the method must be
  repeated at the target model size before drawing conclusions about larger
  models.
- **Qwen3-0.6B LoRA, two P40s, world 2** (LoRA 4, block 128, batch 1, 2 steps,
  Vulkan, three repeats, medians):
  - eval-free steady state 1.862 steps/s (1 process) vs 2.050 (2 ranks):
    throughput speedup 1.98, parallel efficiency about 99%; the two adapters
    are byte-identical (`sha256:4b77add9...`), held-out scores identical on
    both ranks, train losses differ per shard.
  - default cadence with one eval per step: 0.315 vs 0.283 steps/s; each eval
    cost 2.6-3.0 s at block 128 over 20 batches and was replicated on every
    rank, about 83% of the default 2-step time with no gain from a second
    rank. It now runs on rank 0 only and is broadcast.
  - reduction: 392 LoRA tensors, 9.63 MiB of host fp32 per rank per step; the
    worker's socket time is about 0.24 s per step traced, about 2.5% of wall
    in the eval-free run and about 27% of the reduction window. The star is
    asymmetric: the coordinator rarely blocks, the worker waits.
  - fixed per-run overhead was dominated by the per-process SHA-256 of the
    2.2 GiB base (about 14 s), which every rank recomputed at save time; a
    validated sidecar now serves it after the first process hashes it.
- **Qwen3-0.6B LoRA, two P40s, world 2, `--profile-collectives`**
  (LoRA 4, block 128, batch 1, Vulkan, eval-free): 20,185,088 bytes and 784
  all-reduce calls per rank over two steps - 10,092,544 bytes (9.63 MiB) and
  392 reductions per step, the same volume the traced run above saw. Rank 0
  spent 170.1 ms in the collective over the run (85.0 ms/step), rank 1
  364.2 ms (182.1 ms/step): the star is asymmetric, the worker waiting on the
  coordinator's compute. Differencing a 4-step run (4552.8 ms) against the
  2-step one (3325.0 ms) puts a step at 613.9 ms with about 2.10 s of fixed
  setup, so the collective is 13.9% of a step on rank 0 and 29.7% on rank 1;
  the share of the whole training call is 5.1% and 10.9%. Per-step reduction
  time varies between runs (rank 1 measured 94.7 ms/step at 4 steps), so the
  fixed part is differenced rather than assumed.

- **End to end through a durable job controller.** An orchestrator submitted
      two Qwen3-0.6B LoRA ranks on one machine (eval-free, 2 steps, one P40
      each); the job completed 6.6 s after launch with exit 0, the adapters in
      `out0` and `out1` are byte-identical (`sha256:f5e70fce...`), rank 0
      published the step-2 commit, and the controller registered the adapter
      (10,140,344 bytes, matching sha256) from the manifest beside the nested
      model-store copy. The per-rank profiles report 168 and 186 ms of
      collective time in a 3,132 ms call (5.4% and 5.9%).

- **Two serving replicas, one per card**, serving the distributed-trained LoRA
      adapter through two independent `brain serve` processes (no gateway; a
      client splits requests). The first window was cold-start dominated
      (p50 11.6 s); after three warmup requests each, 8+8 concurrent 16-token
      chats gave 8.29 req/s and 132.6 tok/s per replica, p50 921-961 ms, and
      16.57 req/s and 265.2 tok/s combined, 2.0x one replica. Both servers
      list the adapter id. This is replica distribution, not model sharding:
      no request spans cards and no routing layer is implied.

## Open

- [ ] The one-machine profile cannot see a real network. The loopback
      transport's latency is the kernel's TCP stack on one host, there is no
      NIC and no scheduler, and the host transport stages payloads through host
      RAM. The shaped transport is a model of a link, not a measurement of one:
      it charges the transfer term once per op and carries no protocol
      overhead, so it ranks transports and decomposes a step's cost, but it
      cannot predict a cluster's absolute step time.

- [ ] Only the GPT trainer and the Qwen3 LoRA fine-tune take a job plan. The
      GLM trainer, the full-parameter fine-tunes and the other model families
      still run single-process.
- [ ] Exact resume is still refused for a distributed run: the training state
      names one rank's step and options but not the world size, so a resumed
      job could change its group without being refused.
- [ ] `DdpOptimizer`, `DataParallel` and `federated_average` have no
      production caller; the shipped driver reduces through `reduce_grads` so
      each model keeps its own optimizer and schedule.
- [ ] Device-resident collectives: `gpu_core::devgroup::DeviceGroup` (NCCL by
      `dlopen`, host fallback with a stated reason) exists, ledger in
      `serving-runtime.md` item 12; the `Collective` seam in `model::collective`
      does not call it yet, so every distributed-training payload is still
      staged through host fp32.
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
