<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# distributed-execution - roadmap

Scope: Running one training job as ranks across machines, and serving one model fleet across machines - the job plan every rank agrees on, the process group it joins, the transport's failure semantics, device-resident collectives across nodes, the request router over several servers, the cluster launch sample, and the tests that start a cluster as real processes (`crates/model`: `job`, `collective`, `devcollective`, `netcollective`, `distributed`, `grid`; `backend_cuda::nccl`, `gpu_core::devgroup`).

## Delivered

- [x] **A failed collective is an error, not a panic.** `FlatAdam::step`,
      `DdpOptimizer::step`, `fed_avg_flat` and `federated_average` return
      `CollectiveResult`; a lost peer or a rejected payload leaves the
      replica's weights and optimiser state untouched, so a driver can stop at
      the last committed step instead of losing the process. Tests:
      `crates/model/tests/it/collective_errors.rs`.
- [x] **A job plan every rank validates.** `model::job::JobPlan` names the job,
      the world size, the tp/pp/dp grid and the coordinator; it is a value the
      launcher writes, never something brain reads from a scheduler. Every
      rank announces the plan's exact bytes when it joins, and a rank started
      with a different plan is refused with the mismatch rather than joining
      someone else's collective. `JobContext` carries a rank and derives its
      grid coordinate. Tests: the unit tests in `crates/model/src/job.rs`.
- [x] **Only the job's ranks join, and nobody else reads the wire.** The
      launcher hands every rank one owner-only secret file (`openssl rand -hex
      32`, `model::JobSecret::read` refuses a file others may read). Each
      connection proves it both ways by HMAC over fresh nonces, then every
      frame is ChaCha20-Poly1305-sealed with per-connection, per-direction
      HKDF keys and a sequence nonce (`model::sealedwire`). A peer without the
      secret is dropped after a fixed-size exchange on its own thread, so it
      can neither join, read, alter or replay frames, nor tear the forming
      group down; there is no unauthenticated constructor left. Tests: the
      unit tests in `crates/model/src/sealedwire.rs`, and
      `a_peer_without_the_job_secret_cannot_join_or_disturb_the_job` in
      `crates/model/tests/it/distributed_job.rs`. Cost, measured 2026-11-06 on
      a Core Ultra 7 155H over loopback with 16 MiB frames (`fast` profile):
      one sealed stream moves 1.32-1.35 GB/s against 5.5-6.0 GB/s plain, so
      sealing is the ceiling of one connection on links faster than 10 GbE.
- [x] **The group fails together.** The TCP transport has an abort frame: a
      coordinator that rejects a payload or loses a peer sends every remaining
      rank the reason, and each op carries a timeout from the plan, so no rank
      waits forever for a payload that will not come. Join announcements and
      aborts are length-bounded.
- [x] **A cluster can be simulated without a scheduler.** `crates/model/tests/it/distributed_job.rs`
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
      `crates/gpt2/tests/it/distributed_train.rs`, which starts two OS processes
      over loopback and asserts their trained replicas are bit-identical.

- [x] **A train verb takes a job plan.** `brain gpt2 train <data_dir>
      --job-plan FILE --job-rank N --job-secret FILE` joins the plan's
      collective, announces the job and rank, and runs the distributed driver;
      giving only some of the three is a usage error (`args::JobFlags`, shared
      with the fine-tune verb). The flag is `--job-rank` because `--rank` is LoRA
      rank throughout the CLI.

- [x] **The LoRA chat fine-tune takes a job plan.** `FitControl::grad_sync`
      turns the shared `fit_controlled` loop into a data-parallel run: every
      rank reduces each step's gradients over the group and trains on its own
      shard of the training-record stream, derived from the group's rank and
      world size. `brain qwen3 lora_train ... --job-plan FILE
      --job-rank N --job-secret FILE` joins the plan's collective and sets it;
      part of the three is a usage error, and a rank without `--seed` derives one from the plan bytes.
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
      point), and `crates/bench/tests/it/distributed_profile.rs`, which runs the
      profile small and parses its report.

- [x] **A distributed run reports its own collective cost.**
      `brain gpt2 train` and `brain qwen3 lora_train` take
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
      `fetch::file_digest_sidecar` records `sha256:<hex>` with the file's stamp
      (device, inode, size, nanosecond modification and change times) and the
      time hashing began in a `<file>.sha256` sibling, and reuses it only while
      the stamp matches and the record is more than a timestamp tick younger
      than the file's last change; a missing, corrupt or stale sidecar is
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

- [x] **One URL in front of many servers (C4, router part).** `brain serve
      --upstream URL ...` (or `--upstreams-file`) serves the OpenAI dialect
      over other `brain serve` endpoints and loads no model; the library is
      `apiserve::route` (`Fleet`, `router`, `serve`). Routing is
      `residency::affinity`'s replica policy with each upstream as a replica:
      `choose_replica` now weighs load against each replica's own slots, and
      `text_request_chain` keys a text request the same way for the
      in-process dispatcher and the router. Health and capacity come from the
      upstream's new `GET /v1/health` (decode slots, edge concurrency) and its
      `/v1/models`; drain and resume are admin routes; the listing is merged;
      a request is retried elsewhere only when its connection never opened;
      past every upstream's own edge ceiling the router refuses with 503 and
      `Retry-After`. Tests: `crates/apiserve/tests/it/route.rs` (mock
      upstreams on loopback: affinity, load per slot, health, drain,
      streaming and cancel-on-disconnect, at-most-once, model merge,
      overload, the security boundary), `tests/e2e/route.bats` (two real
      `brain serve --mock` behind a real router), the unit tests in
      `crates/residency/src/affinity.rs` and `crates/apiserve/src/route/upstream.rs`.

## Decisions

- The router is a mode of `brain serve` and a module of `apiserve`
  (`apiserve::route`), not a `brain` SDK surface: the SDK is an in-process
  library with no admission control by design (`crates/sdk/src/lib.rs`), and a
  router is a network service whose whole job is admission. An application
  embeds it through `apiserve::route` the way it embeds a serving surface
  through `apiserve::serve_all`. It opens one connection per request: that is
  what makes "the connection never opened" a safe retry condition (a pooled
  connection found dead at write time is ambiguous), at the cost of a TCP
  handshake per request.

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

- **Cluster fabric, NCCL 2.32.3 (2026-11-21, Slurm job on GH200 nodes, fp32,
  20 iterations after 5 warmup, one process per card).** Libfabric network
  plugin 1.22.0 over the hosts' libfabric 2.3.1 CXI provider with GPU direct
  RDMA, against NCCL's TCP sockets on the Slingshot interfaces. Bus bandwidth
  (nccl-tests convention) at 64 MiB / 1 GiB / 4 GiB:
  - 4 ranks, one node, NVLink: all-reduce 302.7 / 339.6 / 340.4 GB/s,
    all-gather 260.3 / 312.7 / 322.3, reduce-scatter 264.7 / 312.7 / 321.0.
  - 2 ranks, one card on each of two nodes: all-reduce over the fabric plugin
    13.1 / 12.6 / 12.9, over sockets 7.9 / 9.8 / 9.8; all-gather 12.4 / 12.5 /
    12.7 vs 10.0 / 9.1 / 9.5. One rank per node drives one NIC.
  - 8 ranks, 4+4 on two nodes, fabric plugin: all-reduce 115.1 / 149.2 /
    153.0, all-gather 84.4 / 92.2 / 92.6, reduce-scatter 85.1 / 91.1 / 92.3:
    the four NICs per node near line rate. Small messages: 8-byte all-reduce
    14.3 us over the fabric, 29.9 us over sockets.
  The data-parallel budget this sets: a 27B LoRA gradient (tens of MiB) reduces
  in about a millisecond across 8 cards, a full bf16 27B gradient (54 GB) in
  about 0.7 s, so full-parameter runs need C3's sharding and overlap.

- **Gradient reduction, device vs host star (2026-11-21, one GH200 node, NCCL
  2.32.3, CUDA backend, one process per card, `nccl_job::measure_*`).** Median
  wall time per reduction, including the wait for completion: the device path
  is `JobCollective::all_reduce_device` (one NCCL all-reduce over NVLink); the
  host star reads the gradient back, all-reduces it over the sealed loopback
  transport in pieces of at most 2^27 floats and writes it back. fp32 payloads:
  - Qwen3-0.6B LoRA r4 (10.1 MB): 2 ranks 0.23 vs 45.5 ms; 4 ranks 0.23 vs
    75.7 ms.
  - 27B LoRA r16 on every linear (108.9 M params, 435 MB): 2 ranks 3.9 ms vs
    2.14 s; 4 ranks 2.1 ms vs 3.55 s.
  - Qwen3-0.6B full gradient (596 M params, 2.38 GB): 2 ranks 20.2 ms vs
    13.2 s; 4 ranks 10.7 ms vs 21.4 s (4 ranks: 222 GB/s algorithm, 333 GB/s
    bus bandwidth, the C0 ceiling).
  The host star is 200-2000x slower and its time grows with the world; for
  the large payloads the device path's time falls with it.

- **`brain qwen3 lora_train --job-plan`, device vs host star (2026-11-21,
  one GH200 node, one card per rank).** Qwen3-4B, LoRA r16 on all seven
  projections (33.0 M fp32 gradient elements, 132 MB per step), bf16 base,
  block 128, batch 1, eval-free; per-step time by differencing a 6-step and a
  2-step run's training call on rank 0 so the fixed setup cancels. The plan's
  `collectives` was `auto` (every rank reported the device path over NCCL)
  or `host`:
  - 2 ranks: 108 ms/step on the device vs 719 ms through the host star (6.7x).
  - 4 ranks: 154 ms/step vs 1128 ms (7.3x).
  - The host star spends about 530 ms (2 ranks) and 1130 ms (4 ranks) of a
    step in 1009 per-tensor reductions; the device path one grouped
    reduction. Forming the device group (rendezvous and NCCL init) is a
    one-off 2.0 s (2 ranks) and 3.1 s (4 ranks) on the first step.
  - Every run's adapters were byte-identical across its ranks, device and
    host alike.

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

## Plan: multi-node clusters (2026-11-21)

Target machine class: Slurm allocations of GH200 nodes, 4 cards per node on
NVLink (NV6, about 127 GB/s card to card measured), 4 Slingshot-11 NICs per
node (`/dev/cxi0-3`, about 25 GB/s each), hosts with libfabric (CXI provider,
CUDA and GDRCopy enabled) but no NCCL network plugin. Every launched step gets
its own fabric VNI, so fabric traffic must come from scheduler-launched ranks.
brain stays scheduler-agnostic: a launcher maps the scheduler's rank and
addresses onto `--job-plan`/`--job-rank`, brain reads no scheduler variable.

- [x] **C0 fabric baseline.** NCCL all-reduce/all-gather/reduce-scatter bus
      bandwidth within a node, across 2 nodes and across 16 cards, over the
      libfabric network plugin and over NCCL's socket fallback, recorded under
      Measured. This is the ceiling every later item is judged against.
      Done 2026-11-21 for 4 cards (NVLink), 2 and 8 ranks over two nodes
      (knowledge #541); 12 and 16 ranks follow with C2's scaling runs.
- [x] **C1 NCCL across processes and nodes.** Done 2026-11-21.
      `backend_cuda::nccl::NcclRank` is one rank per process on one card:
      `ncclGetUniqueId`, nonblocking `ncclCommInitRankConfig`, `ncclCommSplit`
      (`Grid::split_key` gives the color and key of a tp/pp/dp group),
      all-reduce (grouped, with a per-rank pre-multiplied sum), all-gather,
      reduce-scatter and broadcast over fp32 and bf16. Creation, a split and
      every enqueue are bounded by a deadline, and `wait` polls the stream and
      `ncclCommGetAsyncError`, so a dead peer is an error within the plan's op
      timeout (the communicator is then aborted), never a hang; a clean drop
      finalizes and destroys. `gpu_core::devgroup::RankGroup` is the facade;
      the single-process `NcclGroup` (`ncclCommInitAll`) serving uses is
      unchanged. `model::devcollective::rendezvous` forms the group over the
      job's sealed host transport: every rank states whether it can take
      part, rank 0 creates the id and broadcasts it there (so only holders of
      the job secret ever see it), every rank joins and reports, and only a
      group every rank joined is used. `JobCollective` (what
      `JobContext::job_collective` and so every `--job-plan` verb now get)
      keeps payload ops on the host star and reduces device gradients with
      `Collective::all_reduce_device`; `reduce_grads(_weighted)` take that path
      when the model keeps its gradients in its parameter store. The plan's
      `collectives` field (`auto` default, `host`, `device`) chooses; `auto`
      falls back to the host star and every rank says why once. Replicas get
      the same reduced bytes from NCCL, so they stay bit-identical; for a
      fixed world, card set and library a repeat gives the same bytes
      (asserted), but a device run is not bitwise equal to a host-star run.
      Tests: `crates/model/tests/it/nccl_job.rs` (2 and 4 OS processes on one
      node: every op, bf16, tp/dp splits against a host reference, identical
      bytes across ranks and repeats; a rank that dies, error in 5.5 s for a
      5 s op timeout; a rank without the secret never joins), the
      `devcollective`, `grid` and `job` unit tests; `brain qwen3 lora_train
      --job-plan` on 2 and 4 cards (Measured). Open: no two-node run of the
      provider yet (C2's scaling runs).
- [ ] **C2 data parallelism for every trainer.** The plan-file data-parallel
      path moves from the GPT and Qwen3 LoRA bindings into the shared training
      capability, so any architecture with gradients trains as ranks; gradients
      are reduced in buckets overlapped with backward; exact resume carries the
      world size. Gates: replicas bit-identical across 2 nodes; scaling
      efficiency at 4, 8 and 16 cards on Qwen3-0.6B LoRA and 27B LoRA.
- [ ] **C3 sharded training.** Optimizer state and gradient sharding over the dp
      group (reduce-scatter + all-gather), then parameter sharding; tensor
      parallelism inside a node over NVLink, sharded data parallelism across
      nodes. Gate: a full-parameter 27B fine-tune step that does not fit one card.
- [ ] **C4 serving across nodes.** A router capability in front of several
      `brain serve` endpoints, reusing `residency::affinity` (longest cached
      prefix, then least loaded), with health, drain and model listing merged,
      so a client sees one OpenAI-compatible URL for every card of an
      allocation: delivered (`brain serve --upstream`, see Delivered); a
      measurement on real servers across nodes is pending. Open: tensor-parallel serving
      inside a node for a model that does not fit one card; a cross-node layer
      split only for one that does not fit a node; multipart uploads
      (`/audio/transcriptions`) and job routes are not routed.
- [ ] **C5 cluster launch sample and gates.** `samples/` gains a Slurm launch
      sample (job plan and secret written once, scheduler rank mapped onto
      `--job-rank`, rank bound to its card's NUMA node, NCCL network plugin
      selected) and a cluster end-to-end suite that runs under the scheduler
      and skips by name without one.

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
- [ ] Device-resident collectives cover the data-parallel gradient reduction
      of a model whose gradients live in its parameter store (C1). A resident
      full fine-tune (`model::resident_ft`) keeps its linears' gradients
      outside the store and still reduces through the host star; the payload
      ops (`Collective::all_reduce` and friends) stay on the host transport.
- [ ] Wire the commit into a trainer: rank 0 writes the weights and the
      exact-resume state, calls `CheckpointCommit::build`/`write`, and the
      driver resumes from `CheckpointCommit::latest` after verifying it.
      Replicas are bit-identical, so one writer is enough until tensor
      parallelism shards the weights.
- [ ] Ring/tree all-reduce for worlds where the coordinator star's bandwidth
      matters (sealing runs at about 1.3 GB/s per connection, so a ring also
      spreads the encryption over every rank's cores), and cross-rank validation of every op on the worker side (the
      coordinator validates today; a worker's own rank/length is checked
      locally).
- [ ] Sharded optimizer state, and the tensor- and expert-parallel dimensions
      whose groups `Grid` already addresses.
