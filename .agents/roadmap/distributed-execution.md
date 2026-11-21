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
      an upstream is sent at most its decode slots, past the fleet's sum a
      request waits in a bounded FIFO-per-model queue under the admit deadline
      (429 at the deadline, 503 past `--route-queue`); `prompt_cache_key`
      keeps a conversation on one upstream; `/v1/metrics` reports per
      upstream in-flight, capacity, queue depth, recent tok/s and prefix-hit
      rate (from the upstream's cumulative served counters in `/v1/health`)
      and fleet totals; every request carries a router `X-Request-Id` to the
      upstream and back. Tests: `crates/apiserve/tests/it/route.rs` (mock
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

- Cluster topology facts that need no tuning (verified 2026-11-21 on the GH200
  nodes): NIC `cxiN` sits on NUMA node N beside card N, and NCCL's own
  topology search already sends card N's network channels over `cxiN` on every
  node, so no topology file is written. NVLS is unavailable (NCCL: "NVLS
  multicast support is not available"; four GH200s on NVLink have no NVSwitch),
  so no NVLS path is planned. Without a job-level VNI (granted only when the
  allocation itself is requested with `--network=job_vni`), every srun step
  gets its own VNI, so all ranks of one NCCL group are tasks of one step;
  separately launched steps (trainer, rollout servers) talk over TCP/HTTP.
- Three reproducibility levels, chosen per run, not implied by speed settings:
  fast (results within a stated tolerance), repeatable (bit-identical on an
  unchanged layout, library and card set: what resume and exams rely on today,
  since every rank receives NCCL's identical bytes) and strict (a fixed
  reduction tree independent of NCCL's algorithm choice, for comparisons across
  layouts). Fixing NCCL settings alone does not give the last two; data order,
  RNG streams, kernel choice and accumulation order are part of the contract.

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
  block 128, batch 1, eval-free. Per-step times here are the fit loop's own
  step stage (`--kernel-profile`), median of steps 3-10 on rank 0.
  - 1 card (no job plan) 110.5 ms; 2 ranks 118.2 ms; 4 ranks 117.7-121.6 ms
    over four runs: 128-token rows per second 1158 / 2166 / 4210-4350,
    parallel efficiency 93% at 2 and 91-94% at 4. The 108 -> 154 ms "growth"
    first recorded here was an artifact: it differenced whole training calls
    whose fixed part (load, NCCL init) varies by hundreds of ms.
  - Where the 4-rank extra goes, with the backward drained first: the device
    reduction 3.0 ms (504 tensors as one NCCL group; 0.7 ms of it is
    bandwidth, from the payload sweep), the step-weight scalar over the host
    star 0.1-1.8 ms (it is also where a rank waits for the slowest one), and
    the remaining few ms are the drain itself. Overlap can hide at most the
    3.0 ms (overlap efficiency E = 0 today: the reduction starts after the
    backward).
  - Bucket cap sweep at 4 ranks (device reduction per step): none 3.0 ms;
    4 / 16 / 64 / 128 MiB 5.8 / 5.2 / 5.2 / 5.2 ms. Per-tensor copies into the
    flat buckets cost more than the fewer NCCL operations save, so the default
    packs nothing.
  - Repeatability: every run's adapters were byte-identical across its
    ranks, and two 4-rank runs of the same plan gave the same bytes
    (`sha256:506f9ca1...`); a different bucket cap gives different bytes (it
    changes the reduction's layout).
  - Through the host star instead (same model, differenced training calls):
    about 530 ms (2 ranks) and 900 ms (4 ranks) of each step in 504
    per-tensor reductions.

- **Slingshot settings sweep (2026-11-21, two GH200 nodes, NCCL 2.32.3,
  fp32 all-reduce, 20 iterations).** The same ranks under the site's injected
  defaults, the CSCS Alps set (larger CXI queues, no host registration), the
  Isambard-AI set (hybrid matching, rendezvous from 0 bytes, `alt_read`), and
  single changes. Bus bandwidth at 64 MiB / 1 GiB and 8-byte latency:
  - 2 ranks (one card per node): site 13.0 / 12.6 GB/s, 15 us; CSCS 13.1 /
    12.8 (all-gather), 130 us; Isambard 22.0 / 22.2, 79 us; Isambard with the
    site's 16 KiB rendezvous threshold 20.0 / 22.8, 76 us; the site plus only
    `FI_CXI_RDZV_PROTO=alt_read` 19.2 / 22.9, 28 us. More channels per network
    peer (2, 4) change nothing (12.8-12.9).
  - 8 ranks (4+4): site 39.2 / 72.4, CSCS 113.9 / 150.9, Isambard 116.5 /
    151.3, site plus `alt_read` 104.6 / 147.5. These 8-rank rows ran while
    another process kept n442's cards busy (98% utilisation), and the site row
    measured 149.2 at 1 GiB alone earlier the same day, so they bound rather than
    rank the settings.
  The receiver-driven rendezvous is the one setting that matters here: it
  takes a single card's stream from half to nearly all of one 200 Gb/s link,
  without the small-message cost of the full Isambard set. The NIC octet
  counters (`hni_sts_{tx,rx}_ok_octets`) read 2.0x the payload NCCL reports in
  every configuration, so they are used for ratios, not absolute link rate,
  until that factor is explained.

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

- **Router across nodes, 2026-11-21** (`brain serve --upstream`, C4). Four
  `brain serve` processes on n442, one GH200 each (`--device gpuN --backend
  cuda`, Qwen/Qwen3-4B alone in the store, defaults: 263 decode slots each),
  bound on hsn0; the router on n424 (`--device cpu`) reaching them over
  Slingshot TCP. Load: a Python client on n424, non-streamed chat, greedy,
  `max_tokens` 64, fresh prompts per run (no cache carried between runs);
  prefix hit = cached prompt tokens / prompt tokens from each response's
  usage (token-weighted). Another lane shared the node, so early runs were
  noisy; the comparisons below are back-to-back pairs.
  - Multi-turn agents, 256 conversations x 6 turns, distinct ~1400-token
    system prompts, 256 in flight (1536 requests): router 6388 and 6334
    completion tok/s (99.8 and 99.0 req/s, p50 2.51/2.42 s, p95 3.48/3.99 s,
    p99 4.52/4.37 s), hit 0.818/0.822; client round-robin over the same four
    6067 and 6193 tok/s (94.8/96.8 req/s, p95 4.76/4.70 s, p99 5.45/5.73 s),
    hit 0.443/0.445; one server alone (earlier, under contention) 1606
    tok/s, hit 0.822. Affinity doubles the hit rate, cuts p95 by about a
    fifth and gains 3-5% throughput at this short output length.
  - Shared system prompt, 512 single-turn requests, 256 in flight: before
    the balance bound the router pinned all of it to one card (944 vs 4787
    tok/s round-robin, knowledge 543); with `balance_slack` 4907 vs 4978
    tok/s, hit 0.982 vs 0.981.
  - Not measured: TTFT and inter-token latency (non-streamed client),
    independent-prompt workloads, a least-loaded-only policy, and the
    router's queue under overload (the four upstreams' 1052 slots were never
    exceeded).

- **Router follow-up, streamed, 2026-11-21 18:32-18:37** (cost choice,
  token keys, keep-alive pool). Same fleet as above (four Qwen3-4B servers on
  n442, router on n424 CPU), now with no other lane on the cards; client
  streams with `include_usage`; TTFT = first delta, ITL = interval between
  streamed chunks (deltas are coalesced, so a chunk can carry several
  tokens); hit = cached / prompt tokens. Router binary at 10cadb127 (cost
  choice), "previous" = 40c42d559 (balance_slack, one connection per
  request).
  - Multi-turn (256 x 6 turns, 256 in flight): cost router 10067 and 10497
    completion tok/s, hit 0.81, TTFT p50 1.00/1.06 s, p95 3.35/2.67 s, ITL
    p95 0.68/0.69 s; client round-robin 6172 and 6658 tok/s, hit 0.43/0.42,
    TTFT p95 5.24/3.77 s, ITL p95 1.42/1.49 s. Previous router against cost
    router back to back: 8473/8607 against 8546/8340 tok/s, hit 0.80-0.81
    both, TTFT p95 3.33-3.41 against 3.28-3.33 s: no regression, and no gain
    while the KV term had nothing to say (see below).
  - Shared system prompt (512 single-turn): 4964/5096 against round-robin
    5233/5177 tok/s, hit 0.98 both, TTFT p95 3.48/3.20 against 3.78/4.03 s.
  - Independent short prompts (1024, 256 in flight): 12572 against 13110
    tok/s, TTFT p95 1.69 against 2.07 s, ITL p95 0.28 against 0.38 s.
  - Overload (3000 independent, 1500 in flight, 128 tokens): the first run
    reset 1976 client connections at the router's listener cap (knowledge
    545); with the cap sized from the fleet 28297/29814 tok/s, 0 errors, the
    router holding 1052 in flight (the fleet's slots) and about 450 queued,
    TTFT p95 9.33/7.91 s; round-robin 29932 tok/s, TTFT p95 7.63 s.
  - Token keying: 9216 requests, 1.73M tokens encoded in 8.13 s of encoder
    time (about 0.9 ms a request), 9.80M more served from the per-message
    memo (85% of what was keyed).
  - What the run showed of the upstream readings: an idle server reported
    20624 free KV tokens of 1052672, the rest held by its prefix cache, so the
    KV term now counts cache-held blocks as claimable (not re-measured on
    cards); and `free_slots`/KV refresh only between a lane's batch groups,
    so under sustained load they lag and the router's own in-flight count is
    what bounds an upstream.

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
      - [x] Exact resume carries the world size (2026-11-21): the fit loop
        writes the gradient group's world size into the state's identity
        itself, so every distributed caller keeps exact-resume state, a state
        from another world is refused, and a caller naming a contradictory
        world size is refused. Test:
        `a_distributed_state_carries_its_world_and_resumes_only_in_the_same_world`.
      - [x] Gradients are reduced where they live for every model that can
        name them (2026-11-21): `Model::device_grads` answers from the
        parameter store by default, and a resident full fine-tune adds its own
        fp32 or bf16 gradient buffers (`DeviceGrads::from_store_and_resident`,
        used by qwen3 and qwen35; their host `read_grad`/`write_grad` reach the
        resident state too instead of panicking). The reduction runs once per
        element type. A model that keeps state off the device and cannot name
        its gradients (a layer-streamed build) is refused by that reason when a
        data-parallel run starts.
      - [x] The SDK trains as ranks (2026-11-21): `brain::DataParallel` joins a
        launcher's job (`join_job`) or makes in-process ranks, one per card
        (`in_process`), and `ChatFineTune::data_parallel` runs any chat
        family the SDK trains as one rank (same seed required, refused by
        name otherwise). Test: `chat_finetune_data_parallel` (two in-process
        ranks end byte-identical; one process over the whole stream does not).
      - [x] Bucket layout (2026-11-21): `model::gradbucket::plan` packs
        tensors in reverse (backward) order into flat buckets under a cap, the
        same layout on every rank, members word-aligned (bf16), tensors at
        the cap in place; the plan's `bucket_mib` chooses the cap and is part
        of the layout a repeatable run fixes. Default 0 (no packing): measured
        slower with per-tensor copies (see Measured). The layout is the one a
        reduce-scatter (C3) shards.
      - [x] One report for every front end (2026-11-21):
        `ChatFineTuneOutcome::report` is the document the served `train` job
        answers with (totals, held-out scores, curve, selected step); the
        trainjob crate builds its result from it, and `brain finetune` writes
        it as `DIR/report.json` (rank 0 of a job). `brain finetune --lora`
        takes `--job-plan/--job-rank/--job-secret` and runs the SDK fine-tune as
        one rank (sft and policy gradient; the paired objectives are refused).
      - [ ] Overlap with backward. Not done: on the measured LoRA step the
        whole exposed reduction is 3.0 ms of 118 ms, so overlap could gain at
        most about 2.5%; it needs a per-layer gradient-ready hook in each backward and
        a one-kernel bucket gather. Revisit for full fine-tunes, where the
        payload is 30x larger.
      - [ ] The served `train` action on several cards of one server. Design:
        a request field `cards` (default 1) with admission charging every card
        through the residency multi-device path (`MultiDeviceResidentModel`),
        then in-process ranks via `brain::DataParallel::in_process` on those
        cards. Blocker: the multi-device path takes the device set from the
        model's own estimate, while a train job must be placed on any `cards`
        cards with room; the residency layer needs a "N of any devices" claim.
        Cross-node served training would take a job plan in the request (the
        plan, this server's rank range and the secret's digest) and the
        server's own sealed coordinator address.
- [ ] **C3 sharded training.** Optimizer state and gradient sharding over the dp
      group (reduce-scatter + all-gather), then parameter sharding; tensor
      parallelism inside a node over NVLink, sharded data parallelism across
      nodes. Gate: a full-parameter 27B fine-tune step that does not fit one card.
- [ ] **C4 serving across nodes.** A router capability in front of several
      `brain serve` endpoints, reusing `residency::affinity` (longest cached
      prefix, then least loaded), with health, drain and model listing merged,
      so a client sees one OpenAI-compatible URL for every card of an
      allocation: delivered (`brain serve --upstream`, see Delivered) and
      measured across nodes (Measured, 2026-11-21). Open: tensor-parallel serving
      inside a node for a model that does not fit one card; a cross-node layer
      split only for one that does not fit a node; multipart uploads
      (`/audio/transcriptions`) and job routes are not routed. The router
      keys prompts by the model's tokens (tokenizer read from the router's
      model store, text when absent) under a namespace carrying the adapter
      digest, drops an upstream's record when its instance id changes, and
      chooses by cost (`residency::affinity::replica_cost`: uncached prefill,
      load per slot including the upstream's own queue, KV headroom from its
      `/v1/health`). Open: upstreams report no cached-prefix or eviction
      events, so the index stays advisory; the tokenizer is loaded as
      byte-level BPE (`QwenBpe`), exact for that family and only consistent
      (not the model's ids) for others; the base-weight digest is not in the
      namespace (a base changes only with a restart, which the instance id
      catches). The upstream's KV and free-slot readings lag under load (refreshed
      between batch groups); KV-aware choice is unmeasured beyond showing no
      regression. A least-loaded-only policy was not measured. Connections are a bounded
      keep-alive pool per upstream, and a drain cuts what still runs at its
      deadline (`--drain-deadline-ms`, default 300000).
- [ ] **C5 cluster launch sample and gates.** `samples/` gains a Slurm launch
      sample (job plan and secret written once, scheduler rank mapped onto
      `--job-rank`, rank bound to its card's NUMA node, NCCL network plugin
      selected) and a cluster end-to-end suite that runs under the scheduler
      and skips by name without one.
- [ ] **C6 versioned weight publication for live servers.** Rollout servers and
      trainers as separate groups: a new policy (full weights or an adapter) is
      staged into inactive buffers, verified by size and digest, and activated
      atomically as version N+1; requests that started on N finish on N, and
      KV/prefix-cache entries are keyed by weight version so none crosses
      versions. Every generated sample records its policy version. Transfer by
      NCCL broadcast inside one step, by file otherwise. Gate: publication time
      per 27B full/LoRA update and zero mixed-version generations.
- [ ] **C7 node-local model staging and sharded checkpoints.** Each node's
      leader reads a base once from the shared filesystem into the node's
      job-private NVMe (`/scratch/local`, 1.7 TB per node, host side), verifies
      digests and serves its local ranks, instead of every rank reading the
      shared filesystem; sharded runs read only their shards. Checkpoints keep
      the committed-generation manifest and add the parallelism layout and
      per-rank RNG state, with loading into a different layout as a separate
      correctness test (logical tensor digests equal; training behaviour
      compared, not bitwise). Gate: time until all 16 ranks hold a 27B base.
- [ ] **C8 collective failure handling.** A communicator state machine
      (active, suspect, aborting, rebuilding) driven by op deadlines and async
      error polling; recovery is a job-level restart from the last committed
      checkpoint first, shrink/regrow after the transactional step is proven.
      Gate: fault injection that kills or stalls a rank during all-gather,
      reduce-scatter, the optimizer step, weight publication and a checkpoint
      write never leaves a partially applied step.
- [ ] **C9 qualification suite and traces.** One machine-readable result per
      run (hardware, every library version, configuration, p50/p99, algorithm
      and bus bandwidth, NIC octets from CXI telemetry, correctness verdict;
      unmeasured fields stay null) and a cross-node trace (per rank: forward,
      backward, each collective with group, bytes, bucket and wait time,
      optimizer, I/O; clock offsets recorded). Gates: every distributed test
      uses the intended fabric path (a socket fallback fails the test),
      collective results match a reference, multi-rank scaling reaches 80% of
      the measured topology reference for large transfers, and gradient
      overlap hides 70% of overlappable communication where compute allows.

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
