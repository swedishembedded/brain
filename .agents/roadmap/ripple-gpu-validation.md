<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# ripple-gpu-validation - roadmap

Scope: The ordered, runnable plan that takes Ripple-65B from "finished on the CPU backend and the software Vulkan adapter" to "validated and measured on GH200 cards": every step's command, the number it should reproduce and what a failure points to.

Swedish Embedded AB implements validation of large language models on the accelerators
they will be served from for its clients. If your team needs expertise in bringing a
65B hybrid-attention decoder from CPU parity to measured GPU serving then you can
procure our services by sending an email to info@swedishembedded.com.

Status of the model itself (what exists, what was measured, the open design items) is `ripple.md`; this
file only says what to run on a card and what to expect. Nothing below has run on a card: every `[ ]` is
unrun, and a step is ticked only with the date, the commit and the number recorded under it.

Conventions in every command: `$T` is a cargo target directory on the project drive
(`CARGO_TARGET_DIR=$T`, `CARGO_INCREMENTAL=0`, at most 25 GB), `$R=~/project/shared/models-qhf` the data
root (`--brain-data-dir $R`, models in `$R/models`; the home drive has 11 GB free), `$B=$T/release/brain`.
Temporary files go under `~/tmp` (mode 700), never `/tmp`. Server ports are 8798 and up, never 8788 to 8791.
Test-harness variables (`DEVICE`, `BACKEND`, `REQUIRE`, `RIPPLE_*`, `BRAIN_DEVICE`, `BRAIN_BACKEND`) are
not read by `brain serve`/`brain finetune`: those take flags.

## Entry points (all CPU-checked, none run on a card)

| entry point | what it does |
|---|---|
| `make test/gpu/ripple/suites` | lists the eight cargo suites and what each guards |
| `make test/gpu/ripple/cargo DEVICE=gpu0 BACKEND=cuda` | every Ripple-relevant suite on that device, one report per suite under `build/ripple-gpu/`; `REQUIRE=1` turns a skip by name into a failure; `BACKEND=vulkan` for the Vulkan path; `DEVICE=gpu0,gpu1` for staged placements |
| `make test/gpu/ripple/e2e DEVICE=gpu0 BACKEND=cuda` | `tests/e2e/ripple.bats` with every case, the sparse-attention ones included, on that device |
| `make test/gpu/ripple DEVICE=...` | both |
| `tools/goldens/ricursive_dump_reference.py full` / `full-check` | the layer-streamed reference of the whole 80-layer release (step 6) and its tiny-scale proof |
| `crates/ricursive/tests/it/full_release.rs` | replays the reference rows through the engine and prints mean KL, top-1 and load-to-ready per release, KV format and budget (step 6, 11) |
| `brain perf run longctx --target ricursive:DIR[:kvint8]` | decode tokens/s per batch size at a context and cold prefill per length, with the plan's bytes (steps 8 to 10) |
| `brain lsa distill|eval|prefill --arch ricursive` | D-LSA distillation, retention and sparse-vs-dense prefill timing (steps 10, 11) |
| `scripts/test/ripple-needle.py` | a passkey hidden at depth in unrepeating filler, asked of a served model (step 11) |
| `make test/perf/cuda`, `make test/parity`, `make cuda/coverage` | the repository's own CUDA gates, run once per card beside step 3 |

`make test/gpu/ripple DEVICE=cpu` and `DEVICE=software` run end to end GPU-free (2026-11-08): eight suites pass on cpu (ricursive 104, dflash2 8, catalog 16, sdk 14, gradcheck 3, model-shared
105, qwen3-lsa 13, qwen35-dflash2 4), the e2e file passes 16 of 16 on the default split and on `DEVICE=software` (case 9, the 5 GB partial checkpoint, skips by name there: lavapipe
caps a storage binding at 128 MiB). A card run must reproduce those counts; a count that falls is a test that skipped.

## Step 0: take the node and a card

The cards belong to other jobs while the node is shared. This session hides them (`~/tmp/no-gpu-env.sh`,
`CUDA_VISIBLE_DEVICES=""`); a card run starts in a shell that has not sourced it, and
`scripts/test/ripple-gpu.sh` refuses a `gpuN` device in a shell that has.

- [ ] `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv` lists nothing on the cards you will use, and
      `nvidia-smi --query-gpu=index,memory.used --format=csv` shows them near empty. A card with another job's context
      is not free: wait for it, never share it for a measurement (a neighbour's kernels divide every timing by
      the share it takes; kernels.md E.0a).
- [ ] Take the card. A throughput or profile measurement holds `flock ~/project/shared/containers/gpu.lock <command>` for
      the whole command and records `nvidia-smi pmon -c 5` before and after; it counts only if no other context
      exceeded 5% SM (the rule of `throughput-27b.md`). The per-card slot files `~/project/shared/containers/locks/gpu{0,1}.s{0..7}.lock`
      and `gpuN.turn` belong to the card helper the campaigns use (`card --card N --gb G --wait -- cmd`, a script under the
      campaign's `runs/` that has since been wiped): when it is not at hand, `gpu.lock` is the whole-card rule. brain's
      own device-open lock (`cross_process_gpu_lock`) only orders processes that open a device, it does not reserve a card.
- [ ] The job's RAM is a cgroup, not the node's `free` (215 GB on the n444 jobs while `free` shows the node's 858 GB; read the limit
      of the job you are in). Page cache of the mapped checkpoint counts toward it
      but is reclaimable; anonymous memory is what kills the job (an OOM took a brain server down on 2026-11-01). Cap
      cargo with `CARGO_BUILD_JOBS=16` and run no workspace-wide test. Steps that need more than 215 GB of host RAM are
      marked and need a job that has it (step 13, the 4-card full fine-tune: 378 GiB).
- [ ] Disk. `df -h ~/project/shared; df -i ~/project/shared`: on 2026-11-08 the project drive read **500 G of 500 G used
      and 500000 of 500000 inodes, 0 available**, under a 500 GB and 500k-file quota (earlier in the campaign about 100 GB
      was free). Nothing in steps 1 to 14 can be written until space and inodes are freed; step 1 states how much.
- [ ] Build the release binary where the quota allows: `CARGO_TARGET_DIR=$T CARGO_INCREMENTAL=0 make build/release`, then
      `make build/profiling` into the same `$T` only when a step asks for line tables.

## Step 1: downloads, sizes against the disk

| artifact | size | where | verdict |
|---|---|---|---|
| NVFP4 release `ricursiveAI/Ripple-65B-Preview-NVFP4` | 57,213,636,640 bytes in safetensors (1923 tensors) | `$R/models/ricursiveAI/Ripple-65B-Preview-NVFP4` | fits once about 60 GB and a few hundred inodes are free |
| bf16 release `ricursiveAI/Ripple-65B-Preview` | 129,565,378,720 bytes (16 shards, 1203 tensors) | `$R/models/ricursiveAI/Ripple-65B-Preview` | does **not** fit beside the NVFP4 release (130 + 57 GB against a 500 GB quota that is full); needs 135 GB freed. It is only needed for bf16 weights on 2 cards (step 9) and the bf16 rows of step 6 |
| DFlash2 drafter | 8,075,513,104 bytes | already at `~/project/shared/models-qhf/ripple-65b-preview-dflash2` | present; pass it with `--draft-model` (a directory), no pull needed |
| real layers 0 and 4 (bf16, NVFP4) | 5.2 GB, 3.4 GB | `testdata/ricursive/ripple-65b-partial` -> `$R/ripple-65b-partial` | present; the real-weight tests read them |

- [ ] Free space: `du -sh` the largest directories of the drive and remove superseded runs and caches first; the
      headroom wanted is NVFP4 57 GB + a build target (up to 25 GB) + 10 GB of dumps = about 100 GB, and 135 GB more for the bf16 release.
- [ ] `$B pull ricursiveAI/Ripple-65B-Preview-NVFP4 --brain-data-dir $R` (resumable per file). `testdata/ricursive/ripple-65b-nvfp4` is a link to its directory.
      Expect 1923 tensors, `total_size` 57,213,636,640, and the startup log's `static activation scales validated`.
- [ ] The bf16 reference needs **no disk** (step 6): the dumper reads the bf16 release by HTTP range request, one tensor at a time. Pull the bf16
      release (`$B pull ricursiveAI/Ripple-65B-Preview --brain-data-dir $R`) only when 135 GB are free and step 9 or the bf16 rows are wanted;
      the alternative to freeing space is per-layer goldens of the layers in question with `~/resources/tools/hf_fetch_tensors.py REPO OUT 'model\.layers\.(0|4)\..*|...'`
      (the way layers 0 and 4 were fetched).

## Step 2: the CPU and software baseline on this build

- [ ] `make test/gpu/ripple DEVICE=cpu` and `make test/gpu/ripple/cargo DEVICE=software` (and `make test/gpu/ripple/e2e DEVICE=software`)
      in the card-hiding shell, on the commit under test. Expect the counts above. These are the numbers the card must reproduce; keep the logs
      for a diff against step 3.
- A failure here is not a GPU finding: fix it first. One known flake: `tests/e2e/ripple.bats` case 6 (speculation reports its rounds) failed once under
  heavy host load because the request line was read before the metrics were logged, and passed on the rerun; a port in use by another job
  (`RIPPLE_PORT=...`) fails the case that binds it.

## Step 3: device gates on CUDA

- [ ] `make test/gpu/ripple/cargo DEVICE=gpu0 BACKEND=cuda REQUIRE=1`, then the same with `DEVICE=gpu0,gpu1`.
  Expect: all eight suites pass with the counts of step 2, `skipped by name: 0` for every suite. These tests ran only on the CPU
  backend or skipped there and are the first to try on a card: `ricursive` tests `a_budget_past_the_context_equals_dense_through_prefill_chunks_and_decode`
  (full-budget LSA equals dense, tolerance 1e-5; 6.6e-7 on lavapipe), `a_small_budget_reads_fewer_keys_than_dense_attention`,
  `prefix_reuse_and_batching_match_cold_and_alone_runs_with_sparse_attention`, `sparse_attention_at_a_full_budget_decodes_real_layers_0_and_4_as_the_reference`
  (needs `testdata/ricursive/ripple-65b-partial`; tolerance 5e-5), the three real-weight tests lavapipe could not bind
  (`real_layers_0_and_4_match_the_reference` 5e-5, 6.2e-6 measured), and the `brain-model` `paged_sparse` device tests.
- [ ] Shared-kernel regressions the port changed, each already green on lavapipe, each to be watched by name when a suite fails:
  - the fused paged decode `GMAX` change (heads per KV head past 8; `fused` tests and `model` `kv_tier::` tests): f64 attention
    oracle 1.3e-6 to 3.1e-6 relative at f32/bf16/int8/fp8; engine fused vs triad f32 1.5e-6, bf16 2.4e-4, int8 5.3e-4, fp8 1.4e-6.
    A failure on one tier only points at that tier's row-scale variant (`HDS`) or `GMAX` rounds; on all of them, at the widened softmax.
  - the sparse whole-row geometry (`paged_sparse::rows_geometry`, 80 heads = 160 threads, a 4-key tile in 16 KiB, and the per-KV-head kernel at group 10):
    a `create_bind_group`/workgroup-size error means the card's limits, a wrong value the tile shrink.
  - the dynamic convolution `block` change (`dyn_group_conv1d`, several blocks per dispatch; `brain-dflash2` and `qwen35-dflash2`): both must pass, a split
    result (Ripple's draft passes, the Qwen3.8 draft fails) points at the per-block indexing.
  - Qwen3's engine on the shared LSA binding (`qwen3-lsa`) and the Qwen3.8 draft through `dflash2` are the regression guards of code both models share.
- [ ] The int8 adapter case (`train_tiers::an_adapter_over_a_packed_base_trains_like_one_over_its_decoded_values`), with `--nocapture`:
  expect `I8: loss 4.208457 vs 4.2068405`, `worst adapter gradient error 2.61e-1 (model.layers.0.mlp.gate_proj.weight.lora_a)`, then
  `I8: global gradient error 1.280e-1, cosine 0.9921`. These are the numbers seen on a GH200 once and **reproduced bit for bit on the software
  Vulkan adapter on 2026-11-08**: the failure was never card-specific. It is the W8A8 tier (the shared op quantizes activations to int8 as serving
  does) and the test now holds it to a bounded drift. A card number different from these is a finding: it would point at the card's int8 dot
  (DP4A) against lavapipe's. Whether training over int8 should read f32 activations instead is a design question in `ripple.md` (later).
- [ ] `make test/parity` and `make test/perf/cuda` on the same card, then `make cuda/coverage`; compare with the floors on file. A CUDA-only gap in the
      kernel list (kernel refused by name) is listed per kernel in `cuda-coverage.json`.

## Step 4: device gates on Vulkan

- [ ] `make test/gpu/ripple/cargo DEVICE=gpu0 BACKEND=vulkan REQUIRE=1` (native Vulkan on the card, not lavapipe). Expect the same counts. The
      Vulkan device caps a storage binding below the 200 MiB weight matrix on some drivers: a `create_bind_group` failure in the three real-weight
      tests is that limit (record the limit; those tests stay CUDA gates), anything else is a kernel.
- [ ] A CUDA pass with a Vulkan fail (or the reverse) on one test is the finding; run that test alone with `make test/one P=brain-ricursive T=it ARGS="NAME --nocapture --test-threads=1"` under `BRAIN_DEVICE=gpu0 BRAIN_BACKEND=vulkan`.

## Step 5: the e2e file on the card

- [ ] `make test/gpu/ripple/e2e DEVICE=gpu0 BACKEND=cuda BRAIN_BIN=$B` (16 cases; the real-layer case runs on the card, 5 GB of bf16, expect a prompt over 150 tokens answered with 3).
      A case that fails here and passes at step 2 is a card finding. Then the same with `BACKEND=vulkan`, and `DEVICE=gpu0,gpu1` (the two-stage LoRA case then
      cuts across cards: loss equal to the one-stage run within 1e-3 of itself, the layout line `pipeline of 2 stages`).

## Step 6: the 80 layers against the official reference

The dumper needs no card for correctness: the reference is `modeling_ricursive.py` run layer by layer, so one 1.5 GB layer is ever resident.
It needs PyTorch with `transformers` and `safetensors`; the `venv-edgerock` one is CPU-only (fine: about 2.7e14 FLOP for 2048 tokens), a CUDA build of torch
for aarch64 (`sbsa`) makes it minutes on a card. The checkpoint is read over HTTP (the 130 GB never touches the disk) or from a directory.

- [ ] Prove the dumper on this machine first: `python -I tools/goldens/ricursive_dump_reference.py full-check --modeling-dir ~/resources/models/ripple-65b-preview --out ~/tmp/ripple-full`.
      Expect `worst scaled error 0.000e+00`, `reads per tensor [1]`, `resident layers 1`, `over HTTP range requests ... 0.000e+00`, `without the embedding scale the logits move by` about 1.
- [ ] A prompt: 2048 tokens of English and code (`head -c 12000 <some source files> > ~/tmp/prompt.txt`).
- [ ] The dump, bf16 as the release is served:
      `python -I tools/goldens/ricursive_dump_reference.py full --modeling-dir ~/resources/models/ripple-65b-preview --out testdata/golden/ricursive --checkpoint-dir https://huggingface.co/ricursiveAI/Ripple-65B-Preview/resolve/main/ --tokenizer testdata/ricursive/ripple-65b-partial/tokenizer.json --prompt-file ~/tmp/prompt.txt --max-tokens 2048 --device cuda:0 --dtype bfloat16 --logits-rows 256`
      (`--device cpu` where no CUDA torch exists; add a second dump with `--dtype float32 --out testdata/golden/ricursive-f32` when a parity number is wanted.)
      Expect: `streamed 120.7 GiB through one resident layer`, `golden/ricursive/full/ricursive_full.safetensors` with `logits` [256, 100352] and `hidden.0 .. hidden.80`
      (the last 16 rows each), and a manifest whose `source.identity` matches `config.json`. The `source` check of the Rust side refuses a dump of another model.
- [ ] Link the engine's inputs: `ln -s $R/models/ricursiveAI/Ripple-65B-Preview-NVFP4 testdata/ricursive/ripple-65b-nvfp4`, and (disk permitting)
      `.../Ripple-65B-Preview testdata/ricursive/ripple-65b-bf16`.
- [ ] The dense matrix, KL and top-1 against the bf16 reference at bf16/int8/fp8 KV, with the load-to-ready time of each engine:
      `BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda BRAIN_REQUIRE_FIXTURES=1 make test/one P=brain-ricursive T=it ARGS="full_release --nocapture --test-threads=1"` (NVFP4 on one card; the bf16 release needs `BRAIN_DEVICE=gpu0,gpu1`).
      The staged placement is the same test under `BRAIN_DEVICE=gpu0,gpu1`: the `ripple-full: placement` line shows the stage layout and its KL must equal the one-card row's to the printed digits.
  Expect (reference points, CPU/lavapipe, tiny and two-layer models; an 80-layer model compounds them, so up to a decade worse is normal and more is a finding):
  | configuration | reference point | flag when |
  |---|---|---|
  | bf16 weights, bf16 KV | mean KL 9.7e-5, top-1 93% (tiny, bf16 tier); the engine itself reproduces its own f32 decode to 1e-5 | KL above 1e-2 or top-1 under 90% |
  | int8 KV | KL 1.4e-4, top-1 93% (tiny) | KL 10x the bf16-KV row |
  | fp8 KV | KL 2.4e-3, top-1 87% (tiny) | KL 10x the int8-KV row |
  | NVFP4 release (W4A16 here; the release was calibrated W4A4) | KL 3.6e-3, top-1 87.8%, hidden relative L2 4.2e-2 after layers 0 and 4 | KL above 0.1 or top-1 under 70% |
  The harness itself asserts only sanity floors (KL < 1, top-1 >= 50%); the first card run's numbers become the bounds. A KL that is large for the
  bf16 release at bf16 KV points at the engine or the dump (compare `hidden.K` row by row to localize: `tools/goldens/ricursive_dump_reference.py` writes
  `hidden.0 .. hidden.80`), the same large for NVFP4 only at the tile decode. LSA rows are step 11.

## Step 7: placement against the card

- [ ] Serve the NVFP4 release on one card and compare the startup placement line with `nvidia-smi` after the first request:
      `$B serve --brain-data-dir $R --openai 8798 --api-keys-out ~/tmp/keys.json --ready-file ~/tmp/ready --device gpu0 --kv-cache int8 --verbose 2`.
      The plan says 52.3 GiB of weights on the device (the embedding is gathered on the host), 57.0 GiB at int8 KV and 128k context, and about 921k
      tokens of int8 context on one card with the index charged. Expect `nvidia-smi` within the allocator's 14% (`capacity::headroom`) of the plan; the fixed per-stage
      bytes (0.5 to 1.0 GiB) are an estimate to be replaced by the measurement.
- [ ] Load-to-ready: `date +%s.%N` before the start and when `~/tmp/ready` appears. There is no CPU reference point; record it per tier (bf16 two cards,
      NVFP4 one card) and compare with vLLM's cold start at matched settings (step 15). The 2-layer model on the CPU backend was ready in 2 s.

## Step 8: serving on 1 GH200 (NVFP4, int8 KV, D-LSA, DFlash2): tokens/s against the roofline

The roofline for decode is bytes over the measured read bandwidth, never a table value: bandwidth from the kernel gap list of the roofline
microbenchmark (`kernels.md` F.2), bytes from the placement (weights 52.3 GiB per token at batch 1, plus the KV a token reads).

- [ ] Plain decode, batch 1, 8, 64, at 8192 context, kernels profiled:
      `flock ~/project/shared/containers/gpu.lock $B perf run longctx --target ricursive:$R/models/ricursiveAI/Ripple-65B-Preview-NVFP4:kvint8 --device gpu0 --context 8192 --ladder 1,8,64 --steps 16 --kernels --out ~/tmp/ripple-longctx.json`.
      Expect a table with the plan's `needed_bytes` beside each fit and a per-kernel-kind profile. Record tokens/s per batch and the fraction of the bandwidth
      bound; a fraction under the gap-list threshold of the roofline probe is a kernel item (a new candidate, never a hand-tuned path). Batch 64 at 8k context needs 64 sequences
      of KV: if the sweep ends early the boundary is the finding.
- [ ] The timeline and the kernels: `nsys profile -t cuda,osrt,nvtx --cuda-graph-trace=node -o ~/tmp/ripple-decode $B serve ...` for one request, and `ncu -k regex:gemv -c 20`
      on the focused kernel test binary of the kernel under suspicion (`kernels.md` F.0); never `ncu` through the server. Binaries at `~/project/shared/containers/nsight/{nsys,ncu}`.
- [ ] Decode with the distilled indexer (step 11) and with speculation: the same command with a server started with `--sparse-attention FILE`, and `--speculative dflash` (step 9).
- A failure: a brain-level number far under the bound at batch 1 with a clean profile points at the NVFP4 GEMV (`matmul_nvfp4_gemv`; the CUDA native NVFP4 kernels were never validated);
  a profile with the card idle between kernels points at the host (F.2c: allocator, readbacks).

## Step 9: DFlash2 on the real target, and the two open conventions

- [ ] First, on a tiny and real draft against the reference on the card: `BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda make test/one P=brain-dflash2 T=it ARGS="--nocapture --test-threads=1"`,
      then the `real` forward with `testdata/dflash2/ripple-65b-preview-dflash2` and the golden of `tools/goldens/dflash2_reference_forward.py real`. Expect the CPU numbers:
      layers 0 to 4 scaled error 3.2e-6 .. 2.5e-5, head logits 2.7e-5, the selector's 15-token path identical.
- [ ] Equal replies: the same request twice, plain and `$B serve ... --speculative dflash --draft-model ~/project/shared/models-qhf/ripple-65b-preview-dflash2 --verbose 2`; greedy replies byte-equal.
      A difference is an engine bug (verify/rollback), not a convention.
- [ ] Acceptance. `spec_accept_rate` and `spec_tokens_per_round` in the `request done` lines, over 30 prompts (chat, code, tool call), against plain decode at batch 1 and 4. Reference point: a random draft
      accepts 0 of 19 proposals; prompt lookup (`--speculative ngram`) is the floor to beat.
- [ ] The two conventions the published files do not settle, tried in this order, each a one-line change and a rebuild, the winner being the one with the highest accepted tokens per round, a wrong one showing as the random draft's near-zero:
  1. taps: `RicursiveEngine::enable_taps(&cfg.target_layers)` (`crates/ricursive/src/serve.rs`, `set_drafter`) taps the output of decoder layer `i` (current); the alternative is layer `i + 1`
     (`l + 1` for each tap, the stream entering layer `i + 2`).
  2. noise embedding: `DraftHost::embed` for `RicursiveEngine` (`serve.rs`) reads the unscaled `embed_tokens` row (current, `embed_unscaled`); the alternative is the scaled row (`crate::model::embed`, times `embed_scale` 71.554).
  All four combinations at batch 1 on the same prompts. Record the table, make the winner the code and `docs/reference/models/ricursive.md`'s sentence, and delete the flag-free alternative.
- [ ] Memory: the draft's 7.42 GiB and its compact caches in the placement line against `nvidia-smi` (0.29 GiB of caches at 128k with int8 KV).

## Step 10: serving on 2 GH200 (bf16) and long prefill

Needs the bf16 release on disk (step 1: 135 GB).

- [ ] `$B serve --brain-data-dir $R --openai 8798 --device gpu0,gpu1 [--kv-cache int8]`: the placement line expects stages of 63.6 + 64.8 GiB (bf16 KV, 128k) or 76.6 + 77.8 GiB (int8, 1M);
      the peer copy against the host-staged boundary (the log names which path the pair used), bit-equal logits to one stage on the CPU backend, to be re-checked: step 6's staged row.
- [ ] `perf run longctx --target ricursive:...Ripple-65B-Preview:kvint8 --device gpu0,gpu1 --context 8192 --ladder 1,8,64`; expect lower tokens/s than step 8 per the roofline (bytes 2.2x), no pipeline overlap at batch 1.
- [ ] Prefill at 32k, 128k, 512k and 1M, dense and with the index: `$B perf run longctx --target ricursive:... --device gpu0 --context 8192 --ladder 1 --prefill 32768,131072,524288,1048576`. The dense engine holds at most 962k tokens on one card (NVFP4, int8 KV);
      1M needs two cards (`--device gpu0,gpu1`, 43.6 + 43.4 GiB). Record seconds and tokens/s per length; the dense prefill is quadratic in the global layers, the sparse one is linear past the crossover.
- [ ] LSA crossover: `$B lsa prefill --arch ricursive --checkpoint DIR --indexer ripple-lsa.safetensors --tokens 4096,8192,16384,32768,65536,131072 --kv-cache int8 --device gpu0` prints dense and sparse prefill per length and `set_lsa_dense_until` follows the measured crossover. Reference points: Qwen3 at 8 to 12k tokens (P40); Ripple's 80 heads over 8 KV heads are
      group 10, so a different number is expected: record it and the engine's own `measure_crossover` value.

## Step 11: D-LSA distillation of the real indexer, then retention

- [ ] 32k: `$B lsa distill --arch ricursive --checkpoint DIR --corpus <directory of source files> --out ~/tmp/ripple-lsa-32k.safetensors --tokens 32768 --train-docs 48 --epochs 3 --queries 64 --top-k 1024 --top-p 256 --device gpu0`
      (NVFP4 release, f32 KV for the capture: 128 KiB a token over the 16 global layers, 4 GiB at 32k). The teacher costs about 12 s per 32k sample across 16 cores (190 s on one thread); expect the run to take under an hour. Then 128k with `--tokens 131072`: the f32 capture cache is 16 GiB on the card, the keys are 134 MB a layer at 32k (four times at 128k) on the host and only the
      sampled queries are kept, so host RAM stays far under the cgroup; the host teacher is quadratic in the keys (about 0.19 s a query at 32k), so expect hours at 128k, not minutes.
- [ ] Retention: `$B lsa eval --arch ricursive --checkpoint DIR --corpus <dir> --indexer random,pca,~/tmp/ripple-lsa-32k.safetensors --tokens 32768 --skip-docs 48 --held-docs 6 --device gpu0`.
      Expect retained share distilled >= pca >= random (CPU, real layers 0 and 4: 0.513, 0.513, 0.509 against 0.217 for window plus sinks; tiny model: distilled beats random by +0.006 to +0.015) and a next-token loss gap to dense that shrinks from random to distilled (tiny: 0.211 to 0.155).
      On 16 real global layers the margin should be larger than on the one layer measured; if distilled does not beat random the capture or the objective is wrong, not the data.
- [ ] The sparse rows of step 6: re-run `full_release` with `testdata/ricursive/ripple-lsa.safetensors` linked to the artifact: it adds the NVFP4/int8-KV rows at the artifact's budget and at a quarter of it. Reference: a budget past the context equals dense (3.4e-7, 4.2e-7 on lavapipe).
- [ ] Needle: serve with and without the index (`--kv-cache int8`, with `--sparse-attention`), then `python3 -I scripts/test/ripple-needle.py --url http://127.0.0.1:8798 --model ricursiveAI/Ripple-65B-Preview-NVFP4 --keys ~/tmp/keys.json --lengths 8192,32768,131072 --depths 0.05,0.5,0.95 --trials 5`.
      Expect the dense server to find the passkey (a 65B instruct model at 128k); the sparse server may lose it where the retained share is low: the table is the retention result in task terms, the budget is the knob.
      No CPU number exists for this probe; its plumbing was run against the scripted tiny model.

## Step 12: training on cards

All of these ran on the CPU backend and the software adapter at tiny scale and on real layers 0 and 4; none on a card. Order: the small gates, then one real step.

- [ ] The card gates first, each against the CPU number in `ripple.md`: `make test/one P=brain-ricursive T=it ARGS="train_resident train_saved train_lora train_pipeline train_tiers::a_chunked_head --nocapture --test-threads=1"` under `BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda`,
      then `BRAIN_DEVICE=gpu0,gpu1` (already part of step 3). `train_real_lora` with the partial checkpoint linked: expect loss 13.992954 and gradients to 4.3e-5 (CPU: 47 s forward and backward).
- [ ] LoRA over the NVFP4 release on 1 GH200 (52 GiB of weights, adapter state and one row of activations small beside it), tokens/s and the roofline fraction from `--kernel-profile`:
      `$B finetune ricursiveAI/Ripple-65B-Preview-NVFP4 --brain-data-dir $R --arch ricursive --dataset chat.jsonl --out ~/tmp/run-lora --lora 16 --alpha 32 --base-format nvfp4 --activations recompute --steps 20 --lr 1e-4 --max-block 4096 --device gpu0`
      Expect the loss to fall on a small dataset (CPU tiny: 4.1951 -> 0.0551 in 6 steps); record tokens/s (`kernels.md` F). QLoRA over nf4/fp8 from the bf16 release: `--base-format nf4|fp8` on two cards (`--device gpu0,gpu1`, layout printed).
- [ ] The adapter served beside its base: `$B serve ... --adapter ~/tmp/run-lora/adapter.safetensors`; a trained adapter's logits equal the training forward's to 2.3e-7 (CPU) and the bank's memory (3.75 GiB for four rank-16 slots) against `nvidia-smi`.
- [ ] RL: `--policy-gradient --held-out heldout.jsonl` (rollout records with `advantage`) and a verifiable-reward GRPO loop through the SDK trainer (`rl objective_fits`): expect reward up from the first step, the first-step ratio 1 +- 1e-4 (CPU: 0.534 -> 0.844 on a tiny model).
- [ ] N-LSA crossover (the global layers read their selection only): `--native-sparse --lsa-window 4096 --lsa-sink 4 --lsa-top-k 16384 --lsa-top-p 1024 --lsa-cluster-block 512 --max-block 65536` against the dense flash path at head dimension 128, step time at 8k, 16k, 32k, 64k; the Qwen3 crossover near 8k is the
      only reference. Sparse-budget pipeline stages are built but only checked at one stage against dense.
- [ ] Full fine-tune of the bf16 release on **4 cards** (92.2 of 95.6 GiB a card at 4096 tokens, 378 GiB of host RAM for master and int8 state: **more than a 215 GB cgroup**; use a job with >= 450 GB):
      `$B finetune ricursiveAI/Ripple-65B-Preview ... --stages 4 --train-precision weights=bf16,state=int8,grad=bf16 --max-block 4096 --steps 20 --lr 1e-5 --device gpu0,gpu1,gpu2,gpu3`.
      One step first with `--kernel-profile` (host transfers and the host update are the cost to read first). The 1-card variant (`weights=bf16,window=2,state=int8,grad=bf16,clip=two-pass`) needs 370 to 490 GiB of host RAM. The plan's numbers are not measured.
- A failure: gradients that differ from the CPU numbers on a card but not on lavapipe point at a CUDA kernel of the backward; losses equal but slow, at the head (`training head`) or the host.

## Step 13: agentic task evals

Out of brain's scope (agent loops and task scoring are not the engine's). Hand-off: serve the checkpoint at its best measured configuration from steps 8 to 11 (`--openai`; the chat template and the tool-call
reply parser are gated by step 5's e2e file) and point an agent evaluation harness outside this repository (terminal-task suites of the TerminalBench kind) at it over the OpenAI API under the model id the
server lists. brain reports what the run needs from the server (tokens/s, speculation metrics, `lsa_*` read counters); the task score belongs to the harness.

## Step 14: matched comparison against vLLM and SGLang

Only where they support Ripple at all: SGLang ships a native port (`sglang_ricursive`, read for the reply format, never run here); vLLM has no known support, check at the time. Compare only at identical settings: same checkpoint release,
same weight format (NVFP4 W4A4 there against W4A16 here is a difference to state, not to hide), same KV format (int8 or fp8 against the same), same speculation (DFlash2 against DFlash2), same context, same batch, same prompts, a cold start each, on one
free card under the lock. Report tokens/s per batch, load-to-ready, and precision as KL and top-1 against the bf16 reference of step 6 for both stacks (the dump is engine-independent). A faster number at a different precision is not a result.

## Step 15: record and release

- [ ] Each ticked step: the date, the commit, the number, the log under the evidence repository (never in this tree; the run logs of `build/ripple-gpu/` and the reports under `~/tmp` are disposable).
- [ ] Update `ripple.md` (Plan step 8, Open items) and the two model pages, lower the thresholds above to the measured values in the tests that assert them, and release the card and the lock. Delete `~/tmp/ripple-*`.
