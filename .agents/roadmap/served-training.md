<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# served-training

Scope: Training as a job of the serving process (`crates/trainjob`): `brain/train`, its artifact store, scheduling with serving residents, the job event stream.

## Done

- `brain/train` / `train`: supervised (`sft`) and policy-gradient (`policy_gradient`, teacher contexts included) fine-tuning of a chat LoRA through `brain::ChatFineTune`, the same settings the SDK takes; qwen35 first, qwen3 for the supervised objective (the policy-gradient objective is qwen35 only, as in the SDK). `brain serve --training [--artifact-store DIR]`.
- Datasets and the starting adapter by input blob (D-Bus file descriptors) or by blake3 digest in the operator's artifact store (`--artifact-store`); the adapter is published to the store and named by digest, and returned as an output blob when `return_adapter` is set. A digest is 64 lowercase hex digits and the server copies the file into the job's private directory while verifying it. No caller-supplied path exists.
- Progress: one `step` frame per optimiser step (loss, lr, grad norm, seconds; `pg` with clip fraction, KL, importance ratio, advantage statistics and the scoring seconds for a policy-gradient run) over D-Bus `Subscribe` progress frames and over HTTP `GET /v1/jobs/{id}/events` (SSE, `Last-Event-ID` replay of the last 4096 frames, 8 followers per job). `Cancel` / `DELETE /v1/jobs/{id}` stop the run at the next step.
- Scheduling (`residency::ResidentModel::{transient, check_job}`, `Executor::check_job`): the run reserves its footprint (`ChatFineTune::footprint`: base at its tier, activations of one `max_block` row, adapter and optimiser state, a margin), takes the card with the most room or evicts idle serving instances on the best candidate, never an instance a request is running on, waits for the admit deadline and is then refused (429); a run larger than the largest card is refused at the door; the reservation returns when the run ends. One run per card.
- A call a model refuses at the door (`check_job`) is a 400 on HTTP and a bus error, before a job exists. `/v1/run` and `/v1/jobs` bodies refuse unknown fields.
- Tests: `crates/trainjob` (store, request, door refusals, real runs, scheduling, HTTP, D-Bus), `crates/apiserve/tests/capabilities.rs` (event stream), `tests/e2e/api_conformance.bats` (flags, auth, refusals), `crates/residency` (transient, check_job).

## Adapter hot-load (lane rollout, 2026-11-13)

- [x] `--serve-adapters DIR`: every `*.safetensors` adapter under a directory is offered as a model of its own and the offered set follows the files (appear, renamed over, disappear) with no restart; symlinks, hidden files and other names ignored, depth 4, 1024 files, each file read once, size-capped and verified against the base. Gated on the CPU in `crates/cli/src/adapter_directory.rs` (follow, refuse without replacing what serves, id moves).
- [x] A policy version per adapter id (`AdapterIdentity::version`): 1 for the first release, +1 for a replacement with different weights, kept across withdrawal; in `/v1/models` (`adapter.version`), in every reply's `brain.adapter` and in the policy trace's `policy.adapter`. A refit never reads the previous release's prefix-cache entries (one namespace per release digest): `a_refit_adapter_never_reads_the_previous_releases_cached_prompts` (`crates/catalog/tests/it/multi_adapter.rs`).
- [x] `--adapter-rank-cap` defaults to 32; a refusal names the cap to restart with. The cap is fixed when the engine is built (the bank's buffers are bound into recorded tapes), so an adapter above it cannot be hot-loaded.
- [x] Real-weight check (Qwen3.8-27B-FP8, one GH200, 2026-11-13): three rank-32 all-target adapters hot-loaded onto the running server by renaming files into the `--serve-adapters` directory (listed with `version` at once, served concurrently in one batch under their own ids), a `brain/train` job admitted at once after ~600 adapter requests (knowledge 438, the stall does not reproduce).
- [ ] The adapter bank's decode cost on the 27B at 8/32/64 rows is measured (knowledge 444: -50 %, -61 %, -66 % tokens/s with every row on one adapter, 13/26/47 ms extra per step) and the cause is profiled (the row-wise `lora_xa_rows` / `lora_delta_rows_multi` re-read the adapter for every row, L2-bound); a row-tiled kernel variant (`KernelVariant::RowTiled`, one-slot banks, from 2048 output pairs) landed and moves it to -53 / -55 / -63 % (knowledge 449); the launch-bound remainder needs a fused or library-GEMM design.
- [ ] At `--max-batch 64` the 27B on one 96 GB card needs `--reserve-gb 16`; at the default the graph launch runs out of memory and the model goes unservable (knowledge 445).

## Open

- Multi-card runs: the qwen35 trainer is one card per run; a run that needs two cards (a bf16 27B on 80 GB cards) is refused with what it needs. The pipelined qwen3 trainer could be placed over several cards through `register_multi`.
- Fair queueing: a run waits for idle serving; steady light traffic on every card can keep it waiting past the admit deadline. A scheduler-side priority for admitted jobs is not built.
- The footprint is a model (base bytes by tier, calibrated activations, a margin), not a measurement; a first run on a new base should be watched against `nvidia-smi`, and the estimate corrected if it is off.
- Real-weights runs of Qwen3.8-27B-FP8 through the service (policy gradient, rank 16, fp8 base, 8 records a step, 2026-11-12, knowledge #440) have run at lr 1.2e-5 to 3e-5, every one with `behaviour: start_of_run_proxy`: no serve-versus-train numerics measurement existed. `brain finetune --policy-gradient --numerics-gate` (SDK: `PolicyGradientConfig::numerics`) is that gate, run so far on Qwen3-4B (#446); the 27B needs `policy_trace` on its served engine first. The tests of the service itself use a toy `qwen35` on the CPU backend.
- The `brain/train` action takes the step guard and the clip-higher and importance settings (`max_step_kl`, `max_step_grad_norm`, `clip_eps_high`, `importance`) but not the numerics gate, which is a `brain finetune` run (no update, nothing to publish).
- The per-step PG series rides D-Bus progress frames, which are droppable; the complete series is the `record` blob.
