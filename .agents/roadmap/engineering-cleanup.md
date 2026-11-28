<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# engineering-cleanup

Scope: parameters declared once across the RL objectives, the training-job API, the SDK and the CLI; one max-batch policy for every served language model; the environment reads still left; the published training-records contract; and the test-binary layout.

Audit of 2026-11-21 against brain 6d9a31d14 (read-only; every file and line below was read). Execution waits for the
safety nets each item names. Prefer deleting a declaration over adding a layer that keeps two in step.

## Order

EC-1 first (it also closes a resume-identity hazard; done). EC-2 after EC-1 (it changes the type EC-1 tabulates; done). EC-3
independently. EC-4 is incremental, by class. EC-5 and EC-6 any time.

## EC-1 One table for the policy-gradient settings (M, done 2026-11-24)

- **Problem (was).** `clip_eps_high` and every sibling were declared by hand in seven places: the config, the served
  param spec with its guard list, wire list and field-by-field layering (`trainjob`), the SDK's run identity and
  `settings_json`, and the CLI's `Args`, hand parser and struct literal. A field missed in one list was silently not
  settable over the API, not reported, or, worst, not part of the identity a resume is keyed by.
- **Done.** `rl::objective::pg_settings` (beside `PolicyGradientConfig`, re-exported as `brain::policy_gradient_settings`)
  holds one row per field: name (= field = wire param = settings key), served help and range or choices, CLI flag,
  identity and report encoding, and an off-default example. Generated from it: the served params, their implied
  defaults and the "policy gradient only" guard (`trainjob/src/request.rs`), the layering onto the config (a
  `PgSettingsPatch` of wire values, `trainjob/src/job.rs`), the SDK's identity and `settings_json`, and the CLI's
  parser and config (`cli/src/finetune_cli.rs`). `FIELDS` destructures the config without `..`, so a new field does not
  compile until listed, and the table's test then asks for its row. The aggregation and its constant
  (`loss_agg_length`) resolve through one `LossAgg::resolve` for both front ends.
- **Deviations from the target, by the user's constraints.** Flags keep their names (`--ratio-clip` for `clip_eps`,
  `--ratio-clip-high`); the table maps field to flag, renames wait for the major release. The identity is not the
  canonical JSON of the config: each existing row keeps its old encoding (`_bits`, the debug form of `loss_agg`,
  serde of `importance`, `numerics_gate`), so every run that could be started before resumes; a row added later is
  `WhenNotDefault` in the identity and the report (joins only off its default). There is no derived Option-per-field
  patch type (the workspace has no proc-macro crate): the patch is the table's own name-to-wire-value list, which is
  what both front ends read. Explicit exclusions, each with its reason in the row: `log_kl` from the identity and the
  report, `numerics` from the report and the served action, `kv_roundtrip` and `teacher_tokenizer` from the served
  action, `reference`, `temperature`, `teacher`, `advantage_bound`, `sequence_advantage_cap` and `log_kl` from the CLI
  (never offered there; a flag is one row's `Flag::Named`).
- **What proved it.** Pinned before the change on 41aec7d47 and unchanged after: the SHA-256 of the identity and the
  settings of the default, an every-field-off-default and the DAPO configuration
  (`sdk/src/finetune.rs` `the_policy_gradient_identity_and_settings_are_pinned`), and the blake3 of the served params
  `algorithm`..`advantage_norm` (`trainjob` `the_served_policy_gradient_params_are_pinned`). New: every field has one
  row and every omission a reason; every row round-trips its example through its own field and moves the identity and
  the report (`rl` `pg_settings` tests); every served setting's example reaches the resolved config and nothing else,
  and is refused for sft by name (`trainjob`); every named flag likewise (`cli`). The existing `request.rs` tests and
  `trainjob/tests/it/run.rs` pass unchanged.

## EC-2 Online GRPO and the recorded-rollout objective share one loss configuration (M, done 2026-11-24)

- **Problem (was).** `GrpoConfig` (with `GrpoOptions`) and `PolicyGradientConfig` configured the same loss with two
  types: online GRPO built its band without the dual clip and had no step guard, the guard that removed the damage of
  record-driven ratio spikes in the recorded-rollout path.
- **Done.** `rl::objective::pg_loss::PgLoss { clip_eps, clip_eps_high, dual_clip, kl_beta, log_kl, loss_agg,
  advantage_norm, advantage_floor, max_step_kl, max_step_grad_norm }` (re-exported as `brain::PgLoss`) is embedded as
  `loss` in both `GrpoConfig` and `PolicyGradientConfig`; its `band()`/`spec()` build the clip band (dual clip
  included) once, its `validate()` is both objectives' range check, and its `StepGuard` (with
  `MAX_CONSECUTIVE_GUARD_SKIPS`) and `gradient_scan` are the one step guard. What differs stays in the objectives:
  sampling (`GrpoOptions`: resampling, staleness, ratio basis, routing replay) against records (importance correction,
  behaviour source, reference, teacher, truncation, numerics gate, KV round trip). `Grpo` and
  `ExternalPolicyGradient` are not merged; the numerics gate stays the recorded-rollout objective's (online GRPO
  re-scores its own samples). The policy-gradient table's rows read `loss.*`; the identity pins hold.
- **Behaviour change, online GRPO only.** It gains the dual clip, the advantage floor and the step guards, each off by
  default (`PgLoss::ONLINE` keeps every default it had: clip-higher 0.28, token-sum, mean-centred, the KL logged), and
  reports `clipped_dual`, `steps_skipped_guard` and the metrics `grpo_clip_frac_dual`, `grpo_steps_skipped_guard`
  (resume cursors written before read them as zero). Its loss settings are now held to the recorded-rollout objective's
  ranges (`clip_eps` in (0, 1), `clip_eps_high` in [`clip_eps`, 10)); `BehaviourRefusal::InvalidNumeric` carries the
  reason. `rl::continual::StudyConfig` gains `grpo_loss`.
- **Breaking for SDK callers.** `PolicyGradientConfig`'s loss fields moved under `loss` (`PolicyGradientConfig { loss:
  PgLoss { kl_beta: 0.0, ..PgLoss::RECORDED }, ..Default::default() }`), and `GrpoConfig::{clip_eps, kl_beta}` and
  `GrpoOptions::{advantage, loss_agg, clip_eps_high, log_kl}` under `GrpoConfig::loss`. No wire, CLI or identity change.
- **What proved it.** Before and after: `rl` `conformance`, `grpo_fixes`, `it` (with `policy_gradient_stability`),
  the GRPO, DAPO, Dr. GRPO and RLOO gradchecks and `grpo_objective`; the EC-1 pins (identity, settings, served params).
  New spec tests in `grpo_fixes`: a token dual-clipped in online GRPO gives no gradient (7 signed tokens, 3 clipped
  high, the 4 punished ones dual-clipped, gradient L1 exactly 0, against 0 dual-clipped and a non-zero gradient
  without it), and a step over a KL or gradient-norm bound is skipped, logged and counted, bounds above it pass, and
  `MAX_CONSECUTIVE_GUARD_SKIPS` in a row abort. Both fail against the pre-EC-2 band and guard (checked by reverting
  the two lines).

## EC-3 One max-batch policy for every served language model (M, open)

- **Problem.** An unset `--max-batch` is chosen three ways:
  - Qwen3 (`crates/catalog/src/resident_llm.rs:782`, `planned_max_batch`) ranks from the cards' decode operating
    point (`serve_defaults::max_batch_ranking`, up to `MAX_OPERATING_BATCH`) through `given_or_chosen_ranked`;
  - the Qwen3.5 dense and MoE residents rank the static `MAX_BATCH_DEFAULTS` (64 down to 1) through
    `given_or_chosen` (`resident_qwen35.rs:694`, `resident_qwen35moe.rs:423`);
  - LFM, Huginn and Ouro use a fixed constant each (`DEFAULT_BATCH` 2, 4 and 8).
  The batching, prefix cache and admission are generic; the choice of the batch they run at is not.
- **Target.** One `serve_defaults::plan_max_batch(cards, weight_dtype, fits)`: the operating-point ranking for every
  resident, each resident supplying only `fits(batch)` from its own memory estimate. `MAX_BATCH_DEFAULTS`, the
  per-resident constants and one of the two `given_or_chosen*` helpers are deleted. Knowledge 539's open item (the
  27B's activation headroom estimate against its measured peak) decides how wide the hybrid models go; it is a fix to
  their `fits`, not to the policy.
- **Safety net.** Per resident, on the mock card table: the chosen batch is the widest of the ranking its estimate
  places, and an explicit `--max-batch` is kept; `check-serve-conformance` and `check-placement-accounting` pass; the
  apiserve edge test (`crates/apiserve/tests/it/edge.rs`) unchanged.

## EC-4 The environment reads that are left (M, incremental, open)

- **Problem.** `scripts/gates/env-reads.allow` lists 725 reads, 151 outside tests, benches and examples. By class:
  1. device selection: `BRAIN_DEVICE` and `BRAIN_BACKEND` in `gpu-core/src/devices.rs` and `catalog/src/resident_llm.rs`;
  2. test hooks compiled into production code: `capability` `BRAIN_TEST_HOST_*` (5), `cli/src/main.rs`
     `BRAIN_TEST_CONFIG_*` (4);
  3. behaviour toggles: `backend-wgpu` `BRAIN_GPU_*` and `BRAIN_WGPU_*` (7), `flux2` `BRAIN_FLUX2_I8_KEEP_F32` and
     `BRAIN_FLUX2_DUMP_LATENT`, `model::kv_tier::KvTier::from_env`, `QWEN_TOKENIZER` (catalog twice, SDK once);
  4. developer binaries reading weight paths (`npu` export tools, `flux2` bins, `qwen35` `import_profile`);
  5. platform conventions (`HOME`, `XDG_*`, `HF_TOKEN`, `HF_ENDPOINT`, `DBUS_SESSION_BUS_ADDRESS`, `NO_COLOR`): kept.
- **Target.** Classes 1 to 4 become explicit parameters: a `Device` argument (tests take theirs from
  `brain_testutil`), injected test seams under `cfg(test)`, typed options on the owning config (a toggle nobody needs
  is deleted), argv on the binaries. The allowlist shrinks with each class; class 5 stays listed with its reason.
- **Safety net.** The env-read gate itself; per class, the tests that set the variable today rewritten to pass the
  parameter, run on CPU and CUDA.
- **Progress (2026-11-28, class 1, the timeline models).**
  - Done: `horizon` builds every model on an explicit `horizon::placement::Placement`. It is a parameter of
    `Horizon::new`/`load`, `Saved::load`, `Ensemble::load`, `Loaded::load` and `TrainSpec::placement`, with the CPU
    as the default.
  - The SDK's `TimelineModel::load`/`TimelineEnsemble::load` load on the CPU, and `load_on` takes a placement.
  - A capability action, a served model and the generic trainer build on `Placement::Process`, the set the
    process's own `--device`/`--backend` published. A process nothing configured is refused by name; it never
    falls back to the environment.
  - The GPU parity test asks for a GPU by name and runs under `--ignored`.
- **Step 1 done (2026-11-28).**
  - `gpu_core::ambient_compute_set` with nothing published is now the CPU. It probes no card and reads no
    variable. `resolve_ambient_compute_set` and the `BRAIN_BACKEND` text parser are deleted, and so is the CLI's
    warning about the variables.
  - `scripts/gates/check-device-env-unread.sh` (renamed from `check-device-env-single-source.sh`) keeps both
    variables unread under `crates/*/src`. `gpu-core`'s `ambient_device` test runs a child with them set to GPU
    values and asserts the CPU.
  - On the horizon test path, the one caller that still probed the cards was `Horizon::build` →
    `StreamedStore::attach` → `memplan::Capacity::measured` → `devices::selected_device` (the device registry).
    It now measures the handle's own memory (`Capacity::measured_for`, `Gpu::card`).
  - With `CUDA_VISIBLE_DEVICES` empty and neither variable set, the whole `brain-horizon` suite passes. A one-shot
    instrumented run showed no call to the registry, `Inventory::probe`, `capacity::probe_gpus` or
    `visible_gpu_count`.
  - Consequence: until step 2, the GPU legs that steered a test binary through these variables fail loudly and do
    not run. This covers `parity-gate.sh` (Vulkan and CUDA), `perf-floor-gate.sh` (Vulkan and CUDA),
    `scripts/test/cuda-matrix.py` and `scripts/test/ripple-gpu.sh` off the CPU.
- **Work list (steps 2 to 5), in order:**
  2. Give the test binaries a device seam. A `brain_testutil::device()` helper set by the harness (argv after
     `--`, or a cfg feature per backend suite) needs a decision first. Then re-enable the legs listed above, the
     Makefile CPU and CUDA suites (about line 1156) and the remaining callers of `cuda-device.sh`.
  3. Rewrite the tests that read device variables themselves to take the seam and ask for a GPU by name under
     `--ignored`:
     - `BRAIN_DEV_GPU`: `wan` (`device_train`, `dev_grad`, `dev_lora_train`), `s3dit` (`dev_parity`,
       `shard_parity`, `device_train`, `dev_grad`, `model_trait`, `shard_2card`, `lora_train`), `flux2`
       (`int8_base_grads`, `variant_widths`);
     - `BRAIN_VAE_DEVICE`: `vae` (`flux2_parity`, `encode_parity`, `sdxl_decode_parity`, `decode_parity`);
     - `BRAIN_BACKEND`: `huginn` (`serve`, `decode`, `parity`), `ouro` (`forward`);
     - `BRAIN_DEVICE`: `chronos2` (`timing`), `backend-cuda` (`tensormap_template`), `wan` (`gguf_direct_real`).
     Until then, the ones that ask through `BRAIN_DEVICE`/`BRAIN_BACKEND` run on the CPU.
  4. Give `brain-gradcheck`'s horizon cases a placement from the seam (`Placement::Process`, now the CPU).
  5. Check the other model crates for the same probe that `Capacity::measured` was. `qwen35`, `qwen3` and
     `ricursive` finetune call it on the selected card.

## EC-5 The training-records contract, published by version (S, open)

- **Problem.** `crates/data/tests/fixtures/training-records/v2/` is the contract a producer of training data copies
  and checks its records against. A copy is identical today, and nothing lets a producer check that it still is.
- **Target.** The fixture's tree digest published with the contract version (in `docs/reference/training-capabilities.md`
  and returned by the `check_dataset` action), so a producer's gate can compare its copy without reading brain's tree.
- **Safety net.** A test that recomputes the digest from the directory and compares it with the published one.

## EC-6 Loose integration-test binaries (S, open)

- **Problem.** Each `tests/*.rs` is its own test binary and link. `crates/rl/tests` has 23 beside `tests/it`, the SDK 18,
  `qwen35` 11.
- **Target.** Fold the ones that need no device of their own into `tests/it`. `gpu-core` (28) keeps separate binaries:
  the testing rule allows one device per test binary.
- **Safety net.** The same test names pass before and after (`cargo test -- --list` compared).

## Not now

- **The `Objective` trait and the objective family:** sound; their configuration was duplicated (EC-1, EC-2, done).
- **`ChatScanner` (`crates/data/src/qwen_chat.rs`, 2892 lines):** one scanner already serves every chat format and
  `toolcall::parse_tool_call` delegates to it. Splitting the file by responsibility (render, scan, split) is worth
  doing when a non-Qwen chat format arrives and gives the seam a second implementation, not before.
- **`QwenServeConfig` in `catalog`:** move it with the resident when `cli-to-served.md` moves the per-model surfaces.
