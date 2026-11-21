<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# engineering-cleanup

Scope: parameters declared once across the RL objectives, the training-job API, the SDK and the CLI; one max-batch policy for every served language model; the environment reads still left; the published training-records contract; and the test-binary layout.

Audit of 2026-11-21 against brain 6d9a31d14 (read-only; every file and line below was read). Execution waits for the
safety nets each item names. Prefer deleting a declaration over adding a layer that keeps two in step.

## Order

EC-1 first (it also closes a resume-identity hazard). EC-2 after EC-1 (it changes the type EC-1 tabulates). EC-3
independently. EC-4 is incremental, by class. EC-5 and EC-6 any time.

## EC-1 One table for the policy-gradient settings (M, open)

- **Problem.** `clip_eps_high` (and every sibling: `dual_clip`, `max_step_kl`, `max_step_grad_norm`, `importance`,
  `advantage_norm`, `loss_agg`, ...) is declared by hand in seven places:
  1. `rl::PolicyGradientConfig` (`crates/rl/src/objective/external.rs:289`);
  2. `trainjob` `PgParams` (`crates/trainjob/src/request.rs:356`), its parameter spec with docs and ranges (line 126),
     its wire list (line 55) and its "policy gradient only" guard list (line 404);
  3. `trainjob` `job.rs:271-276`, which lays each param over the config field by field;
  4. the SDK's run identity (`crates/sdk/src/finetune.rs:1303-1324`, every field as bits) and
  5. its `settings_json` (`finetune.rs:1640-1660`);
  6. the CLI's `Args`, its hand parser (`crates/cli/src/finetune_cli.rs:254-257`, where `clip_eps_high` is
     `--ratio-clip-high`) and the struct literal that builds the config (line 420).
- **Cost.** A field added to the config and missed in one list is silently not settable over the API, not shown in
  the settings, or, worst, not part of the identity a resume is keyed by, so a resumed run continues under different
  settings without noticing. The names already drift between the CLI and the API.
- **Target.** `PolicyGradientConfig` is the one declaration: serde, with a derived patch type (every field `Option`)
  that the API decodes and `apply`s. The parameter spec (doc, range), the guard, `settings_json` and the CLI flags are
  generated from one descriptor table beside the config, flag names equal to field names (the CLI's renames go in the
  major release `cli-to-served.md` plans, with no aliases). The run identity is the canonical JSON of the resolved
  config, not a hand list.
- **Safety net (first).** A test that walks every field of `PolicyGradientConfig` and requires it in the spec, the
  settings, the identity and the CLI, and that a request setting each field to a non-default value round-trips into the
  resolved config; the existing `trainjob/src/request.rs` and `crates/trainjob/tests/it/run.rs` tests unchanged.

## EC-2 Online GRPO and the recorded-rollout objective share one loss configuration (M, open; needs EC-1)

- **Problem.** The objective family is sound: one `Objective` trait (`crates/model/src/objective.rs:52`) implemented
  by `Grpo`, `ExternalPolicyGradient`, `Dpo`, `Feedback` (KTO), `DistillTopK` and `Mixture`, sharing
  `grpo_term`, `pg_batch` and `advantage`. But the two policy-gradient objectives configure the same loss with two
  types. `GrpoConfig` (`grpo.rs:185`, with `GrpoOptions`) has `clip_eps`, a plain `clip_eps_high`, `kl_beta`,
  `loss_agg`, `advantage`; `PolicyGradientConfig` adds `dual_clip`, `importance`, `max_step_kl`,
  `max_step_grad_norm` and the numerics gate. Online GRPO builds its band without the dual clip
  (`ClipBand::new(clip_eps, clip_eps_high)`, `grpo.rs:445`) and has no step guard, the guard that removed the damage
  of record-driven ratio spikes in the recorded-rollout path.
- **Target.** One `PgLoss { band (with dual clip), kl_beta, loss_agg, advantage_norm, advantage_floor, step guards,
  numerics }` embedded in both configs; the variation that is real (sampled online from an environment against read
  from records) stays in the two objectives. `Grpo` and `ExternalPolicyGradient` are not merged.
- **Safety net.** `crates/rl/src/conformance.rs` (token-term parity), `crates/rl/tests/grpo_fixes.rs`,
  `crates/rl/tests/it/policy_gradient_stability.rs` and gradcheck on both backends, run before and after; a new test
  that online GRPO honours the dual clip and a step guard.

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

- **The `Objective` trait and the objective family:** sound; only their configuration is duplicated (EC-1, EC-2).
- **`ChatScanner` (`crates/data/src/qwen_chat.rs`, 2892 lines):** one scanner already serves every chat format and
  `toolcall::parse_tool_call` delegates to it. Splitting the file by responsibility (render, scan, split) is worth
  doing when a non-Qwen chat format arrives and gives the seam a second implementation, not before.
- **`QwenServeConfig` in `catalog`:** move it with the resident when `cli-to-served.md` moves the per-model surfaces.
