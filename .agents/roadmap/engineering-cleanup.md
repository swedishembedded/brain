<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# engineering-cleanup

Scope: parameters declared once across the RL objectives, the training-job API, the SDK and the CLI; one max-batch policy for every served language model; the environment reads still left; the published training-records contract; and the test-binary layout.

Audit of 2026-11-21 against brain 6d9a31d14 (read-only; every file and line below was read). Execution waits for the
safety nets each item names. Prefer deleting a declaration over adding a layer that keeps two in step.

## Order

EC-1 first (it also closes a resume-identity hazard; done). EC-2 after EC-1 (it changes the type EC-1 tabulates). EC-3
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
