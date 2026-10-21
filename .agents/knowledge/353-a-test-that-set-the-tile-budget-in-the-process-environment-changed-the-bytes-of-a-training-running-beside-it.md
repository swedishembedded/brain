<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 353. A test that set the tile budget in the process environment changed the bytes of a training running beside it

Question: why did `caps::tests::an_unseeded_lora_train_echoes_the_seed_that_reproduces_it` fail once in a loaded run and
never again in dozens of reruns (knowledge 300 left it open)?

Cause: `BRAIN_TILE_BUDGET_WORDS` decides how the LM head is tiled (`block::tile_budget_words_for`), and tiling reorders its
sums. Two qwen3 lib tests (`model::a_built_model_keeps_its_head_tiling_when_the_tile_budget_changes`,
`serve::head_matmul_tiled_matches_untiled_within_tolerance`) wrote the variable with `set_var`; the cargo test runner
executes the whole lib in one process, so any training that built its model inside those windows tiled differently from the
same training outside them. The seed test trains twice and compares adapter bytes, so it failed exactly when one of its two
runs fell inside the window. Both tests took `brain_testutil::env_lock`, which only serialises tests that also take it.

Evidence (n = 1 in 4 full loaded lib runs, but deterministic when the window overlaps): a stress test training unseeded then
seeded 60 times passed 60/60 alone on wgpu, 60/60 on CUDA and 3 x 60 under 40 busy loops; inside a full lib run under load
it failed at iteration 1 with 332 of 8664 adapter bytes different (seed with the high bit set, so not seed-dependent).
After the two tests moved to the thread-scoped `model::block::with_tile_budget_words` the same full-lib run under load
passed.

Fix: tests scope the budget to their thread; `caps::tests::no_lib_test_publishes_the_tile_budget_through_the_process_environment`
keeps the lib sources free of the write. Rule: a value another thread's numerics can read never goes through the process
environment, even under `env_lock`.
