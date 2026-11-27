<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Branch integration of 2026-11-27

Scope: Every brain branch that held a patch not on origin/main on 2026-11-27, what was landed and why the rest was not.

The rule was "all code on trunk, but only if it is sound and good": a change lands only if it is correct, passes the
gates of the crates it touches, fits main's current architecture and adds something main does not already have.
Inventory: `git cherry origin/main <branch>` over every local and `origin/*` branch, against origin/main 5e59cea01.
Each patch was compared with main by content, not only by patch id: a squash, a rebase or a later rewrite lands the
same change under another id.

## Decisions

| branch (tip) | unlanded patches | decision | evidence |
|---|---|---|---|
| `agrpo-review` (6e98e9d64), `agrpo-support-brain` (4d1271b47) | 1 / 2 | **LANDED** as f5d3e8995 | The trainer half of support replay (`pg_batch::SamplerSupport`, `head_ce::renormalise_over_support`), left "in review" on 2026-11-25 while its serving half (43219c67c) and records half (17076edb8 = f00e2186a) landed. 4d1271b47 is the same change before a three-line `dataset_report` fix; 6e98e9d64 is the one landed. Knowledge 591 renumbered to 618 (591 was taken). Gates: fmt, clippy -D warnings on brain-rl, brain-model, brain, brain-trainjob; CPU tests of those crates, brain-rl with `qwen3`; on the rebase a new `it::session_samples` test was given the new `support` field. |
| `minimax-h3` (47319bfd5) | 54 | SUPERSEDED | Landed as the squashed port 6626e37fc (2026-10-12) and developed on main since (64 commits on `crates/minimaxh3`). 10 of the 54 match main by subject; the rest are in the squash: the branch tip's `crates/minimaxh3` differs from 6626e37fc only by the squash's refinements (spec.rs, caps), and its shared-crate pieces (`checkpoint::has_model_weights`, `H3Recipe`, video-VAE tiling, vocoder reclaim) are on main. |
| `koala` (5e9ffdda5) | 3 | SUPERSEDED | Landed under rewritten ids: dd27f0e6f (split-key int8 decode), accdfd234 (int8 prefill GEMM 128x64 tile), c61bc2535 (KV swap staging budgeted with the pool). |
| `frozen/brain-87a32d2fe` | 1 | SUPERSEDED | Chat SFT observations in place and per-message `loss_weight`: on main as bd6f7c6e2 (`ChatSample::observations_in_place`, `ChatMessage::loss_weight`). Kept as a pin: an orchestrator's learning ledger and two of its findings name brain-87a32d2fe as the build of recorded runs, and only this ref keeps that commit reachable. |
| `paper-pins/flywheel-learner` (796871647) | 1 | SUPERSEDED | Its one patch is knowledge 436, on main as b9667ae78 (rebased, evidence ids updated by a88359c96). Kept as a pin: the flywheel paper's reproducibility section names 796871647 as the learner build. |
| `api3` (b6e06e82b) | 1 | SUPERSEDED | Knowledge 436, as above. |
| `api4` (13ff83a94) | 2 | SUPERSEDED | Knowledge 436, and a test-side workaround in `catalog` `it::decision_agreement` for the SDK moving the process backend; main fixed the cause in the SDK (`sdk::device::resolve` keeps the ambient compute set for a default device, `the_default_device_leaves_the_process_selection_alone`). |
| `wip/brain-api4-2026-11-27` (17d6ce7dc) | 3 | SUPERSEDED | Raw snapshot. Its device fix, mirage `caps` row and early arch refusal in `ChatFullFineTune` / `PreferenceFineTune` are on main in another form (`resolve_base` + `chat_for`). |
| `wip/brain-diag-2026-11-27` (b544bae2e) | 2 | SUPERSEDED | Raw snapshot: a separate exponent cap (5) on the k3 KL weight. Main fixed the cause instead: the KL term is ratio-weighted k3 (7a7869433, knowledge 537), whose weight on a token the policy has left is bounded without a second cap. |
| `wip/brain-k2-2026-11-27` (08d2a8413) | 2 | NOT LANDED, open | Raw, unreviewed snapshot: a `TokenObjectives::set_token_argmax` / `batch_token_argmax` readout so in-process held-out scoring reports token accuracy beside loss (`ScoredSet::base_accuracy`, `tuned_accuracy`). The gap is real on main: `sdk::finetune` builds the in-process `HeldOutScore` with `token_accuracy: None`. Open below. |
| `w4a16` (8f5ec7d86), `wip/w4a16-2026-11-27` (3c86f2d91) | 1 / 2 | NOT LANDED, open | `matmul_nvfp4_wgmma_bf16`, a W4A16 wgmma GEMM that `Ops::nvfp4_mma` would select for bf16 linear activations. A CUDA-only kernel on the serving path: it cannot be validated without a Hopper card (none may run tests today), its message records no win over the library product it would replace, and the snapshot is an unfinished rework (a small-M N64 build). Open below. |
| `edgerock-wip-split-read` (76c810382) | 1 | OBSOLETE | Self-described "unverified, mid-edit". The split anchor read it was editing landed as 1fec3b735 and was then changed to bf16 storage (2be5af300). |
| `backup/pre-forcepush-2026-11-20` (f1915226b) | 0 | BACKUP | All 53 patches are on main; a backup of rewritten history is never re-landed. |
| `e1` (3e2e51153) | 0 | LANDED earlier | All 45 patches are on main. |

## Open

- [ ] Held-out token accuracy for in-process scored fine-tunes (from `wip/brain-k2-2026-11-27`): a generic argmax readout
      on `TokenObjectives`, implemented by qwen35 (`enable_token_argmax` exists) and reported in `HeldOutScore`. Port
      it onto main's `model::score` with a test, rather than rebasing the snapshot.
- [ ] W4A16 NVFP4 wgmma GEMM (`w4a16`, `wip/w4a16-2026-11-27`): rebase, finish or drop the N64 build, and land only
      with its CUDA oracle tests run on a Hopper card and a device-event measurement against the product it replaces
      (`LinearActivations::Bf16` only; `auto` stays E4M3).

## Remote refs

Deleted from origin once their content was on main or recorded here as superseded or obsolete. Kept: `w4a16`,
`wip/w4a16-2026-11-27` and `wip/brain-k2-2026-11-27`, the sources of the two open items, and the two pins
`frozen/brain-87a32d2fe` and `paper-pins/flywheel-learner`, whose commits recorded results cite and which nothing else
keeps reachable.

## Found while gating (pre-existing on main, not caused by this integration)

- [ ] `rl` `continual_study::twelve_cycles_run_end_to_end_but_capability_does_not_accumulate_at_this_scale` fails its
      own tripwire on main 5e59cea01 (CPU, `--features qwen3`): ACC 0.280 against the last-task-only control 0.066 + 0.15.
      The test says the P19 verdict is stale and must be re-derived from a fresh run.
- [ ] `rl` `mixture_anchor_regression` draws its seeds and failed one draw in about twelve on CPU ("primary-only training
      raised the anchor's held-out loss": baseline 3.137, primary-only 2.857). Its precondition does not hold for
      every seed.
