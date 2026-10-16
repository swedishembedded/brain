<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 305. The LoRA defaults already cover every linear, but the default rank 8 is far below the rank published for code

Question: which linear layers does the qwen3 LoRA adapt by default, and how does
that compare with the published code fine-tuning recipes? (Read-only research; no
default was changed.)

## What the defaults are today

- **Targets.** `crates/qwen3/src/lora.rs` carries `targets: vec![]` only in
  `fold_adapter_into`'s return value (the fold needs rank and alpha, not the site
  list); it is not the training default. The training default is
  `qwen3::finetune::default_lora_targets()` = `LORA_TARGETS` = all seven
  projections `wq wk wv wo gate up down` (`finetune.rs`), used by the CLI
  (`qwen_cli.rs`), the caps layer and the benches. `LoraCfg::attn` (four attention
  projections only) exists in `config.rs` but is not what a default run uses.
  `qwen35` (dense hybrid) adapts all twelve leaves (`config::lora_targets`: the
  five GDN projections, the four GQA projections and the three MLP ones). Only
  `qwen35moe`/`deepseek2` skip the expert MLPs.
- **Rank and alpha.** `crates/sdk/src/finetune.rs::lora_shape` and
  `qwen35_cli::DEFAULT_LORA_RANK` default to rank 8, alpha = 2 * rank. The qwen3
  CLI has no default (`--lora R` is required); its alpha default is also 2 * rank.
- **Schedule.** `LoraHyper::default`: weight decay 0.1, clip 1.0, warmup 5%, cosine
  to a tenth of the peak. The learning rate is the caller's.

## What the published recipes use

- A study of LoRA against full fine-tuning (2405.09673) targets "all transformer
  modules" (q, k, v, o, gate, up, down), ranks 16/64/256 at alpha = 2 * rank, and
  its stated recommendation is: target all modules with rank 256, because ranks
  16-64 "tend not to suffice for code tasks"; use alpha = 2 * rank (a fixed small
  alpha down-scales high ranks and was inferior); learning rate one order of
  magnitude above full fine-tuning (5e-5 to 5e-4), the highest stable one. MLP or
  all-module targets beat attention-only at every rank; the weight deltas of full
  fine-tuning have a rank 10-100x above typical LoRA ranks. Its advice also holds
  that LoRA suits instruction fine-tuning, not continued pre-training.
- The software-engineering agent trajectory papers in `coding-agents/` that use
  LoRA give little: the environment paper (2412.21139) trains a 32B model with
  LoRA rank 64, learning rate 5e-4, batch 8, 5 epochs and does not state the
  targets in the text; the task-synthesis paper (2504.21798) uses LoRA only for a
  small difficulty classifier. The others fine-tune fully.

## Comparison

Targets already match the recipe (all linears). Alpha = 2 * rank matches. The gap
is the **rank**: the default 8 is 8x below the 64 used by the agent recipe and 32x
below the 256 the study recommends for code. The learning rate (caller-chosen,
no default) is the other lever that interacts with it.

## Proposed change (not applied) and measurement plan

Propose: keep targets; raise the default fresh-adapter rank for code/agent
fine-tuning from 8 to 64 (the sdk, `qwen35_cli`, and a required-or-64 default in
`qwen_cli`), keep alpha = 2 * rank, and document 256 as the high-capacity option
bounded by memory (adapter and AdamW state grow linearly with rank).

Measure before changing, on one model and one code SFT set, same data, steps and
seed-reported runs, ranks 8/32/64/128/256 at alpha = 2 * rank and a learning rate
sweep {5e-5, 1e-4, 2e-4, 5e-4} per rank:

1. held-out code loss and pass rate on a held-out task set (the metric that
   decides);
2. a forgetting probe: loss on a general-text held-out set before and after;
3. cost: step time and peak device memory per rank (rank 256 on the 27B must fit
   the card with the NF4 tier);
4. the attention-only against all-linears control at rank 64, to confirm the
   existing target default rather than assume it.

Adopt rank 64 only if it beats rank 8 on (1) beyond seed noise without a worse
(2); take 256 only if it beats 64 by the same test within the memory of (3).
