<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 459. A refusal by name moved behind the base lookup when the architecture became optional

**Lesson:** When a required input becomes optional (inferred later from something that has to be read), every check
that used it before anything was read must still run first whenever the input is given, and run again on the inferred
value. Moving the checks to after the inference turns a typed refusal into whatever the read fails with.

## What happened

`ChatFullFineTune` took its architecture as a required name and refused a stage count or native sparse attention the
architecture's trainer does not take before reading the base. f2d7d98f5 made the architecture optional (read from the
checkpoint when not named) and moved those two refusals below `resolve_base`. A run naming `qwen3` with `.stages(2)`
or `qwen35` with native sparse attention then failed with `model not found` for a base that does not exist, and in
general only after the store had been consulted. 6082276b7 restored the unknown-architecture refusal for a named
architecture but not the two option refusals. `chat_finetune_ricursive::a_stage_count_is_refused_by_an_architecture_that_lays_itself_out`
and `native_sparse_attention_is_refused_where_it_cannot_run` caught it; they pass at the build before f2d7d98f5 and
were red on trunk in the CPU suite (2026-11-15).

## Fix and evidence

`ChatFullFineTune::refuse_options` holds both refusals; `run_with` applies it to a named architecture before the base
is resolved and to the architecture read from the checkpoint after. The two tests above are the regression gates.
