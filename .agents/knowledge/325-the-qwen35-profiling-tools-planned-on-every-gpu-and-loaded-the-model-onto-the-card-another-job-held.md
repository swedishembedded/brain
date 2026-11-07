<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 325. The qwen35 profiling tools planned on every GPU and loaded the model onto the card another job held

**Lesson:** A tool that plans placement from all GPUs is correct only on a box with one job. On a
shared node the failure looks like an out-of-memory on an empty card, so plan only over the cards
the tool was told to use.

## What happened

Symptom: `qwen35_gguf_prefill_profile ... --device gpu1` died twice with `cuMemAlloc failed: out of
memory` while loading the weights. Card 1 was shown empty by `nvidia-smi` (10 MiB), and a
reservation of the whole card (88 GiB) was in place. Card 0 held another job's 27B server (70 GiB).

## Why

The tool called `gpu_core::select_compute("gpu1")` and then built the device list it hands the
resident from `gpu_core::devices::gpus()`, which is every physical card, not the set `--device`
selected.

The resident's planner places a model on the fewest devices that fit, in index order, assuming every
listed card is empty: it chose card 0. The same pattern was in the decode profile, the batch and
speculation sweeps, the chunk-round profile, the adapter serve check and six real-checkpoint tests.

## What to do

`qwen35::int8_gguf_resident::budgeted_gpus(allowed, reserve)` is the one place that turns a card
list into `(device, usable bytes)`. It keeps only the cards in the process's compute set
(`--device`, or the ambient set for a tool with no flag). Every such site uses it, and a unit test
pins that a tool given card 1 never has card 0 in its budget.

`qwen35_train_probe` also takes `--device` and `--backend` as flags. Without them it fell back to
the box's default backend (Vulkan) and measured a different path than the one it was meant to.
