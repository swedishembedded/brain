<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 325. The qwen35 profiling tools planned on every GPU and loaded the model onto the card another job held

Symptom: `qwen35_gguf_prefill_profile ... --device gpu1` died twice with `cuMemAlloc failed: out of
memory` while loading the weights, on a card 1 that `nvidia-smi` showed empty (10 MiB) and a reservation of
the whole card (88 GiB). Card 0 held another job's 27B server (70 GiB).

Cause: the tool called `gpu_core::select_compute("gpu1")` and then built the device list it hands the
resident from `gpu_core::devices::gpus()`, which is every physical card, not the set `--device` selected.
The resident's planner places a model on the fewest devices that fit, in index order, assuming every
listed card is empty: it chose card 0. The same pattern was in the decode profile, the batch and
speculation sweeps, the chunk-round profile, the adapter serve check and six real-checkpoint tests.

Fix: `qwen35::int8_gguf_resident::budgeted_gpus(allowed, reserve)` is the one place that turns a card
list into `(device, usable bytes)`, keeping only the cards in the process's compute set (`--device`, or
the ambient set for a tool with no flag); every such site uses it, and a unit test pins that a tool
given card 1 never has card 0 in its budget. `qwen35_train_probe` also takes `--device` and `--backend`
as flags: without them it fell back to the box's default backend (Vulkan) and measured a different
path than the one it was meant to.

Lesson: a tool that plans placement from "all GPUs" is correct only on a box with one job. On the shared
two-card node the failure looks like an out-of-memory on an empty card.
