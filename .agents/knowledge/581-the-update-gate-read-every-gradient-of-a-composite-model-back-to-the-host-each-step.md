<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 581. The update gate read every gradient of a composite model back to the host each step

**Lesson:** the shared fit loop's update gate (`model::stepgate::StepGate::admit`) asks
`Model::gradients_finite` before every update, and bounds that question by the trainable floats
`Model::optimizer_parts` reports (`MAX_SCANNED_GRAD_FLOATS`). A model with several parameter stores
(two decoders and an aux store) reports no single `optimizer_parts`, so the bound never applied and
the trait's default answered by reading every gradient to the host with `read_grad`: for the 1.7B
Qwen3-TTS fine-tune, about 2 billion floats, tens of large pageable copies, every step. No timer
showed it: it sat between the backward and the optimizer, outside every phase the objective timed,
and the device looked idle.

A composite model answers `gradients_finite` itself, on the device: every gradient is finite exactly
when their sum of squares is. The update needs that sum for its global norm anyway, so it is measured
once per set of gradients and carried (`qwen3tts::speech_sft::SpeechSftModel`). Any other model that
holds its parameters outside one `optimizer_parts` store pays the same scan until it does the same.

## Evidence (2026-11-25, one GH200, `speech_sft_bench`, 1.7B Base, batch 16)

`nsys` of the device-resident build (rows, routing and tables already on the card): 0.4 s of kernels
a step against a 2.3-2.8 s step; between the backward's end and the optimizer range, 1.9 s with 399
device-to-host copies (`cuda_gpu_trace`): every decoder weight gradient (50.3 MB each), the stacked
heads (125.8 MB) and the codec tables (276.8 MB, sixteen times, once for each table read out of
them). With the device answer: 520 -> 2390 frames/s by the wall clock.
