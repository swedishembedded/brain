<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 468. A bf16 checkpoint was widened to fp32 by default, so every decode step read twice the weight bytes

**Lesson:** A default tier must start from what the checkpoint stores, not from a fixed precision. The Qwen3
dense default was "fp32 below 6B parameters or where fp32 fits, int8 otherwise", so the bf16 Hugging Face
releases were widened to fp32 on load: twice the bytes in memory and twice the bytes every decode step reads,
for values that were bf16 to begin with. And the perf target decided the tier by a rule of its own, so the
number it printed was not what `brain serve` ran.

## What happened

`brain perf run longctx --target qwen:<Qwen3-4B> --context 32768` measured 10.6 ms per one-stream decode step,
the rate of the 27B FP8 on the same card. The 4B is stored bf16 and was held fp32. Three callers each chose
the tier: the serving resident (`weights_int8_for`), the SDK pipeline (two `Precision` defaults) and the
`qwen:` perf target (fp32 unless a `:i8w` suffix). None of them read the checkpoint's encoding.

## Why

The rule was written around one question, "does fp32 fit", that only an fp32 checkpoint asks. The tier a
checkpoint is stored in was available in its tensor headers the whole time (`model::ops::stored_tier`) and
nothing on the default path consulted it.

## What to do

- Decide the default from the stored encoding: `qwen3::stored_weight_tier` (headers only) feeds
  `qwen3::footprint::default_weight_tier`, which keeps a bf16/f16/fp32 tier and falls to int8 only for a 6B+
  checkpoint whose stored-tier weights plus 8 GiB do not fit the card. `--weight-format` always wins.
- One answer per question: `catalog::resident_llm::qwen3_weight_policy` is what serving, the SDK and the
  `qwen:`/`qwen-synth:` perf targets build at, and the perf targets take the global `--weight-format` with
  serve's spellings instead of a suffix.
- Measure the accuracy of a default change, not only its speed, and on more than one small sample: on 16
  prompts of one text the bf16 tensor-core tier read 3.6x the KL of the old default, on 96 prompts of the
  same text 1.17x. The first sample held one outlier prompt.

## Evidence

Qwen3-4B, GH200 (gpu3), int8 KV, `brain perf run longctx --target qwen:<4B> --context 32768 --ladder 1,16,31
--steps 4 --seed 1`, one run each: fp32 default 10.60 ms per step at batch 1, 485 tok/s at batch 16, 30
sequences admitted at 32K; bf16 default 7.92 ms, 527 tok/s, 35 sequences (batch 31: 496 tok/s).

`qwen_bench serve-accuracy --weights <4B> --t 256 --gen 24`, next-token KL from exact fp32 (repeat KL 0):
book text, 96 prompts: old default (fp32 weights, TF32 GEMMs) 2.68e-4 mean / 3.83e-3 max, top-1 95/96; new
default (bf16 weights on bf16 tensor cores) 3.12e-4 / 4.66e-3, 94/96; int8 1.28e-3 / 1.05e-2, 93/96. Knowledge
notes, 96 prompts: 4.85e-4 / 5.82e-3, 96/96 against 5.67e-4 / 5.03e-3, 96/96; int8 1.92e-3 / 1.45e-2. The
largest log-probability difference of the bf16 tier on the second text was 2.23 against 0.60 (one token). The
same order as on the 8B (#326: 1.9e-4 against 1.6e-4). Full table: `.agents/roadmap/qwen3.md`.
