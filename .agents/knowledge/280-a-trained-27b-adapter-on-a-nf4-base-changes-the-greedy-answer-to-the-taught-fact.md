<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 280. A trained 27B adapter on the NF4 base changes the greedy answer to the taught fact

Q6 content gate, measured 2026-10-16 on the GH200 (CUDA backend, release binary, brain at f9f909f7
plus the lane's commits). Real Qwen3.8-27B (FP8 directory streamed onto the device as NF4), no fold.

**Training** (`brain qwen35 finetune CORPUS --base Qwen3.8-27B-FP8 --mode lora --base-dtype nf4 --batch 1
--seq-len 128 --lora-rank 16 --lr 2e-4 --warmup 4 --steps 40 --eval-interval 0`, drawn seed
6132777293037022924): the corpus is 798 tokens, four invented facts (an island capital, a beetle's shell
colour, a planet's moon count, a bridge's material), each written as `Question: ...\nAnswer: ...` and as a
statement, six times over. Loss 1.7469 -> 0.0746 in 586.8 s (load 152 s, then 12 s per step after the first
two). Peak device memory 23,975 MiB, host RSS high-water 6,992 MiB. The adapter file is 445 MiB (fp32, all
64 layers).

A first attempt with the default warmup (100 steps) never left the bottom of the schedule in 40 steps
(lr 1.6e-5 at step 8, loss still 1.5-1.7) and was stopped; the 40-step budget needs `--warmup 4`.

**Sampling** (`brain qwen35 sample-adapter --base-dtype nf4 --adapter FILE --max-new 8 --compare`, greedy,
repeated full forwards at length 29, 2.3 s per token; load 38.5 s; peak device 23,363 MiB, host RSS 7,465
MiB). The adapter is overlaid on the quantised base; "base" is the same process with every `lora_b` zeroed.
Verbatim output, the first four prompts are the trained questions, the next four are held-out
re-wordings, the last is a control:

```
[adapter] Question: What is the capital of Velmora?
Answer: Quenthara.

Quenthara
[base] Question: What is the capital of Velmora?
Answer:

<think>
The user is asking about
[adapter] Question: What colour is the shell of the Drellik beetle?
Answer: violet.

The shell of the D
[base] Question: What colour is the shell of the Drellik beetle?
Answer:

The Drellik beetle (*D
[adapter] Question: How many moons does the planet Ostrava-9 have?
Answer: seven.

The planet Ostrava
[base] Question: How many moons does the planet Ostrava-9 have?
Answer:

Ostrava-9 is not
[adapter] Question: What is the Marnikov Bridge made of?
Answer: red basalt.

The Marn
[base] Question: What is the Marnikov Bridge made of?
Answer:

The Marnikov Bridge, located
[adapter] Question: Which city is the seat of government of the island nation of Velmora?
Answer: Quenthara.

Quenthara
[base] Question: Which city is the seat of government of the island nation of Velmora?
Answer:

Based on the name provided, it
[adapter] Question: What is the shell of a Drellik beetle coloured?
Answer: violet.

The shell of the D
[base] Question: What is the shell of a Drellik beetle coloured?
Answer:

The Drellik beetle (also
[adapter] Question: The planet Ostrava-9 is orbited by how many moons?
Answer: seven.

The planet Ostrava
[base] Question: The planet Ostrava-9 is orbited by how many moons?
Answer:

Ostrava-9 is a
[adapter] Question: Which material was used to build the Marnikov Bridge?
Answer: red basalt.

The Marn
[base] Question: Which material was used to build the Marnikov Bridge?
Answer:

The Marnikov Bridge, located
[adapter] Question: What is the capital of France?
Answer: Quenthara.

Quenthara
[base] Question: What is the capital of France?
Answer:

Paris.<|endoftext|>
```

**Result: partial.** All eight fact questions (four trained, four re-worded) get the taught answer with the
adapter and none gets it from the base (the base declines or restates the question; the zero-delta base
answers the control correctly). The Q6 criteria not met: (1) nothing was served: the adapter was sampled
in-process by `sample-adapter` through repeated full forwards, never loaded into a serving engine; (2) the
control question was overwritten (the adapter answers `Quenthara` for the capital of France), so retention
of unrelated knowledge is not shown and is in fact contradicted; (3) the base's greedy answers were
compared only by eye, no acceptance check that the base lacks the fact was recorded.

**Retention not shown, and a real defect of the adapter:** the control question about France is answered
`Quenthara` with the adapter on. Forty steps on four repeated facts overfit the answer slot, so this
establishes that a trained qwen35 adapter changes behaviour on the real checkpoint, not that it
generalises or leaves other knowledge intact. The re-wordings share the entities of the taught facts. A
corpus with rehearsal rows (the sequential-SFT-with-rehearsal rule) is the fix to try before any claim of
retention.

**Mechanism.** `AdapterSampler` builds the same streamed training-mode model `finetune_streamed` does
(adapter factors beside the NF4 base) and reads the logits at the last real position of a fixed-length
forward; tokens after it are padding that a causal model never reads. A tiny-config CPU test checks it
equals the independent KV-cache decode of the adapter folded into the base, that switching the adapter off
equals the base's own decode, and that the trained fixture adapter makes the two differ.
