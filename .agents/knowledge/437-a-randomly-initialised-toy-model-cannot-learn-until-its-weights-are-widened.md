<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 437. A randomly initialised toy model cannot learn until its weights are widened

**Lesson:** A LoRA fine-tune of a tiny `qwen3` checkpoint at the standard 0.02 initialisation barely
moves: the tied output embedding is so small that no adapter can lift one logit clear of the rest in a
feasible number of steps. Widen the non-norm weights before using it as a test fixture.

## What happened

A 2-layer, 16-wide checkpoint with the real 151936-token vocabulary spent 184 s on 300 CPU steps and its
loss went 11.93 to 11.92 (the uniform loss is 11.93). Shrinking the vocabulary to 264 and widening the
weights sixfold (`brain::toy::WEIGHT_SCALE`) took the loss from 5.9 to 2.0 in 400 steps and a held-out
cipher from 0 to 0.99 in 500 steps (27 s). With the standard scale and the small vocabulary the loss went
5.58 to 4.96 in 1500 steps.

Two more traps on the way: the fixture needs `eos_token` in `tokenizer_config.json` or generation never
stops at the end-of-turn token; and a task that needs a copy circuit from the prompt to an unseen token
was memorised, not learnt (train 1.0, held-out 0.0), while a letter-wise cipher over seen letters in new
combinations generalised.
