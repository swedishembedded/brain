<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 315. A served adapter bank on the GGUF 27B changes the answer and costs about a quarter of batch-8 decode throughput

Question: serve one LoRA adapter, trained by `brain qwen35 finetune`, beside the base on the
released Q8_0 GGUF of Qwen3.8-27B (INT8, one GH200, CUDA), in the same batched decode step, and
see what the bank costs. Measured 2026-10-17, release build, `qwen35_adapter_serve_check`
(`crates/qwen35/src/bin/qwen35_adapter_serve_check.rs`) under the GPU lock, `other_contexts_start=0`.
All figures are **VERIFIED-BY-RUN** from one run; the throughput is a single measurement, not a
distribution.

**Adapter.** Trained on the NF4 FP8 base: 336 corpus tokens (four invented question/answer facts,
six times over), `--seq-len 64 --lora-rank 16 --lora-alpha 32 --lr 2e-4 --warmup 2 --steps 14
--seed 5`, all 12 targetable linears; loss 2.677 to 0.340 in 126.6 s (0.6 s per step after load).

**Bank.** `AdapterTargets::ALL` resolved against the model's sites (12 leaves), 2 slots, rank cap
16: 890 MiB of device memory (all 64 layers, one card). Registered as an ordinary
`VerifiedAdapter` read through the generic `parse_adapter` (which checks the card's targets
against the tensors).

**The served answer differs from the base.** Greedy, `Question: What is the capital of
Velmora?\nAnswer:`, 8 new tokens:

```
base:    "\n\nBased on the name provided, it"
adapter: " Quenthara.\n\nQuestion: How"
```

A base request after the adapter one returns the base's text again.

**Throughput, batch 8** (8 sequences, one short prompt, 48 new tokens each, greedy, prefill of
the 8 prompts included, after a warm-up that uploads the adapter):

| rows | generated tokens/s |
|---|---:|
| 8 base | 212.1 |
| 4 adapter + 4 base | 154.5 |
| 8 adapter | 151.1 |

The mixed batch costs the same as the all-adapter one: the bank adds two dispatches per targeted
linear (12 linears, 64 layers, so about 1,500 extra dispatches per step) whatever the number of
distinct adapters or adapter rows, and a batch with no adapter row records exactly the base's
dispatches. The overhead is therefore about 27 percent at this batch size, consistent with
knowledge 314 (the step is host-bound, so extra launches cost wall time directly). Narrowing the
bank to the nine mixer leaves (`--adapter-targets attention`, this model's default group) or
fusing the two dispatches of a site across the group would shrink it; neither was measured.

## Through `brain serve`

The same adapter, served by `brain serve --openai --models-dir DIR --qwen35-ctx 512 --qwen35-max-batch 8
--adapter-slots 2 --adapter-targets all`, with the GGUF and the adapter in the store
(`unsloth/Qwen3.8-27B-GGUF/adapters/lab/facts/v1/`): the model list shows the GGUF and
`unsloth/Qwen3.8-27B-GGUF-Q8_0:lab:facts:v1`, and a greedy completion of the same prompt gives
`"\n\nBased on the name provided, it"` on the base id and `" Quenthara.\n\nQuestion: How"` on the
adapter id, whose reply names the adapter's id and digest. `--adapter-slots` alone makes the bank the
operator's, so the default `attention` targets refuse this adapter (it trains the MLP too) with the
flag that widens them. The trainer wrote the card id `brain/qwen35-lora` and base `Qwen3.8-27B-FP8`;
the id had to be rewritten to the adapter reference grammar for serving.

## Caveats

* A short prompt (up to 8 rows) of an adapter sequence is prefilled on the chunk tape, not the
  exact-decode tape a base prompt of that length uses (the exact tape splits the round into groups
  of 8 and the bank is set for the whole round). Its arithmetic differs from the base's at the
  tolerance `tests/decode_step.rs` gates; the parity gate for adapter rows is the training
  forward (`crates/qwen35/tests/adapter_bank.rs`, CPU and CUDA), not the decode tape.
* One adapter, one run. Eviction under slot pressure on the real model was not exercised; the
  lifecycle is gated on the tiny config with a one-slot bank.
