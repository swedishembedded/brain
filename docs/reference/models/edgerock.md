<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# EdgeRock

EdgeRock (`EdgeRockForCausalLM`) is the [Qwen3.5](qwen35.md) hybrid decoder
(Gated DeltaNet layers plus gated grouped-query attention) with two changes on
every full-attention layer: attention is restricted to a causal sliding window,
and an anchor memory keeps a fixed number of the best-scoring earlier tokens per
key/value group and reads them back, so the cost of a long context is bounded.

brain recognizes the architecture and runs it on the Qwen3.5 decoder
(`brain-qwen35`): a forward pass over a whole sequence, incremental inference
(`step`, `prefill_chunked`, and slot pools for many sequences) on state whose
size does not depend on how long the sequence is (the last `window` positions'
keys and values and the memory's bank), and training: a full
fine-tune in fp32 and LoRA, batched, with adapters on the attention and MLP
linears and on the memory's seven read-side projections (`memory.q_proj`,
`memory.k_proj`, `memory.v_proj`, `memory.o_proj`, `memory.write_gate_proj`,
`memory.write_up_proj`, `memory.write_down_proj`). The memory's admission
scorer, which picks the records a bank keeps, is never trained: the choice is a
sort, and no loss gradient reaches it. The training capabilities are listed in
the [capability matrix](../training-capabilities.md). The serving engine refuses an
EdgeRock configuration with an explicit error for now.
