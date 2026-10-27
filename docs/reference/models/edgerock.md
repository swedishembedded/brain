<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# EdgeRock

EdgeRock (`EdgeRockForCausalLM`) is the [Qwen3.5](qwen35.md) hybrid decoder
(Gated DeltaNet layers plus gated grouped-query attention) with two changes on
every full-attention layer: attention is restricted to a causal sliding window,
and an anchor memory keeps a fixed number of the best-scoring earlier tokens per
key/value group and reads them back, so the cost of a long context is bounded.

brain runs it on the Qwen3.5 decoder (`brain-qwen35`): a forward pass over a whole
sequence, incremental inference (`step`, `prefill_chunked`, and slot pools for
many sequences) on state whose size does not depend on how long the sequence is
(the last `window` positions' keys and values and the memory's bank), serving,
and training: a full fine-tune in fp32 and LoRA, batched, with adapters on the
attention and MLP linears and on the memory's seven read-side projections
(`memory.q_proj`, `memory.k_proj`, `memory.v_proj`, `memory.o_proj`,
`memory.write_gate_proj`, `memory.write_up_proj`, `memory.write_down_proj`). The
memory's admission scorer, which picks the records a bank keeps, is never
trained: the choice is a sort, and no loss gradient reaches it. The training
capabilities are listed in the [capability matrix](../training-capabilities.md).

## Serving

```bash
brain pull rockmatrix-ai/EdgeRock
brain serve --dbus --openai 8788      # the model is served under its pulled id
brain do edgerock generate --prompt "Hello" --max_new 64   # in process, no server
```

`brain pull` leaves the Hugging Face directory in the store (the vision tower
under `vision/` is a separate component the text decoder does not use); it is
read in place, each tensor on demand, and the linears are held at bf16, the
precision they are released in, so no fp32 copy of the model exists on a card or
on the host. The tied embedding and the anchor memories' projections are held in
fp32 by the model, which the placement estimate charges. Requests go through the
same continuous-batching scheduler as every other model: all running sequences
decode in one batched step, prompts are read in under the scheduler's prefill
budget, and chat uses the Qwen3.5 template with the `<|im_end|>` stop.

A sequence costs the same fixed state at any length (a few hundred megabytes
at the published shape: the recurrent state, the ring of window keys and values
and the bank), so the context is the model's position limit (262144), not a
memory figure, and memory decides only how many sequences run at once. `brain
serve --max-batch N` is the number of live sequences; the same number of
recent prompts is kept for reuse, both capped to what the emptiest card has
free after the weights. A prompt that extends a recent one (an agent resending
its history) resumes from a saved copy of the whole state and reads in only the
new tokens; an exact repeat reads in none. `--ctx` only lowers the position
limit. Adapters are not served for this model yet.

The example `samples/python/llm/edgerock/edgerock.py` asks about a fact
buried in a long document over D-Bus, OpenAI-compatible and
Anthropic-compatible HTTP, and shows the second request resuming from the first.
