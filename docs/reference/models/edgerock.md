<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# EdgeRock

EdgeRock (`EdgeRockForCausalLM`) is the [Qwen3.5](qwen35.md) hybrid decoder
(Gated DeltaNet layers plus gated grouped-query attention) with two changes on
every full-attention layer: attention is restricted to a causal sliding window,
and an anchor memory keeps a fixed number of the best-scoring earlier tokens per
key/value group and reads them back, so the cost of a long context is bounded.

brain recognizes the architecture and runs it on the Qwen3.5 decoder
(`brain-qwen35`): a forward pass over a whole sequence. Generation and serving
refuse an EdgeRock configuration with an explicit error for now.
