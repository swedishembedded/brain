<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# huginn

Scope: Huginn (Raven), the prelude/recurrent-core/coda decoder (`crates/huginn`).

The recurrent contract differs from Ouro's whole-stack loop: the input is
embedded by a prelude, a shared core is applied a variable number of times
with the embedded input re-injected through a linear adapter, and a coda
un-embeds the final latent state into the tied head.

## Done

- Config and parameter layout pinned to the released `huginn-0125` manifest:
  9 tensors per block (four RMSNorms, fused QKV, output projection, Q/K bias,
  fused gate/up FC, projection), the injection adapter, the final norm and the
  tied embedding table; `lm_head.weight` is dropped as a tied copy and
  `freqs_cis` as a rebuilt buffer.
- Hugging Face import with two-way coverage and sharded safetensors support.
- Deterministic initialization for tiny configs and the gradient check.

## Left to do

- Execution graph: prelude, recurrent core with the `cat(x, embeds)` adapter,
  coda, tied head; `like-init` latent state (the reference draws it from
  `init_values["std"]`, so a parity run fixes the same draw).
- Reference golden (tiny and released) and per-step parity; incremental
  decode with one KV history per recurrent step; paged serving.
- Training: backward through the sampled recurrence (the reference backprops
  the last `mean_backprop_depth` steps and runs the earlier ones under
  `no_grad`), then LoRA.
