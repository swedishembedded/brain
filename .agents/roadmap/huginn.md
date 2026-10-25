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
  10 tensors per block (four RMSNorms, fused QKV, output projection, the two
  Q/K bias halves, fused gate/up FC, projection), the injection adapter, the
  final norm and the tied embedding table; `lm_head.weight` is dropped as a
  tied copy and `freqs_cis` as a rebuilt buffer. The checkpoint stores the
  Q/K bias as one `[2, 1, heads, dim]` tensor; the import splits it, because a
  device binding needs wider alignment than one head-width offset.
- Hugging Face import with two-way coverage and sharded safetensors support.
- Deterministic initialization for tiny configs and the gradient check.
- Execution graph: embeddings scaled by `sqrt(width)`, prelude blocks, the
  recurrent core re-entered with `adapter(concat(state, prelude_out))`, the
  final norm, the coda, the norm again and the tied head. Sandwich blocks
  (`x = norm_2(attn(norm_1(x)) + x); x = norm_4(mlp(norm_3(x)) + x)`), fused
  QKV with a per-head Q/K bias, fused gate/up FC and interleaved-pair RoPE
  (the `rope_train` kernel, not the half-split `rope_base`).
- Parity against the tiny golden in `testdata/golden/huginn/tiny`, which is
  driven by a fixed starting state and records the raw state, the normed
  latent and the logits of each of three steps. Green on cpu, wgpu and cuda.
- Incremental decode with one KV history per stage (prelude block, every
  recurrent step's blocks, coda blocks) - the core's repeated use means the
  keys of step 3 cannot share a history with step 2. A cached step equals the
  full forward's row at the same position because every position's recurrence
  starts from the model's initial state; green on cpu, wgpu and cuda.

- Paged serving engine (`crates/huginn/src/serve.rs`) over the same stage
  list: one allocator and one paged K/V plane per stage, one recorded token
  graph, batched prefill with a batched paged scatter. Depth is a serving
  setting (`open_with_steps`): the released 32 iterations are 132 stages,
  which cannot share a card with fp32 weights at a serving context.
- The catalog resident (checkpoint Jinja template, tool schemas described in
  a system turn), the provider/manifest entry, the model-directory scan's
  simple and compound arms, `brain huginn import|generate`, and the docs
  page.
- Training: forward and analytic backward through the sampled recurrence,
  every boundary residual kept per application and shared gradients
  accumulating over all of them; a finite-difference gate over every
  parameter on cpu, wgpu and cuda, plus an AdamW-step test that the loss
  falls and the shared core moves.

## Left to do

- Truncated backprop as the reference's sampled-depth mode runs it (run the
  leading steps under no-grad when `mean_backprop_depth` is smaller than the
  depth), and LoRA.
- A generated golden of the released checkpoint's continuation, so the
  served engine is gated beyond the tiny fixture.
