<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 70. `convert_hf_to_gguf.py` PRE-APPLIES math, and a name-and-shape-correct import can still be silently wrong

The first real two-GPU generation from `unsloth/Qwen3.8-27B-GGUF` produced
fluent-looking garbage: for `"The capital city of France is"` it emitted
`" France is France is France is France is"`, and for a chat prompt it echoed
four words of the prompt and stopped. Every structural check was green - 866
source tensors classified, two-way coverage clean, every numel matching, every
dequantized value finite and plausibly scaled, the norm-fold question already
settled by measurement (lesson from M20).

The cause: llama.cpp's converter stores `ssm_a = -exp(A_log)` so that
`ggml_ssm_scan` can use it directly, while brain's `gdn_decay_gate.wgsl`
implements the REFERENCE formula `g = -exp(A_log) * softplus(a + dt_bias)`
and therefore wants the original `A_log`. Importing verbatim computes
`-exp(-exp(A_log))`, which on the real checkpoint turns a per-head decay
coefficient of 0.0038..0.34 into 0.71..1.00 - up to 260x too strong. The Gated
DeltaNet recurrent state is annihilated within a step or two, so the model
keeps only the last token or so of context and degenerates into copying. Note
what it is NOT: not a crash, not a NaN, not an obviously broken number. All 48
values were negative, small and smooth - exactly what a plausible `A_log`
looks like.

**Rules that follow.**

1. **A GGUF import is not a renaming problem.** Read the converter, not just
   `tensor_mapping.py`: `convert_hf_to_gguf.py`'s `modify_tensors` hooks are
   where a value transform hides. `gguf::import::Mapped` now has a
   `Transformed { into, op }` variant and `ElemOp` is where each such
   convention is written down once, with its reason. The same fix had to be
   applied to `qwen35moe`'s importer, which had copied the same wrong
   assumption.
2. **A transform must be applied on EVERY read path, including the zero-copy
   one.** A streaming resident that lends `raw_words` straight from the
   mapping bypasses any transform its `with_tensor` applies -
   `qwen35::int8_gguf_resident::SsmALogFix` therefore returns `None` from
   `raw_words` for the transformed leaf, and its test drives a source that
   really does lend words so the bypass is a reachable state.
3. **Synthetic import fixtures must be realistic about SIGN and RANGE, not
   just shape.** Both crates' fixtures wrote `ssm_a` as `0.1, 0.2, ...` -
   positive, i.e. a value llama.cpp cannot produce. A fixture that cannot
   occur in the wild cannot exercise the transform, and (worse) made the
   wrong behaviour look tested. `ElemOp::LnNeg` now refuses a non-negative
   input outright, so that fixture would fail loudly today.
4. **Only a real end-to-end generation finds this class.** The gate that
   caught it is four lines: greedy-continue `"The capital city of France is"`
   with the chat template OFF and assert the output contains `"Paris"`. A
   chat-shaped request could not have caught it as cleanly - its short answer
   was explicable as a template quirk. Assert on a fact the model cannot get
   wrong, through the plainest possible request; "finite and non-degenerate"
   is not a correctness check.
