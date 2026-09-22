<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 71. When the loader and the checkpoint agree, only a SECOND implementation can say which of them is wrong - and a checkpoint's own statistics can settle most layout questions before you write one

`crates/qwen35`'s two-GPU Q8_0 route produced fluent-but-wrong text and the
investigation had concluded "a difference between what this GGUF route loads
and what the safetensors route loads, in a tensor whose name and shape are
both right" - the `ssm_a = -exp(A_log)` class of defect (lesson 70), assumed
to have a second instance. Every gate that existed compared brain against
brain: int8 against fp32 from the same bytes, decode tape against prefill
tape, two shards against one. None of them can see a tensor that both sides
read the same wrong way.

A ~350-line pure-CPython reference (no torch, no numpy, no transformers, no
llama.cpp: `tools/goldens/qwen35_gguf_reference_forward.py` parses the GGUF
header, dequantizes Q8_0 and runs the decoder itself) settled it in an
afternoon: brain's fp32 GGUF stack is **bit-equal** to it - cosine 1.000000,
relative L2 0.0000 - at 1/4/8/12 real layers over 4 real decode positions on
the real 27B weights. The loader was never the defect. Four real layers cost
~30 s of CPU, or ~3 s with the row range split over a process pool, which is
cheaper than one wrong hypothesis.

**Rules that follow.**

1. **"No external oracle is available" is usually a statement about torch,
   not about arithmetic.** A decoder layer is a few mat-vecs, a norm, a
   softmax and a recurrence. If the reference is a Python file you can read,
   you can transcribe it; the expensive part (a 27B checkpoint) is already on
   disk. Do this BEFORE the next hypothesis, not after the fifth.
2. **A checkpoint answers most layout questions itself, without any
   forward pass.** Partial RoPE leaves a magnitude break at `rot_dim` in
   every roped projection's per-row profile, and NOT in the unroped ones -
   that alone identified which half of a per-head-doubled `q_proj` is the
   query, ruled out llama.cpp's LLaMA-style q/k permute (the profile is a
   function of `d mod rot/2`, not of `floor(d/2)`), and separated `attn_k`
   from `attn_v`. A depthwise conv's channel-block boundary appears at
   exactly one flat index under the correct `[channel][tap]` reading and
   nowhere under `[tap][channel]`. A `(1+w)`-reparameterized norm that a
   converter has already folded clusters on 1.0; one it has NOT folded
   clusters on 0.0. Per-block norm means that rise monotonically with depth
   prove the block order is not shuffled. Q8_0 block scales are enough for
   all of this - a few MB of reads, no GPU.
3. **A symmetry tells you which hypotheses a position-0 check can even
   test.** In a delta-rule recurrence the first token's output is
   `(q.k) * v * beta`, symmetric in `q` and `k`, so a q/k swap is INVISIBLE
   at position 0 - as is any RoPE error (angle 0), any KV-cache error, and
   the decay `g` (the state is still zero). Work out that set before reading
   a per-position debug dump, or "position 0 looks fine" will be read as
   evidence when it is none.
4. **Publish the oracle as a gate, not as a transcript.** The digests it
   produced are now `tests/gguf_reference_parity_real.rs`, so the next
   refactor of the GGUF plumbing cannot silently un-prove this.
