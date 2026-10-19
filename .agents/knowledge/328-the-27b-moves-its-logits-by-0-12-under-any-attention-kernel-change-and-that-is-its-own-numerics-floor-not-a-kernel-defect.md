<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 328. The 27B moves its logits by 0.12 under any attention kernel change, and that is its own numerics floor, not a kernel defect

Symptom: replaying the same greedy requests through the pre-L7 serving binary and a new one gave 0 of 12 identical
replies, and `qwen35_gguf_logit_check` (one load of the Q8_0 27B, one greedy stream, teacher-forced logits under several
settings) reported mean logits rel-L2 0.11-0.17 between flash prefill version 1 and version 2, with argmax flips at
reference margins up to 1.28. It looked like version 2 was inaccurate.

What the attribution showed (runs of 2026-10-19, `runs/b27/window7.log`, `window8.log`):

- A build at d4afa3de5 (the old serving binary plus counters and comments) is bit-identical to the old binary.
  A build at c13dba6ea, whose only change reachable from the 27B GGUF tape is the head_dim 256 flash kernel
  refactor (the softmax scale moved from Q to the scores, an exact power of two), is already 0.12-0.17 from it.
- Against the portable fp32 attention kernel as the reference, the old binary, c13dba6ea, flash version 1 and version 2
  are all 0.11-0.27 away (v1 mean 0.132, v2 0.134 over four prompts); none is closer than the others.
- At the kernel, the same fp16 flash kernels are 3e-4 (unit scale) to 1.3e-3 (scores 16x larger, int8 KV) from the
  portable fp32 kernel by rel-L2 over the whole output, for both versions
  (`both_flash_versions_stay_close_to_fp32_at_large_score_magnitudes`). That is the accuracy of fp32-accumulated fp16
  tensor cores; the kernels are not the cause.

Cause: the model. The 27B is W8A8 (int8 weights and dynamically quantised int8 activations); a perturbation of the
attention output too small to matter anywhere else flips activation roundings downstream and the logits move by about
0.12. The qwen35 ledger already records the same number: the f32 cache's own decode tape against its own chunk tape
differ by mean logits rel-L2 0.134 on this checkpoint (`gguf_kv_tier_real.rs` holds a candidate to that floor, not to
an absolute). `qwen35_gguf_logit_check` now measures the floor in the same run (`tape_comparison_trace`) and
`--assert-within-floor M` fails when a setting's mean distance exceeds M floors.

Consequence for gates: a logits-level bound below the floor cannot be met by any kernel change, and a kernel-level bound
at unit-scale inputs cannot see score-magnitude effects. Gate the kernel against fp32 at large score magnitudes (3e-3
rel-L2) and the engine against the floor (3x), and report flips together with their margin.

Not a finding about the quality of the old kernel: it is simply one more point at the floor. Replies change under any
numerics change of this size, so a "0 of N replies identical" check is not a defect detector for this model.

Confirmation over eight prompts (window9, 2026-10-19 19:37, `runs/b27/window9.log`, card 1, 3070-3931 token prompts,
24 greedy steps each, int8 KV): the numerics floor is 8.85e-2 mean rel-L2; flash v1 is 1.279e-1 (1.44x the floor), flash
v2 1.292e-1 (1.46x), v1 with the fused front end and no drain 1.271e-1 (1.44x). v2 is within noise of v1 and well under
the 3x gate, so version 2 is the default again (-16 ms per 512-row prefill round, knowledge 327).

W8A16 (bf16 activations through the wgmma GEMM, scratch build) against the same portable-fp32-attention reference is not
closer or more stable than W8A8: mean rel-L2 1.26e-1 vs 1.29e-1 (v1 attention), mean KL 3.3e-2..5.6e-2 vs 3.8e-2..4.2e-2
and the same order of argmax flips; step-0 rel-L2 differs by prompt in both directions (0.19 vs 0.29, 0.49 vs 0.33).
Activation precision is therefore not what sets the 27B's floor, and the wgmma path stays off main.
