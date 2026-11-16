<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 481. A MoE router on the library's TF32 GEMM gave a sequence other tokens beside a second row, because its top-k turns a rounding into another expert

**Lesson:** An fp32 leaf of an int8 model is not exact because its weight is fp32: under `GemmPrecision::Auto` a
product of two rows or more takes the vendor library's TF32 GEMM, whose operand rounding and algorithm depend on the
row count. For a router that rounding picks other experts, and the sequence's logits part by the int8 cascade size.
A decode step that must not depend on its co-batch keeps its fp32 products on the fp32 kernels.

## What happened

`brain perf run serve --target qwen35moe:<Q8_0 GGUF>` failed its own fidelity gate on one GH200
(`greedy_token_match 0.7778 ... first divergence at sequence 0 position 3`) and `qwen35moe` `serve::
batched_decode_matches_each_sequence_decoded_alone` failed on CUDA. Bisected on the real checkpoint with a probe
decoded alone and again beside other sequences:

- the probe's next-token logits parted by relative L2 2.8e-2 at two rows; switching off decode tapes, decode fusion,
  the gather/grouped expert split, or forcing the int8 products onto the packed activations
  (`model::ops::row_count_invariant`) changed nothing, while `GemmPrecision::Fp32Reference` made two rows bit-equal
  to one;
- the fp32 product of an int8-tier decode step is the router with its shared-expert gate (`[257, 2048]`): one row
  went to the native fp32 GEMV, two rows to `library_gemm ... tf32`. The tiny fp32 engine of the failing test has every
  linear fp32, the same switch.

## What to do

- `model::ops::fp32_kernels(|| ..)`: fp32-weight products on the fp32 kernels inside the scope, never the TF32
  library GEMM. The MoE engine's decode step (recorded or built, every head) runs inside it; prefill keeps TF32.
- Measured cost on gpu3 (`perf run longctx --context 32768 --ladder 1,16,64`, two alternating A/B pairs): decode
  4.54 / 12.62 / 28.76 -> 4.83 / 13.02 / 29.12 ms per step at 1 / 16 / 64 rows, prefill unchanged. It is the router on
  `matmul_f32_gemv` (14.4 us a call, a 2 MB weight read at a few hundred GB/s: 17 blocks of 16 weight rows leave most
  multiprocessors idle and each row waits on a chain of load round trips) against 6 us on the library. Narrower blocks and deeper loads did not move it; a GEMV
  that keeps the reference's 64 virtual lanes on 64 threads a row would.
- Not pinned, and not free: past a few rows the int8 GEMV switches to its tensor-core block (from 5 rows) and the
  tuned row buckets (9, 33), and the fp32 GEMV hands 33 rows to `matmul_f32_skinny` (another summation order); the
  probe's logits part by 1.2e-2 to 2.7e-2 from 7 rows. Only an exact-decode regime (#457: row-invariant candidates and
  grouped rows) holds wider batches; the MoE engine has none yet.

## Evidence

`qwen35moe/tests/it/serve.rs` `batched_decode_matches_each_sequence_decoded_alone` (fp32 tiny, now passes on CUDA);
`qwen35moe/tests/it/fast_path_parity_real.rs` `a_sequence_decodes_alike_alone_and_beside_others_on_the_real_checkpoint`
(bit-identical logits at 2 and 3 rows, `BRAIN_QWEN35MOE_GGUF`). The perf serve run's fidelity gate passes. Ledger:
`.agents/roadmap/qwen35moe.md`, "Served from the released GGUF".
