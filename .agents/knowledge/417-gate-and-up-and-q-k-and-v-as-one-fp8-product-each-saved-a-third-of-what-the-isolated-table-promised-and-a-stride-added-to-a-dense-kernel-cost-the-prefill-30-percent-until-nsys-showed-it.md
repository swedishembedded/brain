<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 417. Gate and up, and q, k and v, as one FP8 product each saved a third of what the isolated table promised, and a stride added to a dense kernel cost the prefill 30 % until nsys showed it

Question (the brief of 2026-11-01): the decode step at 16 sequences is 18.6 ms against the reference server's 15.7. It runs four FP8 products a layer, brain five to seven. The tuned table
(`fp8_class_widths`) puts a 1024 x 5120 product at 10.6-11.4 us at any row count and one 34816-row product at 60.9 us against 2 x 36.2 for gate and up: ~11 us x 64 layers for gate|up and
~20 us x 16 for q|k|v, ~0.9 ms of the step. Hold each pair/triple as one weight, as `in_proj_qkv | in_proj_z` already is (knowledge 413).

## What was built

- `Fusion` (`crates/qwen35/src/model.rs`): `Qkvz`, `GateUp`, `Qkv`. An FP8 inference build with native FP8 products holds each group a layer has as one weight (`upload_fused_from_source`, no second
  copy). `Qwen35::linear` answers for every name by its window; prefill, mixed rounds, adapters (bank rows), LoRA and training multiply the windows, which are weights of their own (numerics unchanged).
  A decode or verify step of at most 128 rows multiplies the whole weight once. The A/B switch `set_fused_qkvz_decode` is now `set_fused_projections_decode` and switches all three.
- gate|up: `quant_epilogue` (params `[k, rows, mode, flags, stride, b_offset]`) and `silu_mul_quant_e4m3_128` (`[rows, k, flags, stride, up_offset]`) read gate and up at a row stride and the up
  columns' offset. Dense pair = stride k, offset 0; the halves of one `[n, 2 ff]` product = the product twice, stride 2 ff, offset ff. Byte-identical to the dense launch over the halves copied apart (fp32 and bf16
  rows, with and without the stored product, 1-128 rows). The bf16-output form (`matmul_rows_bf16_out` over the whole weight) is covered by the same kernel; the tiny-model test does not reach it (its candidates
  have no bf16 store at that width), the kernel test and the served run do.
- q|k|v: one `qkv_split` launch (a copy kernel) takes the `[n, 2 qd + 2 kvd]` product apart. Where the native decode front half (`gqa_decode_prep`, f32 and int8 KV) will run, it makes `q_full` (the projection's
  interleaved rows), `k`, `v`; otherwise (bf16 KV chain) it makes the query's values and gates apart, which replaces both `concat_split` launches (9.2 us each, served). `decode_prep_form` asks which, before
  the split. Reading the windows in place in the front-half kernels was not built: the copy is 2 us, and the in-place form would add a stride to the kernels the int8 default and the f32 KV tiers share.

## Numbers (gpu1, FP8 / bf16 KV / f32 state / no speculation, 8K x 16, nsys, decode-only steps)

| | base | fused |
|---|---|---|
| median step span | 17767 us | 17574 us (-1.1 %) |
| kernel busy a step | 17207 us | 16911 us |
| `swap_tma_bn136c1` (launches / us a step) | 176 / 5316 | 111.6 / 5070 |
| `dq_stream_r4` (the k and v products) | 32 / 391 | 0 |
| `gemv_f16_a2w4d4b3` (q, then q|k|v) | 16 / 332 | 15.9 / 500 |
| `concat_split` / `qkv_split` | 32 / 57 and 0 | 0 and 15.9 / 33 |
| `quant_epilogue_cluster` | 80 / 305 | 79.7 / 324 |

q|k|v: -248 us a step (15.5 us a layer, the table said 20); gate|up: ~-230 us (3.6 us a layer, the table said 11); the rest of the step moved +180 us (`gemv_f16x_t8w8d2b1` +78, the others small): the sum
is -296 busy. Why the step gains a third of the table's figure is not diagnosed: the table times the best candidate of each shape alone, and the step's products run back to back under programmatic launch. The kernels, isolated (roof bench, `glue_kernels_against_the_byte_roof_at_a_4096_row_round`): `quant_epilogue` silu*up 16 rows 6.2 us dense / 6.3 in place, 64 rows 11.1 / 11.0;
`silu_mul_quant_e4m3_128` 4096 x 17408 bf16 165.8 us dense / 189.6 in place (the decode widths are 17-128 rows).

Served (matrix.sh, 2 interleaved rounds, mean tok/s base -> fused): 8K x 1 71.7 -> 73.6 (all three pairs of runs positive), 8K x 16 280 -> 293 (rounds 265/296 and 300/286: the cell is +-6 % between
runs of one binary), 8K x 64 346.0 -> 346.2, 32K x 16 80.4 -> 79.7; int8 default 8K x 16 (median TPOT) 46.2 -> 44.9 ms. A real, small, decode-step gain (~1 %) inside the served noise.

## What went wrong on the way (all caught by a measurement, not by reading)

- Adding a row stride to `silu_mul_quant_e4m3_128` made every block divide by `k / 128` in 64 bits: the 4096-row prefill pass went 173 -> 223 us (nsys kernel summary, 4608 instances), and the first served
  A/B showed it as a worse TTFT at 8K x 64. The dense pair (stride == k, uniform per launch) takes the flat index again. A change to a kernel every prefill round runs needs its dense-case time checked
  in the same pass.
- A test that pins one candidate and expects bit-identical windows and whole product needs a width the candidate serves: at d_model 128 `matmul_fp8_dq_stream_r4` declines, the tuner keeps its own choice and
  the two forms differ by 2e-3 (relative L2 of the logits). The fixtures use d_model 512.
- A kernel binding the same buffer as an output twice (the unused fourth output of `qkv_split` in the interleaved form) is refused by the usage-scope check; the unused slot is a one-word buffer.

## Gates

`qwen35` tests as on the base (the failures with `BRAIN_DEVICE=gpu1` set are the CPU-backend tests that want the CPU; `fp8_source`, `lora_seq_len`, `model_i8_smoke`, `model_q4_smoke` fail identically on the base).
Logits of the fused decode equal the windows' bit for bit with one candidate pinned (2 and 20 rows; f32 and bf16 KV for q|k|v); with the tuner free they agree to 5e-2 relative L2 (asserted;
2e-3 where measured). `qwen35_fp8_parity` (8 x ~1300 tokens, 32 steps, model.rs as text, int8 KV) base -> fused: greedy agreement 97.66 -> 97.27 % (one more flip, prompt 5 step 27, at a reference margin of 0.015), mean rel-L2 2.612e-2 -> 2.609e-2, worst rel-L2 8.987e-2 both, mean KL 2.463e-3 -> 2.437e-3 nats, flips at margin > 0.5: 1 both: the wider product's tuned candidate moves a near-tie. Stress (ignored): the three wide-product kernels 2,000 launches each, swap-TMA back-to-back over 5120 x 34816,
5120 x 17408 and 5120 x 14336; `dmesg | grep -ci xid` unchanged.

## Open

The step is still 17.6 ms against 15.7: the products are ~9.3 ms of it at 5120-wide K and the reference's `fp8_gemm_kernel_swapAB` runs the same bytes in 8.5 ms. The fused 34816-row product at 16 rows is the
`swap_tma` kernel at ~60 us (2.1 TB/s of the weight's 178 MB); the 3.3 TB/s the reference reaches on the 16384-row product is the number to chase. Reading q, k and v in place in `gqa_decode_prep` and
`gqa_qk_prep` would remove the 2 us split; a mixed round's leading decode rows still multiply the windows.
