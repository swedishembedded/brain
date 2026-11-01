<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 413. Two projections that read one activation are one FP8 weight, and a tuning class timed at one width ranks the candidates wrongly for the widths around it

Question (the brief of 2026-11-01): a decode step ran five FP8 products per Gated DeltaNet layer where the reference server runs four (one 16384-wide
`qkv | z` product at 3.28 TB/s against brain's 10240-row and 6144-row pair, ~14 us a layer, 48 layers). Can brain hold and multiply the pair as one
product without slowing prefill and without a second copy of the weights?

## What was built

- **One stored weight.** A GDN layer of an FP8 inference build uploads `in_proj_qkv` and `in_proj_z` as one `[16384, 5120]` weight
  (`upload_fused_from_source`: rows `qkv | z`, scales concatenated, because block scales are per 128 x 128 block and 10240 = 80, 6144 = 48 blocks).
  The two names stay the checkpoint's: `Qwen35::linear(name)` returns the weight and the row window of it (tier, values, precision report, bytes all
  answer per projection; a rewrite of one window is refused loudly, it would requantize its neighbour's blocks).
- **Windows are weights.** `fp8_forward` takes a window of whole scale blocks (`first`, `count`): the candidate the tuned table names for the WINDOW's
  shape, bound to the window's bytes and scales through the sliced bindings every native step already had. `fp8_row_windows` holds that a window equals
  its rows as a separate weight bit for bit at 1-2048 rows, f32 and bf16 output, so prefill and mixed rounds (two window products) are unchanged.
- **Decode and verify rows read the product in place.** `params[7]` of `gdn_decode_pool_stream_bf16` / `gdn_verify_pool_stream_bf16` is the row stride
  of the fused product (0 = two arrays); the kernels read `qkv` and `z` columns of the same rows. No split copy. Same kernel, same launch
  configuration (knowledge 412 asks a new name only for a new launch form). Byte-identical to the launches over two arrays; the f32-pool forms are not
  strided (a build without a bf16 pool keeps two products). A step takes the one product for at most 128 rows, no adapter rows and no LoRA.

## Isolated products (gpu1, `fp8_qkvz_one_product_against_two`, best candidate each, us)

| rows | qkv 10240 | z 6144 | sum | one product 16384 | one product TB/s |
|---|---|---|---|---|---|
| 1 | 21.3 | 14.7 | 36.0 | 30.5 | 2.75 |
| 16 | 24.0 | 16.0 | 40.1 | 32.3 | 2.60 |
| 32 | 23.3 | 16.4 | 39.7 | 31.6 | 2.66 |
| 64 | 25.6 | 19.3 | 44.9 | 33.3 | 2.52 |
| 128 | 29.5 | 26.1 | 55.5 | 35.6 | 2.35 |

7.8 us a layer at 16 rows, 11.6 at 64 (the reference's 25.6 us at 16 rows is 3.28 TB/s; brain's best candidate stays at 2.6 TB/s: open).

## A finding on the way: the tuning class ranked candidates at one width

The tuned table names ONE winner per (rows, K, weight bytes) class, timed at one representative width. The FP8 class of 48-128 MB was timed at 12288 rows, but
the model runs 10240 (qkv), 12288 (query), 16384 (fused qkv | z) and 17408 (MLP gate and up, 128 launches a step) in it, and the ranking reorders between them
(`fp8_class_widths`, K = 5120, us, best first):

| rows | n | order |
|---|---|---|
| 16 | 12288 | t4w4d3b3 27.4, a2w4d4b3 27.4 |
| 16 | 16384 | t8w8d2b1 32.5, swap_tma_bn136c1 34.0 |
| 16 | 17408 | swap_tma_bn144c1 34.5, bn136c1 35.1, t8w8d2b1 39.1, t4w4d3b3 41.1 |
| 64 | 12288 | n64b64s2 28.8 |
| 64 | 17408 | swap_tma_bn136c1 36.1 |

The pick for the whole class (a2w4d4b3 at 16 rows) ran the MLP's gate and up ~7 us a launch behind the best candidate. The class now splits at 80 MB
(12288 rows is timed at 12288, 16384 and 17408 at 17408; K = 17408 at 3072 and 5120 rows); 120 classes tune in 37 s here (was 29 s for 96), so the budget is 60 s.
Rule: a representative width must be a width the model serves; list the ranking at every served width of a class before trusting one winner.

## Measured

nsys of the served 8K cell, decode-regime FP8 kernels per decode step (steps counted from the GDN launches), base -> fused weight only -> fused weight + grid:
16 rows 11.79 -> 11.13 -> 10.95 ms (355 -> 314 -> 312 launches a step); 64 rows 10.52 -> 8.15 ms (251 -> 222 launches; the gate and up now run `swap_tma_bn136c1`
at 47 us where `n64b64` ran 52). `qwen35_fp8_parity` (8 x 1500 tokens, 32 steps, model.rs / gpu-core lib.rs as texts) identical to base to every printed digit
(98.05 % / KL 2.314e-3, 97.27 % / 2.238e-3). Served, gpu1, SPEC unset, two interleaved rounds, mean of two (tok/s, base -> new): 8K x 16 311.8 -> 319.5, 8K x 64 379.9 -> 381.7,
32K x 16 86.6 -> 87.6; 8K x 1 92.9 -> 81.6 with speculation on, where the controller's plain step alone ranges 11.4-15.0 ms inside ONE binary (acceptance on random tokens),
and 70.9 -> 69.2 with speculation off (66.1 and 72.3 against 70.7 and 71.1: noise). The strided verify kernel costs ~2 us a launch more at one sequence (17.8 -> 20.1 us).

## Open

- gate and up as ONE 34816-row FP8 product (the reference does: 58.9 us at 64 rows against two products of ~36 us here) is the same lever on the largest family.
- The f32-pool streaming forms and the segments' mixed rounds (decode rows leading a prefill window) still run two products.
- The best 16384-row candidate reaches 2.6 TB/s; the reference's 3.28.
