<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 4. Degenerate test dims hide whole bug classes

**Lesson:** Degenerate test dims hide whole bug classes. Choose toy dims where every dimension
**differs**, and gate fixtures that cover each distinct case.

## What happened
At T5-XXL, `heads == d_kv == 64` **and** `heads*d_kv == d_model == 4096`. Both toy configs
shared that degeneracy, so a head-count/head-width swap, or a `d_model`-vs-attention-inner-width
swap, would have passed a 19 GB gate at cosine 1.0.

The same rule catches the `kv_rows == num_queries + 1` trap in the IP-Adapter resampler: sizing
k/v at `num_queries` is shape-legal only at one query.

## Why
The rule applies to *which fixtures you gate*, not only to their dims. InstantID's 70
cross-attention sites come in two widths (640 and 1280) and `heads = hidden / 64` differs
between them. The first draft dumped sites 1 and 3, which are **both 640**, so a
width-dependent bug would have passed.

## What to do
- Use `heads=2, d_kv=64, d_model=64` for the T5 case, plus a checkpoint-free tiny gate at those
  dims.
- The InstantID dumper now picks one site per distinct width.
- The test **discovers the indices from the goldens** and asserts it saw ≥2 widths, so a
  re-dump cannot silently stop gating one.
