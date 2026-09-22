<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 33. A checker that cannot tell code from prose fails on the best-documented file

The kernel catalogue cross-checks each `@cpu` declaration against the file's
`workgroupBarrier()` count, because a kernel with two or more corrupts memory on
the CPU JIT (#26). It counted the word over the **raw source**, so it also
counted every mention in a comment - and the kernels most likely to discuss
their barrier discipline are exactly the cooperative ones the check exists for.

It fired on a correct new kernel (`paged_decode_scores_wg`: one barrier in code,
green on `backend-cpu`) whose header said "Exactly ONE top-level
`workgroupBarrier()`". Counting code only then revealed the seeded catalogue had
been wrong about **four** kernels all along:

| kernel | published | actual | barriers in code |
|---|---|---|---:|
| `layernorm_rows` | ✗ CPU | ✓ | 1 |
| `gradnorm_part` | ✗ CPU | ✓ | 1 |
| `prelu_bwd_wg` | ✗ CPU | ✓ | 1 |
| `conv2d_tiled` | native only | native | 1 |

Every one had a comment mentioning the barrier. Verified after correcting:
`compile_all` and `make gradcheck` are both green on `BRAIN_DEVICE=cpu`, so the
CPU claims now hold.

Two things worth keeping:

* **The failure mode is inverted from the usual one.** A checker with false
  positives does not merely annoy; it trains you to distrust it, and this one
  fired first on a *correct* file. Had the new kernel not been documented, the
  four wrong rows would have shipped indefinitely.
* **A derived value seeded from a buggy derivation stays buggy after the source
  of truth moves.** The `@` blocks are hand-maintained now, but they were
  *seeded* by the same comment-counting code, so fixing the checker was not
  enough - the seeds had to be recomputed too. Any bootstrap-then-hand-maintain
  migration carries this: the bootstrap's bugs are baked into the data.
* **The bug was in two files because the derivation was.** The seeder and the
  checker each had their own copy of "count the barriers, read the workgroup
  size, decide the tier", so one defect needed two fixes and could be
  half-fixed. They now share `scripts/build/kernelmeta.py`, and the property
  that matters is asserted: for every kernel, what the seeder would *propose*
  for the mechanical fields is exactly what the checker *demands*. Without
  that, seeding a new kernel could emit a block that immediately fails the
  build.

The same comment-blindness was in **production** code:
`backend_api::workgroup_size_of` took the first `@workgroup_size` anywhere in
the source, and several in-repo kernels document the attribute in their header
prose, which sits above the declaration. All of them happen to state the right
number, so nothing was broken - but a single stale or aspirational comment would
have laid out every backend's dispatch grid with the wrong size, while the
kernel reconstructed its flat invocation id from a different one. Its own doc
comment admitted it was relying on the parity tests to notice. It now scans code
only, with a test that pins it.
