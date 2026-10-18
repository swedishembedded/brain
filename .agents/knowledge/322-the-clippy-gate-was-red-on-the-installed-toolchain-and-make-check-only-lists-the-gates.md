<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 322. The clippy gate was red on the installed toolchain, and `make check` only lists the gates

Question: does brain's own gate chain pass at `origin/main` (5c1aa4863) on the toolchain this node has
(rustc/clippy 1.99.0, 2026-09-28, unpinned `stable`)? Every figure is **VERIFIED-BY-RUN** from the log of
`cargo clippy --all-targets --keep-going` over `scripts/build/engine-pkgs.sh` (the gate's own command).

## What was found

- `make check` runs nothing: it lists the `check/*` targets (the Makefile says so; `test/full` is the
  aggregate). A caller who runs `make check` and reads "exit 0" has run no gate. Run each
  `check/<gate>` (clippy, env-reads, files, generated, make-targets, paths, samples, scope, scripts,
  sdk-features, seed-literals, spdx, test-skips, workspace).
- `check/clippy` failed at HEAD, first with two deny-by-default errors that abort the lint pass for every
  crate after them (the gate reports "the lint pass ABORTED" and prints the error line only):
  `not_unsafe_ptr_arg_deref` on `backend_cuda::cublas::Blas::new` (a public fn that hands a raw stream
  handle to the vendor library; it is now `unsafe fn` with the contract written down) and
  `while_immutable_condition` on the prefill admission loop in `model::serve` (the loop now sits in the
  `else` of the `if batched`).
- Behind those two errors the gate counted 1094 distinct warnings against a baseline of 0:
  675 `excessive_precision` (float literals), 234 `chunks_exact` with a constant size
  (`as_chunks`), 49 doc-list indentation, 26 `type_complexity`, and about 110 others
  (`is_multiple_of`, `checked_div`, `sort_by_key`, `large_enum_variant`, `too_many_arguments`, ...).
  All are cleared, with no `#[allow]` added, and the gate prints `0 warnings (baseline 0)`.
- `check/generated/kernels-cuda-table` had drifted from the registry (two catalogue lines); regenerated.

## Why it matters

The toolchain here dates from 2026-10-04; brain commits since then were not held to the gate that
would have caught each addition, or it was masked by the first deny-level error. Lints new in a toolchain
step arrive in bulk; the cheapest defence is to run the clippy gate (not `make check`) on every push and
to read its first lines, since an aborted pass says nothing about the crates it never reached.

## Follow-up not done

`clippy.toml` thresholds and the toolchain are not pinned, so the next `stable` bump will do this again.
