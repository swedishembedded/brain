<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 1. A gate that never runs is worse than no gate

It reports green and nobody looks again. Four separate instances, all found in
one workstream:

| Gate | Why it never ran |
|---|---|
| `sam2` parity | `scripts/fetch-testdata.sh` had no sam2 entry; the checkpoints had been hard-linked in by hand, so the test self-skipped on every machine but the one that created them |
| `flux2` host/device parity | `hidden: 16` put a modulation slice at byte offset 192, violating the 256-byte storage-binding alignment - it failed on every GPU, and had simply stopped being run |
| `cargo clippy` | aborts at the first deny-by-default lint and then reports nothing about everything after it - a 123-file backlog stayed invisible **twice in one day** |
| `wgsl-cpu` `compile_all` | one kernel out of the whole set would not compile on the CPU JIT, so int8 quantization had no working CPU path at all |

**Rules that follow.** A test that self-skips must say so loudly and its fixture
must be provisioned by `make fetch/testdata`. A tool's **exit code** is the
signal, not its stdout (`make clippy` checks both, separately). And when a suite
reports "N passed", check how many of those N were skips - an implementer
reported 9 vqgan tests passing when 7 of them were `SKIP: set BRAIN_VQGAN_WEIGHTS`.
