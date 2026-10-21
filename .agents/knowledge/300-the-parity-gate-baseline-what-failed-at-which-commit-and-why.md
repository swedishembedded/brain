<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 300. The parity gate baseline: what failed, at which commit, and why

`make test/parity` was run on main (f9f909f7) and the failing tests were
re-run at the baseline 87c777c1 (before the NF4 tier, head-CE and the generic
layer commits). The gradcheck suite is the part that failed.

| Failure | Backend | Main | Baseline | Class |
|---|---|---|---|---|
| any test on the shared pooled device, run concurrently (`convtr2d_kernels`: DEVICE_LOST, `cross-group leak d=1.6`; `imaging_models`; a SIGSEGV of the lib binary) | Vulkan | 15/30 runs fail | 12/30 runs fail | pre-existing |
| same tests with `--test-threads=1` | Vulkan | 0/30 fail | 0/30 fail | the race needs concurrency |
| `bf16_train::*` (2 tests) | Vulkan | fail | not run | pre-existing: the Vulkan device has no bf16 storage; the harness asserts that and names the reason |
| `glue_fd_pinball_grad_w` | CPU | one failure (random seed) | not run | pre-existing test flaw: a central difference whose step crossed the kink |
| `conv2d_gd_reproduces_conv2d_at_groups1_dilation1` | CPU, Vulkan | 0/460 fail each | not run | not reproduced; CUDA here lacks NVRTC so that arm cannot run |

Nothing in this list was introduced by the NF4 tier, head-CE, the generic layer
or the trainer commits. On CPU the whole gradcheck suite passes; on Vulkan, serial,
everything except `bf16_train` passes (74 of 76 lib tests, all integration targets).

What changed: the gate's Vulkan line now runs serially and skips `bf16_train::`
(which stays required on the CPU line), each with the reason in the script; the
pinball probe draws clear of the kink. The concurrent shared-device race is a real
backend defect (wrong results, not only a lost device) and is on the Vulkan roadmap.

Also fixed in this pass: the four fused-attention dispatch tests of qwen3 now
expect the fused kernel where the device has workgroup reductions and the triad
where it has none (they failed on the CPU backend); `paged_flash_prefill_hd256`'s
`@tpl pool_k, pool_v` header (space after the comma) was parsed as an empty binding
by the dtype-tier gate; see #301 for the shared-prefix flake.

Open (root cause in knowledge 353): `caps::tests::an_unseeded_lora_train_echoes_the_seed_that_reproduces_it` failed
once under CPU load and could not be reproduced (about 15 runs under 48 to 60 busy
loops, 3 full loaded lib runs: one passed twice, one run hung past 1400 s in the
sibling `a_trained_adapter_blob_folds_into_the_base_and_changes_its_weights` and was
killed by the timeout). The CPU kernels are atomic-free and range-partitioned, so a
bitwise difference is not expected from scheduling alone.
