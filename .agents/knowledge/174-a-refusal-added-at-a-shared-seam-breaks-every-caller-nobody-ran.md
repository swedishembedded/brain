<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 174. A refusal added at a shared seam breaks every caller nobody ran

`Gpu::step` started refusing workgroup-cooperative kernels (a raw thread
count cannot describe their dispatch). The refusal is right, but it landed
without every existing caller being moved to `dispatch(..,
Dispatch::Workgroups(n))`, and the callers it missed are the ones no test on
the machine exercised. From then on they panicked instead of dispatching:

- `qwen3::q8::Q8::mm8` - the int8 matmul of every `Q8` linear;
- the int8 linears of `wan` and `ltxv` (`matmul_i8_dyn`);
- `vae::blocks` and `vae::blocks3d`'s lowered convolutions (`matmul_reg3`
  and the `nlc_bias_nchw` epilogue), so every VAE conv above the GEMM
  threshold, and their backward (`matmul_dw_reg`, `matmul_dw_reg_splitk`,
  `matmul_dx_reg`);
- `audio::conv`'s lowered conv1d, `model::block`'s fused hd256 paged
  prefill, gpt2's and flux2's backward GEMMs, `vision::PReLU`'s backward.

Every one of them had passed `threads = workgroups * wg_size`, so the numbers
were right; only the API had changed under them.

**Rule:** a new refusal at a seam every model shares ships with a sweep of
that seam's callers (a static scan for the refused shape, not only the tests
that happen to run), and each caller is moved in the same change.
