<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 173. An epsilon a kernel compiles in is a config field nothing reads

`rmsnorm.wgsl`, `rms_inv.wgsl`, `rmsnorm_dx.wgsl` and `rmsnorm_dx_rows.wgsl`
added a literal `1e-6`, and `model::block::rmsnorm_fwd`/`rmsnorm_bwd` took no
epsilon at all. Every config struct still parsed `rms_norm_eps` faithfully,
so the value looked honoured everywhere it was read and was ignored
everywhere it mattered. Models normalized at an epsilon they never declared:

| model | declares | ran at |
|---|---|---|
| LLaVA's Vicuna/Llama-2 decoder (`QwenConfig::llama2_13b`) | 1e-5 | 1e-6 |
| GLM-5.2 (`glmdsa`) | 1e-5 | 1e-6 |
| Kronos (reference `RMSNorm(eps=1e-5)`) | 1e-5 | 1e-6, GPU and host decode |
| Mimi's pre-transformer (`mimi`) | 1e-5 | 1e-6 |

and every Llama-3 checkpoint would have joined them. Two models had noticed
and asserted `eps == 1e-6` instead (`deepseek2`, `deepseekocr2`), one of them
with an absolute tolerance of `1e-4` that would have waved a 1e-5 config
through. Six more had worked around it with byte-identical `_eps` kernel
twins that took the value as a param.

The two could not be told apart by the existing variant-agreement gates: their
inputs are O(1), where `mean(x^2)` dwarfs any epsilon. `rmsnorm_variant_
agreement.rs`'s `every_variant_normalizes_at_the_callers_epsilon` (and its dx
twin) use ~1e-3 inputs at eps 1e-2, where a kernel still adding 1e-6 misses by
~81x (measured by reverting `rmsnorm.wgsl` to the literal).

**Rule:** a numeric constant of the architecture (epsilon, RoPE base, scale)
is a kernel `Params` field and a helper argument, never a WGSL literal; and a
test of it uses inputs small enough that the constant moves the answer.
