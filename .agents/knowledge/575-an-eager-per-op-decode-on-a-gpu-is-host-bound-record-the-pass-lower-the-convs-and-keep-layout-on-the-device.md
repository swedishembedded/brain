<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 575. An eager per-op decode on a GPU is host-bound: record the pass, lower the convs, keep the layout on the device

**Lesson:** the Qwen3-TTS codec decode (`mimi::Codec::decode`) was written as a CPU-backend prototype: every op was its
own `gpu.submit` into a freshly allocated output, layout flips and the conv bias broadcast went through the host, and
the convs were the direct one-thread-per-output kernels. On a GH200 that cost 1.9-3.1 s for 100 frames (8 s of audio) and
7-8 s for 600 frames, served through the real TTS path. nsys of the same decode attributed it three ways, none of them
visible in a correctness test:

- 64.5 % + 34.6 % of GPU kernel time was `brain_conv1d` (75 ms average) and `brain_convtr1d` (196 ms average): the
  direct kernels reach a low single-digit percent of the card's roof on 512-1536 channel convs;
- 70 % of CUDA API time was `cuMemcpyHtoDAsync` (1 898 calls: the broadcast biases and re-uploaded transposes), 19 %
  `cuMemFree` and 7 % `cuMemAlloc` (1 815 each over the four decodes profiled: one allocation per op, 0.7-2 ms apiece);
- the host did the layout work: every transpose read the buffer back, permuted on the CPU and uploaded it again, and the
  ConvNeXt LayerNorm and GELU read back a `[L, 4C]` activation to compute on the host.

What fixed it, all from parts that already existed: a `Pass` that records its steps and submits them in batches
(`mimi::model`), `audio::conv::conv1d_bias_fwd` / `convtr1d_bias_fwd` for every conv (GEMM lowering with the bias fused,
the direct kernel where the selector says so), the `nchw_nlc` / `nlc_nchw` permute kernels, the `layernorm` kernel with
the reference's eps as its parameter and `gelu_erf`, and `hold_freed_blocks` / `retain_freed_blocks` so a repeated decode
takes its blocks from the backend's pool instead of the driver. Only the final waveform read crosses to the host.

The recording needs a memory bound. Submitting once at the end keeps every intermediate of the graph alive until then; at
600 frames the SEANet stages alone are tens of GB. The pass submits whenever 1 GiB of fresh activations has been
allocated, so a long clip's peak is bounded, a 100-frame clip still goes out in a few submissions, and the blocks of a
submitted batch return to the pool for the next stage.

The lowered convs reassociate the `Cin*K` sum, so a gate must be a tolerance. Measured against the CPU backend (which
always takes the direct kernels, so it is an independent witness): relative rms 3e-6 on a synthetic codec wide enough to
reach the lowering, 1.2e-4 on the real checkpoint's device-vs-host codec test (gate 0.02).

## Evidence (2026-11-24)

GH200 (a shared card, other tenants running: wall times move by up to 1.5x between runs), real
`Qwen3-TTS-12Hz-0.6B-Base` codec, random codes, second call of the same decode:

| frames | before | after |
|---|---|---|
| 100 (8 s) | 1.9-3.1 s | 0.08-0.25 s |
| 600 (48 s) | 7.6-7.9 s | 0.31-0.75 s |

GPU kernel time summed over the four decodes of one run (100 and 600 frames, cold and warm): 13.6 s before (conv1d 8.8 s,
convtr1d 4.7 s), 0.65 s after (the GEMM `matmul_f32_reg` is 51 %, `im2col1d_at` 17 %). Driver calls: 1 815 `cuMemAlloc`
plus 1 815 `cuMemFree` plus 1 898 host-to-device copies before; after, activations come from the stream-ordered pool
(`cuMemAllocFromPoolAsync`) and the remaining `cuMemAlloc` calls are the weight upload at load.

Reproduce: `cargo test --release -p brain-qwen3tts --test it codec_decode_timing -- --ignored --nocapture`, under
`nsys profile -t cuda --sample=none` for the kernel and API split.
