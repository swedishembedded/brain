<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 260. The 27B FP8 checkpoint streams tensor by tensor in about 1 GiB of host memory

What was measured. `qwen35::fp8_source::Qwen35Source` over the real
`Qwen/Qwen3.8-27B-FP8` directory (64 layer shards + `outside.safetensors` +
`mtp.safetensors`, 29 GB on disk), consuming every tensor the way a model
builder does (whole tensor, except `tok.weight`/`lm_head.weight` in 8M-element
row bands) and calling `advise_drop` after each one: 851 tensors, 107.6 GB of
f32 delivered, host `VmHWM` 1.13 GiB (cold page cache, 184 s) and 1.45 GiB
(warm, 82 s). `import_dir` needs the whole 108 GB as one host map. Run:
`BRAIN_QWEN35_DIR=<dir> cargo test -p brain-qwen35 --test fp8_source -- --ignored --nocapture`.
Every tensor of layers 0, 3 and 63 is bit-identical to `import_layer`'s.

Why it stays small. The FP8 dequantization is per 128-row band
(`dequant_block128` on rows `[r0, r0+n)` with the scale rows `[r0/128, ..)`), so
a 17408 x 5120 matrix never needs its fp32 copy plus its raw decode at once, and
the 5 GB embedding/head (bf16 on disk, no scale) goes through the mmap's own
chunked decode. Dropping a tensor's pages after use is what keeps `VmHWM` at the
working set instead of the file size.

Two traps found on the real files. (1) `mtp.safetensors` and the vision tower
carry their own `*.weight_scale_inv` planes: a scale whose weight is out of scope
must be dropped with it, not reported as an orphan (a synthetic fixture without
an FP8 `mtp.*` weight cannot show this). (2) The builder's `Weight::upload`
takes a whole f32 slice, so loading the 248320 x 5120 head through it still costs
5.1 GB (+ the packed copy) on the host; the source can serve it in bands, the
uploader has to consume them.

The host-peak bound itself is asserted on a synthetic multi-layer FP8 checkpoint
with a counting allocator (peak <= 2x the largest tensor while the model is 12x
that), because RSS cannot tell a streamed tensor from a leaked one at that size.
