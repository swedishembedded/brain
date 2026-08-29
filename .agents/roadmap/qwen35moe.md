# qwen35moe - roadmap

Qwen3.5-35B-A3B: a hybrid decoder mixing Gated DeltaNet (chunked linear-attention) layers with GQA full-attention layers, a 256-expert sparse MoE on every layer, and a natively-multimodal vision tower.

## Not yet done

- [ ] The prefill/training tape's RMSNorm BACKWARD (`rms_inv`/`rmsnorm_dx`) is still the per-element kernel; only the forward selects the coalesced `rmsnorm_rows` (measured 16.3x on one decode token's norms).
- [ ] Multi-GPU INT8/INT4 residency serving (the full `PagedDecoder` engine), including layer-range sharding across GPUs for the served path - the full model does not fit a single GPU's memory.
- [ ] Multi-sequence GPU batching, prefix-cache reuse, chunked/batched prefill, and speculative decode in the serving engine.
- [ ] INT8/INT4 paged KV cache for served decode.
- [ ] INT4 quantized inference (weight quantization) for this model specifically.
- [ ] LoRA adapter folding into the served model, and int8 weights on the serving path (currently only reachable via a separate, non-serving entry point).
- [ ] Incremental/KV-cache decode-time image splice for multimodal generation - only whole-sequence prefill with an image is supported today.
- [ ] Video (multi-frame) multimodal input - only single-image input is supported.
- [ ] A working NPU run: export currently reaches "compiles" only, at a fixed sequence length, text-only (no vision export, no dynamic sequence length).
- [ ] Overlapped (GPipe-style) pipeline execution - the sharded pipeline currently runs strictly sequentially, one GPU active at a time.
- [ ] An INT4/INT8 frozen base with a trainable (backward-capable) path - needs a dequantizing backward matmul kernel that doesn't exist yet.
- [ ] Register-tiled (non-naive) GEMM for weight gradients - a performance gap, not a correctness one.
- [ ] Numerical parity against the reference implementation - only structural correctness of the forward pass is currently established.

The dense (non-MoE) sibling and its MTP head are **not** in scope for this
crate - they are `crates/qwen35` (llama.cpp `LLM_ARCH_QWEN35`, distinct from
this crate's `LLM_ARCH_QWEN35MOE`), a separate architecture with its own
roadmap. See `.agents/roadmap/qwen35.md`.

A full-precision import of this model is impractical at its parameter count and does not fit alongside everything else on typical development hardware, so quantized (INT8/INT4) device buffers must be constructed directly from the compressed checkpoint format rather than via an intermediate full-precision file.
