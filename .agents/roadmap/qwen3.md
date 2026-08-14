# qwen3 - roadmap

Qwen3 dense decoder transformer (GQA, QK-norm, RoPE, SwiGLU) with a
concurrent paged-KV serving engine. Training, LoRA fine-tuning, INT8 weights,
INT8 KV cache, sharding, and continuous batching are built and verified
against the reference.

## Not yet done

- [ ] Prefix caching across requests (the underlying infrastructure exists
      but is not wired into the serving engine)
- [ ] An FP8 / E4M3 weight path (INT8 is the only quantized weight format
      today)
- [ ] Mixture-of-Experts serving - only dense configurations are supported
      end to end
