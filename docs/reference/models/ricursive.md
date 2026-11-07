# Ripple-65B (library engine, not yet served)

Ripple-65B (`RicursiveForCausalLM`, `model_type: ricursive`) is an 80-layer
dense decoder that mixes two kinds of attention layer, four of one then one of
the other: sliding-window layers with RoPE, and global layers with full causal
attention and no positional rotation. Each block also has a per-layer value
residual, a sigmoid-gated attention output, per-head QK RMSNorm, four norms
(`input`, `post_attention`, `pre_feedforward`, `post_feedforward`) and a
clipped SwiGLU feed-forward.

## Support

| Capability | Supported |
|---|---|
| Config and two-way tensor import from the released safetensors | [x] |
| f32 prefill forward (every position's logits) on every backend | [x] |
| Parity with the reference implementation (random weights, real layers) | [x] |
| Paged serving engine under the generic scheduler (library) | [x] |
| Serving over the CLI, HTTP API or D-Bus | [ ] |
| Weights held as the checkpoint stores them (bf16; the official ModelOpt NVFP4 release with the MLP in NVFP4) | [x] |
| Weight formats `--weight-format` f32, bf16, f16, int8, fp8 | [x] |
| KV cache `--kv-cache` f32, bf16, int8, fp8 (global pools and sliding rings) | [x] |
| Layers split over several devices (pipeline stages) | [x] |
| Training | [ ] |

**Read that table before relying on this.** `RicursiveEngine` implements
`model::serve::PagedDecoder`, so a program can run it under
`model::serve::Scheduler` with continuous batching, chunked prefill and prefix
reuse, and it passes the shared serving conformance suite. It is not in the
model catalog: `brain serve` and the capability interface do not know it yet.
Its global layers can read through latent sparse attention
(`RicursiveEngine::set_lsa`, an indexer covering only global layers, over an
f32, int8 or fp8 cache, also when the layers are split over stages), which
needs a device that runs workgroup-barrier kernels. The registry row is the model's name and its
Hugging Face class.

### Weights, KV cache and devices

By default every linear weight is held as the checkpoint stores it: the bf16
release stays bf16 on the device, and
`ricursiveAI/Ripple-65B-Preview-NVFP4` is loaded as published, with no
conversion step, its MLP in NVFP4 (E2M1 codes, an E4M3 scale per 16 weights,
an f32 per-tensor scale) and its attention, embedding and `lm_head` in bf16.
`--weight-format` picks another format (`f32`, `bf16`, `f16`, `int8`, `fp8`,
or a list such as `bf16,mlp=fp8`); NVFP4 cannot be asked for, because nothing
quantizes into it. A format the device cannot hold is refused naming the
tensor and the format it would have used (int8 needs a packed-int8 dot, which
the CPU backend does not have).

The NVFP4 release is W4A4: it stores a static input scale per quantized
module. Brain keeps activations in float (W4A16, at least as precise as the
reference), so the scales are checked at open (one finite positive scalar per
module, none missing) and not applied.

`--kv-cache` stores K and V as `f32`, `bf16`, `int8` or `fp8` in both the
global pools and the rings; each element is widened as the attention kernel
loads it. The D-LSA capture reads unquantized keys and so needs `f32`.

The layers can be cut into pipeline stages, one device each
(`RicursiveEngine::from_source_staged`); `ricursive::placement` computes the
byte-exact cost of each layer at the chosen formats and the fewest cards that
hold the deployment.

Attention runs the fused paged flash kernels (no score slab, so no cap on the context a dense read
spans, at every KV tier) on a device with workgroup barriers, and the three-kernel reference on the CPU
backend.

The sliding layers keep a fixed ring of recent keys per sequence and the
global layers keep a growing paged cache, so the memory a sequence costs is a
fixed ring plus a per-token share of the global layers
(`ricursive::ring_bytes_per_slot`, `ricursive::global_kv_bytes_per_token`).
A cached prompt is reused at the length it was filed at.

## Using it as a library

```rust
let cfg = ricursive::RicursiveConfig::from_file("config.json")?;
let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
let weights = ricursive::open_safetensors("model.safetensors", &cfg, &layers)?;
let out = ricursive::RicursiveModel::new(cfg, &weights, None).forward(&token_ids)?;
// out.logits is [tokens, vocab]; out.layer_outputs holds every layer's output.
```

To serve it, build a `ricursive::RicursiveEngine` from the same source and a
`ricursive::ServeConfig` (block size, block count, sequence limit, live
sequences, cached prompts, prefill chunk) and hand it to
`model::serve::Scheduler::new`.

The importer refuses a checkpoint with a missing tensor, an unknown tensor, a
wrong shape or a non-float dtype, naming the tensor, before it reads any
weights. A config that sets a RoPE scaling, an attention or MLP bias, a
non-SiLU activation or tied embeddings is refused, because this port does not
implement them.

Package: `brain-ricursive`.
