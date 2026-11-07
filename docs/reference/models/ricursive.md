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
| Quantized weights | [ ] |
| Training | [ ] |

**Read that table before relying on this.** `RicursiveEngine` implements
`model::serve::PagedDecoder`, so a program can run it under
`model::serve::Scheduler` with continuous batching, chunked prefill and prefix
reuse, and it passes the shared serving conformance suite. It is not in the
model catalog: `brain serve` and the capability interface do not know it yet.
Weights are resident f32, so it needs a machine that holds them; there is no
quantization yet. Its global layers can read through latent sparse attention
(`RicursiveEngine::set_lsa`, an indexer covering only global layers), which
needs a device that runs workgroup-barrier kernels. The registry row is the model's name
and its Hugging Face class.

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
