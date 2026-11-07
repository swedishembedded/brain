# Ripple-65B (forward parity only, not served)

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
| Serving (KV cache, batching) | [ ] |
| Quantized weights | [ ] |
| CLI | [ ] |
| HTTP API | [ ] |
| D-Bus | [ ] |
| Training | [ ] |

**Read that table before relying on this.** The crate computes a prefill over a
prompt and returns logits; it keeps no KV cache, so it cannot generate, and it
holds one layer's weights in f32 at a time. The registry row is the model's
name and its Hugging Face class; it does not make the model servable.

## Using it as a library

```rust
let cfg = ricursive::RicursiveConfig::from_file("config.json")?;
let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
let weights = ricursive::open_safetensors("model.safetensors", &cfg, &layers)?;
let out = ricursive::RicursiveModel::new(cfg, &weights, None).forward(&token_ids)?;
// out.logits is [tokens, vocab]; out.layer_outputs holds every layer's output.
```

The importer refuses a checkpoint with a missing tensor, an unknown tensor, a
wrong shape or a non-float dtype, naming the tensor, before it reads any
weights. A config that sets a RoPE scaling, an attention or MLP bias, a
non-SiLU activation or tied embeddings is refused, because this port does not
implement them.

Package: `brain-ricursive`.
