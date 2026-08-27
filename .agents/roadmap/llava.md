# llava - roadmap

LLaVA-1.5-13B: a CLIP-L/14@336 vision tower spliced into a Vicuna-1.5
(LLaMA-2 13B) decoder by a two-layer `mm_projector`. Registered as an
architecture id; the port has not started. Brought into this workspace as
[`supir`](supir.md)'s optional captioner - see that ledger's licence section
for why `supir` itself carries no auto-fetch, which is unrelated to this
crate's own (Apache-licensed code, non-commercial LLaMA-2/Vicuna weights)
scope.

## The spec

- **Vision tower**: `openai/clip-vit-large-patch14-336` - 24 layers, 1024-d,
  16 heads, MLP 4096, patch 14, 336px input, 577 positions (576 patches + CLS).
  `mm_vision_select_layer = -2` (penultimate hidden state, not the final
  layer) and `select_feature = "patch"` (CLS dropped → 576 tokens per image).
- **`mm_projector`**: `Linear(1024, 5120) → GELU → Linear(5120, 5120)`,
  projecting each patch token into the decoder's embedding space.
- **Decoder**: Vicuna-1.5-13B, a LLaMA-2 13B fine-tune with no architecture
  changes - 40 layers, `d_model 5120`, 40 heads, MHA (`n_kv_heads == n_heads`,
  unlike Qwen3's GQA), `d_ff 13824`, `rope_theta 10000`, RMSNorm (no bias),
  no QK-norm, no attention/MLP bias, untied embeddings.
- **Conversation template**: `vicuna_v1` - image tokens spliced into the
  prompt at a fixed position, then the standard Vicuna system/user/assistant
  turn format.
- **Tokenizer**: LLaMA's SentencePiece BPE with byte-fallback (not GPT-2 BPE,
  not CLIP's BPE) - the `▁` word-start marker, no merge-table surprises
  otherwise.
- **Default caption prompt** (SUPIR's usage): *"Describe this image and its
  style in a very detailed manner."*, `temperature 0.2`, `top_p 0.7`,
  `num_beams 1`, `max_new_tokens 512`.
- **Strictly optional in SUPIR's pipeline**: `--no_llava` (empty caption,
  replaced entirely by a user-supplied prompt) is a supported upstream path -
  LLaVA never touches the diffusion graph, it only ever emits a string.

## What is and is not started

- [ ] Everything. The architecture id (`crates/arch`) and this placeholder
      crate are the only things that exist.

## Staged plan

Ordered after SUPIR's own forward is parity-proven (see `.agents/roadmap/supir.md`
step 7), since nothing in SUPIR depends on this crate existing yet:

1. `data::llama_bpe` - a sibling of `data::clip_bpe`, reusing `data::bpe`'s
   merge loop and adding only the SentencePiece `▁` word-start marker and
   LLaMA's byte-fallback. Gate at **exact id equality** vs HF
   `LlamaTokenizerFast`, the way `clip_bpe` is gated against `CLIPTokenizer`.
2. `clip::ClipVisionConfig::clip_l336()` - a preset over the existing
   ordinary pre-LN ViT graph (`model::vit`), not a new graph: 24×1024,
   16 heads, MLP 4096, patch 14, `image_size 336`, 577 positions, quick-GELU.
3. `qwen3::QwenConfig::llama2_13b()` - 40 layers, `d_model 5120`, 40 heads,
   `n_kv_heads 40` (plain MHA), `d_ff 13824`, `rope_theta 10000`,
   `qk_norm false`, `attn_bias false`, untied embeddings. Verify every field
   against the real checkpoint header before treating it as load-bearing -
   this repo's `QwenConfig` already has every knob LLaMA-2 needs (it is what
   `qwen2()` sets `attn_bias: true` / `qk_norm: false` for), this is a preset,
   not new capability.
4. `crates/llava` itself: the `mm_projector`, the token splice (image
   embeddings replacing a placeholder token in the prompt sequence), the
   `vicuna_v1` template. Two-way import, parity-gated per stage against a
   hooked reference forward.
5. INT8 from the start - fp32 13B is ~52 GB, well past what this machine's
   30 GB of shared RAM can hold resident. `qwen3::q8` is the template
   (group-wise, `QUANT_GROUP = 32`, never whole-channel).
6. Serving contract + a `caption` action; wired into `supir`'s pipeline
   through `capability::Registry` (the `imgpipe` composition pattern) so
   `supir` itself links no VLM crate directly.

## Deferred, recorded rather than silently skipped

- Multi-turn conversation / visual question answering beyond a single
  caption call (SUPIR only ever needs one caption per image).
- 4-bit/8-bit bitsandbytes-style loading paths upstream offers
  (`--load_4bit`/`--load_8bit_llava`) - brain's own INT8 path supersedes them.
