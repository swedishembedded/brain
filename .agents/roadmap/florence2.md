# florence2

**Status: M1 (resources + tokenizer) done. M2-M7 not started.**

## Goal

Florence-2-base as brain's first UI-element visual-grounding oracle (image +
text query -> bounding box), chosen for the swedishembedded/whale
android-ui-test initiative (see whale/sven repos' own roadmap docs). Picked
over Qwen3-VL/moondream3 specifically for footprint: 0.23B params, 463 MiB
fp16, CPU-viable on hardware too small for a 4B+ VLM. License: MIT
(`microsoft/Florence-2-base`'s own `LICENSE`) - both weights and the
reference `modeling_florence2.py`/`processing_florence2.py`/
`configuration_florence2.py` (`trust_remote_code`, never merged into
`transformers`). That reference code is used as a validation oracle
(numerical parity against it) and as the source of the token vocabulary
data below - never transliterated as Rust/WGSL implementation code, per
this repo's standing practice on every other ported model.

## Confirmed architecture (from the real `config.json`, not the paper)

```
vision (DaViT):  dim_embed=[128,256,512,1024], num_heads=[4,8,16,32],
                 num_groups=[4,8,16,32] (channel-attention groups, separate
                 from spatial num_heads), depths=[1,1,9,1], window_size=12,
                 patch_size/stride/padding per stage=[7,3,3,3]/[4,2,2,2]/[3,1,1,1]
text (BART):     d_model=768, encoder_layers=6, decoder_layers=6, heads=12,
                 ffn_dim=3072, vocab_size=51289, max_position_embeddings=1024,
                 normalize_before=false (post-LN), add_final_layer_norm=false
projection_dim:  768; image input 768x768 -> image_seq_length=577
```

**Vision-token composition (read directly from `_encode_image`/
`Florence2VisionModelWithProjection.forward`, not inferred)**: DaViT's
`forward_features_unpool` gives 576 tokens (24x24 grid, dim 1024) for a
768x768 image. A **learned absolute 2D position embedding** (`image_pos_embed`,
`learned_abs_2d`, up to 50x50) is added, then a **cosine 1D temporal
embedding** (`visual_temporal_embed`, harmless-but-required at T=1 for a
still image - it's still part of the trained weights). Two pooling views are
concatenated: `spatial_avg_pool` (576 tokens -> 1 global token, mean over the
576) then `temporal_avg_pool` (identity at T=1, keeps all 576) ->
**1 + 576 = 577 tokens**, matching `preprocessor_config.json`'s declared
`image_seq_length: 577` exactly. Projected `1024 -> 768` via a bare
`nn.Parameter` matmul (no bias) + LayerNorm, then concatenated with the
text task-prefix embeddings before the shared BART encoder runs
(`_merge_input_ids_with_image_features`: `cat([image_features, text_embeds])`).

**Tokenizer - the one real surprise vs. the original plan**: the checkpoint's
`tokenizer.json`/`vocab.json` carry ONLY the base 50265-id RoBERTa/BART BPE
vocab (verified: `added_tokens` there has just the 5 standard specials). The
1024 extra ids (`vocab_size: 51289` in config) exist ONLY as a literal Python
list in `processing_florence2.py`'s `Florence2Processor.__init__` (lines
87-91): 4 task markers (`<od>`,`</od>`,`<ocr>`,`</ocr>`), then 1000 location
bins (`<loc_0>`..`<loc_999>`, pixel coords normalized by width/height *
1000), then 20 more task-boundary markers (`<cap>`, `<grounding>`, `<poly>`,
etc.) - added programmatically at load time, sequential ids past the base
vocab. `crates/florence2/src/tokenizer.rs::additional_tokens()` reproduces
this exact list/order (necessarily - it's the model's actual trained
vocabulary, not a design choice); loading uses a new, small, generically
useful addition to the shared tokenizer instead of a Florence-2-specific
parser: `data::qwen_tokenizer::QwenBpe::add_special_tokens` (appends new
specials past the current vocab size, mirrors HF's own
`add_special_tokens` semantics). Verified byte-for-byte against a real
`transformers.AutoTokenizer` run on the downloaded checkpoint (pinned
reference vectors in `tokenizer.rs`'s test module) - not just internally
self-consistent.

## Milestones

- **M1 (done)**: resources downloaded (outside this repo, gitignored-by-
  location under the workspace's shared resources mount), `crates/florence2`
  skeleton, tokenizer (`tokenizer.rs`) loading + extending to the real
  51289-id vocab, gated on `FLORENCE2_DIR` (skips cleanly when unset).
  `cargo check --workspace` clean, clippy clean on touched crates.
- **M2**: DaViT vision encoder. Reuses `WindowPlan` (windowed attention -
  already anticipates "DaViT's local window stage" in its own doc),
  `nchw_nlc`/`nlc_nchw` transpose kernels + `chunked_bidir_fwd` for channel
  attention (one real gap: `attn_scores_cross.wgsl`'s hardcoded softmax
  scale needs a new `CrossIds` variant wired to the already-existing
  scale-configurable `attn_scores_qk.wgsl`), FastVLM's `PatchEmbed`
  (overlapping inter-stage downsample) and `repcpe` (depthwise-conv
  positional encoding) as direct precedent, SAM2's `config.rs` stage-table
  shape as the schedule-struct template.
- **M3**: BART-style shared encoder-decoder + generation. NOT built by
  extending `crates/toyseq2seq` (real gaps there: pre-LN not post-LN,
  untied not tied `lm_head`, fused not separate qkv, and critically no
  generation path/KV-cache/padding-mask at all) - built from
  `model::block::CrossIds`/`KeyMinor` (the same cross-attention-over-fixed-
  encoder-output builder every diffusion DiT here already uses) + Kronos's
  KV-cached prefill/decode-step pattern + `model::vlm::splice_fwd` (the
  vision+text concat, same mechanism DeepSeek-OCR's `layout.rs` already
  uses) for the image-token/text-token merge.
- **M4**: location-token sequence -> bbox parser (`<loc_a><loc_b><loc_c><loc_d>`
  -> normalized `[x0,y0,x1,y1]`). New, small, no existing kernel involved -
  decide `<CAPTION_TO_PHRASE_GROUNDING>` vs `<OPEN_VOCABULARY_DETECTION>`
  from real-weight output quality on real UI screenshots, not from the
  paper alone.
- **M5**: `florence2::caps::ground` capability action (`scrfd::caps`'s
  `detect` shape: `Outcome` JSON, no output blob), CLI (`brain florence2
  {import,infer,ground}`), residency + cost-aware scheduler wiring.
  `general.architecture` id: `"florence2"` (the HF config's own
  `model_type` - no GGUF/llama.cpp conversion exists upstream to take an id
  from, confirmed via an open unresolved llama.cpp issue).
- **M6**: LoRA + full fine-tune, single/batch overfit-to-zero gradcheck -
  lower priority than M1-M5 for the grounding-only use case, required by
  this repo's blanket per-model policy.
- **M7**: docs (`docs/models/florence2.md`, README entry, excluded from
  quickstart).

## Open gaps to track (not blocking, recorded so they aren't lost)

- GPU (wgpu) validation: implemented alongside CPU (same kernels/builders),
  but this dev box has no GPU at all - real-weight GPU parity is an open
  gap until validated on hardware that has one.
- NPU: deferred unless it's a straightforward reuse of existing NPU
  plumbing (see `yolov8`/`zipdepth`'s Intel-NPU paths) - otherwise
  explicitly out of scope for now, not silently dropped.
- `data::qwen_tokenizer::encode_with_specials` scans every special token
  at every scan position (`O(text_len × num_specials)`); harmless at ~1000
  specials for interactive/CI use, a prefix index is a real follow-up if
  profiling ever shows it matters for a hot serving path.
