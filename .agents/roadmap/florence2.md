# florence2

**Status: M1 done. M2 done (patch embed, SpatialBlock, ChannelBlock, full
DaViT tower, and the vision-token projection wrapper all verified against
real weights end to end - see below). M3 done (BART encoder-decoder +
greedy generation, verified against real weights). M4 done. M5/M6/M7 not
started.**

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

- **M2 (DaViT vision encoder) - done**:
  - **Patch embed (verified)**: `vision/patch_embed.rs` - all 4 stages'
    conv+LayerNorm match the real checkpoint at cosine >=0.999
    (`davit_patch_embed_parity` test).
  - **SpatialBlock (verified)**: `vision/{dwconv,mlp,window_attn,block}.rs` -
    dwconv residual + window attention (`model::vit::WindowPlan::new` +
    `chunked_attn_fwd`, standard `1/sqrt(head_dim)` scale) + dwconv residual +
    MLP. Matches stage 0's real spatial-block output at cosine >=0.999
    (`davit_spatial_block_parity` test). `WindowPlan::new` (not `::padded`)
    is correct for this checkpoint specifically - asserted, not assumed:
    Florence-2-base's 4 stage grids (192/96/48/24) are all exact multiples
    of `window_size=12`.
  - **ChannelBlock (verified)**: `vision/channel_attn.rs` +
    `vision/block.rs`'s `ChannelBlock`. NOT a drop-in `chunked_bidir_fwd`
    reuse (correcting the original plan's assumption) - the reference's
    `ChannelAttention` computes `attention = (q*N^-0.5).transpose(-1,-2) @
    k` over `q,k: [B,groups,N,Cg]`, i.e. **channel-groups attend to each
    other, contracting over the spatial-token axis N**, the inverse of
    `chunked_bidir_fwd`'s assumption. Built instead from a per-group GEMM
    loop over the existing generic `matmul` primitive: one whole-buffer
    `nlc_nchw` transpose turns the fused `[N,3C]` qkv into `[3C,N]`
    (contiguous per-group row ranges via `step_sliced`), two GEMMs per
    group (`matmul`'s confirmed `A@B^T` convention maps directly onto both),
    `attn_softmax_cross` for the row-wise non-causal softmax. **Real bug
    caught by the parity gate**: `attn_softmax` looked like the right
    kernel (same math description) but its WGSL source hardcodes causal
    masking (`for j in 0..=i`, no toggle) - silently zeroed half of every
    `[Cg,Cg]` score matrix and cost a real cosine-0.987 failure before being
    caught and fixed by switching to `attn_softmax_cross` (genuinely
    non-causal). `1/sqrt(N)` scale folded into the qkv weight's Q-rows at
    `ParamStore`-construction time (N is a compile-time-known per-stage
    constant) - documented as the caller's obligation in
    `ChannelAttn::new`'s doc. Matches stage 0's real channel-block output at
    cosine >=0.999 (`davit_channel_block_parity` test).
  - **Full DaViT tower (verified)**: `vision/davit.rs`'s `Davit` chains all
    4 stages' patch embeds and all 12 `(SpatialBlock, ChannelBlock)` pairs
    into `forward_features_unpool`. Matches the real checkpoint's `unpooled`
    golden end to end at cosine >=0.999 (`davit_full_forward_parity` test) -
    the strongest single check, since no prior test chained every stage and
    block type together.
  - **Vision-token projection (verified)**: `vision/project.rs`'s
    `ImageProject` is `_encode_image`'s tail - everything after
    `forward_features_unpool`. Learned 2D position embed
    (`image_pos_embed.{row,column}_embeddings`, `[50,512]` each) added per
    token, cosine 1D temporal embed added as a per-channel bias (at `T=1`
    the reference indexes only row 0 of `visual_temporal_embed`'s
    `[100,1024]` table - already a contiguous 1024-float run at that
    tensor's own offset 0, used directly, no transform needed), spatial
    mean-pool via a constant `1/N` row-vector `matmul`, concatenated
    `[pooled(1); tokens(576)]` via two `row_scatter` calls (matches
    `image_feature_source=["spatial_avg_pool","temporal_avg_pool"]`'s real
    order), projected `1024->768` (bare `nn.Parameter` matmul, no bias) +
    LayerNorm. Two host-side pre-transforms the caller must do before
    `ParamStore` construction (`vision/project.rs`'s module doc, same
    "caller's obligation" convention as `ChannelAttn::new`'s qkv
    pre-scale): materialize the static `[24,24,1024]` position table
    (`build_pos_embed_table`, column/row broadcast-concat) and transpose
    `image_projection` from the checkpoint's `[in,out]` to `matmul_rows`'
    `[out,in]` convention (`transpose_2d`) - both kept crate-namespaced
    (`florence2::vision::project::*`), not re-exported at `vision`'s flat
    API, matching how `sam2::hostpe`'s equivalent host-math helpers are
    reached only via their own module, never flattened into that crate's
    root. Matches the real checkpoint's `projected` golden (the actual
    `[577,768]` input the BART encoder consumes) at cosine >=0.999
    (`davit_image_project_parity` test).

- **M3 (done)**: BART-style shared encoder-decoder + greedy generation,
  `crates/florence2/src/text/{config,attn,encoder,decoder,lm}.rs`. NOT built
  by extending `crates/toyseq2seq` (real gaps there: pre-LN not post-LN,
  untied not tied `lm_head`, fused not separate qkv, and critically no
  generation path at all) - built instead from ONE shared attention block
  (`text::attn::BartAttn`) covering all three of BART's attention flavors
  (encoder bidirectional self-attn, decoder causal self-attn, decoder
  cross-attn over the fixed encoder memory), composed entirely from the
  EXISTING generic cross-attention kernels
  (`attn_scores_cross`/`attn_softmax{,_cross}`/`attn_apply_cross`) rather
  than a fused-qkv builder: those kernels' `q_stride`/`kv_stride`/`*_off`
  params are already general enough to read three independent `[T,d_model]`
  buffers directly (`attn_scores_cross` only ever touches its `kv`
  argument's K region, `attn_apply_cross` only ever touches its `kv`
  argument's V region, so passing K and V as two DIFFERENT physical buffers
  to that one generic slot needs no fused layout at all) - `causal` swaps
  only the softmax kernel. **No KV cache** (a deliberate, tracked gap, not
  an oversight - see below).

  **Real gap found while generating M2's goldens, now resolved**: the
  checkpoint's language model reports `encoder.embed_tokens.weight`/
  `decoder.embed_tokens.weight`/`lm_head.weight` as MISSING on load (current
  `transformers` doesn't auto-map them) - the real tensor exists under a
  DIFFERENT name, `language_model.model.shared.weight [51289, 768]`, tied
  across encoder input, decoder input, AND the LM head (standard BART weight
  tying). `brain`'s own `text::lm::Florence2Lm` always reads `shared.weight`
  directly for all three uses, so this was never actually a gap for the
  Rust side - but it WAS a live bug in the golden-generation reference
  itself: without forcing the tie explicitly
  (`tools/goldens/florence2_dump_reference.py`'s `main()`, right after
  load), the Python reference would have silently validated `brain`'s
  correct implementation against untrained random embeddings instead of the
  checkpoint. Confirmed via a direct `torch.equal(shared.weight,
  encoder.embed_tokens.weight)` check before/after the fix (`False` then
  `True`) - caught before it ever produced a wrong golden, not after.

  **Second real bug the parity gate caught**: `Gpu::step`'s
  `assert_no_output_alias` (a real wgpu `STORAGE_READ_WRITE` exclusivity
  rule, not CPU-only pedantry) rejected an early draft of `text::encoder`/
  `text::decoder` that bound the same scratch buffer as both an `add2`/
  `layernorm_fwd` input AND its output (to save a buffer allocation per
  sublayer) - fixed by giving every residual-sum and post-LN step its own
  distinct buffer (`text::encoder`'s module doc has the full accounting),
  and using `add_inplace` (a genuinely single read_write binding plus one
  plain read-only operand) for the position-embedding add specifically,
  since that one case has a legitimate two-buffer-only shape.

  **Validation**: `crates/florence2/tests/text_lm_parity.rs` - encoder
  output and one decoder step's teacher-forced logits (fixed `PROMPT_IDS`/
  `DECODER_IDS`, not real tokenizer output - see the dump script's own doc)
  against `tools/goldens/florence2_dump_reference.py::dump_encdec`'s golden,
  both at cosine >=0.999. A third assertion cross-checks
  `Florence2Lm::generate`'s first greedy token against the golden's row-0
  argmax directly (not just cosine-close) - this specifically catches a
  causal-masking leak that a whole-tensor cosine could hide (a later
  position's context leaking into row 0 would barely move the AGGREGATE
  cosine while still corrupting that one row's argmax).

  **No KV cache - tracked as a real, intentional gap**: every
  `Florence2Lm::decode`/`generate` step recomputes the WHOLE decoder prefix
  (including re-projecting cross-attention K/V from the encoder memory,
  which never changes within one generation). Correct, and cheap enough for
  this crate's actual use case (`ground`'s outputs are a handful of
  `<loc_N>` tokens plus a short phrase - generation lengths in the tens, not
  hundreds), but a real optimization gap versus Kronos's KV-cached
  prefill/decode-step pattern - left for whoever profiles the `ground`
  capability action (M5) and finds it worth doing.

- **M4 (done)**: location-token sequence -> bbox parser
  (`grounding.rs::parse_boxes`, `phrase<loc_a><loc_b><loc_c><loc_d>` ->
  normalized `[x0,y0,x1,y1]` + phrase text) - covers both
  `<CAPTION_TO_PHRASE_GROUNDING>` and `<OD>`/`<DENSE_REGION_CAPTION>`'s
  output shape (same `box_pattern` in the reference's own post-processor).
  Hand-written scanner over decoded text, no regex dependency, no Gpu/model
  dependency - 7 unit tests, no real-checkpoint gate needed since it has no
  checkpoint dependency at all.

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
