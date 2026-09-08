# deepseekocr2 - roadmap

DeepSeek-OCR-2: the successor to DeepSeek-OCR (`crates/deepseek2ocr`,
`.agents/roadmap/deepseek2ocr.md`), released 2026-01-27, Apache 2.0. Same
document-image-in, text/markdown-out task. The decoder is unchanged from v1;
the vision front end ("DeepEncoder V2" / "Visual Causal Flow") is new: SAM
ViT-B feeding a 24-layer Qwen2 encoder run as a learned-query resampler under
a prefix-LM attention mask, then a single linear projector - replacing v1's
SAM -> 16x conv compressor -> CLIP-L/14 -> concat arrangement.

Plan: see the session plan this ledger was opened from (12 milestones,
M0-M12). This file tracks facts and decisions as milestones land; it does not
duplicate the plan's implementation detail.

## Facts pinned from the real GGUF headers (not assumed)

Checkpoint used for fact-pinning: the community Q8_0 conversion pair (LM +
mmproj) built against llama.cpp PR #20975 (`deepseekocr2` mtmd support, not
yet merged upstream as of this writing). No official `deepseek-ai`-published
GGUF exists; the reference safetensors checkpoint is `deepseek-ai/DeepSeek-OCR-2`.

### LM (`deepseek2-ocr` architecture)

- **`general.architecture = "deepseek2-ocr"` - IDENTICAL STRING to v1's LM
  GGUF.** This is not a naming accident: the decoder is genuinely the same
  architecture family, byte-for-byte the same config
  (`deepseek2-ocr.{attention.head_count,head_count_kv}=10/10` - plain MHA,
  `block_count=12`, `embedding_length=1280`, `expert_count=64`,
  `expert_used_count=6`, `expert_shared_count=2`,
  `expert_feed_forward_length=896`, `feed_forward_length=6848`,
  `leading_dense_block_count=1`, `vocab_size=129280`,
  `rope.dimension_count=0` - same full-head_dim NEOX rope quirk v1 already
  documents). Tensor names and shapes match v1's decoder exactly: dense
  `blk.0.{attn_k,attn_norm,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_norm,ffn_up}`,
  MoE `blk.1..11.{..., ffn_down_exps[896,1280,64], ffn_down_shexp[1792,1280],
  ffn_gate_exps[1280,896,64], ffn_gate_inp[1280,64], ffn_gate_shexp[1280,1792],
  ffn_up_exps[1280,896,64], ffn_up_shexp[1280,1792]}`. 155 tensors total
  (`3 + 9 + 11*13`), confirming no hidden extra tensor. Same tokenizer
  (`tokenizer.ggml.pre = "deepseek-v3"`, 129280 tokens, same reserved-token
  scheme) and no `scoring_func`/`topk_method`/`norm_topk_prob` KV keys, same
  as v1 (llama.cpp's compiled-in defaults for this arch apply unchanged).
- **Consequence: `crates/gguf/src/deepseek_ocr.rs` reads this file
  unmodified.** v2's decoder needs no new importer, no new gradcheck, no new
  LoRA wiring - `crates/deepseek2` carries over as-is.
- **The collision is real but narrow: it is a `brain_arch::by_gguf` routing
  question, not a decoder-import question.** `crates/arch/src/lib.rs:778-779`
  asserts `by_gguf` is a one-to-one map (one arch string -> one registry id).
  Since v2's LM GGUF's `general.architecture` string is indistinguishable
  from v1's, the GENERIC single-file path (`brain import <file.gguf>`, and
  any `arch!()` row's `gguf:` field keyed on this string) cannot tell the two
  models' LM files apart from the string alone - it would resolve to
  whichever id claims `"deepseek2-ocr"` first. **Decision: `deepseekocr2`'s
  `arch!()` row leaves its LM-side `gguf:` discriminator unclaimed (only
  `deepseek2ocr` claims `"deepseek2-ocr"`); `deepseekocr2` is reached through
  its own composite import (`crates/deepseekocr2/src/import.rs`, reading
  `LM`/`MMPROJ` by role from a directory, exactly like v1's `import.rs` -
  never through `by_gguf`) and through its vision-side discriminator
  instead** (below). A generic `brain import` of a bare v2 LM GGUF with no
  mmproj alongside it is expected to resolve as v1's decoder, correctly - the
  two are the same tensors under the same name, and importing the LM alone
  discards nothing meaningful either model needs from it.

### Vision (`clip` architecture, `clip.projector_type=deepseekocr2`, mmproj file, 473 tensors)

- **`clip.projector_type = "deepseekocr2"` - the unambiguous discriminator**
  (distinct from v1's `"deepseekocr"`). This is what `deepseekocr2`'s
  `arch!()` row's vision-side `gguf:` field should key on, mirroring how v1's
  own vision importer already discriminates on `clip.projector_type` rather
  than `general.architecture="clip"` (shared by every mmproj in the repo).
- SAM tower: **byte-identical to v1** - `clip.vision.sam.{block_count=12,
  embedding_length=768, head_count=12}`, `clip.vision.window_size=14`. Real
  tensor names/shapes confirm the same per-block layout v1 already imports:
  `v.sam.blk.N.{pre_ln,post_ln}.{weight,bias}` (both are LayerNorm, weight
  AND bias present - not RMSNorm), fused `attn.qkv.{weight[768,2304],bias}`,
  decomposed relative position `attn.pos_h.weight[64,27]` /
  `attn.pos_w.weight[64,27]`, `mlp.lin1.weight[768,3072]` /
  `mlp.lin2.weight[3072,768]`. **`crates/sam1` is reused with zero change.**
- **`downsample_channels` in `config.json` (`[512,1024]`) is CONFIRMED STALE.**
  Real tensors: `v.sam.net_2.weight[3,3,256,512]` (256->512, stride 2),
  `v.sam.net_3.weight[3,3,512,896]` (512->**896**, stride 2). 896 is the
  Qwen2 hybrid encoder's hidden size, not 1024 - the neck's final width feeds
  the encoder directly, with no intermediate projection. A 1024x1024 view
  (64x64 SAM patches, `image_size=1024`, `patch_size=16`) downsamples 4x
  spatially to a 16x16=**256**-token grid at width 896; a 768x768 tile
  (48x48 patches) downsamples to 12x12=**144** tokens at width 896.
- **`clip.use_gelu = true` is an inert converter artifact**, same shape as
  v1's `feed_forward_length` bug: the Qwen2 hybrid encoder uses SiLU
  (SwiGLU, `ffn_gate`/`ffn_up`/`ffn_down` triples per block, confirmed by
  the real tensor names `v.blk.N.ffn_{gate,down,up}.weight`), and llama.cpp's
  mtmd graph builder for this projector type does not read this key. Do not
  propagate it into the config.
- **The new Qwen2 hybrid encoder** - real tensors confirm, per block:
  `ln1.weight`/`ln2.weight` (RMSNorm - weight ONLY, no bias, unlike SAM's
  LayerNorm blocks), `attn_q.{weight[896,896],bias[896]}`,
  `attn_k.{weight[896,128],bias[128]}`, `attn_v.{weight[896,128],bias[128]}`
  (128 = 2 KV heads x 64 head_dim - **GQA 14/2 confirmed**, matching
  `clip.vision.attention.{head_count=14,head_count_kv=2}`; **qkv bias
  present, no q/k norm tensors anywhere - matches `QwenConfig::qwen2`'s
  shape exactly, not `qwen3`'s**), `attn_out.weight[896,896]`,
  `ffn_gate/up.weight[896,4864]`, `ffn_down.weight[4864,896]`
  (`clip.vision.feed_forward_length=4864`, matching the reference's
  `Qwen2Config(intermediate_size=4864)`). `clip.vision.block_count=24`
  confirmed. RMSNorm eps `clip.vision.attention.layer_norm_epsilon ~= 1e-6`.
  No `rope_theta`/`max_position_embeddings` KV present - as with v1's
  `rope.dimension_count=0` quirk, llama.cpp's compiled-in Qwen2 default
  applies unread from this file. **Confirmed from
  `tools/mtmd/models/deepseekocr2.cpp`'s `clip_graph_deepseekocr2::build()`**
  (merged to llama.cpp master 2026-05-29, PR #20975):
  `ggml_rope_ext(ctx0, x, inp_pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX,
  131072, 1000000, 1, 0, 1, 0, 0)` - **theta 1e6, NEOX (half-split) layout**,
  matching the reference `Qwen2Config(rope_theta=1e6)` exactly. The mask
  itself is built as an externally-supplied `[seq_len, seq_len]` f32 input
  tensor (constructed host-side before the graph runs, not inside
  `build_vit()`), consistent with the earlier finding that the hybrid
  bidirectional-image/causal-query mask is prepared CPU-side. Also confirmed
  from the same source: image tokens are concatenated with the query bank
  via `ggml_concat(ctx0, inp, query_embed, 1)` - **image tokens first, query
  tokens appended after**, matching the reference's `cat([x, query], dim=1)`
  - and the view separator is appended conditionally
  (`img.add_viewsep`) via a second `ggml_concat`.
- **Learned query embeddings, real shapes**: `v.resample_query_768.weight
  [896,144]` (the 144-token/768-tile bank) and `v.resample_query_1024.weight
  [896,256]` (the 256-token/1024-global bank) - both present, both named
  after the reference's `query_768`/`query_1024`, GGUF's dim order reversed
  from the reference's `nn.Embedding(n_query, 896)`.
- **`v.view_seperator` is `[1280, 1]` - 1280-wide, the DECODER's hidden size,
  not the encoder's 896.** This settles an ordering question the plan left
  open: the per-view projector (`mm.model.fc`, `[896,1280]`) must run BEFORE
  the separator is appended, i.e. concatenation happens in 1280-dim
  projected space, not before projection. `mm.model.fc.{weight,bias}`
  confirmed as a single Linear(896->1280) - matches the reference's `linear`
  `MlpProjector` exactly, no MLP.
- **Not yet resolved from these headers** (needs the llama.cpp mtmd graph
  builder source, or a real forward, per the plan's M0/M5 gate):
  - the Qwen2 encoder's real RoPE theta and position-embedding convention
    (interleaved vs NEOX-half-split) as actually dispatched, not merely
    the reference config's stated default;
  - multi-tile ordering (local tiles vs the global view vs the separator) -
    the reference's own placeholder-token builder and embedding-scatter
    code disagree on order (see the HF-side research), and `config.json`'s
    `global_view_pos: "head"` needs to be read against the graph builder's
    actual concatenation, not the Python reference alone, since the mmproj
    graph is llama.cpp's own construction and is what any GGUF-based
    serving path actually runs.

## Status

M0 (resources + fact-pinning) done as of this entry. Remaining milestones
(M1-M12) not started.
