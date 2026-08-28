# llava - roadmap

LLaVA-1.5-13B: a CLIP-L/14@336 vision tower spliced into a Vicuna-1.5
(LLaMA-2 13B) decoder by a two-layer `mm_projector`. Brought into this
workspace as [`supir`](supir.md)'s optional captioner - see that ledger's
licence section for why `supir` itself carries no auto-fetch, which is
unrelated to this crate's own (Apache-licensed code, non-commercial
LLaMA-2/Vicuna weights) scope.

## The spec

- **Vision tower**: `openai/clip-vit-large-patch14-336` - 24 layers, 1024-d,
  16 heads, MLP 4096, patch 14, 336px input, 577 positions (576 patches + CLS).
  `mm_vision_select_layer = -2` (penultimate hidden state, not the final
  layer) and `select_feature = "patch"` (CLS dropped -> 576 tokens per image).
- **`mm_projector`**: `Linear(1024, 5120) -> GELU -> Linear(5120, 5120)`
  (`mlp2x_gelu`, exact-erf GELU, NOT quick-GELU), projecting each patch token
  into the decoder's embedding space.
- **Decoder**: Vicuna-1.5-13B, a LLaMA-2 13B fine-tune with no architecture
  changes - 40 layers, `d_model 5120`, 40 heads, MHA (`n_kv_heads == n_heads`,
  unlike Qwen3's GQA), `d_ff 13824`, `rope_theta 10000`, `rms_norm_eps 1e-5`,
  no QK-norm, no attention/MLP bias, untied embeddings, `max_position_embeddings
  4096`. Verified against the real `meta-llama/Llama-2-13b-hf` `config.json`
  (mirrored, ungated, at `NousResearch/Llama-2-13b-hf`) this session.
- **Conversation template**: `vicuna_v1` - transcribed from upstream
  `llava/conversation.py`'s `conv_vicuna_v1` (`SeparatorStyle.TWO`), not
  guessed: system message, `USER: <image>\n{question} ASSISTANT:` (the empty
  assistant turn has NO trailing space - `role + ":"`, not `role + ": "`).
- **Tokenizer**: LLaMA's SentencePiece BPE with byte-fallback (not GPT-2 BPE,
  not CLIP's BPE) - the `▁` word-start marker, byte-fallback for any
  character outside the trained vocab, **no pre-tokenizer at all** (the
  reference checkpoint registers none - the whole normalized string is one
  merge sequence, unlike GPT-2/CLIP's regex-split words).
- **The image-token splice**: transcribed from `llava/mm_utils.py`'s
  `tokenizer_image_token` - the prompt is split on the literal `<image>`
  substring BEFORE tokenizing (it is not a vocab token), each chunk is
  tokenized independently, the first chunk keeps its own BOS and every later
  chunk drops its own, and `IMAGE_TOKEN_INDEX = -200` is spliced between
  chunks as a sentinel later expanded into `n_visual` projected image-embed
  rows.
- **Default caption prompt** (SUPIR's usage): *"Describe this image and its
  style in a very detailed manner."*
- **Strictly optional in SUPIR's pipeline**: `--no_llava` (empty caption,
  replaced entirely by a user-supplied prompt) is a supported upstream path -
  LLaVA never touches the diffusion graph, it only ever emits a string.

## What is done

- [x] `data::llama_bpe::LlamaBpe` - gated at **exact id equality** against
  the real `tokenizers` library running `NousResearch/Llama-2-13b-hf`'s
  `tokenizer.json` (Vicuna-1.5's tokenizer, byte-for-byte), 21 strings
  covering SUPIR's own caption prompt, the `vicuna_v1` template shape, the
  literal `<image>` placeholder, whitespace runs, six non-Latin scripts and
  astral-plane emoji. `tools/goldens/llava_tokenizer_dump_reference.py`
  regenerates the pinned corpus; the fetched `tokenizer.json` stays out of
  git under `testdata/llava/tokenizer/` like every other real-checkpoint
  fixture, so the parity test skips cleanly (default `cargo test` stays
  green) when it is absent.
- [x] `clip::config::ClipVisionConfig::clip_l336()` + `penultimate_layer()` -
  a preset over the existing vanilla pre-LN ViT graph, byte-identical
  topology to the already-shipped `deepseek_ocr()` preset (only `image_size`/
  `n_positions` differ).
- [x] `qwen3::config::QwenConfig::llama2_13b()` - verified against the real
  checkpoint header (see above), not just the plan's stated numbers.
- [x] `crates/llava`: `config.rs` (`LlavaConfig`, `SelectFeature`,
  `vision_tap_layer`/`n_visual_tokens`), `model.rs` (the `mlp2x_gelu`
  `Projector` + `Llava` composite forward, weight-free end-to-end smoke test:
  vision tower -> projector -> decoder splice -> finite loss, actually run
  and green this session), `template.rs` (`vicuna_v1`, pinned against
  upstream's literal `get_prompt()` shape), `prompt.rs` (the `-200` splice,
  `tokenize_with_image_splice` + `splice_image_embeds`), `import.rs` (HF
  tensor name mapping for the decoder/projector/vision tower, including the
  q/k/v fusion CLIP's fused `attn.qkv.*` needs - weight-free "mapping-units"
  rung: every declared parameter of all three families is round-tripped by
  name against `param_list()`/`tensor_manifest()`), `caps.rs` (the `caption`
  `capability::Provider` action - resize+center-crop+CLIP-normalize
  preprocessing, the two-stage resident-lock split `fastvlm::caps` also
  uses, `fp32`/`int8` decoder precision), `captioner.rs` (`LlavaCaptioner`
  implementing `captioner::Captioner`, so `brain label --model llava` and any
  future `capability::Registry`-composing caller drive it identically to
  FastVLM/Qwen3-VL).
- [x] INT8 - `qwen3::Qwen::new_shard_i8` reused **unmodified** against
  `QwenConfig::llama2_13b()` (no llava-specific quantization code was
  written; the type is already generic over any `QwenConfig`). Per the
  workspace-wide measured finding, this reduces HOST RAM only - device-
  resident buffers are still fp32-sized at upload.
- [x] Serving/CLI wiring: `crates/cli`'s `resolve.rs` (`ARCH_TO_MODEL`),
  `resident.rs` (stateless resident registration), `label_cli.rs` (`--model
  llava`), `crates/catalog`'s `ModelEntry` - `brain llava caption`, `brain
  label images --model llava` and the D-Bus `Run` path all reach the
  provider. `docs/models/llava.md` updated from "not started" to this state.

## What is deferred, recorded rather than silently skipped

- **No real checkpoint was fetched or imported against this session.**
  LLaVA-1.5-13B is a ~26 GB fp16 / ~52 GB fp32 download (the tokenizer
  alone, ~1.8 MB, WAS fetched and is what gates `data::llama_bpe` above) -
  fetching the full weights was judged impractical within this session's
  time budget. Every piece that COULD be verified without weights was
  (tokenizer against the real tokenizer.json, both config presets against
  the real `config.json`, the template against upstream's literal source,
  the splice logic, the import name mappings' round-trip coverage, and an
  end-to-end weight-free forward smoke test). What remains unverified:
  `import.rs`'s mapping against REAL tensor names (only reconstructed
  plausible names, from the documented HF `CLIPVisionModel`/`LlamaForCausalLM`
  naming conventions, were exercised), and the single-forward/composed-loop
  parity rungs against real reference activations - both need the actual
  checkpoint bytes. A future session with checkpoint access should: fetch
  into `resources/llava/`, run `caps.rs`'s `load_vision`/`load_decode`
  against it, and dump a Python reference forward (CLIP tower + projector +
  a few decoder layers) to gate cosine/rel_l2 at the stage/single-forward
  rungs this session could not climb.
- Multi-turn conversation / visual question answering beyond a single
  caption call (SUPIR only ever needs one caption per image) - `template.rs`
  implements only `caption_prompt`, not the general `Conversation` state
  machine.
- 4-bit/8-bit bitsandbytes-style loading paths upstream offers
  (`--load_4bit`/`--load_8bit_llava`) - brain's own INT8 path supersedes them.
- `crates/supir`'s own pipeline does not yet exist (no `pipeline.rs`/
  `caps.rs` in that crate) to actually COMPOSE against `llava::caps`'s
  registered provider - that wiring is `supir`'s own later phase (its own
  `caption`/`restore` serving contract), not this crate's. What this phase
  delivers is the composable, registrable half of that seam: `LlavaProvider`
  (`capability::Provider`) and `LlavaCaptioner` (`captioner::Captioner`),
  either of which a future `supir::caps` can hold an `Arc<Registry>` /
  `Box<dyn Captioner>` to, with zero direct dependency on `brain-llava`.
- No HF-dumped golden for the `caption` action's own image preprocessing
  (resize-shortest-edge + center-crop + CLIP normalize) - the same honest gap
  `fastvlm::caps`'s own pad+resize carries; both are host-side data pipeline
  code with no learned weights to get wrong, but neither is proven bit-exact
  against `CLIPImageProcessor` yet.
