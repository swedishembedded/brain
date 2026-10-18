# minicpmv - roadmap

MiniCPM-V 4.6 (`openbmb/MiniCPM-V-4.6`, Apache-2.0, ~1.3B parameters): a
NaViT-style SigLIP vision tower with an in-tower token merger, a two-stage
token compressor, and a Qwen3.5-0.8B hybrid Gated-DeltaNet/GQA decoder.
Images, multi-image prompts and video.

## The spec (from the reference code and the checkpoint, not secondhand)

Authority: `transformers.models.minicpmv4_6` (`modeling_`, `processing_`,
`image_processing_`, `video_processing_minicpmv4_6.py`, transformers 5.20.1).
Third opinion: llama.cpp's `PROJECTOR_TYPE_MINICPMV4_6` converter. The
checkpoint (779 BF16 tensors) is the authority on names and shapes.

- **Vision tower** (`model.vision_tower.*`): SigLIP-shape, 27 layers, width
  1152, 16 heads (head dim 72), MLP 4304, `gelu_pytorch_tanh`, LayerNorm eps
  1e-6, biased q/k/v/out, patch 14 (biased conv), `post_layernorm`.
  - **Variable resolution (NaViT)**: each view is its own `(h, w)` patch grid;
    position embeddings are looked up from a 70x70 learned table (`[4900,
    1152]`) at **nearest** ids (`get_vision_nearest_position_ids`), not
    interpolated.
  - **Packing**: every view of every image is one sequence; attention is
    block-diagonal per view (`cu_seqlens`).
  - **In-tower merger** (`vit_merger`, `insert_layer_id = 6`): after encoder
    layer 6 (0-based), 2x2-window attention over the window-major
    reordering, then per window: concat 4 tokens -> LayerNorm(4608) ->
    Linear(4608, 17216) -> gelu_tanh -> Linear(17216, 1152) + mean of the 4
    tokens. Token count / 4; layers 7..26 run on the halved grid. This is a
    VISION-INTERNAL step: nothing is injected into the language model's
    middle layers.
  - Skipped entirely in `4x` mode.
- **Merger** (`model.merger.mlp.0`, `merger_times = 1`): per view, 2x2
  concat -> LayerNorm(4608) -> Linear(4608, 4608) -> GELU(erf) ->
  Linear(4608, 1024). Token count / 4.
- **Downsample mode**: `16x` (default) = in-tower merger + merger; `4x` =
  merger only. Same weights, chosen per request. Placeholder count per view
  = `h*w/16` or `h*w/4` (patch grid `h x w`).
- **Language model** (`model.language_model.*`): Qwen3.5 text, 24 layers
  (3 GDN : 1 GQA), width 1024, FFN 3584, GQA 8/2 heads of 256 with output
  gate and QK-norm, partial rotary 0.25, `rope_theta` 1e7, GDN 16/16 heads of
  128, conv 4, vocab 248094, **tied** embeddings, no MTP tensors.
  Visual features replace placeholder rows of the input embeddings
  (`masked_scatter`), and positions are plain 1-D (`position_ids=None`) -
  no M-RoPE grid for images.
- **Image processor**: torchvision backend, bicubic with antialias, mean/std
  0.5, `scale_resolution` 448, `max_slice_nums` 9, `slice_mode`,
  `use_image_id`. Source view resized by `find_best_resize` (multiples of
  56 px), plus, when `get_sliced_grid` picks a grid, a refine view cut into
  `rows x cols` slices.
- **Prompt layout**: `<image_id>N</image_id>` (optional) + `<image>` +
  source tokens + `</image>` + per row `<slice>` tokens `</slice>` repeated
  `cols` times, rows joined by `\n`.
- **Video**: frames chosen by INDEX `int(t * avg_fps)` from the container's
  `duration`/`fps`/`frame count` (1 fps; 10 Hz + `linspace` thinning to
  `max_num_frames` when the clip is longer than `max_num_frames` seconds).
  Each frame is processed like an image (with slices) and laid out as an
  image placeholder with `<|video_pad|>`. No timestamp tokens.
- **Reference defect**: `stack_frames > 1` builds per-second composites and
  then discards them, so placeholders and features disagree and the
  reference forward raises (`tokens: 756, features: 1134` on the golden
  clip, recorded in the golden manifest). brain refuses `stack_frames > 1`
  until a working reference exists.

## Goldens

`tools/goldens/minicpmv_dump_reference.py` (CPU, fp32) writes
`testdata/minicpmv/`: processor cases (ids, grids, resized views, pixels),
`sample_frames` index lists, model cases (text, image 16x/4x, document,
two images, video) with vision/LM taps, last-row logits, greedy
continuations and one label loss.

## Plan and status

- [x] P0 reference capture: dumper, goldens.
- [x] qwen35 refuses configs missing GDN/FFN dims (they vary per size).
- [x] P1 text-only decoder through `qwen35` from the HF directory, logits
      parity, incremental decode == full prefill. Measured: last-8-row logits
      cosine 1.0000000000, rel_l2 2.9e-6, greedy continuation identical
      (`tests/text_parity.rs`).
- [x] P2 multi-run embedding splice shared in `model::vlm::SpliceRuns`
      (qwen3, qwen35, qwen35moe, deepseek2), FD-gradchecked on three disjoint
      runs (`gradcheck::check_qwen35_splices`); inference prefill from input
      embeddings (`Qwen35::prefill_embeds`) equal to the token prefill and to
      the spliced training forward (`qwen35/tests/prefill_embeds.rs`).
- [x] P3 vision tower forward (`vision.rs`): views packed with one span
      each through `model::vit` blocks, nearest position ids, the window
      merger after `insert_layer_id` on the uniform `attn_*_bidir` family
      (T = 4), the merger; no new kernel. Real checkpoint, reference pixels
      (`tests/vision_parity.rs`): every stage at cosine >= 0.9999999999,
      rel_l2 <= 1.3e-5, at 16x, at 4x and for two packed images.
- [ ] P3b vision backward (cached forward + `vit_block_bwd`, window merger
      and merger adjoints) and its FD gradcheck - lands with P7, the first
      consumer.
- [x] P4 preprocessing + prompt layout (`preprocess.rs`, `prompt.rs`): the
      slicing policy with Python's rounding, torch's uint8 antialiased bicubic
      (`imaging::host::resize_bicubic_torch_u8`, not Pillow's - they differ by
      one on ~0.4% of pixels), the fused `(x - 127.5) / 127.5`, the
      checkpoint's own chat template through `data::chat_template`, the
      placeholder layout. Exact against all 14 processor cases and the 5
      model-case prompts (`tests/processor_parity.rs`): resized views byte
      for byte, grids, patches, rendered text, token ids.
- [ ] P5 end-to-end generate + serving contract (caps, resident, catalog,
      D-Bus, HTTP, sample, docs).
- [ ] P6 video: index-exact frame selection, frame layout, parity.
- [ ] P7 training: decoder LoRA, merger, vision; overfit/held-out/reload
      gates; PEFT export.
