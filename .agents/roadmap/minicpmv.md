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
- [ ] P1 text-only decoder through `qwen35` from the HF directory, logits
      parity, incremental decode == full prefill.
- [ ] P2 multi-run embedding splice shared in `model::vlm` (qwen3 + qwen35),
      in training forward and inference prefill; FD-gradchecked.
- [ ] P3 vision tower: ragged per-view spans, nearest position ids,
      in-tower merger hook, merger; per-stage parity (16x and 4x);
      gradchecked.
- [ ] P4 preprocessing + prompt layout: exact ids/grids, resized views.
- [ ] P5 end-to-end generate + serving contract (caps, resident, catalog,
      D-Bus, HTTP, sample, docs).
- [ ] P6 video: index-exact frame selection, frame layout, parity.
- [ ] P7 training: decoder LoRA, merger, vision; overfit/held-out/reload
      gates; PEFT export.
