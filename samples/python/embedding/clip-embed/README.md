# sample: embedding/clip-embed

CLIP text and image embeddings over brain's D-Bus interface (`brain clip
embed_text`/`embed_image`). `embed_text` batches every `--text` string given
into ONE forward over the resident text tower
(`clip::caps::Session::embed_text_batch`) - the point of `--text` being
repeatable is to show that batching, not just to embed several strings for
convenience. `embed_image` runs the EVA-CLIP-L/336 vision tower on one image
at a time.

```bash
BRAIN_CLIP_DIR=/path/to/stable-diffusion-xl-base-1.0 \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/embedding/clip-embed/clip_embed.py --text "a photo of a cat" \
    --text "a photo of a dog" --image photo.ppm
```

## What it demonstrates

* Both SDXL text towers are available (`--tower clip_l` / `openclip_bigg`);
  the action returns the projected `text_embeds` when the tower projects and
  the pooled EOS row when it does not, so a caller doesn't have to know
  which is which.
* Unlike the face stack, CLIP's `run_batch` is a **genuine batched forward**:
  the residency adapter groups a batch by tower and runs one forward per
  group at `b = N`, because every row is the same fixed 77-token context
  (`crates/cli/src/resident_clip.rs`).
* Verified end to end against HuggingFace on the released checkpoint -
  string in, BPE, tower, pooling - at **cosine 1.0000000000 / max_abs 1.5e-5**
  (CLIP-L) and **0.9999998212 / 1.0e-5** (OpenCLIP-bigG);
  `crates/clip/tests/serving.rs` is the standing gate.

## What it needs

- `BRAIN_CLIP_DIR` pointed at a checkpoint root holding `text_encoder/`
  and/or `text_encoder_2/` (SDXL layout), `tokenizer/` and/or `tokenizer_2/`,
  and `EVA02_CLIP_L_336_psz14_s6B.pt` at the root (for `--image`).
- `jeepney` - `pip install -e brain-py`.

## Options

| flag | default |
|---|---|
| `--text TEXT` | repeatable, batched into one forward |
| `--tower {clip_l,openclip_bigg}` | `clip_l` |
| `--image PATH` | unset - binary PPM (P6) to embed with EVA-CLIP-L/336 |
