# sample: embedding/t5-embed

T5-XXL / umT5-XXL text encoding over brain's D-Bus interface (`brain
t5encoder encode`) - the text conditioning FLUX.1/2 need alongside CLIP-L,
and the text tower Wan2.1/2.2 condition on. `variant` selects the model
(`flux_xxl`, unmasked; `wan_umt5`, masked). Like CLIP's text towers, batching
is a genuine forward at `b = N`: `crates/cli/src/resident_t5encoder.rs`
groups concurrent calls by `(variant, max_len)`.

```bash
BRAIN_T5ENCODER_DIR=/path/to/FLUX.1-dev \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/embedding/t5-embed/t5_embed.py \
    --text "a red cube on a wooden table" --variant flux_xxl --concurrent 4
```

## What it demonstrates

* `encode` returns the last hidden state a diffusion DiT conditions on:
  FLUX.1/2 consume it unmasked (`variant=flux_xxl`); Wan2.1/2.2 condition on
  the masked, zero-padded version (`variant=wan_umt5`) - the served action
  always returns the masked-aware tensor even for the unmasked variant (it
  degrades to the same thing).
* `--concurrent` demonstrates the same real batched-forward grouping
  `samples/python/vision/segment-image/` shows for SAM 2.

## What it needs

`BRAIN_T5ENCODER_DIR` pointed at:

- `text_encoder_2/` + `tokenizer_2/tokenizer.json` (the FLUX.1-*/ release
  layout, unmodified) for `flux_xxl`, and/or
- `wan/models_t5_umt5-xxl-enc-bf16.pth` + `wan/tokenizer.json` for `wan_umt5`.

Plus `jeepney` - `pip install -e brain-py`.

## Options

| flag | default |
|---|---|
| `--text TEXT` | repeatable, one string to encode |
| `--variant {flux_xxl,wan_umt5}` | `flux_xxl` |
| `--max-len N` | `128` |
| `--concurrent N` | `0` - also submit N identical requests at once |
