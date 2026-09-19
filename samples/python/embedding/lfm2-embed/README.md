# sample: embedding/lfm2-embed

LFM2.5-Encoder long-context embeddings and fill-mask over brain's D-Bus
interface (`brain do lfm2 embed` / `brain do lfm2 fill_mask`). Both run the
same chunked long-context path, so an 8k-token document works the same way
as a short sentence.

```bash
BRAIN_LFM2=<ckpt>.safetensors BRAIN_LFM2_TOKENIZER=<tokenizer>.json \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/embedding/lfm2-embed/lfm2_embed.py \
    --text "the quick brown fox jumps over the lazy dog"

python3 samples/python/embedding/lfm2-embed/lfm2_embed.py \
  --action fill_mask --text "the quick brown <|mask|> jumps over the lazy dog"
```

## What it demonstrates

* `embed` returns per-token hidden states (an `[n_tokens, dim]` LE-f32 blob)
  plus a mean-pooled sequence embedding - the two things a downstream
  retrieval/classification head needs.
* `fill_mask` returns top-k predictions at every `<|mask|>` position in the
  input.
* Unlike `t5_embed.py --concurrent`, this sample does not demonstrate
  client-side batching: bidirectional attention makes unmasked padding
  unsound, so the resident model is rebuilt at the exact request length
  rather than grouped into one padded forward (see `crates/lfm2/src/caps.rs`).

## What it needs

`BRAIN_LFM2` (a brain-format LFM2.5-Encoder `.safetensors` checkpoint) and
`BRAIN_LFM2_TOKENIZER` (its `tokenizer.json`).

Plus `jeepney` - `pip install -e brain-py`.

## Options

| flag | default |
|---|---|
| `--action {embed,fill_mask}` | `embed` |
| `--text TEXT` | a built-in example for the selected action |
| `--max-tokens N` | `0` - embed: truncate to N tokens (0 = no limit) |
| `--topk N` | `5` - fill_mask: predictions per mask position |
