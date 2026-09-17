# sample: vision/qwen3vl-caption (D-Bus)

`qwen3vl_caption.py` drives `brain/qwen3vl`'s streaming `generate` action:
an image plus a prompt in, generated text out, streamed token by token.

```bash
BRAIN_QWEN3VL_WEIGHTS=<Qwen3-VL-4B checkpoint dir-or-GGUF> \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/qwen3vl-caption/qwen3vl_caption.py --image photo.ppm --prompt "Describe this image."
```

## What it demonstrates

* **KV-cached decode**, unlike `moondream3-caption` in the sibling
  directory: `Qwen3Vl::generate_cb` carries real M-RoPE + DeepStack through
  the incremental decode path, so one masked prefill seeds the cache and
  each token after that is an `O(1)`-past-prefill step, not a full
  recompute. `generate` is still `.streaming()` and this example still
  uses `Subscribe`, but for the ordinary reason - seeing partial output as
  it's produced - not to hide a quadratic cost.
* **`--precision`.** `fp32` (the default) is the exact decoder tier;
  `int8` is explicitly LOSSY (~4x less weight traffic per token) and must
  be asked for by name - it never arrives by falling back. The scheduler
  budgets the two as separate resident instances.
* **`--max-pixels` sizes a capacity, not a per-request resize limit.** It
  sets the resident's DeepStack/splice buffer capacity at *build* time (a
  practical default around a 1024x1024 image); a request whose
  smart-resized image needs more visual tokens than that fails loudly
  rather than silently truncating. Raise it if your images are bigger -
  the server rebuilds the resident for the new capacity.

The first call pays activation (checkpoint load + upload); every later
call on the same server reuses the resident instance.

## What it needs

`BRAIN_QWEN3VL_WEIGHTS` pointing at a checkpoint directory or GGUF. Input
is a binary PPM (P6).

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* |
| `--prompt TEXT` | `Describe this image.` |
| `--max-new N` | `64` |
| `--precision` | `fp32` (exact); `int8` (lossy, ~4x less weight traffic) |
| `--max-pixels N` | resident's own default |
