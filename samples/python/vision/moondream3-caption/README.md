# sample: vision/moondream3-caption (D-Bus)

`moondream3_caption.py` drives `brain/moondream3`'s streaming `caption`
action: an image plus an instruction in, generated text out, streamed
token by token.

```bash
BRAIN_MOONDREAM3_WEIGHTS=<moondream3-preview checkpoint dir> \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/moondream3-caption/moondream3_caption.py --image photo.ppm --max-new 16
```

## What it demonstrates

* **The stream matters here more than for most models.** Moondream 3's
  decoder has no KV cache, so every generated token re-runs the whole
  sequence through all 24 layers over a 730-row image prefix - `Subscribe`
  delivers progress while the run is still going; waiting for the final
  `Outcome` means waiting for all of them.
* **`--precision`.** `int8` (the default) quantizes the 1280 expert
  tensors onto one shared activation set (~9 GiB); `fp32` is ~43 GiB of
  weights plus per-block scratch. The scheduler budgets them as two
  separate resident instances, so an `fp32` request on a machine without
  room fails placement cleanly instead of evicting a working `int8` one.

The first call pays activation (checkpoint load + expert quantization);
every later call on the same server reuses the resident instance.

## What it needs

`BRAIN_MOONDREAM3_WEIGHTS` pointing at the checkpoint directory. Input is a
binary PPM (P6).

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* |
| `--prompt TEXT` | `Describe this image.` |
| `--max-new N` | `16` (every token is a full recompute) |
| `--precision` | `int8` (~9 GiB); `fp32` (~43 GiB) |
