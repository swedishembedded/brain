# Visual grounding (Florence-2)

Given an image and a short phrase - "the Login button", "a red car" - returns
a normalized bounding box for it. Native text-conditioned open-vocabulary
detection, not a prompt-engineered side effect of a chat decoder: the DaViT
vision tower and BART-style encoder-decoder were trained specifically for
this, at 0.23B params - small enough to run on hardware too small for a
general vision-language model. Built for the android-ui-test workflow's
screenshot -> bounding-box oracle, but the action has no UI-specific
assumptions in it.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI                    | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/florence2` - not auto-fetched. Set `BRAIN_FLORENCE2_DIR` to
a directory holding the released `microsoft/Florence-2-base` checkpoint:
`config.json`, `model.safetensors`, `tokenizer.json`. Read directly - no
separate import/conversion step; the few host-side transforms the vision
tower needs (a channel-attention scale fold, a synthesized position-embedding
table) happen automatically on load.

## Running it

```bash
brain caps florence2

brain florence2 ground --in image=screenshot.png --target "the Login button" --json
```

- **`ground`** - required `image` input and `target` param (the phrase or UI
  element to locate); optional `max_new_tokens` (default a generous ceiling,
  grounding answers are short); returns `{found, boxes: [{phrase, bbox}]}`,
  `bbox` normalized `[x0, y0, x1, y1]` in `[0, 1]`.

The same action is reachable over D-Bus via the generic
`Run(model, action, params, in_fds, in_meta, transport)` call.

## Options

| Param | Effect |
|---|---|
| `target` | the phrase or UI element to locate (required) |
| `max_new_tokens` | cap on generated tokens (`0` = default) |

## Hardware and limits

CPU-viable by design (0.23B params, ~463 MiB fp16 weights) - this is the
whole reason it was chosen over a larger general VLM for grounding. No KV
cache: each generation step recomputes the decoder's full prefix, which is
cheap at the short output lengths a grounding answer actually produces (a
handful of `<loc_N>` tokens plus a short phrase) but would not scale to long
free-form generation - this model's `ground` action is intentionally narrow,
not a general chat/captioning interface. No training support yet (see the
model's own roadmap for the planned LoRA scope: text-decoder-only, vision
tower frozen, matching how `deepseek2ocr` trains its own LoRA adapters).
