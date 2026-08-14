# Face detection (SCRFD)

Finds every face in an image - boxes, scores, 5-point landmarks. Part of the
well-known insightface antelopev2 stack; its sibling architecture is
[ArcFace](arcface.md), the identity embedding this detector's landmarks align
and feed. The two are independently served models with their own weights and
their own weights variable, and this one stands alone: detection needs nothing
from the embedder. (The reverse is not true - `brain/arcface`'s default path
detects with this model first.)

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

Model id: `brain/scrfd` - not auto-fetched. Set `BRAIN_SCRFD_DIR` to a
directory holding the released ONNX graph under its antelopev2 name,
`scrfd_10g_bnkps.onnx`. It is read directly - no import or conversion step.
(The antelopev2 release ships `glintr100.onnx`, the
[ArcFace](arcface.md) embedder, in the same directory; pointing both
`BRAIN_SCRFD_DIR` and `BRAIN_ARCFACE_DIR` at it serves both models.)

## Running it

```bash
brain caps scrfd

brain scrfd detect --in image=photo.ppm --json
```

- **`detect`** - required `image` input; param `max_faces` (default `0` =
  all); returns boxes/scores/5-point landmarks in source-image pixels.

The same action is reachable over D-Bus via the generic
`Run(model, action, params, in_fds, in_meta, transport)` call. Reference
client: `examples/vision/face_id.py` (see `examples/vision/README.md`).

## Options

| Param | Effect |
|---|---|
| `max_faces` | cap on faces returned (default `0` = all) |

## Hardware and limits

The released ONNX graph is pinned to a batch size of 1, so requests are
served one at a time rather than batched. There's no training or fine-tune
verb, and no HTTP surface - reach this model through the CLI or D-Bus.
