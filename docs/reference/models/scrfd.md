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

Model id: `brain/scrfd` - not auto-fetched. Put the released ONNX graph, under
its antelopev2 name `scrfd_10g_bnkps.onnx`, anywhere in the models directory
(`--models-dir` / `BRAIN_MODELS_DIR`); brain finds it by scanning and reads it
directly, with no import or conversion step and no variable to export.

The antelopev2 release ships `glintr100.onnx`, the [ArcFace](arcface.md)
embedder, in the same directory - so one copy of that release serves both
models. `BRAIN_SCRFD_DIR` still works as an explicit pin when you want one
exact directory used.

## Running it

```bash
brain caps scrfd

brain scrfd detect --in image=photo.ppm --json
```

- **`detect`** - required `image` input; param `max_faces` (default `0` =
  all); returns boxes/scores/5-point landmarks in source-image pixels.

The same action is reachable over D-Bus via the generic
`Run(model, action, params, in_fds, in_meta, transport)` call. Reference
client: `samples/python/vision/face-id/face_id.py` (see that sample's README.md).

## Options

| Param | Effect |
|---|---|
| `max_faces` | cap on faces returned (default `0` = all) |

## Hardware and limits

The released ONNX graph is pinned to a batch size of 1, so requests are
served one at a time rather than batched. There's no training or fine-tune
verb, and no HTTP surface - reach this model through the CLI or D-Bus.
