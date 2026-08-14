# Face detection (SCRFD)

Finds every face in an image - boxes, scores, 5-point landmarks. Part of the
well-known insightface antelopev2 stack; its sibling architecture is
[ArcFace](arcface.md), the identity embedding SCRFD's output aligns and feeds.
Both are implemented in one crate (`crates/facenet`) and served as one model,
`brain/facenet` - see that page for why, and for the identity-embedding half
of this pipeline.

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

Model id: `brain/facenet` - not auto-fetched. Set `BRAIN_FACENET_DIR` to a
directory holding both released ONNX graphs under their antelopev2 names:
`glintr100.onnx` (the [ArcFace](arcface.md) embedding model) and
`scrfd_10g_bnkps.onnx` (this detector). Both are read directly - no import or
conversion step.

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
