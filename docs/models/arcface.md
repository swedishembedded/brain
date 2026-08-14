# Face identity embedding (ArcFace)

Turns an aligned face crop into a 512-d embedding you can use for identity
matching and search - "are these two photos the same person?" or "find this
face among a set of known ones." Part of the well-known insightface
antelopev2 stack (IResNet-100 backbone); its sibling architecture is
[SCRFD](scrfd.md), the detector that finds and aligns the face this model
embeds. Both are implemented in one crate (`crates/facenet`) and served as
one model, `brain/facenet`.

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
`glintr100.onnx` (this embedding model) and `scrfd_10g_bnkps.onnx` (the
[SCRFD](scrfd.md) detector). Both are read directly - no import or
conversion step.

## Running it

```bash
brain caps arcface

brain arcface embed --align false \
    --in image=aligned112.ppm --out embedding=id.bin
```

- **`embed`** - required `image` input; params `align` (bool, default
  `true`: detect + similarity-align the primary face via SCRFD first;
  `false` = the input is already an aligned 112x112 face) and `select`
  (`largest`|`score`, default `largest`); returns a 512-d, L2-normalized
  embedding as blob `embedding`.

The same action is reachable over D-Bus via the generic
`Run(model, action, params, in_fds, in_meta, transport)` call. Reference
client: `examples/vision/face_id.py` (see `examples/vision/README.md`).

## Options

| Param | Effect |
|---|---|
| `align` | detect + align the primary face first (default `true`), or `false` if the input is already an aligned 112x112 crop |
| `select` | which face to embed when several are found: `largest` (default) or `score` |

To compare two identities, embed each face and compare the two 512-d
vectors (e.g. cosine similarity) yourself - that comparison isn't a
server-side action.

## Hardware and limits

The released ONNX graph is pinned to a batch size of 1, so requests are
served one at a time rather than batched. There's no training or fine-tune
verb, and no HTTP surface - reach this model through the CLI or D-Bus.
