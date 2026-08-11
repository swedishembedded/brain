# Face detection and identity (FaceNet)

Face detection plus identity embeddings: find every face in an image
(boxes, scores, landmarks), or turn an aligned face crop into a 512-d
embedding you can use for identity matching and search — "are these two
photos the same person?" or "find this face among a set of known ones."
Built on the well-known insightface antelopev2 stack (SCRFD detection +
ArcFace-style embeddings), so it's also the identity input other generative
pipelines condition on.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain do`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/facenet` — not auto-fetched. Set `BRAIN_FACENET_DIR` to a
directory holding both released ONNX graphs under their antelopev2 names:
`glintr100.onnx` (the embedding model) and `scrfd_10g_bnkps.onnx` (the
detector). Both are read directly — no import or conversion step.

## Running it

No dedicated `brain facenet` verb — use the generic `brain do` pair:

```bash
brain caps brain/facenet

brain do brain/facenet detect --in image=photo.ppm --json

brain do brain/facenet embed --align false \
    --in image=aligned112.ppm --out embedding=id.bin
```

Two actions:

- **`detect`** — required `image` input; param `max_faces` (default `0` =
  all); returns boxes/scores/5-point landmarks in source-image pixels.
- **`embed`** — required `image` input; params `align` (bool, default
  `true`: detect + similarity-align the primary face; `false` = the input is
  already an aligned 112x112 face) and `select` (`largest`|`score`, default
  `largest`); returns a 512-d, L2-normalized embedding as blob `embedding`.

The same two actions are reachable over D-Bus via the generic
`Run(model, action, params, in_fds, in_meta, transport)` call. Reference
client: `examples/vision/face_id.py` (see `examples/vision/README.md`).

## Options

| Param | Action | Effect |
|---|---|---|
| `max_faces` | `detect` | cap on faces returned (default `0` = all) |
| `align` | `embed` | detect + align the primary face first (default `true`), or `false` if the input is already an aligned 112x112 crop |
| `select` | `embed` | which face to embed when several are found: `largest` (default) or `score` |

To compare two identities, embed each face and compare the two 512-d
vectors (e.g. cosine similarity) yourself — that comparison isn't a
server-side action.

## Hardware and limits

Both released ONNX graphs are pinned to a batch size of 1, so requests are
served one at a time rather than batched. There's no training or fine-tune
verb, and no HTTP surface — reach this model through `brain do` or D-Bus.
