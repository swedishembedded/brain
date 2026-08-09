# Face recognition — antelopev2 (`crates/facenet`)

insightface's antelopev2 stack: SCRFD-10GF face detection (boxes, scores, 5-point
landmarks) and ArcFace IResNet-100 (`glintr100`) 512-d identity embeddings, for
face search/verification and as the identity conditioning for PuLID-style
generation.

## Model id and weights

- **Id:** `brain/facenet` — reserved vendor `brain/`, never auto-fetched.
- **Weights:** `BRAIN_FACENET_DIR` — a directory holding both released ONNX
  graphs by their antelopev2 names: `glintr100.onnx` (ArcFace) and
  `scrfd_10g_bnkps.onnx` (SCRFD). No import/conversion step; both are read
  directly.

## Surfaces

CLI (`brain do`) and D-Bus — both actions are registered in
`crates/cli/src/catalog.rs` and `crates/cli/src/resident_facenet.rs`. Not HTTP:
`embed` is right-named but requires an `image` blob and no `text` param, which
`crates/apiserve/src/catalog.rs`'s embeddings classifier now checks for
specifically (it used to admit any action literally named `embed`, which
would have listed this model on `/v1/embeddings` and then 400ed on the
missing input — see that module's docs).

## Inference

### CLI

No dedicated `brain facenet` verb; the generic pair:

```bash
brain caps brain/facenet
brain do brain/facenet detect --in image=photo.ppm --json
brain do brain/facenet embed --align false --in image=aligned112.ppm --out embedding=id.bin
```

Two actions:
- **`detect`** — `image` in (required); param `max_faces` (default `0` = all);
  outputs boxes/scores/5-point landmarks in **source-image pixels**.
- **`embed`** — `image` in (required); params `align` (bool, default `true`:
  detect + similarity-align the primary face; `false` = input is already an
  aligned 112×112 face) and `select` (enum `largest`|`score`, default
  `largest`); outputs a 512-d L2-normalised embedding as blob `embedding`
  (`Media::Bytes`).

### D-Bus

Same two actions via the generic `Run(model, action, params, in_fds, in_meta,
transport)`. Reference client:
[`examples/vision/face_id.py`](../../../examples/vision/face_id.py) — see
[`examples/vision/README.md`](../../../examples/vision/README.md).

## Training / Fine-tune / LoRA

A real trainer exists (`crates/facenet/src/train.rs`: `ArcFaceTrainer`,
additive-angular-margin head, finite-diff gradchecked via
`gradcheck::check_arcface`) but has no CLI verb — no command to give.

## Not supported

`training` (verb), `finetune`, `LoRA`, `QLoRA`, `HTTP`, `batch > 1` — both
released ONNX graphs are pinned to `N = 1`; `Instance::run_batch` is
deliberately the serial default (see `crates/cli/src/resident_facenet.rs`).

## See also

- Crate: `crates/facenet`
- Workstream ledger: [`status.md`](status.md)
- [`examples/vision/README.md`](../../../examples/vision/README.md)
