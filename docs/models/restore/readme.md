# CodeFormer face restoration (`crates/restore`)

Blind face restoration: a degraded (ideally aligned) face in, a restored 512x512
face out, with a continuous identity-fidelity dial `w`. Built on `crates/vqgan`'s
VQ autoencoder (see [../vqgan/readme.md](../vqgan/readme.md)).

## Model id and weights

- **Id:** `brain/restore` — reserved vendor `brain/`, never fetched.
- **Weights:** `BRAIN_RESTORE_WEIGHTS` — either `codeformer.pth` itself or the
  directory holding it (resolved by `restore::caps::checkpoint_path`).

## Surfaces

CLI and D-Bus. Not HTTP: the action is `restore_face`, not `generate` (no
`chat`), and it requires an `image` input blob, so it does not qualify as the
`image` (text-to-image) capability either — it is correctly absent from
`/v1/models` and `/v1/images/generations`.

## Inference

### CLI
No dedicated `brain restore` verb. Use the generic pair:
```bash
brain caps brain/restore
brain do brain/restore restore_face --w 0.5 --in image=face.ppm --out image=restored.ppm --json
```

### D-Bus
One action, `restore_face`: required input `image` (`Media::Image`, ideally an
aligned 512x512 face), one float param `w` (`0.0..=1.0`, default `0.5`,
identity-fidelity dial — 0 = maximum quality, 1 = maximum fidelity to the
input), output `image` (the restored 512x512 face, RGB in `[0,1]`).

```bash
BRAIN_RESTORE_WEIGHTS=/path/to/codeformer \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/restore/restore_face.py --image face.ppm --w 0,0.5,1'
```

Reference client: [`examples/restore/restore_face.py`](../../../examples/restore/restore_face.py).
The action takes an already-aligned face; pair it with `facenet detect`
(`examples/vision/face_id.py`) to locate one in a full photo first — CodeFormer's
FFHQ 512x512 alignment template is not `facenet::ARCFACE_DST_112` rescaled, so
the two are not chained in automatically.

## Training / Fine-tune / LoRA

A trainer exists (`crates/restore/src/train.rs`, gradcheck via
`gradcheck::check_codeformer`) but has no CLI verb.

## Not supported

`training`, `finetune`, `LoRA`, `QLoRA`, `batch > 1`, `HTTP`.

## See also

- Crate: [`crates/restore`](../../../crates/restore)
- Workstream ledger: [`docs/models/restore/status.md`](status.md) — forward
  parity: 0/256 predicted-index mismatches, encoder+transformer cosine
  1.000000000; the D-Bus round trip measures cosine 0.999998 at every `w`.
  Deferred: `adain=True`, face detection/alignment, `run_batch` (batch > 1).
- VQ autoencoder this crate builds on: [`../vqgan/readme.md`](../vqgan/readme.md)
