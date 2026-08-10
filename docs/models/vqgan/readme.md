# VQGAN discrete autoencoder (`crates/vqgan`)

Image <-> codebook indices: `encode` an image to a grid of discrete codes,
`decode` codes back to an image. This is the VQ core that
[`crates/restore`](../restore/readme.md)'s CodeFormer restoration is built on
top of — `restore` adds no second copy of it.

## Model id and weights

- **Id:** `brain/vqgan` — reserved vendor `brain/`, never fetched.
- **Weights:** `BRAIN_VQGAN_WEIGHTS` — a released checkpoint
  (`vqgan_code1024.pth` or `codeformer.pth`) or the directory holding one; a
  directory resolves to `vqgan_code1024.pth` first, falling back to
  `codeformer.pth` (`vqgan::caps::checkpoint_path`). Both files share every VQ
  tensor name but are different weights and produce different codes — a
  directory resolution is logged (`vqgan: <dir> -> <file>`) so the choice is
  never silent.

## Surfaces

CLI and D-Bus. Not HTTP: neither `encode` nor `decode` is named `generate` (no
`chat`), and both require an input blob (`image` / `codes`), so neither
qualifies as the `image` (text-to-image) capability — correctly absent from
`/v1/models` and `/v1/images/generations`.

## Inference

### CLI
No dedicated `brain vqgan` verb. Use the generic pair:
```bash
brain caps brain/vqgan
brain do brain/vqgan encode --in image=face.ppm --out codes=codes.bin --json
brain do brain/vqgan decode --in codes=codes.bin --out image=recon.ppm
```

### D-Bus
Two actions, sharing one resident instance per square `size` (`instance_key` is
the size, not the action name):

- **`encode`** — required input `image` (`Media::Image`), optional int param
  `size` (default 512, must be a multiple of the 32x downscale), output `codes`
  (`Media::Bytes`, `u32` little-endian, meta `{lh, lw, codebook_size}`).
- **`decode`** — required input `codes` (`Media::Bytes`, exactly `lh*lw` `u32`
  indices), optional int param `size`, output `image` (`Media::Image`, RGB in
  `[0,1]`).

```bash
BRAIN_VQGAN_WEIGHTS=/path/to/vqgan_code1024.pth \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/restore/vq_roundtrip.py --image face.ppm --corrupt 20'
```

Reference client: [`examples/restore/vq_roundtrip.py`](../../../examples/restore/vq_roundtrip.py).

## Training / Fine-tune / LoRA

A trainer exists (`crates/vqgan/src/train.rs`, gradcheck via
`gradcheck::check_vqgan`) but has no CLI verb.

## Not supported

`training`, `finetune`, `LoRA`, `QLoRA`, `batch > 1`, `HTTP`.

## See also

- Crate: [`crates/vqgan`](../../../crates/vqgan)
- Workstream ledger: [`docs/models/vqgan/status.md`](status.md) — forward
  parity worst case 1-cos 1.63e-10, rel L2 1.80e-5, 0/256 index mismatches;
  `codebook_feat` bit-exact. Training/backward done (gradcheck-gated,
  `gradcheck::check_vqgan`).
- Built on by: [`../restore/readme.md`](../restore/readme.md) (CodeFormer face
  restoration)
