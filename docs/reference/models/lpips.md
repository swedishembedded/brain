# Perceptual image distance (LPIPS)

Scores how different two same-sized RGB images look to a person, with the
learned perceptual metric LPIPS (v0.1, AlexNet trunk; "The Unreasonable
Effectiveness of Deep Features as a Perceptual Metric", 2018,
arXiv:1801.03924). It is the number novel-view synthesis and reconstruction
work reports next to PSNR and SSIM. Lower is closer; identical images give
exactly 0.

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

Model id: `brain/lpips` - not auto-fetched, and assembled from two upstream
files that brain finds by their tensor shapes, never by file name:

- the AlexNet trunk (torchvision's `alexnet-owt-7be5be79.pth`, or the same
  tensors as safetensors), and
- the LPIPS v0.1 linear heads (`lin0` to `lin4`).

Put both anywhere in the models directory (`--models-dir`); no variable to
export. `tools/goldens/lpips_dump_reference.py` puts both into the store.

## Running it

```bash
brain caps lpips
```

One action, `brain/lpips distance`, takes two image blobs, `a` and `b`; it returns the total distance and the contribution of
each of the five AlexNet layers. The two images must be the same size.

## How it is built

The metric runs on the device from the shared conv-net blocks and existing
kernels only; there is no LPIPS-specific kernel. The architecture is fixed
(`crates/lpips/src/config.rs`) and the importer validates every tensor by
shape.
