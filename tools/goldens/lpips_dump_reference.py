#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Put LPIPS v0.1 (AlexNet) into brain's model store and dump the reference
distances `crates/lpips`'s tests compare against.

LPIPS (Zhang et al. 2018, "The Unreasonable Effectiveness of Deep Features as
a Perceptual Metric") is two upstream releases, neither on a model hub brain's
`brain pull` speaks to, so this tool fetches them the way the other non-hub
releases in the store arrived (`pulid_face_parsing_dump_reference.py` for
facexlib's BiSeNet): download, verify the pinned sha256, and re-serialize only
what `checkpoint` cannot read.

  alexnet-owt-7be5be79.pth   torchvision's ImageNet AlexNet, the trunk LPIPS
                             taps. Zip-container `torch.save`, which
                             `checkpoint::torchpt` reads as is, so it lands
                             unchanged at
                             <models>/pytorch/vision/alexnet-owt-7be5be79.pth.
  alex.pth                   LPIPS v0.1's five 1x1 linear heads
                             (richzhang/PerceptualSimilarity). Torch's pre-1.6
                             LEGACY pickle format, which `checkpoint::torchpt`
                             does not read, so it is re-serialized (the same
                             five tensors, byte for byte as f32) to
                             <models>/richzhang/PerceptualSimilarity/alex.safetensors,
                             carrying the source URL and sha256 in its metadata.

`crates/lpips/src/spec.rs` finds both by content, wherever they sit in the
store.

Goldens, under <testdata>/lpips/: the LPIPS repository's own example images
(`ex_ref.png`, `ex_p0.png`, `ex_p1.png`, the pair its `test_network.py` scores)
and `manifest.json` holding the distances the official `lpips` package
computes for (ex_ref, ex_p0) and (ex_ref, ex_p1), total and per layer, with
the trunk and heads loaded from the two store files above - so the reference
number and brain's number come from the identical weights.

Usage:
  python tools/goldens/lpips_dump_reference.py [--models-dir DIR] [--testdata testdata]

`--models-dir` defaults to brain's own resolution (`$BRAIN_MODELS_DIR`, else
`$XDG_DATA_HOME/brain/models`, else `~/.local/share/brain/models`).
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LIN_URL = "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth"
LIN_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
IMG_URL = "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/imgs/{}"
EXAMPLES = ("ex_ref.png", "ex_p0.png", "ex_p1.png")

# The five taps of torchvision's `alexnet().features` LPIPS reads (after each
# ReLU): the convolutions at these indices, each with weight and bias.
ALEXNET_CONVS = (0, 3, 6, 8, 10)
# `lpips.pretrained_networks.alexnet` slices `features` at these bounds; the
# slice a conv index falls in names its module in the LPIPS network.
SLICE_OF_CONV = {0: 1, 3: 2, 6: 3, 8: 4, 10: 5}


def default_models_dir():
    if os.environ.get("BRAIN_MODELS_DIR"):
        return os.environ["BRAIN_MODELS_DIR"]
    if os.environ.get("XDG_DATA_HOME"):
        return os.path.join(os.environ["XDG_DATA_HOME"], "brain", "models")
    return os.path.join(os.path.expanduser("~"), ".local", "share", "brain", "models")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, sha256=None):
    """Download `url` to `dest` through a `.part` sibling, verify `sha256`
    when given, rename into place. An existing `dest` is re-verified, never
    trusted."""
    if os.path.exists(dest):
        if sha256 is None or sha256_file(dest) == sha256:
            print(f"have {dest}", flush=True)
            return
        raise SystemExit(f"{dest}: sha256 {sha256_file(dest)} is not the pinned {sha256}; remove it and re-run")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    print(f"fetch {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    got = sha256_file(part)
    if sha256 is not None and got != sha256:
        os.remove(part)
        raise SystemExit(f"{url}: sha256 {got}, expected {sha256}")
    os.replace(part, dest)
    print(f"wrote {dest} (sha256 {got})", flush=True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402  (tools/goldens is this file's own dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=default_models_dir())
    ap.add_argument("--testdata", default="testdata")
    args = ap.parse_args()

    alexnet_path = os.path.join(args.models_dir, "pytorch", "vision", "alexnet-owt-7be5be79.pth")
    lin_dir = os.path.join(args.models_dir, "richzhang", "PerceptualSimilarity")
    lin_safetensors = os.path.join(lin_dir, "alex.safetensors")
    fetch(ALEXNET_URL, alexnet_path, ALEXNET_SHA256)

    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    # The legacy-format heads only ever live in a scratch directory next to
    # the goldens: the store keeps the re-serialization, not a file its
    # reader cannot open.
    out_dir = os.path.join(args.testdata, "lpips")
    os.makedirs(out_dir, exist_ok=True)
    lin_pth = os.path.join(out_dir, "alex.pth")
    fetch(LIN_URL, lin_pth, LIN_SHA256)
    lin = torch.load(lin_pth, map_location="cpu", weights_only=True)
    lin = {k: v.to(torch.float32).contiguous() for k, v in lin.items()}
    expected = {f"lin{i}.model.1.weight" for i in range(5)}
    if set(lin) != expected:
        raise SystemExit(f"{lin_pth}: tensors {sorted(lin)}, expected {sorted(expected)}")
    os.makedirs(lin_dir, exist_ok=True)
    save_file(lin, lin_safetensors + ".part", metadata={"source": LIN_URL, "source_sha256": LIN_SHA256, "lpips_version": "0.1", "net": "alex"})
    os.replace(lin_safetensors + ".part", lin_safetensors)
    print(f"wrote {lin_safetensors} (sha256 {sha256_file(lin_safetensors)})", flush=True)

    # The official implementation, with its trunk replaced by the store copy
    # of torchvision's AlexNet rather than whatever torchvision would fetch.
    import lpips

    model = lpips.LPIPS(net="alex", version="0.1", pnet_rand=True, model_path=lin_pth, verbose=False)
    trunk = torch.load(alexnet_path, map_location="cpu", weights_only=True)
    net_state = {}
    for i in ALEXNET_CONVS:
        for p in ("weight", "bias"):
            net_state[f"slice{SLICE_OF_CONV[i]}.{i}.{p}"] = trunk[f"features.{i}.{p}"]
    model.net.load_state_dict(net_state, strict=True)
    model.eval()

    images = {}
    manifest = {"images": {}, "pairs": {}}
    for name in EXAMPLES:
        path = os.path.join(out_dir, name)
        fetch(IMG_URL.format(name), path)
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
        manifest["images"][name] = {"sha256": sha256_file(path), "width": rgb.shape[1], "height": rgb.shape[0]}
        # lpips.im2tensor: x / 127.5 - 1, i.e. [-1, 1].
        images[name] = torch.from_numpy(rgb / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        for other in ("ex_p0.png", "ex_p1.png"):
            total, layers = model.forward(images["ex_ref.png"], images[other], retPerLayer=True)
            manifest["pairs"][f"ex_ref.png|{other}"] = {
                "lpips": float(total.flatten()[0]),
                "layers": [float(l.flatten()[0]) for l in layers],
            }
            print(f"lpips(ex_ref, {other}) = {float(total.flatten()[0]):.6f}", flush=True)

    manifest["source"] = source_block(
        checkpoint="richzhang/PerceptualSimilarity v0.1 alex + torchvision alexnet-owt-7be5be79",
        files=[alexnet_path, lin_safetensors],
        identity={"relu1": 64, "relu2": 192, "relu3": 384, "relu4": 256, "relu5": 256},
    )
    manifest["lpips_package"] = getattr(lpips, "__version__", "unknown")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {os.path.join(out_dir, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
