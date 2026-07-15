#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump tiny random-weight DIAMOND reference fixtures for brain parity tests.

Runs the ORIGINAL implementation from resources/world-models/repos/diamond on a
tiny fixed-seed config/input and dumps every module's weights, inputs, and
per-module activations as raw little-endian f32 blobs + manifest.json.

This script is resource-prep tooling: it is NEVER part of brain's build/test
path (see docs/world-models/FIXTURES.md). Output is copied into
crates/wm-diamond/tests/fixtures/diamond/ and committed.

Usage:
  python3 scripts/parity-dump/diamond.py --out /tmp/fixtures-diamond
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

DIAMOND_REPO = pathlib.Path("/data/workspace/resources/world-models/repos/diamond")

import torch  # noqa: E402

# Load blocks.py / inner_model.py directly as a synthetic package, bypassing the
# repo's models/__init__.py (which drags in omegaconf/wandb/hydra we don't need).
import importlib.util  # noqa: E402
import types  # noqa: E402

_SRC = DIAMOND_REPO / "src"
for _pkg, _p in (("models", _SRC / "models"), ("models.diffusion", _SRC / "models" / "diffusion")):
    _m = types.ModuleType(_pkg)
    _m.__path__ = [str(_p)]
    sys.modules[_pkg] = _m


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("models.blocks", _SRC / "models" / "blocks.py")
_inner = _load("models.diffusion.inner_model", _SRC / "models" / "diffusion" / "inner_model.py")
InnerModel, InnerModelConfig = _inner.InnerModel, _inner.InnerModelConfig

SEED = 7
# Tiny but structurally complete: 2 levels (1 downsample), mid attention,
# GroupNorm with 1 group (8 < GN_GROUP_SIZE=32 -> max(1, .)), AdaGroupNorm cond.
CFG = InnerModelConfig(
    img_channels=3,
    num_steps_conditioning=2,
    cond_channels=16,           # act_emb dim = 16 // 2 = 8 per step
    depths=[1, 1],
    channels=[8, 8],
    attn_depths=[False, True],
    num_actions=4,
)
H = W = 8


def tensor_blob(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = InnerModel(CFG).eval()
    # conv_out / attn out_proj are zero-init in the reference; give them small
    # deterministic values so the parity test exercises those paths too.
    with torch.no_grad():
        g = torch.Generator().manual_seed(SEED + 1)
        for name, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    torch.manual_seed(SEED + 2)
    noisy = torch.randn(1, CFG.img_channels, H, W)
    c_noise = torch.tensor([0.4])
    obs = torch.randn(1, CFG.num_steps_conditioning * CFG.img_channels, H, W)
    act = torch.tensor([[1, 3]])  # (B, num_steps_conditioning)

    acts: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _i, o):
            if isinstance(o, torch.Tensor):
                acts[name] = o
            elif isinstance(o, tuple) and isinstance(o[0], torch.Tensor):
                acts[name] = o[0]
        return fn

    for name, mod in model.named_modules():
        if name:
            mod.register_forward_hook(hook(name))

    with torch.no_grad():
        y = model(noisy, c_noise, obs, act)

    manifest = {
        "model": "diamond-inner-tiny",
        "config": {
            "img_channels": CFG.img_channels,
            "num_steps_conditioning": CFG.num_steps_conditioning,
            "cond_channels": CFG.cond_channels,
            "depths": CFG.depths,
            "channels": CFG.channels,
            "attn_depths": CFG.attn_depths,
            "num_actions": CFG.num_actions,
            "h": H, "w": W,
        },
        "provenance": {
            "repo": str(DIAMOND_REPO),
            "repo_commit": subprocess.run(
                ["git", "-C", str(DIAMOND_REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip(),
            "torch": torch.__version__,
            "seed": SEED,
            "cmd": " ".join(sys.argv),
        },
        "weights": {}, "inputs": {}, "activations": {}, "output": {},
    }

    def dump(section: str, name: str, t: torch.Tensor) -> None:
        blob = tensor_blob(t)
        assert len(blob) <= 256 * 1024, f"{name}: {len(blob)}B exceeds 256KB cap"
        fname = f"{section}.{name.replace('.', '_')}.f32"
        (out / fname).write_bytes(blob)
        manifest[section][name] = {
            "file": fname,
            "shape": list(t.shape),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }

    for name, p in model.named_parameters():
        dump("weights", name, p)
    dump("inputs", "noisy", noisy)
    dump("inputs", "c_noise", c_noise)
    dump("inputs", "obs", obs)
    dump("inputs", "act", act.to(torch.float32))
    for name, t in acts.items():
        dump("activations", name, t)
    dump("output", "y", y)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(f.stat().st_size for f in out.iterdir())
    assert total <= 2 * 1024 * 1024, f"total {total}B exceeds 2MB cap"
    print(f"wrote {len(list(out.iterdir()))} files, {total/1024:.0f} KiB -> {out}")


if __name__ == "__main__":
    main()
