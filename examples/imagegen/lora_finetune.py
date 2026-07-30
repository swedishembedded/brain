#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""FLUX.2 Klein LoRA fine-tuning over brain's D-Bus interface.

Submits the streaming `lora_train` action (the dataset dir and save path are
**server-side** — training data never crosses the bus), live-prints the
per-step loss from the progress frames, receives the trained adapter back as
an fd blob (a remote client has no access to the server's filesystem), saves
it locally, then immediately generates a test image **with the adapter
applied** (`text2image` has an `adapter` param wired through the pipeline's
LoRA fold-in).

    dbus-run-session -- bash -c '
      BRAIN_FLUX2_DIT=… BRAIN_FLUX2_VAE=… BRAIN_FLUX2_TE=… BRAIN_FLUX2_TOKENIZER=… \
      brain serve --dbus & sleep 2
      python3 examples/imagegen/lora_finetune.py --data /srv/mydataset \
          --save out/flux2-my.lora --prompt "a photo of sks dog on the moon"'

Requires: jeepney (pip install brain-py[dbus]).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd  # noqa: E402

from generate import MODEL, run_streaming  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="SERVER-side folder of captioned images (data::imageset)")
    ap.add_argument("--save", required=True, help="SERVER-side path for the trained adapter")
    ap.add_argument("--adapter-out", default="adapter.lora", help="local copy of the returned adapter blob")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200, help="training steps")
    ap.add_argument("--size", type=int, default=512, help="training square size (multiple of 16)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="klein-4b", choices=["klein-4b", "klein-9b", "base-4b", "base-9b"])
    ap.add_argument("--prompt", default="", help="if set: generate a test image with the adapter after training")
    ap.add_argument("--out", default="adapted.ppm", help="test-image PPM path (with --prompt)")
    args = ap.parse_args()

    params = {
        "data": args.data,
        "save": args.save,
        "rank": args.rank,
        "steps": args.steps,
        "size": args.size,
        "lr": args.lr,
        "seed": args.seed,
        "variant": args.variant,
    }

    with BrainDBus() as brain:
        models = brain.models()
        if MODEL not in models:
            print(f"{MODEL} not served (models: {models}); set BRAIN_FLUX2_*", file=sys.stderr)
            return 1

        print(f"lora_train rank={args.rank} steps={args.steps} ({args.variant}):")
        adapter = None
        for frame, fds in brain.subscribe(MODEL, "lora_train", params, timeout=48 * 3600.0):
            kind = frame.get("type")
            if kind == "progress":
                # per-step messages look like "step 3/200  loss 0.41231  (95.2 s)"
                print(f"  {frame.get('message', '')}", flush=True)
            elif kind == "blob" and frame.get("name") == "adapter":
                adapter = read_fd(fds[0])
            elif kind == "done":
                print(f"  done: {frame.get('result')}")
            elif kind == "error":
                print(f"  ERROR: {frame.get('message')}", file=sys.stderr)
                return 1
        if adapter is None:
            print("  no adapter blob arrived", file=sys.stderr)
            return 1
        Path(args.adapter_out).write_bytes(adapter)
        print(f"wrote {args.adapter_out} ({len(adapter)} bytes; server copy at {args.save})")

        if not args.prompt:
            return 0
        # Generate with the freshly trained adapter folded into the DiT
        # (`adapter` is the SERVER-side save path).
        print(f"text2image with adapter={args.save}:")
        return run_streaming(
            brain,
            "text2image",
            {"prompt": args.prompt, "variant": args.variant, "adapter": args.save},
            args.out,
        )


if __name__ == "__main__":
    sys.exit(main())
