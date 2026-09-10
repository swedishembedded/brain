#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements resident, session-reusing inference serving
# for its clients - the difference between a dataset pipeline that reloads
# tens of gigabytes of weights per image and one that loads them once. If your
# team needs expertise in model serving, D-Bus/IPC control planes, or
# identity-preserving image generation, you can procure our services by
# sending an email to info@swedishembedded.com.

"""One generated PNG from a RESIDENT `brain serve --dbus` session.

`identity_yolo_pipeline.sh`'s image generator. It replaces the two fresh
CLI subprocesses that script used to spawn per image (`brain pulid text2image`
and `brain flux2 generate`), each of which reloaded its whole weight set -
~27 GB for PuLID (FLUX.1-dev DiT + T5-XXL + CLIP-L + the PuLID adapter +
ArcFace + BiSeNet + EVA-CLIP), ~10-15 GB for FLUX.2 Klein (DiT + Qwen3-8B +
VAE) - so a 75-image run paid that cost 75 times.

Driving the same two actions over `com.swedishembedded.Brain1` instead, the
daemon's residency executor keeps one built instance per instance key and
every later call reuses it. Both backends key on exactly what this script
holds constant across a pipeline run, so all its calls land on one instance:

* `brain/flux1-pulid` - `resident_pulid.rs` uses a single `"default"`
  instance key, and `pulid::caps::Session` then caches one built bundle per
  `(variant, height, width, precision)`.
* `brain/flux2-klein` - `resident_flux2.rs` keys on
  `"{variant}:{precision}:{w}x{h}:{nref}"`, with `variant` bound to the real
  weights at daemon startup (never per request). Note the server's own
  default precision there is `fp32`; `--precision` is always sent so the key
  matches the `int8` build the pipeline actually wants.

Parameters not named here are left to the action's own `ActionSpec` defaults,
which is exactly what the CLI path this replaces did, so generations are
unchanged apart from where the weights came from.

Images cross the bus as HWC f32 RGB in `[0,1]` over sealed memfds; PNG
encode/decode happens here (via Pillow) because the rest of the pipeline
reads and writes PNG.

The daemon must already be running on this session bus with the weight env
vars set - see `identity_yolo_pipeline.sh`, which starts it, or:

    dbus-run-session -- bash -c '
      BRAIN_FLUX1_DIR=... BRAIN_PULID_DIR=... BRAIN_ARCFACE_DIR=... \\
      BRAIN_CLIP_DIR=... BRAIN_BISENET_DIR=... \\
      BRAIN_FLUX2_DIT=... BRAIN_FLUX2_VAE=... BRAIN_FLUX2_TE=... \\
      BRAIN_FLUX2_TOKENIZER=... BRAIN_FLUX2_ALLOW_NC=1 \\
      brain serve --dbus & sleep 2
      python3 examples/imagegen/identity_yolo_gen.py pulid \\
        --prompt "a photo of a person in a park" --face portrait.jpg --out img.png'

Requires: jeepney + Pillow - `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainError  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import from_pil_rgb, to_pil  # noqa: E402

#: Identity-conditioned generation (`crates/pulid/src/caps.rs`). `text2image`
#: is a plain (non-streaming) action, so it goes through `Run`.
PULID_MODEL = "brain/flux1-pulid"
#: Unconditioned generation for negatives/backgrounds/strangers
#: (`crates/flux2/src/caps.rs`). Its `text2image` IS streaming, so it goes
#: through `Subscribe` and reports per-denoise-step progress.
FLUX2_MODEL = "brain/flux2-klein"


def load_face(path: str) -> tuple[bytes, int, int]:
    """A face photo (any format Pillow reads - the pipeline's source is JPEG)
    as an HWC f32 RGB blob plus its size, ready to send as `face_image`."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return from_pil_rgb(img), img.width, img.height


def save_png(path: str, data: bytes, w: int, h: int, c: int) -> None:
    """Write an HWC f32 image blob out as PNG - what the rest of the pipeline
    (letterboxing, dataset packing, auto-labeling) reads."""
    to_pil(data, w, h, c).save(path)


def write_image(outcome, out: str, want_w: int, want_h: int) -> int:
    """Save an outcome's `image` blob at the size the server reported."""
    data = outcome.blobs.get("image")
    if data is None:
        print("  no image blob arrived", file=sys.stderr)
        return 1
    meta = (outcome.meta.get("image") or {}).get("meta") or {}
    w, h, c = int(meta.get("w", want_w)), int(meta.get("h", want_h)), int(meta.get("c", 3))
    save_png(out, data, w, h, c)
    return 0


def wait_for(model: str, timeout: float) -> int:
    """Block until the daemon on this session bus advertises `model`.

    Two distinct failures both surface here rather than at the first (minutes
    long) generation: a daemon that is not up yet, and a daemon that IS up but
    was started without the weight env vars this model is gated on - which
    registers no such model at all, silently, by design.
    """
    deadline = time.monotonic() + timeout
    last = "no connection to the session bus"
    while time.monotonic() < deadline:
        try:
            with BrainDBus() as brain:
                served = brain.models()
                if model in served:
                    print(f"{model}: served", file=sys.stderr)
                    return 0
                last = f"daemon is up but does not serve {model!r} (models: {served}) - check its BRAIN_* weight variables"
        except Exception as e:  # bus not up yet, name not taken yet
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.0)
    print(f"ERROR: waited {timeout:.0f}s for {model}: {last}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["pulid", "flux2"],
                    help="pulid = identity-conditioned FLUX.1; flux2 = plain FLUX.2 Klein")
    ap.add_argument("--prompt", help="text description of the desired image")
    ap.add_argument("--out", help="output PNG path")
    ap.add_argument("--seed", type=int, help="RNG seed (flux2 requires one; pulid honours it)")
    ap.add_argument("--wait", type=float, metavar="SECONDS",
                    help="readiness probe: generate nothing, just block until the daemon advertises this "
                         "backend's model, then exit. A daemon started without the right BRAIN_* weight "
                         "variables then fails here in seconds instead of at the first two-minute generation")
    ap.add_argument("--face", help="pulid only: photo of the identity to condition on (required there)")
    ap.add_argument("--width", type=int, default=512, help="output width (multiple of 16)")
    ap.add_argument("--height", type=int, default=512, help="output height (multiple of 16)")
    ap.add_argument("--steps", type=int, default=0,
                    help="denoise steps; 0 = variant default. Ignored by the distilled klein variants, whose sampler is fixed")
    ap.add_argument("--precision", default="int8", choices=["fp32", "int8"],
                    help="DiT numeric tier; part of flux2's instance key, so it must stay constant across a run")
    ap.add_argument("--variant", help="model variant; flux2's must match the variant the daemon bound from its weights")
    ap.add_argument("--progress", action="store_true", help="print a line per denoise step (flux2 only)")
    args = ap.parse_args()

    model = PULID_MODEL if args.backend == "pulid" else FLUX2_MODEL
    if args.wait is not None:
        return wait_for(model, args.wait)
    for name in ("prompt", "out", "seed"):
        if getattr(args, name) is None:
            ap.error(f"--{name} is required when generating")
    if args.backend == "pulid" and not args.face:
        ap.error("pulid needs --face (the identity to condition on)")

    params = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "precision": args.precision,
    }
    if args.variant:
        params["variant"] = args.variant

    t0 = time.monotonic()
    with BrainDBus() as brain:
        served = brain.models()
        if model not in served:
            print(f"ERROR: {model!r} is not served by the running daemon (models: {served}) - "
                  "check the BRAIN_* weight variables it was started with", file=sys.stderr)
            return 1
        try:
            if args.backend == "pulid":
                face, fw, fh = load_face(args.face)
                outcome = brain.run(
                    model, "text2image", params,
                    blobs={"face_image": face},
                    meta={"face_image": {"media": "image", "w": fw, "h": fh, "c": 3}},
                )
            else:
                on_progress = None
                if args.progress:
                    def on_progress(step: int, total: int, message: str) -> None:  # noqa: E306
                        print(f"  [{step}/{total}] {message}", file=sys.stderr, flush=True)
                outcome = brain.subscribe(model, "text2image", params, timeout=7200.0, on_progress=on_progress)
        except BrainError as e:
            print(f"ERROR: {model} text2image failed: {e}", file=sys.stderr)
            return 1

    rc = write_image(outcome, args.out, args.width, args.height)
    if rc == 0:
        print(f"{args.backend}: {args.out} seed={args.seed} in {time.monotonic() - t0:.1f}s", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
