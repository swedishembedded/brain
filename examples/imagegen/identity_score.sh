#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Is the generated face actually THEIR face? Answer as a number.
#
#   examples/imagegen/identity_score.sh <ref-dir> <generated-dir>...
#
# Prints, per generated folder, the mean ArcFace cosine between every image in
# it and the reference photographs -- the same 512-d identity embedding face
# recognition itself is built on, so the number means what it says.
#
# Read it against these anchors, which this repo measures rather than assumes:
#
#   ~0.0   a stranger. Text-only generation of a named person lands here,
#          because the model has never seen them.
#   ~0.3   recognisably influenced -- family-resemblance territory.
#   ~0.5   the same person to a human, most of the time.
#   ~0.7+  what two genuine photographs of one person score against each other.
#          This is the ceiling, not 1.0: pose, lighting and age move it.
#
# A conditioning path that does not MOVE this number is not working, however
# good the pictures look. That is the whole point of running it: identity is
# the one property of a portrait you cannot grade by eye without fooling
# yourself, because the eye grades "plausible person" and this grades "that
# person".
#
# The reference directory is the folder of their photographs (any name, any
# format brain decodes); the generated directories are what you are grading.
# Faces that fill the entire frame are NOT detectable -- the detector needs
# context around the head -- so generate head-and-shoulders framing if you
# intend to measure it. Undetectable images are reported, never scored as 0.
#
# Weights: BRAIN_ARCFACE_DIR and BRAIN_SCRFD_DIR, both the directory holding
# the insightface antelopev2 `glintr100.onnx` and `scrfd_10g_bnkps.onnx`.
#
# Optional, all env: BRAIN.

set -euo pipefail

[ "$#" -ge 2 ] || { echo "usage: identity_score.sh <ref-dir> <generated-dir>..." >&2; exit 2; }
: "${BRAIN_ARCFACE_DIR:?set BRAIN_ARCFACE_DIR to the antelopev2 directory}"
: "${BRAIN_SCRFD_DIR:=$BRAIN_ARCFACE_DIR}"
export BRAIN_SCRFD_DIR

BRAIN="${BRAIN:-./target/release/brain}" python3 - "$@" <<'PY'
import glob, os, subprocess, sys, tempfile
import numpy as np

BRAIN = os.environ["BRAIN"]
IMG = (".jpg", ".jpeg", ".png", ".ppm", ".webp", ".bmp")


def images(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(IMG))


def embed(path, tmp):
    """One L2-normalised 512-d ArcFace vector, or None if no face is found."""
    out = os.path.join(tmp, os.path.abspath(path).replace(os.sep, "_") + ".bin")
    subprocess.run(
        [BRAIN, "arcface", "embed", "--in", f"image={path}", "--out", f"embedding={out}"],
        capture_output=True, text=True,
    )
    if not os.path.exists(out):
        return None
    v = np.fromfile(out, dtype="<f4")
    n = np.linalg.norm(v)
    return v / n if n else None


ref_dir, gen_dirs = sys.argv[1], sys.argv[2:]
with tempfile.TemporaryDirectory() as tmp:
    refs = [e for e in (embed(p, tmp) for p in images(ref_dir)) if e is not None]
    if not refs:
        sys.exit(f"no face found in any reference under {ref_dir}")
    R = np.stack(refs)
    # The references' agreement with EACH OTHER is the ceiling any generated
    # image is really competing against, so print it rather than let the reader
    # compare against an imagined 1.0.
    if len(refs) > 1:
        off = R @ R.T
        ceiling = off[~np.eye(len(refs), dtype=bool)].mean()
        print(f"reference set: {len(refs)} photos, mean pairwise cosine {ceiling:.4f} (the ceiling)")
    else:
        print("reference set: 1 photo (no ceiling to report - add more references)")

    print(f"\n{'generated':<30} {'n':>3} {'mean':>7} {'best':>7} {'worst':>7}   per image")
    for d in gen_dirs:
        scores, missed = [], 0
        for p in images(d):
            e = embed(p, tmp)
            if e is None:
                missed += 1
            else:
                scores.append(float((R @ e).mean()))
        tag = os.path.basename(d.rstrip("/")) or d
        if not scores:
            print(f"{tag:<30} {0:>3}   no detectable face in {missed} image(s)")
            continue
        note = f"   [{missed} with no detectable face]" if missed else ""
        print(f"{tag:<30} {len(scores):>3} {np.mean(scores):>7.4f} {max(scores):>7.4f} "
              f"{min(scores):>7.4f}   " + " ".join(f"{v:.3f}" for v in scores) + note)
PY
