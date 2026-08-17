#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 schedule reference goldens (pure math, no weights needed).

Runs the OFFICIAL `ltx_core.components.schedulers.LTX2Scheduler` and dumps its
sigma vectors for a spread of `(token_count, steps)` combinations, plus the
hardcoded distilled-8-step sigma constants read directly out of
`ltx-pipelines`' own source file (not transcribed from the roadmap doc, per
the milestone instructions - that doc's copy is unverified against a source
line).

  schedule.safetensors   per (tokens, steps, base_shift, max_shift, stretch):
                         the sigma vector; plus the three DISTILLED_* constant
                         vectors verbatim
  manifest.json          shapes, sha256, run params, library versions

## Self-validation

`LTX2Scheduler.execute` is deterministic pure math (Flux-style token-count
shift + optional terminal stretch, see the module docstring in
`ltx_core/components/schedulers.py`). Two independent checks stand in for a
ground truth comparison:

1. **Closed-form reimplementation**: `expected_sigmas` below re-derives the
   same formula from the paper/docstring description independently (no
   shared code with `LTX2Scheduler.execute`) and is asserted to agree exactly
   (this is fp64 numpy vs fp32 torch, so the tolerance is a few ULPs, not
   "close").
2. **Exact reproducibility**: every case is run twice through the actual
   scheduler and asserted bit-identical.

Usage:
  python tools/goldens/ltxv_schedule_dump_reference.py [--out testdata/golden/ltxv/schedule]
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# The reference clone this port validates against - `LTXV_REFERENCE_ROOT`
# overrides for a checkout elsewhere; the default is repo-relative
# (`scratchpad/reference/ltxv/`, gitignored, populated per that dir's own
# README/roadmap doc), never a machine-specific absolute path.
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.components.schedulers import (  # noqa: E402
    BASE_SHIFT_ANCHOR,
    MAX_SHIFT_ANCHOR,
    LTX2Scheduler,
)


def _load_constants_by_path(path):
    """Import `ltx_pipelines/utils/constants.py` BY FILE PATH, never
    `import ltx_pipelines.utils` - that package's `__init__.py` drags in the
    whole media-io/color stack (needs `av`, not installed here). The module's
    own imports are absolute `ltx_core.*`, which resolve fine off sys.path
    without the `ltx_pipelines` package ever being imported."""
    spec = importlib.util.spec_from_file_location("ltx_pipelines_constants", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ltx_pipelines_constants"] = mod
    spec.loader.exec_module(mod)
    return mod


_DEFAULT_CONSTANTS_PATH = str(
    _REFERENCE_ROOT / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "utils" / "constants.py")

# (token_count, steps, base_shift, max_shift, stretch, terminal). The default
# base_shift=0.95/max_shift=2.05/terminal=0.1 come from `LTX2Scheduler.execute`'s
# own signature defaults (the values a pipeline gets if it never overrides them).
# token_count spans well below BASE_SHIFT_ANCHOR (1024), exactly at it, between
# the two anchors, exactly at MAX_SHIFT_ANCHOR (4096), and above it, which is
# what proves the shift is a FORMULA (linear interpolation extrapolated outside
# [1024, 4096]) and not a lookup table clamped to the anchors.
CASES = [
    (256, 8, 0.95, 2.05, True, 0.1),
    (1024, 8, 0.95, 2.05, True, 0.1),
    (1024, 20, 0.95, 2.05, True, 0.1),
    (2048, 20, 0.95, 2.05, True, 0.1),
    (4096, 8, 0.95, 2.05, True, 0.1),
    (4096, 20, 0.95, 2.05, True, 0.1),
    (4096, 50, 0.95, 2.05, True, 0.1),
    (8192, 20, 0.95, 2.05, True, 0.1),
    # no-stretch variant, and a non-default shift pair, spanning the same anchors
    (4096, 20, 0.95, 2.05, False, 0.1),
    (1024, 20, 0.5, 3.0, True, 0.05),
]
SEED = 1234


def save(out, name, tensors, manifest):
    # everything as f32 - brain's safetensors reader is F32/F16/BF16-only
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: {len(tensors)} tensors", flush=True)


def expected_sigmas(tokens, steps, base_shift, max_shift, stretch, terminal):
    """Independent (numpy, fp64) reimplementation of `LTX2Scheduler.execute`,
    from the formula rather than by calling the class - one half of the
    self-validation."""
    sigmas = np.linspace(1.0, 0.0, steps + 1)
    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = tokens * mm + b

    out = np.zeros_like(sigmas)
    nz = sigmas != 0
    out[nz] = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas[nz] - 1.0))

    if stretch:
        non_zero_mask = out != 0
        non_zero = out[non_zero_mask]
        one_minus_z = 1.0 - non_zero
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        out[non_zero_mask] = stretched

    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", default=_DEFAULT_CONSTANTS_PATH,
                    help="ltx_pipelines/utils/constants.py (source of DISTILLED_SIGMA_VALUES)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.environ.get("BRAIN_TESTDATA") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata"),
        "golden", "ltxv", "schedule")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SEED)

    constants = _load_constants_by_path(args.constants)
    distilled_sigma_values = constants.DISTILLED_SIGMA_VALUES
    stage_2_distilled_sigma_values = constants.STAGE_2_DISTILLED_SIGMA_VALUES
    tdp_distilled_sigmas = constants.TDP_DISTILLED_SIGMAS

    scheduler = LTX2Scheduler()
    store = {}
    manifest_cases = []

    for tokens, steps, base_shift, max_shift, stretch, terminal in CASES:
        # `latent.shape[2:]` product is what `execute` reads as the token
        # count - any tensor shape whose trailing dims multiply to `tokens`
        # works; a real pipeline latent is (B, C, F, H, W).
        latent = torch.zeros(1, 1, tokens)
        sigmas = scheduler.execute(steps, latent=latent, max_shift=max_shift,
                                   base_shift=base_shift, stretch=stretch,
                                   terminal=terminal)

        # ---- self-validation 1: independent closed-form reimplementation --
        want = expected_sigmas(tokens, steps, base_shift, max_shift, stretch, terminal)
        d = np.abs(sigmas.numpy() - want).max()
        print(f"tokens={tokens} steps={steps} base_shift={base_shift} max_shift={max_shift} "
              f"stretch={stretch} terminal={terminal}: closed-form max abs diff {d:.3e}", flush=True)
        assert d < 1e-6, f"tokens={tokens} steps={steps}: formula mismatch ({d:.3e})"

        # ---- self-validation 2: exact reproducibility ----------------------
        sigmas2 = scheduler.execute(steps, latent=latent, max_shift=max_shift,
                                    base_shift=base_shift, stretch=stretch,
                                    terminal=terminal)
        assert torch.equal(sigmas, sigmas2), f"tokens={tokens} steps={steps}: not reproducible"

        pfx = f"tokens{tokens}.steps{steps}.base{base_shift:g}.max{max_shift:g}.stretch{int(stretch)}.term{terminal:g}"
        store[f"{pfx}.sigmas"] = sigmas
        manifest_cases.append({"tokens": tokens, "steps": steps, "base_shift": base_shift,
                               "max_shift": max_shift, "stretch": stretch, "terminal": terminal,
                               "key": f"{pfx}.sigmas"})

    # ---- the hardcoded distilled-8-step sigma constants, read from source --
    store["distilled_8step.sigmas"] = torch.tensor(distilled_sigma_values, dtype=torch.float32)
    store["distilled_stage2.sigmas"] = torch.tensor(stage_2_distilled_sigma_values, dtype=torch.float32)
    store["distilled_tdp.sigmas"] = tdp_distilled_sigmas.to(torch.float32)
    assert list(store["distilled_8step.sigmas"].shape) == [9], store["distilled_8step.sigmas"].shape
    print(f"DISTILLED_SIGMA_VALUES ({len(distilled_sigma_values)} steps): {distilled_sigma_values}", flush=True)
    print(f"STAGE_2_DISTILLED_SIGMA_VALUES: {stage_2_distilled_sigma_values}", flush=True)
    print(f"TDP_DISTILLED_SIGMAS: {tdp_distilled_sigmas.tolist()}", flush=True)

    manifest = {
        "params": {
            "cases": manifest_cases, "seed": SEED,
            "base_shift_anchor": BASE_SHIFT_ANCHOR, "max_shift_anchor": MAX_SHIFT_ANCHOR,
            "distilled_sigma_values": distilled_sigma_values,
            "stage_2_distilled_sigma_values": stage_2_distilled_sigma_values,
            "tdp_distilled_sigmas": tdp_distilled_sigmas.tolist(),
            "source": args.constants,
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
    }
    save(out, "schedule.safetensors", store, manifest)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {out}/manifest.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
