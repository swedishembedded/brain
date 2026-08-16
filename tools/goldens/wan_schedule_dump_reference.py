#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Wan2.1 sampler goldens (schedule + solver trajectory) for brain.

Needs NO model weights: it imports only the two scheduler classes out of the
Wan2.1 checkout and drives them with a deterministic pseudo-denoiser.

  schedule.safetensors  per (solver, shift, steps): sigmas, timesteps and a
                        full `step()` trajectory over a fixed pseudo-denoiser
  manifest.json         shapes, sha256, run parameters, library versions

The two solvers do NOT share a schedule, which is the reason both are dumped:
`unipc` starts from the training grid's top sigma (`1 - 1/1000 = 0.999`, so the
first timestep is 999) while `dpm++` builds its own `linspace(1, 0, n+1)[:n]`
in `get_sampling_sigmas` and starts at exactly 1.0 (first timestep 1000). A
port that shares one sigma vector between them is wrong for one of them.

Usage:
  python tools/goldens/wan_schedule_dump_reference.py \
      [--wan <Wan2.1 checkout>] [--out testdata/golden/wan]
"""

import argparse, hashlib, importlib.util, json, os, sys

import numpy as np
import torch
from safetensors.torch import save_file

# (shift, steps). 5.0/50 is the T2V default and 3.0/40 the I2V-480p default;
# the rest exist so the test proves the shift FORMULA rather than a lookup
# table, and so the small-step cases exercise the multistep warmup and the
# `lower_order_final` drop with the schedule that triggers them.
CASES = [
    (5.0, 50), (3.0, 40), (5.0, 40), (3.0, 50),
    (7.5, 25), (16.0, 50), (1.0, 10), (5.0, 4),
]
SEED = 1234
# The solvers einsum over "b k c ..." internally, so the fixed trajectory has to
# carry a real (batch, channel, rest) shape rather than a flat vector.
SHAPE = (1, 4, 24)
NUM_TRAIN = 1000


def load(name, path):
    """Import a single Wan module by path - `import wan.utils...` drags in
    `wan/__init__.py`, which pulls the whole model stack (and easydict)."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save(out, name, tensors, manifest):
    # everything as f32 - brain's safetensors reader is F32/F16/BF16-only, and
    # the int64 timesteps (<= 1000) are exactly representable
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: {len(tensors)} tensors", flush=True)


def pseudo(i, x):
    """Deterministic stand-in for the denoiser: a pure function of (step, x) so
    the trajectory is reproducible from the goldens alone."""
    return torch.sin(x * (0.7 + 0.1 * i)) * 0.9 - 0.05 * i


def expected_sigmas(solver, shift, steps):
    """Independent reimplementation of the sigma schedule, from the formula
    rather than from the scheduler objects - one half of the self-validation."""
    if solver == "unipc":
        # sigma_max is the top of the training grid AFTER the f32 round-trip
        # the constructor does, not 1.0.
        sigma_max = float(np.float32(1.0 - 1.0 / NUM_TRAIN))
        base = np.linspace(sigma_max, 0.0, steps + 1)[:-1]
    else:
        base = np.linspace(1.0, 0.0, steps + 1)[:steps]
    sig = shift * base / (1 + (shift - 1) * base)
    timesteps = (sig * NUM_TRAIN).astype(np.int64)
    sig = np.concatenate([sig, [0.0]]).astype(np.float32)
    return sig, timesteps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "scratchpad", "reference", "wan", "Wan2.1"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.environ.get("BRAIN_TESTDATA") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata"),
        "golden", "wan")
    os.makedirs(out, exist_ok=True)

    utils = os.path.join(args.wan, "wan", "utils")
    U = load("fm_solvers_unipc", os.path.join(utils, "fm_solvers_unipc.py"))
    D = load("fm_solvers", os.path.join(utils, "fm_solvers.py"))

    g = torch.Generator().manual_seed(SEED)
    x0 = torch.randn(SHAPE, generator=g)
    store = {"traj.x0": x0}
    manifest = {}
    worst_euler = 0.0

    for solver in ("unipc", "dpmpp"):
        for shift, steps in CASES:
            # Constructed exactly as `wan/text2video.py` does it: shift=1 in the
            # constructor, the real shift passed to set_timesteps (unipc) or
            # baked into the supplied sigmas (dpm++). Passing it twice would
            # square the shift.
            if solver == "unipc":
                s = U.FlowUniPCMultistepScheduler(
                    num_train_timesteps=NUM_TRAIN, shift=1,
                    use_dynamic_shifting=False)
                s.set_timesteps(steps, device="cpu", shift=shift)
            else:
                s = D.FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=NUM_TRAIN, shift=1,
                    use_dynamic_shifting=False)
                D.retrieve_timesteps(s, device="cpu",
                                     sigmas=D.get_sampling_sigmas(steps, shift))

            pfx = f"{solver}.shift{shift:g}.steps{steps}"
            store[f"{pfx}.sigmas"] = s.sigmas.to(torch.float32).cpu()
            store[f"{pfx}.timesteps"] = s.timesteps.to(torch.float32).cpu()

            # -- self-validation 1: the formula, recomputed independently -----
            want_s, want_t = expected_sigmas(solver, shift, steps)
            assert np.array_equal(s.sigmas.cpu().numpy(), want_s), \
                f"{pfx}: sigmas differ from the closed form"
            assert np.array_equal(s.timesteps.cpu().numpy(), want_t), \
                f"{pfx}: timesteps differ from the closed form"

            x = x0.clone()
            traj = []
            for i, t in enumerate(s.timesteps):
                m = pseudo(i, x)
                nxt = s.step(m, t, x, return_dict=False)[0]
                if i == 0:
                    # -- self-validation 2 ---------------------------------
                    # Both solvers reduce, on their first step (order 1, no
                    # corrector yet), to the plain flow-matching Euler step
                    # x + (sigma_next - sigma)*v. That closed form is derived
                    # by hand and shares no code with the B(h)/DPM machinery
                    # above, so agreement is a real cross-check of the sigma
                    # indexing and of the x0 conversion.
                    euler = x + (s.sigmas[1] - s.sigmas[0]) * m
                    d = (nxt - euler).abs().max().item()
                    worst_euler = max(worst_euler, d)
                    assert d < 1e-5, f"{pfx}: first step is not Euler ({d:.3e})"
                traj.append(nxt.clone())
                x = nxt
            store[f"{pfx}.traj"] = torch.stack(traj)

    print(f"first-step vs hand-derived Euler: max abs {worst_euler:.3e}", flush=True)
    save(out, "schedule.safetensors", store, manifest)
    manifest["params"] = {
        "cases": [list(c) for c in CASES], "seed": SEED, "shape": list(SHAPE),
        "num_train_timesteps": NUM_TRAIN,
        "solvers": {"unipc": "FlowUniPCMultistepScheduler(bh2, order 2, predict_x0)",
                    "dpmpp": "FlowDPMSolverMultistepScheduler(dpmsolver++, midpoint, order 2)"},
        "wan": os.path.abspath(args.wan),
        "torch": torch.__version__, "numpy": np.__version__,
        "diffusers": __import__("diffusers").__version__,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
