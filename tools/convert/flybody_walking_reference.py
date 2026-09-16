#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Convert flybody's walking-imitation HDF5 into a flat binary brain can read.

The reference trajectories for the fly's DeepMimic-style imitation reward ship
as HDF5, which would put an hdf5 dependency in the engine's load path for data
that is a few hundred megabytes of plain f32 arrays. This is the same split
`tools/` exists for everywhere else: a human runs the conversion once, and the
Rust side reads a format it can parse with no dependency at all.

WHAT IT KEEPS, AND WHY THAT IS THE WHOLE FILE

flybody's reward tracks four features (`flybody/tasks/rewards.py`). Two of them
- centre of mass and joint velocities - are reachable through MuJoCo's flat
state API, which is all `crates/mujoco` binds. The other two, egocentric
end-effector vectors and joint orientation quaternions, need site positions and
joint axes out of mjData/mjModel, which that binding deliberately does not
mirror. So this writes the first two and says so, rather than writing all four
and leaving half of them unused with no explanation.

The dataset stores the root joint separately from the other 102, and the two
concatenate to exactly the model's own state vector: 7 + 102 = nq = 109 and
6 + 102 = nv = 108. That is asserted here, because a silent mismatch would
produce a reward that tracks the wrong joints.

    pip install h5py
    tools/convert/flybody_walking_reference.py IN.hdf5 OUT.bin [--snippets N]

FORMAT (little-endian throughout)

    magic     8 bytes   "BRNFLYW1"
    timestep  f64       seconds per reference frame
    nq        u32       109, checked against the model on load
    nv        u32       108
    count     u32       number of snippets
    lengths   u32 * count
    then, per snippet, frames in order:
        qpos  f32[nq]   root_qpos (7) then the other 102
        qvel  f32[nv]   root_qvel (6) then the other 102
"""

import argparse
import struct
import sys

import h5py
import numpy as np

MAGIC = b"BRNFLYW1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--snippets", type=int, default=0, help="keep only the first N (0 = all)")
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        timestep = float(f["timestep_seconds"][()])
        names = sorted(f["trajectories"].keys())
        if args.snippets:
            names = names[: args.snippets]

        first = f["trajectories"][names[0]]
        nq = first["root_qpos"].shape[1] + first["qpos"].shape[1]
        nv = first["root_qvel"].shape[1] + first["qvel"].shape[1]
        # The model this reward is for. A reference whose joint count does not
        # match is not a reference for this fly, and every downstream number
        # would be a comparison against the wrong animal.
        assert nq == 109, f"expected nq = 109 for flybody, got {nq}"
        assert nv == 108, f"expected nv = 108 for flybody, got {nv}"

        lengths = []
        blocks = []
        for name in names:
            t = f["trajectories"][name]
            n = int(t["qpos"].shape[0])
            qpos = np.concatenate([np.asarray(t["root_qpos"]), np.asarray(t["qpos"])], axis=1)
            qvel = np.concatenate([np.asarray(t["root_qvel"]), np.asarray(t["qvel"])], axis=1)
            assert qpos.shape == (n, nq) and qvel.shape == (n, nv), f"{name}: ragged snippet"
            # Interleaved per frame, so a reader streams one frame at a time
            # instead of seeking between two arrays.
            frames = np.concatenate([qpos.astype("<f4"), qvel.astype("<f4")], axis=1)
            lengths.append(n)
            blocks.append(frames.tobytes())

    with open(args.output, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<dIII", timestep, nq, nv, len(lengths)))
        out.write(struct.pack(f"<{len(lengths)}I", *lengths))
        for b in blocks:
            out.write(b)

    total = sum(lengths)
    print(
        f"{len(lengths)} snippets, {total} frames, {total * timestep:.2f} s of walking, "
        f"nq={nq} nv={nv}, timestep {timestep * 1e3:.3f} ms -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
