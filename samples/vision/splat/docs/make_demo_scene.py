# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Write an Inria-layout PLY of a procedurally generated scene.

The README's pictures come from here, so they can be regenerated rather than
being binaries nobody can reproduce. This is a SYNTHETIC scene - it exercises
the rasterizer, not the reconstruction - which is why it lives beside the
pictures and not in the test fixtures.

brain's world is y-down: -Y is up.

Every gaussian is a FLAT DISC lying in the surface, not a ball. A surface built
from isotropic gaussians is as thick along its own normal as it is wide, which
is what makes such a scene render as fluff however many of them there are - the
eye sees through each one into the ones behind. Flattening each against the
surface it belongs to is the same thing a reconstruction does when it aligns a
splat to a measured normal, and it is the whole difference between a floor and
a fog.
"""
import math, random, struct, sys

random.seed(11)
G = []
FLAT = 0.16   # short axis as a fraction of the long ones


def quat_to(n):
    """Unit quaternion (wxyz) taking local +z onto the unit normal `n`.

    The gaussian's third scale is its short axis, so putting local z on the
    surface normal is what lays the disc in the surface.
    """
    nx, ny, nz = n
    d = nx * nx + ny * ny + nz * nz
    if d <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    k = 1.0 / math.sqrt(d)
    nx, ny, nz = nx * k, ny * k, nz * k
    # axis = z_hat x n , angle = acos(z_hat . n)
    ax, ay, az = -ny, nx, 0.0
    s = math.sqrt(ax * ax + ay * ay)
    c = max(-1.0, min(1.0, nz))
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    ang = math.acos(c)
    sh = math.sin(ang * 0.5)
    return (math.cos(ang * 0.5), ax / s * sh, ay / s * sh, az / s * sh)


def add(p, c, op, s, n):
    q = quat_to(n)
    G.append((p[0], p[1], p[2], c[0], c[1], c[2], op, s, s, s * FLAT, q[0], q[1], q[2], q[3]))


def shade(n, base, key=(0.45, -0.80, -0.40), amb=0.34):
    """Lambert against one key light, in the y-down frame."""
    k = math.sqrt(sum(v * v for v in key))
    d = sum(a * b for a, b in zip(n, key)) / k
    lit = amb + (1.0 - amb) * max(0.0, d)
    return tuple(min(1.0, c * lit) for c in base)


# --- ground: a wide checkerboard, fading out at the edges.
# A jittered LATTICE rather than uniform random points: independent samples
# clump, and a clump of translucent discs reads as a stain on what is supposed
# to be a flat painted floor.
SPAN = 11.0
STEP = 0.021
UP = (0.0, -1.0, 0.0)
n_side = int(2 * SPAN / STEP)
for iy in range(n_side):
    for ix in range(n_side):
        x = -SPAN + (ix + 0.5 + 0.35 * (random.random() - 0.5)) * STEP
        z = -SPAN + (iy + 0.5 + 0.35 * (random.random() - 0.5)) * STEP
        r = math.hypot(x, z)
        if r > SPAN:
            continue
        check = (int(x + 40) + int(z + 40)) % 2
        t = (0.46 if check else 0.28) + 0.015 * random.random()
        # Opaque across the floor and soft only at the rim: a surface that is
        # translucent everywhere is the fluff this scene exists to avoid, but
        # one that simply stops at a circle draws a hard edge across the sky.
        fade = max(0.0, 1.0 - (r / SPAN) ** 8)
        add((x, 0.0, z), (t * 0.97, t, t * 1.06), 0.02 + 0.97 * fade, 0.016, UP)

# --- a torus standing upright, warm orange
for _ in range(90000):
    u = random.random() * math.tau
    v = random.random() * math.tau
    R, rr = 1.00, 0.30
    cx, cy = math.cos(u), math.sin(u)
    nx, ny, nz = cx * math.cos(v), cy * math.cos(v), math.sin(v)
    x = (R + rr * math.cos(v)) * cx
    y = (R + rr * math.cos(v)) * cy
    z = rr * math.sin(v)
    add((x, y - 1.15, z), shade((nx, ny, nz), (0.98, 0.56, 0.16)), 0.99, 0.011, (nx, ny, nz))

# --- a sphere on the ground, cool blue
for _ in range(52000):
    th = math.acos(1 - 2 * random.random())
    ph = random.random() * math.tau
    r = 0.62
    nx, ny, nz = math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)
    add((r * nx + 2.15, r * ny - 0.62, r * nz + 0.65),
        shade((nx, ny, nz), (0.20, 0.52, 0.92)), 0.99, 0.011, (nx, ny, nz))

# --- a tapered column, pale stone
for _ in range(36000):
    ph = random.random() * math.tau
    h = random.random()
    r = 0.19 * (1.0 - 0.30 * h)
    nx, nz = math.cos(ph), math.sin(ph)
    add((r * nx - 2.25, -h * 2.05, r * nz + 0.30),
        shade((nx, 0.0, nz), (0.86, 0.85, 0.80)), 0.99, 0.010, (nx, 0.0, nz))

SH_C0 = 0.28209479177387814


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


props = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
]
out = sys.argv[1]
with open(out, "wb") as f:
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(G)
    hdr += "".join("property float %s\n" % p for p in props) + "end_header\n"
    f.write(hdr.encode())
    for (x, y, z, r, g, b, op, sx, sy, sz, qw, qx, qy, qz) in G:
        rec = [
            x, y, z, 0.0, 0.0, 0.0,
            (r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0,
            logit(op), math.log(sx), math.log(sy), math.log(sz),
            qw, qx, qy, qz,
        ]
        f.write(struct.pack("<%df" % len(rec), *rec))
print("%s: %d gaussians" % (out, len(G)))
