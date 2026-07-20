#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Regenerate crates/splat/tests/golden/gradcheck.json — the float64
torch-autograd oracle for the rasterizer-backward gradcheck.

The scene/loss replicate crates/splat/tests/s5_bwd_fit.rs bit-for-bit (same
LCG). Finite differences are deliberately NOT used as the oracle: the 1/255
truncation boundary makes them biased for scale/mean gradients.

Run from the repo root:  python3 tools/splat_dump_gradcheck.py
"""
import torch, json
torch.set_default_dtype(torch.float64)

class Lcg:
    def __init__(s, seed): s.v = seed
    def next(s):
        s.v = (s.v * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (s.v >> 33) / float(1 << 31)

r = Lcg(0x5eed)
means, quats, scales, ops, cols = [], [], [], [], []
for _ in range(6):
    means.append([(r.next()-0.5)*2, (r.next()-0.5)*2, 3.0+r.next()*2])
    quats.append([0.5+r.next(), r.next()-0.5, r.next()-0.5, r.next()-0.5])
    scales.append([0.1+r.next()*0.15 for _ in range(3)])
    ops.append(0.35+0.5*r.next())
    cols.append([r.next(), r.next(), r.next()])
means = torch.tensor(means, requires_grad=True)
quats = torch.tensor(quats, requires_grad=True)
scales = torch.tensor(scales, requires_grad=True)
ops = torch.tensor(ops, requires_grad=True)
cols = torch.tensor(cols, requires_grad=True)
W = H = 32
fy = 0.5*H/torch.tan(torch.tensor(30.0*torch.pi/180))
fx, cx, cy = fy, W/2, H/2
eps2d = 0.3

def rotm(q):
    q = q/q.norm(); w, x, y, z = q
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]),
    ])

gs = []
for i in range(6):
    m = means[i]; z = m[2]
    M = rotm(quats[i]) @ torch.diag(scales[i])
    Sc = M @ M.T  # camera at origin looking +z: R = I
    rz = 1/z
    tan_fovx = 0.5*W/fx
    limx = cx/fx + 0.3*tan_fovx
    txc = z*torch.clamp(m[0]*rz, -limx, (W-cx)/fx+0.3*tan_fovx)
    tyc = z*torch.clamp(m[1]*rz, -limx, (H-cy)/fy+0.3*tan_fovx)
    J = torch.stack([
        torch.stack([fx*rz, torch.tensor(0.0), -fx*txc*rz*rz]),
        torch.stack([torch.tensor(0.0), fy*rz, -fy*tyc*rz*rz]),
    ])
    S2 = J @ Sc @ J.T + eps2d*torch.eye(2)
    C = torch.linalg.inv(S2)
    mean2d = torch.stack([fx*m[0]*rz + cx, fy*m[1]*rz + cy])
    gs.append((z, mean2d, C, ops[i], cols[i]))
order = sorted(range(6), key=lambda i: gs[i][0].item())

r2 = Lcg(0xabcd)
wimg = [[(r2.next()-0.5) if c < 3 else 0.0 for c in range(4)] for _ in range(W*H)]
L = torch.tensor(0.0)
for py in range(H):
    for px in range(W):
        T = torch.tensor(1.0); color = torch.zeros(3)
        for i in order:
            z, m2, C, op, col = gs[i]
            d = m2 - torch.tensor([px+0.5, py+0.5])
            sigma = 0.5*(C[0, 0]*d[0]*d[0] + C[1, 1]*d[1]*d[1]) + C[0, 1]*d[0]*d[1]
            if sigma.item() < 0: continue
            alpha = torch.minimum(torch.tensor(0.99), op*torch.exp(-sigma))
            if alpha.item() < 1/255: continue
            nt = T*(1-alpha)
            if nt.item() <= 1e-4: break
            color = color + col*alpha*T
            T = nt
        w = wimg[py*W+px]
        L = L + color[0]*w[0] + color[1]*w[1] + color[2]*w[2]
L.backward()
golden = {
    "d_means": means.grad.flatten().tolist(),
    "d_scales": scales.grad.flatten().tolist(),
    "d_quats": quats.grad.flatten().tolist(),
    "d_opac": ops.grad.tolist(),
    "d_colors": cols.grad.flatten().tolist(),
}
json.dump(golden, open("crates/splat/tests/golden/gradcheck.json", "w"), indent=0)
print("wrote crates/splat/tests/golden/gradcheck.json")
