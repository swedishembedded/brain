// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Binary Spherical Quantization (Kronos), inference form
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
//
// Binary Spherical Quantization (Kronos), inference form. The reference
// L2-normalizes the k-dim latent then takes the sign — but positive scaling
// preserves the sign, so the normalize is irrelevant to the quantized value:
//   zq[i] = sign(z[i]) * (1/sqrt(k))
// where sign follows torch.where(z>0, +1, -1) (i.e. z==0 -> -1). In place.
// One invocation per element; `inv_sqrt_k = 1/sqrt(codebook_dim)`.

struct Params {
    total: u32,
    inv_sqrt_k: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.total) { return; }
    let v = buf[gidx];
    let s = select(-1.0, 1.0, v > 0.0);   // z>0 -> +1 else -1
    buf[gidx] = s * p.inv_sqrt_k;
}
