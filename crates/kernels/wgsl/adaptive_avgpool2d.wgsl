// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// torch adaptive_avg_pool2d (forward): average a variable [N,C,H,W] map into a
// fixed [N,C,OH,OW] grid. Bin (oh,ow) spans input rows [floor(oh·H/OH),
// ceil((oh+1)·H/OH)) × cols [floor(ow·W/OW), ceil((ow+1)·W/OW)) — bins may overlap
// when H/OH is non-integer, exactly like torch. One invocation per OUTPUT element.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    OH: u32,
    OW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.N * p.C * p.OH * p.OW) { return; }
    let ow = idx % p.OW;
    let oh = (idx / p.OW) % p.OH;
    let c = (idx / (p.OW * p.OH)) % p.C;
    let n = idx / (p.OW * p.OH * p.C);

    let sh = (oh * p.H) / p.OH;
    let eh = ((oh + 1u) * p.H + p.OH - 1u) / p.OH;
    let sw = (ow * p.W) / p.OW;
    let ew = ((ow + 1u) * p.W + p.OW - 1u) / p.OW;

    let plane = (n * p.C + c) * p.H * p.W;
    var acc = 0.0;
    for (var ih = sh; ih < eh; ih = ih + 1u) {
        for (var iw = sw; iw < ew; iw = iw + 1u) {
            acc = acc + x[plane + ih * p.W + iw];
        }
    }
    let count = f32((eh - sh) * (ew - sw));
    y[idx] = acc / count;
}
