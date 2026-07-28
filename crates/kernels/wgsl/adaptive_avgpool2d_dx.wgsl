// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// adaptive_avg_pool2d backward: scatter each output bin's gradient (divided by
// its region size) back to the input pixels it covered. One invocation per INPUT
// element (n,c,h,w); loops the OH×OW bins and accumulates from those containing
// this pixel (no atomics — each input element sums its own contributions).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    OH: u32,
    OW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.N * p.C * p.H * p.W) { return; }
    let w = idx % p.W;
    let h = (idx / p.W) % p.H;
    let c = (idx / (p.W * p.H)) % p.C;
    let n = idx / (p.W * p.H * p.C);

    let out_plane = (n * p.C + c) * p.OH * p.OW;
    var acc = 0.0;
    for (var oh = 0u; oh < p.OH; oh = oh + 1u) {
        let sh = (oh * p.H) / p.OH;
        let eh = ((oh + 1u) * p.H + p.OH - 1u) / p.OH;
        if (h >= sh && h < eh) {
            for (var ow = 0u; ow < p.OW; ow = ow + 1u) {
                let sw = (ow * p.W) / p.OW;
                let ew = ((ow + 1u) * p.W + p.OW - 1u) / p.OW;
                if (w >= sw && w < ew) {
                    let count = f32((eh - sh) * (ew - sw));
                    acc = acc + dy[out_plane + oh * p.OW + ow] / count;
                }
            }
        }
    }
    dx[idx] = acc;
}
