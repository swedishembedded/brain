// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D depth-to-space (channel-outer grouping), NCTHW expand
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32

//
// 3D depth-to-space: the exact inverse of `space_to_depth3d.wgsl` - unfolds a
// channel-OUTER `(pt,ph,pw)` group back into the time/height/width axes. The
// LTX video-VAE decoder's `DepthToSpaceUpsample` rearrange (`einops`
// `'b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)'`). EXPAND direction:
// channel count divides by `pt*ph*pw`, the spatial/temporal extents multiply
// by it.
//
//   x : [Cin,        T,    H,    W   ]
//   y : [Cin/(pt*ph*pw), T*pt, H*ph, W*pw]   one invocation per OUTPUT element
//
//   y[c, t*pt+it, h*ph+ih, w*pw+iw] = x[((c*pt+it)*ph+ih)*pw+iw, t, h, w]

struct Params {
    Cin: u32, T: u32, H: u32, W: u32,
    pt: u32, ph: u32, pw: u32,
    Cout: u32,   // = Cin / (pt*ph*pw)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let To = p.T * p.pt;
    let Ho = p.H * p.ph;
    let Wo = p.W * p.pw;
    let total = p.Cout * To * Ho * Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (c, to, ho, wo).
    let wo = idx % Wo;
    let t1 = idx / Wo;
    let ho = t1 % Ho;
    let t2 = t1 / Ho;
    let to = t2 % To;
    let c  = t2 / To;

    let iw = wo % p.pw;
    let w  = wo / p.pw;
    let ih = ho % p.ph;
    let h  = ho / p.ph;
    let it = to % p.pt;
    let t  = to / p.pt;

    let cin = ((c * p.pt + it) * p.ph + ih) * p.pw + iw;
    let xi = ((cin * p.T + t) * p.H + h) * p.W + w;
    y[idx] = x[xi];
}
