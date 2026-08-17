// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D space-to-depth (channel-outer grouping), NCTHW contract
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32

//
// 3D space-to-depth: folds `(pt,ph,pw)`-sized blocks of the time/height/width
// axes into new, channel-OUTER groups - the LTX video-VAE encoder's
// `SpaceToDepthDownsample` rearrange (`einops`
// `'b c (d p1)(h p2)(w p3) -> b (c p1 p2 p3) d h w'`), applied both to the
// block's raw input (for its parameter-free group-mean skip) and to its own
// conv's output. This is the CONTRACT direction: channel count multiplies by
// `pt*ph*pw`, the spatial/temporal extents divide by it.
//
// Channel-OUTER is the reason this cannot reuse `pixel_shuffle.wgsl`: that
// kernel assumes batch-major NCHW (`x[n,cin,h,w]`) and has no time axis, while
// this volume is channel-major (`x[c,t,h,w]`, batch folded away) and needs the
// (p1,p2,p3) triple factored out of one combined channel axis.
//
//   x : [Cin,        T,    H,    W   ]
//   y : [Cin*pt*ph*pw, T/pt, H/ph, W/pw]   one invocation per OUTPUT element
//
//   y[((c*pt+it)*ph+ih)*pw+iw, to, ho, wo] = x[c, to*pt+it, ho*ph+ih, wo*pw+iw]
//
// `depth_to_space3d.wgsl` is the exact inverse (EXPAND direction), used by the
// decoder's `DepthToSpaceUpsample`.

struct Params {
    Cin: u32, T: u32, H: u32, W: u32,
    pt: u32, ph: u32, pw: u32,
    To: u32, Ho: u32, Wo: u32,   // = T/pt, H/ph, W/pw
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let Cout = p.Cin * p.pt * p.ph * p.pw;
    let total = Cout * p.To * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (co, to, ho, wo).
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let to = t2 % p.To;
    let co = t2 / p.To;

    // co = ((c*pt + it)*ph + ih)*pw + iw
    let iw = co % p.pw;
    let r1 = co / p.pw;
    let ih = r1 % p.ph;
    let r2 = r1 / p.ph;
    let it = r2 % p.pt;
    let c  = r2 / p.pt;

    let ti = to * p.pt + it;
    let hi = ho * p.ph + ih;
    let wi = wo * p.pw + iw;
    let xi = ((c * p.T + ti) * p.H + hi) * p.W + wi;
    y[idx] = x[xi];
}
