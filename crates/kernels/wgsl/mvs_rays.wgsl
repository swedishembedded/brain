// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS ray map: the unit ray through every pixel centre of a view, through its lens
// @how   one thread per pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
//
// The PatchMatch cost warps every patch sample of the reference through the
// ray of that sample's pixel, and inverting a distorted lens is an iterative
// solve (`lens_unproject`), so the rays are solved once per view and level
// here and read back as three planes (x, y, z of `plane` floats each). A
// pixel the lens has no ray for (past a fisheye's field, outside a folding
// Brown lens's valid disc) gets the zero vector, which every consumer reads as
// "no pixel here".

struct Params {
    w: u32,
    h: u32,
    plane: u32,
    pad0: u32,
    lens: Lens,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> rays: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let px = vec2<f32>(f32(i % p.w) + 0.5, f32(i / p.w) + 0.5);
    let u = lens_unproject(p.lens, px);
    var d = u.xyz;
    if (u.w == 0.0) { d = vec3<f32>(0.0, 0.0, 0.0); }
    rays[i] = d.x;
    rays[p.plane + i] = d.y;
    rays[2u * p.plane + i] = d.z;
}
