// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS pyramid step: plane hypotheses of a level -> the next finer level, edge-aware
// @how   one thread per fine pixel, four parent candidates
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import mvs
//
// Coarse-to-fine hand-over of the multi-scale scheme (Xu & Tao, CVPR 2019,
// §3.3). A fine pixel takes the plane of one of the four coarse pixels whose
// centres surround its own - the one whose gray value is closest to its own
// fine gray value (a joint, colour-guided choice in the spirit of joint
// bilateral upsampling, Kopf et al., SIGGRAPH 2007), so a hypothesis never
// leaks across an intensity edge, and planes are never averaged. The plane is
// re-intersected with the fine pixel's OWN ray, which is exact for any lens:
// a coarse pixel centre (i + 0.5, j + 0.5) is the fine continuous coordinate
// (2i + 1, 2j + 1).
//
// parent: the coarse level's five state planes (range, normal, cost);
// pgray/cgray: coarse and fine gray; rays: the fine ray map; child: the fine
// state, written in full.

struct Params {
    w: u32,
    h: u32,
    plane: u32,
    pw: u32,
    ph: u32,
    pplane: u32,
    pad0: u32,
    pad1: u32,
    lens: Lens,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       parent: array<f32>;
@group(0) @binding(2) var<storage, read>       pgray:  array<f32>;
@group(0) @binding(3) var<storage, read>       cgray:  array<f32>;
@group(0) @binding(4) var<storage, read>       rays:   array<f32>;
@group(0) @binding(5) var<storage, read_write> child:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let x = i % p.w;
    let y = i / p.w;
    let pl = p.plane;
    let ppl = p.pplane;
    let d = vec3<f32>(rays[i], rays[pl + i], rays[2u * pl + i]);
    child[i] = 0.0;
    child[pl + i] = 0.0;
    child[2u * pl + i] = 0.0;
    child[3u * pl + i] = 0.0;
    child[4u * pl + i] = 2.0;
    if (dot(d, d) == 0.0) { return; }
    // parent coordinates of this pixel's centre, and the 2x2 block of parent
    // centres around it
    let cx = (f32(x) + 0.5) * 0.5 - 0.5;
    let cy = (f32(y) + 0.5) * 0.5 - 0.5;
    let x0 = i32(floor(cx));
    let y0 = i32(floor(cy));
    let g = cgray[i];
    var best = 1e30;
    var bj = 0u;
    var found = false;
    for (var k = 0; k < 4; k = k + 1) {
        let qx = clamp(x0 + (k & 1), 0, i32(p.pw) - 1);
        let qy = clamp(y0 + (k >> 1), 0, i32(p.ph) - 1);
        let j = u32(qy) * p.pw + u32(qx);
        if (parent[j] <= 0.0) { continue; }
        // colour distance first, spatial distance breaks ties
        let dist = abs(pgray[j] - g) + 1e-3 * length(vec2<f32>(f32(qx) - cx, f32(qy) - cy));
        if (dist < best) {
            best = dist;
            bj = j;
            found = true;
        }
    }
    if (!found) { return; }
    let n = vec3<f32>(parent[ppl + bj], parent[2u * ppl + bj], parent[3u * ppl + bj]);
    let pd = lens_unproject(p.lens, vec2<f32>(2.0 * f32(bj % p.pw) + 1.0, 2.0 * f32(bj / p.pw) + 1.0));
    var r = parent[bj];
    if (pd.w != 0.0) {
        let rr = mvs_plane_range(n, parent[bj] * pd.xyz, d);
        if (rr > 0.0) { r = rr; }
    }
    child[i] = r;
    child[pl + i] = n.x;
    child[2u * pl + i] = n.y;
    child[3u * pl + i] = n.z;
    child[4u * pl + i] = parent[4u * ppl + bj];
}
