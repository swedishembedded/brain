// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  splat environment: gradient of the spherical-harmonic coefficients, as per-workgroup partial sums
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import sh_env
//
// dL/dcoeff_{k,c} = sum over pixels of (1 - alpha) [E_c > 0] Y_k(d) dL/dC_c
// (`splat_env.wgsl`'s forward). Every pixel feeds every coefficient, so
// there is nothing to scatter: workgroup w covers pixel block w % n_blocks
// (grid-strided over the frame) and the 16 coefficient entries of chunk
// w / n_blocks (entry k*3 + c), each thread summing its pixels, and after one
// barrier threads 0..15 total the block into partial[w*16 + j]. The host sums
// the blocks.

struct Grad {
    degree: u32,
    n_blocks: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    v: View,
    g: Grad,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       coeffs:  array<f32>; // (degree+1)^2 * 3
@group(0) @binding(2) var<storage, read>       img:     array<f32>; // W*H*4, alpha out in [3]
@group(0) @binding(3) var<storage, read>       dimg:    array<f32>; // W*H*4
@group(0) @binding(4) var<storage, read_write> partial: array<f32>; // n_wg*16

var<workgroup> acc: array<f32, 1024>; // 16 x 64

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wid.y * nwg.x + wid.x;
    let t = lid.x;
    let n = (p.g.degree + 1u) * (p.g.degree + 1u);
    // a grid padded past the last chunk
    if (w >= p.g.n_blocks * ((n * 3u + 15u) / 16u)) { return; }
    let block = w % p.g.n_blocks;
    let chunk = w / p.g.n_blocks;
    var s: array<f32, 16>;
    for (var j = 0u; j < 16u; j = j + 1u) { s[j] = 0.0; }
    let n_pix = p.v.width * p.v.height;
    for (var idx = block * 64u + t; idx < n_pix; idx = idx + p.g.n_blocks * 64u) {
        let pr = pixel_ray(p.v, vec2<f32>(f32(idx % p.v.width) + 0.5, f32(idx / p.v.width) + 0.5));
        if (pr.ok == 0.0) { continue; }
        let o = idx * 4u;
        let tr = 1.0 - img[o + 3u];
        let g = vec3<f32>(dimg[o], dimg[o + 1u], dimg[o + 2u]) * tr;
        if (tr <= 0.0 || dot(g, g) == 0.0) { continue; }
        let dw = pr.d.x * p.v.r0.xyz + pr.d.y * p.v.r1.xyz + pr.d.z * p.v.r2.xyz;
        let y = sh_env_basis(dw, p.g.degree);
        var e = vec3<f32>(0.0, 0.0, 0.0);
        for (var k = 0u; k < n; k = k + 1u) {
            e = e + y[k] * vec3<f32>(coeffs[k * 3u], coeffs[k * 3u + 1u], coeffs[k * 3u + 2u]);
        }
        let live = vec3<f32>(select(0.0, 1.0, e.x > 0.0), select(0.0, 1.0, e.y > 0.0), select(0.0, 1.0, e.z > 0.0));
        let gl = g * live;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let entry = chunk * 16u + j;
            if (entry < n * 3u) {
                s[j] = s[j] + y[entry / 3u] * gl[entry % 3u];
            }
        }
    }
    for (var j = 0u; j < 16u; j = j + 1u) { acc[j * 64u + t] = s[j]; }
    workgroupBarrier();
    if (t < 16u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) { total = total + acc[t * 64u + i]; }
        partial[w * 16u + t] = total;
    }
}
