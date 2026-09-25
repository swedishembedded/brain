// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  splat environment: composite the distant radiance behind a ray-evaluated render, or its alpha VJP
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import camera
// @import splat_view
// @import sh_env
//
// What a pixel sees past every gaussian: the environment's radiance along the
// pixel's own world-space ray direction, a spherical-harmonic expansion of
// `degree` (`splat::env`). The render arrives composited over black, so
//
//   C = C_splats + (1 - alpha) E(d),    E = max(0, sum_k Y_k(d) coeffs_k)
//
// mode 0 adds (1 - alpha) E into img's colour. mode 1 is the part of the
// backward the rasterizer needs: dL/dalpha_out gains -E . dL/dC, written into
// dimg's alpha in place (the coefficients' own gradient is
// `splat_env_grad.wgsl`).

struct Env {
    degree: u32,
    mode: u32,
    pad0: u32,
    pad1: u32,
};

struct Params {
    v: View,
    e: Env,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       coeffs: array<f32>; // (degree+1)^2 * 3
@group(0) @binding(2) var<storage, read_write> img:    array<f32>; // W*H*4
@group(0) @binding(3) var<storage, read_write> dimg:   array<f32>; // W*H*4

fn env_at(d: vec3<f32>) -> vec3<f32> {
    let y = sh_env_basis(d, p.e.degree);
    let n = (p.e.degree + 1u) * (p.e.degree + 1u);
    var e = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < n; k = k + 1u) {
        e = e + y[k] * vec3<f32>(coeffs[k * 3u], coeffs[k * 3u + 1u], coeffs[k * 3u + 2u]);
    }
    return max(e, vec3<f32>(0.0, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.v.width * p.v.height) { return; }
    let pr = pixel_ray(p.v, vec2<f32>(f32(idx % p.v.width) + 0.5, f32(idx / p.v.width) + 0.5));
    if (pr.ok == 0.0) { return; }
    let dw = pr.d.x * p.v.r0.xyz + pr.d.y * p.v.r1.xyz + pr.d.z * p.v.r2.xyz;
    let e = env_at(dw);
    let o = idx * 4u;
    if (p.e.mode == 0u) {
        let t = 1.0 - img[o + 3u];
        img[o] = img[o] + t * e.x;
        img[o + 1u] = img[o + 1u] + t * e.y;
        img[o + 2u] = img[o + 2u] + t * e.z;
    } else {
        dimg[o + 3u] = dimg[o + 3u] - dot(e, vec3<f32>(dimg[o], dimg[o + 1u], dimg[o + 2u]));
    }
}
