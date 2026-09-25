// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The per-view uniform of the ray-evaluated splat renderer, and the ray every
// pixel of that view casts. Imported after `camera` by the splat_ray_*
// kernels, so the forward, the backward and the camera gradient agree on
// exactly one definition of where a pixel looks.
//
// Rolling shutter: a row is exposed at time tau = v / H - 1/2 of the frame
// (v the row's continuous coordinate), and the camera at tau is the mid-frame
// camera moved by the twist (rs_w, rs_v) scaled by tau, in its own frame:
// rotation Exp(tau rs_w), translation tau rs_v. The renderer works in the
// MID-frame camera, so a pixel's ray starts at tau rs_v and points along
// Exp(tau rs_w) d, where d is the lens's ray through that pixel. A global
// shutter is rs_w = rs_v = 0 and every ray starts at the camera centre.

struct View {
    n: u32,
    width: u32,
    height: u32,
    flags: u32,    // bit 0: 2D Mip compensation; bit 1: rolling shutter; bit 2: wrap x (equirect)
    near: f32,
    far: f32,
    eps2d: f32,    // screen-space filter variance, pixels^2
    kbuf: u32,     // reserved
    r0: vec4<f32>, // world-to-camera rows [R | t]
    r1: vec4<f32>,
    r2: vec4<f32>,
    rs_w: vec4<f32>,
    rs_v: vec4<f32>,
    bg: vec4<f32>,
    lens: Lens,
};

// A pixel's ray in the mid-frame camera: origin, unit direction, the time it
// was exposed at, and whether the lens has a ray there at all.
struct PixelRay {
    o: vec3<f32>,
    d: vec3<f32>,
    tau: f32,
    ok: f32,
};

fn view_rs(v: View) -> bool {
    return (v.flags & 2u) != 0u;
}

// Exp of a rotation vector (Rodrigues), applied to `x`.
fn rot_exp_apply(w: vec3<f32>, x: vec3<f32>) -> vec3<f32> {
    let th2 = dot(w, w);
    if (th2 < 1e-16) {
        return x + cross(w, x);
    }
    let th = sqrt(th2);
    let k = w / th;
    let c = cos(th);
    let s = sin(th);
    return x * c + cross(k, x) * s + k * dot(k, x) * (1.0 - c);
}

fn pixel_ray(v: View, px: vec2<f32>) -> PixelRay {
    var r: PixelRay;
    let u = lens_unproject(v.lens, px);
    r.ok = u.w;
    r.d = u.xyz;
    r.o = vec3<f32>(0.0, 0.0, 0.0);
    r.tau = 0.0;
    if (view_rs(v)) {
        r.tau = px.y / f32(v.height) - 0.5;
        r.o = r.tau * v.rs_v.xyz;
        r.d = rot_exp_apply(r.tau * v.rs_w.xyz, r.d);
    }
    return r;
}

fn view_to_camera(v: View, x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v.r0.xyz, x) + v.r0.w, dot(v.r1.xyz, x) + v.r1.w, dot(v.r2.xyz, x) + v.r2.w);
}
