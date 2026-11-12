// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The sky beyond the tank's open back: stars and a faint nebula at
// INFINITY. Drawn first as a full-screen triangle at the far plane; the ray
// for each pixel comes from the inverse view-projection, so the sky shifts
// only with viewing DIRECTION - it shows no parallax, which is what makes
// everything nearer read as near.

struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    let ndc = uv * 2.0 - 1.0;
    var o: SkyOut;
    o.clip = vec4<f32>(ndc, 1.0, 1.0);
    o.ndc = ndc;
    return o;
}

fn star_layer(d: vec3<f32>, scale: f32, density: f32, t: f32) -> vec3<f32> {
    // longitude/latitude cells looking down -Z
    let lon = atan2(d.x, -d.z);
    let lat = asin(clamp(d.y, -1.0, 1.0));
    let g = vec2<f32>(lon, lat) * scale;
    let cell = vec2<i32>(floor(g));
    var c = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let id = cell + vec2<i32>(dx, dy);
            let h = hash3(vec3<i32>(id, i32(scale)));
            if (h > density) {
                let o = vec2<f32>(hash3(vec3<i32>(id, 7)), hash3(vec3<i32>(id, 13)));
                let r = length(g - (vec2<f32>(id) + o));
                let b = pow((h - density) / (1.0 - density), 2.0);
                let tw = 0.65 + 0.35 * sin(t * (1.0 + 3.0 * o.x) + 40.0 * o.y);
                let warm = mix(vec3<f32>(0.6, 0.75, 1.0), vec3<f32>(1.0, 0.85, 0.6), o.y);
                c = c + warm * b * tw * exp(-r * r * 90.0) * 6.0;
            }
        }
    }
    return c;
}

@fragment
fn fs_sky(v: SkyOut) -> @location(0) vec4<f32> {
    let far = frame.inv_view_proj * vec4<f32>(v.ndc, 1.0, 1.0);
    let d = normalize(far.xyz / far.w - frame.eye.xyz);
    let t = frame.eye.w;
    let neb = fbm(d * 2.2 + vec3<f32>(0.0, 0.0, t * 0.01));
    let wisps = fbm(d * 5.0 - vec3<f32>(t * 0.02, 0.0, 0.0));
    var c = mix(vec3<f32>(0.004, 0.002, 0.012), vec3<f32>(0.05, 0.015, 0.09), smoothstep(0.35, 0.8, neb));
    c = c + vec3<f32>(0.0, 0.05, 0.07) * smoothstep(0.55, 0.9, wisps) * neb;
    c = c + star_layer(d, 90.0, 0.90, t) + star_layer(d, 220.0, 0.93, t) * 0.6;
    return vec4<f32>(c, 1.0);
}
