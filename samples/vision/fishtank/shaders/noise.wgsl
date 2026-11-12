// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Integer hashing and value noise, prepended to every shader module by the
// renderer (WGSL has no include). An integer hash rather than the
// `fract(sin(x))` idiom: the latter loses its bits differently on every GPU,
// and the starfield would differ by vendor.

fn hash_u(x: u32) -> u32 {
    var v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}

fn hash3(p: vec3<i32>) -> f32 {
    let h = hash_u(u32(p.x) ^ hash_u(u32(p.y) ^ hash_u(u32(p.z))));
    return f32(h) / 4294967295.0;
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i);
    let n100 = hash3(i + vec3<i32>(1, 0, 0));
    let n010 = hash3(i + vec3<i32>(0, 1, 0));
    let n110 = hash3(i + vec3<i32>(1, 1, 0));
    let n001 = hash3(i + vec3<i32>(0, 0, 1));
    let n101 = hash3(i + vec3<i32>(1, 0, 1));
    let n011 = hash3(i + vec3<i32>(0, 1, 1));
    let n111 = hash3(i + vec3<i32>(1, 1, 1));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var a = 0.5;
    var s = 0.0;
    var q = p;
    for (var i = 0; i < 5; i = i + 1) {
        s = s + a * value_noise(q);
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 4.1);
        a = a * 0.5;
    }
    return s;
}
