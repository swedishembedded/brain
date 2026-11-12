// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Integer hashing, prepended to the shader modules that need it by the
// renderer (WGSL has no include). An integer hash rather than the
// `fract(sin(x))` idiom: the latter loses its bits differently on every GPU,
// and the generated textures would differ by vendor.

fn hash_u(x: u32) -> u32 {
    var v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}

fn hash3(p: vec3<i32>) -> f32 {
    let h = hash_u(u32(p.x) ^ hash_u(u32(p.y) ^ hash_u(u32(p.z))));
    return f32(h) / 4294967295.0;
}
