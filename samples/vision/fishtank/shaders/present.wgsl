// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Present: the tonemapped scene, scaled to the output (it renders at a
// lower resolution when the GPU is short of time), dithered against
// banding, with the HUD drawn over it pixel for pixel.

struct Present {
    // x, y, w, h of the HUD in output pixels; w = 0: no HUD
    hud: vec4<f32>,
    // x seconds (the dither's seed)
    misc: vec4<f32>,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> present: Present;
@group(0) @binding(3) var hud: texture_2d<f32>;

struct FullOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FullOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return FullOut(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(uv.x, 1.0 - uv.y));
}

@fragment
fn fs_present(v: FullOut) -> @location(0) vec4<f32> {
    var c = textureSample(scene, smp, v.uv).rgb;
    let px = vec2<i32>(v.clip.xy);
    c = c + vec3<f32>((hash3(vec3<i32>(px, i32(present.misc.x * 60.0))) - 0.5) / 255.0);
    let r = present.hud;
    if (r.z > 0.0 && v.clip.x >= r.x && v.clip.y >= r.y && v.clip.x < r.x + r.z && v.clip.y < r.y + r.w) {
        let h = textureLoad(hud, vec2<i32>(v.clip.xy - r.xy), 0);
        c = mix(c, h.rgb, h.a);
    }
    return vec4<f32>(c, 1.0);
}
