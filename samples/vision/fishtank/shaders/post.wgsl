// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Post-processing: a bloom pyramid over the HDR scene, then tonemapping and
// the HUD overlay into the presentable (sRGB) target.
//
// Bloom is the usual progressive downsample / upsample: the first downsample
// keeps only what is brighter than 1 (soft knee), each later one halves the
// resolution, and the upsample pass adds every level back into the one above
// with a tent filter. Cheap - a handful of small passes - and it is what
// makes the neon read as light rather than as paint.

struct Post {
    // xy: 1 / source size; z: threshold pass (1) or not (0); w: unused
    texel: vec4<f32>,
    // composite: x exposure, y bloom strength, z seconds, w HUD on
    grade: vec4<f32>,
    // composite: HUD rectangle in output pixels (x, y, w, h)
    hud: vec4<f32>,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> post: Post;
@group(0) @binding(3) var aux: texture_2d<f32>;
@group(0) @binding(4) var hud: texture_2d<f32>;

struct FullOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FullOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FullOut;
    o.clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return o;
}

fn soft_threshold(c: vec3<f32>) -> vec3<f32> {
    let b = max(c.r, max(c.g, c.b));
    let knee = 0.5;
    let soft = clamp(b - 1.0 + knee, 0.0, 2.0 * knee);
    let w = max(soft * soft / (4.0 * knee + 1e-4), b - 1.0) / max(b, 1e-4);
    return c * w;
}

@fragment
fn fs_down(v: FullOut) -> @location(0) vec4<f32> {
    let o = post.texel.xy;
    var c = textureSample(src, smp, v.uv + vec2<f32>(-o.x, -o.y)).rgb;
    c = c + textureSample(src, smp, v.uv + vec2<f32>(o.x, -o.y)).rgb;
    c = c + textureSample(src, smp, v.uv + vec2<f32>(-o.x, o.y)).rgb;
    c = c + textureSample(src, smp, v.uv + vec2<f32>(o.x, o.y)).rgb;
    c = c * 0.25;
    if (post.texel.z > 0.5) {
        c = soft_threshold(min(c, vec3<f32>(64.0)));
    }
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_up(v: FullOut) -> @location(0) vec4<f32> {
    let o = post.texel.xy;
    var c = textureSample(src, smp, v.uv).rgb * 4.0;
    c = c + (textureSample(src, smp, v.uv + vec2<f32>(o.x, 0.0)).rgb + textureSample(src, smp, v.uv - vec2<f32>(o.x, 0.0)).rgb) * 2.0;
    c = c + (textureSample(src, smp, v.uv + vec2<f32>(0.0, o.y)).rgb + textureSample(src, smp, v.uv - vec2<f32>(0.0, o.y)).rgb) * 2.0;
    c = c + textureSample(src, smp, v.uv + o).rgb + textureSample(src, smp, v.uv - o).rgb;
    c = c + textureSample(src, smp, v.uv + vec2<f32>(o.x, -o.y)).rgb + textureSample(src, smp, v.uv + vec2<f32>(-o.x, o.y)).rgb;
    return vec4<f32>(c / 16.0, 1.0);
}

// ACES filmic curve (Narkowicz's fit).
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(v: FullOut) -> @location(0) vec4<f32> {
    let scene = textureSample(src, smp, v.uv).rgb;
    let bloom = textureSample(aux, smp, v.uv).rgb;
    var c = aces((scene + bloom * post.grade.y) * post.grade.x);
    // vignette
    let q = v.uv - 0.5;
    c = c * (1.0 - 0.35 * dot(q, q) * 2.0);
    // dither away gradient banding
    let px = vec2<i32>(v.clip.xy);
    let n = hash3(vec3<i32>(px, i32(post.grade.z * 60.0))) - 0.5;
    c = c + vec3<f32>(n / 255.0);
    // the HUD, pixel for pixel
    let r = post.hud;
    if (post.grade.w > 0.5 && v.clip.x >= r.x && v.clip.y >= r.y && v.clip.x < r.x + r.z && v.clip.y < r.y + r.w) {
        let h = textureLoad(hud, vec2<i32>(v.clip.xy - r.xy), 0);
        c = mix(c, h.rgb, h.a);
    }
    return vec4<f32>(c, 1.0);
}
