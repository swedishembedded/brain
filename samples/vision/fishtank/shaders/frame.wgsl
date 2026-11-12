// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Shared by every scene pass: the per-frame uniform block, fog and the
// crystal's light. Prepended to each pass's source by the renderer (WGSL has no
// include), so the layout is declared exactly once.
//
// Everything is in the SCREEN frame: metres, origin at the display centre,
// +X right, +Y up, +Z out of the glass toward the viewer.

struct Frame {
    // off-axis projection * view from the tracked eye (headpose::OffAxis)
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz: the eye; w: seconds
    eye: vec4<f32>,
    // half width, half height and depth of the tank; w: grid on (1) / off (0)
    tank: vec4<f32>,
    // xyz: the crystal (the scene's point light); w: its pulse in [0, 1]
    light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

const FOG_COLOR: vec3<f32> = vec3<f32>(0.010, 0.006, 0.028);
const KEY_DIR: vec3<f32> = vec3<f32>(-0.40, 0.75, 0.53);

// Exponential fog by distance from the eye, starting at the glass.
fn fogged(c: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let d = max(length(p - frame.eye.xyz) - 0.45, 0.0);
    return mix(FOG_COLOR, c, exp(-d * 1.4));
}

// The crystal lights its surroundings: inverse-square, softened near it.
fn crystal_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let l = frame.light.xyz - p;
    let d2 = dot(l, l);
    let ndl = max(dot(n, normalize(l)), 0.0);
    let pulse = 0.75 + 0.25 * frame.light.w;
    return vec3<f32>(0.25, 0.75, 1.0) * ndl * pulse * 0.004 / (d2 + 0.0015);
}
