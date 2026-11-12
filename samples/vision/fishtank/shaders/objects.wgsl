// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Everything floating in the tank (and in front of the glass), instanced.
// Materials (scene.rs): 0 crystal, 1 pillar, 2 ring, 3 orb. Output is linear
// HDR; values above 1 feed the bloom.
//
// Models carry rotation, uniform scale and (pillars only) a stretch along an
// axis their normals are perpendicular to, so transforming normals by the
// model matrix and normalizing is exact here; a general scene would need the
// inverse-transpose.

struct ObjIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(7) material: vec4<f32>,
};

struct ObjOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) local: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) material: vec4<f32>,
};

@vertex
fn vs_obj(v: ObjIn) -> ObjOut {
    let m = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let w = m * vec4<f32>(v.position, 1.0);
    var o: ObjOut;
    o.world = w.xyz;
    o.normal = normalize((m * vec4<f32>(v.normal, 0.0)).xyz);
    o.local = v.position;
    o.color = v.color.rgb;
    o.material = v.material;
    o.clip = frame.view_proj * w;
    return o;
}

fn palette(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.2831853 * (vec3<f32>(t) + vec3<f32>(0.0, 0.33, 0.67)));
}

@fragment
fn fs_obj(v: ObjOut) -> @location(0) vec4<f32> {
    let p = v.world;
    var n = normalize(v.normal);
    let view = normalize(frame.eye.xyz - p);
    if (dot(n, view) < 0.0) {
        n = -n;
    }
    let t = frame.eye.w;
    let kind = i32(v.material.x + 0.5);
    let emit = v.material.y;
    let phase = v.material.z;
    let ndv = max(dot(n, view), 0.0);
    let fresnel = pow(1.0 - ndv, 3.0);
    let key = normalize(KEY_DIR);
    let diffuse = max(dot(n, key), 0.0);
    let spec = pow(max(dot(reflect(-key, n), view), 0.0), 48.0);
    var c = vec3<f32>(0.0);
    if (kind == 0) {
        // crystal: iridescent facets, a pulsing core, white glints
        let film = palette(fresnel * 1.4 + dot(n, vec3<f32>(0.3, 0.6, 0.2)) + t * 0.05);
        let pulse = 0.7 + 0.3 * frame.light.w;
        c = v.color * (0.25 + 0.35 * diffuse) * pulse + film * fresnel * 3.5 + vec3<f32>(spec * 4.0);
        c = c + v.color * emit * 0.6 * pow(ndv, 4.0);
    } else if (kind == 1) {
        // pillar: dark metal, scrolling emissive bands, lit by the crystal
        let band = smoothstep(0.90, 1.0, fract(p.y * 18.0 - t * 0.4 + phase * 0.37));
        c = vec3<f32>(0.015, 0.016, 0.022) * (0.3 + diffuse) + crystal_light(p, n) + vec3<f32>(spec * 0.5);
        c = c + v.color * (band * 2.5 * emit + fresnel * 0.6);
    } else if (kind == 2) {
        // ring: a glow travelling around it
        let a = atan2(v.local.y, v.local.x);
        let wave = 0.5 + 0.5 * sin(a * 3.0 - t * 2.2 + phase * 2.1);
        c = v.color * emit * (0.35 + 1.6 * pow(wave, 3.0)) + vec3<f32>(spec) + v.color * fresnel;
    } else {
        // orb: a hot core fading to its rim
        let core = pow(ndv, 2.5);
        let pulse = 0.8 + 0.2 * sin(t * 2.0 + phase * 1.9);
        c = v.color * emit * pulse * (0.25 + 1.6 * core) + vec3<f32>(spec * 2.0);
    }
    return vec4<f32>(fogged(c, p), 1.0);
}
