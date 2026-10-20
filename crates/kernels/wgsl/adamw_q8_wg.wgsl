// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in 8-bit optimiser state on GPUs then
// you can procure our services by sending an email to info@swedishembedded.com.

// @what  AdamW with blockwise-int8 moments, cooperative form: fused decode, update, re-encode (stochastic rounding optional)
// @how   one workgroup per 64-element block, one thread per element (coalesced); the block maximum by a tree reduction
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The barrier-using sibling of `adamw_q8.wgsl` (same state layout, same
// arithmetic, same stochastic-rounding hash; read its header). The reference
// form gives each thread a whole block, so lanes of a warp are 256 bytes apart
// and the kernel runs far below memory bandwidth; here consecutive lanes touch
// consecutive elements. The block maxima are exact under any reduction order, so
// the result is bit-identical to the reference form. The CPU JIT mis-executes
// barriers, so `optim` selects this one only where the device reports
// workgroup reductions (`backend_api::select::Op::AdamWQ8`).

struct Params {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
    scale: f32,
    seed_lo: u32,
    seed_hi: u32,
    step: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read>       grad:  array<f32>;
@group(0) @binding(3) var<storage, read_write> mq:    array<u32>;
@group(0) @binding(4) var<storage, read_write> vq:    array<u32>;
@group(0) @binding(5) var<storage, read_write> sc:    array<f32>;
@group(0) @binding(6) var<storage, read>       desc:  array<u32>;
@group(0) @binding(7) var<storage, read>       coef:  array<f32>;

const BLOCK: u32 = 64u;

fn mix32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn key_lo() -> u32 {
    return mix32(p.seed_lo ^ mix32(p.step));
}

fn key_hi(tensor: u32, slot: u32) -> u32 {
    return mix32(p.seed_hi ^ mix32(((tensor << 8u) | slot) + 0x9E3779B9u));
}

fn unit(k0: u32, k1: u32, idx: u32) -> f32 {
    return f32(mix32(mix32(k0 ^ idx) ^ k1) >> 8u) * (1.0 / 16777216.0);
}

fn signed_code(c: u32) -> i32 {
    // sign-extend one byte
    return i32(c << 24u) >> 24u;
}

fn decode_m(c: u32, scale: f32) -> f32 {
    let q = signed_code(c);
    let r = f32(abs(q)) * 0.0078740157;
    let m = r * r * scale;
    if (q < 0) { return -m; }
    return m;
}

fn decode_v(c: u32, scale: f32) -> f32 {
    let r = f32(c) * 0.0039215689;
    let r2 = r * r;
    return r2 * r2 * scale;
}

fn round_code(x: f32, sr: bool, u: f32) -> f32 {
    if (sr) { return floor(x + u); }
    return floor(x + 0.5);
}

fn encode_m(v: f32, inv: f32, sr: bool, u: f32) -> u32 {
    if (inv <= 0.0) { return 0u; }
    let q = i32(clamp(round_code(sqrt(min(abs(v) * inv, 1.0)) * 127.0, sr, u), 0.0, 127.0));
    var s = q;
    if (v < 0.0) { s = -q; }
    return u32(s) & 0xffu;
}

fn encode_v(v: f32, inv: f32, sr: bool, u: f32) -> u32 {
    if (inv <= 0.0 || v <= 0.0) { return 0u; }
    let q = clamp(round_code(sqrt(sqrt(min(v * inv, 1.0))) * 255.0, sr, u), 1.0, 255.0);
    return u32(q);
}

const BPW: u32 = 4u; // blocks per workgroup: independent chains share the barriers (latency hiding)

var<workgroup> red_m: array<f32, 256>;
var<workgroup> red_v: array<f32, 256>;
var<workgroup> codes_m: array<u32, 256>;
var<workgroup> codes_v: array<u32, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) t: u32,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = wid.y * nwg.x + wid.x;
    let numel = desc[0];
    let nb = (numel + BLOCK - 1u) / BLOCK;
    let lr = p.lr * bitcast<f32>(desc[1]);
    let tensor = desc[2];
    let sr = desc[3] != 0u;
    var m: array<f32, 4>;
    var v: array<f32, 4>;
    for (var k = 0u; k < BPW; k = k + 1u) {
        let b = wg * BPW + k;
        let idx = b * BLOCK + t;
        m[k] = 0.0;
        v[k] = 0.0;
        if (idx < numel) {
            let shift = (t & 3u) * 8u;
            let mc = (mq[idx >> 2u] >> shift) & 0xffu;
            let vc = (vq[idx >> 2u] >> shift) & 0xffu;
            var mk = decode_m(mc, sc[b]);
            var vk = decode_v(vc, sc[nb + b]);
            let g = grad[idx] * p.scale * coef[0];
            mk = p.beta1 * mk + (1.0 - p.beta1) * g;
            vk = p.beta2 * vk + (1.0 - p.beta2) * g * g;
            let mhat = mk / p.bc1;
            let vhat = vk / p.bc2;
            var w = param[idx];
            w = w - lr * p.wd * w;
            w = w - lr * mhat / (sqrt(vhat) + p.eps);
            param[idx] = w;
            m[k] = mk;
            v[k] = vk;
        }
        red_m[k * BLOCK + t] = abs(m[k]);
        red_v[k * BLOCK + t] = max(v[k], 0.0);
    }
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (t < s) {
            for (var k = 0u; k < BPW; k = k + 1u) {
                let o = k * BLOCK + t;
                red_m[o] = max(red_m[o], red_m[o + s]);
                red_v[o] = max(red_v[o], red_v[o + s]);
            }
        }
        workgroupBarrier();
    }
    let k0 = key_lo();
    let k1m = key_hi(tensor, 0u);
    let k1v = key_hi(tensor, 1u);
    for (var k = 0u; k < BPW; k = k + 1u) {
        let b = wg * BPW + k;
        let idx = b * BLOCK + t;
        let mmax = red_m[k * BLOCK];
        let vmax = red_v[k * BLOCK];
        var minv = 0.0;
        if (mmax > 0.0) { minv = 1.0 / mmax; }
        var vinv = 0.0;
        if (vmax > 0.0) { vinv = 1.0 / vmax; }
        var cm = 0u;
        var cv = 0u;
        if (idx < numel) {
            cm = encode_m(m[k], minv, sr, unit(k0, k1m, idx));
            cv = encode_v(v[k], vinv, sr, unit(k0, k1v, idx));
        }
        codes_m[k * BLOCK + t] = cm;
        codes_v[k * BLOCK + t] = cv;
    }
    workgroupBarrier();
    for (var k = 0u; k < BPW; k = k + 1u) {
        let b = wg * BPW + k;
        if (t < 16u && b * BLOCK + t * 4u < numel) {
            let j = k * BLOCK + t * 4u;
            mq[(b * BLOCK + t * 4u) >> 2u] = codes_m[j] | (codes_m[j + 1u] << 8u) | (codes_m[j + 2u] << 16u) | (codes_m[j + 3u] << 24u);
            vq[(b * BLOCK + t * 4u) >> 2u] = codes_v[j] | (codes_v[j + 1u] << 8u) | (codes_v[j + 2u] << 16u) | (codes_v[j + 3u] << 24u);
        }
        if (t == 0u && b < nb) {
            sc[b] = red_m[k * BLOCK];
            sc[nb + b] = red_v[k * BLOCK];
        }
    }
}
