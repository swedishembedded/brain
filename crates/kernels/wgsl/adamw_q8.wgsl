// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in 8-bit optimiser state on GPUs then
// you can procure our services by sending an email to info@swedishembedded.com.

// @what  AdamW with blockwise-int8 first/second moments: fused decode, update, re-encode (stochastic rounding optional)
// @how   one thread per 64-element block; the block's scale is its absolute maximum, so no barrier is needed
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The device twin of `optim::rule::Rule::AdamW` over `StateCodec` (Int8Block
// 64): the CPU reference in the `optim` crate is the definition and
// `optim/tests/adamw_q8_parity.rs` gates this kernel against it.
//
// State layout per tensor: `mq`/`vq` hold one code byte per element packed four
// to a u32 word (little endian); `sc` holds the f32 block scales, the `nb`
// first-moment scales followed by the `nb` second-moment scales, where
// `nb = ceil(numel / 64)`.
//   first moment : sign + 7-bit magnitude, x = sign * (q/127)^2 * scale
//   second moment: 8-bit magnitude,        x = (q/255)^4 * scale, never 0 for v > 0
// A block starts on a word boundary (64 = 16 words), so threads never share a
// word. Stochastic rounding hashes (seed, step, tensor, slot, element) with the
// 32-bit mixer below, bit-identical to `optim::codec`.
//
// `desc` = [numel, lr_mult (f32 bits), tensor index, stochastic flag].
// `coef[0]` is the device-resident clip coefficient, exactly as in `adamw.wgsl`.

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = gid.y * (nwg.x * 64u) + gid.x;
    let numel = desc[0];
    let base = b * BLOCK;
    if (base >= numel) { return; }
    let n = min(BLOCK, numel - base);
    let lr = p.lr * bitcast<f32>(desc[1]);
    let tensor = desc[2];
    let sr = desc[3] != 0u;
    let nb = (numel + BLOCK - 1u) / BLOCK;
    let mscale = sc[b];
    let vscale = sc[nb + b];
    let wbase = base >> 2u;

    var mvals: array<f32, 64>;
    var vvals: array<f32, 64>;
    var mwords: array<u32, 16>;
    var vwords: array<u32, 16>;
    var mmax = 0.0;
    var vmax = 0.0;
    for (var j = 0u; j < n; j = j + 1u) {
        let idx = base + j;
        let word = j >> 2u;
        let shift = (j & 3u) * 8u;
        let mc = (mq[wbase + word] >> shift) & 0xffu;
        let vc = (vq[wbase + word] >> shift) & 0xffu;
        var m = decode_m(mc, mscale);
        var v = decode_v(vc, vscale);
        let g = grad[idx] * p.scale * coef[0];
        m = p.beta1 * m + (1.0 - p.beta1) * g;
        v = p.beta2 * v + (1.0 - p.beta2) * g * g;
        let mhat = m / p.bc1;
        let vhat = v / p.bc2;
        var w = param[idx];
        w = w - lr * p.wd * w;
        w = w - lr * mhat / (sqrt(vhat) + p.eps);
        param[idx] = w;
        mvals[j] = m;
        vvals[j] = v;
        mmax = max(mmax, abs(m));
        vmax = max(vmax, max(v, 0.0));
    }

    var minv = 0.0;
    if (mmax > 0.0) { minv = 1.0 / mmax; }
    var vinv = 0.0;
    if (vmax > 0.0) { vinv = 1.0 / vmax; }
    let k0m = key_lo();
    let k1m = key_hi(tensor, 0u);
    let k1v = key_hi(tensor, 1u);
    for (var i = 0u; i < 16u; i = i + 1u) { mwords[i] = 0u; vwords[i] = 0u; }
    for (var j = 0u; j < n; j = j + 1u) {
        let idx = base + j;
        let um = unit(k0m, k1m, idx);
        let uv = unit(k0m, k1v, idx);
        let shift = (j & 3u) * 8u;
        mwords[j >> 2u] = mwords[j >> 2u] | (encode_m(mvals[j], minv, sr, um) << shift);
        vwords[j >> 2u] = vwords[j >> 2u] | (encode_v(vvals[j], vinv, sr, uv) << shift);
    }
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (i * 4u < n) {
            mq[wbase + i] = mwords[i];
            vq[wbase + i] = vwords[i];
        }
    }
    sc[b] = mmax;
    sc[nb + b] = vmax;
}
