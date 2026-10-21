// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in half-precision optimiser state on
// GPUs then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  AdamW with bf16 first/second moments: fused decode, update, re-encode (stochastic rounding optional)
// @how   one thread per word of two moment elements, so threads never share a word
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The device twin of `optim::rule::Rule::AdamW` over `StateCodec` (Bf16):
// the CPU reference in the `optim` crate is the definition and
// `optim/tests/adamw_bf16_parity.rs` gates this kernel against it.
//
// `mq`/`vq` hold the moments as bf16, two elements to a u32 word (low half
// first, `model::half::pack_bf16`'s layout). Stochastic rounding adds 16 random
// bits below the kept mantissa before truncating, keyed on (seed, step,
// tensor, slot, element) with the same 32-bit mixer as `adamw_q8.wgsl`, so the
// draws are bit-identical to `optim::codec`.
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
@group(0) @binding(5) var<storage, read>       desc:  array<u32>;
@group(0) @binding(6) var<storage, read>       coef:  array<f32>;

fn mix32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn unit(k0: u32, k1: u32, idx: u32) -> f32 {
    return f32(mix32(mix32(k0 ^ idx) ^ k1) >> 8u) * (1.0 / 16777216.0);
}

fn bf16_to_f32(h: u32) -> f32 {
    return bitcast<f32>(h << 16u);
}

// f32 -> bf16 bits, nearest-even or stochastic (`u` in [0, 1)).
fn f32_to_bf16(x: f32, sr: bool, u: f32) -> u32 {
    let b = bitcast<u32>(x);
    if ((b & 0x7f800000u) == 0x7f800000u) {
        // inf or NaN: keep the class, set a mantissa bit for NaN
        var h = b >> 16u;
        if ((b & 0x007fffffu) != 0u) { h = h | 0x0040u; }
        return h & 0xffffu;
    }
    if (sr) {
        let r = u32(u * 65536.0) & 0xffffu;
        return ((b + r) >> 16u) & 0xffffu;
    }
    let lsb = (b >> 16u) & 1u;
    return ((b + 0x7fffu + lsb) >> 16u) & 0xffffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let wi = gid.y * (nwg.x * 64u) + gid.x;
    let numel = desc[0];
    let base = wi * 2u;
    if (base >= numel) { return; }
    let lr = p.lr * bitcast<f32>(desc[1]);
    let tensor = desc[2];
    let sr = desc[3] != 0u;
    let mword = mq[wi];
    let vword = vq[wi];
    let k0 = mix32(p.seed_lo ^ mix32(p.step));
    let k1m = mix32(p.seed_hi ^ mix32(((tensor << 8u) | 0u) + 0x9E3779B9u));
    let k1v = mix32(p.seed_hi ^ mix32(((tensor << 8u) | 1u) + 0x9E3779B9u));
    var mout = 0u;
    var vout = 0u;
    for (var j = 0u; j < 2u; j = j + 1u) {
        let idx = base + j;
        if (idx >= numel) { break; }
        let shift = j * 16u;
        var m = bf16_to_f32((mword >> shift) & 0xffffu);
        var v = bf16_to_f32((vword >> shift) & 0xffffu);
        let g = grad[idx] * p.scale * coef[0];
        m = p.beta1 * m + (1.0 - p.beta1) * g;
        v = p.beta2 * v + (1.0 - p.beta2) * g * g;
        let mhat = m / p.bc1;
        let vhat = v / p.bc2;
        var w = param[idx];
        w = w - lr * p.wd * w;
        w = w - lr * mhat / (sqrt(vhat) + p.eps);
        param[idx] = w;
        mout = mout | (f32_to_bf16(m, sr, unit(k0, k1m, idx)) << shift);
        vout = vout | (f32_to_bf16(v, sr, unit(k0, k1v, idx)) << shift);
    }
    mq[wi] = mout;
    vq[wi] = vout;
}
