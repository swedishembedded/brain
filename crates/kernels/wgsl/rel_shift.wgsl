// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Transformer-XL relative-position shift (NeMo / Conformer rel-pos attention)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Transformer-XL relative-position shift (NeMo / Conformer rel-pos attention).
// Reproduces the pad → reshape → drop-first-row → reshape sequence:
//   x:  [rows, q, p]           (rows = B*H)
//   pad last dim left by 1 (zeros) -> [rows, q, p+1]
//   view as [rows, p+1, q]; drop row 0 -> [rows, p, q]; view back -> [rows, q, p]
// Closed form for out[r,i,j]:  f=i*p+j; s=(f/q + 1)*q + f%q;
//   ip=s/(p+1); kp=s%(p+1);  out = 0 if kp==0 else x[r,ip,kp-1].
// A pure reindex (linear); its transpose is rel_shift_bwd.

struct Params {
    rows: u32,
    q: u32,
    p: u32,
};

@group(0) @binding(0) var<uniform> pm: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let qp = pm.q * pm.p;
    if (idx >= pm.rows * qp) { return; }
    let r = idx / qp;
    let rem = idx % qp;
    let i = rem / pm.p;
    let j = rem % pm.p;
    let f = i * pm.p + j;
    let s = (f / pm.q + 1u) * pm.q + (f % pm.q);
    let ip = s / (pm.p + 1u);
    let kp = s % (pm.p + 1u);
    if (kp == 0u) {
        out[idx] = 0.0;
    } else {
        out[idx] = x[r * qp + ip * pm.p + (kp - 1u)];
    }
}
