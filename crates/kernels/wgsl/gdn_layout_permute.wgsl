// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Token-major [B,T,H,D] <-> chunk-major [n_chunks,B,H,C,D] permute for model::gdn
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `model::gdn::gdn_chunk_fwd` requires every per-token buffer (query/key/
// value/decay-gate/beta) laid out CHUNK-MAJOR (`[n_chunks, B, H, C, D]`,
// chunk outermost — see that module's doc, "Layout" section, for why: it
// makes each chunk's `(b,h)` batch range a plain contiguous element offset
// for `bmm`/`bmm_acc` instead of a strided one). Every other kernel in this
// engine produces/consumes TOKEN-major buffers instead (`[B, T, H, D]` =
// `[B*T, H*D]` row-major, T = n_chunks*C), so the qwen35 model wiring needs a
// permute at the GDN layer's boundary in each direction. Neither existing
// layout-permutation kernel (`nlc_nchw`/`nchw_nlc`) covers this: those swap
// exactly two axes around a fixed batch axis, but chunk-major additionally
// SPLITS the token axis into `(chunk, c)` and moves `chunk` ahead of `B`,
// which is a genuine 5-index permute, not a 3-index one.
//
// One invocation per element of the CHUNK-MAJOR index space (used for BOTH
// directions — the mapping is a bijection over the same `(b,h,chunk,c,d)`
// tuple regardless of which side is being read from and which is being
// written to):
//   chunk_major_addr = (((chunk*B + b)*H + h)*C + c)*D + d      (== idx)
//   token_major_addr = ((b*T + chunk*C + c)*H + h)*D + d,  T = n_chunks*C
// `to_chunk_major=1`: dst[chunk_major_addr] = src[token_major_addr] (token -> chunk-major).
// `to_chunk_major=0`: dst[token_major_addr] = src[chunk_major_addr] (chunk-major -> token).
//
// `D` is passed as 1 for the scalar-per-(token,head) buffers this module
// also permutes (`raw_g`, `beta` — `GdnShape.dk`/`dv`-shaped buffers use the
// real head width instead).

struct Params {
    b: u32,
    h: u32,
    n_chunks: u32,
    c: u32,
    d: u32,
    to_chunk_major: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.b * p.h * p.n_chunks * p.c * p.d;
    if (idx >= total) { return; }

    var rem = idx;
    let d = rem % p.d; rem = rem / p.d;
    let c = rem % p.c; rem = rem / p.c;
    let h = rem % p.h; rem = rem / p.h;
    let b = rem % p.b; rem = rem / p.b;
    let chunk = rem;

    let t = chunk * p.c + c;
    let t_total = p.n_chunks * p.c;
    let token_major_addr = ((b * t_total + t) * p.h + h) * p.d + d;
    let chunk_major_addr = idx;

    if (p.to_chunk_major != 0u) {
        dst[chunk_major_addr] = src[token_major_addr];
    } else {
        dst[token_major_addr] = src[chunk_major_addr];
    }
}
