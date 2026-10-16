// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bound training kernels for linear-
// attention language models for its clients. If your team needs expertise in
// collapsing the layout and normalisation glue around a recurrent layer's
// backward into one pass over the gradients, then you can procure our services
// by sending an email to info@swedishembedded.com.
//
// The last leg of Gated DeltaNet's mixer backward for query and key, in ONE
// launch. The chunk backward leaves d_query and d_key per VALUE head in the
// chunk-major layout; the projections want them per KEY head in token order,
// through the adjoint of the L2 normalisation. The portable path takes six
// dispatches for it (two token-major permutes, two group sums, two
// normalisation backwards), each a full pass over [tokens, heads, 128]. Here a
// warp owns one (token, key head) row: it gathers the `group` value heads that
// share the key head from the chunk-major gradient, sums them, and applies the
//
//   dx = r * (dn - x * r^2 * sum(dn * x)),   r = rsqrt(sum(x^2) + eps)
//
// normalisation adjoint against the raw (pre-normalisation) projection x - the
// formula of `l2norm_scale_dx` with a unit gain - before one coalesced store.
//
// Contract (u32 params, then pointers):
//   params : [tokens, kv_heads, group, heads, c_len, seq_len, batch, eps (f32 bits)]
//   read   : d_query_cm, d_key_cm [n_chunks * batch * heads, c_len, 128] (chunk-major, value heads);
//            query, key [tokens, kv_heads, 128] (raw projections)
//   write  : d_query, d_key [tokens, kv_heads, 128]
// Head width 128 only (the provider checks); seq_len is a multiple of c_len and
// tokens = batch * seq_len. Every element of the outputs is overwritten.
//
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define THREADS 256

extern "C" __global__ void __launch_bounds__(THREADS)
brain_gdn_qk_bwd_tok(const unsigned int* params, const float* d_query_cm, const float* d_key_cm, const float* query, const float* key, float* d_query,
                     float* d_key) {
    const unsigned int tokens = params[0], kv_heads = params[1], group = params[2], heads = params[3], C = params[4], T = params[5], batch = params[6];
    const float eps = __uint_as_float(params[7]);

    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned long long rows = (unsigned long long)tokens * kv_heads;
    const unsigned long long block_row = (unsigned long long)(blockIdx.y * gridDim.x + blockIdx.x) * (THREADS / 32) + warp;
    // Rows [0, rows) are query rows, [rows, 2 * rows) key rows.
    if (block_row >= 2 * rows) { return; }  // warp-uniform
    const bool is_key = block_row >= rows;
    const unsigned long long row = is_key ? block_row - rows : block_row;
    const float* dcm = is_key ? d_key_cm : d_query_cm;
    const float* x_src = is_key ? key : query;
    float* dst = is_key ? d_key : d_query;

    const unsigned int hs = row % kv_heads;
    const unsigned int tok = row / kv_heads;
    const unsigned int b = tok / T, ti = tok % T, ci = ti / C, r = ti % C;

    float4 dn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (unsigned int k = 0; k < group; ++k) {
        const unsigned int h = hs * group + k;
        const unsigned long long bhc = (unsigned long long)ci * batch * heads + (unsigned long long)b * heads + h;
        const float4 v = *reinterpret_cast<const float4*>(dcm + (bhc * C + r) * D + lane * 4);
        dn.x += v.x;
        dn.y += v.y;
        dn.z += v.z;
        dn.w += v.w;
    }
    const float4 x = *reinterpret_cast<const float4*>(x_src + row * D + lane * 4);
    float s = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    float dot = dn.x * x.x + dn.y * x.y + dn.z * x.z + dn.w * x.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        dot += __shfl_xor_sync(0xffffffffu, dot, o);
    }
    const float inv = rsqrtf(s + eps);
    const float k2 = inv * inv * dot;
    *reinterpret_cast<float4*>(dst + row * D + lane * 4) =
        make_float4(inv * (dn.x - x.x * k2), inv * (dn.y - x.y * k2), inv * (dn.z - x.z * k2), inv * (dn.w - x.w * k2));
}
