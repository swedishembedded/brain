// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels in front of a decode-time attention into one launch, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// Everything a gated-attention decode step does between the q/k/v projections
// and the attention scores, for ONE token, in a single launch:
//
//   q_full holds, per head, [value | gate] (2 * head_dim)
//   q     = rope(rmsnorm_head(value) * q_norm)        written to `q_out`
//   gate  = the gate half, de-interleaved             written to `q_gate`
//   k     = rope(rmsnorm_head(k_proj) * k_norm)       appended to `pool_k`
//   v     = v_proj                                    appended to `pool_v`
//
// It replaces eight kernels (two `concat_split`, two per-head `rmsnorm`, two
// `rope2d_partial`, two `paged_kv_append_batched`) that a decode token runs 16
// times, and produces what they produce to the last bit. The generated per-head
// RMSNorm alone was ~44 us a call: one thread per head walking 256 dependent
// loads.
//
//   params : u32 [nh, nkv, head_dim, half, eps (f32 bits), block_size, 0, 0]
//            `half` is the rotary table width (rot_dim / 2); head_dim <= 512
//   q_full : [nh * 2 * head_dim] f32     k, v : [nkv * head_dim] f32
//   q_norm, k_norm : [head_dim] f32      cos, sin : [half] f32 (one row: batch 1)
//   blocks, offsets : [1] u32            where this token's K and V land
//   q_out, q_gate : [nh * head_dim] f32  pool_k, pool_v : the paged KV pools
//
// Work split: one 256-thread block per head - the nh query heads, then the nkv
// key heads (which also copy their value head into the pool). A head is
// independent of every other, so there is no ordering to preserve between
// blocks.
//
// How it stays bit-identical to the chain it replaces
// ---------------------------------------------------
// The per-head `rmsnorm.wgsl` is one thread per row: `ss = ss + v * v` for
// c = 0..d-1 in ascending order, `inv = 1 / sqrt(ss / d + eps)`, then
// `weight[c] * x * inv` - the product order matters, it is `(weight * x) *
// inv`. One thread of the block does that ascending sum from shared memory
// (every other element of the head is already in place), and the elementwise
// parts run across all threads. The rotation is `rope2d_partial.wgsl`'s: for
// d < half, `x1 * c - x2 * s` and `x2 * c + x1 * s` with `s = sin * 1.0f`,
// channels past 2 * half passing through. `--fmad=false` is the backend's
// default and every product and sum is an explicit round-to-nearest operation
// besides.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_GPREP_THREADS 256
#define BRAIN_GPREP_MAX_HD 512

extern "C" __global__ void __launch_bounds__(BRAIN_GPREP_THREADS)
brain_gqa_decode_prep(const unsigned int* params, const float* q_full, const float* k, const float* v,
                      const float* q_norm, const float* k_norm, const float* cos_t, const float* sin_t,
                      const unsigned int* blocks, const unsigned int* offsets,
                      float* q_out, float* q_gate, float* pool_k, float* pool_v) {
    const unsigned int nh = params[0];
    const unsigned int nkv = params[1];
    const unsigned int hd = params[2];
    const unsigned int half = params[3];
    const float eps = __uint_as_float(params[4]);
    const unsigned int block_size = params[5];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= nh + nkv) { return; }  // block-uniform
    const bool is_q = blk < nh;
    const unsigned int h = is_q ? blk : blk - nh;
    const unsigned int t = threadIdx.x;

    __shared__ float xs[BRAIN_GPREP_MAX_HD];
    __shared__ float inv_s;

    // The head's own values, loaded before anything is written.
    const float* src = is_q ? (q_full + (unsigned long long)h * 2u * hd) : (k + (unsigned long long)h * hd);
    const float* gain = is_q ? q_norm : k_norm;
    float x[BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS];
    float g[BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS];
    float gate[BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS];
    float vv[BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS];
#pragma unroll
    for (int i = 0; i < BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS; ++i) {
        const unsigned int d = t + BRAIN_GPREP_THREADS * i;
        const bool live = d < hd;
        x[i] = live ? src[d] : 0.0f;
        g[i] = live ? gain[d] : 0.0f;
        gate[i] = (live && is_q) ? src[hd + d] : 0.0f;
        vv[i] = (live && !is_q) ? v[(unsigned long long)h * hd + d] : 0.0f;
        if (live) { xs[d] = x[i]; }
    }
    __syncthreads();

    // The sum of squares: one thread, ascending, as the reference's one thread.
    if (t == 0) {
        float ss = 0.0f;
        for (unsigned int c = 0; c < hd; ++c) { ss = __fadd_rn(ss, __fmul_rn(xs[c], xs[c])); }
        inv_s = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(hd)), eps)));
    }
    __syncthreads();
    const float inv = inv_s;
    // `(weight * x) * inv`, then keep the normalised head in shared memory so
    // the rotation can read both halves of each rotated pair.
#pragma unroll
    for (int i = 0; i < BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS; ++i) {
        const unsigned int d = t + BRAIN_GPREP_THREADS * i;
        if (d < hd) { xs[d] = __fmul_rn(__fmul_rn(g[i], x[i]), inv); }
    }
    __syncthreads();

    float* dst = is_q ? (q_out + (unsigned long long)h * hd)
                      : (pool_k + ((unsigned long long)blocks[0] * block_size + offsets[0]) * (nkv * hd) + (unsigned long long)h * hd);
#pragma unroll
    for (int i = 0; i < BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS; ++i) {
        const unsigned int d = t + BRAIN_GPREP_THREADS * i;
        if (d < hd) {
            float y = xs[d];
            if (d < half) {
                const float c = cos_t[d];
                const float s = __fmul_rn(sin_t[d], 1.0f);
                y = __fsub_rn(__fmul_rn(xs[d], c), __fmul_rn(xs[d + half], s));
            } else if (d >= half && d < 2u * half) {
                const unsigned int e = d - half;
                const float c = cos_t[e];
                const float s = __fmul_rn(sin_t[e], 1.0f);
                y = __fadd_rn(__fmul_rn(xs[d], c), __fmul_rn(xs[e], s));
            }
            dst[d] = y;
            if (is_q) {
                q_gate[(unsigned long long)h * hd + d] = gate[i];
            } else {
                pool_v[((unsigned long long)blocks[0] * block_size + offsets[0]) * (nkv * hd) + (unsigned long long)h * hd + d] = vv[i];
            }
        }
    }
}
