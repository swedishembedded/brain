// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels in front of a decode-time attention into one launch, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// Everything a gated-attention decode step does between the q/k/v projections
// and the attention scores, for each of `rows` tokens, in a single launch:
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
//   params : u32 [nh, nkv, head_dim, half, eps (f32 bits), block_size, rows, 0]
//            `half` is the rotary table width (rot_dim / 2); head_dim <= 512
//   q_full : [rows, nh * 2 * head_dim] f32   k, v : [rows, nkv * head_dim] f32
//   q_norm, k_norm : [head_dim] f32          cos, sin : [rows, half] f32
//   blocks, offsets : [rows] u32             where each token's K and V land
//   q_out, q_gate : [rows, nh * head_dim] f32   pool_k, pool_v : the paged KV pools
//
// The rows are independent tokens - the rows of a decode batch of different
// sequences, or consecutive tokens of one (a speculative verify round, whose
// rows differ only in `offsets`) - and a row's heads write only its own
// outputs and its own pool row, so no ordering between them is needed.
//
// Work split: one 256-thread block per head of each row - the nh query heads,
// then the nkv key heads (which also copy their value head into the pool). A
// head is independent of every other, so there is no ordering to preserve
// between blocks.
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
// The int8 KV tier (`brain_gqa_decode_prep_i8`, knowledge 386)
// ----------------------------------------------------------
// With `--kv-cache int8` the chain after the projections is the same eight kernels with the append replaced by
// `paged_kv_append_i8_clipped_batched`, and the f32 pools of the kernel above cannot take it. The second entry point runs the same
// body and ends each key head's block with that append's own arithmetic on the values it holds: the maximum of |value| over the head
// (order-free), `scale = min(max, clip[head]) / 127` (1 for an all-zero head), `q = clamp(rint(value / scale), -127, 127)` with the
// correctly rounded division, four int8 to a word, the scale stored per (pool row, kv head). Extra bindings: `clip_k`, `clip_v`
// (`[nkv]` f32), the two int8 pools as words, and their scale tables.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_GPREP_THREADS 256
#define BRAIN_GPREP_MAX_HD 512

// Programmatic dependent launch (compute capability 9.0): wait until the kernel before this one in the stream has
// finished and its writes are visible, then let the kernel after this one become resident at once. A launch without
// the programmatic attribute has nothing to wait for and no early dependents, so both are no-ops there.
#ifndef BRAIN_PDL_ENTRY
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;\n" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); } while (0)
#else
#define BRAIN_PDL_ENTRY() do {} while (0)
#endif
#endif

template <bool I8>
__device__ __forceinline__ void brain_gqa_prep_body(const unsigned int* params, const float* q_full, const float* k, const float* v,
                                                    const float* q_norm, const float* k_norm, const float* cos_t, const float* sin_t,
                                                    const unsigned int* blocks, const unsigned int* offsets, float* q_out, float* q_gate,
                                                    float* pool_k, float* pool_v, const float* clip_k, const float* clip_v,
                                                    unsigned int* words_k, unsigned int* words_v, float* scales_k, float* scales_v) {
    const unsigned int nh = params[0];
    const unsigned int nkv = params[1];
    const unsigned int hd = params[2];
    const unsigned int half = params[3];
    const float eps = __uint_as_float(params[4]);
    const unsigned int block_size = params[5];
    const unsigned int rows = params[6];

    const unsigned int blk_all = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk_all >= (nh + nkv) * rows) { return; }  // block-uniform
    const unsigned int row = blk_all / (nh + nkv);
    const unsigned int blk = blk_all - row * (nh + nkv);
    const bool is_q = blk < nh;
    const unsigned int h = is_q ? blk : blk - nh;
    const unsigned int t = threadIdx.x;

    __shared__ float xs[BRAIN_GPREP_MAX_HD];
    __shared__ float inv_s;
    // The int8 tier's staging: the head's rotated K and its V, and the two maxima per warp.
    __shared__ float ys[I8 ? BRAIN_GPREP_MAX_HD : 1];
    __shared__ float vs[I8 ? BRAIN_GPREP_MAX_HD : 1];
    __shared__ float maxes[I8 ? 2 * (BRAIN_GPREP_THREADS / 32) : 1];

    // The head's own values, loaded before anything is written.
    const unsigned long long kv_row = (unsigned long long)row * nkv * hd;
    const float* src = is_q ? (q_full + (unsigned long long)row * nh * 2u * hd + (unsigned long long)h * 2u * hd) : (k + kv_row + (unsigned long long)h * hd);
    const float* cos_r = cos_t + (unsigned long long)row * half;
    const float* sin_r = sin_t + (unsigned long long)row * half;
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
        vv[i] = (live && !is_q) ? v[kv_row + (unsigned long long)h * hd + d] : 0.0f;
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

    const unsigned long long slot = (unsigned long long)blocks[row] * block_size + offsets[row];
    const unsigned long long pool_row = slot * (nkv * hd);
    float* dst = is_q ? (q_out + (unsigned long long)row * nh * hd + (unsigned long long)h * hd) : (I8 ? nullptr : pool_k + pool_row + (unsigned long long)h * hd);
    float mk = 0.0f, mv = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_GPREP_MAX_HD / BRAIN_GPREP_THREADS; ++i) {
        const unsigned int d = t + BRAIN_GPREP_THREADS * i;
        if (d < hd) {
            float y = xs[d];
            if (d < half) {
                const float c = cos_r[d];
                const float s = __fmul_rn(sin_r[d], 1.0f);
                y = __fsub_rn(__fmul_rn(xs[d], c), __fmul_rn(xs[d + half], s));
            } else if (d >= half && d < 2u * half) {
                const unsigned int e = d - half;
                const float c = cos_r[e];
                const float s = __fmul_rn(sin_r[e], 1.0f);
                y = __fadd_rn(__fmul_rn(xs[d], c), __fmul_rn(xs[e], s));
            }
            if (is_q) {
                dst[d] = y;
                q_gate[(unsigned long long)row * nh * hd + (unsigned long long)h * hd + d] = gate[i];
            } else if (I8) {
                ys[d] = y;
                vs[d] = vv[i];
                mk = fmaxf(mk, fabsf(y));
                mv = fmaxf(mv, fabsf(vv[i]));
            } else {
                dst[d] = y;
                pool_v[pool_row + (unsigned long long)h * hd + d] = vv[i];
            }
        }
    }
    if (I8 && !is_q) {
        // The maxima: a warp's by shuffles (exact in any order), the block's from the per-warp ones.
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            mk = fmaxf(mk, __shfl_xor_sync(0xffffffffu, mk, o));
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, o));
        }
        if ((t & 31u) == 0u) { maxes[2 * (t >> 5)] = mk; maxes[2 * (t >> 5) + 1] = mv; }
        __syncthreads();
        float bk = 0.0f, bv = 0.0f;
#pragma unroll
        for (int w = 0; w < BRAIN_GPREP_THREADS / 32; ++w) { bk = fmaxf(bk, maxes[2 * w]); bv = fmaxf(bv, maxes[2 * w + 1]); }
        float sk = __fdiv_rn(fminf(bk, clip_k[h]), 127.0f);
        float sv = __fdiv_rn(fminf(bv, clip_v[h]), 127.0f);
        if (sk == 0.0f) { sk = 1.0f; }
        if (sv == 0.0f) { sv = 1.0f; }
        if (t == 0) {
            scales_k[slot * nkv + h] = sk;
            scales_v[slot * nkv + h] = sv;
        }
        // Four consecutive elements to a word, each by its head's scale.
        const unsigned long long word0 = (pool_row + (unsigned long long)h * hd) / 4u;
        if (t < hd / 4u) {
            unsigned int pk = 0u, pv = 0u;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int qk = max(-127, min(127, static_cast<int>(rintf(__fdiv_rn(ys[4 * t + j], sk)))));
                const int qv = max(-127, min(127, static_cast<int>(rintf(__fdiv_rn(vs[4 * t + j], sv)))));
                pk |= (static_cast<unsigned int>(qk) & 0xffu) << (8 * j);
                pv |= (static_cast<unsigned int>(qv) & 0xffu) << (8 * j);
            }
            words_k[word0 + t] = pk;
            words_v[word0 + t] = pv;
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_GPREP_THREADS)
brain_gqa_decode_prep(const unsigned int* params, const float* q_full, const float* k, const float* v,
                      const float* q_norm, const float* k_norm, const float* cos_t, const float* sin_t,
                      const unsigned int* blocks, const unsigned int* offsets,
                      float* q_out, float* q_gate, float* pool_k, float* pool_v) {
    BRAIN_PDL_ENTRY();
    brain_gqa_prep_body<false>(params, q_full, k, v, q_norm, k_norm, cos_t, sin_t, blocks, offsets, q_out, q_gate, pool_k, pool_v, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr);
}

extern "C" __global__ void __launch_bounds__(BRAIN_GPREP_THREADS)
brain_gqa_decode_prep_i8(const unsigned int* params, const float* q_full, const float* k, const float* v,
                         const float* q_norm, const float* k_norm, const float* cos_t, const float* sin_t,
                         const unsigned int* blocks, const unsigned int* offsets,
                         float* q_out, float* q_gate, const float* clip_k, const float* clip_v,
                         unsigned int* words_k, unsigned int* words_v, float* scales_k, float* scales_v) {
    BRAIN_PDL_ENTRY();
    brain_gqa_prep_body<true>(params, q_full, k, v, q_norm, k_norm, cos_t, sin_t, blocks, offsets, q_out, q_gate, nullptr, nullptr, clip_k, clip_v,
                              words_k, words_v, scales_k, scales_v);
}
