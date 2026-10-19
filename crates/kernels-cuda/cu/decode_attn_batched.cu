// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference throughput for its
// clients. If your team needs expertise in making the decode attention of a
// batched, paged-KV serving stack run at memory bandwidth, then you can procure
// our services by sending an email to info@swedishembedded.com.
//
// Batched paged-KV decode attention at head_dim 128: one query token per
// sequence attends its whole cached context, for 2..32+ sequences at once. The
// CUDA form of the triad `paged_decode_scores_batched` -> `decode_softmax_batched` ->
// `paged_decode_apply_batched` (and their int8-KV twins) in ONE launch, with the
// triad's argument contract and addressing:
//
//   params       : [bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt] u32
//   q            : [bsz, n_heads * 128]                         f32
//   pool_k/pool_v: [num_blocks * block_size, n_kv_heads * 128]  paged pool, f32 or (BRAIN_KV_INT8) int8
//   k_scales/v_scales (int8 only): [pool rows, n_kv_heads] f32, one scale per (token, kv head) row
//   block_tables : [bsz, max_bt] u32     seq_lens : [bsz] u32 (live keys of each sequence)
//   ctx          : [bsz, n_heads * 128]                         f32
//
// head_dim must be 128 and group (n_heads / n_kv_heads) at most 8; the caller checks.
//
// Structure. A block is one (sequence, kv head): its `group` query heads share every K and V row they read, so the
// pool is streamed once per kv head, not once per query head as the triad does. The block is 8 warps; the warps
// stride over the keys, a warp reads one key row per step (a lane owns 4 of the 128 channels: 128 contiguous int8
// bytes, or 512 f32 bytes, per row, fully coalesced), reduces each head's dot product with an xor butterfly so every
// lane holds the score, and keeps the online-softmax state (running max, sum and the lane's 4 output channels) of
// each head in registers. The 8 warps' partial states are merged once at the end through shared memory. A sequence's
// walk is serial per block, so 2 sequences fill only 16 of the card's SMs: the caller keeps the split-key kernel for
// one sequence and takes this one from two up, where bsz * n_kv_heads blocks cover the SMs.
//
// NUMERICS. fp32 throughout (scores, softmax, accumulation); an int8 plane is dequantised as int8 * scale exactly as
// the triad's. The exp2 domain differs from the triad's exp by rounding only.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_KV_INT8
#define BRAIN_KV_INT8 0
#endif

#define HD 128
#define THREADS 256
#define WARPS 8
#define MAXG 8
#define LOG2E 1.4426950408889634f
#define NEG_BIG (-1.0e30f)

#if BRAIN_KV_INT8
typedef signed char KvT;
#else
typedef float KvT;
#endif

// Four channels of a key or value row for lane `lane`, as floats (an int8 plane multiplied by its row scale).
__device__ __forceinline__ float4 brain_load4(const KvT* row, unsigned int lane, float scale) {
#if BRAIN_KV_INT8
    const int w = *reinterpret_cast<const int*>(row + lane * 4u);
    return make_float4(static_cast<float>(static_cast<signed char>(w & 0xff)) * scale, static_cast<float>(static_cast<signed char>((w >> 8) & 0xff)) * scale,
                       static_cast<float>(static_cast<signed char>((w >> 16) & 0xff)) * scale, static_cast<float>(static_cast<signed char>((w >> 24) & 0xff)) * scale);
#else
    (void)scale;
    return *reinterpret_cast<const float4*>(row + lane * 4u);
#endif
}

#if BRAIN_KV_INT8
#define SCALE_PARAMS , const float* k_scales, const float* v_scales
#else
#define SCALE_PARAMS
#endif

extern "C" __global__ void __launch_bounds__(THREADS) brain_decode_attn_batched(const unsigned int* params, const float* q, const KvT* pool_k, const KvT* pool_v,
#if BRAIN_KV_INT8
                                                                                 const float* k_scales, const float* v_scales,
#endif
                                                                                 const unsigned int* block_tables, const unsigned int* seq_lens, float* ctx) {
    const unsigned int bsz = params[0];
    const unsigned int n_heads = params[1];
    const unsigned int n_kv_heads = params[2];
    const unsigned int group = params[4];
    const unsigned int block_size = params[5];
    const unsigned int max_bt = params[6];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int seq = blk / n_kv_heads;
    const unsigned int hkv = blk % n_kv_heads;
    if (seq >= bsz) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int len = seq_lens[seq];
    const unsigned long long q_row = (unsigned long long)n_heads * HD;
    const unsigned long long kv_row = (unsigned long long)n_kv_heads * HD;
    const float qscale = 0.08838834764831845f * LOG2E;   // 1/sqrt(128), into the exp2 domain

    // This lane's 4 query channels of each head of the group.
    float4 qv[MAXG];
#pragma unroll
    for (int g = 0; g < MAXG; ++g) {
        qv[g] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (static_cast<unsigned int>(g) < group) {
            const float4 x = *reinterpret_cast<const float4*>(q + seq * q_row + (hkv * group + g) * HD + lane * 4u);
            qv[g] = make_float4(x.x * qscale, x.y * qscale, x.z * qscale, x.w * qscale);
        }
    }
    float m[MAXG], l[MAXG];
    float4 acc[MAXG];
#pragma unroll
    for (int g = 0; g < MAXG; ++g) {
        m[g] = NEG_BIG;
        l[g] = 0.f;
        acc[g] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    const unsigned int* bt = block_tables + (unsigned long long)seq * max_bt;
    for (unsigned int j = warp; j < len; j += WARPS) {
        const unsigned long long row = (unsigned long long)bt[j / block_size] * block_size + (j % block_size);
        const KvT* kp = pool_k + row * kv_row + hkv * HD;
        const KvT* vp = pool_v + row * kv_row + hkv * HD;
#if BRAIN_KV_INT8
        const float sk = k_scales[row * n_kv_heads + hkv], sv = v_scales[row * n_kv_heads + hkv];
#else
        const float sk = 1.f, sv = 1.f;
#endif
        const float4 kk = brain_load4(kp, lane, sk);
        const float4 vv = brain_load4(vp, lane, sv);
#pragma unroll
        for (int g = 0; g < MAXG; ++g) {
            if (static_cast<unsigned int>(g) >= group) { break; }
            float s = qv[g].x * kk.x + qv[g].y * kk.y + qv[g].z * kk.z + qv[g].w * kk.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xffffffffu, s, o); }
            const float mn = fmaxf(m[g], s);
            const float corr = exp2f(m[g] - mn);
            const float p = exp2f(s - mn);
            m[g] = mn;
            l[g] = l[g] * corr + p;
            acc[g].x = acc[g].x * corr + p * vv.x;
            acc[g].y = acc[g].y * corr + p * vv.y;
            acc[g].z = acc[g].z * corr + p * vv.z;
            acc[g].w = acc[g].w * corr + p * vv.w;
        }
    }

    // Merge the 8 warps' states: M = max m, L = sum l 2^(m - M), O = sum acc 2^(m - M) / L.
    __shared__ float sm_m[WARPS][MAXG];
    __shared__ float sm_l[WARPS][MAXG];
    __shared__ float4 sm_acc[WARPS][MAXG][32];
#pragma unroll
    for (int g = 0; g < MAXG; ++g) {
        if (lane == 0u) {
            sm_m[warp][g] = m[g];
            sm_l[warp][g] = l[g];
        }
        sm_acc[warp][g][lane] = acc[g];
    }
    __syncthreads();
    // Thread (g, c4): head g of the group, channels 4 c4 .. 4 c4 + 3.
    for (unsigned int t = tid; t < group * 32u; t += THREADS) {
        const unsigned int g = t / 32u, c4 = t % 32u;
        float big = NEG_BIG;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) { big = fmaxf(big, sm_m[w][g]); }
        float lsum = 0.f;
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float f = sm_l[w][g] > 0.f ? exp2f(sm_m[w][g] - big) : 0.f;
            lsum += sm_l[w][g] * f;
            const float4 a = sm_acc[w][g][c4];
            o.x += a.x * f;
            o.y += a.y * f;
            o.z += a.z * f;
            o.w += a.w * f;
        }
        const float inv = lsum > 0.f ? 1.0f / lsum : 0.f;
        *reinterpret_cast<float4*>(ctx + seq * q_row + (hkv * group + g) * HD + c4 * 4u) = make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
    }
}
