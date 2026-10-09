// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fused attention kernels for GPUs without
// tensor cores for its clients. If your team needs expertise in getting
// attention onto the fp32 roof of the hardware it actually has then you can
// procure our services by sending an email to info@swedishembedded.com.
//
// Bidirectional (non-causal) flash attention over a packed qkv slab, fp32,
// head_dim 128 - the hand-written CUDA form of `flash_attn_bidir_reg2.wgsl`,
// with its uniform, its two bindings and its output layout:
//
//   params : u32 [bsz, n_heads, T, head_dim, qkv_stride, q_off, k_off, v_off, d_model]
//   qkv    : [bsz*T, qkv_stride] f32; row (b*T + j) holds q at q_off, k at
//            k_off, v at v_off, each n_heads * head_dim wide, head h at h*head_dim
//   out    : [bsz*T, d_model] f32; head h of row (b*T + i) at h*head_dim
//
//   out[i] = sum_j softmax_j(q_i . k_j / sqrt(head_dim)) v_j
//
// Only head_dim == 128 is served (`gpu_core::native_upgrade` checks it).
//
// What it computes, and how that differs from the WGSL kernel
// ------------------------------------------------------------
// The same online softmax, not the same bits. Each score is two chains of
// 64 products (even and odd 16-byte channel chunks), each chunk's four
// products summed on their own before they join a chain - the summation depth
// of the WGSL kernel's lane partials, so the scores are no less accurate. The
// softmax runs in base 2: a row's running maximum is kept already scaled, and
// each exponent is `fma(score, scale * log2(e), -max)`, one rounding. Each
// thread keeps its own share of a row's sum, folded across lanes once at the
// end. The gate (`gpu-core/tests/flash_bidir_native.rs`) holds both tiers to
// one stated tolerance against an f64 oracle, and this one to no more than
// twice the WGSL tier's own error on the same data.
//
// Tiling
// ------
// A 128-thread block owns 64 query rows of one (sample, head); their q stays
// in shared memory for the whole key loop (32 KiB - the reason the row tile is
// 64: q is read 128 floats a row, and staging it per key tile re-reads it
// from L2 once per tile). Keys stream through ONE 16 KiB tile of 32 rows
// holding k and then v of the same 32 keys: 48 KiB in all, the most a block
// may declare, and two blocks share an SM.
//
// Warp w owns rows 16w..16w+15; lane = 8*rg + cg. For the scores a thread
// holds a 4 x 4 block: rows 16w + 4rg + {0..3} by keys cg + {0, 8, 16, 24}.
// For the output the same thread owns the same four rows by the 16 channels
// of chunks cg + {0, 8, 16, 24} - 64 accumulators. A row's probabilities are
// in the eight lanes of its row group, so they move by shuffle, never through
// shared memory.
//
// Banks: a quarter-warp (eight lanes of one rg) reads one q row (a broadcast)
// and eight k rows at one channel chunk. k and v rows store chunk c at
// c ^ (row & 7), q rows at c ^ ((row >> 2) & 7), so the eight k reads, the
// eight consecutive chunks of one v row, and the four q rows a half-warp
// reads land on distinct banks.
//
// The output loop is ROLLED over the key's position in its lane group (eight
// trips of 4 keys x 64 multiply-adds). Unrolled whole it is some 25 KB of
// straight-line code a warp streams once per tile, and on a GP102 that ran at
// half the rate of the same arithmetic in a loop that stays in the
// instruction cache.
//
// Per key tile: v's global loads are issued before the scores, so they land
// while the tile is being multiplied; the next tile's k loads before the
// output update. Four barriers per tile, each covered by the other resident
// block's arithmetic.
//
// Bindings and strides that are not 16-byte multiples take scalar loads and
// stores with the identical arithmetic. No `__restrict__`: brain's device
// buffers alias by design.

#define BRAIN_FB_HD 128
#define BRAIN_FB_BQ 64
#define BRAIN_FB_BK 32
#define BRAIN_FB_THREADS 128
#define BRAIN_FB_CH (BRAIN_FB_HD / 4)
#define BRAIN_FB_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float4 brain_fb_ld(const float* p, size_t i, bool vec) {
    if (vec) { return __ldg(reinterpret_cast<const float4*>(p + i)); }
    return make_float4(__ldg(p + i), __ldg(p + i + 1), __ldg(p + i + 2), __ldg(p + i + 3));
}

// The four products of one 16-byte chunk, summed on their own.
__device__ __forceinline__ float brain_fb_dot4(float4 a, float4 b) {
    float acc = a.x * b.x;
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    return fmaf(a.w, b.w, acc);
}

// Stage the global loads of one 32-key tile (k or v, chosen by `off`) into
// registers: thread t carries chunk t%32 of key rows t/32 + 4s.
__device__ __forceinline__ void brain_fb_fetch(float4 (&r)[8], const float* qkv, size_t row0, unsigned int keys_left, unsigned int stride,
                                               unsigned int off, bool vec) {
    const unsigned int c = threadIdx.x % BRAIN_FB_CH;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const unsigned int j = threadIdx.x / BRAIN_FB_CH + 4u * s;
        r[s] = j < keys_left ? brain_fb_ld(qkv, (row0 + j) * stride + off + 4u * c, vec) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

__device__ __forceinline__ void brain_fb_stage(float4* kv, const float4 (&r)[8]) {
    const unsigned int c = threadIdx.x % BRAIN_FB_CH;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const unsigned int j = threadIdx.x / BRAIN_FB_CH + 4u * s;
        kv[j * BRAIN_FB_CH + (c ^ (j & 7u))] = r[s];
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_FB_THREADS, 2)
brain_flash_bidir_f32(const unsigned int* params, const float* qkv, float* out) {
    const unsigned int bsz = params[0], heads = params[1], T = params[2];
    const unsigned int stride = params[4], q_off = params[5], k_off = params[6], v_off = params[7], d_model = params[8];

    const unsigned int qtiles = (T + BRAIN_FB_BQ - 1) / BRAIN_FB_BQ;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= bsz * heads * qtiles) { return; }
    const unsigned int qt = blk % qtiles;
    const unsigned int h = (blk / qtiles) % heads;
    const unsigned int b = blk / qtiles / heads;

    __shared__ float4 qs[BRAIN_FB_BQ * BRAIN_FB_CH];  // 32 KiB, chunk ^ ((row >> 2) & 7)
    __shared__ float4 kv[BRAIN_FB_BK * BRAIN_FB_CH];  // 16 KiB, chunk ^ (row & 7)

    const bool vec = ((reinterpret_cast<size_t>(qkv) | reinterpret_cast<size_t>(out)) & 15u) == 0
                     && ((stride | q_off | k_off | v_off | d_model) & 3u) == 0;
    const size_t seq0 = (size_t)b * T;  // first slab row of this sample
    const unsigned int hoff = h * BRAIN_FB_HD;

    // q: 64 rows x 32 chunks, sixteen 16-byte pieces a thread; rows past T are zero.
    {
        const unsigned int c = threadIdx.x % BRAIN_FB_CH;
        const unsigned int i0 = qt * BRAIN_FB_BQ;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int r = threadIdx.x / BRAIN_FB_CH + 4u * s;
            qs[r * BRAIN_FB_CH + (c ^ ((r >> 2) & 7u))] =
                i0 + r < T ? brain_fb_ld(qkv, (seq0 + i0 + r) * stride + q_off + hoff + 4u * c, vec) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int rg = lane / 8, cg = lane % 8;
    const unsigned int row0 = 16u * warp + 4u * rg;  // this thread's first of four rows in the tile
    const unsigned int ntiles = (T + BRAIN_FB_BK - 1) / BRAIN_FB_BK;
    // exp(x / sqrt(128)) = exp2(x * c2); the running maxima m are in these units.
    const float c2 = 0.08838834764831845f * 1.4426950408889634f;

    float4 o[4][4];
    float m[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = BRAIN_FB_NEG_INF;
        l[i] = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) { o[i][d] = make_float4(0.f, 0.f, 0.f, 0.f); }
    }

    float4 pre[8];
    brain_fb_fetch(pre, qkv, seq0, T, stride, k_off + hoff, vec);
    brain_fb_stage(kv, pre);
    __syncthreads();

    for (unsigned int t = 0; t < ntiles; ++t) {
        const unsigned int j0 = t * BRAIN_FB_BK;
        const unsigned int left = T - j0;
        // v of this tile, in flight while its scores are computed.
        brain_fb_fetch(pre, qkv, seq0 + j0, left, stride, v_off + hoff, vec);

        // Scores: even chunks into s, odd ones into s2.
        float s[4][4], s2[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s[i][k] = 0.f;
                s2[i][k] = 0.f;
            }
        }
        const unsigned int qswz = (row0 >> 2) & 7u;
#pragma unroll 2
        for (unsigned int c = 0; c < BRAIN_FB_CH; c += 2) {
            float4 qa[4], qb[4], ka[4], kb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qa[i] = qs[(row0 + i) * BRAIN_FB_CH + (c ^ qswz)];
                qb[i] = qs[(row0 + i) * BRAIN_FB_CH + ((c + 1) ^ qswz)];
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                ka[k] = kv[(cg + 8u * k) * BRAIN_FB_CH + (c ^ cg)];
                kb[k] = kv[(cg + 8u * k) * BRAIN_FB_CH + ((c + 1) ^ cg)];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s[i][k] += brain_fb_dot4(qa[i], ka[k]);
                    s2[i][k] += brain_fb_dot4(qb[i], kb[k]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int k = 0; k < 4; ++k) { s[i][k] += s2[i][k]; }
        }

        // Online softmax in base 2. A key past T scores -inf, so its weight is 0.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = BRAIN_FB_NEG_INF;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s[i][k] = cg + 8u * k < left ? s[i][k] : BRAIN_FB_NEG_INF;
                mx = fmaxf(mx, s[i][k]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float mn = fmaxf(m[i], mx * c2);
            const float corr = exp2f(m[i] - mn);
            m[i] = mn;
            float ls = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s[i][k] = exp2f(fmaf(s[i][k], c2, -mn));
                ls += s[i][k];
            }
            l[i] = fmaf(l[i], corr, ls);
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                o[i][d].x *= corr;
                o[i][d].y *= corr;
                o[i][d].z *= corr;
                o[i][d].w *= corr;
            }
        }

        __syncthreads();  // every warp is done reading k
        brain_fb_stage(kv, pre);
        __syncthreads();  // v is visible
        if (t + 1 < ntiles) { brain_fb_fetch(pre, qkv, seq0 + j0 + BRAIN_FB_BK, left - BRAIN_FB_BK, stride, k_off + hoff, vec); }

        // Output: key j = g + 8k's probabilities come from lane g of the row group.
        const unsigned int base = lane & ~7u;
#pragma unroll 1
        for (unsigned int g = 0; g < 8; ++g) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const unsigned int j = g + 8u * k;
                float p[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { p[i] = __shfl_sync(0xffffffffu, s[i][k], base + g); }
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const float4 v = kv[j * BRAIN_FB_CH + ((cg + 8u * d) ^ g)];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        o[i][d].x = fmaf(p[i], v.x, o[i][d].x);
                        o[i][d].y = fmaf(p[i], v.y, o[i][d].y);
                        o[i][d].z = fmaf(p[i], v.z, o[i][d].z);
                        o[i][d].w = fmaf(p[i], v.w, o[i][d].w);
                    }
                }
            }
        }

        __syncthreads();  // every warp is done reading v
        if (t + 1 < ntiles) {
            brain_fb_stage(kv, pre);
            __syncthreads();  // the next k is visible
        }
    }

    // Fold each row's sum across its eight lanes, normalise, store.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float ls = l[i];
        ls += __shfl_xor_sync(0xffffffffu, ls, 1);
        ls += __shfl_xor_sync(0xffffffffu, ls, 2);
        ls += __shfl_xor_sync(0xffffffffu, ls, 4);
        const float inv = 1.0f / ls;
        const unsigned int r = qt * BRAIN_FB_BQ + row0 + i;
        if (r >= T) { continue; }
        float* dst = out + (seq0 + r) * d_model + hoff;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const unsigned int ch = 4u * (cg + 8u * d);
            const float4 v = make_float4(o[i][d].x * inv, o[i][d].y * inv, o[i][d].z * inv, o[i][d].w * inv);
            if (vec) {
                *reinterpret_cast<float4*>(dst + ch) = v;
            } else {
                dst[ch] = v.x;
                dst[ch + 1] = v.y;
                dst[ch + 2] = v.z;
                dst[ch + 3] = v.w;
            }
        }
    }
}
