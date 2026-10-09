// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fused attention kernels for GPUs without
// tensor cores for its clients. If your team needs expertise in getting
// attention onto the fp32 roof of the hardware it actually has then you can
// procure our services by sending an email to info@swedishembedded.com.
//
// Flash attention in fp32 at head_dim 128, one core and two entry points:
//
// brain_flash_bidir_f32 - bidirectional attention over a packed qkv slab, the
// hand-written form of `flash_attn_bidir_reg2.wgsl` (its uniform, its two
// bindings, its output layout; `gpu_core::native_upgrade` redirects it):
//
//   params : u32 [bsz, n_heads, T, head_dim, qkv_stride, q_off, k_off, v_off, d_model]
//   qkv    : [bsz*T, qkv_stride]; row (b*T + j) holds q at q_off, k at k_off,
//            v at v_off, each n_heads * head_dim wide, head h at h*head_dim
//   out    : [bsz*T, d_model]; head h of row (b*T + i) at h*head_dim
//   out[i] = sum_j softmax_j(q_i . k_j / sqrt(head_dim)) v_j
//
// brain_flash_gqa_kmask_f32 - causal grouped-query attention with an additive
// per-key mask, the `gqa_scores_kmask` -> softmax -> `gqa_apply` chain of a
// right-padded encoder in one launch (`gpu_core::Fused::FlashGqaKmask`):
//
//   params : u32 [bsz, n_heads, n_kv_heads, T, head_dim, group]
//   q      : [bsz*T, n_heads*head_dim]     k, v : [bsz*T, n_kv_heads*head_dim]
//   kmask  : [T], added to every score of key j (0 live, -3.4e38 excluded)
//   out    : [bsz*T, n_heads*head_dim]
//   out[i] = sum_{j<=i} softmax_j(q_i . k_j / sqrt(head_dim) + kmask[j]) v_j
//            (query head h reads key/value head h / group)
//   Every query row must see at least one key whose mask is finite in base 2;
//   key 0 live is the encoder's case.
//
// What it computes, and how that differs from the WGSL kernels
// -------------------------------------------------------------
// The same online softmax, not the same bits. Each score is two chains of
// 64 products (even and odd 16-byte channel chunks), each chunk's four
// products summed on their own before they join a chain - the summation depth
// of the WGSL bidirectional kernel's lane partials, and shallower than the
// masked chain's single 128-long sum, so the scores are no less accurate. The
// softmax runs in base 2 with the running maxima in base-2 units: each
// exponent is `fma(dot, scale*log2(e), mask*log2(e) - max)`, one rounding of
// the product (for a live key the mask is 0 and the subtraction exact), and an
// excluded key's mask is -inf there, so its weight is exactly 0. Each thread
// keeps its own share of a row's sum, folded across lanes once at the end.
// The gates (`gpu-core/tests/flash_bidir_native.rs`,
// `flash_gqa_kmask_native.rs`) hold both tiers to one stated tolerance
// against an f64 oracle, and these kernels to no more than twice the WGSL
// tier's own error on the same data.
//
// What the masked kernel does not compute
// ---------------------------------------
// A key tile is visited only if some key in it can carry weight: the causal
// bound stops at the block's last query row, and the keys after the last one
// whose mask is finite are never loaded. Skipping them is exact - each would
// add a zero weight, leave the running maximum where it is and scale the
// accumulators by exp2(0) - and it is what makes a short caption padded to a
// long fixed length cost its content, not its padding. Pad QUERY rows are
// computed in full: an encoder hands their outputs on.
//
// Tiling
// ------
// A 128-thread block owns 64 query rows of one (sample, query head); their q
// stays in shared memory for the whole key loop (32 KiB - the reason the row
// tile is 64: q is read 128 floats a row, and staging it per key tile
// re-reads it from L2 once per tile). Keys stream through ONE 16 KiB tile of
// 32 rows holding k and then v of the same 32 keys: 48 KiB in all, the most a
// block may declare, and two blocks share an SM.
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

#define BRAIN_FA_HD 128
#define BRAIN_FA_BQ 64
#define BRAIN_FA_BK 32
#define BRAIN_FA_THREADS 128
#define BRAIN_FA_CH (BRAIN_FA_HD / 4)
#define BRAIN_FA_NEG_INF __int_as_float(0xff800000)
#define BRAIN_FA_LOG2E 1.4426950408889634f

// Where one (sample, head)'s rows live. Each pointer is already offset to the
// head's first channel; a row r of the sample is at `(seq0 + r) * stride`.
struct BrainFaRows {
    const float* q;
    const float* k;
    const float* v;
    const float* kmask;  // per key, or null
    float* out;
    unsigned int q_stride, kv_stride, out_stride;
    unsigned int T;
    size_t seq0;
    bool vec;
};

__device__ __forceinline__ float4 brain_fa_ld(const float* p, size_t i, bool vec) {
    if (vec) { return __ldg(reinterpret_cast<const float4*>(p + i)); }
    return make_float4(__ldg(p + i), __ldg(p + i + 1), __ldg(p + i + 2), __ldg(p + i + 3));
}

// The four products of one 16-byte chunk, summed on their own.
__device__ __forceinline__ float brain_fa_dot4(float4 a, float4 b) {
    float acc = a.x * b.x;
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    return fmaf(a.w, b.w, acc);
}

// Stage the global loads of one 32-key tile of `src` into registers: thread t
// carries chunk t%32 of key rows t/32 + 4s; keys at or past `keys_left` are 0.
__device__ __forceinline__ void brain_fa_fetch(float4 (&r)[8], const BrainFaRows& a, const float* src, unsigned int j0, unsigned int keys_left) {
    const unsigned int c = threadIdx.x % BRAIN_FA_CH;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const unsigned int j = threadIdx.x / BRAIN_FA_CH + 4u * s;
        r[s] = j < keys_left ? brain_fa_ld(src, (a.seq0 + j0 + j) * a.kv_stride + 4u * c, a.vec) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

__device__ __forceinline__ void brain_fa_stage(float4* kv, const float4 (&r)[8]) {
    const unsigned int c = threadIdx.x % BRAIN_FA_CH;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const unsigned int j = threadIdx.x / BRAIN_FA_CH + 4u * s;
        kv[j * BRAIN_FA_CH + (c ^ (j & 7u))] = r[s];
    }
}

// One past the last key below `limit` whose base-2 mask is finite (0 if none):
// keys from there on can carry no weight.
__device__ __forceinline__ unsigned int brain_fa_live_end(const float* kmask, unsigned int limit, int* scratch) {
    int last = -1;
    for (unsigned int j = threadIdx.x; j < limit; j += BRAIN_FA_THREADS) {
        if (__ldg(kmask + j) * BRAIN_FA_LOG2E != BRAIN_FA_NEG_INF) { last = (int)j; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { last = max(last, __shfl_xor_sync(0xffffffffu, last, o)); }
    if (threadIdx.x % 32 == 0) { scratch[threadIdx.x / 32] = last; }
    __syncthreads();
    last = max(max(scratch[0], scratch[1]), max(scratch[2], scratch[3]));
    __syncthreads();
    return (unsigned int)(last + 1);
}

// Query tile `qt` of one (sample, head). MASKED selects the causal bound and
// the per-key mask.
template <bool MASKED>
__device__ __forceinline__ void brain_fa_core(const BrainFaRows& a, unsigned int qt, float4* qs, float4* kv) {
    const unsigned int T = a.T;
    const unsigned int q0 = qt * BRAIN_FA_BQ;

    // Keys that can carry weight: all of them, or up to the block's last row
    // and the last live key.
    unsigned int kend = T;
    if (MASKED) {
        kend = min(T, q0 + BRAIN_FA_BQ);
        kend = brain_fa_live_end(a.kmask, kend, reinterpret_cast<int*>(kv));
    }

    // q: 64 rows x 32 chunks, sixteen 16-byte pieces a thread; rows past T are zero.
    {
        const unsigned int c = threadIdx.x % BRAIN_FA_CH;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const unsigned int r = threadIdx.x / BRAIN_FA_CH + 4u * s;
            qs[r * BRAIN_FA_CH + (c ^ ((r >> 2) & 7u))] =
                q0 + r < T ? brain_fa_ld(a.q, (a.seq0 + q0 + r) * a.q_stride + 4u * c, a.vec) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int rg = lane / 8, cg = lane % 8;
    const unsigned int row0 = 16u * warp + 4u * rg;  // this thread's first of four rows in the tile
    const unsigned int ntiles = (kend + BRAIN_FA_BK - 1) / BRAIN_FA_BK;
    // exp(x / sqrt(128)) = exp2(x * c2); the running maxima m are in these units.
    const float c2 = 0.08838834764831845f * BRAIN_FA_LOG2E;

    float4 o[4][4];
    float m[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = BRAIN_FA_NEG_INF;
        l[i] = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) { o[i][d] = make_float4(0.f, 0.f, 0.f, 0.f); }
    }

    float4 pre[8];
    if (ntiles > 0) {
        brain_fa_fetch(pre, a, a.k, 0, kend);
        brain_fa_stage(kv, pre);
    }
    __syncthreads();

    for (unsigned int t = 0; t < ntiles; ++t) {
        const unsigned int j0 = t * BRAIN_FA_BK;
        const unsigned int left = kend - j0;
        // v of this tile, in flight while its scores are computed.
        brain_fa_fetch(pre, a, a.v, j0, left);

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
        for (unsigned int c = 0; c < BRAIN_FA_CH; c += 2) {
            float4 qa[4], qb[4], ka[4], kb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qa[i] = qs[(row0 + i) * BRAIN_FA_CH + (c ^ qswz)];
                qb[i] = qs[(row0 + i) * BRAIN_FA_CH + ((c + 1) ^ qswz)];
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                ka[k] = kv[(cg + 8u * k) * BRAIN_FA_CH + (c ^ cg)];
                kb[k] = kv[(cg + 8u * k) * BRAIN_FA_CH + ((c + 1) ^ cg)];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s[i][k] += brain_fa_dot4(qa[i], ka[k]);
                    s2[i][k] += brain_fa_dot4(qb[i], kb[k]);
                }
            }
        }

        // Each key's base-2 mask: -inf past the tile's keys, and for a
        // masked key whose mask is excluded.
        float mk[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned int j = cg + 8u * k;
            mk[k] = j < left ? 0.f : BRAIN_FA_NEG_INF;
            if (MASKED && j < left) { mk[k] = __ldg(a.kmask + j0 + j) * BRAIN_FA_LOG2E; }
        }

        // Online softmax in base 2.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = BRAIN_FA_NEG_INF;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s[i][k] += s2[i][k];
                if (MASKED && j0 + cg + 8u * k > q0 + row0 + i) { s[i][k] = BRAIN_FA_NEG_INF; }
                mx = fmaxf(mx, fmaf(s[i][k], c2, mk[k]));
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float mn = fmaxf(m[i], mx);
            // A row with no live key yet keeps exp2's argument finite: its
            // weights and its correction are 0 and 1, not NaN.
            const float base = mn == BRAIN_FA_NEG_INF ? 0.f : mn;
            const float corr = exp2f(m[i] - base);
            m[i] = mn;
            float ls = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s[i][k] = exp2f(fmaf(s[i][k], c2, mk[k] - base));
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
        brain_fa_stage(kv, pre);
        __syncthreads();  // v is visible
        if (t + 1 < ntiles) { brain_fa_fetch(pre, a, a.k, j0 + BRAIN_FA_BK, left - BRAIN_FA_BK); }

        // Output: key j = g + 8k's probabilities come from lane g of the row group.
        const unsigned int lane0 = lane & ~7u;
#pragma unroll 1
        for (unsigned int g = 0; g < 8; ++g) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const unsigned int j = g + 8u * k;
                float p[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { p[i] = __shfl_sync(0xffffffffu, s[i][k], lane0 + g); }
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const float4 v = kv[j * BRAIN_FA_CH + ((cg + 8u * d) ^ g)];
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
            brain_fa_stage(kv, pre);
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
        const unsigned int r = q0 + row0 + i;
        if (r >= T) { continue; }
        float* dst = a.out + (a.seq0 + r) * a.out_stride;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const unsigned int ch = 4u * (cg + 8u * d);
            const float4 v = make_float4(o[i][d].x * inv, o[i][d].y * inv, o[i][d].z * inv, o[i][d].w * inv);
            if (a.vec) {
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

extern "C" __global__ void __launch_bounds__(BRAIN_FA_THREADS, 2)
brain_flash_bidir_f32(const unsigned int* params, const float* qkv, float* out) {
    __shared__ float4 qs[BRAIN_FA_BQ * BRAIN_FA_CH];  // 32 KiB, chunk ^ ((row >> 2) & 7)
    __shared__ float4 kv[BRAIN_FA_BK * BRAIN_FA_CH];  // 16 KiB, chunk ^ (row & 7)
    const unsigned int bsz = params[0], heads = params[1], T = params[2];
    const unsigned int stride = params[4], q_off = params[5], k_off = params[6], v_off = params[7], d_model = params[8];
    const unsigned int qtiles = (T + BRAIN_FA_BQ - 1) / BRAIN_FA_BQ;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= bsz * heads * qtiles) { return; }
    const unsigned int qt = blk % qtiles, h = (blk / qtiles) % heads, b = blk / qtiles / heads;
    const unsigned int hoff = h * BRAIN_FA_HD;
    BrainFaRows a;
    a.q = qkv + q_off + hoff;
    a.k = qkv + k_off + hoff;
    a.v = qkv + v_off + hoff;
    a.kmask = nullptr;
    a.out = out + hoff;
    a.q_stride = stride;
    a.kv_stride = stride;
    a.out_stride = d_model;
    a.T = T;
    a.seq0 = (size_t)b * T;
    a.vec = ((reinterpret_cast<size_t>(qkv) | reinterpret_cast<size_t>(out)) & 15u) == 0 && ((stride | q_off | k_off | v_off | d_model) & 3u) == 0;
    brain_fa_core<false>(a, qt, qs, kv);
}

extern "C" __global__ void __launch_bounds__(BRAIN_FA_THREADS, 2)
brain_flash_gqa_kmask_f32(const unsigned int* params, const float* q, const float* k, const float* kmask, const float* v, float* out) {
    __shared__ float4 qs[BRAIN_FA_BQ * BRAIN_FA_CH];
    __shared__ float4 kv[BRAIN_FA_BK * BRAIN_FA_CH];
    const unsigned int bsz = params[0], heads = params[1], kv_heads = params[2], T = params[3], group = params[5];
    const unsigned int qtiles = (T + BRAIN_FA_BQ - 1) / BRAIN_FA_BQ;
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= bsz * heads * qtiles) { return; }
    // Heaviest query tiles first: under the causal bound the last tile of a
    // head attends the most keys, so it should not be the wave's straggler.
    const unsigned int qt = qtiles - 1 - blk % qtiles, h = (blk / qtiles) % heads, b = blk / qtiles / heads;
    BrainFaRows a;
    a.q = q + h * BRAIN_FA_HD;
    a.k = k + (h / group) * BRAIN_FA_HD;
    a.v = v + (h / group) * BRAIN_FA_HD;
    a.kmask = kmask;
    a.out = out + h * BRAIN_FA_HD;
    a.q_stride = heads * BRAIN_FA_HD;
    a.kv_stride = kv_heads * BRAIN_FA_HD;
    a.out_stride = heads * BRAIN_FA_HD;
    a.T = T;
    a.seq0 = (size_t)b * T;
    a.vec = ((reinterpret_cast<size_t>(q) | reinterpret_cast<size_t>(k) | reinterpret_cast<size_t>(v) | reinterpret_cast<size_t>(out)) & 15u) == 0;
    brain_fa_core<true>(a, qt, qs, kv);
}
