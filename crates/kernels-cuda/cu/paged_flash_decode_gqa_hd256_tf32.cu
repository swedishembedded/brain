// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference serving for clients whose GPU decides how many users one
// card carries. If your team needs expertise in decode attention that runs at the memory roofline over any KV cache
// format, you can procure our services by sending an email to info@swedishembedded.com.
//
// Split-key fused GQA decode attention at head_dim 256 over an f32 paged pool (or a bf16 one with BRAIN_KV_BF16), on
// TF32 TENSOR CORES: the CUDA form of `paged_flash_decode_gqa_hd256.wgsl` (and its `kv=bf16` variant) at GRP =
// BRAIN_GRP, with its whole contract - the same uniform, bindings, grid and partial records - so
// `paged_flash_decode_gqa_combine` merges either kernel's output.
//
//   params       : [batch, n_heads, n_kv_heads, head_dim (256), group (BRAIN_GRP), block_size, max_bt, n_splits,
//                   tiles_per_split] u32
//   q            : [batch, n_heads * 256]                          f32
//   pool_k/pool_v: [num_blocks * block_size, n_kv_heads * 256]     f32 (or bf16: element 2w the low half of word w)
//   block_tables : [batch, max_bt] u32       seq_lens : [batch] u32 (live keys, the new token included)
//   part         : [batch, n_heads, n_splits, 258] f32  the unnormalised output (256), the running max, the sum
//
// The structure is `paged_flash_decode_gqa_hd256_kv8.cu`'s (read that file's header for the map): each warp walks
// its own 16-key chunks with its own running softmax, S = Q . K^T with the group padded to 16 rows, then
// O^T += V^T . P^T with the first product's accumulator as the second's B operand. What differs is the element:
// the products run on `mma.sync.m16n8k8` with TF32 operands, an f32 split into a TF32 high part and the TF32
// rounding of the remainder, and the three products hi.hi + hi.lo + lo.hi kept (about 22 significant bits, the
// f32 kernel's own bound). A bf16 value is exact in TF32, so a bf16 cache needs only the query and probabilities
// split (two products). Each lane reads contiguous 16-byte pieces of its rows, in halves so the loads and the
// accumulators fit the register file.
//
// NUMERICS. f32 scores, softmax, state and merge; not bit-identical to the WGSL kernel (another summation order),
// held to the same bound against exact f64 attention.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_GRP
#define BRAIN_GRP 6
#endif
#ifndef BRAIN_KV_BF16
#define BRAIN_KV_BF16 0
#endif

#define HD 256
#define TILE 128
#define WARPS 4
#define THREADS (WARPS * 32)
#define CHUNK 16
#define NEG_BIG (-3.4e38f)

#if BRAIN_KV_BF16
typedef unsigned short KvElem;
#else
typedef float KvElem;
#endif

__device__ __forceinline__ unsigned int brain_tf32(float x) {
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

// x as a TF32 pair: `hi` its TF32 rounding, `lo` the TF32 rounding of the remainder.
__device__ __forceinline__ void brain_split_tf32(float x, unsigned int& hi, unsigned int& lo) {
    hi = brain_tf32(x);
    lo = brain_tf32(x - __uint_as_float(hi));
}

__device__ __forceinline__ void brain_mma8(float* d, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int brain_row(const unsigned int* bt, unsigned int j, unsigned int block_size, unsigned int n_kv, unsigned int kvh) {
    return (bt[j / block_size] * block_size + (j % block_size)) * n_kv + kvh;
}

// Elements `e0 .. e0 + 7` (e0 a multiple of 8) of a pool row as f32, from two 16-byte loads (f32) or one (bf16).
__device__ __forceinline__ void brain_load8(const KvElem* row, unsigned int e0, float (&v)[8]) {
#if BRAIN_KV_BF16
    const uint4 w = *reinterpret_cast<const uint4*>(row + e0);
    const unsigned int ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(ws[i] << 16);
        v[2 * i + 1] = __uint_as_float(ws[i] & 0xffff0000u);
    }
#else
    const float4 a = *reinterpret_cast<const float4*>(row + e0), b = *reinterpret_cast<const float4*>(row + e0 + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
#endif
}

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

extern "C" __global__ void __launch_bounds__(THREADS) brain_paged_flash_decode_gqa_hd256_tf32(
    const unsigned int* params, const float* q, const KvElem* pool_k, const KvElem* pool_v, const unsigned int* block_tables,
    const unsigned int* seq_lens, float* part) {
    BRAIN_PDL_ENTRY();
    const unsigned int batch = params[0], n_heads = params[1], n_kv = params[2], block_size = params[5], max_bt = params[6];
    const unsigned int n_splits = params[7], tiles_per_split = params[8];
    const unsigned int wg = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int s = wg % n_splits;
    const unsigned int wg2 = wg / n_splits;
    const unsigned int kvh = wg2 % n_kv;
    const unsigned int b = wg2 / n_kv;
    if (b >= batch) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int gid = lane >> 2, tig = lane & 3u;

    // The query as the first product's A fragments at k-step st (logical k tig <-> channel 64 tig + 2 st, tig + 4 <->
    // 64 tig + 2 st + 1): (hi a0, hi a2, lo a0, lo a2) per lane; rows gid + 8 (the padding of the group) are zero.
    // The query fragments are dead once the walk is over, so the merge tile reuses their bytes (the two together
    // would pass the 48 KiB of static shared memory).
    __shared__ __align__(16) unsigned char smem[WARPS * 8 * HD * 4];
    uint4 (*qf)[32] = reinterpret_cast<uint4 (*)[32]>(smem);
    float (*red_o)[8][HD] = reinterpret_cast<float (*)[8][HD]>(smem);
    __shared__ float red_m[WARPS][8], red_l[WARPS][8];

    const float* qb = q + (static_cast<unsigned long long>(b) * n_heads + kvh * BRAIN_GRP) * HD;
    for (unsigned int e = tid; e < 32u * 32u; e += THREADS) {
        const unsigned int st = e >> 5, ln = e & 31u, g = ln >> 2, t = ln & 3u;
        uint4 f = make_uint4(0u, 0u, 0u, 0u);
        if (g < BRAIN_GRP) {
            const float2 x = *reinterpret_cast<const float2*>(qb + g * HD + t * 64u + st * 2u);
            brain_split_tf32(x.x, f.x, f.z);
            brain_split_tf32(x.y, f.y, f.w);
        }
        qf[st][ln] = f;
    }
    __syncthreads();

    const unsigned int tlen = seq_lens[b];
    const unsigned int k_begin = s * tiles_per_split * TILE;
    const unsigned int k_end = min(k_begin + tiles_per_split * TILE, tlen);
    const unsigned int* bt = block_tables + static_cast<unsigned long long>(b) * max_bt;
    const unsigned long long row_elems = HD;

    // O^T accumulators: channel tile t, rows gid (channel 32 gid + 2t) and gid + 8 (32 gid + 2t + 1), columns heads
    // 2 tig and 2 tig + 1.
    float o[16][4];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { o[t][i] = 0.f; }
    }
    float m_run = NEG_BIG, l_run = 0.f;

    for (unsigned int c0 = k_begin + warp * CHUNK; c0 < k_end; c0 += WARPS * CHUNK) {
        // ---- scores: key rows gid and gid + 8; this lane's channels 64 tig .. 64 tig + 63, in two halves.
        const KvElem* krow[2];
        bool klive[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const unsigned int j = c0 + gid + 8u * r;
            klive[r] = j < k_end;
            krow[r] = pool_k + (klive[r] ? static_cast<unsigned long long>(brain_row(bt, j, block_size, n_kv, kvh)) * row_elems : 0ull);
        }
        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float kv[2][32];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float v8[8];
                    if (klive[r]) {
                        brain_load8(krow[r], tig * 64u + half * 32u + p * 8u, v8);
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i) { v8[i] = 0.f; }
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) { kv[r][p * 8 + i] = v8[i]; }
                }
            }
#pragma unroll
            for (int ss = 0; ss < 16; ++ss) {
                const int st = half * 16 + ss;
                const uint4 a = qf[st][lane];
                float* acc[2] = {s0, s1};
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const float k0 = kv[r][2 * ss], k1 = kv[r][2 * ss + 1];
#if BRAIN_KV_BF16
                    const unsigned int b0 = __float_as_uint(k0), b1 = __float_as_uint(k1);  // bf16 is exact in TF32
                    brain_mma8(acc[r], a.x, 0u, a.y, 0u, b0, b1);
                    brain_mma8(acc[r], a.z, 0u, a.w, 0u, b0, b1);
#else
                    unsigned int h0, l0, h1, l1;
                    brain_split_tf32(k0, h0, l0);
                    brain_split_tf32(k1, h1, l1);
                    brain_mma8(acc[r], a.x, 0u, a.y, 0u, h0, h1);
                    brain_mma8(acc[r], a.x, 0u, a.y, 0u, l0, l1);
                    brain_mma8(acc[r], a.z, 0u, a.w, 0u, h0, h1);
#endif
                }
            }
        }

        // s0[0], s0[1]: keys 2 tig, 2 tig + 1; s1[0], s1[1]: keys 2 tig + 8, 2 tig + 9 (rows gid + 8 are padding).
        float sc[4] = {s0[0], s0[1], s1[0], s1[1]};
        bool live[4];
        float mx = NEG_BIG;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            live[r] = c0 + 2u * tig + (r & 1) + 8u * (r >> 1) < k_end;
            sc[r] = live[r] ? sc[r] * 0.0625f : NEG_BIG;
            mx = fmaxf(mx, sc[r]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float m_new = fmaxf(m_run, mx);
        const float corr = expf(m_run - m_new);
        m_run = m_new;
        float p[4];
        float psum = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            p[r] = live[r] ? expf(sc[r] - m_new) : 0.f;
            psum += p[r];
        }
        l_run = l_run * corr + psum;
        // P^T as the B operand of k-step 0 (keys 2 tig, 2 tig + 1) and 1 (keys 2 tig + 8, + 9), high and low parts.
        unsigned int ph[4], pl[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) { brain_split_tf32(p[r], ph[r], pl[r]); }

        const float ca = __shfl_sync(0xffffffffu, corr, (2u * tig) * 4u), cb = __shfl_sync(0xffffffffu, corr, (2u * tig + 1u) * 4u);
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            o[t][0] *= ca;
            o[t][1] *= cb;
            o[t][2] *= ca;
            o[t][3] *= cb;
        }

        // ---- values: rows of keys 2 tig, 2 tig + 1, 2 tig + 8, 2 tig + 9; this lane's channels 32 gid .. + 31, in
        // two halves of 16 (channel tiles 8 h .. 8 h + 7).
        const KvElem* vrow[4];
        bool vlive[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const unsigned int j = c0 + 2u * tig + (r & 1) + 8u * (r >> 1);
            vlive[r] = j < k_end;
            vrow[r] = pool_v + (vlive[r] ? static_cast<unsigned long long>(brain_row(bt, j, block_size, n_kv, kvh)) * row_elems : 0ull);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float vv[4][16];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int pp = 0; pp < 2; ++pp) {
                    float v8[8];
                    if (vlive[r]) {
                        brain_load8(vrow[r], gid * 32u + half * 16u + pp * 8u, v8);
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i) { v8[i] = 0.f; }
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) { vv[r][pp * 8 + i] = v8[i]; }
                }
            }
#pragma unroll
            for (int tt = 0; tt < 8; ++tt) {
                const int t = half * 8 + tt;
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    // A = V^T: (row gid = channel 2t, keys kk = tig <-> key 2 tig, tig + 4 <-> key 2 tig + 1) of this k-step.
                    const float x0 = vv[2 * ks][2 * tt], x1 = vv[2 * ks][2 * tt + 1];          // key 2 tig (+8)
                    const float y0 = vv[2 * ks + 1][2 * tt], y1 = vv[2 * ks + 1][2 * tt + 1];  // key 2 tig + 1 (+8)
#if BRAIN_KV_BF16
                    const unsigned int a0 = __float_as_uint(x0), a1 = __float_as_uint(x1), a2 = __float_as_uint(y0), a3 = __float_as_uint(y1);
                    brain_mma8(o[t], a0, a1, a2, a3, ph[2 * ks], ph[2 * ks + 1]);
                    brain_mma8(o[t], a0, a1, a2, a3, pl[2 * ks], pl[2 * ks + 1]);
#else
                    unsigned int h0, l0, h1, l1, h2, l2, h3, l3;
                    brain_split_tf32(x0, h0, l0);
                    brain_split_tf32(x1, h1, l1);
                    brain_split_tf32(y0, h2, l2);
                    brain_split_tf32(y1, h3, l3);
                    brain_mma8(o[t], h0, h1, h2, h3, ph[2 * ks], ph[2 * ks + 1]);
                    brain_mma8(o[t], h0, h1, h2, h3, pl[2 * ks], pl[2 * ks + 1]);
                    brain_mma8(o[t], l0, l1, l2, l3, ph[2 * ks], ph[2 * ks + 1]);
#endif
                }
            }
        }
    }

    // ---- merge the four warps. This lane holds heads 2 tig, 2 tig + 1 of channels 32 gid + 2t (+ 1).
    __syncthreads();  // every warp is past its last read of the query fragments the merge tile overwrites
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        const unsigned int ch = gid * 32u + 2u * t;
        red_o[warp][2u * tig][ch] = o[t][0];
        red_o[warp][2u * tig + 1u][ch] = o[t][1];
        red_o[warp][2u * tig][ch + 1u] = o[t][2];
        red_o[warp][2u * tig + 1u][ch + 1u] = o[t][3];
    }
    float l_quad = l_run + __shfl_xor_sync(0xffffffffu, l_run, 1);
    l_quad += __shfl_xor_sync(0xffffffffu, l_quad, 2);
    if (tig == 0u) {
        red_m[warp][gid] = m_run;
        red_l[warp][gid] = l_quad;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < BRAIN_GRP; ++g) {
        float mw[WARPS], m = NEG_BIG;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            mw[w] = red_m[w][g];
            m = fmaxf(m, mw[w]);
        }
        float f[WARPS], l = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            f[w] = expf(mw[w] - m);
            l += red_l[w][g] * f[w];
        }
        const unsigned long long base = ((static_cast<unsigned long long>(b) * n_heads + kvh * BRAIN_GRP + g) * n_splits + s) * (HD + 2u);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int ch = tid * 2u + h;
            float acc = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) { acc = fmaf(red_o[w][g][ch], f[w], acc); }
            part[base + ch] = acc;
        }
        if (tid == 0u) {
            part[base + HD] = m;
            part[base + HD + 1u] = l;
        }
    }
}
