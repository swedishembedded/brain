// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference serving for clients whose GPU decides how many users one
// card carries. If your team needs expertise in decode attention that runs at the memory roofline over a compact KV
// cache, you can procure our services by sending an email to info@swedishembedded.com.
//
// Split-key fused GQA decode attention at head_dim 256 over an 8-bit paged pool (int8, or FP8 E4M3 with
// BRAIN_KV_FP8), on TENSOR CORES: the CUDA form of the `kv=int8` and `kv=fp8` variants of
// `paged_flash_decode_gqa_hd256.wgsl` at GRP = BRAIN_GRP (at most 8), with their whole
// contract - the same uniform, bindings, grid and partial records - so `paged_flash_decode_gqa_combine` merges
// either kernel's output and the host cannot tell them apart.
//
//   params       : [batch, n_heads, n_kv_heads, head_dim (256), group (BRAIN_GRP), block_size, max_bt, n_splits,
//                   tiles_per_split] u32
//   q            : [batch, n_heads * 256]                          f32
//   pool_k/pool_v: [num_blocks * block_size, n_kv_heads * 256]     int8 or FP8 E4M3 bytes
//   block_tables : [batch, max_bt] u32       seq_lens : [batch] u32 (live keys, the new token included)
//   part         : [batch, n_heads, n_splits, 258] f32  the unnormalised output (256), the running max, the sum
//   k_scales/v_scales: [num_blocks * block_size, n_kv_heads] f32, one scale per (token, kv head) row
//
// WHY. The generated form of the WGSL kernel gives one key to one thread and widens every int8 with a convert; it
// read the cache at about 5 percent of HBM bandwidth on a GH200 (122 us per layer at 12.8k keys). A CUDA-core
// rewrite with whole-row loads was still bound by instruction latency (about 1,500 dependent FMAs per key and kv
// head for the scores and as many for the values, with two warps per scheduler). Here both products run on
// `mma.sync.m16n8k16` (f16 in, f32 accumulate), so a key costs a few tensor instructions.
//
// THE MAP. One block per (sequence, kv head, key split), 4 warps. Each warp walks its own 16-key chunks of the split
// (chunks w, w + 4, ...) with its own running softmax, and the four warps merge once at the end - no barrier per
// tile. Per chunk:
//   S[heads, keys] = Q[heads (16, the group padded), 256] . K^T[256, 16 keys]    16 k-steps x 2 key tiles
//   O^T[256, heads] += V^T[256, 16 keys] . P^T[16 keys, heads (8)]               16 channel tiles
// The first product's accumulator IS the second's B operand (row = head, columns = the quad's keys), so P never
// leaves registers. The order of channels inside a dot product, and of keys inside a chunk, is free; it is chosen so
// that every lane reads CONTIGUOUS bytes: a lane loads 64 bytes of each of its two key rows (channels 64 tig ..) and
// 32 bytes of each of its four value rows (channels 32 gid ..), all as 16-byte loads, issued before any is used.
//
// NUMERICS. An int8 value is exact in f16 (byte permute into the mantissa of 1024, one subtract). The f32 query and
// the f32 probabilities (times the value row scale) are split into an f16 high part and the f16 rounding of the
// remainder, and both parts are multiplied: the products then carry about 22 significant bits, the f32 kernel's
// bound, at twice the tensor instructions (which are not the limit). Scores, softmax, the running state and the
// merge are f32. Not bit-identical to the WGSL kernel (another summation order); both are gated against exact f64
// attention at the same bound.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_GRP
#define BRAIN_GRP 6
#endif
// The 8-bit element format of the pool: 0 = int8 (the `kv=int8` tier), 1 = FP8 E4M3 (`kv=fp8`). Both carry one f32
// scale per (token, kv head) row and both are exact in f16; only the widening differs.
#ifndef BRAIN_KV_FP8
#define BRAIN_KV_FP8 0
#endif

#define HD 256
#define TILE 128
#define WARPS 4
#define THREADS (WARPS * 32)
#define CHUNK 16
#define NEG_BIG (-3.4e38f)

// The two bytes `sel` picks of an int8 word whose sign bits are flipped, as an f16 pair (exact): the byte lands in
// the mantissa of 1024, which carries it as 1024 + v + 128, and one subtract of 1152 leaves v.
__device__ __forceinline__ unsigned int brain_i8h2(unsigned int u, unsigned int sel) {
    const unsigned int h = __byte_perm(u, 0x64646464u, sel);
    unsigned int r;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(h), "r"(0x64806480u));
    return r;
}

// The four 8-bit elements of `w` (byte 0 first) as two f16 pairs: `lo` = (element 0, 1), `hi` = (element 2, 3).
__device__ __forceinline__ void brain_8bit_pairs(unsigned int w, unsigned int& lo, unsigned int& hi) {
#if BRAIN_KV_FP8
    const unsigned short l = static_cast<unsigned short>(w & 0xffffu), h = static_cast<unsigned short>(w >> 16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(lo) : "h"(l));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(hi) : "h"(h));
#else
    const unsigned int u = w ^ 0x80808080u;
    lo = brain_i8h2(u, 0x5140u);
    hi = brain_i8h2(u, 0x5342u);
#endif
}

// f32 -> nearest f16 -> f32.
__device__ __forceinline__ float brain_f16_round(float x) {
    float r;
    asm("{ .reg .f16 h;\n cvt.rn.f16.f32 h, %1;\n cvt.f32.f16 %0, h;\n }\n" : "=f"(r) : "f"(x));
    return r;
}

// a and b as f16 pairs (low half a, high half b): `hi` their f16 roundings, `lo` the f16 roundings of the
// remainders, so hi + lo carries about 22 significant bits of each.
__device__ __forceinline__ void brain_split2(float a, float b, unsigned int& hi, unsigned int& lo) {
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(hi) : "f"(b), "f"(a));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(lo) : "f"(b - brain_f16_round(b)), "f"(a - brain_f16_round(a)));
}

__device__ __forceinline__ void brain_mma(float* d, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The pool row (slot * n_kv + kv head) of key `j` of the sequence whose block table is `bt`.
__device__ __forceinline__ unsigned int brain_row(const unsigned int* bt, unsigned int j, unsigned int block_size, unsigned int n_kv, unsigned int kvh) {
    return (bt[j / block_size] * block_size + (j % block_size)) * n_kv + kvh;
}

extern "C" __global__ void __launch_bounds__(THREADS) brain_paged_flash_decode_gqa_hd256_kv8(
    const unsigned int* params, const float* q, const signed char* pool_k, const signed char* pool_v, const unsigned int* block_tables,
    const unsigned int* seq_lens, float* part, const float* k_scales, const float* v_scales) {
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

    // The query as the first product's A fragments, high and low parts: [k-step][lane] -> (hi a0, hi a2, lo a0, lo a2)
    // (rows gid + 8, the padding of the group to 16, are zero and never stored).
    __shared__ uint4 qf[16][32];
    __shared__ __align__(16) float red_o[WARPS][8][HD];
    __shared__ float red_m[WARPS][8], red_l[WARPS][8];

    const float* qb = q + (static_cast<unsigned long long>(b) * n_heads + kvh * BRAIN_GRP) * HD;
    for (unsigned int e = tid; e < 16u * 32u; e += THREADS) {
        const unsigned int st = e >> 5, ln = e & 31u, g = ln >> 2, t = ln & 3u;
        uint4 f = make_uint4(0u, 0u, 0u, 0u);
        if (g < BRAIN_GRP) {
            const float4 x = *reinterpret_cast<const float4*>(qb + g * HD + t * 64u + st * 4u);
            brain_split2(x.x, x.y, f.x, f.z);
            brain_split2(x.z, x.w, f.y, f.w);
        }
        qf[st][ln] = f;
    }
    __syncthreads();

    const unsigned int tlen = seq_lens[b];
    const unsigned int k_begin = s * tiles_per_split * TILE;
    const unsigned int k_end = min(k_begin + tiles_per_split * TILE, tlen);
    const unsigned int* bt = block_tables + static_cast<unsigned long long>(b) * max_bt;

    // O^T accumulators: channel tile t, rows gid (channel 32 gid + 2t) and gid + 8 (channel 32 gid + 2t + 1), columns
    // heads 2 tig and 2 tig + 1.
    float o[16][4];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { o[t][i] = 0.f; }
    }
    float m_run = NEG_BIG, l_run = 0.f;  // head gid, this lane's share of the sum

    for (unsigned int c0 = k_begin + warp * CHUNK; c0 < k_end; c0 += WARPS * CHUNK) {
        // ---- loads: key rows gid and gid + 8 (64 bytes each), value rows 2 tig, 2 tig + 1, 2 tig + 8, 2 tig + 9 (32 bytes each).
        uint4 kw[2][4];
        uint4 vw[4][2];
        float ksc[4], vsc[4];
        bool live[4];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const unsigned int j = c0 + gid + 8u * r;
            if (j < k_end) {
                const uint4* src = reinterpret_cast<const uint4*>(pool_k + static_cast<unsigned long long>(brain_row(bt, j, block_size, n_kv, kvh)) * HD + tig * 64u);
#pragma unroll
                for (int i = 0; i < 4; ++i) { kw[r][i] = src[i]; }
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) { kw[r][i] = make_uint4(0u, 0u, 0u, 0u); }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const unsigned int j = c0 + 2u * tig + (r & 1) + 8u * (r >> 1);
            live[r] = j < k_end;
            if (live[r]) {
                const unsigned int row = brain_row(bt, j, block_size, n_kv, kvh);
                const uint4* src = reinterpret_cast<const uint4*>(pool_v + static_cast<unsigned long long>(row) * HD + gid * 32u);
                vw[r][0] = src[0];
                vw[r][1] = src[1];
                ksc[r] = k_scales[row];
                vsc[r] = v_scales[row];
            } else {
                vw[r][0] = make_uint4(0u, 0u, 0u, 0u);
                vw[r][1] = vw[r][0];
                ksc[r] = 0.f;
                vsc[r] = 0.f;
            }
        }

        // ---- scores: S[head gid, keys 2 tig, 2 tig + 1] (sc[0..1]) and [.., keys 2 tig + 8, + 9] (sc[2..3]).
        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int st = 0; st < 16; ++st) {
            const uint4 a = qf[st][lane];
            // Key word of k-step st: channels 64 tig + 4 st .. + 3 of the lane's row.
            unsigned int b00, b01, b10, b11;
            brain_8bit_pairs(reinterpret_cast<const unsigned int*>(&kw[0][st >> 2])[st & 3], b00, b01);
            brain_8bit_pairs(reinterpret_cast<const unsigned int*>(&kw[1][st >> 2])[st & 3], b10, b11);
            brain_mma(s0, a.x, 0u, a.y, 0u, b00, b01);
            brain_mma(s0, a.z, 0u, a.w, 0u, b00, b01);
            brain_mma(s1, a.x, 0u, a.y, 0u, b10, b11);
            brain_mma(s1, a.z, 0u, a.w, 0u, b10, b11);
        }
        // s0[0], s0[1]: keys 2 tig, 2 tig + 1; s1[0], s1[1]: keys 2 tig + 8, 2 tig + 9 (rows gid + 8 are padding).
        float sc[4] = {s0[0], s0[1], s1[0], s1[1]};
        float mx = NEG_BIG;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            sc[r] = live[r] ? sc[r] * ksc[r] * 0.0625f : NEG_BIG;
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
        unsigned int ph0, pl0, ph1, pl1;
        brain_split2(p[0] * vsc[0], p[1] * vsc[1], ph0, pl0);
        brain_split2(p[2] * vsc[2], p[3] * vsc[3], ph1, pl1);

        // ---- values: rescale by each column head's correction (held by the quad of that head), then accumulate.
        const float ca = __shfl_sync(0xffffffffu, corr, (2u * tig) * 4u), cb = __shfl_sync(0xffffffffu, corr, (2u * tig + 1u) * 4u);
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            o[t][0] *= ca;
            o[t][1] *= cb;
            o[t][2] *= ca;
            o[t][3] *= cb;
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            // Channels 32 gid + 2t (row gid) and + 1 (row gid + 8): bytes 2t, 2t + 1 of the lane's 32-byte value spans.
            const int wi = t >> 1, sh = (t & 1) * 2;
            unsigned int x[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) { x[r] = reinterpret_cast<const unsigned int*>(&vw[r][wi >> 2])[wi & 3]; }
            // [key r_a byte sh, key r_b byte sh, key r_a byte sh + 1, key r_b byte sh + 1]
            const unsigned int sel = static_cast<unsigned int>(sh) | ((sh + 4u) << 4) | ((sh + 1u) << 8) | ((sh + 5u) << 12);
            const unsigned int p01 = __byte_perm(x[0], x[1], sel), p23 = __byte_perm(x[2], x[3], sel);
            unsigned int a0, a1, a2, a3;
            brain_8bit_pairs(p01, a0, a1);
            brain_8bit_pairs(p23, a2, a3);
            brain_mma(o[t], a0, a1, a2, a3, ph0, ph1);
            brain_mma(o[t], a0, a1, a2, a3, pl0, pl1);
        }
    }

    // ---- merge the four warps. This lane holds heads 2 tig, 2 tig + 1 of channels 32 gid + 2t (+ 1).
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
