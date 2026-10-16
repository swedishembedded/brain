// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core training kernels for linear-
// attention language models for its clients. If your team needs expertise in
// running the sequential gradient of a recurrent layer at tensor-core rate
// instead of on a single block of CUDA cores, then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// The reverse sweep of Gated DeltaNet's across-chunk recurrence (`model::gdn`'s
// `gdn_chunk_bwd`, "the sequential part") on tensor cores, split over value
// columns. With D the gradient of the loss with respect to the state a chunk
// leaves (D = d_final_state for the last chunk), each chunk, newest first, does
//
//   d_v_new   = intra^T @ d_out  +  decayed_k @ D          (rows: the chunk's tokens)
//   D_before  = D * exp(g_last) + q_scaled^T @ d_out - w^T @ d_v_new
//
// where decayed_k = key * decay_scale and q_scaled = scale * query * exp_g_cs.
// Every product is per value column: D's columns never mix (the transition acts
// on the key axis), so a head is split over four blocks of 32 columns exactly
// like the forward recurrence (`gdn_chunk_loop_tc`), 192 blocks for 48 heads
// instead of the 48 of the fp32 sweep. The one quantity that contracts over the
// columns, d_decayed_k = v_new @ D^T, does not feed back into the recurrence, so
// it is not computed here: the kernel writes the history of D (the state
// gradient entering each chunk's step) and `gdn_chunk_bwd_post_tc` forms
// d_decayed_k and the decay gradient from it.
//
// NUMERICS. Three-pass TF32 `mma.sync.m16n8k8` per product (about 20 mantissa
// bits), fp32 accumulation and fp32 state gradient: not bit-identical to
// `gdn_bwd_sweep_f32`, but within a few units in 1e-6 relative.
//
// Contract (u32 params, then pointers; chunk-major, `bhc = chunk * bh + head`):
//   params : [bh, c_len, dk, dv, n_chunks, scale (f32 bits)]
//   read   : key, query, w, d_out [n_chunks, bh, c_len, 128];
//            intra [n_chunks, bh, c_len, c_len]; g_cs, exp_g_cs, decay_scale_hist
//            [n_chunks, bh, c_len]; d_final_state [bh, 128, 128]
//   write  : d_v_new [n_chunks, bh, c_len, 128]      (the final value)
//            d_state_hist [n_chunks, bh, 128, 128]   (D entering chunk ci's step)
//            d_initial_state [bh, 128, 128]
// dk = dv = 128 and c_len <= 64 (the provider checks; the entry does not).
// Dynamic shared memory: SWEEP_SMEM_FLOATS floats.
//
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define MAXC 64
#define CS 32                 // value columns per block
#define MMA_NT (CS / 8)       // 8-column MMA tiles per block
#define SPLITS (D / CS)       // blocks per head
#define THREADS 128           // 4 warps, one 16-row tile of the chunk each
#define SA (D + 4)            // row stride of the staged A tiles (conflict-free fragment loads)
#define SS (CS + 8)           // row stride of the state slice and of the d_out / d_v_new slices

#define SWEEP_SMEM_FLOATS (2 * MAXC * SA + D * SS + 2 * MAXC * SS)

__device__ __forceinline__ unsigned int brain_tf32(float x) {
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void brain_split_tf32(float x, unsigned int& hi, unsigned int& lo) {
    hi = brain_tf32(x);
    lo = brain_tf32(x - __uint_as_float(hi));
}

__device__ __forceinline__ void brain_mma_tf32(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// acc(16 x 8*NT) += sum_{k < K} fa(r, k) * fb(k, n) for this warp's rows m0 .. and columns n0 .., as
// three TF32 products per tile (head*head + head*remainder + remainder*head). The caller zeroes acc.
template <int NT, class FA, class FB>
__device__ __forceinline__ void warp_mma_acc(float (&acc)[NT][4], unsigned int K, unsigned int m0, unsigned int n0, FA fa, FB fb) {
    const unsigned int lane = threadIdx.x & 31u, g = lane >> 2, t4 = lane & 3u;
    for (unsigned int k0 = 0; k0 < K; k0 += 8) {
        unsigned int ah[4], al[4];
        brain_split_tf32(fa(m0 + g, k0 + t4), ah[0], al[0]);
        brain_split_tf32(fa(m0 + g + 8, k0 + t4), ah[1], al[1]);
        brain_split_tf32(fa(m0 + g, k0 + t4 + 4), ah[2], al[2]);
        brain_split_tf32(fa(m0 + g + 8, k0 + t4 + 4), ah[3], al[3]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const unsigned int n = n0 + t * 8 + g;
            unsigned int bh0, bl0, bh1, bl1;
            brain_split_tf32(fb(k0 + t4, n), bh0, bl0);
            brain_split_tf32(fb(k0 + t4 + 4, n), bh1, bl1);
            brain_mma_tf32(acc[t], al, bh0, bh1);
            brain_mma_tf32(acc[t], ah, bl0, bl1);
            brain_mma_tf32(acc[t], ah, bh0, bh1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
brain_gdn_bwd_sweep_tc(const unsigned int* params, const float* key, const float* query, const float* w, const float* d_out, const float* intra,
                       const float* g_cs, const float* exp_g_cs, const float* ds_hist, const float* d_final_state, float* d_v_new, float* d_state_hist,
                       float* d_initial_state) {
    const unsigned int bh = params[0];
    const unsigned int C = params[1];
    const unsigned int n_chunks = params[4];
    const float scale = __uint_as_float(params[5]);

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bhi = blk / SPLITS;
    const unsigned int j0 = (blk % SPLITS) * CS;
    if (bhi >= bh) { return; }  // block-uniform

    extern __shared__ float smem[];
    float* sA = smem;                   // staged decayed key, then w, [64][SA]
    float* sB = sA + MAXC * SA;         // staged intra, then scaled query, [64][SA]
    float* sS = sB + MAXC * SA;         // this block's slice of the state gradient, [128][SS]
    float* sO = sS + D * SS;            // d_out columns j0 .., [64][SS]
    float* sV = sO + MAXC * SS;         // d_v_new columns j0 .., [64][SS]

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int m0 = warp * 16u;                  // this warp's chunk rows
    const unsigned int c8 = (C + 7u) & ~7u;              // contraction length over chunk rows

    for (unsigned int i = tid; i < D * CS; i += THREADS) {
        const unsigned int k = i / CS, j = i % CS;
        sS[k * SS + j] = d_final_state[((unsigned long long)bhi * D + k) * D + j0 + j];
    }
    __syncthreads();

    for (unsigned int step = 0; step < n_chunks; ++step) {
        const unsigned int ci = n_chunks - 1 - step;
        const unsigned long long row_base = ((unsigned long long)ci * bh + bhi) * C;
        const unsigned long long off_dk = row_base * D;
        const unsigned long long off_cc = row_base * C;
        const unsigned long long hist_base = ((unsigned long long)ci * bh + bhi) * D * D;

        // The state gradient entering this step, for the post kernel.
        for (unsigned int i = tid; i < D * CS; i += THREADS) {
            const unsigned int k = i / CS, j = i % CS;
            d_state_hist[hist_base + (unsigned long long)k * D + j0 + j] = sS[k * SS + j];
        }
        // Stage key * decay_scale into sA, intra into sB and this block's d_out columns into sO.
        for (unsigned int e = tid; e < MAXC * (D / 4); e += THREADS) {
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (r < C) {
                kv = *reinterpret_cast<const float4*>(key + off_dk + (unsigned long long)r * D + c4 * 4);
                const float s = ds_hist[row_base + r];
                kv = make_float4(kv.x * s, kv.y * s, kv.z * s, kv.w * s);
            }
            *reinterpret_cast<float4*>(sA + r * SA + c4 * 4) = kv;
        }
        for (unsigned int e = tid; e < MAXC * MAXC; e += THREADS) {
            const unsigned int r = e / MAXC, k = e % MAXC;
            sB[r * SA + k] = (r < C && k < C) ? intra[off_cc + (unsigned long long)r * C + k] : 0.0f;
        }
        for (unsigned int e = tid; e < MAXC * CS; e += THREADS) {
            const unsigned int r = e / CS, j = e % CS;
            sO[r * SS + j] = r < C ? d_out[off_dk + (unsigned long long)r * D + j0 + j] : 0.0f;
        }
        __syncthreads();

        // d_v_new = decayed_k @ D + intra^T @ d_out (the second product is the intra-chunk term).
        {
            float acc[MMA_NT][4];
#pragma unroll
            for (int t = 0; t < MMA_NT; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[t][i] = 0.0f; }
            }
            warp_mma_acc<MMA_NT>(acc, c8, m0, 0, [&](unsigned int j, unsigned int i) { return sB[i * SA + j]; }, [&](unsigned int i, unsigned int n) { return sO[i * SS + n]; });
            warp_mma_acc<MMA_NT>(acc, D, m0, 0, [&](unsigned int r, unsigned int k) { return sA[r * SA + k]; }, [&](unsigned int k, unsigned int n) { return sS[k * SS + n]; });
#pragma unroll
            for (int t = 0; t < MMA_NT; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                    const unsigned int col = t * 8 + 2 * t4 + (i & 1);
                    sV[r * SS + col] = r < C ? acc[t][i] : 0.0f;
                    if (r < C) { d_v_new[off_dk + (unsigned long long)r * D + j0 + col] = acc[t][i]; }
                }
            }
        }
        __syncthreads();

        // Restage: w into sA and scale * query * exp_g_cs into sB.
        for (unsigned int e = tid; e < MAXC * (D / 4); e += THREADS) {
            const unsigned int r = e / (D / 4), c4 = e % (D / 4);
            float4 wv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qv = wv;
            if (r < C) {
                wv = *reinterpret_cast<const float4*>(w + off_dk + (unsigned long long)r * D + c4 * 4);
                qv = *reinterpret_cast<const float4*>(query + off_dk + (unsigned long long)r * D + c4 * 4);
                const float ex = exp_g_cs[row_base + r];
                qv = make_float4((scale * qv.x) * ex, (scale * qv.y) * ex, (scale * qv.z) * ex, (scale * qv.w) * ex);
            }
            *reinterpret_cast<float4*>(sA + r * SA + c4 * 4) = wv;
            *reinterpret_cast<float4*>(sB + r * SA + c4 * 4) = qv;
        }
        __syncthreads();

        // D = D * exp(g_last) + q_scaled^T @ d_out - w^T @ d_v_new: 128 state rows, two 16-row tiles per warp.
        const float decay = expf(g_cs[row_base + C - 1]);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            float up[MMA_NT][4];
#pragma unroll
            for (int t = 0; t < MMA_NT; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { up[t][i] = 0.0f; }
            }
            const unsigned int p0 = (warp + 4u * mi) * 16u;
            warp_mma_acc<MMA_NT>(up, c8, p0, 0, [&](unsigned int p, unsigned int c) { return -sA[c * SA + p]; }, [&](unsigned int c, unsigned int n) { return sV[c * SS + n]; });
            warp_mma_acc<MMA_NT>(up, c8, p0, 0, [&](unsigned int p, unsigned int i) { return sB[i * SA + p]; }, [&](unsigned int i, unsigned int n) { return sO[i * SS + n]; });
#pragma unroll
            for (int t = 0; t < MMA_NT; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float* s = &sS[(p0 + g + ((i & 2) ? 8u : 0u)) * SS + t * 8 + 2 * t4 + (i & 1)];
                    *s = *s * decay + up[t][i];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < D * CS; i += THREADS) {
        const unsigned int k = i / CS, j = i % CS;
        d_initial_state[((unsigned long long)bhi * D + k) * D + j0 + j] = sS[k * SS + j];
    }
}
