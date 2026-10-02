// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency chunked linear-attention kernels
// for its clients. If your team needs expertise in collapsing a sequential
// GPU dependency chain of small launches into one persistent kernel, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// Gated DeltaNet's whole across-chunk recurrence (step 10 of `model::gdn`'s
// chunked forward) in ONE launch. The portable path walks the chunks on the
// host and issues nine dispatches per chunk - two `bmm`, two `bmm_acc`, a
// subtraction, two row scales, a decay and a state decay - behind each of which
// the next waits. At 256 rows that is 36 launches per layer on top of the
// whole-tensor ones, 1,700 per prefill round of the 64-layer model, each a few
// microseconds of work behind a launch that costs about as much.
//
// Per (head, chunk), with `state` [dk, dv] carried from chunk to chunk:
//   v_prime = w_c @ state                 v_new  = u_c - v_prime
//   out_c   = (query_c * exp_g_cs * scale) @ state  +  intra_c @ v_new
//   state   = state * exp(g_last)  +  (key_c * exp(g_last - g_cs))^T @ v_new
//
// Every matrix product is column-independent in dv, so a head is split across
// four blocks of 32 columns each (192 blocks for the model's 48 value heads,
// enough to fill the part), each keeping its 128 x 32 slice of the state in
// shared memory for all chunks.
//
// NUMERICS: bit-identical to the dispatch sequence it replaces. Every output is
// the same ascending-k fp32 sum starting from 0 (`acc = acc + a * b`, compiled
// without contraction), added to its previous value exactly where the portable
// kernels add it (`out = inter + intra`, `state = state * decay + update`), and
// the exponentials are `expf`, as the generated kernels use.
//
// Contract (u32 params, then pointers; every buffer is the chunk-major layout
// `model::gdn` documents, `bhc = chunk * (B*H) + b*H + h`):
//   params : [bh, c_len, dk, dv, n_chunks, scale (f32 bits)]
//   key, query, w : [n_chunks, bh, c_len, dk]   u, out : [n_chunks, bh, c_len, dv]
//   g_cs, exp_g_cs : [n_chunks, bh, c_len]      intra : [n_chunks, bh, c_len, c_len]
//   initial_state, final_state : [bh, dk, dv]
// dk = dv = 128 and c_len <= 64 (the provider checks; the entry does not).
//
// No `__restrict__`: brain's device buffers alias by design.

#define DK 128
#define DV 128
#define SLICE 32              // dv columns per block
#define SPLITS (DV / SLICE)   // blocks per head
#define MAXC 64               // largest chunk
#define KC 32                 // k-depth staged per step
#define THREADS 128
#define APAD (MAXC + 1)       // staged A tile row stride (rows of the 64-row GEMMs)
#define APAD4 (DK + 1)        // ... and of the 128-row state update

extern "C" __global__ void __launch_bounds__(THREADS) brain_gdn_chunk_loop_f32(const unsigned int* params,
                                                                               const float* key,
                                                                               const float* query,
                                                                               const float* w,
                                                                               const float* u,
                                                                               const float* g_cs,
                                                                               const float* exp_g_cs,
                                                                               const float* intra,
                                                                               const float* initial_state,
                                                                               float* out,
                                                                               float* final_state) {
    const unsigned int bh = params[0];
    const unsigned int C = params[1];
    const unsigned int n_chunks = params[4];
    const float scale = __uint_as_float(params[5]);

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bhi = blk / SPLITS;
    const unsigned int j0 = (blk % SPLITS) * SLICE;
    if (bhi >= bh) { return; }  // block-uniform

    __shared__ float st[DK * SLICE];           // state slice [k][j]
    __shared__ float vn[MAXC * SLICE];         // v_new [c][j]
    __shared__ float as[KC * APAD4];           // staged A tile, [kk][row]; sized for the larger of the two shapes

    const unsigned int tid = threadIdx.x;
    const unsigned int ty = tid >> 3;          // 0..15
    const unsigned int tx = tid & 7u;          // 0..7: columns 4*tx .. 4*tx+3

    for (unsigned int i = tid; i < DK * SLICE; i += THREADS) {
        const unsigned int k = i / SLICE, j = i % SLICE;
        st[i] = initial_state[((unsigned long long)bhi * DK + k) * DV + j0 + j];
    }
    __syncthreads();

    for (unsigned int ci = 0; ci < n_chunks; ++ci) {
        // Chunk ci's flat element offsets, as `gdn_chunk_fwd`'s own `off_*`.
        const unsigned long long row_base = ((unsigned long long)ci * bh + bhi) * C;   // first row of this (chunk, head)
        const unsigned long long off_dk = row_base * DK;
        const unsigned long long off_dv = row_base * DV;
        const unsigned long long off_g = row_base;
        const unsigned long long off_cc = row_base * C;

        float acc[4][4];

        // ---- v_new = u - w @ state   (rows 4*ty .. +3 of this block's 64 x 32 tile)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
        }
        for (unsigned int kc = 0; kc < DK; kc += KC) {
            for (unsigned int e = tid; e < MAXC * KC; e += THREADS) {
                const unsigned int r = e / KC, kk = e % KC;
                as[kk * APAD + r] = (r < C) ? w[off_dk + (unsigned long long)r * DK + kc + kk] : 0.0f;
            }
            __syncthreads();
#pragma unroll 8
            for (unsigned int kk = 0; kk < KC; ++kk) {
                const float a0 = as[kk * APAD + ty * 4 + 0], a1 = as[kk * APAD + ty * 4 + 1];
                const float a2 = as[kk * APAD + ty * 4 + 2], a3 = as[kk * APAD + ty * 4 + 3];
                const float* sr = &st[(kc + kk) * SLICE + tx * 4];
                const float b0 = sr[0], b1 = sr[1], b2 = sr[2], b3 = sr[3];
                acc[0][0] = acc[0][0] + a0 * b0; acc[0][1] = acc[0][1] + a0 * b1; acc[0][2] = acc[0][2] + a0 * b2; acc[0][3] = acc[0][3] + a0 * b3;
                acc[1][0] = acc[1][0] + a1 * b0; acc[1][1] = acc[1][1] + a1 * b1; acc[1][2] = acc[1][2] + a1 * b2; acc[1][3] = acc[1][3] + a1 * b3;
                acc[2][0] = acc[2][0] + a2 * b0; acc[2][1] = acc[2][1] + a2 * b1; acc[2][2] = acc[2][2] + a2 * b2; acc[2][3] = acc[2][3] + a2 * b3;
                acc[3][0] = acc[3][0] + a3 * b0; acc[3][1] = acc[3][1] + a3 * b1; acc[3][2] = acc[3][2] + a3 * b2; acc[3][3] = acc[3][3] + a3 * b3;
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int r = ty * 4 + i;
            if (r < C) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    vn[r * SLICE + tx * 4 + j] = u[off_dv + (unsigned long long)r * DV + j0 + tx * 4 + j] - acc[i][j];
                }
            }
        }

        // ---- out = (query * exp_g_cs * scale) @ state   (the OLD state)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
        }
        for (unsigned int kc = 0; kc < DK; kc += KC) {
            for (unsigned int e = tid; e < MAXC * KC; e += THREADS) {
                const unsigned int r = e / KC, kk = e % KC;
                // `gdn_row_scale_off`: alpha * x * s, left to right.
                as[kk * APAD + r] = (r < C) ? (scale * query[off_dk + (unsigned long long)r * DK + kc + kk]) * exp_g_cs[off_g + r] : 0.0f;
            }
            __syncthreads();
#pragma unroll 8
            for (unsigned int kk = 0; kk < KC; ++kk) {
                const float a0 = as[kk * APAD + ty * 4 + 0], a1 = as[kk * APAD + ty * 4 + 1];
                const float a2 = as[kk * APAD + ty * 4 + 2], a3 = as[kk * APAD + ty * 4 + 3];
                const float* sr = &st[(kc + kk) * SLICE + tx * 4];
                const float b0 = sr[0], b1 = sr[1], b2 = sr[2], b3 = sr[3];
                acc[0][0] = acc[0][0] + a0 * b0; acc[0][1] = acc[0][1] + a0 * b1; acc[0][2] = acc[0][2] + a0 * b2; acc[0][3] = acc[0][3] + a0 * b3;
                acc[1][0] = acc[1][0] + a1 * b0; acc[1][1] = acc[1][1] + a1 * b1; acc[1][2] = acc[1][2] + a1 * b2; acc[1][3] = acc[1][3] + a1 * b3;
                acc[2][0] = acc[2][0] + a2 * b0; acc[2][1] = acc[2][1] + a2 * b1; acc[2][2] = acc[2][2] + a2 * b2; acc[2][3] = acc[2][3] + a2 * b3;
                acc[3][0] = acc[3][0] + a3 * b0; acc[3][1] = acc[3][1] + a3 * b1; acc[3][2] = acc[3][2] + a3 * b2; acc[3][3] = acc[3][3] + a3 * b3;
            }
            __syncthreads();
        }

        // ---- out += intra @ v_new   (a second accumulator, added to `out` once, like `bmm_acc`)
        float acc2[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc2[i][j] = 0.0f; }
        }
        for (unsigned int kc = 0; kc < C; kc += KC) {
            for (unsigned int e = tid; e < MAXC * KC; e += THREADS) {
                const unsigned int r = e / KC, kk = e % KC;
                as[kk * APAD + r] = (r < C && kc + kk < C) ? intra[off_cc + (unsigned long long)r * C + kc + kk] : 0.0f;
            }
            __syncthreads();
            const unsigned int kn = min((unsigned int)KC, C - kc);
            for (unsigned int kk = 0; kk < kn; ++kk) {
                const float a0 = as[kk * APAD + ty * 4 + 0], a1 = as[kk * APAD + ty * 4 + 1];
                const float a2 = as[kk * APAD + ty * 4 + 2], a3 = as[kk * APAD + ty * 4 + 3];
                const float* vr = &vn[(kc + kk) * SLICE + tx * 4];
                const float b0 = vr[0], b1 = vr[1], b2 = vr[2], b3 = vr[3];
                acc2[0][0] = acc2[0][0] + a0 * b0; acc2[0][1] = acc2[0][1] + a0 * b1; acc2[0][2] = acc2[0][2] + a0 * b2; acc2[0][3] = acc2[0][3] + a0 * b3;
                acc2[1][0] = acc2[1][0] + a1 * b0; acc2[1][1] = acc2[1][1] + a1 * b1; acc2[1][2] = acc2[1][2] + a1 * b2; acc2[1][3] = acc2[1][3] + a1 * b3;
                acc2[2][0] = acc2[2][0] + a2 * b0; acc2[2][1] = acc2[2][1] + a2 * b1; acc2[2][2] = acc2[2][2] + a2 * b2; acc2[2][3] = acc2[2][3] + a2 * b3;
                acc2[3][0] = acc2[3][0] + a3 * b0; acc2[3][1] = acc2[3][1] + a3 * b1; acc2[3][2] = acc2[3][2] + a3 * b2; acc2[3][3] = acc2[3][3] + a3 * b3;
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int r = ty * 4 + i;
            if (r < C) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    out[off_dv + (unsigned long long)r * DV + j0 + tx * 4 + j] = acc[i][j] + acc2[i][j];
                }
            }
        }

        // ---- state = state * exp(g_last) + (key * exp(g_last - g_cs))^T @ v_new
        const float g_last = g_cs[off_g + C - 1];
        const float decay = expf(g_last);
        float up[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { up[i][j] = 0.0f; }
        }
        for (unsigned int kc = 0; kc < C; kc += KC) {
            for (unsigned int e = tid; e < KC * DK; e += THREADS) {
                const unsigned int cc = e / DK, m = e % DK;
                const unsigned int c = kc + cc;
                // `gdn_decay_scale` then `gdn_row_scale_off` (alpha = 1): 1 * x * s.
                as[cc * APAD4 + m] = (c < C) ? (1.0f * key[off_dk + (unsigned long long)c * DK + m]) * expf(g_last - g_cs[off_g + c]) : 0.0f;
            }
            __syncthreads();
            const unsigned int kn = min((unsigned int)KC, C - kc);
            for (unsigned int cc = 0; cc < kn; ++cc) {
                const float* vr = &vn[(kc + cc) * SLICE + tx * 4];
                const float b0 = vr[0], b1 = vr[1], b2 = vr[2], b3 = vr[3];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float a = as[cc * APAD4 + ty * 8 + i];
                    up[i][0] = up[i][0] + a * b0; up[i][1] = up[i][1] + a * b1; up[i][2] = up[i][2] + a * b2; up[i][3] = up[i][3] + a * b3;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float* s = &st[(ty * 8 + i) * SLICE + tx * 4 + j];
                *s = *s * decay + up[i][j];
            }
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < DK * SLICE; i += THREADS) {
        const unsigned int k = i / SLICE, j = i % SLICE;
        final_state[((unsigned long long)bhi * DK + k) * DV + j0 + j] = st[i];
    }
}
