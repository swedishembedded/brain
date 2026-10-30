// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core inference kernels for linear-
// attention language models for its clients. If your team needs expertise in
// running a sequential recurrence over chunks at the tensor-core rate of the
// card instead of emulating fp32 products, then you can procure our services
// by sending an email to info@swedishembedded.com.
//
// Gated DeltaNet's across-chunk recurrence (`model::gdn` step 10) with bf16 MMA operands and an fp32 state, for the
// serving prefill. Same launch geometry as `gdn_chunk_loop_tc`; the four matrix products of a chunk
//
//   v_new = u_c - w_c @ state              out_c = qs_c @ state + intra_c @ v_new
//   state = state * exp(g_last) + kd_c^T @ v_new
//
// run as ONE bf16 `mma.sync.m16n8k16` per tile instead of three TF32 ones. The matrices a chunk multiplies are
// produced as bf16 by `gdn_chunk_prep_tok_bf16` (w, qs = scale * query * exp_g_cs, kd = key * exp(g_last - g_cs) and
// intra: rounded once where they are made instead of once per use, half the bytes through L2, no convert here); the
// accumulators and the state stay fp32. The state lives in the registers of the warp that owns it and only a bf16
// copy of it sits in shared memory as the B operand of the next chunk, so the rounding of a product is bounded by
// 2^-9 of its operands and does not compound through the state. Not bit-identical to the TF32 kernel; chosen by the
// measured selection of the GDN path (never by device name), the TF32 kernel stays the exact option.
//
// Eight warps a block: four 16-row tiles of the chunk, each split over two halves of the block's 32 value columns;
// the state update gives each warp two 16-row tiles of the 128 state rows. A head is split over four blocks.
//
// Contract (u32 params, then pointers; chunk-major layout, `bhc = chunk*bh + head`) as `gdn_chunk_loop_tc`, with the
// buffers reinterpreted: `key` holds kd, `query` holds qs, `w` holds w and `intra` holds intra, all bf16 with the
// element counts of the fp32 tensors they replace (so the same scratch buffers serve); `u`, `g_cs`, the states and
// `out` are fp32; `exp_g_cs` is not read. `out` is token-major when heads_tok != 0.
// dk = dv = 128 and c_len <= 64.
// No `__restrict__`: brain's device buffers alias by design.

#define D 128
#define MAXC 64
#define CS 32                 // value columns per block
#define SPLITS (D / CS)       // blocks per head
#define THREADS 256
#define WARPS_M 4             // warps down the chunk's rows
#define SAH (D + 8)           // bf16 row stride of the staged w / query / key tiles (ldmatrix rows 16 B apart mod 128 B: conflict-free)
#define SIH (MAXC + 8)        // bf16 row stride of the staged intra tile
#define SSH (CS + 8)          // bf16 row stride of the state copy and of v_new

// Dynamic shared memory: (3 * MAXC * SAH + MAXC * SIH + D * SSH + MAXC * SSH) * 2 bytes.

// Where one chunk of the walk lives: its `C` rows, the scratch row its operands start at (`intra` at `cc_base`), and where its output
// rows go (`out_chunk`, rows `out_stride` floats apart).
struct LoopChunk {
    unsigned int C;
    unsigned long long row_base, cc_base;
    float* out_chunk;
    unsigned long long out_stride;
};

struct LoopOperands {
    const unsigned short *kd, *qs, *w, *intra;
    const float *u, *g_cs;
};

// One block's walk: the chunks of one value head and one 32-column slice of its state, in order. The state is read from
// `state_in` before the first chunk and written to `state_out` after the last (both point at the head's [128][128]
// matrix; they may be the same memory, since a block reads and writes only its own slice). `chunk(ci)` places chunk `ci`.
template <class ChunkAt>
__device__ __forceinline__ void brain_gdn_loop_walk(const LoopOperands& op, const unsigned int n_chunks, const unsigned int j0, const float* state_in, float* state_out,
                                                    ChunkAt chunk_at) {
    const unsigned short *kd = op.kd, *qs = op.qs, *w = op.w, *intra = op.intra;
    const float *u = op.u, *g_cs = op.g_cs;
    const float* initial_state = state_in;
    float* final_state = state_out;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned short* sW = reinterpret_cast<unsigned short*>(smem_raw);  // w, [64][SAH]
    unsigned short* sQ = sW + MAXC * SAH;                              // scale * query * exp_g_cs
    unsigned short* sK = sQ + MAXC * SAH;                              // key * exp(g_last - g_cs), [chunk row][state row]
    unsigned short* sI = sK + MAXC * SAH;                              // intra, [64][SIH]
    unsigned short* sS = sI + MAXC * SIH;                              // bf16 copy of the state slice, [128][SSH]
    unsigned short* sV = sS + D * SSH;                                 // bf16 v_new, [64][SSH]

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int wr = warp % WARPS_M, wc = warp / WARPS_M;
    const unsigned int m0 = wr * 16u;   // this warp's chunk rows
    const unsigned int n0 = wc * 16u;   // and its first column of the block's slice (two 8-column MMA tiles)

    // ldmatrix lane addressing: matrix q = lane >> 3, row l8 = lane & 7.
    const unsigned int q = lane >> 3, l8 = lane & 7u;
    // A operand, tile stored [m][k] (w, query, intra): matrices (m,k) (m+8,k) (m,k+8) (m+8,k+8) = a0..a3.
    const unsigned int a_row = (q & 1u) * 8u + l8, a_col = (q >> 1) * 8u;
    // A operand, tile stored [k][m] (key, loaded transposed).
    const unsigned int at_row = (q >> 1) * 8u + l8, at_col = (q & 1u) * 8u;
    // B operand, tile stored [k][n] (state, v_new, loaded transposed): (k,n) (k+8,n) (k,n+8) (k+8,n+8) = b0,b1 of tile 0 then of tile 1.
    const unsigned int b_row = (q & 1u) * 8u + l8, b_col = (q >> 1) * 8u;

    // The state this warp owns, in fp32: state rows (wr + 4*mi)*16 + g (+8), columns n0 + 8*nt + 2*t4 (+1).
    float st[2][2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int p = (wr + 4u * mi) * 16u + g + ((i & 2) ? 8u : 0u);
                st[mi][nt][i] = initial_state[(unsigned long long)p * D + j0 + n0 + nt * 8 + 2 * t4 + (i & 1)];
            }
        }
    }

    // A chunk's tiles are fetched into registers one phase ahead of their use: w and qs for chunk ci + 1 while chunk
    // ci's second half runs, kd and intra for chunk ci while its first half runs, so the loads overlap tensor work
    // instead of stalling it. Rows past C load zeros.
    constexpr int NV = MAXC * (D / 8) / THREADS;  // 16-byte vectors (8 bf16) a thread stages of each 64 x 128 tile
    constexpr int NI = MAXC * MAXC / 8 / THREADS;  // and of the 64 x 64 intra tile
    uint4 wv[NV], qv[NV], kv[NV], iv[NI];
    auto fetch_wq = [&](unsigned int ci) {
        const LoopChunk ch = chunk_at(ci);
        const unsigned int C = ch.C;
        const unsigned long long row_base = ch.row_base;
#pragma unroll
        for (int it = 0; it < NV; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (D / 8), c8 = e % (D / 8);
            wv[it] = make_uint4(0u, 0u, 0u, 0u);
            qv[it] = wv[it];
            if (r < C) {
                const unsigned long long at = (row_base + r) * D + c8 * 8;
                wv[it] = *reinterpret_cast<const uint4*>(w + at);
                qv[it] = *reinterpret_cast<const uint4*>(qs + at);
            }
        }
    };
    auto fetch_ki = [&](unsigned int ci) {
        const LoopChunk ch = chunk_at(ci);
        const unsigned int C = ch.C;
        const unsigned long long row_base = ch.row_base;
#pragma unroll
        for (int it = 0; it < NV; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (D / 8), c8 = e % (D / 8);
            kv[it] = r < C ? *reinterpret_cast<const uint4*>(kd + (row_base + r) * D + c8 * 8) : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int it = 0; it < NI; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (MAXC / 8), c8 = (e % (MAXC / 8)) * 8;
            if (C == MAXC) {  // rows of 128 bytes: aligned vectors
                iv[it] = *reinterpret_cast<const uint4*>(intra + ch.cc_base + (unsigned long long)r * C + c8);
            } else {          // a short chunk's rows are not 16-byte aligned: gather the eight elements
                unsigned short h[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { h[j] = (r < C && c8 + j < C) ? intra[ch.cc_base + (unsigned long long)r * C + c8 + j] : (unsigned short)0; }
                iv[it] = make_uint4(h[0] | (unsigned int)h[1] << 16, h[2] | (unsigned int)h[3] << 16, h[4] | (unsigned int)h[5] << 16, h[6] | (unsigned int)h[7] << 16);
            }
        }
    };

    fetch_wq(0);
    for (unsigned int ci = 0; ci < n_chunks; ++ci) {
        const LoopChunk ch = chunk_at(ci);
        const unsigned int C = ch.C;
        const unsigned int c16 = (C + 15u) & ~15u;  // contraction length over chunk rows (staged rows past C are zero)
        const unsigned long long row_base = ch.row_base;
        const unsigned long long off_dk = row_base * D;
        const float g_last = g_cs[row_base + C - 1];  // (unused beyond the decay below)
        float* out_chunk = ch.out_chunk;
        const unsigned long long out_stride = ch.out_stride;

        // Publish the state entering this chunk: the bf16 B operand of the products below, and the training history.
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const unsigned int p = (wr + 4u * mi) * 16u + g + h2 * 8u;
                    const unsigned int col = n0 + nt * 8 + 2 * t4;
                    *reinterpret_cast<unsigned int*>(sS + p * SSH + col) = brain_pack_bf16(st[mi][nt][2 * h2], st[mi][nt][2 * h2 + 1]);
                }
            }
        }
#pragma unroll
        for (int it = 0; it < NV; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (D / 8), c8 = e % (D / 8);
            *reinterpret_cast<uint4*>(sW + r * SAH + c8 * 8) = wv[it];
            *reinterpret_cast<uint4*>(sQ + r * SAH + c8 * 8) = qv[it];
        }
        __syncthreads();

        fetch_ki(ci);
        // u is read in the accumulator layout.
        float uv[2][4];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int r = m0 + g + ((i & 2) ? 8u : 0u);
                uv[t][i] = r < C ? u[off_dk + (unsigned long long)r * D + j0 + n0 + t * 8 + 2 * t4 + (i & 1)] : 0.0f;
            }
        }

        // v_new = u - w @ state and the inter-chunk output (scaled query) @ state, both from the entering state.
        float acc[2][4], acc2[2][4];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[t][i] = 0.0f; acc2[t][i] = 0.0f; }
        }
#pragma unroll 2
        for (unsigned int k0 = 0; k0 < D; k0 += 16) {
            unsigned int aw[4], aq[4], b[4];
            brain_ldmatrix_x4(aw, sW + (m0 + a_row) * SAH + k0 + a_col);
            brain_ldmatrix_x4(aq, sQ + (m0 + a_row) * SAH + k0 + a_col);
            brain_ldmatrix_x4_trans(b, sS + (k0 + b_row) * SSH + n0 + b_col);
            brain_mma_bf16(acc[0], aw, b[0], b[1]);
            brain_mma_bf16(acc[1], aw, b[2], b[3]);
            brain_mma_bf16(acc2[0], aq, b[0], b[1]);
            brain_mma_bf16(acc2[1], aq, b[2], b[3]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                *reinterpret_cast<unsigned int*>(sV + r * SSH + n0 + t * 8 + 2 * t4) = brain_pack_bf16(uv[t][2 * h2] - acc[t][2 * h2], uv[t][2 * h2 + 1] - acc[t][2 * h2 + 1]);
            }
        }
        // The kd and intra tiles of this chunk (fetched above) for the second half.
#pragma unroll
        for (int it = 0; it < NV; ++it) {
            const unsigned int e = tid + it * THREADS;
            const unsigned int r = e / (D / 8), c8 = e % (D / 8);
            *reinterpret_cast<uint4*>(sK + r * SAH + c8 * 8) = kv[it];
        }
#pragma unroll
        for (int it = 0; it < NI; ++it) {
            const unsigned int e = tid + it * THREADS;
            *reinterpret_cast<uint4*>(sI + (e / (MAXC / 8)) * SIH + (e % (MAXC / 8)) * 8) = iv[it];
        }
        __syncthreads();

        if (ci + 1 < n_chunks) { fetch_wq(ci + 1); }

        // out += intra @ v_new.
        for (unsigned int k0 = 0; k0 < c16; k0 += 16) {
            unsigned int ai[4], b[4];
            brain_ldmatrix_x4(ai, sI + (m0 + a_row) * SIH + k0 + a_col);
            brain_ldmatrix_x4_trans(b, sV + (k0 + b_row) * SSH + n0 + b_col);
            brain_mma_bf16(acc2[0], ai, b[0], b[1]);
            brain_mma_bf16(acc2[1], ai, b[2], b[3]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int r = m0 + g + h2 * 8u;
                if (r < C) {
                    *reinterpret_cast<float2*>(out_chunk + (unsigned long long)r * out_stride + j0 + n0 + t * 8 + 2 * t4) = make_float2(acc2[t][2 * h2], acc2[t][2 * h2 + 1]);
                }
            }
        }

        // state = state * exp(g_last) + decayed_key^T @ v_new, in the registers that own the state.
        const float decay = expf(g_last);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            float up[2][4];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { up[t][i] = 0.0f; }
            }
            const unsigned int p0 = (wr + 4u * mi) * 16u;
            for (unsigned int c0 = 0; c0 < c16; c0 += 16) {
                unsigned int ak[4], b[4];
                brain_ldmatrix_x4_trans(ak, sK + (c0 + at_row) * SAH + p0 + at_col);
                brain_ldmatrix_x4_trans(b, sV + (c0 + b_row) * SSH + n0 + b_col);
                brain_mma_bf16(up[0], ak, b[0], b[1]);
                brain_mma_bf16(up[1], ak, b[2], b[3]);
            }
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { st[mi][t][i] = st[mi][t][i] * decay + up[t][i]; }
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const unsigned int p = (wr + 4u * mi) * 16u + g + h2 * 8u;
                const float2 v = make_float2(st[mi][nt][2 * h2], st[mi][nt][2 * h2 + 1]);
                const unsigned int col = j0 + n0 + nt * 8 + 2 * t4;
                *reinterpret_cast<float2*>(final_state + (unsigned long long)p * D + col) = v;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS, 2) brain_gdn_chunk_loop_bf16(const unsigned int* params, const unsigned short* kd, const unsigned short* qs,
                                                                                 const unsigned short* w, const float* u, const float* g_cs, const float* exp_g_cs,
                                                                                 const unsigned short* intra, const float* initial_state, float* out, float* final_state) {
    const unsigned int bh = params[0];
    const unsigned int C = params[1];
    const unsigned int n_chunks = params[4];
    const unsigned int heads_tok = params[6];

    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bhi = blk / SPLITS;
    const unsigned int j0 = (blk % SPLITS) * CS;
    if (bhi >= bh) { return; }  // block-uniform

    const LoopOperands op = {kd, qs, w, intra, u, g_cs};
    auto chunk_at = [&](unsigned int ci) {
        LoopChunk ch;
        ch.C = C;
        ch.row_base = ((unsigned long long)ci * bh + bhi) * C;
        ch.cc_base = ch.row_base * C;
        ch.out_chunk = out + ch.row_base * D;
        ch.out_stride = D;
        if (heads_tok != 0) {
            const unsigned long long tok0 = ((unsigned long long)(bhi / heads_tok) * n_chunks + ci) * C;
            ch.out_chunk = out + (tok0 * heads_tok + bhi % heads_tok) * D;
            ch.out_stride = (unsigned long long)heads_tok * D;
        }
        return ch;
    };
    brain_gdn_loop_walk(op, n_chunks, j0, initial_state + (unsigned long long)bhi * D * D, final_state + (unsigned long long)bhi * D * D, chunk_at);
}

// The recurrence of every prompt segment of a round in one launch, on the state rows of the pools themselves: block
// (segment, head, column slice) walks the segment's chunks from the pool's state row and leaves the end state in the same
// row, so a round needs no gather or scatter of the state around the kernel. `table` is the chunk table of
// `brain_gdn_chunk_prep_seg_bf16` (a chunk's operands sit at a 64-row slot of the scratch, its output rows at its token
// rows of the token-major `out`).
// Contract: params [segments * H * SPLITS, unused..., H at 6, nseg at 7, state words per pool row at 8]; buffers kd, qs, w, u, g_cs,
// intra, table, state pool (read and written in place), out.
extern "C" __global__ void __launch_bounds__(THREADS, 2) brain_gdn_chunk_loop_seg_bf16(const unsigned int* params, const unsigned short* kd, const unsigned short* qs,
                                                                                     const unsigned short* w, const float* u, const float* g_cs,
                                                                                     const unsigned short* intra, const unsigned int* table, float* state_pool, float* out) {
    const unsigned int blocks = params[0];
    const unsigned int H = params[6], nseg = params[7];
    const unsigned long long state_words = params[8];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= blocks) { return; }  // block-uniform
    const unsigned int seg = blk / (H * SPLITS), head = (blk / SPLITS) % H;
    const unsigned int j0 = (blk % SPLITS) * CS;
    const unsigned int* sg = table + 4 * seg;
    const unsigned int rows = sg[1], slot = sg[2], chunk0 = sg[3];
    const unsigned int n_chunks = (rows + MAXC - 1) / MAXC;
    const unsigned int* chunks = table + 4 * nseg;

    const LoopOperands op = {kd, qs, w, intra, u, g_cs};
    auto chunk_at = [&](unsigned int ci) {
        const unsigned int global = chunk0 + ci;
        LoopChunk ch;
        ch.C = chunks[2 * global + 1];
        ch.row_base = ((unsigned long long)global * H + head) * MAXC;
        ch.cc_base = ch.row_base * MAXC;
        ch.out_stride = (unsigned long long)H * D;
        ch.out_chunk = out + ((unsigned long long)chunks[2 * global] * H + head) * D;
        return ch;
    };
    float* state = state_pool + slot * state_words + (unsigned long long)head * D * D;
    brain_gdn_loop_walk(op, n_chunks, j0, state, state, chunk_at);
}
