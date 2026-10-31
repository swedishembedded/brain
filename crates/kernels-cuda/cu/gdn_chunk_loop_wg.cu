// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tensor-core inference kernels for linear-attention language models for its clients.
// If your team needs expertise in running the sequential state recurrence of a gated delta rule on Hopper warpgroup
// tensor cores with the state held in registers, then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// Gated DeltaNet's across-chunk recurrence on Hopper warpgroup tensor cores (`wgmma`), reading the matrices the bf16
// prep (`gdn_chunk_prep_bf16`) writes: the same arithmetic as `gdn_chunk_loop_bf16` (bf16 operands, fp32 accumulation and
// state), restructured so that the state never visits shared memory.
//
// The state is held TRANSPOSED, S^T [value column][key], as the fp32 accumulator of the state-update product, so the
// registers that receive an update are exactly the registers the next chunk's products read: converted to bf16 pairs in
// place they are the register-resident A operand of
//
//   v_new^T = u^T - S^T w^T          out^T = S^T qs^T + v_new^T intra^T          S^T = S^T exp(g_last) + v_new^T kd
//
// (`w`, `qs`, `kd` and `intra` stay in shared memory as the B operands, read by descriptor), and the bf16 v_new^T the
// first product leaves is the register-resident A operand of the other two. Each chunk is therefore three groups of
// asynchronous products and one barrier-free epilogue; the old kernel's per-chunk publishing of the state to shared
// memory, its ldmatrix traffic and four block barriers are gone.
//
// One block per (segment or sequence, value head): 288 threads, two consumer warpgroups of 64 value columns each and a
// producer warp (so the whole kernel compiles under 224 registers a thread, not the 168 a third warpgroup would leave). The producer streams the chunk operands (`cp.async`, 128-byte swizzled tiles, zero rows past a
// short chunk) through a two-stage ring and publishes a stage with an mbarrier; the consumers release it when their
// products have drained.
//
// Contract: as `gdn_chunk_loop_bf16` / `gdn_chunk_loop_seg_bf16` (chunk-major scratch, bf16 `kd`, `qs`, `w`, `intra`, fp32
// `u` and `g_cs`; `exp_g_cs` is not read), dk = dv = 128, c_len <= 64. No `__restrict__`: brain's device buffers alias
// by design.

#define D 128
#define MAXC 64
#define WGT 256                // two warpgroups, each 64 value columns; every thread also stages the next chunk's operands
#define TILE_BYTES 16384        // a 64-row x 128-value bf16 tile: two 128-byte-swizzle atom columns of 8 KiB
#define INTRA_BYTES 8192
#define SU (D + 4)              // fp32 row stride of the staged u tile (conflict-free for the accumulator-layout reads)
#define U_BYTES (MAXC * SU * 4)
#define STAGE_W 0
#define STAGE_Q TILE_BYTES
#define STAGE_K (2 * TILE_BYTES)
#define STAGE_I (3 * TILE_BYTES)
#define STAGE_U (STAGE_I + INTRA_BYTES)
#define STAGE_DEC (STAGE_U + U_BYTES)
#define STAGE_BYTES (((STAGE_DEC + 16 + 1023) / 1024) * 1024)
#define OUT_BYTES 8192          // a consumer warpgroup's 64-token x 64-column bf16 output tile
#define OFF_OUT (2 * STAGE_BYTES)
#define OFF_BAR (OFF_OUT + 2 * OUT_BYTES)
#define LOOP_WG_SMEM_BYTES (OFF_BAR + 64 + 1024)

// Where one chunk of the walk lives (the `LoopChunk` of the mma.sync recurrence, which this kernel replaces on Hopper).
struct WgChunk {
    unsigned int C;
    unsigned long long row_base, cc_base;
    // The output rows: `out_row` is the element offset of the chunk's first row and head in `out`, `out_stride` the elements between rows.
    unsigned long long out_row, out_stride;
};

struct WgOperands {
    const unsigned short *kd, *qs, *w, *intra;
    const float *u, *g_cs;
    void* out;
    bool out_bf16;
};

// One block's walk over `n_chunks` chunks of one value head. The state is read from / written to `state` (bf16 elements when
// `state_bf16`), a [key][value column] matrix of the head.
template <class ChunkAt>
__device__ __forceinline__ void gdn_loop_wg_walk(const WgOperands& op, const unsigned int n_chunks, void* state, const bool state_bf16, ChunkAt chunk_at) {
    extern __shared__ unsigned char gw_smem_raw[];
    const unsigned int raw_addr = brain_smem_addr(gw_smem_raw);
    const unsigned int base = (raw_addr + 1023u) & ~1023u;
    unsigned char* const gen = gw_smem_raw + (base - raw_addr);
    const unsigned int tid = threadIdx.x;
    const unsigned short *kd = op.kd, *qs = op.qs, *w = op.w, *intra = op.intra;
    const float *u = op.u, *g_cs = op.g_cs;

    // Stage chunk `ci`'s operands into ring slot `ci & 1` with `cp.async` (one commit group). Three 64 x 128 bf16 tiles into
    // 128-byte-swizzled atom columns: 16-byte chunk c of row r lands at chunk (c & 7) ^ (r & 7) of atom column c >> 3. Rows past
    // the chunk's length read nothing and arrive as zeros.
    auto issue = [&](unsigned int ci) {
        const WgChunk ch = chunk_at(ci);
        const unsigned int stage = base + (ci & 1u) * STAGE_BYTES;
        const unsigned int C = ch.C;
        const unsigned long long rb = ch.row_base;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const unsigned int e = tid + it * WGT;
            const unsigned int r = e >> 4, c = e & 15u;
            const unsigned int dst = (c >> 3) * ATOM_BYTES + r * 128u + (((c & 7u) ^ (r & 7u)) << 4);
            const unsigned long long at = (rb + r) * D + c * 8;
            const unsigned int bytes = r < C ? 16u : 0u;
            gw_cp16(stage + STAGE_W + dst, w + (r < C ? at : 0), bytes);
            gw_cp16(stage + STAGE_Q + dst, qs + (r < C ? at : 0), bytes);
            gw_cp16(stage + STAGE_K + dst, kd + (r < C ? at : 0), bytes);
        }
        // u: fp32 rows of 128 into the padded rows the consumers read in accumulator layout.
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const unsigned int e = tid + it * WGT;
            const unsigned int r = e >> 5, c = e & 31u;
            gw_cp16(stage + STAGE_U + (r * SU + c * 4) * 4u, u + (r < C ? (rb + r) * D + c * 4 : 0), r < C ? 16u : 0u);
        }
        // intra: 64 x 64 bf16, one atom column. A full chunk's rows are 128 bytes (vector copies); a short chunk's rows are C
        // elements apart and gathered.
        if (C == MAXC) {
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const unsigned int e = tid + it * WGT;
                const unsigned int r = e >> 3, c = e & 7u;
                gw_cp16(stage + STAGE_I + r * 128u + (((c ^ r) & 7u) << 4), intra + ch.cc_base + r * 64u + c * 8, 16u);
            }
        } else {
            for (unsigned int e = tid; e < MAXC * 8; e += WGT) {
                const unsigned int r = e >> 3, c = e & 7u;
                unsigned short h[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { h[j] = (r < C && c * 8 + j < C) ? intra[ch.cc_base + (unsigned long long)r * C + c * 8 + j] : (unsigned short)0; }
                const uint4 v = make_uint4(h[0] | (unsigned int)h[1] << 16, h[2] | (unsigned int)h[3] << 16, h[4] | (unsigned int)h[5] << 16, h[6] | (unsigned int)h[7] << 16);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(stage + STAGE_I + r * 128u + (((c ^ r) & 7u) << 4)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
            }
        }
        if (tid == 0u) {
            const float dec = expf(g_cs[rb + C - 1]);
            asm volatile("st.shared.f32 [%0], %1;\n" ::"r"(stage + STAGE_DEC), "f"(dec));
        }
        gw_cp_commit();
    };
    // The copies of the chunk the next iteration works on have landed, and are visible to the tensor cores and to every thread.
    auto publish = [&]() {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        gw_fence_proxy();
        __syncthreads();
    };

    issue(0);
    publish();

    // ================= consumer warpgroups =================
    const unsigned int cwg = tid >> 7;                     // 0 or 1: value columns 64 cwg ..
    const unsigned int ct = tid & 127u;
    const unsigned int warp = ct >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2, t4 = lane & 3u;

    // The state, transposed: S^T [value column (m)][key (n)] as the accumulator of a 64 x 128 product. Element (j, e): column
    // 64 cwg + 16 warp + g + 8 (e >> 1), key 8 j + 2 t4 + (e & 1).
    float st[64];
    unsigned short* const state_h = reinterpret_cast<unsigned short*>(state);
    float* const state_f = reinterpret_cast<float*>(state);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const unsigned long long at = (unsigned long long)(8 * j + 2 * t4 + (e & 1)) * D + 64 * cwg + 16 * warp + g + 8 * (e >> 1);
            st[4 * j + e] = state_bf16 ? __uint_as_float(static_cast<unsigned int>(state_h[at]) << 16) : state_f[at];
        }
    }

    const unsigned int out_tile = base + OFF_OUT + cwg * OUT_BYTES;
    for (unsigned int ci = 0; ci < n_chunks; ++ci) {
        const unsigned int s = ci & 1u;
        const WgChunk ch = chunk_at(ci);
        const unsigned int C = ch.C;
        const unsigned int stage = base + s * STAGE_BYTES;
        // The next chunk's operands stream in while this one computes.
        if (ci + 1 < n_chunks) { issue(ci + 1); }

        // S^T as the bf16 A operand of the first two products (the state entering the chunk).
        unsigned int sb[8][4];
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            sb[kk][0] = brain_pack_bf16(st[8 * kk + 0], st[8 * kk + 1]);
            sb[kk][1] = brain_pack_bf16(st[8 * kk + 2], st[8 * kk + 3]);
            sb[kk][2] = brain_pack_bf16(st[8 * kk + 4], st[8 * kk + 5]);
            sb[kk][3] = brain_pack_bf16(st[8 * kk + 6], st[8 * kk + 7]);
        }
        // u^T in the accumulator layout, the start of v_new^T = u^T - S^T w^T.
        float va[32], qa[32];
        const float* const us = reinterpret_cast<const float*>(gen + s * STAGE_BYTES + STAGE_U);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { va[4 * j + e] = us[(8 * j + 2 * t4 + (e & 1)) * SU + 64 * cwg + 16 * warp + g + 8 * (e >> 1)]; }
        }
        gw_fence();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) { gw_mma_n64<-1>(va, sb[kk], gw_desc_k(stage + STAGE_W + (kk >> 2) * ATOM_BYTES + (kk & 3) * 32u), true); }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) { gw_mma_n64<1>(qa, sb[kk], gw_desc_k(stage + STAGE_Q + (kk >> 2) * ATOM_BYTES + (kk & 3) * 32u), kk > 0); }
        gw_commit();
        gw_wait0();

        // v_new^T as the bf16 A operand of the intra-chunk output and the state update.
        unsigned int vb[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            vb[kk][0] = brain_pack_bf16(va[8 * kk + 0], va[8 * kk + 1]);
            vb[kk][1] = brain_pack_bf16(va[8 * kk + 2], va[8 * kk + 3]);
            vb[kk][2] = brain_pack_bf16(va[8 * kk + 4], va[8 * kk + 5]);
            vb[kk][3] = brain_pack_bf16(va[8 * kk + 6], va[8 * kk + 7]);
        }
        const float decay = *reinterpret_cast<const float*>(gen + s * STAGE_BYTES + STAGE_DEC);
#pragma unroll
        for (int i = 0; i < 64; ++i) { st[i] *= decay; }
        gw_fence();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) { gw_mma_n64<1>(qa, vb[kk], gw_desc_k(stage + STAGE_I + kk * 32u), true); }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) { gw_mma_n128_acc(st, vb[kk], gw_desc_mn(stage + STAGE_K + kk * 2048u)); }
        gw_commit();
        gw_wait0();
        // out^T -> the output rows. Rows past the chunk are not written.
        if (op.out_bf16) {
            // The accumulator pairs are adjacent tokens of one column: `stmatrix.trans` writes them as token rows of 16 bytes.
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {
                const unsigned int r0 = brain_pack_bf16(qa[8 * j2 + 0], qa[8 * j2 + 1]);  // tile 2 j2, columns g
                const unsigned int r1 = brain_pack_bf16(qa[8 * j2 + 2], qa[8 * j2 + 3]);  // tile 2 j2, columns g + 8
                const unsigned int r2 = brain_pack_bf16(qa[8 * j2 + 4], qa[8 * j2 + 5]);  // tile 2 j2 + 1, columns g
                const unsigned int r3 = brain_pack_bf16(qa[8 * j2 + 6], qa[8 * j2 + 7]);  // tile 2 j2 + 1, columns g + 8
                // Lane l addresses stored row (l & 7) of matrix (l >> 3): token 8 (2 j2 + (m >> 1)) + (l & 7), column chunk 2 warp + (m & 1).
                const unsigned int m = lane >> 3, rr = lane & 7u;
                const unsigned int token = 8u * (2 * j2 + (m >> 1)) + rr;
                const unsigned int chunk = 2u * warp + (m & 1u);
                const unsigned int addr = out_tile + token * 128u + ((chunk ^ (token & 7u)) << 4);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
            }
            gw_named_barrier(1u + cwg, 128u);
            unsigned short* const out = reinterpret_cast<unsigned short*>(op.out) + ch.out_row + 64 * cwg;
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const unsigned int e = ct + it * 128u;
                const unsigned int r = e >> 3, c = e & 7u;
                uint4 v;
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(out_tile + r * 128u + ((c ^ (r & 7u)) << 4)));
                if (r < C) { *reinterpret_cast<uint4*>(out + r * ch.out_stride + c * 8) = v; }
            }
            // The next chunk's stmatrix must not overwrite the tile before every thread has read it.
            gw_named_barrier(1u + cwg, 128u);
        } else {
            float* const out = reinterpret_cast<float*>(op.out) + ch.out_row + 64 * cwg;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const unsigned int token = 8 * j + 2 * t4 + (e & 1);
                    if (token < C) { out[token * ch.out_stride + 16 * warp + g + 8 * (e >> 1)] = qa[4 * j + e]; }
                }
            }
        }
        // The staged chunk is visible to every thread and no product still reads the slot the next staging will overwrite.
        if (ci + 1 < n_chunks) { publish(); }
    }

    // The state leaves as it arrived.
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const unsigned long long at = (unsigned long long)(8 * j + 2 * t4 + (e & 1)) * D + 64 * cwg + 16 * warp + g + 8 * (e >> 1);
            if (state_bf16) {
                state_h[at] = static_cast<unsigned short>(brain_pack_bf16(st[4 * j + e], 0.0f));
            } else {
                state_f[at] = st[4 * j + e];
            }
        }
    }
}

// The recurrence of one sequence batch in the chunk-major scratch of `gdn_chunk_prep_tok_bf16`: block = one (sequence, head).
// Contract (u32 params, then pointers) as `brain_gdn_chunk_loop_bf16`.
extern "C" __global__ void __launch_bounds__(WGT, 1) brain_gdn_chunk_loop_wg(const unsigned int* params, const unsigned short* kd, const unsigned short* qs,
                                                                          const unsigned short* w, const float* u, const float* g_cs, const float* exp_g_cs,
                                                                          const unsigned short* intra, const float* initial_state, float* out, float* final_state) {
    const unsigned int bh = params[0];
    const unsigned int C = params[1];
    const unsigned int n_chunks = params[4];
    const unsigned int heads_tok = params[6];
    const unsigned int bhi = blockIdx.y * gridDim.x + blockIdx.x;
    if (bhi >= bh) { return; }  // block-uniform
    // The kernel keeps the state in place: seed the output buffer with the initial state when they differ.
    float* const state = final_state + (unsigned long long)bhi * D * D;
    if (initial_state != final_state) {
        const float4* src = reinterpret_cast<const float4*>(initial_state + (unsigned long long)bhi * D * D);
        float4* dst = reinterpret_cast<float4*>(state);
        for (unsigned int i = threadIdx.x; i < D * D / 4; i += WGT) { dst[i] = src[i]; }
        __syncthreads();
    }
    const WgOperands op = {kd, qs, w, intra, u, g_cs, out, false};
    auto chunk_at = [&](unsigned int ci) {
        WgChunk ch;
        ch.C = C;
        ch.row_base = ((unsigned long long)ci * bh + bhi) * C;
        ch.cc_base = ch.row_base * C;
        ch.out_row = ch.row_base * D;
        ch.out_stride = D;
        if (heads_tok != 0) {
            const unsigned long long tok0 = ((unsigned long long)(bhi / heads_tok) * n_chunks + ci) * C;
            ch.out_row = (tok0 * heads_tok + bhi % heads_tok) * D;
            ch.out_stride = (unsigned long long)heads_tok * D;
        }
        return ch;
    };
    gdn_loop_wg_walk(op, n_chunks, state, false, chunk_at);
}
