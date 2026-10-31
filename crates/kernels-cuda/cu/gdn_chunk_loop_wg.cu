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
#define STAGE_W 0
#define STAGE_Q TILE_BYTES
#define STAGE_K (2 * TILE_BYTES)
#define STAGE_I (3 * TILE_BYTES)
#define STAGE_U (STAGE_I + INTRA_BYTES)
#define OUT_BYTES 8192          // a consumer warpgroup's 64-token x 64-column bf16 output tile

// The shared-memory layout of a block with `WGS` consumer warpgroups (2: both 64-column halves of the value dimension; 1: the half the block
// is given): a two-stage ring of operand tiles and one output tile a warpgroup. The staged fp32 u tile holds only the block's own columns
// and is padded to a conflict-free row stride for the accumulator-layout reads.
template <int WGS, bool FOLD, int STAGES = 2>
struct WgLayout {
    static constexpr unsigned SU = (WGS == 2 ? D : 64) + 4;
    // Folded form: u arrives as the prep's fp16 thread-order vectors, 8 KiB per warpgroup, copied as they are.
    static constexpr unsigned U_BYTES = FOLD ? WGS * 8192 : MAXC * SU * 4;
    static constexpr unsigned DEC = STAGE_U + U_BYTES;  // folded form: scale exp(g_t) of each of the chunk's 64 tokens, then exp(g_last - g_t), then exp(g_last)
    static constexpr unsigned ES = DEC;
    static constexpr unsigned DD = ES + MAXC * 4;
    static constexpr unsigned BYTES = ((DD + MAXC * 4 + 16 + 1023) / 1024) * 1024;
    static constexpr unsigned OFF_OUT = STAGES * BYTES;
    static constexpr unsigned OFF_BAR = OFF_OUT + WGS * OUT_BYTES;
    static constexpr unsigned SMEM_BYTES = OFF_BAR + 64 + 1024;
};

// Where one chunk of the walk lives (the `LoopChunk` of the mma.sync recurrence, which this kernel replaces on Hopper).
struct WgChunk {
    unsigned int C;
    unsigned long long row_base, cc_base;
    // Folded form: the element offset of the chunk's first token row and key head in the query / key rows, and the elements between rows.
    unsigned long long qk_at, qk_stride;
    // The output rows: `out_row` is the element offset of the chunk's first row and head in `out`, `out_stride` the elements between rows.
    unsigned long long out_row, out_stride;
};

struct WgOperands {
    const unsigned short *kd, *qs, *w, *intra;
    const float *u, *g_cs;
    void* out;
    bool out_bf16;
    // Folded form (`FOLD`): `qs` and `kd` are the L2-normalised query and key rows as the convolution wrote them (token-major, one
    // key head shared by its value heads); the row scalings the prep otherwise bakes into them (`scale exp(g_t)` on the query, `exp(g_last - g_t)` on
    // the key) are applied here, to the accumulator of the query product and to the bf16 operand of the state update.
    float scale;
};

// One block's walk over `n_chunks` chunks of one value head. The state is read from / written to `state` (bf16 elements when
// `state_bf16`), a [key][value column] matrix of the head.
template <bool FOLD, int WGS, int STAGES, class ChunkAt>
__device__ __forceinline__ void gdn_loop_wg_walk(const WgOperands& op, const unsigned int n_chunks, void* state, const bool state_bf16, const unsigned int half, ChunkAt chunk_at) {
    using L = WgLayout<WGS, FOLD, STAGES>;
    constexpr unsigned int NT = 128 * WGS;  // threads: every one also stages the next chunk's operands
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
        const unsigned int stage = base + (STAGES == 2 ? (ci & 1u) : 0u) * L::BYTES;
        const unsigned int C = ch.C;
        const unsigned long long rb = ch.row_base;
#pragma unroll
        for (int it = 0; it < 1024 / NT; ++it) {
            const unsigned int e = tid + it * NT;
            const unsigned int r = e >> 4, c = e & 15u;
            const unsigned int dst = (c >> 3) * ATOM_BYTES + r * 128u + (((c & 7u) ^ (r & 7u)) << 4);
            const unsigned long long at = (rb + r) * D + c * 8;
            const unsigned int bytes = r < C ? 16u : 0u;
            gw_cp16(stage + STAGE_W + dst, w + (r < C ? at : 0), bytes);
            const unsigned long long qk = FOLD ? (r < C ? ch.qk_at + r * ch.qk_stride + c * 8 : 0) : (r < C ? at : 0);
            gw_cp16(stage + STAGE_Q + dst, qs + qk, bytes);
            gw_cp16(stage + STAGE_K + dst, kd + qk, bytes);
        }
        // u: fp32 rows of the block's columns (128, or its 64) into the padded rows the consumers read in accumulator layout, or in the folded
        // form the fp16 thread-order vectors of the block's warpgroups as one contiguous copy.
        if (FOLD) {
            const unsigned short* const uh = reinterpret_cast<const unsigned short*>(u) + (rb / MAXC) * (MAXC * D) + (WGS == 2 ? 0u : 4096u * half);
#pragma unroll
            for (int it = 0; it < WGS * 512 / NT; ++it) {
                const unsigned int e = tid + it * NT;
                gw_cp16(stage + STAGE_U + 16u * e, uh + 8u * e, 16u);
            }
        } else {
            constexpr unsigned int UV = WGS == 2 ? 32u : 16u;  // 16-byte vectors a row
#pragma unroll
            for (int it = 0; it < 64 * UV / NT; ++it) {
                const unsigned int e = tid + it * NT;
                const unsigned int r = e / UV, c = e % UV;
                gw_cp16(stage + STAGE_U + (r * L::SU + c * 4) * 4u, u + (r < C ? (rb + r) * D + (WGS == 2 ? 0u : 64u * half) + c * 4 : 0), r < C ? 16u : 0u);
            }
        }
        // intra: 64 x 64 bf16, one atom column. A full chunk's rows are 128 bytes (vector copies); a short chunk's rows are C
        // elements apart and gathered.
        if (C == MAXC) {
#pragma unroll
            for (int it = 0; it < 512 / NT; ++it) {
                const unsigned int e = tid + it * NT;
                const unsigned int r = e >> 3, c = e & 7u;
                gw_cp16(stage + STAGE_I + r * 128u + (((c ^ r) & 7u) << 4), intra + ch.cc_base + r * 64u + c * 8, 16u);
            }
        } else {
            for (unsigned int e = tid; e < MAXC * 8; e += NT) {
                const unsigned int r = e >> 3, c = e & 7u;
                unsigned short h[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { h[j] = (r < C && c * 8 + j < C) ? intra[ch.cc_base + (unsigned long long)r * C + c * 8 + j] : (unsigned short)0; }
                const uint4 v = make_uint4(h[0] | (unsigned int)h[1] << 16, h[2] | (unsigned int)h[3] << 16, h[4] | (unsigned int)h[5] << 16, h[6] | (unsigned int)h[7] << 16);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(stage + STAGE_I + r * 128u + (((c ^ r) & 7u) << 4)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
            }
        }
        if (FOLD) {
            // The prep staged scale exp(g_t), exp(g_last - g_t) and exp(g_last) of this chunk-head contiguously (192 floats).
            if (tid < 33u) { gw_cp16(stage + L::ES + 16u * tid, g_cs + rb * 3 + 4u * tid, 16u); }
        } else if (tid == 0u) {
            const float dec = expf(g_cs[rb + C - 1]);
            asm volatile("st.shared.f32 [%0], %1;\n" ::"r"(stage + L::DEC), "f"(dec));
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
    const unsigned int cwg = WGS == 2 ? tid >> 7 : half;   // 0 or 1: value columns 64 cwg ..
    const unsigned int wgi = WGS == 2 ? tid >> 7 : 0u;      // this warpgroup's index within the block
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

    const unsigned int out_tile = base + L::OFF_OUT + wgi * OUT_BYTES;
    for (unsigned int ci = 0; ci < n_chunks; ++ci) {
        const unsigned int s = STAGES == 2 ? (ci & 1u) : 0u;
        const WgChunk ch = chunk_at(ci);
        const unsigned int C = ch.C;
        const unsigned int stage = base + s * L::BYTES;
        // The next chunk's operands stream in while this one computes.
        if (STAGES == 2 && ci + 1 < n_chunks) { issue(ci + 1); }

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
        if (FOLD) {
            // This thread's 32 values, four 16-byte vectors lane after lane, fp16.
            const unsigned char* const uv = gen + s * L::BYTES + STAGE_U + (wgi * 4 + warp) * 2048u + lane * 16u;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint4 p;
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(p.x), "=r"(p.y), "=r"(p.z), "=r"(p.w) : "r"(brain_smem_addr(uv + v * 512)));
                const unsigned int w4[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    asm("{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, %2;\ncvt.f32.f16 %0, lo;\ncvt.f32.f16 %1, hi;\n}\n" : "=f"(va[8 * v + 2 * k]), "=f"(va[8 * v + 2 * k + 1]) : "r"(w4[k]));
                }
            }
        } else {
            const float* const us = reinterpret_cast<const float*>(gen + s * L::BYTES + STAGE_U);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { va[4 * j + e] = us[(8 * j + 2 * t4 + (e & 1)) * L::SU + (WGS == 2 ? 64 * cwg : 0u) + 16 * warp + g + 8 * (e >> 1)]; }
            }
        }
        gw_fence();
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) { gw_mma_n64<-1>(va, sb[kk], gw_desc_k(stage + STAGE_W + (kk >> 2) * ATOM_BYTES + (kk & 3) * 32u), true); }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) { gw_mma_n64<1>(qa, sb[kk], gw_desc_k(stage + STAGE_Q + (kk >> 2) * ATOM_BYTES + (kk & 3) * 32u), kk > 0); }
        gw_commit();
        gw_wait0();
        if (FOLD) {
            // The query product carries no row scaling yet: token n of the accumulator is scaled by scale exp(g_n).
            const float* const es = reinterpret_cast<const float*>(gen + s * L::BYTES + L::ES);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 t = *reinterpret_cast<const float2*>(es + 8 * j + 2 * t4);
                qa[4 * j + 0] *= t.x; qa[4 * j + 1] *= t.y; qa[4 * j + 2] *= t.x; qa[4 * j + 3] *= t.y;
            }
        }

        // v_new^T as the bf16 A operand of the intra-chunk output and the state update.
        unsigned int vb[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            vb[kk][0] = brain_pack_bf16(va[8 * kk + 0], va[8 * kk + 1]);
            vb[kk][1] = brain_pack_bf16(va[8 * kk + 2], va[8 * kk + 3]);
            vb[kk][2] = brain_pack_bf16(va[8 * kk + 4], va[8 * kk + 5]);
            vb[kk][3] = brain_pack_bf16(va[8 * kk + 6], va[8 * kk + 7]);
        }
        // The state-update operand: v_new^T with the key's row scaling folded in (token n of the accumulator scaled by exp(g_last - g_n)).
        unsigned int vd[4][4];
        if (FOLD) {
            const float* const dd = reinterpret_cast<const float*>(gen + s * L::BYTES + L::DD);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const float2 t0 = *reinterpret_cast<const float2*>(dd + 16 * kk + 2 * t4), t1 = *reinterpret_cast<const float2*>(dd + 16 * kk + 8 + 2 * t4);
                vd[kk][0] = brain_pack_bf16(va[8 * kk + 0] * t0.x, va[8 * kk + 1] * t0.y);
                vd[kk][1] = brain_pack_bf16(va[8 * kk + 2] * t0.x, va[8 * kk + 3] * t0.y);
                vd[kk][2] = brain_pack_bf16(va[8 * kk + 4] * t1.x, va[8 * kk + 5] * t1.y);
                vd[kk][3] = brain_pack_bf16(va[8 * kk + 6] * t1.x, va[8 * kk + 7] * t1.y);
            }
        }
        const float decay = *reinterpret_cast<const float*>(gen + s * L::BYTES + (FOLD ? L::DD + MAXC * 4 : L::DEC));
#pragma unroll
        for (int i = 0; i < 64; ++i) { st[i] *= decay; }
        gw_fence();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) { gw_mma_n64<1>(qa, vb[kk], gw_desc_k(stage + STAGE_I + kk * 32u), true); }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) { gw_mma_n128_acc(st, FOLD ? vd[kk] : vb[kk], gw_desc_mn(stage + STAGE_K + kk * 2048u)); }
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
            gw_named_barrier(1u + wgi, 128u);
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
            gw_named_barrier(1u + wgi, 128u);
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
        if (ci + 1 < n_chunks) {
            if (STAGES == 1) {
                // One stage: every thread is done with the tiles before the next chunk overwrites them.
                __syncthreads();
                issue(ci + 1);
            }
            publish();
        }
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
    const WgOperands op = {kd, qs, w, intra, u, g_cs, out, false, 0.0f};
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
    gdn_loop_wg_walk<false, 2, 2>(op, n_chunks, state, false, 0u, chunk_at);
}

// The recurrence of every prompt segment of a round in one launch, on the state rows of the pools themselves (the contract of
// `brain_gdn_chunk_loop_seg_bf16`, whose chunk table and scratch slots it reads): block = one (segment, value head).
// params [blocks, unused..., scale (f32 bits) at 5, H at 6, nseg at 7, state elements per pool row at 8, flags at 9 (bit 0: `out` is bf16
// rows; bit 1: the state pool holds bf16 elements; bit 2: folded form), key_dim at 10, value heads per key head at 11]; buffers kd, qs, w, u,
// g_cs, intra, table, state pool (read and written in place), out. In the folded form `kd` and `qs` are the normalised key and query rows of the
// round (token-major, `key_dim` apart) instead of the prep's scaled chunk-major copies.
extern "C" __global__ void __launch_bounds__(WGT, 1) brain_gdn_chunk_loop_seg_wg(const unsigned int* params, const unsigned short* kd, const unsigned short* qs,
                                                                              const unsigned short* w, const float* u, const float* g_cs,
                                                                              const unsigned short* intra, const unsigned int* table, float* state_pool, float* out) {
    const unsigned int blocks = params[0];
    const unsigned int H = params[6], nseg = params[7];
    const unsigned long long state_words = params[8];
    const bool out_bf16 = (params[9] & 1u) != 0u;
    const bool state_bf16 = (params[9] & 2u) != 0u;
    const bool fold = (params[9] & 4u) != 0u;
    const unsigned long long key_dim = params[10];
    const unsigned int group = max(params[11], 1u);
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= blocks) { return; }  // block-uniform
    const unsigned int seg = blk / H, head = blk % H;
    const unsigned int* sg = table + 4 * seg;
    const unsigned int rows = sg[1], slot = sg[2], chunk0 = sg[3];
    const unsigned int n_chunks = (rows + MAXC - 1) / MAXC;
    const unsigned int* chunks = table + 4 * nseg;

    const WgOperands op = {kd, qs, w, intra, u, g_cs, out, out_bf16, __uint_as_float(params[5])};
    auto chunk_at = [&](unsigned int ci) {
        const unsigned int global = chunk0 + ci;
        WgChunk ch;
        ch.C = chunks[2 * global + 1];
        ch.row_base = ((unsigned long long)global * H + head) * MAXC;
        ch.cc_base = ch.row_base * MAXC;
        ch.out_stride = (unsigned long long)H * D;
        ch.out_row = ((unsigned long long)chunks[2 * global] * H + head) * D;
        ch.qk_at = (unsigned long long)chunks[2 * global] * key_dim + (unsigned long long)(head / group) * D;
        ch.qk_stride = key_dim;
        return ch;
    };
    // A bf16 pool is addressed in elements: the same count of two-byte elements instead of four-byte words.
    const unsigned long long state_at = slot * state_words + (unsigned long long)head * D * D;
    void* const state = state_bf16 ? static_cast<void*>(reinterpret_cast<unsigned short*>(state_pool) + state_at) : static_cast<void*>(state_pool + state_at);
    if (fold) {
        gdn_loop_wg_walk<true, 2, 2>(op, n_chunks, state, state_bf16, 0u, chunk_at);
    } else {
        gdn_loop_wg_walk<false, 2, 2>(op, n_chunks, state, state_bf16, 0u, chunk_at);
    }
}

// `brain_gdn_chunk_loop_seg_wg` with one consumer warpgroup a block: block = (segment, value head, 64-column half), twice the blocks for the
// device's multiprocessors and half the tensor-core work and staged bytes of u on each. Contract and buffers as above (blocks counts the halves).
extern "C" __global__ void __launch_bounds__(128, 1) brain_gdn_chunk_loop_seg_wg_half(const unsigned int* params, const unsigned short* kd, const unsigned short* qs,
                                                                                    const unsigned short* w, const float* u, const float* g_cs,
                                                                                    const unsigned short* intra, const unsigned int* table, float* state_pool, float* out) {
    const unsigned int blocks = params[0];
    const unsigned int H = params[6], nseg = params[7];
    const unsigned long long state_words = params[8];
    const bool out_bf16 = (params[9] & 1u) != 0u;
    const bool state_bf16 = (params[9] & 2u) != 0u;
    const bool fold = (params[9] & 4u) != 0u;
    const unsigned long long key_dim = params[10];
    const unsigned int group = max(params[11], 1u);
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= blocks) { return; }  // block-uniform
    const unsigned int half = blk & 1u;
    const unsigned int seg = (blk >> 1) / H, head = (blk >> 1) % H;
    const unsigned int* sg = table + 4 * seg;
    const unsigned int rows = sg[1], slot = sg[2], chunk0 = sg[3];
    const unsigned int n_chunks = (rows + MAXC - 1) / MAXC;
    const unsigned int* chunks = table + 4 * nseg;

    const WgOperands op = {kd, qs, w, intra, u, g_cs, out, out_bf16, __uint_as_float(params[5])};
    auto chunk_at = [&](unsigned int ci) {
        const unsigned int global = chunk0 + ci;
        WgChunk ch;
        ch.C = chunks[2 * global + 1];
        ch.row_base = ((unsigned long long)global * H + head) * MAXC;
        ch.cc_base = ch.row_base * MAXC;
        ch.out_stride = (unsigned long long)H * D;
        ch.out_row = ((unsigned long long)chunks[2 * global] * H + head) * D;
        ch.qk_at = (unsigned long long)chunks[2 * global] * key_dim + (unsigned long long)(head / group) * D;
        ch.qk_stride = key_dim;
        return ch;
    };
    const unsigned long long state_at = slot * state_words + (unsigned long long)head * D * D;
    void* const state = state_bf16 ? static_cast<void*>(reinterpret_cast<unsigned short*>(state_pool) + state_at) : static_cast<void*>(state_pool + state_at);
    if (fold) {
        gdn_loop_wg_walk<true, 1, 2>(op, n_chunks, state, state_bf16, half, chunk_at);
    } else {
        gdn_loop_wg_walk<false, 1, 2>(op, n_chunks, state, state_bf16, half, chunk_at);
    }
}

// `brain_gdn_chunk_loop_seg_wg_half` with a single stage of operand tiles, so three blocks share a multiprocessor: the form for rounds of
// many short segments, where the blocks outnumber the multiprocessors and each is short, and prologue, epilogue and waves matter more than
// overlapping a chunk's staging with the previous chunk's products. Contract and buffers as above.
extern "C" __global__ void __launch_bounds__(128, 3) brain_gdn_chunk_loop_seg_wg_lean(const unsigned int* params, const unsigned short* kd, const unsigned short* qs,
                                                                                    const unsigned short* w, const float* u, const float* g_cs,
                                                                                    const unsigned short* intra, const unsigned int* table, float* state_pool, float* out) {
    const unsigned int blocks = params[0];
    const unsigned int H = params[6], nseg = params[7];
    const unsigned long long state_words = params[8];
    const bool out_bf16 = (params[9] & 1u) != 0u;
    const bool state_bf16 = (params[9] & 2u) != 0u;
    const bool fold = (params[9] & 4u) != 0u;
    const unsigned long long key_dim = params[10];
    const unsigned int group = max(params[11], 1u);
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk >= blocks) { return; }  // block-uniform
    const unsigned int half = blk & 1u;
    const unsigned int seg = (blk >> 1) / H, head = (blk >> 1) % H;
    const unsigned int* sg = table + 4 * seg;
    const unsigned int rows = sg[1], slot = sg[2], chunk0 = sg[3];
    const unsigned int n_chunks = (rows + MAXC - 1) / MAXC;
    const unsigned int* chunks = table + 4 * nseg;

    const WgOperands op = {kd, qs, w, intra, u, g_cs, out, out_bf16, __uint_as_float(params[5])};
    auto chunk_at = [&](unsigned int ci) {
        const unsigned int global = chunk0 + ci;
        WgChunk ch;
        ch.C = chunks[2 * global + 1];
        ch.row_base = ((unsigned long long)global * H + head) * MAXC;
        ch.cc_base = ch.row_base * MAXC;
        ch.out_stride = (unsigned long long)H * D;
        ch.out_row = ((unsigned long long)chunks[2 * global] * H + head) * D;
        ch.qk_at = (unsigned long long)chunks[2 * global] * key_dim + (unsigned long long)(head / group) * D;
        ch.qk_stride = key_dim;
        return ch;
    };
    const unsigned long long state_at = slot * state_words + (unsigned long long)head * D * D;
    void* const state = state_bf16 ? static_cast<void*>(reinterpret_cast<unsigned short*>(state_pool) + state_at) : static_cast<void*>(state_pool + state_at);
    // Only the folded form is offered (its smem footprint is what the launch reserves).
    if (fold) { gdn_loop_wg_walk<true, 1, 1>(op, n_chunks, state, state_bf16, half, chunk_at); }
}
