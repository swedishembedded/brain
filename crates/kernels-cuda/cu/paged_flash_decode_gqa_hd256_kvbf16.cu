// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference serving for clients whose GPU decides how many users one
// card carries. If your team needs expertise in decode attention that runs at the memory roofline over a bf16 KV
// cache, you can procure our services by sending an email to info@swedishembedded.com.
//
// Split-key fused GQA decode attention at head_dim 256 over a bf16 paged pool, on bf16 TENSOR CORES: the CUDA form of
// the `kv=bf16` variant of `paged_flash_decode_gqa_hd256.wgsl` at GRP = BRAIN_GRP (at most 8), with its whole
// contract - the same uniform, bindings, grid and partial records - so `paged_flash_decode_gqa_combine` merges
// its output like any other decode kernel's.
//
//   params       : [batch, n_heads, n_kv_heads, head_dim (256), group (BRAIN_GRP), block_size, max_bt, n_splits,
//                   tiles_per_split] u32
//   q            : [batch, n_heads * 256]                          f32
//   pool_k/pool_v: [num_blocks * block_size, n_kv_heads * 256]     bf16 (element 2w the low half of word w)
//   block_tables : [batch, max_bt] u32       seq_lens : [batch] u32 (live keys, the new token included)
//   part         : [batch, n_heads, n_splits, 258] f32  the unnormalised output (256), the running max, the sum
//
// WHY. The TF32 form of this kernel (`paged_flash_decode_gqa_hd256_tf32.cu`) reads each bf16 element into a TF32
// operand through registers and runs `m16n8k8` products at half the tensor rate: 224 us a layer at 8K keys x 16
// sequences against 77 us for the bytes. The 8-bit kernel (`paged_flash_decode_gqa_hd256_kv8.cu`) reaches 112 us for
// half the bytes with `cp.async` stages and whole-sector loads; this is that design for a 16-bit element.
//
// THE MAP (the 8-bit kernel's, with the operands swapped). One block per (sequence, kv head, key split), 4 warps; each
// warp walks its own 16-key chunks with its own running softmax and the four warps merge once at the end. Per chunk:
//   S^T[keys (16), heads (8, the group padded)] = K[keys, 256] . Q^T[256, heads]    16 k-steps, ONE product a step
//   O^T[256, heads] += V^T[256, 16 keys] . P^T[16 keys, heads]                      16 channel tiles
// Keys are the rows of the first product, so the 6 heads of a group pad to 8 columns, not 16 rows: half the tensor
// instructions of the 8-bit kernel's form. Its accumulator holds (key gid | gid + 8, heads 2 tig | 2 tig + 1), which is
// the TRANSPOSE of the second product's B operand (key 2 tig .. | head gid); `movmatrix` turns one into the other in
// registers. The running maxima and sums then belong to the lane's own two heads (every row of the output accumulator
// belongs to heads 2 tig, 2 tig + 1 too), so the per-chunk correction needs no shuffle. Keys inside a chunk and
// channels inside a dot product are in any order that K, V and Q agree on, chosen so that every load instruction reads
// WHOLE 32-byte sectors: a lane copies 16-byte pieces of its two key rows (piece w at bytes 64 w + 16 tig, so the four
// lanes of a row read 64 contiguous bytes) and of its four value rows (piece p at bytes 128 p + 16 gid, so the eight
// lanes of a row read 128 contiguous bytes).
//
// THE PIPELINE. A chunk's keys (8 KiB) and its values (8 KiB) are two units; a warp's ring holds BRAIN_DECODE_STAGES
// of them, and the units after the one being computed are in flight by `cp.async` (the keys of chunk i + 1 while the
// values of chunk i are consumed). A lane reads back the keys it copied itself and the values of its warp (the lanes copy for each other, see the value
// fetch), so the only synchronisation is the warp's wait on its copies and a `__syncwarp`. Past the live keys a copy of size 0 zero-fills, so dead keys and values are
// zero. The same bytes hold the merge of the four warps after the loop.
//
// NUMERICS. A bf16 key or value is an exact bf16 operand; there is nothing to round. The f32 query and the f32
// probabilities are split into BRAIN_QP and BRAIN_PP bf16 parts (each the bf16 rounding of what the earlier parts left,
// 8 significant bits a part) and every part is multiplied: three parts carry 24 bits, the f32 mantissa. Never f16:
// a bf16 element of magnitude over 65504 would become infinity. Scores, softmax, the running state and the merge are
// f32. Not bit-identical to the WGSL kernel (another summation order); both are gated against exact f64 attention.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_GRP
#define BRAIN_GRP 6
#endif
// bf16 parts of the query and of the probabilities (1 to 3).
#ifndef BRAIN_QP
#define BRAIN_QP 3
#endif
#ifndef BRAIN_PP
#define BRAIN_PP 3
#endif
// Units (8 KiB each) in a warp's ring; blocks resident on a multiprocessor are bounded by it.
#ifndef BRAIN_DECODE_STAGES
#define BRAIN_DECODE_STAGES 3
#endif
#ifndef BRAIN_BLOCKS_PER_SM
#define BRAIN_BLOCKS_PER_SM 2
#endif

#define HD 256
#define TILE 128
#define WARPS 4
#define THREADS (WARPS * 32)
#define CHUNK 16
#define NEG_BIG (-3.4e38f)
#define UNIT_BYTES 8192
#define ROW_BYTES (HD * 2)

// f32 -> nearest bf16 -> f32.
__device__ __forceinline__ float brain_bf16_round(float x) {
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(h) : "f"(x));
    return __uint_as_float(static_cast<unsigned int>(h) << 16);
}

// a and b as N bf16 pairs (low half a, high half b): part i is the bf16 rounding of what the parts before it left.
template <int N>
__device__ __forceinline__ void brain_split_bf16(float a, float b, unsigned int (&out)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const float ha = brain_bf16_round(a), hb = brain_bf16_round(b);
        out[i] = (__float_as_uint(hb) & 0xffff0000u) | (__float_as_uint(ha) >> 16);
        a -= ha;
        b -= hb;
    }
}

__device__ __forceinline__ void brain_mma(float* d, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The 8 x 8 matrix of 16-bit elements whose row r the lanes 4 r .. 4 r + 3 hold (two elements a lane), transposed.
__device__ __forceinline__ unsigned int brain_transpose8(unsigned int x) {
    unsigned int r;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(x));
    return r;
}

// Copy 16 bytes global -> shared without passing through registers; with `ok` false the destination is zero-filled.
__device__ __forceinline__ void brain_copy16(unsigned int dst, const void* src, bool ok) {
    const unsigned int n = ok ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n) : "memory");
}

// The 16-byte slot, in a value unit's 512-byte row of slots, that lane (gid, tig) reads its piece from: the pieces of channel
// groups gid = 2 m and 2 m + 1 of one key pair are neighbours, so the two halves of a 32-byte sector of a value row sit side by
// side in shared memory.
__device__ __forceinline__ unsigned int brain_value_slot(unsigned int gid, unsigned int tig) { return ((gid >> 1) * 4u + tig) * 2u + (gid & 1u); }

// The pool row (slot * n_kv + kv head) of key `j` of the sequence whose block table is `bt`.
__device__ __forceinline__ unsigned int brain_row(const unsigned int* bt, unsigned int j, unsigned int block_size, unsigned int n_kv, unsigned int kvh) {
    return (bt[j / block_size] * block_size + (j % block_size)) * n_kv + kvh;
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

extern "C" __global__ void __launch_bounds__(THREADS, BRAIN_BLOCKS_PER_SM) brain_paged_flash_decode_gqa_hd256_kvbf16(
    const unsigned int* params, const float* q, const unsigned char* pool_k, const unsigned char* pool_v, const unsigned int* block_tables,
    const unsigned int* seq_lens, float* part) {
    BRAIN_PDL_ENTRY();
    const unsigned int batch = params[0], n_heads = params[1], n_kv = params[2], block_size = params[5], max_bt = params[6];
    const unsigned int n_splits = params[7], tiles_per_split = params[8];
    // The row is the fastest-varying part of the block index: the rows of one sequence in a speculative verify round are
    // consecutive and read the same keys, so their blocks run at the same time and the second read of a tile is served by the
    // L2 the first filled.
    const unsigned int wg = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int b = wg % batch;
    const unsigned int wg2 = wg / batch;
    const unsigned int s = wg2 % n_splits;
    const unsigned int kvh = wg2 / n_splits;
    if (kvh >= n_kv) { return; }  // block-uniform

    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int gid = lane >> 2, tig = lane & 3u;
    const unsigned int vslot = brain_value_slot(gid, tig);

    // The query as the first product's B fragments: [k-step][part][lane] -> (b0, b1), the bf16 pairs of head gid at
    // channels 32 (st / 2) + 8 tig + 4 (st mod 2) + 0 .. 1 and + 2 .. 3. Heads past the group are zero and never stored.
    __shared__ uint2 qf[16 * BRAIN_QP][32];
    __shared__ float red_m[WARPS][8], red_l[WARPS][8];
    // The key and value ring (BRAIN_DECODE_STAGES units a warp); after the loop the same bytes hold the merge of the four warps.
    extern __shared__ __align__(16) unsigned char ring[];
    float(*red_o)[8][HD] = reinterpret_cast<float(*)[8][HD]>(ring);

    const float* qb = q + (static_cast<unsigned long long>(b) * n_heads + kvh * BRAIN_GRP) * HD;
    for (unsigned int e = tid; e < 16u * 32u; e += THREADS) {
        const unsigned int st = e >> 5, ln = e & 31u, g = ln >> 2, t = ln & 3u;
        unsigned int pa[BRAIN_QP], pb[BRAIN_QP];
#pragma unroll
        for (int i = 0; i < BRAIN_QP; ++i) { pa[i] = pb[i] = 0u; }
        if (g < BRAIN_GRP) {
            const float4 x = *reinterpret_cast<const float4*>(qb + g * HD + (st >> 1) * 32u + t * 8u + (st & 1u) * 4u);
            brain_split_bf16<BRAIN_QP>(x.x, x.y, pa);
            brain_split_bf16<BRAIN_QP>(x.z, x.w, pb);
        }
#pragma unroll
        for (int i = 0; i < BRAIN_QP; ++i) { qf[st * BRAIN_QP + i][ln] = make_uint2(pa[i], pb[i]); }
    }
    __syncthreads();

    const unsigned int tlen = seq_lens[b];
    const unsigned int k_begin = s * tiles_per_split * TILE;
    const unsigned int k_end = min(k_begin + tiles_per_split * TILE, tlen);
    const unsigned int* bt = block_tables + static_cast<unsigned long long>(b) * max_bt;

    // O^T accumulators of channel tile t (p = t / 4, jj = t mod 4): rows gid and gid + 8 are channels 64 p + 8 gid + 2 jj
    // and + 1, columns the heads 2 tig and 2 tig + 1.
    float o[16][4];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { o[t][i] = 0.f; }
    }
    // The running maximum and this lane's share of the sum of heads 2 tig and 2 tig + 1.
    float m_run[2] = {NEG_BIG, NEG_BIG}, l_run[2] = {0.f, 0.f};

    // The chunks this warp walks: k_begin + (warp + WARPS i) * CHUNK for i < n_mine; unit 2 i is its keys, 2 i + 1 its values.
    const unsigned int first = k_begin + warp * CHUNK;
    const unsigned int n_mine = first < k_end ? (k_end - first + WARPS * CHUNK - 1u) / (WARPS * CHUNK) : 0u;
    const unsigned int my_ring = static_cast<unsigned int>(__cvta_generic_to_shared(ring)) + warp * (BRAIN_DECODE_STAGES * UNIT_BYTES);
    const unsigned char* my_ring_ptr = ring + warp * (BRAIN_DECODE_STAGES * UNIT_BYTES);

    // Unit u goes to this lane's slots of ring slot (u mod STAGES). Past the live keys a copy of size 0 zero-fills.
    unsigned int issue_slot = 0;
    auto issue = [&](unsigned int u) {
        if (u < 2u * n_mine) {
            const unsigned int c0 = first + (u >> 1) * (WARPS * CHUNK);
            const unsigned int at = my_ring + issue_slot * UNIT_BYTES;
            // The 16 keys of a chunk are in one pool block when the block size is a multiple of 16 (the served 64): one table
            // lookup and one division for the chunk, then a row is the first row plus the key offset times the heads.
            const bool in_block = (block_size & 15u) == 0u;
            const unsigned int row0 = in_block ? (bt[c0 / block_size] * block_size + (c0 % block_size)) * n_kv + kvh : 0u;
            auto row_of = [&](unsigned int d) { return in_block ? row0 + d * n_kv : brain_row(bt, c0 + d, block_size, n_kv, kvh); };
            if ((u & 1u) == 0u) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const unsigned int j = c0 + gid + 8u * r;
                    const bool ok = j < k_end;
                    const unsigned char* src = ok ? pool_k + static_cast<unsigned long long>(row_of(j - c0)) * ROW_BYTES + tig * 16u : pool_k;
#pragma unroll
                    for (int w = 0; w < 8; ++w) { brain_copy16(at + (r * 8 + w) * 512u + lane * 16u, src + w * 64, ok); }
                }
            } else {
                // The values are fetched as the keys are: eight rows an instruction, a row's 64 contiguous bytes read by four
                // adjacent lanes (piece P = 4 w + b of the row's 32 pieces of 16 bytes). Eight lanes of one row reading 128 bytes
                // made the memory system return every sector twice (ncu: L2 to L1 bytes 2.15 GB for 1.08 GB of values), and so did
                // a layout that put the two halves of a sector in distant slots; here the two halves of a 32-byte sector (b = 0, 1
                // and 2, 3) land in ADJACENT slots (`brain_value_slot`), and the lanes read their pieces back from there after the
                // warp's wait and `__syncwarp`. The piece goes to the slot of its consumer: gid_v = P mod 8 (a channel group),
                // tig_v = a / 2 (the key pair), p = P / 8, r = 2 h + a mod 2.
                const unsigned int a = lane >> 2, b = lane & 3u;
                const unsigned int lane_slot = (a & 1u) * 2048u + (((b >> 1) * 4u + (a >> 1)) * 2u + (b & 1u)) * 16u;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const unsigned int j = c0 + 8u * h + a;
                    const bool ok = j < k_end;
                    const unsigned char* src = ok ? pool_v + static_cast<unsigned long long>(row_of(j - c0)) * ROW_BYTES + b * 16u : pool_v;
#pragma unroll
                    for (int w = 0; w < 8; ++w) { brain_copy16(at + h * 4096u + (w >> 1) * 512u + (w & 1) * 256u + lane_slot, src + w * 64, ok); }
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        issue_slot = issue_slot + 1u == BRAIN_DECODE_STAGES ? 0u : issue_slot + 1u;
    };
#pragma unroll
    for (int u = 0; u < BRAIN_DECODE_STAGES - 1; ++u) { issue(u); }

    unsigned int slot = 0;
    auto next_unit = [&](unsigned int u) -> const unsigned char* {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(BRAIN_DECODE_STAGES - 2) : "memory");
        __syncwarp();  // every lane's copies are visible to the warp, and every lane is done with the slot issued next
        issue(u + BRAIN_DECODE_STAGES - 1u);
        const unsigned char* at = my_ring_ptr + slot * UNIT_BYTES;
        slot = slot + 1u == BRAIN_DECODE_STAGES ? 0u : slot + 1u;
        return at;
    };

    for (unsigned int i = 0; i < n_mine; ++i) {
        const unsigned int c0 = first + i * (WARPS * CHUNK);

        // ---- scores S^T: two accumulators a part (even and odd k-steps) keep the tensor pipe fed; the parts add smallest first.
        const unsigned char* kbuf = next_unit(2u * i);
        float sacc[BRAIN_QP][2][4];
#pragma unroll
        for (int p = 0; p < BRAIN_QP; ++p) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { sacc[p][h][e] = 0.f; }
            }
        }
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint4 k0 = *reinterpret_cast<const uint4*>(kbuf + w * 512u + lane * 16u);        // key gid
            const uint4 k1 = *reinterpret_cast<const uint4*>(kbuf + (8 + w) * 512u + lane * 16u);  // key gid + 8
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int st = 2 * w + h;
                const unsigned int a0 = h ? k0.z : k0.x, a2 = h ? k0.w : k0.y, a1 = h ? k1.z : k1.x, a3 = h ? k1.w : k1.y;
#pragma unroll
                for (int p = 0; p < BRAIN_QP; ++p) {
                    const uint2 qv = qf[st * BRAIN_QP + p][lane];
                    brain_mma(sacc[p][h], a0, a1, a2, a3, qv.x, qv.y);
                }
            }
        }
        // sc[0], sc[1]: key gid, heads 2 tig, 2 tig + 1; sc[2], sc[3]: key gid + 8.
        float sc[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float tot = 0.f;
#pragma unroll
            for (int p = BRAIN_QP - 1; p >= 0; --p) { tot += sacc[p][0][e] + sacc[p][1][e]; }
            sc[e] = tot * 0.0625f;
        }
        const bool live_a = c0 + gid < k_end, live_b = c0 + gid + 8u < k_end;
        sc[0] = live_a ? sc[0] : NEG_BIG;
        sc[1] = live_a ? sc[1] : NEG_BIG;
        sc[2] = live_b ? sc[2] : NEG_BIG;
        sc[3] = live_b ? sc[3] : NEG_BIG;
        // The maximum over the chunk's 16 keys for each of the lane's heads: its two keys, then the 8 lanes of a quad column.
        float mx[2] = {fmaxf(sc[0], sc[2]), fmaxf(sc[1], sc[3])};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 4));
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 8));
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 16));
        }
        float corr[2], p4[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float m_new = fmaxf(m_run[h], mx[h]);
            corr[h] = expf(m_run[h] - m_new);
            m_run[h] = m_new;
            p4[h] = live_a ? expf(sc[h] - m_new) : 0.f;
            p4[2 + h] = live_b ? expf(sc[2 + h] - m_new) : 0.f;
            l_run[h] = l_run[h] * corr[h] + p4[h] + p4[2 + h];
        }
        // P^T as the second product's B fragments: the pair (heads 2 tig, 2 tig + 1) of key gid is row gid of an 8 x 8 matrix
        // (keys 0 .. 7 of the chunk; keys 8 .. 15 the second), whose transpose gives the lane head gid at keys 2 tig, 2 tig + 1.
        unsigned int pa[BRAIN_PP], pb[BRAIN_PP];
        brain_split_bf16<BRAIN_PP>(p4[0], p4[1], pa);
        brain_split_bf16<BRAIN_PP>(p4[2], p4[3], pb);
#pragma unroll
        for (int p = 0; p < BRAIN_PP; ++p) {
            pa[p] = brain_transpose8(pa[p]);
            pb[p] = brain_transpose8(pb[p]);
        }

        // ---- values: rescale by the lane's own heads' corrections, then accumulate (smallest part first).
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            o[t][0] *= corr[0];
            o[t][1] *= corr[1];
            o[t][2] *= corr[0];
            o[t][3] *= corr[1];
        }
        const unsigned char* vbuf = next_unit(2u * i + 1u);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            uint4 x[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) { x[r] = *reinterpret_cast<const uint4*>(vbuf + (r * 4 + p) * 512u + vslot * 16u); }
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const unsigned int w0 = jj == 0 ? x[0].x : jj == 1 ? x[0].y : jj == 2 ? x[0].z : x[0].w;
                const unsigned int w1 = jj == 0 ? x[1].x : jj == 1 ? x[1].y : jj == 2 ? x[1].z : x[1].w;
                const unsigned int w2 = jj == 0 ? x[2].x : jj == 1 ? x[2].y : jj == 2 ? x[2].z : x[2].w;
                const unsigned int w3 = jj == 0 ? x[3].x : jj == 1 ? x[3].y : jj == 2 ? x[3].z : x[3].w;
                // Rows gid (channel c) and gid + 8 (c + 1) by keys (2 tig, 2 tig + 1) | (2 tig + 8, 2 tig + 9): the two
                // channels of a word are the low and high half, so one permute pairs two keys' same channel.
                const unsigned int a0 = __byte_perm(w0, w1, 0x5410u), a1 = __byte_perm(w0, w1, 0x7632u);
                const unsigned int a2 = __byte_perm(w2, w3, 0x5410u), a3 = __byte_perm(w2, w3, 0x7632u);
#pragma unroll
                for (int pp = BRAIN_PP - 1; pp >= 0; --pp) { brain_mma(o[p * 4 + jj], a0, a1, a2, a3, pa[pp], pb[pp]); }
            }
        }
    }

    // ---- merge the four warps. The ring bytes become the merge buffer: every copy has landed and every warp is out of its loop.
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        const unsigned int ch = 64u * (t >> 2) + 8u * gid + 2u * (t & 3);
        red_o[warp][2u * tig][ch] = o[t][0];
        red_o[warp][2u * tig + 1u][ch] = o[t][1];
        red_o[warp][2u * tig][ch + 1u] = o[t][2];
        red_o[warp][2u * tig + 1u][ch + 1u] = o[t][3];
    }
    // The sum of a head is spread over the 8 lanes of its column (one per key row).
    float l_col[2] = {l_run[0], l_run[1]};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l_col[h] += __shfl_xor_sync(0xffffffffu, l_col[h], 4);
        l_col[h] += __shfl_xor_sync(0xffffffffu, l_col[h], 8);
        l_col[h] += __shfl_xor_sync(0xffffffffu, l_col[h], 16);
    }
    if (gid == 0u) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            red_m[warp][2u * tig + h] = m_run[h];
            red_l[warp][2u * tig + h] = l_col[h];
        }
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
