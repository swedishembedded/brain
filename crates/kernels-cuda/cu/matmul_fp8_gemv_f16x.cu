// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM decode on GPUs for its clients. If your
// team needs expertise in streaming FP8 weights from HBM at the memory system's rate for a batch of sequences, then you
// can procure our services by sending an email to info@swedishembedded.com.
//
// FP8 E4M3 weight-streaming matmul for 1 to 16 rows of bf16 x with the ROWS OF x AS THE MMA's A OPERAND:
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per weight row, K a multiple of 128
//   x_bf   : [M, K] bf16      activations (two per u32)
//   w_q    : [N, kg] u32      weights, 4 E4M3 bytes per word
//   sw     : [ceil(N/128), K/128] f32   one scale per 128 x 128 block (`weight_scale_inv`)
//   out    : [M, N] f32       out[m, n] = sum_k x[m,k] * decode(w[n,k]) * sw[n/128, k/128]
//
// `matmul_fp8_gemv_f16_*` puts 16 weight rows in the A operand and the rows of x in the 8-wide B operand, which is the right
// shape up to 8 rows and doubles the activation work at 9 to 16 (two B tiles, each converted from bf16 to f16 per MMA, and the
// L1 serving the B fragments at one line a row). Here the 16 rows of x are the A operand (m16), loaded and converted to f16 once
// per k16 step, and 8 weight rows are the 8-wide B operand: one MMA covers 16 rows of x against 8 weight rows, the activation
// work does not depend on the row count (1 to 16) and is shared by the WT weight tiles a warp streams. Everything else is
// `matmul_fp8_gemv_f16.cu`: per-lane `cp.async` fragments into a ring only the warp reads, hardware E4M3 -> f16 decode
// (`cvt.rn.f16x2.e4m3x2`, exact), the 128 x 128 block scale folded once per 128 k in f32, K taken in turn by the warps of the
// block and their partial sums added in warp order, the copies of the first line steps issued before `griddepcontrol.wait`.
//
// The k permutation: lane (g, t) holds bytes 16t..16t+15 of a 64-byte half line of weight row g (B operand: k position (2t, 2t+1)
// of step j is true k 16t + 4j, +1 and position (2t+8, 2t+9) is 16t + 4j + 2, +3); A takes the same four k of x rows g and g + 8.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_FX_WT
#define BRAIN_FX_WT 4
#endif
#ifndef BRAIN_FX_WARPS
#define BRAIN_FX_WARPS 4
#endif
#ifndef BRAIN_FX_DEPTH
#define BRAIN_FX_DEPTH 3
#endif
#ifndef BRAIN_FX_MIN_BLOCKS
#define BRAIN_FX_MIN_BLOCKS 3
#endif

#define BRAIN_FX_THREADS (32 * BRAIN_FX_WARPS)
#define BRAIN_FX_BN (8 * BRAIN_FX_WT)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_FX_PDL_WAIT() asm volatile("griddepcontrol.wait;\n" ::: "memory")
#define BRAIN_FX_PDL_LAUNCH() asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory")
#else
#define BRAIN_FX_PDL_WAIT() do {} while (0)
#define BRAIN_FX_PDL_LAUNCH() do {} while (0)
#endif

__device__ __forceinline__ void brain_fx_cp16(unsigned int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void brain_fx_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_fx_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint4 brain_fx_lds16(unsigned int addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ uint4 brain_fx_ld_x(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

// Two E4M3 bytes -> two f16, exact.
__device__ __forceinline__ unsigned int brain_fx_e4m3x2(unsigned int w) {
    unsigned int d;
    asm("{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.rn.f16x2.e4m3x2 %0, h; }\n" : "=r"(d) : "r"(w));
    return d;
}

// Two bf16 -> two f16 (exact for 6e-5 <= |x| <= 65504, saturating past it).
__device__ __forceinline__ unsigned int brain_fx_bf16x2_f16x2(unsigned int p) {
    unsigned int d;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(__uint_as_float(p & 0xFFFF0000u)), "f"(__uint_as_float(p << 16)));
    return d;
}

__device__ __forceinline__ void brain_fx_mma(float (&d)[4], unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The activations of one line step a lane needs: rows g and g + 8, two 64-byte halves of the line, 16 bf16 each.
struct BrainFxX {
    uint4 v[2][2][2];  // [row g / g + 8][half line c][first / second 8 bf16]
};

extern "C" __global__ void __launch_bounds__(BRAIN_FX_THREADS, BRAIN_FX_MIN_BLOCKS)
brain_matmul_fp8_gemv_f16x(const unsigned int* params, const unsigned int* xb, const unsigned int* wq, const float* sw, float* out) {
    constexpr int WT = BRAIN_FX_WT;
    constexpr int D = BRAIN_FX_DEPTH;
    extern __shared__ __align__(16) unsigned char brain_fx_smem_base[];
    const unsigned int m = params[0];
    const unsigned int k = params[1] * 4u;
    const unsigned int n = params[2];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int n0 = blk * BRAIN_FX_BN;
    if (n0 >= n || m == 0u) { return; }  // block-uniform
    const unsigned char* w8 = reinterpret_cast<const unsigned char*>(wq);
    const unsigned char* x8 = reinterpret_cast<const unsigned char*>(xb);

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int lines = k >> 7;
    const unsigned int mine = warp < lines ? (lines - warp + BRAIN_FX_WARPS - 1u) / BRAIN_FX_WARPS : 0u;

    // Weight row pointers (rows past n read the last row and are dropped at the store), shifted to this lane's 16 bytes.
    const unsigned char* wp[WT];
#pragma unroll
    for (int a = 0; a < WT; ++a) { wp[a] = w8 + (unsigned long long)min(n0 + 8u * a + g, n - 1u) * k + 16u * t4; }
    // x rows g and g + 8 (rows past m read the last row and are dropped at the store).
    const unsigned char* xp[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) { xp[h] = x8 + (unsigned long long)min(g + 8u * h, m - 1u) * k * 2u + 32u * t4; }
    const float* sp = sw + (unsigned long long)(n0 >> 7) * lines;

    // The ring of this warp: slot s holds the line step's WT * 2 chunks of 512 bytes, lane l's 16 bytes at l * 16 of each.
    constexpr unsigned int SLOT_BYTES = WT * 1024u;
    const unsigned int ring = static_cast<unsigned int>(__cvta_generic_to_shared(brain_fx_smem_base)) + warp * D * SLOT_BYTES + lane * 16u;
    auto issue = [&](unsigned int step) {
        const unsigned int dst = ring + (step % D) * SLOT_BYTES;
        const unsigned int off = (warp + step * BRAIN_FX_WARPS) * 128u;
#pragma unroll
        for (int a = 0; a < WT; ++a)
#pragma unroll
            for (int c = 0; c < 2; ++c) { brain_fx_cp16(dst + (a * 2 + c) * 512u, wp[a] + off + 64u * c); }
    };
#pragma unroll
    for (int d = 0; d < D - 1; ++d) {
        if (d < (int)mine) { issue(d); }
        brain_fx_commit();
    }
    BRAIN_FX_PDL_LAUNCH();
    BRAIN_FX_PDL_WAIT();

    float acc[WT][4];
#pragma unroll
    for (int a = 0; a < WT; ++a) { acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.0f; }

    auto load_x = [&](BrainFxX& x, unsigned int step) {
        const unsigned int off = (warp + step * BRAIN_FX_WARPS) * 256u;  // bf16: 256 bytes a line
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int q = 0; q < 2; ++q) { x.v[h][c][q] = brain_fx_ld_x(xp[h] + off + 128u * c + 16u * q); }
    };
    BrainFxX xn;
    if (mine > 0u) { load_x(xn, 0); }

    for (unsigned int i = 0; i < mine; ++i) {
        const BrainFxX xc = xn;
        const float s = sp[warp + i * BRAIN_FX_WARPS];
        brain_fx_wait<D - 2>();  // this warp's copies of step i have landed
        if (i + D - 1u < mine) { issue(i + D - 1u); }
        brain_fx_commit();
        if (i + 1u < mine) { load_x(xn, i + 1u); }
        const unsigned int slot = ring + (i % D) * SLOT_BYTES;
        float part[WT][4];
#pragma unroll
        for (int a = 0; a < WT; ++a) { part[a][0] = part[a][1] = part[a][2] = part[a][3] = 0.0f; }
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            // The weights of every tile for this half line, read once: word j holds the four k of step j.
            uint4 bw[WT];
#pragma unroll
            for (int a = 0; a < WT; ++a) { bw[a] = brain_fx_lds16(slot + (a * 2 + c) * 512u); }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 r0 = xc.v[0][c][j >> 1], r1 = xc.v[1][c][j >> 1];
                const unsigned int w0 = (j & 1) ? r0.z : r0.x, w1 = (j & 1) ? r0.w : r0.y;
                const unsigned int v0 = (j & 1) ? r1.z : r1.x, v1 = (j & 1) ? r1.w : r1.y;
                const unsigned int a0 = brain_fx_bf16x2_f16x2(w0), a2 = brain_fx_bf16x2_f16x2(w1);
                const unsigned int a1 = brain_fx_bf16x2_f16x2(v0), a3 = brain_fx_bf16x2_f16x2(v1);
#pragma unroll
                for (int a = 0; a < WT; ++a) {
                    const unsigned int q = j == 0 ? bw[a].x : j == 1 ? bw[a].y : j == 2 ? bw[a].z : bw[a].w;
                    brain_fx_mma(part[a], a0, a1, a2, a3, brain_fx_e4m3x2(q), brain_fx_e4m3x2(q >> 16));
                }
            }
        }
#pragma unroll
        for (int a = 0; a < WT; ++a)
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc[a][q] = fmaf(s, part[a][q], acc[a][q]); }
    }
    brain_fx_wait<0>();

    // Add the warps' partial sums in warp order and store: D's c0, c1 are x row g, weight rows 2 t + (0, 1); c2, c3 x row g + 8.
    float* red = reinterpret_cast<float*>(brain_fx_smem_base);
    constexpr int PER_WARP = WT * 4 * 32;
    if (BRAIN_FX_WARPS > 1) {
        __syncthreads();  // every warp is done with the ring the partial sums alias
#pragma unroll
        for (int a = 0; a < WT; ++a)
#pragma unroll
            for (int q = 0; q < 4; ++q) { red[warp * PER_WARP + (a * 4 + q) * 32 + lane] = acc[a][q]; }
        __syncthreads();
        for (unsigned int idx = threadIdx.x; idx < PER_WARP; idx += BRAIN_FX_THREADS) {
            float total = 0.0f;
#pragma unroll
            for (int ww = 0; ww < BRAIN_FX_WARPS; ++ww) { total += red[ww * PER_WARP + idx]; }
            const unsigned int l = idx & 31u, q = (idx >> 5) & 3u, a = idx >> 7;
            const unsigned int xrow = (l >> 2) + 8u * (q >> 1);
            const unsigned int col = n0 + 8u * a + 2u * (l & 3u) + (q & 1u);
            if (xrow < m && col < n) { out[(unsigned long long)xrow * n + col] = total; }
        }
    } else {
#pragma unroll
        for (int a = 0; a < WT; ++a)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int xrow = g + 8u * (q >> 1);
                const unsigned int col = n0 + 8u * a + 2u * t4 + (q & 1u);
                if (xrow < m && col < n) { out[(unsigned long long)xrow * n + col] = acc[a][q]; }
            }
    }
}
