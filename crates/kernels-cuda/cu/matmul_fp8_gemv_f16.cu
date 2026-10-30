// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bandwidth-bound LLM decode on GPUs for its clients. If your
// team needs expertise in streaming FP8 weights from HBM at the memory system's rate, then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// FP8 E4M3 weight-streaming matmul for up to 32 rows of bf16 x, without a staging ring:
//
//   params : u32 [m, kg, n]   kg = K/4 packed words per weight row, K a multiple of 128
//   x_bf   : [M, K] bf16      activations (two per u32)
//   w_q    : [N, kg] u32      weights, 4 E4M3 bytes per word
//   sw     : [ceil(N/128), K/128] f32   one scale per 128 x 128 block (`weight_scale_inv`)
//   out    : [M, N] f32       out[m, n] = sum_k x[m,k] * decode(w[n,k]) * sw[n/128, k/128]
//
// Why this kernel exists (knowledge 386)
// --------------------------------------
// `matmul_fp8_dq_stream_*` copies every weight byte global -> shared (`cp.async`) -> registers (`ldmatrix`), decodes it to bf16
// in about 2.5 instructions per weight and synchronises the block once per 8 KB stage: 4.5 thread instructions per weight, the
// issue slots 42 % busy at 2.1 TB/s, three blocks an SM. Here:
//  * each lane copies its own A fragment (16-byte `cp.async` chunks, so the bytes in flight cost no registers) into a ring of
//    BRAIN_FG_DEPTH slots that only its own warp reads, and the same lane reads it back: no barrier inside the loop;
//  * the weight bytes are decoded by the hardware (`cvt.rn.f16x2.e4m3x2`, one instruction per two weights) to f16, which holds
//    every E4M3 code exactly, so the block scale is NOT in the operand: the tensor core accumulates the 128 k of a block
//    unscaled and one f32 fma per accumulator folds `scale * partial` once per 128 k (more exact than the bf16 rounding
//    of code * scale the bf16 kernels feed the MMA);
//  * the bf16 activations are widened to f16 (`satfinite`) once per fragment and shared by the AT 16-row tiles of the warp;
//  * the copies of the first line steps are issued before `griddepcontrol.wait`: they do not depend on the previous
//    kernel, so under programmatic dependent launch the stream starts while the producer of x is finishing.
//
// Work split: a block owns 16 * AT weight rows and BRAIN_FG_WARPS warps that take the 128-wide line steps of K in turn; the
// warps' partial sums are added in warp order through shared memory, so the result does not depend on scheduling.
//
// The k permutation (as in `matmul_i8_dq_stream.cu`): lane (g, t) loads bytes 16t..16t+15 of a 64-byte half line of weight rows
// g and g + 8. Word j (bytes 4j..4j+3) hands A position (2t, 2t+1) the true k (4j, 4j+1) and position (2t+8, 2t+9) the true k
// (4j+2, 4j+3); B takes the same four k of the activation row g as one eight-byte slice. The MMA does not care which true k a
// position stands for as long as A and B agree.
//
// No `__restrict__`: brain's device buffers alias by design.

#ifndef BRAIN_FG_AT
#define BRAIN_FG_AT 2
#endif
#ifndef BRAIN_FG_WARPS
#define BRAIN_FG_WARPS 4
#endif
#ifndef BRAIN_FG_DEPTH
#define BRAIN_FG_DEPTH 3
#endif
#ifndef BRAIN_FG_MIN_BLOCKS
#define BRAIN_FG_MIN_BLOCKS 4
#endif

#define BRAIN_FG_THREADS (32 * BRAIN_FG_WARPS)
#define BRAIN_FG_BN (16 * BRAIN_FG_AT)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BRAIN_FG_PDL_WAIT() asm volatile("griddepcontrol.wait;\n" ::: "memory")
#define BRAIN_FG_PDL_LAUNCH() asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory")
#else
#define BRAIN_FG_PDL_WAIT() do {} while (0)
#define BRAIN_FG_PDL_LAUNCH() do {} while (0)
#endif

__device__ __forceinline__ void brain_fg_cp16(unsigned int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void brain_fg_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void brain_fg_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint4 brain_fg_lds16(unsigned int addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ uint4 brain_fg_ld_x(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

// Two E4M3 bytes -> two f16, exact.
__device__ __forceinline__ unsigned int brain_fg_e4m3x2(unsigned int w) {
    unsigned int d;
    asm("{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.rn.f16x2.e4m3x2 %0, h; }\n" : "=r"(d) : "r"(w));
    return d;
}

// Two bf16 -> two f16 (an activation is exact in f16 unless it is under 6e-5 or past 65504, which saturates).
__device__ __forceinline__ unsigned int brain_fg_bf16x2_f16x2(unsigned int p) {
    unsigned int d;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(__uint_as_float(p & 0xFFFF0000u)), "f"(__uint_as_float(p << 16)));
    return d;
}

__device__ __forceinline__ void brain_fg_mma(float (&d)[4], unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int T>
struct BrainFgX {
    uint4 v[T][2][2];  // [x tile][64-byte half line = 32 bf16][first / second 8 bf16]
};

template <int T>
__device__ __forceinline__ void brain_fg_block(const unsigned char* wq, const unsigned char* xb, const float* sw, float* out, unsigned int m, unsigned int k,
                                               unsigned int n, unsigned int n0, unsigned char* smem) {
    constexpr int AT = BRAIN_FG_AT;
    constexpr int D = BRAIN_FG_DEPTH;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2, t4 = lane & 3u;
    const unsigned int lines = k >> 7;
    const unsigned int mine = warp < lines ? (lines - warp + BRAIN_FG_WARPS - 1u) / BRAIN_FG_WARPS : 0u;

    // Row pointers (rows past n read the last row and are dropped at the store), shifted to this lane's 16 bytes.
    const unsigned char* wp[AT][2];
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int h = 0; h < 2; ++h) { wp[a][h] = wq + (unsigned long long)min(n0 + 16u * a + g + 8u * h, n - 1u) * k + 16u * t4; }
    const unsigned char* xp[T];
#pragma unroll
    for (int t = 0; t < T; ++t) { xp[t] = xb + (unsigned long long)min(8u * t + g, m - 1u) * k * 2u + 32u * t4; }
    const float* sp = sw + (unsigned long long)(n0 >> 7) * lines;

    // The ring of this warp: slot s holds the line step's 4 * AT chunks of 512 bytes, lane l's 16 bytes at l * 16 of each.
    constexpr unsigned int SLOT_BYTES = AT * 2048u;
    const unsigned int ring = static_cast<unsigned int>(__cvta_generic_to_shared(smem)) + warp * D * SLOT_BYTES + lane * 16u;
    auto issue = [&](unsigned int step) {
        const unsigned int dst = ring + (step % D) * SLOT_BYTES;
        const unsigned int off = (warp + step * BRAIN_FG_WARPS) * 128u;
#pragma unroll
        for (int a = 0; a < AT; ++a)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 2; ++c) { brain_fg_cp16(dst + ((a * 2 + h) * 2 + c) * 512u, wp[a][h] + off + 64u * c); }
    };
#pragma unroll
    for (int d = 0; d < D - 1; ++d) {
        if (d < (int)mine) { issue(d); }
        brain_fg_commit();
    }
    // Let the next kernel's blocks start as soon as every block of this grid is resident (they stream their own weights
    // meanwhile), then wait for the previous kernel: the activations are its output and nothing of them is read before.
    BRAIN_FG_PDL_LAUNCH();
    BRAIN_FG_PDL_WAIT();

    float acc[AT][T][4];
#pragma unroll
    for (int a = 0; a < AT; ++a)
#pragma unroll
        for (int t = 0; t < T; ++t) { acc[a][t][0] = acc[a][t][1] = acc[a][t][2] = acc[a][t][3] = 0.0f; }

    BrainFgX<T> xn;
    if (mine > 0u) {
        const unsigned int off = warp * 256u;  // bf16: 256 bytes a line
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) { xn.v[t][c][hh] = brain_fg_ld_x(xp[t] + off + 128u * c + 16u * hh); }
    }

    for (unsigned int i = 0; i < mine; ++i) {
        const BrainFgX<T> xc = xn;
        const float s = sp[warp + i * BRAIN_FG_WARPS];
        brain_fg_wait<D - 2>();  // this warp's copies of step i have landed
        // The slot the previous step read is free: refill it with the step D - 1 ahead.
        if (i + D - 1u < mine) { issue(i + D - 1u); }
        brain_fg_commit();
        // The next line step's activations, one ahead.
        if (i + 1u < mine) {
            const unsigned int off = (warp + (i + 1u) * BRAIN_FG_WARPS) * 256u;
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int c = 0; c < 2; ++c)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) { xn.v[t][c][hh] = brain_fg_ld_x(xp[t] + off + 128u * c + 16u * hh); }
        }
        const unsigned int slot = ring + (i % D) * SLOT_BYTES;
        float part[AT][T][2][4];
#pragma unroll
        for (int a = 0; a < AT; ++a)
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int c = 0; c < 2; ++c) { part[a][t][c][0] = part[a][t][c][1] = part[a][t][c][2] = part[a][t][c][3] = 0.0f; }
#pragma unroll
        for (int a = 0; a < AT; ++a) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint4 r0 = brain_fg_lds16(slot + ((a * 2 + 0) * 2 + c) * 512u), r1 = brain_fg_lds16(slot + ((a * 2 + 1) * 2 + c) * 512u);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned int q0 = j == 0 ? r0.x : j == 1 ? r0.y : j == 2 ? r0.z : r0.w;
                    const unsigned int q1 = j == 0 ? r1.x : j == 1 ? r1.y : j == 2 ? r1.z : r1.w;
                    const unsigned int a0 = brain_fg_e4m3x2(q0), a2 = brain_fg_e4m3x2(q0 >> 16);
                    const unsigned int a1 = brain_fg_e4m3x2(q1), a3 = brain_fg_e4m3x2(q1 >> 16);
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        const uint4 xv = xc.v[t][c][j >> 1];
                        const unsigned int p0 = (j & 1) ? xv.z : xv.x, p1 = (j & 1) ? xv.w : xv.y;
                        brain_fg_mma(part[a][t][c], a0, a1, a2, a3, brain_fg_bf16x2_f16x2(p0), brain_fg_bf16x2_f16x2(p1));
                    }
                }
            }
        }
#pragma unroll
        for (int a = 0; a < AT; ++a)
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int q = 0; q < 4; ++q) { acc[a][t][q] = fmaf(s, part[a][t][0][q] + part[a][t][1][q], acc[a][t][q]); }
    }
    brain_fg_wait<0>();

    // Add the warps' partial sums in warp order and store.
    float* red = reinterpret_cast<float*>(smem);
    constexpr int PER_WARP = AT * T * 4 * 32;
    if (BRAIN_FG_WARPS > 1) {
        __syncthreads();  // every warp is done with the ring the partial sums alias
#pragma unroll
        for (int a = 0; a < AT; ++a)
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int q = 0; q < 4; ++q) { red[warp * PER_WARP + ((a * T + t) * 4 + q) * 32 + lane] = acc[a][t][q]; }
        __syncthreads();
        for (unsigned int idx = threadIdx.x; idx < PER_WARP; idx += BRAIN_FG_THREADS) {
            float total = 0.0f;
#pragma unroll
            for (int ww = 0; ww < BRAIN_FG_WARPS; ++ww) { total += red[ww * PER_WARP + idx]; }
            const unsigned int l = idx & 31u, q = (idx >> 5) & 3u, at = idx >> 7;
            const unsigned int a = at / T, t = at % T;
            const unsigned int row = n0 + 16u * a + (l >> 2) + 8u * (q >> 1);
            const unsigned int col = 8u * t + 2u * (l & 3u) + (q & 1u);
            if (row < n && col < m) { out[(unsigned long long)col * n + row] = total; }
        }
    } else {
#pragma unroll
        for (int a = 0; a < AT; ++a)
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const unsigned int row = n0 + 16u * a + g + 8u * (q >> 1);
                    const unsigned int col = 8u * t + 2u * t4 + (q & 1u);
                    if (row < n && col < m) { out[(unsigned long long)col * n + row] = acc[a][t][q]; }
                }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_FG_THREADS, BRAIN_FG_MIN_BLOCKS)
brain_matmul_fp8_gemv_f16(const unsigned int* params, const unsigned int* xb, const unsigned int* wq, const float* sw, float* out) {
    extern __shared__ __align__(16) unsigned char brain_fg_smem_base[];
    const unsigned int m = params[0];
    const unsigned int k = params[1] * 4u;
    const unsigned int n = params[2];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int n0 = blk * BRAIN_FG_BN;
    if (n0 >= n || m == 0u) { return; }
    const unsigned char* w8 = reinterpret_cast<const unsigned char*>(wq);
    const unsigned char* x8 = reinterpret_cast<const unsigned char*>(xb);
    if (m <= 8u)       { brain_fg_block<1>(w8, x8, sw, out, m, k, n, n0, brain_fg_smem_base); }
    else if (m <= 16u) { brain_fg_block<2>(w8, x8, sw, out, m, k, n, n0, brain_fg_smem_base); }
    else if (m <= 24u) { brain_fg_block<3>(w8, x8, sw, out, m, k, n, n0, brain_fg_smem_base); }
    else               { brain_fg_block<4>(w8, x8, sw, out, m, k, n, n0, brain_fg_smem_base); }
}
