// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements convolutional-network training kernels for
// its clients. If your team needs expertise in getting a CNN's forward and
// backward passes onto the fp32 roofline of a GPU, you can procure our
// services by sending an email to info@swedishembedded.com.
//
// Dense fp32 2D convolution, NCHW, square K x K kernel, any stride and zero
// padding: the forward, the input gradient and the weight gradient as
// implicit GEMMs, and the 3x3 / stride 1 / pad 1 forward of a layer at least
// 64 channels wide as a direct convolution with the same bits. The hand-written forms of `conv2d.wgsl`, `conv2d_dx.wgsl`
// and `conv2d_dw.wgsl`, reading the identical buffers and the identical
// uniform `[N, Cin, H, W, Cout, K, stride, pad, Ho, Wo]`:
//
//   x  : [N, Cin,  H,  W ]
//   w  : [Cout, Cin, K, K]
//   y  : [N, Cout, Ho, Wo]
//
// What the WGSL kernels do and why it is slow
// -------------------------------------------
// One thread per output element, a serial walk of the whole reduction in
// global memory: every operand is fetched once per multiply-add, the
// reduction-major loads of neighbouring threads are a whole channel apart,
// and the weight-gradient kernel gives one thread a reduction over every
// output position of the batch (hundreds of thousands of terms) while
// launching only as many threads as the weight has elements.
//
// What these kernels do instead
// -----------------------------
// Each is a GEMM whose operands are gathered from the NCHW tensors as they
// are staged, so no im2col buffer is ever written:
//
//   forward  y[co, p]  = sum_k  w[co, k]        * patch(x)[k, p]
//            p = (n, ho, wo),  k = (ci, kh, kw)
//   dx       dx[ci, p] = sum_k' w[co, ci, kh, kw] * dy[n, co, ho, wo]
//            p = (n, h, w) of one STRIDE CLASS, k' = (co, a, b)
//   dw       dw[co, k] = sum_p  dy[n, co, p]     * patch(x)[k, p]
//
// A block stages a 16-deep slice of the reduction for both operands in
// shared memory (double-buffered: the next slice's global loads are in
// flight while the current one is multiplied), and every thread owns a
// register block of outputs that it accumulates with explicit fused
// multiply-adds. The backend compiles with `--fmad=false`, which only stops
// the compiler from contracting `a*b + c` on its own; `fmaf` is an explicit
// request and is kept.
//
// The input gradient is split by STRIDE CLASS. With stride s, an input
// position h receives a contribution from tap kh only when
// (h + pad - kh) is a multiple of s, so for s = 2 three of every four
// (kh, kw) pairs of a 3x3 kernel are dead for any given position. Gathering
// over all taps would multiply by zero three times out of four; instead the
// grid is split into the s*s classes (h mod s, w mod s), and inside one
// class the live taps are exactly kh = kh0 + s*a, kw = kw0 + s*b - a dense
// sub-convolution with no dead work. Stride 1 is the one-class case.
//
// The weight gradient's reduction runs over every output position of the
// batch, which for the early layers of a detector is far longer than its
// output is wide: a 16 x 144 gradient summed over half a million positions
// would be three blocks. So the positions are SPLIT across blocks
// (`S` slices of `chunk` positions each); each block writes its slice's
// partial sums to a scratch plane and `brain_conv2d_dw_reduce` adds the
// planes in ascending slice order. No atomics: the result is deterministic
// run to run. With S = 1 the partial kernel accumulates straight into dw.
//
// Accuracy
// --------
// fp32 throughout, fp32 accumulation. The forward and the input gradient
// sum their reduction in the reference's own order (ascending k) into one
// register per output; the difference to the reference is the rounding of
// a fused multiply-add (one rounding where the reference has two). The
// weight gradient re-associates its position sum into S slices. Both stay
// inside the fp32 summation-order bound the gate derives.
//
// Index decoding without division
// -------------------------------
// The gathers need (ci, kh, kw) from a flat k and (n, ho, wo) from a flat p.
// A thread's rows advance by a fixed stride every stage, so each thread
// decodes its rows ONCE with divisions and afterwards adds the stride in
// mixed radix (`brain_cv_adv3`), carrying by comparison - two compares and
// a few adds per load instead of two integer divisions.
//
// No `__restrict__` anywhere: brain's device buffers alias by design.
// Nothing here names a card: tile geometry is a property of this source and
// whether a device can host it is asked of the driver.

#define BRAIN_CV_THREADS 256
#define BRAIN_CV_BK 16         // reduction depth staged per iteration (fwd, dx)
#define BRAIN_CV_BN 128        // output positions per block (fwd, dx)
#define BRAIN_CV_TN 8          // positions per thread (fwd, dx)
#define BRAIN_CV_DW_BP 32      // positions staged per iteration (dw)
#define BRAIN_CV_DW_BN 64      // weight columns (ci, kh, kw) per block (dw)

struct BrainConvP {
    unsigned N, Cin, H, W, Cout, K, stride, pad, Ho, Wo;
};

__device__ __forceinline__ BrainConvP brain_cv_params(const unsigned int* p) {
    BrainConvP c;
    c.N = p[0]; c.Cin = p[1]; c.H = p[2]; c.W = p[3]; c.Cout = p[4];
    c.K = p[5]; c.stride = p[6]; c.pad = p[7]; c.Ho = p[8]; c.Wo = p[9];
    return c;
}

// Output channels a block covers: the smallest of 16/32/64 that holds Cout,
// so a 16-channel layer does not multiply three blocks' worth of zeros.
// `gpu_core::native_upgrade` derives the block count with the SAME rule.
__device__ __forceinline__ unsigned brain_cv_bm(unsigned rows) {
    return rows <= 16u ? 16u : (rows <= 32u ? 32u : 64u);
}

// A position in mixed radix (hi digit, mid digit, lo digit) = (q, r, s) with
// radices (., R1, R0), advanced by a precomputed (dq, dr, ds) with carries.
// Exact for any advance whose digits are already reduced (dr < R1, ds < R0).
struct BrainCvRadix {
    int q, r, s;
};

__device__ __forceinline__ void brain_cv_adv3(BrainCvRadix& v, const BrainCvRadix& d, int r1, int r0) {
    v.s += d.s;
    int c = v.s >= r0;
    v.s -= c ? r0 : 0;
    v.r += d.r + c;
    c = v.r >= r1;
    v.r -= c ? r1 : 0;
    v.q += d.q + c;
}

__device__ __forceinline__ BrainCvRadix brain_cv_split3(unsigned v, unsigned r1, unsigned r0) {
    BrainCvRadix o;
    const unsigned hi = v / r0;
    o.s = (int)(v - hi * r0);
    o.q = (int)(hi / r1);
    o.r = (int)(hi - (unsigned)o.q * r1);
    return o;
}

// ---------------------------------------------------------------------------
// The register-block multiply shared by the forward and the input gradient:
// a [BK x BM] A slice and a [BK x BN] B slice in shared memory, thread
// (ty, tx) owning rows ty*TM.. and the two runs of four positions
// [tx*4, tx*4+4) and [64 + tx*4, 64 + tx*4 + 4) - so the eight threads of a
// quarter-warp read 128 contiguous bytes of a B row per vector load and hit
// every bank once. (Eight contiguous positions per thread would put threads
// tx and tx+4 on the same banks: a two-way conflict on the hottest load.)
// ---------------------------------------------------------------------------
#define BRAIN_CV_HALF (BRAIN_CV_BN / 2)

// The block-relative position of a thread's register column j.
__device__ __forceinline__ unsigned brain_cv_col(int tx, int j) { return (j < 4 ? 0u : (unsigned)BRAIN_CV_HALF) + 4u * tx + (j & 3); }

template <int TM>
__device__ __forceinline__ void brain_cv_mma(const float* as, int as_stride, const float* bs, float (&acc)[TM][BRAIN_CV_TN],
                                             int ty, int tx) {
#pragma unroll
    for (int kk = 0; kk < BRAIN_CV_BK; ++kk) {
        float a[TM];
        if (TM == 4) {
            const float4 v = *reinterpret_cast<const float4*>(as + kk * as_stride + ty * TM);
            a[0] = v.x; a[1 % TM] = v.y; a[2 % TM] = v.z; a[3 % TM] = v.w;
        } else if (TM == 2) {
            const float2 v = *reinterpret_cast<const float2*>(as + kk * as_stride + ty * TM);
            a[0] = v.x; a[1 % TM] = v.y;
        } else {
            a[0] = as[kk * as_stride + ty * TM];
        }
        const float4 b0 = *reinterpret_cast<const float4*>(bs + kk * BRAIN_CV_BN + 4 * tx);
        const float4 b1 = *reinterpret_cast<const float4*>(bs + kk * BRAIN_CV_BN + BRAIN_CV_HALF + 4 * tx);
        const float b[BRAIN_CV_TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < BRAIN_CV_TN; ++j) { acc[i][j] = fmaf(a[i], b[j], acc[i][j]); }
        }
    }
}

// Shared memory for the forward and the input gradient: two A slices of
// 16 x (64 + 4) and two B slices of 16 x 128 floats. The A row is padded by
// four floats so a row stays 16-byte aligned for the vector reads.
#define BRAIN_CV_AS (BRAIN_CV_BK * (64 + 4))
#define BRAIN_CV_BS (BRAIN_CV_BK * BRAIN_CV_BN)

// ---------------------------------------------------------------------------
// Forward.
// ---------------------------------------------------------------------------
template <int BM>
__device__ __forceinline__ void brain_conv2d_fwd_tile(const BrainConvP& c, const float* x, const float* w, const float* bias, float* y,
                                                      unsigned tile_m, unsigned tile_n, float* smem) {
    constexpr int TM = BM / 16;
    constexpr int AS = BM + 4;                      // A row stride, floats
    constexpr int A_LOADS = BM * BRAIN_CV_BK / BRAIN_CV_THREADS;
    // Slice `b` of each operand, by arithmetic: an array of the two pointers
    // indexed by the runtime buffer number would live in local memory.
    auto as_buf = [smem](int b) { return smem + b * BRAIN_CV_AS; };
    auto bs_buf = [smem](int b) { return smem + 2 * BRAIN_CV_AS + b * BRAIN_CV_BS; };

    const int tid = threadIdx.x;
    const int ty = tid / 16, tx = tid % 16;
    const unsigned KK = c.K * c.K;
    const unsigned Kd = c.Cin * KK;
    const unsigned HW = c.H * c.W;
    const unsigned HoWo = c.Ho * c.Wo;
    const unsigned P = c.N * HoWo;
    const unsigned co0 = tile_m * BM;
    const unsigned p0 = tile_n * BRAIN_CV_BN;

    // A (weights) loads: column acol of the k slice is the same for every one
    // of a thread's A_LOADS rows.
    const int acol = tid % BRAIN_CV_BK;
    const int arow = tid / BRAIN_CV_BK;  // + 16 * i

    // B (input patch) loads: one fixed position per thread, rows krow + 2i.
    const int pcol = tid % BRAIN_CV_BN;
    const int krow = tid / BRAIN_CV_BN;  // 0 or 1
    const unsigned p = p0 + pcol;
    const bool p_ok = p < P;
    int hi0 = 0, wi0 = 0;
    long long xbase = 0;
    if (p_ok) {
        const unsigned n = p / HoWo;
        const unsigned q = p - n * HoWo;
        const unsigned ho = q / c.Wo;
        const unsigned wo = q - ho * c.Wo;
        hi0 = (int)(ho * c.stride) - (int)c.pad;
        wi0 = (int)(wo * c.stride) - (int)c.pad;
        xbase = (long long)n * c.Cin * HW;
    }
    // (ci, kh, kw) of this thread's first row, and the advances by 2 (to the
    // next row of the same slice) and by 16 (to the next slice).
    const int K = (int)c.K;
    BrainCvRadix k_first = brain_cv_split3((unsigned)krow, c.K, c.K);
    const BrainCvRadix d2 = brain_cv_split3(2u, c.K, c.K);
    const BrainCvRadix d16 = brain_cv_split3((unsigned)BRAIN_CV_BK, c.K, c.K);

    float ra[A_LOADS];
    float rb[8];

    auto load = [&](unsigned k0, BrainCvRadix kv) {
#pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const unsigned co = co0 + arow + 16 * i;
            const unsigned k = k0 + acol;
            ra[i] = (co < c.Cout && k < Kd) ? w[(unsigned long long)co * Kd + k] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned k = k0 + krow + 2 * i;
            const int hi = hi0 + kv.r;
            const int wi = wi0 + kv.s;
            const bool ok = p_ok && k < Kd && (unsigned)hi < c.H && (unsigned)wi < c.W;
            rb[i] = ok ? x[xbase + (long long)kv.q * HW + hi * (int)c.W + wi] : 0.0f;
            if (i < 7) { brain_cv_adv3(kv, d2, K, K); }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < A_LOADS; ++i) { as_buf(buf)[acol * AS + arow + 16 * i] = ra[i]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) { bs_buf(buf)[(krow + 2 * i) * BRAIN_CV_BN + pcol] = rb[i]; }
    };

    float acc[TM][BRAIN_CV_TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < BRAIN_CV_TN; ++j) { acc[i][j] = 0.0f; }
    }

    // `kv` tracks (ci, kh, kw) of the thread's first B row of the CURRENT
    // slice being loaded.
    BrainCvRadix kv = k_first;
    load(0u, kv);
    store(0);
    __syncthreads();
    int cur = 0;
    for (unsigned k0 = 0; k0 < Kd; k0 += BRAIN_CV_BK) {
        const bool more = k0 + BRAIN_CV_BK < Kd;
        if (more) {
            brain_cv_adv3(kv, d16, K, K);
            load(k0 + BRAIN_CV_BK, kv);
        }
        brain_cv_mma<TM>(as_buf(cur), AS, bs_buf(cur), acc, ty, tx);
        if (more) { store(cur ^ 1); }
        __syncthreads();
        cur ^= 1;
    }

    // Write: rows co0 + ty*TM + i; columns as `brain_cv_col` lays them out, in
    // two runs of four positions. A run is one 16-byte store where the map is
    // a whole number of vectors (four consecutive positions then never straddle
    // two images) and the binding is aligned.
    const bool vec = (HoWo % 4u == 0u) && ((reinterpret_cast<unsigned long long>(y) & 15ull) == 0ull);
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const unsigned co = co0 + ty * TM + i;
        if (co >= c.Cout) { continue; }
        if (bias != nullptr) {
            // `conv_bias`: the bias joins the finished sum, as in the reference.
            const float b = bias[co];
#pragma unroll
            for (int j = 0; j < BRAIN_CV_TN; ++j) { acc[i][j] = acc[i][j] + b; }
        }
#pragma unroll
        for (int run = 0; run < 2; ++run) {
            const unsigned pr = p0 + brain_cv_col(tx, 4 * run);
            if (pr >= P) { continue; }
            if (vec) {
                const unsigned n = pr / HoWo, q = pr - (pr / HoWo) * HoWo;
                *reinterpret_cast<float4*>(y + ((unsigned long long)n * c.Cout + co) * HoWo + q) =
                    make_float4(acc[i][4 * run], acc[i][4 * run + 1], acc[i][4 * run + 2], acc[i][4 * run + 3]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned p = pr + j;
                    if (p < P) {
                        const unsigned n = p / HoWo;
                        y[((unsigned long long)n * c.Cout + co) * HoWo + (p - n * HoWo)] = acc[i][4 * run + j];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Forward, 3x3 / stride 1 / pad 1, as a DIRECT convolution
// (`brain_conv2d_fwd3x3`, `brain_conv2d_bias_fwd3x3`).
//
// The implicit GEMM above gathers every input element once per tap it feeds -
// nine times for a 3x3 kernel - each with its own index arithmetic and bounds
// check, and on a wide layer (the diffusion VAEs: 128-512 channels at large
// maps) that staging, not the multiply, is what bounds it. Here a block owns
// 64 output channels x a 16 x 16 output patch of one image, and per 4-channel
// slice stages the patch's input WITH its one-pixel halo (4 x 18 x 18, zero
// outside the image) and the slice's 4 x 9 x 64 weights (transposed so one
// tap's channels are contiguous) - each staged element is read once from
// global memory. A thread owns 8 channels x 8 consecutive positions of one
// output row; for each (channel, kernel row) it reads the 10 inputs that row
// needs as three 16-byte loads and slides them across the three taps in
// registers, so a tap costs two 16-byte weight loads (a warp-wide broadcast)
// for 64 multiply-adds.
//
// Every output is summed in ascending (ci, kh, kw) - the implicit GEMM's own
// reduction order - with one fused multiply-add per term, so the two forms
// produce the same bits, and the gate holds them to it.
//
// Banks: the staged input row stride is 20 floats (five 16-byte units, odd),
// and the eight lanes of a quarter-warp read eight consecutive rows at the
// same column, so their 16-byte loads land on distinct bank groups.
//
// One block per SM: the 64 accumulators, the register-staged next slice
// (15 values) and the 12 inputs and 8 weights of a tap need more than the 128
// registers two resident blocks would allow, and capped there the compiler
// spilled; the uncapped kernel ran at twice the rate of the capped one.
// ---------------------------------------------------------------------------
#define BRAIN_C3_CO 64       // output channels per block
#define BRAIN_C3_TH 16       // output rows per block
#define BRAIN_C3_TW 16       // output columns per block
#define BRAIN_C3_CS 4        // input channels staged per iteration
#define BRAIN_C3_IW 20       // staged input row stride: TW + 2 halo, padded to an odd number of float4
#define BRAIN_C3_IN (BRAIN_C3_CS * (BRAIN_C3_TH + 2) * BRAIN_C3_IW)
#define BRAIN_C3_WS 68       // staged weight row stride (64 channels + 4)
#define BRAIN_C3_WN (BRAIN_C3_CS * 9 * BRAIN_C3_WS)
#define BRAIN_C3_STAGE (BRAIN_C3_IN + BRAIN_C3_WN)

__device__ __forceinline__ bool brain_c3_serves(const BrainConvP& c) {
    return c.K == 3u && c.stride == 1u && c.pad == 1u && c.Ho == c.H && c.Wo == c.W && c.Cout >= 64u;
}

__device__ __forceinline__ void brain_conv3x3_tile(const BrainConvP& c, const float* x, const float* w, const float* bias, float* y,
                                                   unsigned blk, float* smem) {
    const unsigned tiles_w = (c.W + BRAIN_C3_TW - 1u) / BRAIN_C3_TW;
    const unsigned tiles_h = (c.H + BRAIN_C3_TH - 1u) / BRAIN_C3_TH;
    const unsigned tiles_co = (c.Cout + BRAIN_C3_CO - 1u) / BRAIN_C3_CO;
    const unsigned tco = blk % tiles_co;
    unsigned r = blk / tiles_co;
    const unsigned tw = r % tiles_w;
    r /= tiles_w;
    const unsigned th = r % tiles_h;
    const unsigned n = r / tiles_h;
    const unsigned co0 = tco * BRAIN_C3_CO, h0 = th * BRAIN_C3_TH, w0 = tw * BRAIN_C3_TW;
    const unsigned HW = c.H * c.W;
    const unsigned KK9 = c.Cin * 9u;

    const int tid = threadIdx.x;
    const int cog = tid / 32;          // eight output channels: co0 + 8*cog
    const int lane = tid % 32;
    const int orow = lane % 16;        // output row within the tile
    const int ocol = (lane / 16) * 8;  // first of eight output columns

    auto in_buf = [smem](int b) { return smem + b * BRAIN_C3_STAGE; };
    auto w_buf = [smem](int b) { return smem + b * BRAIN_C3_STAGE + BRAIN_C3_IN; };

    // Staging: the input slice (CS channels x 18 rows x 18 columns, zero
    // outside the image) and the weight slice (CS x 9 taps x 64 channels,
    // transposed so a tap's channels are contiguous).
    constexpr int IN_ELEMS = BRAIN_C3_CS * (BRAIN_C3_TH + 2) * (BRAIN_C3_TW + 2);
    constexpr int IN_LOADS = (IN_ELEMS + BRAIN_CV_THREADS - 1) / BRAIN_CV_THREADS;
    constexpr int W_ELEMS = BRAIN_C3_CS * 9 * BRAIN_C3_CO;
    constexpr int W_LOADS = W_ELEMS / BRAIN_CV_THREADS;
    float rin[IN_LOADS];
    float rw[W_LOADS];
    const float* xn = x + (unsigned long long)n * c.Cin * HW;
    auto load = [&](unsigned ci0) {
#pragma unroll
        for (int i = 0; i < IN_LOADS; ++i) {
            const int e = tid + BRAIN_CV_THREADS * i;
            const int ci = e / ((BRAIN_C3_TH + 2) * (BRAIN_C3_TW + 2));
            const int rem = e - ci * ((BRAIN_C3_TH + 2) * (BRAIN_C3_TW + 2));
            const int ih = rem / (BRAIN_C3_TW + 2);
            const int iw = rem - ih * (BRAIN_C3_TW + 2);
            const int hh = (int)h0 + ih - 1, ww = (int)w0 + iw - 1;
            const bool ok = e < IN_ELEMS && ci0 + ci < c.Cin && (unsigned)hh < c.H && (unsigned)ww < c.W;
            rin[i] = ok ? xn[(unsigned long long)(ci0 + ci) * HW + hh * c.W + ww] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < W_LOADS; ++i) {
            const int e = tid + BRAIN_CV_THREADS * i;
            const int col = e / (BRAIN_C3_CS * 9);       // output channel within the tile
            const int kt = e - col * (BRAIN_C3_CS * 9);  // (ci, kh, kw) within the slice
            const unsigned co = co0 + col;
            const unsigned k = ci0 * 9u + kt;
            rw[i] = (co < c.Cout && k < KK9) ? w[(unsigned long long)co * KK9 + k] : 0.0f;
        }
    };
    auto store = [&](int b) {
        float* ib = in_buf(b);
#pragma unroll
        for (int i = 0; i < IN_LOADS; ++i) {
            const int e = tid + BRAIN_CV_THREADS * i;
            if (e < IN_ELEMS) {
                const int ci = e / ((BRAIN_C3_TH + 2) * (BRAIN_C3_TW + 2));
                const int rem = e - ci * ((BRAIN_C3_TH + 2) * (BRAIN_C3_TW + 2));
                const int ih = rem / (BRAIN_C3_TW + 2);
                const int iw = rem - ih * (BRAIN_C3_TW + 2);
                ib[(ci * (BRAIN_C3_TH + 2) + ih) * BRAIN_C3_IW + iw] = rin[i];
            }
        }
        float* wb = w_buf(b);
#pragma unroll
        for (int i = 0; i < W_LOADS; ++i) {
            const int e = tid + BRAIN_CV_THREADS * i;
            const int col = e / (BRAIN_C3_CS * 9);
            const int kt = e - col * (BRAIN_C3_CS * 9);
            wb[kt * BRAIN_C3_WS + col] = rw[i];
        }
    };

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
    }

    load(0u);
    store(0);
    __syncthreads();
    int cur = 0;
    for (unsigned ci0 = 0; ci0 < c.Cin; ci0 += BRAIN_C3_CS) {
        const bool more = ci0 + BRAIN_C3_CS < c.Cin;
        if (more) { load(ci0 + BRAIN_C3_CS); }
        const float* ib = in_buf(cur);
        const float* wb = w_buf(cur);
#pragma unroll 1
        for (int ci = 0; ci < BRAIN_C3_CS; ++ci) {
#pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                const float* row = ib + (ci * (BRAIN_C3_TH + 2) + orow + kh) * BRAIN_C3_IW + ocol;
                const float4 r0 = *reinterpret_cast<const float4*>(row);
                const float4 r1 = *reinterpret_cast<const float4*>(row + 4);
                const float4 r2 = *reinterpret_cast<const float4*>(row + 8);
                const float b[12] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w};
#pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    const float* wr = wb + (ci * 9 + kh * 3 + kw) * BRAIN_C3_WS + 8 * cog;
                    const float4 a0 = *reinterpret_cast<const float4*>(wr);
                    const float4 a1 = *reinterpret_cast<const float4*>(wr + 4);
                    const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) { acc[i][j] = fmaf(a[i], b[kw + j], acc[i][j]); }
                    }
                }
            }
        }
        if (more) { store(cur ^ 1); }
        __syncthreads();
        cur ^= 1;
    }

    const unsigned oh = h0 + orow;
    if (oh >= c.H) { return; }
    const bool vec = (c.W % 4u == 0u) && ((reinterpret_cast<unsigned long long>(y) & 15ull) == 0ull);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned co = co0 + 8 * cog + i;
        if (co >= c.Cout) { continue; }
        if (bias != nullptr) {
            const float bv = bias[co];
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[i][j] = acc[i][j] + bv; }
        }
        float* dst = y + ((unsigned long long)n * c.Cout + co) * HW + oh * c.W;
#pragma unroll
        for (int run = 0; run < 2; ++run) {
            const unsigned ow = w0 + ocol + 4 * run;
            if (vec && ow + 3 < c.W) {
                *reinterpret_cast<float4*>(dst + ow) = make_float4(acc[i][4 * run], acc[i][4 * run + 1], acc[i][4 * run + 2], acc[i][4 * run + 3]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (ow + j < c.W) { dst[ow + j] = acc[i][4 * run + j]; }
                }
            }
        }
    }
}

// The forward's block: decode the tile, pick the channel-tile width. `bias`
// is null for `conv2d` and the per-output-channel bias for `conv_bias`.
__device__ __forceinline__ void brain_conv2d_fwd_block(const unsigned int* params, const float* x, const float* w, const float* bias,
                                                       float* y, float* smem) {
    const BrainConvP c = brain_cv_params(params);
    const unsigned P = c.N * c.Ho * c.Wo;
    const unsigned tiles_n = (P + BRAIN_CV_BN - 1u) / BRAIN_CV_BN;
    const unsigned bm = brain_cv_bm(c.Cout);
    const unsigned tiles_m = (c.Cout + bm - 1u) / bm;
    const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (tiles_m == 0u || blk >= tiles_m * tiles_n) { return; }  // block-uniform
    const unsigned tile_m = blk % tiles_m;
    const unsigned tile_n = blk / tiles_m;
    if (bm == 16u) {
        brain_conv2d_fwd_tile<16>(c, x, w, bias, y, tile_m, tile_n, smem);
    } else if (bm == 32u) {
        brain_conv2d_fwd_tile<32>(c, x, w, bias, y, tile_m, tile_n, smem);
    } else {
        brain_conv2d_fwd_tile<64>(c, x, w, bias, y, tile_m, tile_n, smem);
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS, 2) brain_conv2d_fwd(const unsigned int* params, const float* x, const float* w,
                                                                              float* y) {
    __shared__ __align__(16) float smem[2 * BRAIN_CV_AS + 2 * BRAIN_CV_BS];
    brain_conv2d_fwd_block(params, x, w, nullptr, y, smem);
}

// `conv_bias.wgsl`'s form: the same forward, plus `bias[co]` on each output.
extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS, 2) brain_conv2d_bias_fwd(const unsigned int* params, const float* x,
                                                                                   const float* w, const float* bias, float* y) {
    __shared__ __align__(16) float smem[2 * BRAIN_CV_AS + 2 * BRAIN_CV_BS];
    brain_conv2d_fwd_block(params, x, w, bias, y, smem);
}

// The direct 3x3 forward's block: decode the tile, or exit if the grid has
// more blocks than the shape (block-uniform).
__device__ __forceinline__ void brain_conv3x3_block(const unsigned int* params, const float* x, const float* w, const float* bias, float* y,
                                                    float* smem) {
    const BrainConvP c = brain_cv_params(params);
    if (!brain_c3_serves(c)) { return; }
    const unsigned blocks = ((c.Cout + BRAIN_C3_CO - 1u) / BRAIN_C3_CO) * c.N * ((c.H + BRAIN_C3_TH - 1u) / BRAIN_C3_TH) *
                            ((c.W + BRAIN_C3_TW - 1u) / BRAIN_C3_TW);
    const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (blk < blocks) { brain_conv3x3_tile(c, x, w, bias, y, blk, smem); }
}

extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS, 1) brain_conv2d_fwd3x3(const unsigned int* params, const float* x,
                                                                                const float* w, float* y) {
    __shared__ __align__(16) float smem[2 * BRAIN_C3_STAGE];
    brain_conv3x3_block(params, x, w, nullptr, y, smem);
}

extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS, 1) brain_conv2d_bias_fwd3x3(const unsigned int* params, const float* x,
                                                                                     const float* w, const float* bias, float* y) {
    __shared__ __align__(16) float smem[2 * BRAIN_C3_STAGE];
    brain_conv3x3_block(params, x, w, bias, y, smem);
}

// ---------------------------------------------------------------------------
// Input gradient, one stride class per block.
// ---------------------------------------------------------------------------
template <int BM>
__device__ __forceinline__ void brain_conv2d_dx_tile(const BrainConvP& c, const float* dy, const float* w, float* dx,
                                                     unsigned tile_m, unsigned cls, unsigned tile_n, float* smem) {
    constexpr int TM = BM / 16;
    constexpr int AS = BM + 4;
    constexpr int A_LOADS = BM * BRAIN_CV_BK / BRAIN_CV_THREADS;
    auto as_buf = [smem](int b) { return smem + b * BRAIN_CV_AS; };
    auto bs_buf = [smem](int b) { return smem + 2 * BRAIN_CV_AS + b * BRAIN_CV_BS; };

    const int tid = threadIdx.x;
    const int ty = tid / 16, tx = tid % 16;
    const unsigned s = c.stride;
    const unsigned rh = cls / s, rw = cls - (cls / s) * s;
    const unsigned Hc = c.H > rh ? (c.H - rh + s - 1u) / s : 0u;
    const unsigned Wc = c.W > rw ? (c.W - rw + s - 1u) / s : 0u;
    const unsigned HWc = Hc * Wc;
    const unsigned Pc = c.N * HWc;
    const unsigned p0 = tile_n * BRAIN_CV_BN;
    if (p0 >= Pc) { return; }  // block-uniform: this class has fewer positions
    // Live taps of this class: kh = kh0 + s*a for a < Ah (likewise kw).
    const unsigned kh0 = (rh + c.pad) % s, kw0 = (rw + c.pad) % s;
    const unsigned Ah = c.K > kh0 ? (c.K - kh0 + s - 1u) / s : 0u;
    const unsigned Aw = c.K > kw0 ? (c.K - kw0 + s - 1u) / s : 0u;
    const unsigned AA = Ah * Aw;
    const unsigned Kd = c.Cout * AA;
    const int hb = (int)((rh + c.pad - kh0) / s);
    const int wb = (int)((rw + c.pad - kw0) / s);
    const unsigned HoWo = c.Ho * c.Wo;
    const unsigned KK = c.K * c.K;
    const unsigned ci0 = tile_m * BM;

    // A (gathered weights): fixed k' column per thread, rows ci0 + arow + 16i.
    const int acol = tid % BRAIN_CV_BK;
    const int arow = tid / BRAIN_CV_BK;
    // B (dy): fixed class position per thread, rows krow + 2i.
    const int pcol = tid % BRAIN_CV_BN;
    const int krow = tid / BRAIN_CV_BN;
    const unsigned pp = p0 + pcol;
    const bool p_ok = pp < Pc;
    int i_pos = 0, j_pos = 0;
    long long dybase = 0;
    if (p_ok) {
        const unsigned n = pp / HWc;
        const unsigned r = pp - n * HWc;
        i_pos = (int)(r / Wc);
        j_pos = (int)(r - (unsigned)i_pos * Wc);
        dybase = (long long)n * c.Cout * HoWo;
    }
    const int Ahi = (int)Ah, Awi = (int)Aw;
    // (co, a, b) of the A column and of the first B row; advances by 2 and 16.
    BrainCvRadix ka = AA ? brain_cv_split3((unsigned)acol, Ah, Aw) : BrainCvRadix{0, 0, 0};
    BrainCvRadix kb = AA ? brain_cv_split3((unsigned)krow, Ah, Aw) : BrainCvRadix{0, 0, 0};
    const BrainCvRadix d2 = AA ? brain_cv_split3(2u, Ah, Aw) : BrainCvRadix{0, 0, 0};
    const BrainCvRadix d16 = AA ? brain_cv_split3((unsigned)BRAIN_CV_BK, Ah, Aw) : BrainCvRadix{0, 0, 0};

    float ra[A_LOADS];
    float rb[8];

    auto load = [&](unsigned k0, BrainCvRadix kav, BrainCvRadix kbv) {
        {
            const unsigned k = k0 + acol;
            const unsigned kh = kh0 + s * (unsigned)kav.r;
            const unsigned kw = kw0 + s * (unsigned)kav.s;
            const unsigned long long wbase = ((unsigned long long)kav.q * c.Cin) * KK + kh * c.K + kw;
#pragma unroll
            for (int i = 0; i < A_LOADS; ++i) {
                const unsigned ci = ci0 + arow + 16 * i;
                ra[i] = (ci < c.Cin && k < Kd) ? w[wbase + (unsigned long long)ci * KK] : 0.0f;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned k = k0 + krow + 2 * i;
            const int ho = hb + i_pos - kbv.r;
            const int wo = wb + j_pos - kbv.s;
            const bool ok = p_ok && k < Kd && (unsigned)ho < c.Ho && (unsigned)wo < c.Wo;
            rb[i] = ok ? dy[dybase + (long long)kbv.q * HoWo + ho * (int)c.Wo + wo] : 0.0f;
            if (i < 7) { brain_cv_adv3(kbv, d2, Ahi, Awi); }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < A_LOADS; ++i) { as_buf(buf)[acol * AS + arow + 16 * i] = ra[i]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) { bs_buf(buf)[(krow + 2 * i) * BRAIN_CV_BN + pcol] = rb[i]; }
    };

    float acc[TM][BRAIN_CV_TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < BRAIN_CV_TN; ++j) { acc[i][j] = 0.0f; }
    }

    if (Kd > 0u) {
        load(0u, ka, kb);
        store(0);
        __syncthreads();
        int cur = 0;
        for (unsigned k0 = 0; k0 < Kd; k0 += BRAIN_CV_BK) {
            const bool more = k0 + BRAIN_CV_BK < Kd;
            if (more) {
                brain_cv_adv3(ka, d16, Ahi, Awi);
                brain_cv_adv3(kb, d16, Ahi, Awi);
                load(k0 + BRAIN_CV_BK, ka, kb);
            }
            brain_cv_mma<TM>(as_buf(cur), AS, bs_buf(cur), acc, ty, tx);
            if (more) { store(cur ^ 1); }
            __syncthreads();
            cur ^= 1;
        }
    }

    // Write: rows ci0 + ty*TM + i; class positions as `brain_cv_col` lays them
    // out, scattered back to (n, rh + s*i, rw + s*j). A class with no live tap
    // writes zeros: dx is overwritten, never accumulated. With stride 1 the
    // class IS the whole map and a run of four is one 16-byte store where the
    // map is a whole number of vectors and the binding is aligned.
    const unsigned HW = c.H * c.W;
    const bool vec = (s == 1u) && (HW % 4u == 0u) && ((reinterpret_cast<unsigned long long>(dx) & 15ull) == 0ull);
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const unsigned ci = ci0 + ty * TM + i;
        if (ci >= c.Cin) { continue; }
#pragma unroll
        for (int run = 0; run < 2; ++run) {
            const unsigned pr = p0 + brain_cv_col(tx, 4 * run);
            if (pr >= Pc) { continue; }
            if (vec) {
                const unsigned n = pr / HWc, r = pr - (pr / HWc) * HWc;
                *reinterpret_cast<float4*>(dx + ((unsigned long long)n * c.Cin + ci) * HW + r) =
                    make_float4(acc[i][4 * run], acc[i][4 * run + 1], acc[i][4 * run + 2], acc[i][4 * run + 3]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned p = pr + j;
                    if (p < Pc) {
                        const unsigned n = p / HWc, r = p - (p / HWc) * HWc;
                        const unsigned ii = r / Wc, jj = r - (r / Wc) * Wc;
                        dx[((unsigned long long)n * c.Cin + ci) * HW + (rh + s * ii) * c.W + (rw + s * jj)] = acc[i][4 * run + j];
                    }
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS, 2) brain_conv2d_dx(const unsigned int* params, const float* dy, const float* w,
                                                                             float* dx) {
    __shared__ __align__(16) float smem[2 * BRAIN_CV_AS + 2 * BRAIN_CV_BS];
    const BrainConvP c = brain_cv_params(params);
    const unsigned s = c.stride;
    // Class (0, 0) has the most positions; smaller classes exit early.
    const unsigned pc_max = s ? c.N * ((c.H + s - 1u) / s) * ((c.W + s - 1u) / s) : 0u;
    const unsigned tiles_n = (pc_max + BRAIN_CV_BN - 1u) / BRAIN_CV_BN;
    const unsigned classes = s * s;
    const unsigned bm = brain_cv_bm(c.Cin);
    const unsigned tiles_m = (c.Cin + bm - 1u) / bm;
    const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (tiles_m == 0u || s == 0u || blk >= tiles_m * classes * tiles_n) { return; }
    const unsigned tile_m = blk % tiles_m;
    const unsigned rest = blk / tiles_m;
    const unsigned cls = rest % classes;
    const unsigned tile_n = rest / classes;
    if (bm == 16u) {
        brain_conv2d_dx_tile<16>(c, dy, w, dx, tile_m, cls, tile_n, smem);
    } else if (bm == 32u) {
        brain_conv2d_dx_tile<32>(c, dy, w, dx, tile_m, cls, tile_n, smem);
    } else {
        brain_conv2d_dx_tile<64>(c, dy, w, dx, tile_m, cls, tile_n, smem);
    }
}

// ---------------------------------------------------------------------------
// Weight gradient, split over positions.
//
// params: the conv uniform, then [S, chunk]: S position slices of `chunk`
// positions each (a multiple of 32). With S == 1 `out` IS dw and the block
// accumulates into it; with S > 1 `out` is an [S, Cout, Cin*K*K] scratch
// that `brain_conv2d_dw_reduce` folds into dw.
//
// Tile: BM output channels x 64 weight columns (ci, kh, kw); a stage stages
// 32 positions of both operands, lane = position so both gathers coalesce.
// ---------------------------------------------------------------------------
#define BRAIN_CV_DW_AS(BM) (BRAIN_CV_DW_BP * ((BM) + 4))
#define BRAIN_CV_DW_BS (BRAIN_CV_DW_BP * (BRAIN_CV_DW_BN + 4))

template <int BM>
__device__ __forceinline__ void brain_conv2d_dw_tile(const BrainConvP& c, const float* dy, const float* x, float* out, unsigned S,
                                                     unsigned chunk, unsigned tile_m, unsigned tile_n, unsigned split, float* smem) {
    constexpr int TM = BM / 16;
    constexpr int AS = BM + 4;
    constexpr int BS = BRAIN_CV_DW_BN + 4;
    constexpr int A_ROWS = BM / 8;                 // dy rows per warp
    constexpr int B_COLS = BRAIN_CV_DW_BN / 8;     // weight columns per warp
    auto as_buf = [smem](int b) { return smem + b * BRAIN_CV_DW_AS(BM); };
    auto bs_buf = [smem](int b) { return smem + 2 * BRAIN_CV_DW_AS(BM) + b * BRAIN_CV_DW_BS; };

    const int tid = threadIdx.x;
    const int ty = tid / 16, tx = tid % 16;
    const int lane = tid % 32, warp = tid / 32;
    const unsigned KK = c.K * c.K;
    const unsigned Kd = c.Cin * KK;
    const unsigned HW = c.H * c.W;
    const unsigned HoWo = c.Ho * c.Wo;
    const unsigned P = c.N * HoWo;
    const unsigned co0 = tile_m * BM;
    const unsigned kc0 = tile_n * BRAIN_CV_DW_BN;
    const unsigned pbeg = split * chunk;
    const unsigned pend = min(P, pbeg + chunk);

    // This thread's B columns: tap offset ci*HW + kh*W + kw and (kh, kw).
    int toff[B_COLS];
    int tkh[B_COLS];
    int tkw[B_COLS];
#pragma unroll
    for (int j = 0; j < B_COLS; ++j) {
        const unsigned k = kc0 + warp * B_COLS + j;
        if (k < Kd) {
            const unsigned ci = k / KK;
            const unsigned r = k - ci * KK;
            const unsigned kh = r / c.K;
            const unsigned kw = r - kh * c.K;
            toff[j] = (int)(ci * HW + kh * c.W + kw);
            tkh[j] = (int)kh;
            tkw[j] = (int)kw;
        } else {
            toff[j] = 0;
            tkh[j] = 1 << 28;  // out of every range: the load is skipped
            tkw[j] = 0;
        }
    }
    // This lane's position, in radix (N, Ho, Wo), advanced by 32 per stage.
    const int Ho = (int)c.Ho, Wo = (int)c.Wo;
    BrainCvRadix pv = brain_cv_split3(pbeg + lane, c.Ho, c.Wo);
    const BrainCvRadix d32 = brain_cv_split3((unsigned)BRAIN_CV_DW_BP, c.Ho, c.Wo);

    float ra[A_ROWS];
    float rb[B_COLS];
    auto load = [&](unsigned pstage, const BrainCvRadix& v) {
        const unsigned p = pstage + lane;
        const bool p_ok = p < pend;
        const long long q = (long long)v.r * Wo + v.s;
        const long long dyb = (long long)v.q * c.Cout * HoWo + q;
#pragma unroll
        for (int i = 0; i < A_ROWS; ++i) {
            const unsigned co = co0 + warp * A_ROWS + i;
            ra[i] = (p_ok && co < c.Cout) ? dy[dyb + (long long)co * HoWo] : 0.0f;
        }
        const int hi0 = v.r * (int)c.stride - (int)c.pad;
        const int wi0 = v.s * (int)c.stride - (int)c.pad;
        const long long xb = (long long)v.q * c.Cin * HW + (long long)hi0 * c.W + wi0;
#pragma unroll
        for (int j = 0; j < B_COLS; ++j) {
            const bool ok = p_ok && (unsigned)(hi0 + tkh[j]) < c.H && (unsigned)(wi0 + tkw[j]) < c.W;
            rb[j] = ok ? x[xb + toff[j]] : 0.0f;
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < A_ROWS; ++i) { as_buf(buf)[lane * AS + warp * A_ROWS + i] = ra[i]; }
#pragma unroll
        for (int j = 0; j < B_COLS; ++j) { bs_buf(buf)[lane * BS + warp * B_COLS + j] = rb[j]; }
    };

    float acc[TM][4];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    if (pbeg < pend) {
        load(pbeg, pv);
        store(0);
        __syncthreads();
        int cur = 0;
        for (unsigned ps = pbeg; ps < pend; ps += BRAIN_CV_DW_BP) {
            const bool more = ps + BRAIN_CV_DW_BP < pend;
            if (more) {
                brain_cv_adv3(pv, d32, Ho, Wo);
                load(ps + BRAIN_CV_DW_BP, pv);
            }
            const float* as = as_buf(cur);
            const float* bs = bs_buf(cur);
#pragma unroll 8
            for (int pp = 0; pp < BRAIN_CV_DW_BP; ++pp) {
                float a[TM];
                if (TM == 4) {
                    const float4 v = *reinterpret_cast<const float4*>(as + pp * AS + ty * TM);
                    a[0] = v.x; a[1 % TM] = v.y; a[2 % TM] = v.z; a[3 % TM] = v.w;
                } else if (TM == 2) {
                    const float2 v = *reinterpret_cast<const float2*>(as + pp * AS + ty * TM);
                    a[0] = v.x; a[1 % TM] = v.y;
                } else {
                    a[0] = as[pp * AS + ty * TM];
                }
                const float4 b = *reinterpret_cast<const float4*>(bs + pp * BS + tx * 4);
                const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < TM; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { acc[i][j] = fmaf(a[i], bv[j], acc[i][j]); }
                }
            }
            if (more) { store(cur ^ 1); }
            __syncthreads();
            cur ^= 1;
        }
    }

    const unsigned long long plane = (unsigned long long)c.Cout * Kd;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const unsigned co = co0 + ty * TM + i;
        if (co >= c.Cout) { continue; }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned k = kc0 + tx * 4 + j;
            if (k >= Kd) { continue; }
            const unsigned long long idx = (unsigned long long)co * Kd + k;
            if (S == 1u) {
                out[idx] = out[idx] + acc[i][j];
            } else {
                out[split * plane + idx] = acc[i][j];
            }
        }
    }
}

// Shared memory for the weight gradient at its widest tile (BM = 64).
#define BRAIN_CV_DW_SMEM (2 * BRAIN_CV_DW_AS(64) + 2 * BRAIN_CV_DW_BS)

extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS, 2) brain_conv2d_dw_partial(const unsigned int* params, const float* dy,
                                                                                     const float* x, float* out) {
    __shared__ __align__(16) float smem[BRAIN_CV_DW_SMEM];
    const BrainConvP c = brain_cv_params(params);
    const unsigned S = params[10];
    const unsigned chunk = params[11];
    const unsigned bm = brain_cv_bm(c.Cout);
    const unsigned tiles_m = (c.Cout + bm - 1u) / bm;
    const unsigned Kd = c.Cin * c.K * c.K;
    const unsigned tiles_n = (Kd + BRAIN_CV_DW_BN - 1u) / BRAIN_CV_DW_BN;
    const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
    if (tiles_m == 0u || S == 0u || chunk % BRAIN_CV_DW_BP != 0u || blk >= tiles_m * tiles_n * S) { return; }
    const unsigned tile_m = blk % tiles_m;
    const unsigned rest = blk / tiles_m;
    const unsigned tile_n = rest % tiles_n;
    const unsigned split = rest / tiles_n;
    if (bm == 16u) {
        brain_conv2d_dw_tile<16>(c, dy, x, out, S, chunk, tile_m, tile_n, split, smem);
    } else if (bm == 32u) {
        brain_conv2d_dw_tile<32>(c, dy, x, out, S, chunk, tile_m, tile_n, split, smem);
    } else {
        brain_conv2d_dw_tile<64>(c, dy, x, out, S, chunk, tile_m, tile_n, split, smem);
    }
}

// dw[i] += part[0][i] + part[1][i] + ... + part[S-1][i], ascending slice
// order. params: [total, S]. One thread per weight element.
extern "C" __global__ void __launch_bounds__(BRAIN_CV_THREADS) brain_conv2d_dw_reduce(const unsigned int* params, const float* part,
                                                                                    float* dw) {
    const unsigned total = params[0];
    const unsigned S = params[1];
    const unsigned i = (blockIdx.y * gridDim.x + blockIdx.x) * BRAIN_CV_THREADS + threadIdx.x;
    if (i >= total || S == 0u) { return; }
    float sum = part[i];
    for (unsigned s = 1; s < S; ++s) { sum += part[(unsigned long long)s * total + i]; }
    dw[i] = dw[i] + sum;
}
