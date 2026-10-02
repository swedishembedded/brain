// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels between two weight streams into one launch, then you can
// procure our services by sending an email to info@swedishembedded.com.
//
// The activation front end of one int8 linear in a single launch:
//
//   sum  = a + b                                   (optional residual add)
//   xn   = rmsnorm(sum) * w
//   sx   = max(max|xn|, 1e-8) / 127                (per-row activation scale)
//   xq   = pack4(clamp(rint(xn / sx), -127, 127))  (four int8 per u32)
//
// It replaces `add2` + `rmsnorm_rows` + `max_abs_rows` + `quant_pack`, which a
// decode token runs 128 times back to back on a 5120-wide row, and produces
// exactly what that chain produces, to the last bit.
//
//   params : u32 [d, rows, eps (f32 bits), flags]   flags bit 0: add `b` first
//   a, b   : [rows, d] f32     b is not read (and sum is not written) without bit 0
//   w      : [d] f32           the norm gain
//   sum    : [rows, d] f32     a + b, the residual stream's next value
//   xn     : [rows, d] f32     the normalised activation (an fp32 linear reads it)
//   xq     : [rows, d/4] u32   packed int8 activation
//   sx     : [rows] f32        per-row scale
//
// Why a decode token wanted this
// ------------------------------
// At one row the model is bandwidth-bound on its own weights, and the 2500
// kernels around the weight GEMVs were a third of the device time: the
// generated RMSNorm kernel takes ~39 us on one row because each of its 64
// threads runs a chain of 80 dependent global loads, and every kernel
// boundary costs about as much again in launch latency. Here a thread issues
// ALL of its loads before it consumes one, and the four stages share one
// pass over the row in registers.
//
// How it stays bit-identical to the chain it replaces
// ---------------------------------------------------
// The sum of squares is the only order-sensitive step. `rmsnorm_rows.wgsl`
// gives each of 64 lanes the elements c = t, t+64, t+128, ... added in
// ascending order into one f32 accumulator, then every thread folds the 64
// partials in ascending lane order. This kernel is one 64-thread block per
// row with the same lane ownership, so a thread's elements stay in its
// registers from the load to the quantiser and the accumulation order is the
// reference's by construction. The max is exact in any order, the scale and
// the quantiser are the reference's expressions (`/` and `rint` are correctly
// rounded; WGSL `round` is half-to-even, which is `rintf`), and
// `--fmad=false` is the backend's default and every product and sum below is
// an explicit round-to-nearest operation besides.
//
// The four int8 of an output word come from four different threads (element c
// belongs to lane c % 64), so the quantised bytes are exchanged through shared
// memory and each thread then packs and stores whole words.
//
// A row wider than BRAIN_ARQ_MAX_ELEMS * 64 (5120) is not served; the host keeps
// the WGSL chain for it.
//
// Why the body is written as three phases
// ---------------------------------------
// A version that interleaved loads with the arithmetic that consumes them (and
// stores with later loads) ran at 26 us on one row: the compiler may not move a
// load above a store through a pointer that might alias it (brain's buffers do
// alias by design), and it placed each load's dependent add right behind it,
// so the 160 loads of a row were 80 serialised round trips to L2. So every
// global load of the row is issued first, with nothing between them that
// could make one wait on another; then the arithmetic, in registers and
// shared memory; then every global store. 64 threads is the reference's own
// lane count and what keeps its accumulation order.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_ARQ_THREADS 64
// Elements one thread keeps in registers: rows up to 64 * this wide. Qwen3.8-27B's
// 5120 uses all 80; the value and the gain together are 160 registers.
#define BRAIN_ARQ_THREADS 64
#define BRAIN_ARQ_MAX_ELEMS 80

extern "C" __global__ void __launch_bounds__(BRAIN_ARQ_THREADS)
brain_add_rms_quant(const unsigned int* params, const float* a, const float* b, const float* w,
                    float* sum, float* xn, unsigned int* xq, float* sx) {
    const unsigned int d = params[0];
    const unsigned int rows = params[1];
    const float eps = __uint_as_float(params[2]);
    const bool add = (params[3] & 1u) != 0u;

    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }  // block-uniform
    const unsigned int t = threadIdx.x;
    const unsigned long long base = (unsigned long long)row * d;

    // Shared memory is static (a native launch requests no dynamic shared
    // memory), sized for the widest served row.
    __shared__ float partial[BRAIN_ARQ_THREADS];
    __shared__ unsigned int qwords[BRAIN_ARQ_THREADS * BRAIN_ARQ_MAX_ELEMS / 4];
    signed char* qbytes = reinterpret_cast<signed char*>(qwords);

    // Phase 1: every load, nothing else.
    float v[BRAIN_ARQ_MAX_ELEMS];
    float g[BRAIN_ARQ_MAX_ELEMS];
    float u[BRAIN_ARQ_MAX_ELEMS];
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_MAX_ELEMS; ++i) {
        const unsigned int c = t + BRAIN_ARQ_THREADS * i;
        const bool live = c < d;
        v[i] = live ? a[base + c] : 0.0f;
        u[i] = (live && add) ? b[base + c] : 0.0f;
        g[i] = live ? w[c] : 0.0f;
    }

    // Phase 2: the residual add, then this lane's sum of squares in the
    // reference's ascending order. `u` is dead after the add, `v` is the sum.
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_MAX_ELEMS; ++i) {
        const unsigned int c = t + BRAIN_ARQ_THREADS * i;
        if (add) { v[i] = __fadd_rn(v[i], u[i]); }
        if (c < d) { acc = __fadd_rn(acc, __fmul_rn(v[i], v[i])); }
    }
    partial[t] = acc;
    __syncthreads();

    // Every thread folds the 64 partials in ascending lane order, exactly as
    // the reference's every thread does.
    float ss = 0.0f;
    for (int i = 0; i < BRAIN_ARQ_THREADS; ++i) { ss = __fadd_rn(ss, partial[i]); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(d)), eps)));

    // The sum goes out now: it is only stored, and the normalisation below
    // overwrites it in registers.
    if (add) {
#pragma unroll
        for (int i = 0; i < BRAIN_ARQ_MAX_ELEMS; ++i) {
            const unsigned int c = t + BRAIN_ARQ_THREADS * i;
            if (c < d) { sum[base + c] = v[i]; }
        }
    }

    // Normalise in registers and track the lane's max |xn|.
    float mx = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_MAX_ELEMS; ++i) {
        const float y = __fmul_rn(__fmul_rn(v[i], inv), g[i]);
        v[i] = y;
        mx = fmaxf(mx, fabsf(y));
    }
    __syncthreads();  // `partial` is about to be reused
    partial[t] = mx;
    __syncthreads();
    float rowmax = 0.0f;
    for (int i = 0; i < BRAIN_ARQ_THREADS; ++i) { rowmax = fmaxf(rowmax, partial[i]); }
    const float scale = __fdiv_rn(fmaxf(rowmax, 1e-8f), 127.0f);
    if (t == 0) { sx[row] = scale; }
    const float rscale = __fdiv_rn(1.0f, scale);

    // Phase 3: quantise from registers into shared bytes, then store.
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_MAX_ELEMS; ++i) {
        const unsigned int c = t + BRAIN_ARQ_THREADS * i;
        if (c < d) {
            xn[base + c] = v[i];
            const float q = fminf(fmaxf(rintf(__fmul_rn(v[i], rscale)), -127.0f), 127.0f);
            qbytes[c] = static_cast<signed char>(static_cast<int>(q));
        }
    }
    __syncthreads();
    const unsigned int words = d >> 2;
    for (unsigned int gw = t; gw < words; gw += BRAIN_ARQ_THREADS) {
        xq[(unsigned long long)row * words + gw] = qwords[gw];
    }
}
