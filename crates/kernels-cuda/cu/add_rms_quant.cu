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
//   sx   = max(max|xn|, 1e-8) / 127 per 32 elements (the activation group scales)
//   xq   = pack4(clamp(rint(xn / sx), -127, 127))  (four int8 per u32, each by its group's scale)
//
// It replaces `add2` + `rmsnorm_rows` + `max_abs_rows` + `quant_pack`, which a
// decode token runs 128 times back to back on a 5120-wide row, and produces
// exactly what that chain produces, to the last bit.
//
//   params : u32 [d, rows, eps (f32 bits), flags]   flags bit 0: add `b` first; bit 1: no int8 packing (`xq`, `sx` are not
//                                                   written) for consumers that read bf16 or E4M3 rows
//   a, b   : [rows, d] f32     b is not read (and sum is not written) without bit 0
//   w      : [d] f32           the norm gain
//   sum    : [rows, d] f32     a + b, the residual stream's next value
//   xn     : [rows, d] f32     the normalised activation (an fp32 linear reads it)
//   xq     : [rows, d/4] u32   packed int8 activation
//   sx     : [rows, d/32] f32  per-group scale
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
// partials in ascending lane order. This kernel is one block per row with the
// same lane ownership, four threads to a lane (each a contiguous quarter of the
// lane's elements, the lane's running sum handed from quarter to quarter), so
// an element stays in one thread's registers from the load to the quantiser
// and the accumulation order is the reference's by construction. Element c = lane + 64j of a thread lies in
// group 2j + lane/32, so a group's 32 elements are one warp's element j and its max is a warp reduction. The max
// is exact in any order, the scale and
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
// shared memory; then every global store. The reference's 64 lanes are what
// fix its accumulation order; the threads that serve them are free.
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

// The reference's lane count: lane l owns the elements l, l + 64, l + 128, ... and its sum of squares adds them in
// that order.
#define BRAIN_ARQ_LANES 64
// Elements of one lane: rows up to 64 * this wide (Qwen3.8-27B's 5120 uses all 80).
#define BRAIN_ARQ_MAX_ELEMS 80
// Threads that share one lane, each holding a contiguous quarter of its elements: thread `q * 64 + l` holds lane
// l's elements 20q .. 20q + 19. The loads, the norm and the quantiser run four times as wide as the reference's
// lane count, while the lane's sum still walks its 80 elements in order (each quarter continues the previous
// quarter's running sum), so the result is the same bits. One 64-thread block per row was latency-bound at about
// 8 us a 5120 row - 2,500 dependent instructions a warp - in a decode token that runs it 128 times.
#define BRAIN_ARQ_SPLIT 4
#define BRAIN_ARQ_THREADS (BRAIN_ARQ_LANES * BRAIN_ARQ_SPLIT)
#define BRAIN_ARQ_PER (BRAIN_ARQ_MAX_ELEMS / BRAIN_ARQ_SPLIT)

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

extern "C" __global__ void __launch_bounds__(BRAIN_ARQ_THREADS)
brain_add_rms_quant(const unsigned int* params, const float* a, const float* b, const float* w,
                    float* sum, float* xn, unsigned int* xq, float* sx) {
    BRAIN_PDL_ENTRY();
    const unsigned int d = params[0];
    const unsigned int rows = params[1];
    const float eps = __uint_as_float(params[2]);
    const bool add = (params[3] & 1u) != 0u;
    const bool pack = (params[3] & 2u) == 0u;

    const unsigned int row = blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) { return; }  // block-uniform
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % BRAIN_ARQ_LANES, quarter = tid / BRAIN_ARQ_LANES;
    const unsigned long long base = (unsigned long long)row * d;

    // Shared memory is static (a native launch requests no dynamic shared
    // memory), sized for the widest served row.
    __shared__ float carry[BRAIN_ARQ_LANES];     // a lane's running sum of squares, handed quarter to quarter
    __shared__ float partial[BRAIN_ARQ_LANES];   // the lanes' sums
    __shared__ unsigned int qwords[BRAIN_ARQ_LANES * BRAIN_ARQ_MAX_ELEMS / 4];
    signed char* qbytes = reinterpret_cast<signed char*>(qwords);

    // Phase 1: every load, nothing else. Element i of this thread is the lane's element quarter * PER + i.
    float v[BRAIN_ARQ_PER];
    float g[BRAIN_ARQ_PER];
    float u[BRAIN_ARQ_PER];
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_PER; ++i) {
        const unsigned int c = lane + BRAIN_ARQ_LANES * (quarter * BRAIN_ARQ_PER + i);
        const bool live = c < d;
        v[i] = live ? a[base + c] : 0.0f;
        u[i] = (live && add) ? b[base + c] : 0.0f;
        g[i] = live ? w[c] : 0.0f;
    }

    // Phase 2: the residual add and the squares, in parallel; then the lane's sum in the reference's ascending order,
    // one quarter after the other through `carry`.
    float sq[BRAIN_ARQ_PER];
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_PER; ++i) {
        if (add) { v[i] = __fadd_rn(v[i], u[i]); }
        sq[i] = __fmul_rn(v[i], v[i]);
    }
    for (unsigned int qq = 0; qq < BRAIN_ARQ_SPLIT; ++qq) {
        if (quarter == qq) {
            float acc = qq == 0 ? 0.0f : carry[lane];
#pragma unroll
            for (int i = 0; i < BRAIN_ARQ_PER; ++i) {
                const unsigned int c = lane + BRAIN_ARQ_LANES * (quarter * BRAIN_ARQ_PER + i);
                if (c < d) { acc = __fadd_rn(acc, sq[i]); }
            }
            if (qq + 1 < BRAIN_ARQ_SPLIT) { carry[lane] = acc; } else { partial[lane] = acc; }
        }
        __syncthreads();
    }

    // Every thread folds the 64 partials in ascending lane order, exactly as the reference's every thread does: all
    // 64 are read first, then added in a register chain.
    float pv[BRAIN_ARQ_LANES];
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_LANES; ++i) { pv[i] = partial[i]; }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_LANES; ++i) { ss = __fadd_rn(ss, pv[i]); }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(d)), eps)));

    // The sum goes out now: it is only stored, and the normalisation below
    // overwrites it in registers.
    if (add) {
#pragma unroll
        for (int i = 0; i < BRAIN_ARQ_PER; ++i) {
            const unsigned int c = lane + BRAIN_ARQ_LANES * (quarter * BRAIN_ARQ_PER + i);
            if (c < d) { sum[base + c] = v[i]; }
        }
    }

    // Normalise in registers; then each element's group scale. Element i of this thread is element
    // j = quarter * PER + i of its lane, in group 2j + lane / 32: the same group as element i of every lane of
    // its warp, so the group's max is a warp reduction (exact in any order).
    const unsigned int groups = d >> 5;
    float rscale[BRAIN_ARQ_PER];
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_PER; ++i) {
        const float y = __fmul_rn(__fmul_rn(v[i], inv), g[i]);
        v[i] = y;
        if (!pack) { continue; }
        float mx = fabsf(y);
        for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
        const float scale = __fdiv_rn(fmaxf(mx, 1e-8f), 127.0f);
        const unsigned int c = lane + BRAIN_ARQ_LANES * (quarter * BRAIN_ARQ_PER + i);
        if ((lane & 31u) == 0u && c < d) { sx[(unsigned long long)row * groups + (c >> 5)] = scale; }
        rscale[i] = __fdiv_rn(1.0f, scale);
    }

    // Phase 3: quantise from registers into shared bytes, then store.
#pragma unroll
    for (int i = 0; i < BRAIN_ARQ_PER; ++i) {
        const unsigned int c = lane + BRAIN_ARQ_LANES * (quarter * BRAIN_ARQ_PER + i);
        if (c < d) {
            xn[base + c] = v[i];
            if (!pack) { continue; }
            const float q = fminf(fmaxf(rintf(__fmul_rn(v[i], rscale[i])), -127.0f), 127.0f);
            qbytes[c] = static_cast<signed char>(static_cast<int>(q));
        }
    }
    if (!pack) { return; }  // block-uniform
    __syncthreads();
    const unsigned int words = d >> 2;
    for (unsigned int gw = tid; gw < words; gw += BRAIN_ARQ_THREADS) {
        xq[(unsigned long long)row * words + gw] = qwords[gw];
    }
}
