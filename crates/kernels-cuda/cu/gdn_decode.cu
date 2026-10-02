// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels that makes up a recurrent layer's token step into one launch,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// One or several consecutive Gated DeltaNet decode steps of ONE sequence, in a
// single launch. Each step is:
//
//   1. causal depthwise conv1d (4 taps) + SiLU on q|k|v, window shifted in place
//   2. q and k L2-normalised per key head
//   3. beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
//   4. the delta-rule state update per value head
//        S *= exp(g);  kv = k^T S;  delta = beta * (v - kv);  S += k (x) delta
//        out = (1/sqrt(dk)) * q^T S
//   5. gated RMSNorm per value head: rmsnorm(out) * silu(z)
//
// It replaces nineteen kernels (causal_conv1d_step, silu, three concat_split,
// two l2norm_scale, sigmoid, gdn_decay_gate, two kv_expand, gdn_state_decay,
// bmm, sub, scale_row, bmm_acc, bmm, rmsnorm, silu, mul) that a decode token
// runs 48 times, and produces what that chain produces to the last bit.
//
//   params : u32 [nkh, nvh, group, l2_eps (f32 bits), rms_eps (f32 bits),
//                 q_scale (f32 bits, 1/sqrt(dk)), rows, 0]   nvh = nkh * group
//   mixed  : [rows, conv_dim] f32   in_proj_qkv output, q | k | v, per token
//   conv_w : [conv_dim, 4] f32      depthwise filter taps
//   hist   : [conv_dim, 3] f32      the conv window (RW; shifted by every step)
//   bproj  : [rows, nvh] f32, aproj : [rows, nvh] f32
//   a_log  : [nvh] f32, dt_bias : [nvh] f32
//   state  : [nvh, 128, 128] f32    recurrent state (RW; updated in place)
//   z      : [rows, nvh * 128] f32  the output gate's pre-activation
//   norm_w : [128] f32              gated-norm gain
//   gated  : [rows, nvh * 128] f32  the layer's output before out_proj
//   conv_dim = 2 * nkh * 128 + nvh * 128; key and value head dims are both 128.
//
// Rows. The `rows` (at most 8) tokens are consecutive positions of one
// sequence, so row r+1 starts from the state row r left, and a launch of R rows
// is byte-identical to R launches of one - which is what a speculative verify
// round needs from its recurrent layers: the same arithmetic as plain decode,
// not an algebraically equal one.
//
// Only the delta-rule update itself is sequential over the rows. The conv, the
// L2 norms, the gates and the gated norm of a row depend on that row's inputs
// and on the raw inputs of the rows before it, never on the state, so they run
// for every row at once and the block walks the rows in order through nothing
// but the state recurrence:
//
//   A  conv + SiLU for every row (the window slides over the raw inputs, which
//      are all in hand), the gates of every row
//   B  the q and k L2 norms of every row, one thread per vector
//   C  the recurrence, row by row - the only serial stretch
//   D  the gated-norm inverse RMS of every row, one thread per row and head
//   E  the layer output for every row
//
// The state stays in registers across the rows - read once, written once - and
// each channel's conv window in the registers of the thread that owns it.
//
// Work split. One block per KEY head with `group` x 128 threads: the q and k
// channels of a key head are used by its `group` value heads, so a block that
// owns the key head owns every channel it reads, and the conv window of a
// channel is updated by exactly one block - which is why the whole chain can
// be one launch (a block per value head would have `group` blocks reading and
// one of them rewriting the same window with no order between them). Thread j
// of value-head slot s owns column j of that head's 128 x 128 state, so the
// state lives in registers for the whole update and is read and written once.
//
// How it stays bit-identical to the chain it replaces
// ---------------------------------------------------
// Every reduction is written in the reference's own order: the conv taps
// 0..3, the L2 and RMS sums over 0..127 by one thread in ascending order, the
// two state contractions over the key index ascending (the reference `bmm`
// accumulates `acc + a * b` for kk = 0.. into one f32), and each product and
// sum is an explicit round-to-nearest operation (`--fmad=false` is the
// backend's default besides). `expf`, `logf` and `sqrtf` are the functions the
// WGSL translator emits for `exp`, `log` and `sqrt`; `inverseSqrt` is
// `1.0f / sqrtf(x)` there and here. The rank-1 update goes through the same
// `0.0f + k*delta` the reference's `bmm_acc` computes, so a negative zero
// behaves identically too.
//
// The batched, pooled entry (`brain_gdn_decode_pool`)
// ---------------------------------------------------
// A serving engine keeps each resident sequence's recurrent state and conv window
// as a ROW of a per-layer pool, and a decode step is a batch of such sequences.
// `brain_gdn_decode_pool` is this kernel over a batch: block (key head `kh`,
// batch row `bi`) runs the identical body for a single row (one step) on that
// batch row's inputs (`mixed`, `bproj`, `aproj`, `z`, `gated` are `[b, ...]`) and
// on pool row `rows[bi]` of `hist` and `state`, which it updates in place. No
// copy of the state is staged in or out, and nothing is shared between batch
// rows (the pool rows of a batch are distinct), so each batch row's result is
// the single-sequence kernel's, bit for bit. params index 6 is `b` there, not a
// step count: the batch rows are different sequences, never consecutive steps.
//
//   params : u32 [nkh, nvh, group, l2_eps, rms_eps, q_scale, b, 0]
//   mixed  : [b, conv_dim]    bproj, aproj : [b, nvh]    z, gated : [b, nvh * 128]
//   hist   : [pool, conv_dim, 3]    state : [pool, nvh, 128, 128]  (pools, RW)
//   rows   : [b] u32                the pool row of each batch row
//
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_GDN_HD 128   // key and value head dimension
#define BRAIN_GDN_KW 4     // conv taps
// Value heads per key head the block shape serves: 3 x 128 threads at 128 live
// state registers each is what one SM's register file holds.
#define BRAIN_GDN_MAX_GROUP 3
// Rows one launch serves: the per-row staging below is sized for it.
#define BRAIN_GDN_MAX_ROWS 8

__device__ __forceinline__ float brain_silu(float v) {
    return __fdiv_rn(v, __fadd_rn(1.0f, expf(-v)));
}

// Sum of squares of 128 values, one thread's ascending walk - the reference's
// own order (its kernels are one thread per row) - four values per shared load.
__device__ __forceinline__ float brain_sum_sq(const float* v) {
    const float4* v4 = reinterpret_cast<const float4*>(v);
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
        const float4 x = v4[i];
        ss = __fadd_rn(ss, __fmul_rn(x.x, x.x));
        ss = __fadd_rn(ss, __fmul_rn(x.y, x.y));
        ss = __fadd_rn(ss, __fmul_rn(x.z, x.z));
        ss = __fadd_rn(ss, __fmul_rn(x.w, x.w));
    }
    return ss;
}

// One causal conv output: the four taps in order, then SiLU. `x` is the newest
// input; the window `h` slides by one.
__device__ __forceinline__ float brain_conv_silu(float (&h)[3], const float (&w)[4], float x) {
    float acc = 0.0f;
    acc = __fadd_rn(acc, __fmul_rn(h[0], w[0]));
    acc = __fadd_rn(acc, __fmul_rn(h[1], w[1]));
    acc = __fadd_rn(acc, __fmul_rn(h[2], w[2]));
    acc = __fadd_rn(acc, __fmul_rn(x, w[3]));
    h[0] = h[1];
    h[1] = h[2];
    h[2] = x;
    return brain_silu(acc);
}

// `rows` consecutive steps of one sequence for key head `kh`: every pointer is
// already the sequence's own (its tokens' inputs, its window and its state).
__device__ __forceinline__ void brain_gdn_decode_body(const unsigned int* params, unsigned int kh, unsigned int rows,
                                                      const float* mixed, const float* conv_w, float* hist,
                                                      const float* bproj, const float* aproj, const float* a_log,
                                                      const float* dt_bias, float* state, const float* z,
                                                      const float* norm_w, float* gated) {
    const unsigned int nkh = params[0];
    const unsigned int group = params[2];
    const float l2_eps = __uint_as_float(params[3]);
    const float rms_eps = __uint_as_float(params[4]);
    const float q_scale = __uint_as_float(params[5]);

    const unsigned int t = threadIdx.x;
    const unsigned int slot = t / BRAIN_GDN_HD;       // which of this key head's value heads
    const unsigned int j = t % BRAIN_GDN_HD;          // column of that head's state
    const unsigned int key_dim = nkh * BRAIN_GDN_HD;
    const unsigned int vh = kh * group + slot;        // this thread's value head
    const unsigned int nvh_total = nkh * group;
    const unsigned int conv_dim = 2u * key_dim + nvh_total * BRAIN_GDN_HD;
    const unsigned int value_dim = nvh_total * BRAIN_GDN_HD;
    const bool live = slot < group;

    // 16-byte aligned: the serial walks read them four floats at a time.
    //   qk[0][r], qk[1][r] : row r's q and k, conv output then L2-normalised in place
    //   vo[r][s]           : row r's v of value head s, then (in place) the
    //                        recurrence's output before the gated norm
    __shared__ __align__(16) float qk[2][BRAIN_GDN_MAX_ROWS][BRAIN_GDN_HD];
    __shared__ __align__(16) float vo[BRAIN_GDN_MAX_ROWS][BRAIN_GDN_MAX_GROUP][BRAIN_GDN_HD];
    __shared__ float beta_s[BRAIN_GDN_MAX_ROWS][BRAIN_GDN_MAX_GROUP];
    __shared__ float decay_s[BRAIN_GDN_MAX_ROWS][BRAIN_GDN_MAX_GROUP];
    __shared__ float inv_rms[BRAIN_GDN_MAX_ROWS][BRAIN_GDN_MAX_GROUP];

    // The state column: loaded once, kept in registers across the rows.
    float s[BRAIN_GDN_HD];
    float* state_col = state + ((unsigned long long)vh * BRAIN_GDN_HD) * BRAIN_GDN_HD + j;
#pragma unroll
    for (int i = 0; i < BRAIN_GDN_HD; ++i) { s[i] = live ? state_col[(unsigned long long)i * BRAIN_GDN_HD] : 0.0f; }

    // Phase A. The channels this block owns: q and k of key head `kh` and v of
    // each of its value heads. A channel's window is read and rewritten by its
    // owner alone.
    const unsigned int q_ch = kh * BRAIN_GDN_HD + j;
    const unsigned int k_ch = key_dim + kh * BRAIN_GDN_HD + j;
    const unsigned int v_ch = 2u * key_dim + vh * BRAIN_GDN_HD + j;
    // q: the threads of slot 0, k: those of slot 1 (slot 0 again when the key
    // head has a single value head); v: every live thread.
    const bool do_q = (slot == 0);
    const bool do_k = (group > 1) ? (slot == 1) : (slot == 0);

    float h0[3], h1[3], h2[3], w0[4], w1[4], w2[4];
    if (do_q) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h0[i] = hist[(unsigned long long)q_ch * 3 + i]; }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w0[i] = conv_w[(unsigned long long)q_ch * 4 + i]; }
    }
    if (do_k) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h1[i] = hist[(unsigned long long)k_ch * 3 + i]; }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w1[i] = conv_w[(unsigned long long)k_ch * 4 + i]; }
    }
    if (live) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h2[i] = hist[(unsigned long long)v_ch * 3 + i]; }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w2[i] = conv_w[(unsigned long long)v_ch * 4 + i]; }
    }
#pragma unroll
    for (int r = 0; r < BRAIN_GDN_MAX_ROWS; ++r) {
        if (static_cast<unsigned int>(r) < rows) {
            const float* mixed_r = mixed + (unsigned long long)r * conv_dim;
            if (do_q) { qk[0][r][j] = brain_conv_silu(h0, w0, mixed_r[q_ch]); }
            if (do_k) { qk[1][r][j] = brain_conv_silu(h1, w1, mixed_r[k_ch]); }
            if (live) { vo[r][slot][j] = brain_conv_silu(h2, w2, mixed_r[v_ch]); }
        }
    }
    // The windows are final: write them back now so the registers are free.
    if (do_q) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { hist[(unsigned long long)q_ch * 3 + i] = h0[i]; }
    }
    if (do_k) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { hist[(unsigned long long)k_ch * 3 + i] = h1[i]; }
    }
    if (live) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { hist[(unsigned long long)v_ch * 3 + i] = h2[i]; }
    }
    // The gates of every (row, value head): beta = sigmoid(b), decay = exp(g).
    if (t < rows * group) {
        const unsigned int r = t / group;
        const unsigned int sl = t % group;
        const unsigned int head = kh * group + sl;
        const float b_in = bproj[(unsigned long long)r * nvh_total + head];
        const float a_in = aproj[(unsigned long long)r * nvh_total + head];
        const float al = a_log[head];
        const float dtb = dt_bias[head];
        beta_s[r][sl] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-b_in)));
        const float xg = __fadd_rn(a_in, dtb);
        const float softplus = __fadd_rn(fmaxf(xg, 0.0f), logf(__fadd_rn(1.0f, expf(-fabsf(xg)))));
        const float g = __fmul_rn(-expf(al), softplus);
        decay_s[r][sl] = expf(g);
    }
    __syncthreads();

    // Phase B. L2-normalise every row's q and k, in place. The sum of squares
    // is one thread's ascending walk, as in the reference: a thread per vector.
    if (t < 2u * rows) {
        float* vec = qk[t & 1u][t >> 1];
        const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(brain_sum_sq(vec), l2_eps)));
        float4* v4 = reinterpret_cast<float4*>(vec);
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
            const float4 x = v4[i];
            v4[i] = make_float4(__fmul_rn(x.x, r), __fmul_rn(x.y, r), __fmul_rn(x.z, r), __fmul_rn(x.w, r));
        }
    }
    __syncthreads();

    // Phase C. The delta-rule recurrence, row by row. Nothing here is shared
    // between threads but the read-only q and k, so no barrier is needed
    // between rows.
    if (live) {
        for (unsigned int r = 0; r < rows; ++r) {
            const float4* qn4 = reinterpret_cast<const float4*>(qk[0][r]);
            const float4* kn4 = reinterpret_cast<const float4*>(qk[1][r]);
            const float decay = decay_s[r][slot];
#pragma unroll
            for (int i = 0; i < BRAIN_GDN_HD; ++i) { s[i] = __fmul_rn(s[i], decay); }
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
                const float4 k4 = kn4[i];
                kv = __fadd_rn(kv, __fmul_rn(k4.x, s[4 * i + 0]));
                kv = __fadd_rn(kv, __fmul_rn(k4.y, s[4 * i + 1]));
                kv = __fadd_rn(kv, __fmul_rn(k4.z, s[4 * i + 2]));
                kv = __fadd_rn(kv, __fmul_rn(k4.w, s[4 * i + 3]));
            }
            const float delta = __fmul_rn(beta_s[r][slot], __fsub_rn(vo[r][slot][j], kv));
            // The rank-1 update goes through `0.0f + k * delta`, as the
            // reference's `bmm_acc` computes it, so a negative zero behaves
            // identically too.
#pragma unroll
            for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
                const float4 k4 = kn4[i];
                s[4 * i + 0] = __fadd_rn(s[4 * i + 0], __fadd_rn(0.0f, __fmul_rn(k4.x, delta)));
                s[4 * i + 1] = __fadd_rn(s[4 * i + 1], __fadd_rn(0.0f, __fmul_rn(k4.y, delta)));
                s[4 * i + 2] = __fadd_rn(s[4 * i + 2], __fadd_rn(0.0f, __fmul_rn(k4.z, delta)));
                s[4 * i + 3] = __fadd_rn(s[4 * i + 3], __fadd_rn(0.0f, __fmul_rn(k4.w, delta)));
            }
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
                const float4 q4 = qn4[i];
                o = __fadd_rn(o, __fmul_rn(q4.x, s[4 * i + 0]));
                o = __fadd_rn(o, __fmul_rn(q4.y, s[4 * i + 1]));
                o = __fadd_rn(o, __fmul_rn(q4.z, s[4 * i + 2]));
                o = __fadd_rn(o, __fmul_rn(q4.w, s[4 * i + 3]));
            }
            vo[r][slot][j] = __fmul_rn(q_scale, o);
        }
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { state_col[(unsigned long long)i * BRAIN_GDN_HD] = s[i]; }
    }
    __syncthreads();

    // Phase D. The gated RMSNorm's inverse RMS of every (row, value head): the
    // sum of squares is one thread's ascending walk over the head.
    if (t < rows * group) {
        const unsigned int r = t / group;
        const unsigned int sl = t % group;
        inv_rms[r][sl] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(brain_sum_sq(vo[r][sl]), static_cast<float>(BRAIN_GDN_HD)), rms_eps)));
    }
    __syncthreads();

    // Phase E. The layer's output for every row.
    if (live) {
        const float nw = norm_w[j];
#pragma unroll
        for (int r = 0; r < BRAIN_GDN_MAX_ROWS; ++r) {
            if (static_cast<unsigned int>(r) < rows) {
                const float zin = z[(unsigned long long)r * value_dim + (unsigned long long)vh * BRAIN_GDN_HD + j];
                const float nrm = __fmul_rn(__fmul_rn(nw, vo[r][slot][j]), inv_rms[r][slot]);
                gated[(unsigned long long)r * value_dim + (unsigned long long)vh * BRAIN_GDN_HD + j] = __fmul_rn(nrm, brain_silu(zin));
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_decode(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                 const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                 float* state, const float* z, const float* norm_w, float* gated) {
    const unsigned int kh = blockIdx.y * gridDim.x + blockIdx.x;  // key head
    if (kh >= params[0]) { return; }                               // block-uniform
    brain_gdn_decode_body(params, kh, params[6], mixed, conv_w, hist, bproj, aproj, a_log, dt_bias, state, z, norm_w,
                          gated);
}

// Each batch row is a different sequence with one token: the body runs a single
// step on that row's inputs and on pool row `rows[bi]` of the state and window.
extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_decode_pool(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                      const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                      float* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int b = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / nkh;                // batch row
    const unsigned int kh = blk - bi * nkh;           // key head
    if (bi >= b) { return; }                          // block-uniform
    const unsigned long long conv_dim = 2ull * nkh * BRAIN_GDN_HD + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long row = rows[bi];
    brain_gdn_decode_body(params, kh, 1u, mixed + bi * conv_dim, conv_w, hist + row * conv_dim * (BRAIN_GDN_KW - 1),
                          bproj + static_cast<unsigned long long>(bi) * nvh, aproj + static_cast<unsigned long long>(bi) * nvh,
                          a_log, dt_bias, state + row * nvh * BRAIN_GDN_HD * BRAIN_GDN_HD,
                          z + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD, norm_w,
                          gated + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD);
}
