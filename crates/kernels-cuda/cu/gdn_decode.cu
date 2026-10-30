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
// The verify entry (`brain_gdn_verify_pool`)
// ------------------------------------------
// A speculative verify step scores SEVERAL consecutive tokens of each sequence
// of a batch at once, and some of them will be rejected - after which the
// recurrent state must be the state after the last ACCEPTED token, which the
// in-place update has already moved past. `brain_gdn_verify_pool` is the pooled
// kernel with a row count per sequence (`counts[bi]`, 1..=8) and a snapshot of
// the state and conv window after every row but the last: the block already
// holds the state in registers between rows, so writing it out costs one
// coalesced store of the column per row and nothing is recomputed. The last
// row's state is the one left in the pool, exactly as a plain decode step
// leaves it, so a verify that accepts everything needs no repair.
// `brain_gdn_commit_pool` then copies the snapshot of the last accepted row
// back over the pool row of each sequence that rejected something.
//
//   params : u32 [nkh, nvh, group, l2_eps, rms_eps, q_scale, b, 0]
//   mixed  : [n, conv_dim]    bproj, aproj : [n, nvh]    z, gated : [n, nvh * 128]
//            where n = sum(counts) and sequence bi owns rows offs[bi] .. +counts[bi]
//   rows   : [b] u32   pool row of each sequence
//   counts : [b] u32   rows of each sequence (1..=8)       offs : [b] u32   first row
//   snap_state : [n, nvh, 128, 128]   snap_hist : [n, conv_dim, 3]
//            row offs[bi] + r holds the state / window AFTER row r (r < counts[bi] - 1)
//
//   commit params : u32 [b, state_len, hist_len, 0]
//   sel    : [b] u32   snapshot row to restore for each sequence, or 0xffffffff for none
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

// The recurrent state and the conv window are stored as f32 or as bf16 (an `unsigned short` holding the top half of the
// f32): every arithmetic step is f32, a bf16 element is widened on load and rounded to nearest even on store.
__device__ __forceinline__ float brain_ld_state(const float* p) { return *p; }
__device__ __forceinline__ float brain_ld_state(const unsigned short* p) { return __uint_as_float(static_cast<unsigned int>(*p) << 16); }
__device__ __forceinline__ void brain_st_state(float* p, float v) { *p = v; }
__device__ __forceinline__ void brain_st_state(unsigned short* p, float v) {
    const unsigned int b = __float_as_uint(v);
    *p = static_cast<unsigned short>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

// `rows` consecutive steps of one sequence for key head `kh`: every pointer is
// already the sequence's own (its tokens' inputs, its window and its state).
template <typename S>
__device__ __forceinline__ void brain_gdn_decode_body(const unsigned int* params, unsigned int kh, unsigned int rows,
                                                      const float* mixed, const float* conv_w, S* hist,
                                                      const float* bproj, const float* aproj, const float* a_log,
                                                      const float* dt_bias, S* state, const float* z,
                                                      const float* norm_w, float* gated, float* snap_state,
                                                      float* snap_hist) {
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
    S* state_col = state + ((unsigned long long)vh * BRAIN_GDN_HD) * BRAIN_GDN_HD + j;
#pragma unroll
    for (int i = 0; i < BRAIN_GDN_HD; ++i) { s[i] = live ? brain_ld_state(state_col + (unsigned long long)i * BRAIN_GDN_HD) : 0.0f; }

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
        for (int i = 0; i < 3; ++i) { h0[i] = brain_ld_state(hist + (unsigned long long)q_ch * 3 + i); }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w0[i] = conv_w[(unsigned long long)q_ch * 4 + i]; }
    }
    if (do_k) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h1[i] = brain_ld_state(hist + (unsigned long long)k_ch * 3 + i); }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w1[i] = conv_w[(unsigned long long)k_ch * 4 + i]; }
    }
    if (live) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h2[i] = brain_ld_state(hist + (unsigned long long)v_ch * 3 + i); }
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
            if (snap_hist != nullptr && static_cast<unsigned int>(r) + 1u < rows) {
                float* sh = snap_hist + static_cast<unsigned long long>(r) * conv_dim * (BRAIN_GDN_KW - 1);
                if (do_q) {
#pragma unroll
                    for (int i = 0; i < 3; ++i) { sh[(unsigned long long)q_ch * 3 + i] = h0[i]; }
                }
                if (do_k) {
#pragma unroll
                    for (int i = 0; i < 3; ++i) { sh[(unsigned long long)k_ch * 3 + i] = h1[i]; }
                }
                if (live) {
#pragma unroll
                    for (int i = 0; i < 3; ++i) { sh[(unsigned long long)v_ch * 3 + i] = h2[i]; }
                }
            }
        }
    }
    // The windows are final: write them back now so the registers are free.
    if (do_q) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { brain_st_state(hist + (unsigned long long)q_ch * 3 + i, h0[i]); }
    }
    if (do_k) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { brain_st_state(hist + (unsigned long long)k_ch * 3 + i, h1[i]); }
    }
    if (live) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { brain_st_state(hist + (unsigned long long)v_ch * 3 + i, h2[i]); }
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
            if (snap_state != nullptr && r + 1u < rows) {
                float* snap_col = snap_state + static_cast<unsigned long long>(r) * nvh_total * BRAIN_GDN_HD * BRAIN_GDN_HD +
                                  ((unsigned long long)vh * BRAIN_GDN_HD) * BRAIN_GDN_HD + j;
                // Streaming stores: the snapshot is written once and read only on a rejection. They
                // also keep the 128 live state registers from being spilled around the stores (3140
                // spill bytes per thread with plain stores, 1500 with these).
#pragma unroll
                for (int i = 0; i < BRAIN_GDN_HD; ++i) { __stcs(snap_col + static_cast<unsigned long long>(i) * BRAIN_GDN_HD, s[i]); }
            }
        }
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { brain_st_state(state_col + (unsigned long long)i * BRAIN_GDN_HD, s[i]); }
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

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_decode(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                 const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                 float* state, const float* z, const float* norm_w, float* gated) {
    BRAIN_PDL_ENTRY();
    const unsigned int kh = blockIdx.y * gridDim.x + blockIdx.x;  // key head
    if (kh >= params[0]) { return; }                               // block-uniform
    brain_gdn_decode_body<float>(params, kh, params[6], mixed, conv_w, hist, bproj, aproj, a_log, dt_bias, state, z, norm_w,
                          gated, nullptr, nullptr);
}

// Each batch row is a different sequence with one token: the body runs a single
// step on that row's inputs and on pool row `rows[bi]` of the state and window.
extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_decode_pool(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                      const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                      float* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    BRAIN_PDL_ENTRY();
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int b = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / nkh;                // batch row
    const unsigned int kh = blk - bi * nkh;           // key head
    if (bi >= b) { return; }                          // block-uniform
    const unsigned long long conv_dim = 2ull * nkh * BRAIN_GDN_HD + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long row = rows[bi];
    brain_gdn_decode_body<float>(params, kh, 1u, mixed + bi * conv_dim, conv_w, hist + row * conv_dim * (BRAIN_GDN_KW - 1),
                          bproj + static_cast<unsigned long long>(bi) * nvh, aproj + static_cast<unsigned long long>(bi) * nvh,
                          a_log, dt_bias, state + row * nvh * BRAIN_GDN_HD * BRAIN_GDN_HD,
                          z + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD, norm_w,
                          gated + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD, nullptr, nullptr);
}

// The verify entry: sequence `bi` runs `counts[bi]` consecutive steps from pool
// row `rows[bi]`, its inputs and outputs the rows `offs[bi] ..` of the packed
// token arrays, snapshotting the state and window after every row but the last.
extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_verify_pool(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                      const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                      float* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows,
                      const unsigned int* counts, const unsigned int* offs, float* snap_state, float* snap_hist) {
    BRAIN_PDL_ENTRY();
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int b = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / nkh;                // sequence
    const unsigned int kh = blk - bi * nkh;           // key head
    if (bi >= b) { return; }                          // block-uniform
    const unsigned long long conv_dim = 2ull * nkh * BRAIN_GDN_HD + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long state_len = static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD * BRAIN_GDN_HD;
    const unsigned long long hist_len = conv_dim * (BRAIN_GDN_KW - 1);
    const unsigned long long row = rows[bi];
    const unsigned long long off = offs[bi];
    const unsigned int n = counts[bi];
    brain_gdn_decode_body<float>(params, kh, n, mixed + off * conv_dim, conv_w, hist + row * hist_len,
                          bproj + off * nvh, aproj + off * nvh, a_log, dt_bias, state + row * state_len,
                          z + off * nvh * BRAIN_GDN_HD, norm_w, gated + off * nvh * BRAIN_GDN_HD,
                          snap_state + off * state_len, snap_hist + off * hist_len);
}

// Put the snapshot row `sel[bi]` back over pool row `rows[bi]` for every
// sequence that has one. Block (bi, c) copies the c-th 4096-element tile of the
// state, and of the window for the first tiles that cover it, as 16-byte
// vectors (the lengths are multiples of four floats, which `Fused::serves`
// checks).
#define BRAIN_GDN_COMMIT_TILE 4096
extern "C" __global__ void __launch_bounds__(256)
brain_gdn_commit_pool(const unsigned int* params, float* pool_state, float* pool_hist, const float* snap_state,
                      const float* snap_hist, const unsigned int* rows, const unsigned int* sel) {
    BRAIN_PDL_ENTRY();
    const unsigned int b = params[0];
    const unsigned long long state_len = params[1];
    const unsigned long long hist_len = params[2];
    const unsigned int tiles = static_cast<unsigned int>((state_len + BRAIN_GDN_COMMIT_TILE - 1) / BRAIN_GDN_COMMIT_TILE);
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / tiles;
    const unsigned int tile = blk - bi * tiles;
    if (bi >= b) { return; }                          // block-uniform
    const unsigned int pick = sel[bi];
    if (pick == 0xffffffffu) { return; }              // block-uniform
    const unsigned long long row = rows[bi];
    const unsigned long long lo = static_cast<unsigned long long>(tile) * BRAIN_GDN_COMMIT_TILE;
    {
        const float4* src = reinterpret_cast<const float4*>(snap_state + static_cast<unsigned long long>(pick) * state_len + lo);
        float4* dst = reinterpret_cast<float4*>(pool_state + row * state_len + lo);
        const unsigned long long n4 = (state_len - lo < BRAIN_GDN_COMMIT_TILE ? state_len - lo : BRAIN_GDN_COMMIT_TILE) / 4;
#pragma unroll
        for (unsigned int it = 0; it < BRAIN_GDN_COMMIT_TILE / 4 / 256; ++it) {
            const unsigned int i = threadIdx.x + it * 256u;
            if (i < n4) { dst[i] = src[i]; }
        }
    }
    if (lo < hist_len) {
        const float4* src = reinterpret_cast<const float4*>(snap_hist + static_cast<unsigned long long>(pick) * hist_len + lo);
        float4* dst = reinterpret_cast<float4*>(pool_hist + row * hist_len + lo);
        const unsigned long long n4 = (hist_len - lo < BRAIN_GDN_COMMIT_TILE ? hist_len - lo : BRAIN_GDN_COMMIT_TILE) / 4;
#pragma unroll
        for (unsigned int it = 0; it < BRAIN_GDN_COMMIT_TILE / 4 / 256; ++it) {
            const unsigned int i = threadIdx.x + it * 256u;
            if (i < n4) { dst[i] = src[i]; }
        }
    }
}

// The pooled decode over a pool that stores the state and the conv window as bf16 (two bytes an element): the same body
// with the state widened on load and rounded to nearest even on store. `hist` and `state` are `[pool, ...]` arrays of
// bf16 elements, indexed exactly as the f32 pool is.
extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_decode_pool_bf16(const unsigned int* params, const float* mixed, const float* conv_w, unsigned short* hist,
                           const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                           unsigned short* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    BRAIN_PDL_ENTRY();
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int b = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / nkh;                // batch row
    const unsigned int kh = blk - bi * nkh;           // key head
    if (bi >= b) { return; }                          // block-uniform
    const unsigned long long conv_dim = 2ull * nkh * BRAIN_GDN_HD + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long row = rows[bi];
    brain_gdn_decode_body<unsigned short>(params, kh, 1u, mixed + bi * conv_dim, conv_w, hist + row * conv_dim * (BRAIN_GDN_KW - 1),
                                          bproj + static_cast<unsigned long long>(bi) * nvh, aproj + static_cast<unsigned long long>(bi) * nvh,
                                          a_log, dt_bias, state + row * nvh * BRAIN_GDN_HD * BRAIN_GDN_HD,
                                          z + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD, norm_w,
                                          gated + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD, nullptr, nullptr);
}

// ---- the wide pooled decode ------------------------------------------------------------------------------------------
//
// `brain_gdn_decode_pool_wide` (and `_wide_bf16`) is the pooled decode step for one token per sequence built for
// occupancy rather than for bit identity with the reference chain. The state column a thread holds is half as long: a
// pair of adjacent lanes owns one column of the 128 x 128 state, thread (j, half) the 64 rows `half * 64 ..`, so each
// thread keeps 64 state registers instead of 128, walks two 64-term dependent chains instead of two 128-term ones, and a
// block of `group` x 256 threads is twice as many warps as the 128-thread-per-column layout. The two contractions
// (`k . s` and `q . s` over the key index) are folded across the pair with one shuffle; the sums of squares of the gated
// norm are a warp-shuffle tree instead of one thread's 128-term walk. Products and sums use fused multiply-adds.
//
// Numerics: against the exact body above the results differ by f32 rounding of reassociated sums (the summation order
// is: each half ascending over its 64 rows, then half 0 + half 1), which the gate bounds (`tests/gdn_decode_wide.rs`).
// Everything outside those sums is the same code: the conv taps and SiLU, the gates, the L2 normalisation of q and k.
//
// Same params, bindings and block-per-(key head, batch row) mapping as `brain_gdn_decode_pool`; `group` is at most 2
// here (a block is `group * 256` threads, 512 at most), which is the model's own 32 value heads over 16 key heads.

#define BRAIN_GDN_WIDE_MAX_GROUP 2
#define BRAIN_GDN_WIDE_THREADS (BRAIN_GDN_WIDE_MAX_GROUP * 256)

// One channel's causal conv step: its window and taps loaded, the SiLU of the four-tap sum returned, the window slid.
template <typename S>
__device__ __forceinline__ float brain_conv_channel(S* hist, const float* conv_w, const float* mixed, unsigned int ch) {
    float h[3], w[4];
#pragma unroll
    for (int i = 0; i < 3; ++i) { h[i] = brain_ld_state(hist + static_cast<unsigned long long>(ch) * 3 + i); }
#pragma unroll
    for (int i = 0; i < 4; ++i) { w[i] = conv_w[static_cast<unsigned long long>(ch) * 4 + i]; }
    const float out = brain_conv_silu(h, w, mixed[ch]);
#pragma unroll
    for (int i = 0; i < 3; ++i) { brain_st_state(hist + static_cast<unsigned long long>(ch) * 3 + i, h[i]); }
    return out;
}

template <typename S>
__device__ __forceinline__ void brain_gdn_decode_wide_body(const unsigned int* params, unsigned int kh, const float* mixed,
                                                           const float* conv_w, S* hist, const float* bproj, const float* aproj,
                                                           const float* a_log, const float* dt_bias, S* state, const float* z,
                                                           const float* norm_w, float* gated) {
    const unsigned int nkh = params[0];
    const unsigned int group = params[2];
    const float l2_eps = __uint_as_float(params[3]);
    const float rms_eps = __uint_as_float(params[4]);
    const float q_scale = __uint_as_float(params[5]);

    const unsigned int t = threadIdx.x;
    const unsigned int slot = t >> 8;                 // which of this key head's value heads
    const unsigned int tt = t & 255u;
    const unsigned int j = tt >> 1;                   // the state column this thread's pair owns
    const unsigned int half = tt & 1u;                // which 64 rows of it
    const unsigned int key_dim = nkh * BRAIN_GDN_HD;
    const unsigned int vh = kh * group + slot;
    const unsigned int nvh_total = nkh * group;
    const unsigned int value_dim = nvh_total * BRAIN_GDN_HD;
    const bool live = slot < group;

    __shared__ __align__(16) float qk[2][BRAIN_GDN_HD];
    __shared__ __align__(16) float vo[BRAIN_GDN_WIDE_MAX_GROUP][BRAIN_GDN_HD];
    __shared__ float beta_s[BRAIN_GDN_WIDE_MAX_GROUP];
    __shared__ float decay_s[BRAIN_GDN_WIDE_MAX_GROUP];
    __shared__ float sq_part[BRAIN_GDN_WIDE_MAX_GROUP][8];

    // The state half-column: loaded once, kept in registers.
    float s[64];
    S* state_col = state + (static_cast<unsigned long long>(vh) * BRAIN_GDN_HD + half * 64u) * BRAIN_GDN_HD + j;
    if (live) {
#pragma unroll
        for (int i = 0; i < 64; ++i) { s[i] = brain_ld_state(state_col + static_cast<unsigned long long>(i) * BRAIN_GDN_HD); }
    }

    // Phase A. The conv windows of the channels this block owns, each rewritten by one thread: q and k of the key head
    // (threads 0..255 of slot 0 and, with two value heads, the first 128 of slot 1 for k), v of each value head.
    const bool do_q = (slot == 0) && (tt < BRAIN_GDN_HD);
    const bool do_k = (group > 1) ? (slot == 1 && tt < BRAIN_GDN_HD) : (slot == 0 && tt >= BRAIN_GDN_HD);
    const unsigned int qk_ch = do_q ? tt : (tt & (BRAIN_GDN_HD - 1u));
    const unsigned int q_ch = kh * BRAIN_GDN_HD + qk_ch;
    const unsigned int k_ch = key_dim + kh * BRAIN_GDN_HD + qk_ch;
    const unsigned int v_ch = 2u * key_dim + vh * BRAIN_GDN_HD + j;
    const bool do_v = live && half == 0u;
    if (do_q || do_k) {
        const float out = brain_conv_channel(hist, conv_w, mixed, do_q ? q_ch : k_ch);
        if (do_q) { qk[0][qk_ch] = out; } else { qk[1][qk_ch] = out; }
    }
    if (do_v) { vo[slot][j] = brain_conv_channel(hist, conv_w, mixed, v_ch); }
    if (t < group) {
        const unsigned int head = kh * group + t;
        const float b_in = bproj[head];
        const float a_in = aproj[head];
        beta_s[t] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-b_in)));
        const float xg = __fadd_rn(a_in, dt_bias[head]);
        const float softplus = __fadd_rn(fmaxf(xg, 0.0f), logf(__fadd_rn(1.0f, expf(-fabsf(xg)))));
        decay_s[t] = expf(__fmul_rn(-expf(a_log[head]), softplus));
    }
    __syncthreads();

    // Phase B. L2-normalise q and k (one thread each, the reference's own walk).
    if (t < 2u) {
        float* vec = qk[t];
        const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(brain_sum_sq(vec), l2_eps)));
        float4* v4 = reinterpret_cast<float4*>(vec);
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
            const float4 x = v4[i];
            v4[i] = make_float4(__fmul_rn(x.x, r), __fmul_rn(x.y, r), __fmul_rn(x.z, r), __fmul_rn(x.w, r));
        }
    }
    __syncthreads();

    // Phase C. The delta rule on this thread's 64 rows of column j, the two contractions folded across the pair.
    float o = 0.0f;
    if (live) {
        const float4* qn4 = reinterpret_cast<const float4*>(qk[0] + half * 64u);
        const float4* kn4 = reinterpret_cast<const float4*>(qk[1] + half * 64u);
        const float decay = decay_s[slot];
#pragma unroll
        for (int i = 0; i < 64; ++i) { s[i] = __fmul_rn(s[i], decay); }
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float4 k4 = kn4[i];
            kv = fmaf(k4.x, s[4 * i + 0], kv);
            kv = fmaf(k4.y, s[4 * i + 1], kv);
            kv = fmaf(k4.z, s[4 * i + 2], kv);
            kv = fmaf(k4.w, s[4 * i + 3], kv);
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        const float delta = __fmul_rn(beta_s[slot], __fsub_rn(vo[slot][j], kv));
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float4 k4 = kn4[i];
            s[4 * i + 0] = fmaf(k4.x, delta, s[4 * i + 0]);
            s[4 * i + 1] = fmaf(k4.y, delta, s[4 * i + 1]);
            s[4 * i + 2] = fmaf(k4.z, delta, s[4 * i + 2]);
            s[4 * i + 3] = fmaf(k4.w, delta, s[4 * i + 3]);
        }
        float op = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float4 q4 = qn4[i];
            op = fmaf(q4.x, s[4 * i + 0], op);
            op = fmaf(q4.y, s[4 * i + 1], op);
            op = fmaf(q4.z, s[4 * i + 2], op);
            op = fmaf(q4.w, s[4 * i + 3], op);
        }
        op += __shfl_xor_sync(0xffffffffu, op, 1);
        o = __fmul_rn(q_scale, op);
#pragma unroll
        for (int i = 0; i < 64; ++i) { brain_st_state(state_col + static_cast<unsigned long long>(i) * BRAIN_GDN_HD, s[i]); }
    }
    // Phase D. The gated RMSNorm's inverse RMS of each value head: the sum of squares of its 128 outputs, a warp tree.
    // The half-0 thread of each pair holds its column's output; the other lane of the pair holds the same value.
    float sq = (live && half == 0u) ? __fmul_rn(o, o) : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { sq += __shfl_xor_sync(0xffffffffu, sq, off); }
    if (live && (t & 31u) == 0u) { sq_part[slot][tt >> 5] = sq; }
    if (live && half == 0u) { vo[slot][j] = o; }
    __syncthreads();
    if (live && half == 0u) {
        float total = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { total += sq_part[slot][i]; }
        const float inv_rms = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(BRAIN_GDN_HD)), rms_eps)));
        const float zin = z[static_cast<unsigned long long>(vh) * BRAIN_GDN_HD + j];
        const float nrm = __fmul_rn(__fmul_rn(norm_w[j], o), inv_rms);
        gated[static_cast<unsigned long long>(vh) * BRAIN_GDN_HD + j] = __fmul_rn(nrm, brain_silu(zin));
    }
}

template <typename S>
__device__ __forceinline__ void brain_gdn_decode_pool_wide_entry(const unsigned int* params, const float* mixed, const float* conv_w, S* hist,
                                                                 const float* bproj, const float* aproj, const float* a_log,
                                                                 const float* dt_bias, S* state, const float* z, const float* norm_w,
                                                                 float* gated, const unsigned int* rows) {
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int b = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / nkh;
    const unsigned int kh = blk - bi * nkh;
    if (bi >= b) { return; }                          // block-uniform
    const unsigned long long conv_dim = 2ull * nkh * BRAIN_GDN_HD + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long row = rows[bi];
    brain_gdn_decode_wide_body<S>(params, kh, mixed + bi * conv_dim, conv_w, hist + row * conv_dim * (BRAIN_GDN_KW - 1),
                                  bproj + static_cast<unsigned long long>(bi) * nvh, aproj + static_cast<unsigned long long>(bi) * nvh,
                                  a_log, dt_bias, state + row * nvh * BRAIN_GDN_HD * BRAIN_GDN_HD,
                                  z + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD, norm_w,
                                  gated + static_cast<unsigned long long>(bi) * nvh * BRAIN_GDN_HD);
}

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_WIDE_THREADS, 1)
brain_gdn_decode_pool_wide(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                           const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                           float* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    BRAIN_PDL_ENTRY();
    brain_gdn_decode_pool_wide_entry<float>(params, mixed, conv_w, hist, bproj, aproj, a_log, dt_bias, state, z, norm_w, gated, rows);
}

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_WIDE_THREADS, 1)
brain_gdn_decode_pool_wide_bf16(const unsigned int* params, const float* mixed, const float* conv_w, unsigned short* hist,
                                const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                                unsigned short* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    BRAIN_PDL_ENTRY();
    brain_gdn_decode_pool_wide_entry<unsigned short>(params, mixed, conv_w, hist, bproj, aproj, a_log, dt_bias, state, z, norm_w, gated, rows);
}

// The split pooled decode (knowledge 386)
// ----------------------------------------
// `brain_gdn_decode_pool` is one block of 384 threads per (key head, batch row): 16 blocks at one sequence, so 16 of 132 SMs
// stream the 3 MB of recurrent state, and 3 value heads in lockstep (load all, compute all, store all) leave the memory idle
// while the one resident block computes: 17 us at one sequence, 72 us at 16, 270 us at 64 (the state traffic is 2 us, 28 us and
// 110 us at the memory roof). The reason one block owns a whole key head is the conv window of its q and k channels, shared by
// its value heads and rewritten in place. Take the conv out and the rest has no sharing at all:
//
//   brain_gdn_conv_pool   one thread per (batch row, channel): the causal conv + SiLU of q, k and v, the window slid in place by
//                         the channel's only owner, the activated channels written to `conv_out` [b, conv_dim]
//   brain_gdn_recur_pool  one block of 128 threads per (batch row, value head): the L2 norms of its key head's q and k (each
//                         block recomputes them: one thread's ascending walk of 128 values, the reference's own order), the gates,
//                         the delta-rule recurrence of its 128 state columns, the gated norm
//
// Every expression is the fused body's, in its order, so the output and the state and window pools are bit-identical to it
// (and to the chain it replaces); the blocks are independent, three or more resident an SM, so one block's state load overlaps
// another's recurrence. The state loads do not depend on the kernel before, so they are issued before `griddepcontrol.wait`.

template <typename S>
__device__ __forceinline__ void brain_gdn_conv_pool_entry(const unsigned int* params, const float* mixed, const float* conv_w, S* hist,
                                                          float* conv_out, const unsigned int* rows) {
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int b = params[2];
    const unsigned long long conv_dim = 2ull * nkh * BRAIN_GDN_HD + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long idx = static_cast<unsigned long long>(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= conv_dim * b) { return; }
    const unsigned long long bi = idx / conv_dim;
    const unsigned int ch = static_cast<unsigned int>(idx - bi * conv_dim);
    const unsigned long long row = rows[bi];
    conv_out[idx] = brain_conv_channel(hist + row * conv_dim * (BRAIN_GDN_KW - 1), conv_w, mixed + bi * conv_dim, ch);
}

extern "C" __global__ void __launch_bounds__(256)
brain_gdn_conv_pool(const unsigned int* params, const float* mixed, const float* conv_w, float* hist, float* conv_out, const unsigned int* rows) {
    BRAIN_PDL_ENTRY();
    brain_gdn_conv_pool_entry<float>(params, mixed, conv_w, hist, conv_out, rows);
}

extern "C" __global__ void __launch_bounds__(256)
brain_gdn_conv_pool_bf16(const unsigned int* params, const float* mixed, const float* conv_w, unsigned short* hist, float* conv_out, const unsigned int* rows) {
    BRAIN_PDL_ENTRY();
    brain_gdn_conv_pool_entry<unsigned short>(params, mixed, conv_w, hist, conv_out, rows);
}

// 16-byte global -> shared copies (`cp.async`) of the state tile, and the shared -> global store of the result.
__device__ __forceinline__ void brain_gdn_cp16(unsigned int dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}

template <typename S>
__device__ __forceinline__ void brain_gdn_recur_pool_entry(const unsigned int* params, const float* conv_out, const float* bproj, const float* aproj,
                                                           const float* a_log, const float* dt_bias, S* state, const float* z, const float* norm_w,
                                                           float* gated, const unsigned int* rows) {
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int group = params[2];
    const float l2_eps = __uint_as_float(params[3]);
    const float rms_eps = __uint_as_float(params[4]);
    const float q_scale = __uint_as_float(params[5]);
    const unsigned int b = params[6];
    const unsigned int blk = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int bi = blk / nvh;                // batch row
    const unsigned int vh = blk - bi * nvh;           // value head
    if (bi >= b) { return; }                          // block-uniform
    const unsigned int kh = vh / group;
    const unsigned int j = threadIdx.x;               // column of the head's state
    const unsigned long long key_dim = static_cast<unsigned long long>(nkh) * BRAIN_GDN_HD;
    const unsigned long long conv_dim = 2ull * key_dim + static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;
    const unsigned long long value_dim = static_cast<unsigned long long>(nvh) * BRAIN_GDN_HD;

    extern __shared__ __align__(16) unsigned char brain_gdn_tile_base[];   // the head's [128][128] state, as stored (f32 or bf16)
    S* tile = reinterpret_cast<S*>(brain_gdn_tile_base);
    __shared__ __align__(16) float qk[2][BRAIN_GDN_HD];
    __shared__ __align__(16) float vo[BRAIN_GDN_HD];
    __shared__ float gate_s[2];   // beta, decay
    __shared__ float inv_rms;

    // The whole state tile is requested at once, as 16-byte copies that cost no registers: it is nothing the previous kernel
    // wrote, so the copies are in flight while that kernel finishes.
    constexpr unsigned int TILE_BYTES = BRAIN_GDN_HD * BRAIN_GDN_HD * sizeof(S);
    const unsigned long long row = rows[bi];
    S* gtile = state + (row * nvh + vh) * BRAIN_GDN_HD * BRAIN_GDN_HD;
    const unsigned int tile_smem = static_cast<unsigned int>(__cvta_generic_to_shared(tile));
    for (unsigned int c = j; c < TILE_BYTES / 16u; c += BRAIN_GDN_HD) { brain_gdn_cp16(tile_smem + 16u * c, reinterpret_cast<const unsigned char*>(gtile) + 16ull * c); }
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    // The conv outputs come from the kernel before.
    BRAIN_PDL_ENTRY();
    const float* co = conv_out + bi * conv_dim;
    qk[0][j] = co[kh * BRAIN_GDN_HD + j];
    qk[1][j] = co[key_dim + kh * BRAIN_GDN_HD + j];
    const float v_in = co[2ull * key_dim + vh * BRAIN_GDN_HD + j];
    if (j == 0) {
        const float b_in = bproj[bi * nvh + vh];
        const float a_in = aproj[bi * nvh + vh];
        gate_s[0] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-b_in)));
        const float xg = __fadd_rn(a_in, dt_bias[vh]);
        const float softplus = __fadd_rn(fmaxf(xg, 0.0f), logf(__fadd_rn(1.0f, expf(-fabsf(xg)))));
        gate_s[1] = expf(__fmul_rn(-expf(a_log[vh]), softplus));
    }
    __syncthreads();

    // L2-normalise q and k: one thread each, the reference's own ascending walk.
    if (j < 2u) {
        const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(brain_sum_sq(qk[j]), l2_eps)));
        float4* v4 = reinterpret_cast<float4*>(qk[j]);
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD / 4; ++i) {
            const float4 x = v4[i];
            v4[i] = make_float4(__fmul_rn(x.x, r), __fmul_rn(x.y, r), __fmul_rn(x.z, r), __fmul_rn(x.w, r));
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();   // q and k are normalised and every thread's copies have landed

    // The delta rule on column j (the fused body's Phase C for one row), two passes over the column in shared memory: the
    // decayed value is recomputed in the second pass (the same product), so no 128-register copy of the column is live.
    const float4* qn4 = reinterpret_cast<const float4*>(qk[0]);
    const float4* kn4 = reinterpret_cast<const float4*>(qk[1]);
    const float decay = gate_s[1];
    S* col = tile + j;
    float kv = 0.0f;
#pragma unroll 2
    for (int i4 = 0; i4 < BRAIN_GDN_HD / 4; ++i4) {
        const float4 k4 = kn4[i4];
        kv = __fadd_rn(kv, __fmul_rn(k4.x, __fmul_rn(brain_ld_state(col + (4 * i4 + 0) * BRAIN_GDN_HD), decay)));
        kv = __fadd_rn(kv, __fmul_rn(k4.y, __fmul_rn(brain_ld_state(col + (4 * i4 + 1) * BRAIN_GDN_HD), decay)));
        kv = __fadd_rn(kv, __fmul_rn(k4.z, __fmul_rn(brain_ld_state(col + (4 * i4 + 2) * BRAIN_GDN_HD), decay)));
        kv = __fadd_rn(kv, __fmul_rn(k4.w, __fmul_rn(brain_ld_state(col + (4 * i4 + 3) * BRAIN_GDN_HD), decay)));
    }
    const float delta = __fmul_rn(gate_s[0], __fsub_rn(v_in, kv));
    // The rank-1 update goes through `0.0f + k * delta`, as the reference's `bmm_acc` computes it (a negative zero behaves identically).
    float o = 0.0f;
#pragma unroll 2
    for (int i4 = 0; i4 < BRAIN_GDN_HD / 4; ++i4) {
        const float4 k4 = kn4[i4];
        const float4 q4 = qn4[i4];
        float sn[4];
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            S* p = col + (4 * i4 + u) * BRAIN_GDN_HD;
            const float sd = __fmul_rn(brain_ld_state(p), decay);
            sn[u] = __fadd_rn(sd, __fadd_rn(0.0f, __fmul_rn(kk[u], delta)));
            o = __fadd_rn(o, __fmul_rn(qq[u], sn[u]));
            brain_st_state(p, sn[u]);
        }
    }
    vo[j] = __fmul_rn(q_scale, o);
    __syncthreads();   // every column is final: the tile goes back as 16-byte vectors, and the norm reads the head

    const uint4* src = reinterpret_cast<const uint4*>(tile);
    uint4* dst = reinterpret_cast<uint4*>(gtile);
    for (unsigned int c = j; c < TILE_BYTES / 16u; c += BRAIN_GDN_HD) { dst[c] = src[c]; }

    // The gated RMSNorm: the inverse RMS of the head is one thread's ascending walk.
    if (j == 0) { inv_rms = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(brain_sum_sq(vo), static_cast<float>(BRAIN_GDN_HD)), rms_eps))); }
    const float nw = norm_w[j];
    const float zin = z[bi * value_dim + static_cast<unsigned long long>(vh) * BRAIN_GDN_HD + j];
    __syncthreads();
    const float nrm = __fmul_rn(__fmul_rn(nw, vo[j]), inv_rms);
    gated[bi * value_dim + static_cast<unsigned long long>(vh) * BRAIN_GDN_HD + j] = __fmul_rn(nrm, brain_silu(zin));
}

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_HD, 3)
brain_gdn_recur_pool(const unsigned int* params, const float* conv_out, const float* bproj, const float* aproj, const float* a_log,
                     const float* dt_bias, float* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    brain_gdn_recur_pool_entry<float>(params, conv_out, bproj, aproj, a_log, dt_bias, state, z, norm_w, gated, rows);
}

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_HD, 3)
brain_gdn_recur_pool_bf16(const unsigned int* params, const float* conv_out, const float* bproj, const float* aproj, const float* a_log,
                          const float* dt_bias, unsigned short* state, const float* z, const float* norm_w, float* gated, const unsigned int* rows) {
    brain_gdn_recur_pool_entry<unsigned short>(params, conv_out, bproj, aproj, a_log, dt_bias, state, z, norm_w, gated, rows);
}
