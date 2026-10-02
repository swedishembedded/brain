// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on
// GPUs for its clients. If your team needs expertise in collapsing the chain of
// tiny kernels that makes up a recurrent layer's token step into one launch,
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// One Gated DeltaNet decode step, single sequence, in a single launch:
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
//                 q_scale (f32 bits, 1/sqrt(dk)), 0, 0]      nvh = nkh * group
//   mixed  : [conv_dim] f32      in_proj_qkv output, q | k | v, one token
//   conv_w : [conv_dim, 4] f32   depthwise filter taps
//   hist   : [conv_dim, 3] f32   the conv window (RW; shifted by this step)
//   bproj  : [nvh] f32, aproj : [nvh] f32, a_log : [nvh] f32, dt_bias : [nvh] f32
//   state  : [nvh, 128, 128] f32 recurrent state (RW; updated in place)
//   z      : [nvh * 128] f32     the output gate's pre-activation
//   norm_w : [128] f32           gated-norm gain
//   gated  : [nvh * 128] f32     the layer's output before out_proj
//   conv_dim = 2 * nkh * 128 + nvh * 128; key and value head dims are both 128.
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
// No `__restrict__` anywhere, deliberately: brain's device buffers alias by
// design (a sliced step binds ranges of one allocation).

#define BRAIN_GDN_HD 128   // key and value head dimension
#define BRAIN_GDN_KW 4     // conv taps
// Value heads per key head the block shape serves: 3 x 128 threads at 128 live
// state registers each is what one SM's register file holds.
#define BRAIN_GDN_MAX_GROUP 3

__device__ __forceinline__ float brain_silu(float v) {
    return __fdiv_rn(v, __fadd_rn(1.0f, expf(-v)));
}

extern "C" __global__ void __launch_bounds__(BRAIN_GDN_MAX_GROUP * BRAIN_GDN_HD, 1)
brain_gdn_decode(const unsigned int* params, const float* mixed, const float* conv_w, float* hist,
                 const float* bproj, const float* aproj, const float* a_log, const float* dt_bias,
                 float* state, const float* z, const float* norm_w, float* gated) {
    const unsigned int nkh = params[0];
    const unsigned int nvh = params[1];
    const unsigned int group = params[2];
    const float l2_eps = __uint_as_float(params[3]);
    const float rms_eps = __uint_as_float(params[4]);
    const float q_scale = __uint_as_float(params[5]);

    const unsigned int kh = blockIdx.y * gridDim.x + blockIdx.x;  // key head
    if (kh >= nkh) { return; }                                     // block-uniform
    const unsigned int t = threadIdx.x;
    const unsigned int slot = t / BRAIN_GDN_HD;       // which of this key head's value heads
    const unsigned int j = t % BRAIN_GDN_HD;          // column of that head's state
    const unsigned int key_dim = nkh * BRAIN_GDN_HD;
    const unsigned int vh = kh * group + slot;        // this thread's value head

    __shared__ float qn[BRAIN_GDN_HD];
    __shared__ float kn[BRAIN_GDN_HD];
    __shared__ float conv_q[BRAIN_GDN_HD];
    __shared__ float conv_k[BRAIN_GDN_HD];
    __shared__ float normed[BRAIN_GDN_MAX_GROUP][BRAIN_GDN_HD];
    __shared__ float inv_rms[BRAIN_GDN_MAX_GROUP];

    // Phase 1: every global load of the step, issued before any is consumed.
    // The conv window and taps of this block's channels, the current token's
    // pre-conv values, and this thread's whole state column.
    float s[BRAIN_GDN_HD];
    float* state_col = state + ((unsigned long long)vh * BRAIN_GDN_HD) * BRAIN_GDN_HD + j;
    const bool live = slot < group;
#pragma unroll
    for (int i = 0; i < BRAIN_GDN_HD; ++i) { s[i] = live ? state_col[(unsigned long long)i * BRAIN_GDN_HD] : 0.0f; }

    // Conv + SiLU for the channels this block owns: q and k of key head `kh`
    // (threads 0..127 take q, then k) and v of each of its value heads (one
    // channel per thread). A channel's window is read and rewritten by its
    // owner alone.
    const unsigned int q_ch = kh * BRAIN_GDN_HD + j;
    const unsigned int k_ch = key_dim + kh * BRAIN_GDN_HD + j;
    const unsigned int v_ch = 2u * key_dim + vh * BRAIN_GDN_HD + j;

    float h0[3], h1[3], h2[3], w0[4], w1[4], w2[4];
    // q: the threads of slot 0, k: those of slot 1 (slot 0 again when the key
    // head has a single value head); v: every thread.
    const bool do_q = (slot == 0);
    const bool do_k = (group > 1) ? (slot == 1) : (slot == 0);
    float qy = 0.0f, ky = 0.0f, vy = 0.0f;
    float q_x = 0.0f, k_x = 0.0f, v_x = 0.0f;
    if (do_q) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h0[i] = hist[(unsigned long long)q_ch * 3 + i]; }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w0[i] = conv_w[(unsigned long long)q_ch * 4 + i]; }
        q_x = mixed[q_ch];
    }
    if (do_k) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h1[i] = hist[(unsigned long long)k_ch * 3 + i]; }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w1[i] = conv_w[(unsigned long long)k_ch * 4 + i]; }
        k_x = mixed[k_ch];
    }
    if (live) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { h2[i] = hist[(unsigned long long)v_ch * 3 + i]; }
#pragma unroll
        for (int i = 0; i < 4; ++i) { w2[i] = conv_w[(unsigned long long)v_ch * 4 + i]; }
        v_x = mixed[v_ch];
    }
    float b_in = 0.0f, a_in = 0.0f, al = 0.0f, dtb = 0.0f, zin = 0.0f, nw = 0.0f;
    if (live) {
        b_in = bproj[vh];
        a_in = aproj[vh];
        al = a_log[vh];
        dtb = dt_bias[vh];
        zin = z[(unsigned long long)vh * BRAIN_GDN_HD + j];
        nw = norm_w[j];
    }

    // Phase 2: the causal conv (taps 0..3 in order) + SiLU, and the window shift.
    if (do_q) {
        float acc = 0.0f;
        acc = __fadd_rn(acc, __fmul_rn(h0[0], w0[0]));
        acc = __fadd_rn(acc, __fmul_rn(h0[1], w0[1]));
        acc = __fadd_rn(acc, __fmul_rn(h0[2], w0[2]));
        acc = __fadd_rn(acc, __fmul_rn(q_x, w0[3]));
        qy = brain_silu(acc);
        hist[(unsigned long long)q_ch * 3 + 0] = h0[1];
        hist[(unsigned long long)q_ch * 3 + 1] = h0[2];
        hist[(unsigned long long)q_ch * 3 + 2] = q_x;
        conv_q[j] = qy;
    }
    if (do_k) {
        float acc = 0.0f;
        acc = __fadd_rn(acc, __fmul_rn(h1[0], w1[0]));
        acc = __fadd_rn(acc, __fmul_rn(h1[1], w1[1]));
        acc = __fadd_rn(acc, __fmul_rn(h1[2], w1[2]));
        acc = __fadd_rn(acc, __fmul_rn(k_x, w1[3]));
        ky = brain_silu(acc);
        hist[(unsigned long long)k_ch * 3 + 0] = h1[1];
        hist[(unsigned long long)k_ch * 3 + 1] = h1[2];
        hist[(unsigned long long)k_ch * 3 + 2] = k_x;
        conv_k[j] = ky;
    }
    if (live) {
        float acc = 0.0f;
        acc = __fadd_rn(acc, __fmul_rn(h2[0], w2[0]));
        acc = __fadd_rn(acc, __fmul_rn(h2[1], w2[1]));
        acc = __fadd_rn(acc, __fmul_rn(h2[2], w2[2]));
        acc = __fadd_rn(acc, __fmul_rn(v_x, w2[3]));
        vy = brain_silu(acc);
        hist[(unsigned long long)v_ch * 3 + 0] = h2[1];
        hist[(unsigned long long)v_ch * 3 + 1] = h2[2];
        hist[(unsigned long long)v_ch * 3 + 2] = v_x;
    }
    __syncthreads();

    // Phase 3: L2-normalise q and k (the sum of squares is one thread's
    // ascending walk, as in the reference). Two threads, one per vector.
    if (t == 0 || t == 1) {
        const float* src = (t == 0) ? conv_q : conv_k;
        float ss = 0.0f;
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { ss = __fadd_rn(ss, __fmul_rn(src[i], src[i])); }
        const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, l2_eps)));
        float* dst = (t == 0) ? qn : kn;
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { dst[i] = __fmul_rn(src[i], r); }
    }
    __syncthreads();

    if (live) {
        // Phase 4: gates, then the delta-rule update on this thread's state column.
        const float beta = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-b_in)));
        const float xg = __fadd_rn(a_in, dtb);
        const float softplus = __fadd_rn(fmaxf(xg, 0.0f), logf(__fadd_rn(1.0f, expf(-fabsf(xg)))));
        const float g = __fmul_rn(-expf(al), softplus);
        const float decay = expf(g);

#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { s[i] = __fmul_rn(s[i], decay); }
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { kv = __fadd_rn(kv, __fmul_rn(kn[i], s[i])); }
        const float delta = __fmul_rn(beta, __fsub_rn(vy, kv));
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) {
            float acc = 0.0f;
            acc = __fadd_rn(acc, __fmul_rn(kn[i], delta));
            s[i] = __fadd_rn(s[i], acc);
        }
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { o = __fadd_rn(o, __fmul_rn(qn[i], s[i])); }
        o = __fmul_rn(q_scale, o);

#pragma unroll
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { state_col[(unsigned long long)i * BRAIN_GDN_HD] = s[i]; }
        normed[slot][j] = o;
    }
    __syncthreads();

    // Phase 5: gated RMSNorm per value head. The sum of squares is one thread's
    // ascending walk over the head (the reference kernel is one thread per row).
    if (live && j == 0) {
        float ss = 0.0f;
        for (int i = 0; i < BRAIN_GDN_HD; ++i) { ss = __fadd_rn(ss, __fmul_rn(normed[slot][i], normed[slot][i])); }
        inv_rms[slot] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(BRAIN_GDN_HD)), rms_eps)));
    }
    __syncthreads();
    if (live) {
        const float nrm = __fmul_rn(__fmul_rn(nw, normed[slot][j]), inv_rms[slot]);
        gated[(unsigned long long)vh * BRAIN_GDN_HD + j] = __fmul_rn(nrm, brain_silu(zin));
    }
}
