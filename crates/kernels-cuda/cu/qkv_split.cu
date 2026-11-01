// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for latency-bound LLM decode on GPUs for its clients. If your team needs
// expertise in cutting a decode step's launches between the weight streams of an attention layer then you can procure
// our services by sending an email to info@swedishembedded.com.
//
// The rows of the ONE product a fused q|k|v weight makes, taken apart in a single launch. A row of the product is
//
//   [ q: n_heads x (hd query values, hd gate values) | k: kvd | v: kvd ]        (2 qd + 2 kvd floats, qd = n_heads * hd)
//
// and the attention front half wants them as separate dense buffers. Pure copies: every output element is bit for bit
// the product's.
//
//   params : u32 [rows, qd, hd, kvd, flags]   qd, hd, kvd multiples of 4. flags bit 0: de-interleave the query heads.
//   x      : [rows, 2 qd + 2 kvd] f32     the product
//   a, b, c, d : f32 outputs, by flag
//     bit 0 clear : a = [rows, 2 qd] the query projection as the unfused product writes it (value and gate interleaved by head),
//                   b = [rows, kvd] k,  c = [rows, kvd] v,  d not written
//     bit 0 set   : a = [rows, qd] query values, b = [rows, qd] gates (what two `concat_split` launches make),
//                   c = [rows, kvd] k,  d = [rows, kvd] v
//
// One thread per four consecutive elements of a row (16-byte accesses). No `__restrict__`: brain's device buffers alias by design.

extern "C" __global__ void __launch_bounds__(256)
brain_qkv_split(const unsigned int* params, const float* x, float* a, float* b, float* c, float* d) {
    const unsigned long long rows = params[0];
    const unsigned long long qd = params[1];
    const unsigned long long hd = params[2];
    const unsigned long long kvd = params[3];
    const bool split_heads = (params[4] & 1u) != 0u;
    const unsigned long long width = 2ull * qd + 2ull * kvd;
    const unsigned long long quads = width / 4ull;
    const unsigned long long t = ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * 256ull + threadIdx.x;
    if (t >= rows * quads) { return; }
    const unsigned long long row = t / quads;
    const unsigned long long col = (t % quads) * 4ull;
    const float4 v = *reinterpret_cast<const float4*>(x + row * width + col);
    if (col < 2ull * qd) {
        if (!split_heads) {
            *reinterpret_cast<float4*>(a + row * 2ull * qd + col) = v;
        } else {
            const unsigned long long head = col / (2ull * hd);
            const unsigned long long within = col % (2ull * hd);
            float* out = within < hd ? a : b;
            *reinterpret_cast<float4*>(out + row * qd + head * hd + (within < hd ? within : within - hd)) = v;
        }
    } else if (col < 2ull * qd + kvd) {
        *reinterpret_cast<float4*>((split_heads ? c : b) + row * kvd + (col - 2ull * qd)) = v;
    } else {
        *reinterpret_cast<float4*>((split_heads ? d : c) + row * kvd + (col - 2ull * qd - kvd)) = v;
    }
}
