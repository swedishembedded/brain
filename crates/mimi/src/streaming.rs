// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming state for the codec's causal (transposed-)convolutions — the core
//! of a **stateful streaming decoder** that decodes only the *new* frames each
//! chunk instead of re-decoding a warmup window.
//!
//! Each causal module carries the minimal state that makes chunked decoding
//! bit-identical to a single full decode:
//!
//! * [`StreamConv1d`] (causal `Conv1d`, stride 1): the full decode left-pads the
//!   input by `dilation·(K-1)` zeros. Streaming keeps the last `dilation·(K-1)`
//!   input columns as the next chunk's left context, so each chunk emits exactly
//!   its `L_new` output columns.
//! * [`StreamConvTr1d`] (causal `ConvTranspose1d`, the upsamplers): the full
//!   decode keeps the first `L·stride` outputs (`pad_end = K-stride`). Each input
//!   spreads over `K` outputs at `stride`, so adjacent inputs overlap by
//!   `K-stride`. Streaming emits `L_new·stride` finalized samples per chunk and
//!   carries the trailing `K-stride` raw samples into the next chunk's head.
//!
//! Both are exact (see the tests): concatenating the per-chunk outputs equals the
//! one-shot output, which equals the reference causal op. Built on the pure-CPU
//! [`audio::conv`] references so the state math is validated without a GPU/NPU.

use backend_cpu::par;

/// Streaming causal `Conv1d` (stride 1). Tensors are channel-major `[C, L]`.
pub struct StreamConv1d {
    cin: usize,
    cout: usize,
    k: usize,
    dil: usize,
    groups: usize,
    w: Vec<f32>,   // [cout, cin/groups, k]
    bias: Vec<f32>, // [cout]
    ctx: usize,    // dilation*(k-1) left-context columns
    buf: Vec<f32>, // [cin, ctx] previous-chunk tail (zeros at start)
}

impl StreamConv1d {
    pub fn new(cin: usize, cout: usize, k: usize, dil: usize, groups: usize, w: Vec<f32>, bias: Vec<f32>) -> StreamConv1d {
        let ctx = dil * (k - 1);
        StreamConv1d { cin, cout, k, dil, groups, w, bias, ctx, buf: vec![0.0; cin * ctx] }
    }

    /// Reset to a fresh utterance (zero left context).
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Feed `x = [cin, l_new]`, return `[cout, l_new]`.
    pub fn step(&mut self, x: &[f32], l_new: usize) -> Vec<f32> {
        let (cin, cout, ctx) = (self.cin, self.cout, self.ctx);
        let lin = ctx + l_new;
        // Prepend the saved left context to the new input.
        let mut xin = vec![0.0f32; cin * lin];
        for c in 0..cin {
            xin[c * lin..c * lin + ctx].copy_from_slice(&self.buf[c * ctx..c * ctx + ctx]);
            xin[c * lin + ctx..c * lin + lin].copy_from_slice(&x[c * l_new..c * l_new + l_new]);
        }
        // Causal conv (stride 1, pad 0 — context is prepended), parallel over
        // output channels. Same per-output accumulation order as the scalar
        // reference, so the result is bit-identical (the exactness tests hold).
        let (k, dil) = (self.k, self.dil);
        let cin_g = cin / self.groups;
        let cout_g = cout / self.groups;
        let (w, bias) = (&self.w, &self.bias);
        let mut y = vec![0.0f32; cout * l_new];
        par::rows_mut(&mut y, l_new, |co, yrow| {
            let g = co / cout_g;
            let b = bias[co];
            for (lo, slot) in yrow.iter_mut().enumerate() {
                let mut acc = b;
                for cl in 0..cin_g {
                    let xbase = (g * cin_g + cl) * lin;
                    let wbase = (co * cin_g + cl) * k;
                    for kw in 0..k {
                        acc += xin[xbase + lo + kw * dil] * w[wbase + kw];
                    }
                }
                *slot = acc;
            }
        });
        // Save the last `ctx` input columns for the next chunk.
        for c in 0..cin {
            self.buf[c * ctx..c * ctx + ctx].copy_from_slice(&xin[c * lin + lin - ctx..c * lin + lin]);
        }
        y
    }
}

/// Streaming causal `ConvTranspose1d` (upsample by `stride`). `[C, L]` in/out.
pub struct StreamConvTr1d {
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    w: Vec<f32>,    // [cin, cout, k]
    bias: Vec<f32>, // [cout]
    ov: usize,      // k - stride overlap columns
    carry: Vec<f32>, // [cout, ov] not-yet-finalized output tail
}

impl StreamConvTr1d {
    pub fn new(cin: usize, cout: usize, k: usize, stride: usize, w: Vec<f32>, bias: Vec<f32>) -> StreamConvTr1d {
        let ov = k - stride;
        StreamConvTr1d { cin, cout, k, stride, w, bias, ov, carry: vec![0.0; cout * ov] }
    }

    pub fn reset(&mut self) {
        self.carry.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Feed `x = [cin, l_new]`, return the finalized `[cout, l_new*stride]`.
    pub fn step(&mut self, x: &[f32], l_new: usize) -> Vec<f32> {
        let (cin, cout, k, stride, ov) = (self.cin, self.cout, self.k, self.stride, self.ov);
        let raw_len = (l_new - 1) * stride + k;
        // Transposed conv (groups 1, no bias), parallel over output channels —
        // same accumulation order as the scalar reference (bit-identical).
        let w = &self.w;
        let mut raw = vec![0.0f32; cout * raw_len];
        par::rows_mut(&mut raw, raw_len, |co, rrow| {
            for (lo, slot) in rrow.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for kw in 0..k {
                    if lo >= kw && (lo - kw) % stride == 0 {
                        let li = (lo - kw) / stride;
                        if li < l_new {
                            for cl in 0..cin {
                                acc += x[cl * l_new + li] * w[(cl * cout + co) * k + kw];
                            }
                        }
                    }
                }
                *slot = acc;
            }
        });
        // Add the previous carry to the first `ov` (overlapping) output columns.
        for co in 0..cout {
            for j in 0..ov {
                raw[co * raw_len + j] += self.carry[co * ov + j];
            }
        }
        let fin = l_new * stride;
        let mut out = vec![0.0f32; cout * fin];
        for co in 0..cout {
            for j in 0..fin {
                out[co * fin + j] = raw[co * raw_len + j] + self.bias[co];
            }
        }
        // New carry = the trailing `ov` raw samples (future inputs still add here).
        for co in 0..cout {
            for j in 0..ov {
                self.carry[co * ov + j] = raw[co * raw_len + fin + j];
            }
        }
        out
    }
}

/// Streaming causal `ConvTranspose1d` with the SYMMETRIC crop convention
/// `mimi::model::Codec::causal_convtr_sym` uses for Qwen3-Omni's Code2Wav
/// SEANet decoder (`(l-1)*stride` output length, cropping `pad` samples off
/// BOTH sides of the raw `(l-1)*stride+k` buffer) — [`StreamConvTr1d`] above
/// implements the DIFFERENT `pad=0` convention the standalone Qwen3-TTS
/// codec's own `causal_convtr` uses (`l*stride`, right-crop only), which is
/// why this is a new type rather than a parameter on that one.
///
/// Built as a thin wrapper around [`StreamConvTr1d`], not a re-derivation:
/// [`StreamConvTr1d`]'s own carry mechanism already reconstructs the raw
/// (uncropped) buffer's finalized segments exactly, chunk by chunk (proven
/// by its own `stream_convtr1d_chunked_equals_full` test) — cropping is a
/// pure post-hoc slice on top of that, not new streaming-state math.
///
/// **Right crop is automatic, not implemented explicitly**: every real
/// caller in this codebase (`causal_convtr_sym`'s SEANet decoder call sites)
/// uses `k = 2·stride`, so `ov = k - stride == stride == pad` always — and
/// [`StreamConvTr1d`] already never emits its final `ov`-sample carry (the
/// caller simply stops calling `step` once the true input is exhausted), so
/// the trailing `pad` samples the symmetric convention wants dropped are
/// exactly the ones that were never going to be emitted anyway. `new`
/// asserts `k - stride == pad` so a future caller outside that shape fails
/// loudly instead of silently keeping (or dropping too many) trailing
/// samples — a general right-crop (needing an explicit `is_final` flush
/// path) is real, separable work this shape doesn't need.
///
/// **Left crop IS implemented**: the true one-shot buffer's first `pad`
/// samples must never reach the caller. Held as a simple "how many samples
/// are still owed to the crop" counter, consumed (chunk-spanning if a chunk
/// is smaller than `pad`) on the way out of the FIRST calls to [`Self::step`]
/// only.
pub struct StreamConvTr1dSym {
    inner: StreamConvTr1d,
    cout: usize,
    stride: usize,
    left_pad_remaining: usize,
    pad: usize,
}

impl StreamConvTr1dSym {
    pub fn new(cin: usize, cout: usize, k: usize, stride: usize, pad: usize, w: Vec<f32>, bias: Vec<f32>) -> StreamConvTr1dSym {
        assert_eq!(k - stride, pad, "StreamConvTr1dSym only implements the ov==pad shape causal_convtr_sym's real callers use (k=2*stride, pad=stride) -- a mismatched (k,stride,pad) needs an explicit is_final right-crop path, not built here");
        StreamConvTr1dSym { inner: StreamConvTr1d::new(cin, cout, k, stride, w, bias), cout, stride, left_pad_remaining: pad, pad }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.left_pad_remaining = self.pad;
    }

    /// Feed `x = [cin, l_new]`, return the finalized (left-cropped)
    /// `[cout, <=l_new*stride]` — shorter than `l_new*stride` only while the
    /// left crop is still being consumed (at most across the first few
    /// calls, until `pad` total samples have been dropped).
    pub fn step(&mut self, x: &[f32], l_new: usize) -> Vec<f32> {
        let raw = self.inner.step(x, l_new);
        let seg = l_new * self.stride;
        if self.left_pad_remaining == 0 {
            return raw;
        }
        let drop = self.left_pad_remaining.min(seg);
        self.left_pad_remaining -= drop;
        let out_len = seg - drop;
        let mut out = vec![0.0f32; self.cout * out_len];
        for c in 0..self.cout {
            out[c * out_len..(c + 1) * out_len].copy_from_slice(&raw[c * seg + drop..c * seg + seg]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    fn concat_chunks(cout: usize, parts: &[Vec<f32>], per: &[usize]) -> Vec<f32> {
        // Re-stitch channel-major chunks [cout, li] into one [cout, sum(li)].
        let total: usize = per.iter().sum();
        let mut out = vec![0.0f32; cout * total];
        let mut off = 0;
        for (p, &li) in parts.iter().zip(per) {
            for co in 0..cout {
                out[co * total + off..co * total + off + li].copy_from_slice(&p[co * li..co * li + li]);
            }
            off += li;
        }
        out
    }

    #[test]
    fn stream_conv1d_chunked_equals_full() {
        let mut seed = Lcg::new(12345);
        let (cin, cout, k, dil) = (3usize, 4usize, 3usize, 2usize);
        let l = 20usize;
        let w: Vec<f32> = seed.vec(cout * cin * k);
        let bias: Vec<f32> = seed.vec(cout);
        let x: Vec<f32> = seed.vec(cin * l);

        // One-shot.
        let mut full = StreamConv1d::new(cin, cout, k, dil, 1, w.clone(), bias.clone());
        let y_full = full.step(&x, l);

        // Chunked (3 + 7 + 10).
        let mut s = StreamConv1d::new(cin, cout, k, dil, 1, w, bias);
        let sizes = [3usize, 7, 10];
        let mut parts = vec![];
        let mut off = 0;
        for &li in &sizes {
            let mut chunk = vec![0.0f32; cin * li];
            for c in 0..cin {
                chunk[c * li..c * li + li].copy_from_slice(&x[c * l + off..c * l + off + li]);
            }
            parts.push(s.step(&chunk, li));
            off += li;
        }
        let y_chunk = concat_chunks(cout, &parts, &sizes);
        let maxd = y_full.iter().zip(&y_chunk).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-6, "conv1d chunked != full: {maxd}");
    }

    #[test]
    fn stream_convtr1d_chunked_equals_full() {
        let mut seed = Lcg::new(999);
        let (cin, cout, stride) = (3usize, 2usize, 4usize);
        let k = 2 * stride; // SEANet block.1: k = 2*rate
        let l = 16usize;
        let w: Vec<f32> = seed.vec(cin * cout * k);
        let bias: Vec<f32> = seed.vec(cout);
        let x: Vec<f32> = seed.vec(cin * l);

        let mut full = StreamConvTr1d::new(cin, cout, k, stride, w.clone(), bias.clone());
        let y_full = full.step(&x, l);

        let mut s = StreamConvTr1d::new(cin, cout, k, stride, w, bias);
        let sizes = [5usize, 4, 7];
        let mut parts = vec![];
        let mut off = 0;
        for &li in &sizes {
            let mut chunk = vec![0.0f32; cin * li];
            for c in 0..cin {
                chunk[c * li..c * li + li].copy_from_slice(&x[c * l + off..c * l + off + li]);
            }
            parts.push(s.step(&chunk, li)); // input li frames -> li*stride output samples
            off += li;
        }
        // Fix per-chunk lengths to li*stride for the stitch.
        let outs: Vec<usize> = sizes.iter().map(|&li| li * stride).collect();
        let y_chunk = concat_chunks(cout, &parts, &outs);
        let maxd = y_full.iter().zip(&y_chunk).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-6, "convtr1d chunked != full: {maxd}");
    }

    #[test]
    fn stream_convtr1d_sym_chunked_equals_full() {
        let mut seed = Lcg::new(2026);
        let (cin, cout, stride) = (3usize, 2usize, 4usize);
        let k = 2 * stride; // causal_convtr_sym's real shape: k = 2*rate, pad = rate
        let pad = stride;
        let l = 16usize;
        let w: Vec<f32> = seed.vec(cin * cout * k);
        let bias: Vec<f32> = seed.vec(cout);
        let x: Vec<f32> = seed.vec(cin * l);

        let mut full = StreamConvTr1dSym::new(cin, cout, k, stride, pad, w.clone(), bias.clone());
        let y_full = full.step(&x, l);
        assert_eq!(y_full.len(), cout * (l - 1) * stride, "one-shot symmetric-crop length must be (l-1)*stride");

        let mut s = StreamConvTr1dSym::new(cin, cout, k, stride, pad, w, bias);
        let sizes = [5usize, 4, 7];
        let mut parts = vec![];
        let mut lens = vec![];
        let mut off = 0;
        for &li in &sizes {
            let mut chunk = vec![0.0f32; cin * li];
            for c in 0..cin {
                chunk[c * li..c * li + li].copy_from_slice(&x[c * l + off..c * l + off + li]);
            }
            let part = s.step(&chunk, li);
            lens.push(part.len() / cout);
            parts.push(part);
            off += li;
        }
        assert_eq!(lens.iter().sum::<usize>(), (l - 1) * stride, "chunked total length must match the one-shot length");
        let y_chunk = concat_chunks(cout, &parts, &lens);
        let maxd = y_full.iter().zip(&y_chunk).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-6, "convtr1d_sym chunked != full: {maxd}");
    }

    /// `StreamConvTr1dSym`'s ONE-SHOT call (not just its own chunked-vs-full
    /// self-consistency, which `stream_convtr1d_sym_chunked_equals_full`
    /// already proves) against the REAL GPU kernel dispatch
    /// (`audio::conv::convtr1d_fwd`, the exact call `mimi::model::Codec::
    /// causal_convtr_sym` makes) on the SAME weights -- an independent
    /// witness, not a self-check, so this test cannot pass merely because
    /// the streaming primitive and its own oracle share a bug.
    #[test]
    fn stream_convtr1d_sym_one_shot_matches_the_real_kernel_dispatch() {
        let mut seed = Lcg::new(4242);
        let (cin, cout, stride) = (3u32, 2u32, 4u32);
        let k = 2 * stride;
        let pad = stride;
        let l = 12u32;
        let lo = (l - 1) * stride; // causal_convtr_sym's own `new_l` formula
        let w: Vec<f32> = seed.vec((cin * cout * k) as usize);
        let bias: Vec<f32> = seed.vec(cout as usize);
        let x: Vec<f32> = seed.vec((cin * l) as usize);

        // Real kernel dispatch, same params causal_convtr_sym itself builds.
        let g = gpu_core::testgpu::dev(&[("convtr1d", kernels::CONVTR1D)]);
        let ck = audio::conv::ConvKernels { fwd: 0, dx: 0, dw: 0 };
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
        let xb = g.storage_init("x", &x);
        let wb = g.storage_init("w", &w);
        let out = g.storage((cout * lo) as u64);
        g.submit(&[], &[audio::conv::convtr1d_fwd(&g, &ck, &c, &xb, &wb, &out)]);
        let mut want = g.read(&out, (cout * lo) as usize).to_vec();
        // The kernel dispatch has no bias add (causal_convtr_sym adds it
        // separately via add_ncl_bias) -- add it here to compare apples to
        // apples against StreamConvTr1dSym::step, which DOES include bias
        // (mirroring StreamConvTr1d's own convention).
        for co in 0..cout as usize {
            for j in 0..lo as usize {
                want[co * lo as usize + j] += bias[co];
            }
        }

        let mut s = StreamConvTr1dSym::new(cin as usize, cout as usize, k as usize, stride as usize, pad as usize, w, bias);
        let got = s.step(&x, l as usize);

        let maxd = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-4, "StreamConvTr1dSym one-shot != real kernel dispatch: {maxd}");
    }

    #[test]
    #[should_panic(expected = "only implements the ov==pad shape")]
    fn stream_convtr1d_sym_rejects_a_mismatched_shape() {
        StreamConvTr1dSym::new(1, 1, 5, 2, 2, vec![0.0; 5], vec![0.0]); // k-stride=3 != pad=2
    }
}
