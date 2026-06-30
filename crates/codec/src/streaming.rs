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

use audio::conv::{conv1d_ref, convtr1d_ref, Conv1d};

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
        let conv = Conv1d {
            n: 1,
            cin: cin as u32,
            l: lin as u32,
            cout: cout as u32,
            k: self.k as u32,
            stride: 1,
            pad: 0,
            dilation: self.dil as u32,
            groups: self.groups as u32,
            lo: l_new as u32,
        };
        let mut y = conv1d_ref(&conv, &xin, &self.w);
        for co in 0..cout {
            for lo in 0..l_new {
                y[co * l_new + lo] += self.bias[co];
            }
        }
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
        let c = Conv1d {
            n: 1,
            cin: cin as u32,
            l: l_new as u32,
            cout: cout as u32,
            k: k as u32,
            stride: stride as u32,
            pad: 0,
            dilation: 1,
            groups: 1,
            lo: raw_len as u32,
        };
        let mut raw = convtr1d_ref(&c, x, &self.w); // [cout, raw_len], no bias
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

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
        let mut seed = 12345u64;
        let (cin, cout, k, dil) = (3usize, 4usize, 3usize, 2usize);
        let l = 20usize;
        let w: Vec<f32> = (0..cout * cin * k).map(|_| rng(&mut seed)).collect();
        let bias: Vec<f32> = (0..cout).map(|_| rng(&mut seed)).collect();
        let x: Vec<f32> = (0..cin * l).map(|_| rng(&mut seed)).collect();

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
        let mut seed = 999u64;
        let (cin, cout, stride) = (3usize, 2usize, 4usize);
        let k = 2 * stride; // SEANet block.1: k = 2*rate
        let l = 16usize;
        let w: Vec<f32> = (0..cin * cout * k).map(|_| rng(&mut seed)).collect();
        let bias: Vec<f32> = (0..cout).map(|_| rng(&mut seed)).collect();
        let x: Vec<f32> = (0..cin * l).map(|_| rng(&mut seed)).collect();

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
}
