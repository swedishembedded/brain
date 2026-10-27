// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in int8 quantization for embedded accelerators then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The NPU's int8 arithmetic: how a float convolution becomes the integers the
//! hardware computes with, and the exact integer result it produces.
//!
//! Activations are per-tensor affine int8 (`real = scale * (q - zero_point)`).
//! Weights are per-output-channel affine int8. The accumulator is
//! `sum((x) * (w - w_zp)) + bias`, where `x` is the raw int8 input and the
//! input zero point is folded into `bias`, so the accumulator is proportional
//! to the real output of its channel. The output conversion is two rounded
//! fixed-point steps: a Q14 per-channel factor (the channel's weight scale over
//! the largest one), then one Q`shift` factor shared by the layer. Both round
//! half to even. A fused ReLU clamps the accumulator at zero before conversion.
//!
//! This arithmetic, its rounding and the register encoding were measured on the
//! board by open-rknpu (`quantization.py`, `chain.py::native_quantize`,
//! `native.py::native_input_reference`; MIT, see `/NOTICE.md`).

use crate::{Error, Result};

/// Affine int8 quantization of one activation tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QParams {
    pub scale: f32,
    pub zero_point: i8,
}

impl QParams {
    /// The asymmetric parameters covering `[lo, hi]` (widened to include 0).
    pub fn from_range(lo: f32, hi: f32) -> Result<QParams> {
        let (lo, hi) = (lo.min(0.0), hi.max(0.0));
        if !(lo.is_finite() && hi.is_finite()) {
            return Err(Error::Quantization(format!("activation range [{lo}, {hi}] is not finite")));
        }
        let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
        let zero_point = (-128.0 - lo / scale).round_ties_even().clamp(-128.0, 127.0) as i8;
        Ok(QParams { scale, zero_point })
    }

    /// Symmetric parameters covering `[-m, m]`, zero point 0. Tensors feeding a
    /// residual Add must be quantized this way (see `eltwise`).
    pub fn symmetric(max_abs: f32) -> Result<QParams> {
        if !(max_abs.is_finite() && max_abs >= 0.0) {
            return Err(Error::Quantization(format!("activation bound {max_abs} is not finite")));
        }
        Ok(QParams { scale: if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 }, zero_point: 0 })
    }

    pub fn quantize(&self, x: f32) -> i8 {
        (x / self.scale + f32::from(self.zero_point)).round_ties_even().clamp(-128.0, 127.0) as i8
    }

    pub fn dequantize(&self, q: i8) -> f32 {
        self.scale * (f32::from(q) - f32::from(self.zero_point))
    }
}

/// The output conversion shared by every int8-producing task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requant {
    /// Q`shift` factor, 1..=32767.
    pub multiplier: u16,
    pub shift: u8,
    pub output_zero_point: i8,
}

impl Requant {
    /// Convert one accumulator, already scaled by its channel's Q14 factor
    /// `channel` (16384 = 1.0), to the output code.
    pub fn apply(&self, acc: i64, channel: u16) -> i8 {
        let scaled = round_shift(acc * i64::from(channel), 14);
        let mut product = scaled * i64::from(self.multiplier);
        let out = if self.shift > 0 {
            product += i64::from(self.output_zero_point) << self.shift;
            round_shift(product, u32::from(self.shift))
        } else {
            product + i64::from(self.output_zero_point)
        };
        out.clamp(-128, 127) as i8
    }

    /// The multiplier/shift pair for a real factor `factor` (input scale times
    /// the largest weight scale over the output scale).
    pub fn for_factor(factor: f64, output_zero_point: i8) -> Result<Requant> {
        if !(factor.is_finite() && factor > 0.0) {
            return Err(Error::Quantization(format!("output conversion factor {factor} is not positive")));
        }
        let shift = (32767.0 / factor).log2().floor().min(31.0);
        if shift < 0.0 {
            return Err(Error::Quantization(format!("output conversion factor {factor} is too large for a Q15 multiplier")));
        }
        let multiplier = (factor * 2f64.powi(shift as i32)).round();
        if !(1.0..=32767.0).contains(&multiplier) {
            return Err(Error::Quantization(format!("output conversion factor {factor} underflows the multiplier")));
        }
        Ok(Requant { multiplier: multiplier as u16, shift: shift as u8, output_zero_point })
    }
}

/// `x >> shift`, rounding half to even (the hardware's rounding).
fn round_shift(x: i64, shift: u32) -> i64 {
    (x + (1 << (shift - 1)) - 1 + ((x >> shift) & 1)) >> shift
}

/// A source of weights in order: each call feeds every weight, in chunks, to
/// the sink it is given.
pub type WeightStream<'a> = &'a mut dyn FnMut(&mut dyn FnMut(&[f32]));

/// A convolution's weights, bias and output conversion in the NPU's integers.
#[derive(Clone, Debug)]
pub struct ConvQuant {
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel: usize,
    /// `[out][in][ky][kx]`, the ONNX/PyTorch order.
    pub weights: Vec<i8>,
    pub weight_zero_points: Vec<i8>,
    /// Bias in accumulator units, input zero point folded in.
    pub bias: Vec<i32>,
    /// Per-channel Q14 factor, the channel's weight scale over the largest.
    pub channel_multipliers: Vec<u16>,
    pub requant: Requant,
    pub relu: bool,
}

impl ConvQuant {
    /// Quantize a float convolution (`weights` `[out][in][k][k]`, `bias`
    /// `[out]`) that reads `input` and writes `output`.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(weights: &[f32], bias: &[f32], out_channels: usize, in_channels: usize, kernel: usize, input: QParams, output: QParams, relu: bool) -> Result<ConvQuant> {
        ConvQuant::quantize_streamed(&mut |sink| sink(weights), bias, out_channels, in_channels, kernel, input, output, relu)
    }

    /// [`ConvQuant::quantize`] over weights that `stream` produces in order,
    /// in chunks, each time it is called (twice: ranges, then codes), so no
    /// float copy of the weights is ever held.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_streamed(stream: WeightStream<'_>, bias: &[f32], out_channels: usize, in_channels: usize, kernel: usize, input: QParams, output: QParams, relu: bool) -> Result<ConvQuant> {
        let per_out = in_channels * kernel * kernel;
        let total = out_channels * per_out;
        if bias.len() != out_channels || bias.iter().any(|v| !v.is_finite()) {
            return Err(Error::Quantization(format!("conv [{out_channels}, {in_channels}, {kernel}, {kernel}] needs {out_channels} finite biases, got {}", bias.len())));
        }
        // Pass 1: each output channel's range, widened to include 0.
        let mut ranges = vec![(0f32, 0f32); out_channels];
        let (mut seen, mut finite) = (0usize, true);
        stream(&mut |chunk| {
            for &v in chunk {
                if seen < total {
                    let r = &mut ranges[seen / per_out];
                    *r = (r.0.min(v), r.1.max(v));
                }
                finite &= v.is_finite();
                seen += 1;
            }
        });
        if seen != total || !finite {
            return Err(Error::Quantization(format!("conv [{out_channels}, {in_channels}, {kernel}, {kernel}] expects {total} finite weights, got {seen}")));
        }
        let raw_scales: Vec<f32> = ranges.iter().map(|(lo, hi)| (hi - lo) / 255.0).collect();
        let max_scale = raw_scales.iter().fold(0f32, |a, &b| a.max(b));
        let max_scale = if max_scale > 0.0 { max_scale } else { 1.0 / 32768.0 };
        // An all-zero channel still carries its bias; give it the common scale.
        let scales: Vec<f32> = raw_scales.iter().map(|&s| if s > 0.0 { s } else { max_scale }).collect();
        let zero_points: Vec<i8> = ranges.iter().zip(&scales).map(|((lo, _), s)| (-128.0 - lo / s).round_ties_even().clamp(-128.0, 127.0) as i8).collect();

        // Pass 2: the codes, and each channel's sum of centered codes (the
        // input zero point's share of the bias).
        let mut weights = Vec::with_capacity(total);
        let mut centered = vec![0i64; out_channels];
        stream(&mut |chunk| {
            for &v in chunk {
                let o = weights.len() / per_out;
                let q = ((v / scales[o]).round_ties_even() + f32::from(zero_points[o])).clamp(-128.0, 127.0) as i8;
                centered[o] += i64::from(q) - i64::from(zero_points[o]);
                weights.push(q);
            }
        });
        let mut q = ConvQuant {
            out_channels,
            in_channels,
            kernel,
            weights,
            weight_zero_points: zero_points,
            bias: Vec::with_capacity(out_channels),
            channel_multipliers: scales.iter().map(|&s| (f64::from(s) / f64::from(max_scale) * 16384.0).round_ties_even() as u16).collect(),
            requant: Requant::for_factor(f64::from(max_scale) * f64::from(input.scale) / f64::from(output.scale), output.zero_point)?,
            relu,
        };
        for o in 0..out_channels {
            let units = (f64::from(bias[o]) / (f64::from(scales[o]) * f64::from(input.scale))).round_ties_even();
            let folded = units as i64 - i64::from(input.zero_point) * centered[o];
            q.bias.push(i32::try_from(folded).map_err(|_| Error::Quantization(format!("conv bias of channel {o} overflows int32")))?);
        }
        Ok(q)
    }

    /// The same convolution with a `k2 x k2` kernel (`k2 > kernel`): the new
    /// taps (last rows and columns) get each channel's weight zero point, a
    /// zero weight, so the results are unchanged given the padding grows to
    /// match.
    pub fn grow_kernel(mut self, k2: usize) -> ConvQuant {
        let k = self.kernel;
        if k2 == k {
            return self;
        }
        let filters = self.out_channels * self.in_channels;
        let mut w = Vec::with_capacity(filters * k2 * k2);
        for f in 0..filters {
            let zp = self.weight_zero_points[f / self.in_channels];
            for ky in 0..k2 {
                for kx in 0..k2 {
                    w.push(if ky < k && kx < k { self.weights[(f * k + ky) * k + kx] } else { zp });
                }
            }
        }
        self.weights = w;
        self.kernel = k2;
        self
    }

    /// The accumulator of output channel `o` for one receptive field
    /// `patch` (`[in][ky][kx]` raw int8 inputs, padding already applied).
    pub fn accumulate(&self, o: usize, patch: impl Iterator<Item = i8>) -> i64 {
        let per_out = self.in_channels * self.kernel * self.kernel;
        let zp = i64::from(self.weight_zero_points[o]);
        let w = &self.weights[o * per_out..(o + 1) * per_out];
        let acc: i64 = w.iter().zip(patch).map(|(&w, x)| (i64::from(w) - zp) * i64::from(x)).sum();
        let acc = acc + i64::from(self.bias[o]);
        if self.relu {
            acc.max(0)
        } else {
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_half_to_even() {
        assert_eq!(round_shift(5, 1), 2); // 2.5 -> 2
        assert_eq!(round_shift(7, 1), 4); // 3.5 -> 4
        assert_eq!(round_shift(-5, 1), -2); // -2.5 -> -2
        assert_eq!(round_shift(-7, 1), -4); // -3.5 -> -4
        assert_eq!(round_shift(6, 2), 2); // 1.5 -> 2
    }

    #[test]
    fn quantized_conv_tracks_the_float_conv() {
        // One 1x1 conv, two output channels: y = w.x + b on a known input.
        let input = QParams::from_range(-1.0, 1.0).unwrap();
        let output = QParams::from_range(-4.0, 4.0).unwrap();
        let w = [0.5f32, -0.25, 1.0, 0.75];
        let b = [0.1f32, -0.2];
        let q = ConvQuant::quantize(&w, &b, 2, 2, 1, input, output, false).unwrap();
        let x = [0.6f32, -0.3];
        let xq: Vec<i8> = x.iter().map(|&v| input.quantize(v)).collect();
        for o in 0..2 {
            let want = w[o * 2] * x[0] + w[o * 2 + 1] * x[1] + b[o];
            let acc = q.accumulate(o, xq.iter().copied());
            let got = output.dequantize(q.requant.apply(acc, q.channel_multipliers[o]));
            assert!((got - want).abs() < 2.0 * output.scale, "channel {o}: {got} vs {want}");
        }
    }
}
