// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Dense int8 convolution on the CNA -> CORE -> DPU pipeline.
//!
//! One layer becomes one task per band of output rows: the CNA's input buffer
//! holds at most [`INPUT_ATOMS`] atoms, so a layer whose input does not fit is
//! split into height tiles, each reading the input rows its outputs need. The
//! register fields, the tiling rule (64-byte aligned band starts) and the
//! weight-table layout are ported from open-rknpu's board-verified
//! `native.py` (`native_fields`, `tile_geometry`, `weight_byte`; MIT, see
//! `/NOTICE.md`).

use crate::layout::{Shape, LANES};
use crate::quant::ConvQuant;
use crate::regs::*;
use crate::{Error, Result};

/// The CNA input buffer, in 16-byte atoms (measured by open-rknpu).
pub const INPUT_ATOMS: usize = 6144;

/// Engine enable mask and interrupt mask of a convolution task.
pub const ENABLE: u32 = 0x1d;
pub const INT_MASK: u32 = 0x300;

/// Geometry of one convolution: square kernel, no dilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvGeometry {
    pub input: Shape,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    /// Top, left, bottom, right.
    pub pad: [usize; 4],
}

impl ConvGeometry {
    pub fn output(&self) -> Shape {
        let [t, l, b, r] = self.pad;
        Shape::new((self.input.h + t + b - self.kernel) / self.stride + 1, (self.input.w + l + r - self.kernel) / self.stride + 1, self.out_channels)
    }

    fn lanes(&self) -> usize {
        self.input.c.next_multiple_of(LANES)
    }

    /// Bytes of the packed weight table.
    pub fn weight_bytes(&self) -> usize {
        self.out_channels * self.lanes() * self.kernel * self.kernel
    }

    /// Bytes of the bias/conversion block: 32 bytes per four output channels.
    pub fn bias_bytes(&self) -> usize {
        self.out_channels.div_ceil(4) * 32
    }

    fn validate(&self, q: &ConvQuant) -> Result<()> {
        let [t, l, b, r] = self.pad;
        let k = self.kernel;
        let bad = |why: String| Err(Error::Unsupported(format!("conv {self:?}: {why}")));
        if q.out_channels != self.out_channels || q.in_channels != self.input.c || q.kernel != k {
            return bad(format!("quantized weights are [{}, {}, {k2}, {k2}]", q.out_channels, q.in_channels, k2 = q.kernel));
        }
        if !(1..=31).contains(&k) || !(1..=4).contains(&self.stride) {
            return bad("kernel must be 1..=31 and stride 1..=4".into());
        }
        if t >= k || l >= k || b >= k || r >= k {
            return bad("padding must be smaller than the kernel".into());
        }
        if self.input.h + t + b < k || self.input.w + l + r < k {
            return bad("the padded input is smaller than the kernel".into());
        }
        // Structural walls measured by open-rknpu: 511 weight parts of 32
        // lanes, and a nine-bit output block index.
        if self.lanes().div_ceil(32) > 511 || self.out_channels > 8192 {
            return bad("at most 16352 input and 8192 output channels".into());
        }
        Ok(())
    }
}

/// One band of output rows and the input rows it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Band {
    pub out_rows: (usize, usize),
    pub in_rows: (usize, usize),
    /// Zero rows the band's top edge needs (only the first band has any).
    pub pad_top: usize,
}

/// Split the layer into bands whose input fits the CNA buffer, with every band
/// start on a 64-byte boundary in both the input and the output plane.
pub fn bands(g: &ConvGeometry) -> Result<Vec<Band>> {
    let (ih, iw, k, s) = (g.input.h, g.input.w, g.kernel, g.stride);
    let pt = g.pad[0];
    let out = g.output();
    let tiles = g.lanes() / LANES;
    if ih * iw * tiles <= INPUT_ATOMS {
        return Ok(vec![Band { out_rows: (0, out.h), in_rows: (0, ih), pad_top: pt }]);
    }
    let max_rows = INPUT_ATOMS / (iw * tiles);
    if max_rows < k {
        return Err(Error::Unsupported(format!("conv {g:?}: one kernel window of rows ({k} x {iw} x {tiles} atoms) exceeds the CNA buffer")));
    }
    let mut result = Vec::new();
    let mut oy0 = 0;
    while oy0 < out.h {
        let mut chosen = None;
        for oy1 in (oy0 + 1..=out.h).rev() {
            let first = (oy0 * s) as isize - pt as isize;
            let last = ((oy1 - 1) * s) as isize - pt as isize + k as isize - 1;
            let iy0 = first.max(0) as usize;
            let iy1 = ((last + 1) as usize).min(ih);
            if iy1 - iy0 > max_rows || !(iy0 * iw).is_multiple_of(4) || !(oy0 * out.w).is_multiple_of(4) {
                continue;
            }
            if oy1 < out.h {
                let next_first = ((oy1 * s) as isize - pt as isize).max(0) as usize;
                if !(next_first * iw).is_multiple_of(4) || !(oy1 * out.w).is_multiple_of(4) {
                    continue;
                }
            }
            chosen = Some(Band { out_rows: (oy0, oy1), in_rows: (iy0, iy1), pad_top: (-first).max(0) as usize });
            break;
        }
        let band = chosen.ok_or_else(|| Error::Unsupported(format!("conv {g:?}: no 64-byte aligned height tiling from output row {oy0}")))?;
        result.push(band);
        oy0 = band.out_rows.1;
    }
    Ok(result)
}

/// Where one convolution's operands live, relative to the job base address.
#[derive(Clone, Copy, Debug)]
pub struct ConvAddrs {
    /// Start of the input tensor's first plane.
    pub input: u32,
    /// Plane stride of the input tensor (its full height, even when it is a
    /// slice of a larger concatenation).
    pub input_plane_bytes: usize,
    /// Start of the output's first plane.
    pub output: u32,
    pub output_plane_bytes: usize,
    pub weights: u32,
    pub bias: u32,
}

/// The task programs of one convolution, one per band.
pub fn programs(g: &ConvGeometry, q: &ConvQuant, input_zero_point: i8, at: &ConvAddrs) -> Result<Vec<Program>> {
    g.validate(q)?;
    let (ih, iw, ic) = (g.input.h, g.input.w, g.input.c);
    let (k, s) = (g.kernel, g.stride);
    let lanes = g.lanes();
    let tiles = lanes / LANES;
    let out = g.output();
    let (ow, oc) = (out.w, out.c);
    let oc_lanes = oc.next_multiple_of(LANES);
    let data_entries = (iw * tiles).div_ceil(2);
    let plane_grains = 16usize >> tiles.ilog2();
    let feature_grains = plane_grains.min(64usize.div_ceil(data_entries)).max(k.div_ceil(2) + 1);
    if at.input_plane_bytes != (ih * iw * LANES).next_multiple_of(64) {
        return Err(Error::Unsupported(format!("conv {g:?}: input plane stride {} is not the dense stride of its height", at.input_plane_bytes)));
    }
    let (rq, relu) = (q.requant, q.relu);
    let mut result = Vec::new();
    for band in bands(g)? {
        let tih = band.in_rows.1 - band.in_rows.0;
        let toh = band.out_rows.1 - band.out_rows.0;
        let scan_flags = if tih * iw * tiles <= 2048 { if k == 1 { 4 } else { 8 } } else { 0 };
        let input = at.input + (band.in_rows.0 * iw * LANES) as u32;
        let output = at.output + (band.out_rows.0 * ow * LANES) as u32;
        let mut p = Program::default();
        // open-rknpu's dense-convolution base values for fields its emitter
        // does not derive from the geometry.
        p.set(CNA_CONV_CON1, 0)
            .set(CNA_UNDECODED_101C, 0)
            .set(CNA_CVT_CON0, 9)
            .set(CNA_CVT_CON1, 0x10001)
            .set(CNA_CVT_CON2, 0x10001)
            .set(CNA_CVT_CON3, 0)
            .set(CNA_CVT_CON4, 0)
            .set(CNA_UNDECODED_108C, 0)
            .set(CORE_CLIP_TRUNCATE, 0);
        p.set(CNA_CONV_CON2, ((feature_grains << 4) | scan_flags) as u32)
            .set(CNA_CONV_CON3, ((s << 3) | s) as u32)
            .set(CNA_DATA_SIZE0, ((iw << 16) | tih) as u32)
            .set(CNA_DATA_SIZE1, (((ic - 1) << 16) | lanes) as u32)
            .set(CNA_DATA_SIZE2, ow as u32)
            .set(CNA_DATA_SIZE3, (toh * ow) as u32)
            .set(CNA_WEIGHT_SIZE0, g.weight_bytes() as u32)
            .set(CNA_WEIGHT_SIZE1, (lanes * k * k) as u32)
            .set(CNA_WEIGHT_SIZE2, ((k << 24) | (k << 16) | oc) as u32)
            .set(CNA_DATA_ENTRIES_HI, (data_entries << 16) as u32)
            .set(CNA_CBUF_CON1, ((iw << 16) | data_entries) as u32)
            .set(CNA_INPUT_EXTENT, ((tih * iw * tiles).div_ceil(2048) * 0x0400_0000) as u32)
            .set(CNA_PAD_CON0, ((g.pad[1] << 8) | band.pad_top) as u32)
            .set(CNA_FEATURE_DATA_ADDR, input)
            .set(CNA_DMA_CON1, (iw * 4usize.min(tih.div_ceil(4) * 2)) as u32)
            .set(CNA_DMA_CON2, ((ih * iw).div_ceil(4) * 4) as u32)
            .set(CNA_FC_DATA_SIZE0, ((iw << 16) | tih) as u32)
            .set(CNA_FC_DATA_SIZE1, lanes as u32)
            .set(CNA_DCOMP_ADDR0, at.weights)
            // The input zero point, as a 16-bit two's-complement value; it is
            // also what the CNA pads the border with.
            .set(CNA_PAD_CON1, u32::from(i16::from(input_zero_point) as u16))
            .set(CNA_WEIGHT_STRIDE, (8 * k * k * tiles) as u32)
            .set(CNA_FEATURE_WIDTH, (((iw - 1) << 16) | (iw - 1)) as u32);
        p.set(CORE_DATAOUT_SIZE_0, (((toh - 1) << 16) | (ow - 1)) as u32).set(CORE_DATAOUT_SIZE_1, (oc_lanes - 1) as u32);
        p.set(DPU_DST_BASE_ADDR, output)
            .set(DPU_DST_SURF_STRIDE, at.output_plane_bytes as u32)
            .set(DPU_DATA_CUBE_WIDTH, (ow - 1) as u32)
            .set(DPU_DATA_CUBE_HEIGHT, (toh - 1) as u32)
            .set(DPU_DATA_CUBE_CHANNEL, (((oc - 1) << 16) | (oc_lanes - 1)) as u32)
            .set(DPU_WDMA_SIZE_0, ((((oc - 1) / 16) << 16) | 3) as u32)
            .set(DPU_WDMA_SIZE_1, (((toh - 1) << 16) | (ow - 1)) as u32)
            .set(DPU_SURFACE_ADD, at.output_plane_bytes as u32)
            .set(DPU_OUT_CVT_OFFSET, i32::from(rq.output_zero_point) as u32)
            .set(DPU_OUT_CVT_SCALE, u32::from(rq.multiplier))
            .set(DPU_OUT_CVT_SHIFT, u32::from(rq.shift))
            .set(DPU_BN_CFG, if relu { 0x12 } else { 0x13 })
            .set(DPU_BN_RELUX_CMP_VALUE, if relu { 0 } else { 0x8000_0000 })
            .set(DPU_RELUX_CMP_VALUE_2, if relu { 0 } else { 0x8000_0000 });
        p.set(RDMA_DATA_CUBE_WIDTH, (ow - 1) as u32)
            .set(RDMA_DATA_CUBE_HEIGHT, (toh - 1) as u32)
            .set(RDMA_DATA_CUBE_CHANNEL, (oc_lanes - 1) as u32)
            .set(RDMA_BS_BASE_ADDR, at.bias);
        result.push(p);
    }
    Ok(result)
}

/// Write the weight table into `dst` ([`ConvGeometry::weight_bytes`] long).
///
/// Output channels go in blocks of 16; inside a block the input lanes are
/// grouped in parts of 32 (the last part may hold 16), and a part is laid out
/// `[tap][output lane][input lane]`. Lanes past the real input channels hold
/// the channel's weight zero point, so they contribute nothing.
pub fn pack_weights(g: &ConvGeometry, q: &ConvQuant, dst: &mut [u8]) {
    let (oc, ic, k) = (g.out_channels, g.input.c, g.kernel);
    let lanes = g.lanes();
    let taps = k * k;
    for o in 0..oc {
        let block = o / 16;
        let lane = o % 16;
        let block_oc = 16.min(oc - block * 16);
        let block_base = block * 16 * taps * lanes;
        let zp = q.weight_zero_points[o] as u8;
        for t in 0..taps {
            let (ky, kx) = (t / k, t % k);
            for c in 0..lanes {
                let (part, within) = (c / 32, c % 32);
                let part_size = 32.min(lanes - part * 32);
                let part_base: usize = (0..part).map(|p| taps * 32.min(lanes - p * 32) * block_oc).sum();
                let at = block_base + part_base + t * part_size * block_oc + lane * part_size + within;
                dst[at] = if c < ic { q.weights[((o * ic + c) * k + ky) * k + kx] as u8 } else { zp };
            }
        }
    }
}

/// Write the bias/conversion block into `dst` ([`ConvGeometry::bias_bytes`]
/// long): per four channels, four int32 biases, four int16 negated weight zero
/// points and four uint16 Q14 channel factors.
pub fn pack_bias(q: &ConvQuant, dst: &mut [u8]) {
    dst[..q.out_channels.div_ceil(4) * 32].fill(0);
    for o in 0..q.out_channels {
        let block = (o / 4) * 32;
        let lane = o % 4;
        dst[block + lane * 4..block + lane * 4 + 4].copy_from_slice(&q.bias[o].to_le_bytes());
        dst[block + 16 + lane * 2..block + 18 + lane * 2].copy_from_slice(&(-i16::from(q.weight_zero_points[o])).to_le_bytes());
        dst[block + 24 + lane * 2..block + 26 + lane * 2].copy_from_slice(&q.channel_multipliers[o].to_le_bytes());
    }
}

/// The exact integer result the NPU computes for this convolution, HWC in and
/// out. The border is padded with the input zero point, as the CNA does.
pub fn reference(g: &ConvGeometry, q: &ConvQuant, input_zero_point: i8, input: &[i8]) -> Vec<i8> {
    let out = g.output();
    let (ih, iw, ic, k) = (g.input.h as isize, g.input.w as isize, g.input.c, g.kernel);
    let [pt, pl, _, _] = g.pad;
    let mut result = vec![0i8; out.h * out.w * out.c];
    let mut patch = vec![0i8; ic * k * k];
    for oy in 0..out.h {
        for ox in 0..out.w {
            for c in 0..ic {
                for ky in 0..k {
                    for kx in 0..k {
                        let y = (oy * g.stride + ky) as isize - pt as isize;
                        let x = (ox * g.stride + kx) as isize - pl as isize;
                        patch[(c * k + ky) * k + kx] = if (0..ih).contains(&y) && (0..iw).contains(&x) { input[((y * iw + x) as usize) * ic + c] } else { input_zero_point };
                    }
                }
            }
            for o in 0..out.c {
                let acc = q.accumulate(o, patch.iter().copied());
                result[(oy * out.w + ox) * out.c + o] = q.requant.apply(acc, q.channel_multipliers[o]);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(h: usize, w: usize, c: usize, oc: usize, k: usize, s: usize) -> ConvGeometry {
        let p = k / 2;
        ConvGeometry { input: Shape::new(h, w, c), out_channels: oc, kernel: k, stride: s, pad: [p; 4] }
    }

    #[test]
    fn small_layers_are_one_band() {
        assert_eq!(bands(&geometry(8, 8, 16, 16, 3, 1)).unwrap().len(), 1);
    }

    #[test]
    fn large_layers_tile_into_aligned_bands_that_cover_the_output() {
        for g in [geometry(160, 160, 32, 64, 3, 2), geometry(80, 80, 64, 64, 3, 1), geometry(320, 320, 3, 32, 3, 2), geometry(20, 20, 256, 256, 3, 1)] {
            let b = bands(&g).unwrap();
            let out = g.output();
            assert!(b.len() > 1, "{g:?}");
            assert_eq!(b.first().unwrap().out_rows.0, 0);
            assert_eq!(b.last().unwrap().out_rows.1, out.h);
            for pair in b.windows(2) {
                assert_eq!(pair[0].out_rows.1, pair[1].out_rows.0);
            }
            for band in &b {
                assert!((band.in_rows.1 - band.in_rows.0) * g.input.w * g.lanes() / LANES <= INPUT_ATOMS);
                assert_eq!((band.in_rows.0 * g.input.w) % 4, 0);
                assert_eq!((band.out_rows.0 * out.w) % 4, 0);
            }
        }
    }

    #[test]
    fn weight_table_is_a_permutation_of_the_padded_weights() {
        let g = geometry(4, 4, 40, 20, 3, 1);
        let q = ConvQuant {
            out_channels: 20,
            in_channels: 40,
            kernel: 3,
            weights: (0..20 * 40 * 9).map(|i| (i % 200) as i8).collect(),
            weight_zero_points: vec![-1; 20],
            bias: vec![0; 20],
            channel_multipliers: vec![16384; 20],
            requant: crate::quant::Requant { multiplier: 1, shift: 0, output_zero_point: 0 },
            relu: false,
        };
        let mut dst = vec![0u8; g.weight_bytes()];
        pack_weights(&g, &q, &mut dst);
        let mut want: Vec<u8> = q.weights.iter().map(|&v| v as u8).collect();
        want.extend(std::iter::repeat_n(0xffu8, 20 * (48 - 40) * 9));
        let mut got = dst.clone();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }
}
