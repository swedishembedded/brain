// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The NPU's feature-map layout ("native16").
//!
//! Channels are split into planes of 16 lanes. A plane stores the image row by
//! row, one 16-byte atom per pixel (lane `c % 16` of pixel `(y, x)` at
//! `(y * w + x) * 16 + c % 16`), and planes follow each other at a stride
//! rounded up to 64 bytes. Because planes are independent, concatenating
//! tensors whose channel counts are multiples of 16 along the channel axis is
//! just placing them in adjacent planes.

pub const LANES: usize = 16;
/// Every engine base address is 64-byte aligned (four atoms).
pub const ADDRESS_ALIGN: usize = 64;

/// A feature map's geometry, height x width x channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Shape {
    pub const fn new(h: usize, w: usize, c: usize) -> Shape {
        Shape { h, w, c }
    }

    pub fn planes(&self) -> usize {
        self.c.div_ceil(LANES)
    }

    /// Stride between planes in bytes.
    pub fn plane_bytes(&self) -> usize {
        (self.h * self.w * LANES).next_multiple_of(ADDRESS_ALIGN)
    }

    pub fn bytes(&self) -> usize {
        self.plane_bytes() * self.planes()
    }

    /// Byte offset of channel `c` of pixel `(y, x)`.
    pub fn offset(&self, y: usize, x: usize, c: usize) -> usize {
        (c / LANES) * self.plane_bytes() + (y * self.w + x) * LANES + c % LANES
    }

    /// Pack an HWC int8 tensor into `dst` (at least [`Shape::bytes`] long).
    /// Lanes past the last channel are zeroed.
    pub fn pack(&self, hwc: &[i8], dst: &mut [u8]) {
        assert_eq!(hwc.len(), self.h * self.w * self.c, "pack: HWC length does not match {self:?}");
        dst[..self.bytes()].fill(0);
        for (pixel, values) in hwc.chunks(self.c).enumerate() {
            for (c, &v) in values.iter().enumerate() {
                dst[(c / LANES) * self.plane_bytes() + pixel * LANES + c % LANES] = v as u8;
            }
        }
    }

    /// Unpack into an HWC int8 tensor.
    pub fn unpack(&self, src: &[u8]) -> Vec<i8> {
        let mut out = vec![0i8; self.h * self.w * self.c];
        for (pixel, values) in out.chunks_mut(self.c).enumerate() {
            for (c, v) in values.iter_mut().enumerate() {
                *v = src[(c / LANES) * self.plane_bytes() + pixel * LANES + c % LANES] as i8;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips_across_planes() {
        let s = Shape::new(3, 5, 35);
        let hwc: Vec<i8> = (0..s.h * s.w * s.c).map(|i| (i % 251) as i8).collect();
        let mut buf = vec![0xffu8; s.bytes()];
        s.pack(&hwc, &mut buf);
        assert_eq!(s.unpack(&buf), hwc);
        // Lane 3 of plane 2 holds channel 35's neighbour slot: zero padding.
        assert_eq!(buf[2 * s.plane_bytes() + 3], 0);
        assert_eq!(s.plane_bytes() % ADDRESS_ALIGN, 0);
    }
}
