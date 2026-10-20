// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of quantized
// checkpoints straight into device memory for its clients. If your team needs
// expertise in uploading packed weights tile by tile without ever expanding a
// tensor to full precision, you can procure our services by sending an email
// to info@swedishembedded.com.

//! [`DeviceWeightSink`]: the device end of a tiled checkpoint load. It receives the tiles a
//! `checkpoint::packed::Loader` produces in the destination encoding and writes the payload bytes and
//! scales straight into the buffers of the [`Weight`] tier that encoding is, so a quantized checkpoint
//! becomes a resident quantized weight with one tile of host staging and no full-precision copy
//! (invariant M of the quantized-training plan: Preserve and lossless Repack only; an encoding with no
//! tier is a clear error, never a silent widening).
//!
//! | destination encoding | tier |
//! |---|---|
//! | f32 / bf16 / f16 | `F32` / `BF16` / `F16` |
//! | signed int8, f32 scale per 32 | `I8` |
//! | NF4 codebook, f32 scale per 32 | `NF4` |
//! | FP8 E4M3 / E5M2, bf16 or f32 scale per `b x b` tile | `FP8` (scales widened to f32, exact) |

use checkpoint::packed::{Codebook, Element, FloatFormat, QuantEncoding, ScaleGranularity, TensorDescriptor, TensorSink, Tile};
use gpu_core::DeviceBuffer;

use super::{Ops, Weight};
use crate::fp8::Fp8Format;

/// The tier a destination encoding maps to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    F32,
    Bf16,
    F16,
    I8,
    Nf4,
    Fp8(Fp8Format, u32),
}

/// The tier of `enc`, or why none exists.
fn tier_of(enc: &QuantEncoding) -> Result<Tier, String> {
    let scale = |i: usize| enc.scales.get(i);
    let unsupported = || format!("no resident Weight tier holds the encoding {enc:?}; supported: f32, bf16, f16, int8 and NF4 (f32 scale per 32), FP8 E4M3/E5M2 (2-D tile scales)");
    match (&enc.element, enc.scales.len()) {
        (Element::Float(FloatFormat::F32), 0) => Ok(Tier::F32),
        (Element::Float(FloatFormat::Bf16), 0) => Ok(Tier::Bf16),
        (Element::Float(FloatFormat::F16), 0) => Ok(Tier::F16),
        (Element::Int { bits: 8, signed: true }, 1) if matches!((scale(0).map(|s| s.granularity), scale(0).map(|s| s.format)), (Some(ScaleGranularity::Block(32)), Some(FloatFormat::F32))) => Ok(Tier::I8),
        (Element::Codebook(Codebook::Nf4), 1) if matches!((scale(0).map(|s| s.granularity), scale(0).map(|s| s.format)), (Some(ScaleGranularity::Block(32)), Some(FloatFormat::F32))) => Ok(Tier::Nf4),
        (Element::Float(f @ (FloatFormat::E4M3Fn | FloatFormat::E5M2)), 1) => match scale(0).map(|s| (s.granularity, s.format)) {
            Some((ScaleGranularity::Tile2d { rows, cols }, FloatFormat::Bf16 | FloatFormat::F32)) if rows == cols => {
                Ok(Tier::Fp8(if *f == FloatFormat::E4M3Fn { Fp8Format::E4M3 } else { Fp8Format::E5M2 }, rows))
            }
            _ => Err(unsupported()),
        },
        _ => Err(unsupported()),
    }
}

struct Pending {
    tier: Tier,
    n: u32,
    k: u32,
    payload: DeviceBuffer,
    scales: Option<DeviceBuffer>,
    scale_is_bf16: bool,
    /// Payload bytes handed to the device so far.
    written: u64,
    /// The bytes of a trailing partial word, held until the next tile completes it.
    carry: Vec<u8>,
    payload_bytes: u64,
}

/// Uploads a tiled load into a resident [`Weight`]. One sink loads one tensor at a time; take the result
/// with [`Self::take_weight`] after the load finished.
pub struct DeviceWeightSink<'a> {
    ops: &'a Ops,
    pending: Option<Pending>,
    done: Option<Weight>,
}

impl<'a> DeviceWeightSink<'a> {
    /// A sink uploading through `ops`' device.
    pub fn new(ops: &'a Ops) -> Self {
        DeviceWeightSink { ops, pending: None, done: None }
    }

    /// The weight the last finished load produced.
    pub fn take_weight(&mut self) -> Option<Weight> {
        self.done.take()
    }

    fn flush_words(&self, p: &mut Pending, bytes: &[u8]) {
        let mut data = std::mem::take(&mut p.carry);
        data.extend_from_slice(bytes);
        let whole = data.len() / 4 * 4;
        if whole > 0 {
            let words: Vec<u32> = data[..whole].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            self.ops.gpu().write_at(&p.payload, p.written / 4, &words);
            p.written += whole as u64;
        }
        p.carry = data[whole..].to_vec();
    }
}

impl TensorSink for DeviceWeightSink<'_> {
    fn begin(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        let tier = tier_of(&dst.encoding).map_err(|e| format!("tensor '{}': {e}", dst.name))?;
        if dst.shape.len() != 2 {
            return Err(format!("tensor '{}': a Weight is a rank-2 [n, k] tensor, got shape {:?}", dst.name, dst.shape));
        }
        let (n, k) = (dst.shape[0], dst.shape[1]);
        if n > u32::MAX as u64 || k > u32::MAX as u64 {
            return Err(format!("tensor '{}': shape {:?} exceeds the 32-bit extents of a Weight", dst.name, dst.shape));
        }
        let multiple = match tier {
            Tier::F32 => 1,
            Tier::Bf16 | Tier::F16 => 2,
            Tier::I8 | Tier::Fp8(..) => 4,
            Tier::Nf4 => 8,
        };
        let group = if matches!(tier, Tier::I8 | Tier::Nf4) { 32 } else { multiple };
        if k % group != 0 {
            return Err(format!("tensor '{}': K = {k} must be a multiple of {group} for this tier", dst.name));
        }
        let gpu = self.ops.gpu();
        let payload_bytes = dst.payload_bytes();
        let scales = (!dst.encoding.scales.is_empty()).then(|| gpu.storage(dst.scale_numel(0)));
        self.done = None;
        self.pending = Some(Pending {
            tier,
            n: n as u32,
            k: k as u32,
            payload: gpu.storage(payload_bytes.div_ceil(4)),
            scales,
            scale_is_bf16: dst.encoding.scales.first().is_some_and(|s| s.format == FloatFormat::Bf16),
            written: 0,
            carry: Vec::new(),
            payload_bytes,
        });
        Ok(())
    }

    fn write_tile(&mut self, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        let mut p = self.pending.take().ok_or("DeviceWeightSink::write_tile before begin")?;
        let range = dst.payload_range(&tile.elems);
        if range.start != p.written + p.carry.len() as u64 {
            let at = p.written + p.carry.len() as u64;
            return Err(format!("tensor '{}': tiles must arrive in order (tile starts at byte {}, expected {at})", dst.name, range.start));
        }
        self.flush_words(&mut p, tile.payload);
        if let (Some(buf), Some(bytes)) = (&p.scales, tile.scales.first()) {
            let at = dst.scale_range(0, &tile.elems).start;
            let words: Vec<u32> = if p.scale_is_bf16 {
                bytes.as_chunks::<2>().0.iter().map(|c| (u16::from_le_bytes(*c) as u32) << 16).collect()
            } else {
                bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
            };
            self.ops.gpu().write_at(buf, at, &words);
        }
        self.pending = Some(p);
        Ok(())
    }

    fn finish(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        let mut p = self.pending.take().ok_or("DeviceWeightSink::finish before begin")?;
        if !p.carry.is_empty() {
            // The final partial word (an odd number of 16-bit elements): zero padded, never read.
            let pad = vec![0u8; 4 - p.carry.len()];
            self.flush_words(&mut p, &pad);
        }
        if p.written < p.payload_bytes.div_ceil(4) * 4 {
            return Err(format!("tensor '{}': the load wrote {} of {} payload bytes", dst.name, p.written, p.payload_bytes));
        }
        let (n, k) = (p.n, p.k);
        let scales = p.scales;
        self.done = Some(match (p.tier, scales) {
            (Tier::F32, _) => Weight::F32 { w: p.payload, n, k },
            (Tier::Bf16, _) => Weight::BF16 { w: p.payload, n, k },
            (Tier::F16, _) => Weight::F16 { w: p.payload, n, k },
            (Tier::I8, Some(s)) => Weight::I8 { w: p.payload, s, n, k },
            (Tier::Nf4, Some(s)) => Weight::NF4 { w: p.payload, s, n, k },
            (Tier::Fp8(fmt, block), Some(s)) => Weight::FP8 { w: p.payload, s, n, k, fmt, block },
            (_, None) => unreachable!("a scaled tier always allocates its scale plane in begin"),
        });
        Ok(())
    }

    fn abort(&mut self, _dst: &TensorDescriptor) {
        self.pending = None;
        self.done = None;
    }
}
