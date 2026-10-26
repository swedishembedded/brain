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
//! | E2M1 with E4M3 per 16 + f32 per tensor, E2M1 / FP8 with E8M0 per 32 | `Micro` (NVFP4, MXFP4, MXFP8; scale bytes kept as bytes) |

use checkpoint::packed::{Codebook, Element, FloatFormat, QuantEncoding, ScaleGranularity, TensorDescriptor, TensorSink, Tile};
use gpu_core::{DeviceBuffer, Gpu};

use super::{DeviceIo, MicroScaled, Ops, Weight, FP8_BLOCK};
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
    Micro(MicroScaled),
}

/// The tier of `enc`, or why none exists.
fn tier_of(enc: &QuantEncoding) -> Result<Tier, String> {
    let level = |i: usize| enc.scales.get(i).map(|s| (s.granularity, s.format));
    let unsupported = || {
        format!(
            "no resident Weight tier holds the encoding {enc:?}; supported: f32, bf16, f16, int8 and NF4 (f32 scale per 32), FP8 E4M3/E5M2 (2-D tile scales), \
             NVFP4, MXFP4 and MXFP8 (byte scales per 16/32)"
        )
    };
    let f32_per_32 = Some((ScaleGranularity::Block(32), FloatFormat::F32));
    let e8m0_per_32 = Some((ScaleGranularity::Block(32), FloatFormat::E8M0));
    match (&enc.element, enc.scales.len()) {
        (Element::Float(FloatFormat::F32), 0) => Ok(Tier::F32),
        (Element::Float(FloatFormat::Bf16), 0) => Ok(Tier::Bf16),
        (Element::Float(FloatFormat::F16), 0) => Ok(Tier::F16),
        (Element::Int { bits: 8, signed: true }, 1) if level(0) == f32_per_32 => Ok(Tier::I8),
        (Element::Codebook(Codebook::Nf4), 1) if level(0) == f32_per_32 => Ok(Tier::Nf4),
        (Element::Float(f @ (FloatFormat::E4M3Fn | FloatFormat::E5M2)), 1) => match level(0) {
            // One scale for the whole tensor (static per-tensor FP8): held as that scale on every tile of the grid.
            Some((ScaleGranularity::Tensor, FloatFormat::F32 | FloatFormat::Bf16)) => {
                Ok(Tier::Fp8(if *f == FloatFormat::E4M3Fn { Fp8Format::E4M3 } else { Fp8Format::E5M2 }, FP8_BLOCK))
            }
            Some((ScaleGranularity::Tile2d { rows, cols }, FloatFormat::Bf16 | FloatFormat::F32)) if rows == cols => {
                Ok(Tier::Fp8(if *f == FloatFormat::E4M3Fn { Fp8Format::E4M3 } else { Fp8Format::E5M2 }, rows))
            }
            _ if level(0) == e8m0_per_32 => Ok(Tier::Micro(if *f == FloatFormat::E4M3Fn { MicroScaled::Mxfp8E4m3 } else { MicroScaled::Mxfp8E5m2 })),
            _ => Err(unsupported()),
        },
        (Element::Float(FloatFormat::E2M1), 1) if level(0) == e8m0_per_32 => Ok(Tier::Micro(MicroScaled::Mxfp4)),
        (Element::Float(FloatFormat::E2M1), 2)
            if level(0) == Some((ScaleGranularity::Tensor, FloatFormat::F32)) && level(1) == Some((ScaleGranularity::Block(16), FloatFormat::E4M3Fn)) =>
        {
            Ok(Tier::Micro(MicroScaled::Nvfp4))
        }
        _ => Err(unsupported()),
    }
}

/// An in-order byte stream into a device buffer of whole words: tiles arrive ascending and contiguous, so a
/// trailing partial word is carried to the next tile and only whole words are written. The buffer is allocated
/// uninitialised: every word is written before the stream finishes.
struct Stream {
    buf: DeviceBuffer,
    written: u64,
    carry: Vec<u8>,
    total_bytes: u64,
}

impl Stream {
    fn new(gpu: &Gpu, total_bytes: u64) -> Stream {
        Stream { buf: gpu.storage_uninit(total_bytes.div_ceil(4)), written: 0, carry: Vec::new(), total_bytes }
    }

    /// A stream filling bytes `start..start + len` of the existing buffer `buf` (`start` a whole word).
    fn within(buf: &DeviceBuffer, start: u64, len: u64) -> Stream {
        debug_assert!(start.is_multiple_of(4), "a bank stream starts on a word");
        Stream { buf: buf.clone(), written: start, carry: Vec::new(), total_bytes: start + len }
    }

    /// Where the next byte goes.
    fn position(&self) -> u64 {
        self.written + self.carry.len() as u64
    }

    fn push(&mut self, gpu: &Gpu, mut bytes: &[u8]) {
        if !self.carry.is_empty() {
            let take = (4 - self.carry.len()).min(bytes.len());
            self.carry.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.carry.len() < 4 {
                return;
            }
            gpu.write_bytes_at(&self.buf, self.written / 4, &self.carry);
            self.written += 4;
            self.carry.clear();
        }
        let whole = bytes.len() / 4 * 4;
        if whole > 0 {
            gpu.write_bytes_at(&self.buf, self.written / 4, &bytes[..whole]);
            self.written += whole as u64;
        }
        self.carry.extend_from_slice(&bytes[whole..]);
    }

    /// Every byte arrived: flush a final partial word (zero padded, never read).
    fn finish(&mut self, gpu: &Gpu, what: &str, name: &str) -> Result<DeviceBuffer, String> {
        if self.position() != self.total_bytes {
            return Err(format!("tensor '{name}': the load wrote {} of {} {what} bytes", self.position(), self.total_bytes));
        }
        if !self.carry.is_empty() {
            let pad = vec![0u8; 4 - self.carry.len()];
            self.push(gpu, &pad);
        }
        Ok(self.buf.clone())
    }
}

/// The scale plane of a tier.
enum Scales {
    /// f32 words, one scale per entry (bf16 scale bytes are widened on the way in).
    F32 { buf: DeviceBuffer, from_bf16: bool },
    /// One byte per block (E4M3 or E8M0 codes), kept as bytes.
    Bytes(Stream),
    /// One scale for the whole tensor (an FP8 weight with a per-tensor scale): the resident tier keeps a scale per
    /// tile, so the value is replicated over the grid when the load finishes.
    Uniform(f32),
}

/// A scale value of `bytes` in `format`, read as f32.
fn scale_value(format: FloatFormat, bytes: &[u8]) -> Result<f32, String> {
    match (format, bytes) {
        (FloatFormat::F32, [a, b, c, d, ..]) => Ok(f32::from_le_bytes([*a, *b, *c, *d])),
        (FloatFormat::Bf16, [a, b, ..]) => Ok(f32::from_bits((u16::from_le_bytes([*a, *b]) as u32) << 16)),
        _ => Err(format!("a {format:?} tensor scale needs its bytes")),
    }
}

struct Pending {
    tier: Tier,
    n: u32,
    k: u32,
    payload: Stream,
    scales: Option<Scales>,
    /// NVFP4's per-tensor scale, read from the first tile.
    global: f32,
}

/// Uploads a tiled load into a resident [`Weight`]. One sink loads one tensor at a time; take the result
/// with [`Self::take_weight`] after the load finished.
pub struct DeviceWeightSink<'a> {
    ops: &'a dyn DeviceIo,
    pending: Option<Pending>,
    done: Option<Weight>,
}

impl<'a> DeviceWeightSink<'a> {
    /// A sink uploading through `ops`' device.
    pub fn new(ops: &'a dyn DeviceIo) -> Self {
        DeviceWeightSink { ops, pending: None, done: None }
    }

    /// The weight the last finished load produced.
    pub fn take_weight(&mut self) -> Option<Weight> {
        self.done.take()
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
        let group = match tier {
            Tier::F32 => 1,
            Tier::Bf16 | Tier::F16 => 2,
            Tier::Fp8(..) => 4,
            Tier::I8 | Tier::Nf4 | Tier::Micro(_) => 32,
        };
        let group = if tier == Tier::Micro(MicroScaled::Nvfp4) { 16 } else { group };
        if k % group != 0 {
            return Err(format!("tensor '{}': K = {k} must be a multiple of {group} for this tier", dst.name));
        }
        let gpu = self.ops.upload_gpu();
        let scales = match (tier, dst.encoding.scales.last()) {
            (_, None) => None,
            (Tier::Micro(_), Some(_)) => Some(Scales::Bytes(Stream::new(gpu, dst.scale_bytes(dst.encoding.scales.len() - 1)))),
            (Tier::Fp8(..), Some(level)) if level.granularity == ScaleGranularity::Tensor => Some(Scales::Uniform(1.0)),
            (_, Some(level)) => Some(Scales::F32 { buf: gpu.storage_uninit(dst.scale_numel(0)), from_bf16: level.format == FloatFormat::Bf16 }),
        };
        self.done = None;
        self.pending = Some(Pending { tier, n: n as u32, k: k as u32, payload: Stream::new(gpu, dst.payload_bytes()), scales, global: 1.0 });
        Ok(())
    }

    fn write_tile(&mut self, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        let mut p = self.pending.take().ok_or("DeviceWeightSink::write_tile before begin")?;
        let gpu = self.ops.upload_gpu();
        let range = dst.payload_range(&tile.elems);
        if range.start != p.payload.position() {
            return Err(format!("tensor '{}': tiles must arrive in order (tile starts at byte {}, expected {})", dst.name, range.start, p.payload.position()));
        }
        p.payload.push(gpu, tile.payload);
        match &mut p.scales {
            Some(Scales::Uniform(value)) => {
                let level = &dst.encoding.scales[0];
                *value = scale_value(level.format, tile.scales.first().ok_or("a scaled tile carries its scales")?)?;
            }
            Some(Scales::F32 { buf, from_bf16 }) => {
                let bytes = tile.scales.first().ok_or("a scaled tile carries its scales")?;
                let at = dst.scale_range(0, &tile.elems).start;
                if *from_bf16 {
                    let words: Vec<u32> = bytes.as_chunks::<2>().0.iter().map(|c| (u16::from_le_bytes(*c) as u32) << 16).collect();
                    gpu.write_at(buf, at, &words);
                } else {
                    gpu.write_bytes_at(buf, at, bytes);
                }
            }
            Some(Scales::Bytes(stream)) => {
                let last = dst.encoding.scales.len() - 1;
                if last == 1 {
                    let g = tile.scales.first().and_then(|b| b.as_chunks::<4>().0.first().copied()).ok_or("an NVFP4 tile carries the tensor scale")?;
                    p.global = f32::from_le_bytes(g);
                }
                let bytes = tile.scales.get(last).ok_or("a scaled tile carries its scales")?;
                let at = dst.scale_range(last, &tile.elems).start;
                if at != stream.position() {
                    return Err(format!("tensor '{}': scale tiles must arrive in order (starts at {at}, expected {})", dst.name, stream.position()));
                }
                stream.push(gpu, bytes);
            }
            None => {}
        }
        self.pending = Some(p);
        Ok(())
    }

    fn finish(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        let mut p = self.pending.take().ok_or("DeviceWeightSink::finish before begin")?;
        let gpu = self.ops.upload_gpu();
        let payload = p.payload.finish(gpu, "payload", &dst.name)?;
        let (n, k) = (p.n, p.k);
        let scales = match p.scales {
            Some(Scales::F32 { buf, .. }) => Some(buf),
            Some(Scales::Bytes(mut stream)) => Some(stream.finish(gpu, "scale", &dst.name)?),
            Some(Scales::Uniform(value)) => {
                let tiles = (n as u64).div_ceil(FP8_BLOCK as u64) * (k as u64).div_ceil(FP8_BLOCK as u64);
                Some(gpu.storage_init(&format!("{}.scale_grid", dst.name), &vec![value; tiles as usize]))
            }
            None => None,
        };
        // The weight is complete: whatever handle uses it next waits for the writes (a no-op on the compute handle).
        let written: Vec<&DeviceBuffer> = std::iter::once(&payload).chain(scales.as_ref()).collect();
        gpu.flush_writes(&written);
        self.done = Some(match (p.tier, scales) {
            (Tier::F32, _) => Weight::F32 { w: payload, n, k },
            (Tier::Bf16, _) => Weight::BF16 { w: payload, n, k },
            (Tier::F16, _) => Weight::F16 { w: payload, n, k },
            (Tier::I8, Some(s)) => Weight::I8 { w: payload, s, n, k },
            (Tier::Nf4, Some(s)) => Weight::NF4 { w: payload, s, n, k },
            (Tier::Fp8(fmt, block), Some(s)) => Weight::FP8 { w: payload, s, n, k, fmt, block },
            (Tier::Micro(kind), Some(s)) => Weight::Micro { w: payload, s, n, k, kind, global: p.global },
            (_, None) => unreachable!("a scaled tier always allocates its scale plane in begin"),
        });
        Ok(())
    }

    fn abort(&mut self, _dst: &TensorDescriptor) {
        self.pending = None;
        self.done = None;
    }
}

/// Overwrite the elements of `tile` in the resident `weight` with a freshly encoded tile: the update half of
/// training a quantized value in place (the load half is [`DeviceWeightSink`]). `dst` is the descriptor the
/// weight was loaded from; the tile must start and end on a payload word boundary (a multiple of the descriptor's
/// tile quantum satisfies this for every tier but the byte-scaled microscaling family, which this refuses: its
/// per-tensor scale is not a buffer).
pub fn rewrite_tile(ops: &dyn DeviceIo, weight: &Weight, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
    let gpu = ops.gpu();
    let (payload, scales) = match weight {
        Weight::F32 { w, .. } | Weight::BF16 { w, .. } | Weight::F16 { w, .. } => (w, None),
        Weight::I8 { w, s, .. } | Weight::NF4 { w, s, .. } | Weight::FP8 { w, s, .. } => (w, Some(s)),
        Weight::Q4 { .. } | Weight::KQuant { .. } | Weight::Micro { .. } => {
            return Err(format!("tensor '{}': a {:?} weight cannot be rewritten in place", dst.name, weight.dtype()));
        }
    };
    let range = dst.payload_range(&tile.elems);
    if !range.start.is_multiple_of(4) || !tile.payload.len().is_multiple_of(4) {
        return Err(format!("tensor '{}': tile {:?} is not word aligned ({} payload bytes at byte {})", dst.name, tile.elems, tile.payload.len(), range.start));
    }
    gpu.write_bytes_at(payload, range.start / 4, tile.payload);
    if let Some(buf) = scales {
        let level = dst.encoding.scales.last().ok_or_else(|| format!("tensor '{}': a scaled weight needs a scale level", dst.name))?;
        let bytes = tile.scales.first().ok_or("a scaled tile carries its scales")?;
        let at = dst.scale_range(0, &tile.elems).start;
        let words: Vec<u32> = if level.format == FloatFormat::Bf16 {
            bytes.as_chunks::<2>().0.iter().map(|c| (u16::from_le_bytes(*c) as u32) << 16).collect()
        } else {
            bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
        };
        gpu.write_at(buf, at, &words);
    }
    Ok(())
}


/// A bank of `blocks` same-shape microscaled matrices (the routed experts of one MoE projection) in ONE resident
/// buffer set: block `b`'s `[n, k]` matrix is rows `b * n .. (b + 1) * n` of the payload and scale planes, and its
/// per-tensor scale is `global[b]`. One buffer per projection is what lets a single gather kernel serve every expert a
/// token routes to; the bytes are the checkpoint's own (Preserve), one tile of host staging at a time.
pub struct MicroBank {
    /// Element codes of every block, back to back, in the tier's packing.
    pub w: DeviceBuffer,
    /// Scale bytes of every block, back to back, four to a word.
    pub s: DeviceBuffer,
    /// `[blocks]` f32 per-tensor scale of each block.
    pub global: DeviceBuffer,
    pub n: u32,
    pub k: u32,
    pub blocks: u32,
    pub kind: MicroScaled,
}

/// Loads the matrices of a [`MicroBank`] one block at a time: [`Self::start_block`] then a tiled load into this sink.
struct MicroBankSink<'a> {
    ops: &'a Ops,
    w: DeviceBuffer,
    s: DeviceBuffer,
    n: u32,
    k: u32,
    block: u32,
    global: Vec<f32>,
    payload: Option<Stream>,
    scales: Option<Stream>,
}

impl MicroBankSink<'_> {
    fn bytes(&self, dst: &TensorDescriptor) -> (u64, u64) {
        (dst.payload_bytes(), dst.scale_bytes(dst.encoding.scales.len() - 1))
    }
}

impl TensorSink for MicroBankSink<'_> {
    fn begin(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        let tier = tier_of(&dst.encoding).map_err(|e| format!("tensor '{}': {e}", dst.name))?;
        if tier != Tier::Micro(MicroScaled::Nvfp4) || dst.shape != [self.n as u64, self.k as u64] {
            return Err(format!("tensor '{}': a bank of NVFP4 [{}, {}] matrices cannot take {:?} of shape {:?}", dst.name, self.n, self.k, dst.encoding, dst.shape));
        }
        let (pb, sb) = self.bytes(dst);
        self.payload = Some(Stream::within(&self.w, self.block as u64 * pb, pb));
        self.scales = Some(Stream::within(&self.s, self.block as u64 * sb, sb));
        Ok(())
    }

    fn write_tile(&mut self, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        let gpu = self.ops.upload_gpu();
        let (pb, sb) = self.bytes(dst);
        let (payload, scales) = self.payload.as_mut().zip(self.scales.as_mut()).ok_or("MicroBankSink::write_tile before begin")?;
        let range = dst.payload_range(&tile.elems);
        if self.block as u64 * pb + range.start != payload.position() {
            return Err(format!("tensor '{}': tiles must arrive in order", dst.name));
        }
        payload.push(gpu, tile.payload);
        let g = tile.scales.first().and_then(|b| b.as_chunks::<4>().0.first().copied()).ok_or("an NVFP4 tile carries the tensor scale")?;
        self.global[self.block as usize] = f32::from_le_bytes(g);
        let bytes = tile.scales.get(1).ok_or("an NVFP4 tile carries its block scales")?;
        let at = self.block as u64 * sb + dst.scale_range(1, &tile.elems).start;
        if at != scales.position() {
            return Err(format!("tensor '{}': scale tiles must arrive in order", dst.name));
        }
        scales.push(gpu, bytes);
        Ok(())
    }

    fn finish(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        let gpu = self.ops.upload_gpu();
        let (mut payload, mut scales) = self.payload.take().zip(self.scales.take()).ok_or("MicroBankSink::finish before begin")?;
        payload.finish(gpu, "payload", &dst.name)?;
        scales.finish(gpu, "scale", &dst.name)?;
        Ok(())
    }

    fn abort(&mut self, _dst: &TensorDescriptor) {
        self.payload = None;
        self.scales = None;
    }
}

/// Upload the NVFP4 matrices `names` (each `[n, k]`, in block order) of `src` as one [`MicroBank`], in the encoding
/// they are stored in. `staging` bounds the host memory a tile may take. An encoding that is not NVFP4, a shape that
/// differs, or a source without a packed view of a name is an error naming it: nothing is widened.
pub fn upload_micro_bank(ops: &Ops, src: &dyn checkpoint::TensorSource, names: &[String], n: usize, k: usize, staging: &checkpoint::packed::BoundedWorkspace) -> Result<MicroBank, String> {
    use checkpoint::packed::{LoadConfig, LoadPolicy, LoadRequest};
    let blocks = names.len() as u32;
    let first = names.first().ok_or("an empty expert bank")?;
    let probe = src.packed_descriptor(first).ok_or_else(|| format!("{first}: the source has no packed view of this tensor"))?;
    let (pb, sb) = (probe.payload_bytes(), probe.scale_bytes(probe.encoding.scales.len().saturating_sub(1)));
    let gpu = ops.upload_gpu();
    let mut sink = MicroBankSink {
        ops,
        w: gpu.storage_uninit((pb * blocks as u64).div_ceil(4)),
        s: gpu.storage_uninit((sb * blocks as u64).div_ceil(4)),
        n: n as u32,
        k: k as u32,
        block: 0,
        global: vec![1.0; blocks as usize],
        payload: None,
        scales: None,
    };
    let config = LoadConfig::default();
    let req = LoadRequest { policy: &LoadPolicy::Preserve, workspace: staging, config: &config };
    for (b, name) in names.iter().enumerate() {
        sink.block = b as u32;
        match src.load_packed(&[name.as_str()], &req, &mut sink)? {
            Some(_) => {}
            None => return Err(format!("{name}: the source cannot serve this tensor packed")),
        }
    }
    let global = gpu.storage_init("micro_bank.global", &sink.global);
    Ok(MicroBank { w: sink.w, s: sink.s, global, n: n as u32, k: k as u32, blocks, kind: MicroScaled::Nvfp4 })
}
