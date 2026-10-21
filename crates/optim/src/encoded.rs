// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in training and exporting quantized
// models without full-precision intermediates then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Parameter values stored in any registered quantized encoding, and the
//! streamed export of values into any other.
//!
//! [`EncodedValues`] is the value half of [`ParamTiles`] for a tensor kept in
//! a `checkpoint::packed::QuantEncoding` (bf16, FP8, int8, NF4, ...) - the
//! encoding comes from the codec registry, so a new format needs no change
//! here. Reading decodes a tile to f32; writing re-encodes one. With an
//! optimiser master copy this is quantization-aware training: the master
//! accumulates updates in high precision, the stored value is its quantization
//! (the straight-through estimator is the trainer applying the gradient taken
//! at the quantized value to the master). Without a master, small updates
//! vanish in a low-precision value, which is why the optimiser's
//! stochastic-rounding request is refused for encoded values rather than
//! silently run with nearest rounding.
//!
//! [`export_values`] converts values tile by tile to a target encoding and
//! hands them to a `TensorSink` (a file writer, a device weight sink): the
//! staging is a tile, never a tensor.

use std::io;
use std::ops::Range;

use checkpoint::packed::codec::EncodeCtx;
use checkpoint::packed::{CodecRegistry, ScaleGranularity, Tile, TileBuf, TensorDescriptor, TensorSink};

use crate::codec::Rounding;
use crate::storage::{Backing, BlockStateStorage, OptimizerStateStorage};
use crate::streamed::ParamTiles;

fn err(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

fn lcm(a: u64, b: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}

/// Smallest tile a writer may use for `d`: the descriptor's tile quantum, and
/// for 2-D tile scales a whole number of scale bands, so every scale value is
/// computed from all the elements it covers by exactly one tile.
fn write_quantum(d: &TensorDescriptor) -> u64 {
    let mut q = d.tile_quantum();
    for s in &d.encoding.scales {
        if let ScaleGranularity::Tile2d { rows, .. } = s.granularity {
            q = lcm(q, rows as u64 * d.shape.last().copied().unwrap_or(1));
        }
    }
    q
}

/// The tile size to use over `descs`: `requested` rounded up to a multiple of
/// every descriptor's write quantum.
pub fn aligned_tile(descs: &[TensorDescriptor], requested: usize) -> usize {
    descs.iter().fold(requested.max(1) as u64, |t, d| lcm(t, write_quantum(d))) as usize
}

/// Where each tensor's payload and scale sidecars live in the backing.
struct Placement {
    payload: u64,
    scales: Vec<u64>,
}

/// Values of every tensor, each in its own encoding, in one backing.
pub struct EncodedValues<B: Backing> {
    descs: Vec<TensorDescriptor>,
    at: Vec<Placement>,
    backing: B,
    registry: CodecRegistry,
    tile: usize,
    payload: Vec<u8>,
    scales: Vec<Vec<u8>>,
    f32s: Vec<f32>,
}

impl<B: Backing> EncodedValues<B> {
    /// Bytes of backing the descriptors need (payload plus scale sidecars).
    pub fn required_bytes(descs: &[TensorDescriptor]) -> u64 {
        descs.iter().map(|d| d.payload_bytes() + (0..d.encoding.scales.len()).map(|l| d.scale_bytes(l)).sum::<u64>()).sum()
    }

    /// `tile_elems` must be a multiple of every descriptor's tile quantum (see
    /// [`aligned_tile`]); every encoding must be decodable and encodable by `registry`.
    pub fn new(descs: Vec<TensorDescriptor>, registry: CodecRegistry, backing: B, tile_elems: usize) -> io::Result<Self> {
        let mut off = 0u64;
        let mut at = Vec::with_capacity(descs.len());
        for d in &descs {
            d.validate().map_err(err)?;
            registry.decoder(&d.encoding).map_err(err)?;
            registry.encoder(&d.encoding).map_err(err)?;
            if !(tile_elems as u64).is_multiple_of(write_quantum(d)) {
                return Err(err(format!("{}: tile {tile_elems} is not a multiple of the write quantum {}", d.name, write_quantum(d))));
            }
            let payload = off;
            off += d.payload_bytes();
            let mut scales = Vec::new();
            for l in 0..d.encoding.scales.len() {
                scales.push(off);
                off += d.scale_bytes(l);
            }
            at.push(Placement { payload, scales });
        }
        if backing.len() != off {
            return Err(err(format!("backing is {} bytes, descriptors need {off}", backing.len())));
        }
        Ok(EncodedValues { descs, at, backing, registry, tile: tile_elems, payload: Vec::new(), scales: Vec::new(), f32s: Vec::with_capacity(tile_elems) })
    }

    fn check(&self, ti: usize, start: usize, n: usize) -> io::Result<(TensorDescriptor, Range<u64>)> {
        let d = self.descs.get(ti).ok_or_else(|| err(format!("no tensor {ti}")))?;
        let e = start as u64..(start + n) as u64;
        if e.end > d.numel() || n > self.tile {
            return Err(err(format!("{}: tile {e:?} out of range ({} elements, max tile {})", d.name, d.numel(), self.tile)));
        }
        Ok((d.clone(), e))
    }

    /// The stored bytes of tile `[start, start + n)` of tensor `ti` as they are encoded, undecoded: what a
    /// device upload of the stored value copies word for word.
    pub fn read_encoded(&mut self, ti: usize, start: usize, n: usize, out: &mut TileBuf) -> io::Result<()> {
        let (d, e) = self.check(ti, start, n)?;
        let pr = d.payload_range(&e);
        out.payload.resize((pr.end - pr.start) as usize, 0);
        self.backing.read_at(self.at[ti].payload + pr.start, &mut out.payload)?;
        out.scales.resize(d.encoding.scales.len(), Vec::new());
        for l in 0..d.encoding.scales.len() {
            let w = d.encoding.scales[l].format.bits() as u64 / 8;
            let r = d.scale_range(l, &e);
            out.scales[l].resize(((r.end - r.start) * w) as usize, 0);
            self.backing.read_at(self.at[ti].scales[l] + r.start * w, &mut out.scales[l])?;
        }
        Ok(())
    }

    /// Decode tile `[start, start + out.len())` of tensor `ti`.
    pub fn read(&mut self, ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        let (d, e) = self.check(ti, start, out.len())?;
        let mut raw = TileBuf { payload: std::mem::take(&mut self.payload), scales: std::mem::take(&mut self.scales) };
        self.read_encoded(ti, start, out.len(), &mut raw)?;
        (self.payload, self.scales) = (raw.payload, raw.scales);
        let tile = Tile { elems: e, payload: &self.payload, scales: self.scales.iter().map(|s| s.as_slice()).collect() };
        self.f32s.clear();
        self.registry.decoder(&d.encoding).map_err(err)?.decode_tile(&d, &tile, &mut self.f32s).map_err(err)?;
        out.copy_from_slice(&self.f32s);
        Ok(())
    }

    /// Encode `vals` into tile `[start, start + vals.len())` of tensor `ti`.
    pub fn write(&mut self, ti: usize, start: usize, vals: &[f32]) -> io::Result<()> {
        let (d, e) = self.check(ti, start, vals.len())?;
        let mut buf = TileBuf::default();
        self.registry.encoder(&d.encoding).map_err(err)?.encode_tile(&d, &e, vals, &EncodeCtx::default(), &mut buf).map_err(err)?;
        let pr = d.payload_range(&e);
        self.backing.write_at(self.at[ti].payload + pr.start, &buf.payload)?;
        for (l, s) in buf.scales.iter().enumerate() {
            let w = d.encoding.scales[l].format.bits() as u64 / 8;
            let r = d.scale_range(l, &e);
            self.backing.write_at(self.at[ti].scales[l] + r.start * w, s)?;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.backing.sync()
    }
}

/// [`ParamTiles`] over encoded values and gradients in a [`BlockStateStorage`]
/// (one slot per tensor, see `streamed::tensor_layout`).
pub struct EncodedParams<V: Backing, G: Backing> {
    pub values: EncodedValues<V>,
    pub grads: BlockStateStorage<G>,
}

impl<V: Backing, G: Backing> ParamTiles for EncodedParams<V, G> {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.values.read(tensor, start, out)
    }

    fn write_value(&mut self, tensor: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()> {
        if rnd.stochastic {
            return Err(err(format!(
                "tensor {tensor}: stochastic rounding of codec-encoded values is not supported; keep a master copy and round to nearest"
            )));
        }
        self.values.write(tensor, start, vals)
    }

    fn read_grad(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.grads.read_tile(tensor, 0, start, out)
    }

    fn write_grad(&mut self, tensor: usize, start: usize, grad: &[f32]) -> io::Result<()> {
        self.grads.write_tile(tensor, 0, start, grad, Rounding::NEAREST)
    }
}

/// What an export moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportReport {
    pub tensors: usize,
    pub payload_bytes: u64,
    /// Host staging used: the f32 tile plus the encoded tile, independent of tensor size.
    pub staging_bytes: usize,
}

/// Stream every tensor's values, read from `params`, into `sink` encoded as
/// `dst[i]` (any encoding `registry` can encode), one tile at a time. On a
/// failure the sink is told to abort the tensor in flight.
pub fn export_values(
    params: &mut dyn ParamTiles,
    dst: &[TensorDescriptor],
    registry: &CodecRegistry,
    sink: &mut dyn TensorSink,
    tile_elems: usize,
) -> io::Result<ExportReport> {
    let tile = aligned_tile(dst, tile_elems);
    let mut vals = vec![0f32; tile];
    let mut buf = TileBuf::default();
    let mut payload_bytes = 0u64;
    let mut staging = tile * 4;
    for (ti, d) in dst.iter().enumerate() {
        d.validate().map_err(err)?;
        let codec = registry.encoder(&d.encoding).map_err(err)?;
        sink.begin(d).map_err(err)?;
        let numel = d.numel() as usize;
        let mut run = || -> io::Result<()> {
            for start in (0..numel).step_by(tile) {
                let n = tile.min(numel - start);
                params.read_value(ti, start, &mut vals[..n])?;
                let e = start as u64..(start + n) as u64;
                codec.encode_tile(d, &e, &vals[..n], &EncodeCtx::default(), &mut buf).map_err(err)?;
                let t = Tile { elems: e, payload: &buf.payload, scales: buf.scales.iter().map(|s| s.as_slice()).collect() };
                sink.write_tile(d, &t).map_err(err)?;
                staging = staging.max(tile * 4 + buf.payload.len() + buf.scales.iter().map(|s| s.len()).sum::<usize>());
            }
            Ok(())
        };
        if let Err(e) = run() {
            sink.abort(d);
            return Err(e);
        }
        sink.finish(d).map_err(err)?;
        payload_bytes += d.payload_bytes();
    }
    Ok(ExportReport { tensors: dst.len(), payload_bytes, staging_bytes: staging })
}
