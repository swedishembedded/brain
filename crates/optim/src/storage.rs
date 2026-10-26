// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in streaming optimiser state from disk
// for very large models then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Tile-addressed storage for optimiser state and any other per-element
//! training tensor (values and gradients reuse it, one slot per tensor).
//!
//! [`OptimizerStateStorage`] is the whole contract: read a tile of a slot as
//! f32, write a tile back. How the bytes are held is a [`Backing`]: RAM
//! ([`MemBacking`]) or a file ([`FileBacking`], positional I/O, never mapped
//! whole), so the resident footprint of a step is the staging workspace, not
//! the parameter count. The workspace is allocated once from the maximum tile
//! size and its size is reported by [`OptimizerStateStorage::staging_bytes`].
//!
//! A fresh backing is all zero bytes, which is a valid empty state in every
//! codec (see `codec`), so a new file is just `set_len`.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

use paramstore::precision::StateFormat;

use crate::codec::{Rounding, StateCodec};

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Byte-addressable store a state layout lives in.
pub trait Backing {
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()>;
    fn write_at(&mut self, off: u64, buf: &[u8]) -> io::Result<()>;
    fn len(&self) -> u64;
    /// Bytes `off..off + len` in place, when the backing is addressable memory: a tile is then decoded straight from
    /// the state instead of through a staging copy. `None` (the default) means read through [`Self::read_at`].
    fn view(&self, _off: u64, _len: usize) -> Option<&[u8]> {
        None
    }
    /// [`Self::view`] for writing.
    fn view_mut(&mut self, _off: u64, _len: usize) -> Option<&mut [u8]> {
        None
    }
    /// Whether [`Self::view`], [`Self::view_mut`] and [`Self::views_mut`] answer for every in-range request.
    fn addressable(&self) -> bool {
        false
    }
    /// Disjoint mutable views of `ranges` (`(offset, length)`, ascending and non-overlapping), when the backing is
    /// addressable memory: the pieces of a tile can then be updated in place from several threads.
    fn views_mut(&mut self, _ranges: &[(u64, usize)]) -> Option<Vec<&mut [u8]>> {
        None
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Make everything written so far durable.
    fn sync(&mut self) -> io::Result<()>;
}

#[derive(Default)]
pub struct MemBacking(Vec<u8>);

impl MemBacking {
    /// `len` zero bytes, resident: every page is faulted in here, across the CPU pool, instead of one page at a time
    /// by the first step that touches it (tens of gigabytes of optimizer state cost seconds of single-threaded page
    /// faults inside that step).
    pub fn zeroed(len: u64) -> MemBacking {
        const PAGE: usize = 4096;
        const PREFAULT_CHUNK: usize = 1 << 24;
        let mut bytes = vec![0u8; len as usize];
        crate::par::chunks_mut(&mut bytes, PREFAULT_CHUNK, |_, chunk| chunk.iter_mut().step_by(PAGE).for_each(|b| *b = 0));
        MemBacking(bytes)
    }
}

/// Bytes below which a copy runs on the calling thread.
const PARALLEL_COPY_MIN: usize = 1 << 20;

/// `dst.copy_from_slice(src)`, cut into contiguous pieces on the CPU pool when it is large: one core does not saturate
/// the memory system, and a tile of optimizer state is megabytes.
fn copy_parallel(dst: &mut [u8], src: &[u8]) {
    let pieces = crate::par::threads().min(dst.len() / PARALLEL_COPY_MIN);
    if pieces <= 1 {
        return dst.copy_from_slice(src);
    }
    let piece = dst.len().div_ceil(pieces);
    crate::par::chunks_mut(dst, piece, |i, d| d.copy_from_slice(&src[i * piece..i * piece + d.len()]));
}

impl Backing for MemBacking {
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let end = off as usize + buf.len();
        copy_parallel(buf, self.0.get(off as usize..end).ok_or_else(|| invalid(format!("read past end: {end} > {}", self.0.len())))?);
        Ok(())
    }
    fn write_at(&mut self, off: u64, buf: &[u8]) -> io::Result<()> {
        let end = off as usize + buf.len();
        let len = self.0.len();
        copy_parallel(self.0.get_mut(off as usize..end).ok_or_else(|| invalid(format!("write past end: {end} > {len}")))?, buf);
        Ok(())
    }
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn addressable(&self) -> bool {
        true
    }
    fn view(&self, off: u64, len: usize) -> Option<&[u8]> {
        self.0.get(off as usize..off as usize + len)
    }
    fn view_mut(&mut self, off: u64, len: usize) -> Option<&mut [u8]> {
        self.0.get_mut(off as usize..off as usize + len)
    }
    fn views_mut(&mut self, ranges: &[(u64, usize)]) -> Option<Vec<&mut [u8]>> {
        let mut rest = self.0.as_mut_slice();
        let mut consumed = 0usize;
        let mut views = Vec::with_capacity(ranges.len());
        for &(off, len) in ranges {
            let skip = (off as usize).checked_sub(consumed)?;
            if skip + len > rest.len() {
                return None;
            }
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(skip + len);
            views.push(&mut head[skip..]);
            rest = tail;
            consumed += skip + len;
        }
        Some(views)
    }
    fn sync(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct FileBacking {
    file: File,
    len: u64,
}

impl FileBacking {
    /// New zero-filled (sparse) file of `len` bytes; fails if it exists.
    pub fn create(path: &Path, len: u64) -> io::Result<FileBacking> {
        let file = OpenOptions::new().read(true).write(true).create_new(true).open(path)?;
        file.set_len(len)?;
        Ok(FileBacking { file, len })
    }

    /// Open an existing state file read/write.
    pub fn open(path: &Path) -> io::Result<FileBacking> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        Ok(FileBacking { file, len })
    }
}

impl Backing for FileBacking {
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        self.file.read_exact_at(buf, off)
    }
    fn write_at(&mut self, off: u64, buf: &[u8]) -> io::Result<()> {
        if off + buf.len() as u64 > self.len {
            return Err(invalid(format!("write past end: {} > {}", off + buf.len() as u64, self.len)));
        }
        self.file.write_all_at(buf, off)
    }
    fn len(&self) -> u64 {
        self.len
    }
    fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }
}

/// Where one slot of one tensor lives in the backing.
#[derive(Clone, Copy, Debug)]
struct SlotLayout {
    codec: StateCodec,
    data_off: u64,
    scale_off: u64,
}

/// Placement of every (tensor, slot) in one flat backing: for each slot a code
/// region, then its scale region. Deterministic from the specs, so a reopened
/// file needs no stored index.
#[derive(Clone, Debug)]
pub struct StateLayout {
    tensors: Vec<(usize, Vec<SlotLayout>)>,
    total: u64,
}

impl StateLayout {
    /// One entry per tensor: its element count and the codec of each slot.
    pub fn new(tensors: &[(usize, Vec<StateCodec>)]) -> StateLayout {
        let mut off = 0u64;
        let mut out = Vec::with_capacity(tensors.len());
        for (numel, codecs) in tensors {
            let mut slots = Vec::with_capacity(codecs.len());
            for c in codecs {
                let data_off = off;
                off += c.data_bytes(*numel) as u64;
                let scale_off = off;
                off += c.scale_bytes(*numel) as u64;
                slots.push(SlotLayout { codec: *c, data_off, scale_off });
            }
            out.push((*numel, slots));
        }
        StateLayout { tensors: out, total: off }
    }

    /// Bytes of the backing this layout needs. Physical: scales included.
    pub fn total_bytes(&self) -> u64 {
        self.total
    }

    fn min_block(&self) -> usize {
        self.tensors.iter().flat_map(|(_, s)| s.iter()).map(|s| s.codec.format.block()).filter(|&b| b > 1).min().unwrap_or(1)
    }
}

/// The bytes of one slot over an element range, mutable and disjoint from every other [`SlotView`] of the same
/// request: the form in which a tile of state is updated in place, piece by piece, from several threads.
pub struct SlotView<'a> {
    pub codec: StateCodec,
    data: &'a mut [u8],
    scales: &'a mut [u8],
}

/// One piece of a [`SlotView`], a whole number of the slot's blocks.
pub struct SlotPiece<'a> {
    codec: StateCodec,
    data: &'a mut [u8],
    scales: &'a mut [u8],
}

impl<'a> SlotView<'a> {
    /// Cut the view into pieces of `piece_elems` elements (a multiple of the slot's block); the last may be shorter.
    pub fn pieces(self, piece_elems: usize) -> Vec<SlotPiece<'a>> {
        debug_assert!(piece_elems.is_multiple_of(self.codec.format.block()), "a piece is whole blocks");
        let SlotView { codec, mut data, mut scales } = self;
        let mut out = Vec::new();
        while !data.is_empty() {
            let elems = piece_elems.min(data.len() / codec.data_bytes(1));
            let (d, rest_d) = std::mem::take(&mut data).split_at_mut(codec.data_bytes(elems));
            let (sc, rest_s) = std::mem::take(&mut scales).split_at_mut(codec.scale_bytes(elems));
            (data, scales) = (rest_d, rest_s);
            out.push(SlotPiece { codec, data: d, scales: sc });
        }
        out
    }
}

impl SlotPiece<'_> {
    /// The piece as the f32 values it stores, in place, when it is an f32 slot on a little-endian host whose bytes are
    /// 4-byte aligned: the update then needs no decode into scratch and no encode back.
    #[cfg(target_endian = "little")]
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        matches!(self.codec.format, StateFormat::F32).then(|| bytemuck::try_cast_slice_mut(&mut *self.data).ok()).flatten()
    }

    #[cfg(not(target_endian = "little"))]
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        None
    }

    /// Decode the piece into `out` (its element count).
    pub fn decode(&self, out: &mut [f32]) {
        self.codec.decode_serial(self.data, self.scales, out);
    }

    /// Encode `vals` into the piece; `base` is the absolute index of its first element in the tensor (the rounding is
    /// keyed on it).
    pub fn encode(&mut self, vals: &[f32], rnd: Rounding, base: u64) {
        self.codec.encode_serial(vals, self.data, self.scales, rnd, base);
    }
}

/// Tile-addressed access to per-element training tensors.
pub trait OptimizerStateStorage {
    /// Whether [`Self::slot_views`] answers for every valid tile: the state is addressable memory.
    fn addressable(&self) -> bool {
        false
    }
    /// In-place views of `slots` (ascending) of `tensor` over elements `[start, start + n)`, when the state is
    /// addressable memory and the range is a valid tile. `None` means use [`Self::read_tile`] and
    /// [`Self::write_tile`].
    fn slot_views(&mut self, _tensor: usize, _slots: &[usize], _start: usize, _n: usize) -> Option<Vec<SlotView<'_>>> {
        None
    }
    /// Decode `out.len()` elements of `slot` of `tensor`, starting at element
    /// `start`, into `out`. `start` must be a multiple of the slot's block.
    fn read_tile(&mut self, tensor: usize, slot: usize, start: usize, out: &mut [f32]) -> io::Result<()>;
    /// Encode `vals` into the same range; the inverse of [`Self::read_tile`].
    fn write_tile(&mut self, tensor: usize, slot: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()>;
    /// Physical bytes of the whole state (scales included).
    fn physical_bytes(&self) -> u64;
    /// Bytes of host staging this storage holds, independent of tensor sizes.
    fn staging_bytes(&self) -> usize;
    /// Stream the raw state bytes, in order, to `w` in chunks of the staging size.
    fn export(&mut self, w: &mut dyn Write) -> io::Result<()>;
    /// Replace the raw state bytes from `r`; fails if `r` is short or long.
    fn import(&mut self, r: &mut dyn Read) -> io::Result<()>;
    /// Make the backing durable.
    fn sync(&mut self) -> io::Result<()>;
}

/// [`OptimizerStateStorage`] over any [`Backing`] with a bounded workspace.
pub struct BlockStateStorage<B: Backing> {
    layout: StateLayout,
    backing: B,
    max_tile: usize,
    data: Vec<u8>,
    scales: Vec<u8>,
}

impl<B: Backing> BlockStateStorage<B> {
    /// `max_tile_elems` bounds every tile and therefore the staging workspace.
    pub fn new(layout: StateLayout, backing: B, max_tile_elems: usize) -> io::Result<Self> {
        if backing.len() != layout.total_bytes() {
            return Err(invalid(format!("backing is {} bytes, layout needs {}", backing.len(), layout.total_bytes())));
        }
        let block = layout.min_block();
        // The widest codec is f32 (4 bytes/element); scales are 4 bytes per block.
        let data = vec![0u8; 4 * max_tile_elems];
        let scales = vec![0u8; 4 * max_tile_elems.div_ceil(block)];
        Ok(BlockStateStorage { layout, backing, max_tile: max_tile_elems, data, scales })
    }

    pub fn backing(&self) -> &B {
        &self.backing
    }

    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn slot(&self, tensor: usize, slot: usize, start: usize, n: usize) -> io::Result<(SlotLayout, usize)> {
        let (numel, slots) = self.layout.tensors.get(tensor).ok_or_else(|| invalid(format!("no tensor {tensor}")))?;
        let s = *slots.get(slot).ok_or_else(|| invalid(format!("tensor {tensor} has no slot {slot}")))?;
        let block = s.codec.format.block();
        if n > self.max_tile || start + n > *numel || !start.is_multiple_of(block) || (!n.is_multiple_of(block) && start + n != *numel) {
            return Err(invalid(format!(
                "tile [{start}, {}) of tensor {tensor} slot {slot} ({numel} elements, block {block}, max tile {}) is out of range or unaligned",
                start + n,
                self.max_tile
            )));
        }
        Ok((s, block))
    }
}

impl<B: Backing> OptimizerStateStorage for BlockStateStorage<B> {
    fn addressable(&self) -> bool {
        self.backing.addressable()
    }

    fn slot_views(&mut self, tensor: usize, slots: &[usize], start: usize, n: usize) -> Option<Vec<SlotView<'_>>> {
        if !slots.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        let mut layouts = Vec::with_capacity(slots.len());
        let mut ranges = Vec::with_capacity(2 * slots.len());
        for &slot in slots {
            let (s, block) = self.slot(tensor, slot, start, n).ok()?;
            let elem = s.codec.data_bytes(1);
            ranges.push((s.data_off + (start * elem) as u64, s.codec.data_bytes(n)));
            // An unscaled slot has an empty scale region: it sits at the region's start, which keeps the ranges ascending.
            let scale_at = if s.codec.scale_bytes(n) == 0 { s.scale_off } else { s.scale_off + (4 * (start / block)) as u64 };
            ranges.push((scale_at, s.codec.scale_bytes(n)));
            layouts.push(s);
        }
        let mut views = self.backing.views_mut(&ranges)?.into_iter();
        Some(layouts.into_iter().map(|s| SlotView { codec: s.codec, data: views.next().expect("two views per slot"), scales: views.next().expect("two views per slot") }).collect())
    }

    fn read_tile(&mut self, tensor: usize, slot: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        let n = out.len();
        let (s, block) = self.slot(tensor, slot, start, n)?;
        let db = s.codec.data_bytes(n);
        let sb = s.codec.scale_bytes(n);
        let elem = if n == 0 { 0 } else { s.codec.data_bytes(1) };
        let (data_at, scale_at) = (s.data_off + (start * elem) as u64, s.scale_off + (4 * (start / block)) as u64);
        let scales = if sb == 0 { Some(&[][..]) } else { self.backing.view(scale_at, sb) };
        if let (Some(data), Some(scales)) = (self.backing.view(data_at, db), scales) {
            s.codec.decode(data, scales, out);
            return Ok(());
        }
        self.backing.read_at(data_at, &mut self.data[..db])?;
        if sb > 0 {
            self.backing.read_at(scale_at, &mut self.scales[..sb])?;
        }
        s.codec.decode(&self.data[..db], &self.scales[..sb], out);
        Ok(())
    }

    fn write_tile(&mut self, tensor: usize, slot: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()> {
        let n = vals.len();
        let (s, block) = self.slot(tensor, slot, start, n)?;
        let db = s.codec.data_bytes(n);
        let sb = s.codec.scale_bytes(n);
        let elem = if n == 0 { 0 } else { s.codec.data_bytes(1) };
        // A format without scales is encoded in place when the backing is addressable.
        if sb == 0 {
            if let Some(data) = self.backing.view_mut(s.data_off + (start * elem) as u64, db) {
                s.codec.encode(vals, data, &mut [], rnd, start as u64);
                return Ok(());
            }
        }
        s.codec.encode(vals, &mut self.data[..db], &mut self.scales[..sb], rnd, start as u64);
        self.backing.write_at(s.data_off + (start * elem) as u64, &self.data[..db])?;
        if sb > 0 {
            self.backing.write_at(s.scale_off + (4 * (start / block)) as u64, &self.scales[..sb])?;
        }
        Ok(())
    }

    fn physical_bytes(&self) -> u64 {
        self.layout.total
    }

    fn staging_bytes(&self) -> usize {
        self.data.len() + self.scales.len()
    }

    fn export(&mut self, w: &mut dyn Write) -> io::Result<()> {
        let mut off = 0u64;
        while off < self.layout.total {
            let n = ((self.layout.total - off) as usize).min(self.data.len());
            self.backing.read_at(off, &mut self.data[..n])?;
            w.write_all(&self.data[..n])?;
            off += n as u64;
        }
        Ok(())
    }

    fn import(&mut self, r: &mut dyn Read) -> io::Result<()> {
        let mut off = 0u64;
        while off < self.layout.total {
            let n = ((self.layout.total - off) as usize).min(self.data.len());
            r.read_exact(&mut self.data[..n])?;
            self.backing.write_at(off, &self.data[..n])?;
            off += n as u64;
        }
        let mut extra = [0u8; 1];
        if r.read(&mut extra)? != 0 {
            return Err(invalid("state stream is longer than the layout".to_string()));
        }
        Ok(())
    }

    fn sync(&mut self) -> io::Result<()> {
        self.backing.sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::SlotKind;
    use paramstore::precision::StateFormat;

    fn codec(format: StateFormat, kind: SlotKind) -> StateCodec {
        StateCodec { format, kind }
    }

    fn storage(max_tile: usize) -> BlockStateStorage<MemBacking> {
        let layout = StateLayout::new(&[
            (100, vec![codec(StateFormat::Int8Block { block: 16 }, SlotKind::Signed), codec(StateFormat::Bf16, SlotKind::Signed)]),
            (7, vec![codec(StateFormat::F32, SlotKind::Signed)]),
        ]);
        let b = MemBacking::zeroed(layout.total_bytes());
        BlockStateStorage::new(layout, b, max_tile).unwrap()
    }

    /// Spec: tiles of any slot round-trip independently, a tile write touches
    /// only its own range, and the tail tile of a tensor need not be a whole block.
    #[test]
    fn tiles_roundtrip_and_do_not_bleed() {
        let mut s = storage(32);
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 10.0) * 0.1).collect();
        s.write_tile(0, 1, 32, &vals, Rounding::NEAREST).unwrap();
        let mut out = vec![9.0f32; 32];
        s.read_tile(0, 1, 32, &mut out).unwrap();
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() <= a.abs() * 2f32.powi(-8) + 1e-9);
        }
        s.read_tile(0, 1, 0, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0), "neighbouring tile untouched");
        // tail tile (96..100) of the int8 slot: 4 elements, not a whole block
        s.write_tile(0, 0, 96, &[1.0, -1.0, 0.5, 0.0], Rounding::NEAREST).unwrap();
        let mut tail = [0f32; 4];
        s.read_tile(0, 0, 96, &mut tail).unwrap();
        assert!((tail[0] - 1.0).abs() < 0.02 && (tail[1] + 1.0).abs() < 0.02);
        s.write_tile(1, 0, 0, &[1.5; 7], Rounding::NEAREST).unwrap();
    }

    /// Spec: tiles beyond the workspace bound, or unaligned to the block, are
    /// refused with a message naming the range - never silently clamped.
    #[test]
    fn oversized_or_unaligned_tiles_are_refused() {
        let mut s = storage(32);
        let mut big = vec![0f32; 48];
        assert!(s.read_tile(0, 1, 0, &mut big).unwrap_err().to_string().contains("max tile 32"));
        let mut t = vec![0f32; 16];
        assert!(s.read_tile(0, 0, 8, &mut t).is_err(), "start 8 is inside a 16-block");
        assert!(s.read_tile(0, 5, 0, &mut t).is_err());
    }

    /// Spec: export/import moves the raw state in staging-sized chunks and
    /// reproduces it exactly; a truncated stream is an error.
    #[test]
    fn export_import_is_exact_and_bounded() {
        let mut a = storage(16);
        a.write_tile(0, 0, 0, &(0..16).map(|i| i as f32).collect::<Vec<_>>(), Rounding::NEAREST).unwrap();
        a.write_tile(1, 0, 0, &[2.0; 7], Rounding::NEAREST).unwrap();
        let mut bytes = Vec::new();
        a.export(&mut bytes).unwrap();
        assert_eq!(bytes.len() as u64, a.physical_bytes());
        let mut b = storage(16);
        b.import(&mut &bytes[..]).unwrap();
        let (mut x, mut y) = (vec![0f32; 16], vec![0f32; 16]);
        a.read_tile(0, 0, 0, &mut x).unwrap();
        b.read_tile(0, 0, 0, &mut y).unwrap();
        assert_eq!(x, y);
        assert!(storage(16).import(&mut &bytes[..bytes.len() - 1]).is_err());
    }
}
