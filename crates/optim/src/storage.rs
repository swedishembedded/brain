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

use crate::codec::{Rounding, StateCodec};

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Byte-addressable store a state layout lives in.
pub trait Backing {
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()>;
    fn write_at(&mut self, off: u64, buf: &[u8]) -> io::Result<()>;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Make everything written so far durable.
    fn sync(&mut self) -> io::Result<()>;
}

#[derive(Default)]
pub struct MemBacking(Vec<u8>);

impl MemBacking {
    pub fn zeroed(len: u64) -> MemBacking {
        MemBacking(vec![0u8; len as usize])
    }
}

impl Backing for MemBacking {
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let end = off as usize + buf.len();
        buf.copy_from_slice(self.0.get(off as usize..end).ok_or_else(|| invalid(format!("read past end: {end} > {}", self.0.len())))?);
        Ok(())
    }
    fn write_at(&mut self, off: u64, buf: &[u8]) -> io::Result<()> {
        let end = off as usize + buf.len();
        let len = self.0.len();
        self.0.get_mut(off as usize..end).ok_or_else(|| invalid(format!("write past end: {end} > {len}")))?.copy_from_slice(buf);
        Ok(())
    }
    fn len(&self) -> u64 {
        self.0.len() as u64
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

/// Tile-addressed access to per-element training tensors.
pub trait OptimizerStateStorage {
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
    fn read_tile(&mut self, tensor: usize, slot: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        let n = out.len();
        let (s, block) = self.slot(tensor, slot, start, n)?;
        let db = s.codec.data_bytes(n);
        let sb = s.codec.scale_bytes(n);
        let elem = if n == 0 { 0 } else { s.codec.data_bytes(1) };
        self.backing.read_at(s.data_off + (start * elem) as u64, &mut self.data[..db])?;
        if sb > 0 {
            self.backing.read_at(s.scale_off + (4 * (start / block)) as u64, &mut self.scales[..sb])?;
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
