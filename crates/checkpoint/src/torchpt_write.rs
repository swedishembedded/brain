// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A from-scratch, minimal `torch.save` (`.pt`/`.pth`) writer - the mirror of
//! [`crate::torchpt`]'s reader, for the one shape every caller in this
//! workspace actually needs: a FLAT `{name: tensor}` state_dict of contiguous
//! f32 tensors, written as protocol-2 pickle opcodes inside an uncompressed
//! (STORED) zip container - exactly the layout torch's own writer produces
//! for `torch.save(state_dict, path)`.
//!
//! Every synthetic `.pth` fixture in this workspace (an `ArchSpec`'s own test
//! suite, mainly - a real multi-gigabyte checkpoint is never committed) is
//! built through this rather than hand-assembling zip/pickle bytes per crate,
//! the same role [`crate::gguf_write`]/[`crate::st::save_safetensors`] already
//! play for their own formats.
//!
//! Swedish Embedded AB implements checkpoint format writers like this one for
//! clients whose test suites need byte-exact synthetic fixtures instead of
//! committed multi-gigabyte binaries. If your team needs the same discipline
//! for its own checkpoint formats, you can procure our services by emailing
//! info@swedishembedded.com.

/// One tensor to write: a name (dotted state_dict key), its shape, and its
/// contiguous row-major f32 values (`data.len()` must equal the shape's
/// element count).
pub struct TensorOut {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

// ---------------------------------------------------------------------------
// zip container (STORED entries, torch's own layout)
// ---------------------------------------------------------------------------

struct ZipWriter {
    out: Vec<u8>,
    central: Vec<u8>,
    count: u16,
}

impl ZipWriter {
    fn new() -> ZipWriter {
        ZipWriter { out: Vec::new(), central: Vec::new(), count: 0 }
    }

    fn add(&mut self, name: &str, data: &[u8]) {
        let lho = self.out.len() as u32;
        self.out.extend_from_slice(b"PK\x03\x04");
        self.out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.out.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.out.extend_from_slice(&0u16.to_le_bytes()); // method: STORED
        self.out.extend_from_slice(&[0u8; 4]); // mod time + date
        self.out.extend_from_slice(&[0u8; 4]); // crc32 (unchecked by the reader)
        self.out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // comp size
        self.out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncomp size
        self.out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        self.out.extend_from_slice(name.as_bytes());
        self.out.extend_from_slice(data);

        self.central.extend_from_slice(b"PK\x01\x02");
        self.central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        self.central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.central.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.central.extend_from_slice(&0u16.to_le_bytes()); // method
        self.central.extend_from_slice(&[0u8; 4]); // mod time + date
        self.central.extend_from_slice(&[0u8; 4]); // crc32
        self.central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.central.extend_from_slice(&0u16.to_le_bytes()); // extra len (central)
        self.central.extend_from_slice(&0u16.to_le_bytes()); // comment len
        self.central.extend_from_slice(&0u16.to_le_bytes()); // disk number
        self.central.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        self.central.extend_from_slice(&[0u8; 4]); // external attrs
        self.central.extend_from_slice(&lho.to_le_bytes());
        self.central.extend_from_slice(name.as_bytes());
        self.count += 1;
    }

    fn finish(mut self) -> Vec<u8> {
        let cd_off = self.out.len() as u32;
        let cd_size = self.central.len() as u32;
        self.out.extend_from_slice(&self.central);
        self.out.extend_from_slice(b"PK\x05\x06");
        self.out.extend_from_slice(&0u16.to_le_bytes()); // this disk
        self.out.extend_from_slice(&0u16.to_le_bytes()); // cd disk
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&cd_size.to_le_bytes());
        self.out.extend_from_slice(&cd_off.to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        self.out
    }
}

// ---------------------------------------------------------------------------
// pickle opcodes (protocol 2, the subset torch's pickler emits)
// ---------------------------------------------------------------------------

fn op_str(out: &mut Vec<u8>, s: &str) {
    out.push(0x58); // BINUNICODE
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn op_int(out: &mut Vec<u8>, v: i64) {
    if (0..256).contains(&v) {
        out.push(0x4b); // BININT1
        out.push(v as u8);
    } else if (256..65536).contains(&v) {
        out.push(0x4d); // BININT2
        out.extend_from_slice(&(v as u16).to_le_bytes());
    } else {
        out.push(0x4a); // BININT
        out.extend_from_slice(&(v as i32).to_le_bytes());
    }
}

fn op_global(out: &mut Vec<u8>, module: &str, name: &str) {
    out.push(0x63); // GLOBAL
    out.extend_from_slice(module.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(name.as_bytes());
    out.push(b'\n');
}

/// `MARK <ints...> TUPLE` - a size/stride tuple of any rank, including 0-dim.
fn op_usize_tuple(out: &mut Vec<u8>, dims: &[usize]) {
    out.push(0x28); // MARK
    for &d in dims {
        op_int(out, d as i64);
    }
    out.push(0x74); // TUPLE
}

/// The row-major contiguous stride for `shape` - what every tensor this
/// writer produces actually has (no views, no offsets).
fn contiguous_stride(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

/// `torch._utils._rebuild_tensor_v2(storage, 0, size, stride, False, {})` for
/// one f32 tensor at storage key `key`.
fn op_tensor(out: &mut Vec<u8>, key: &str, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    op_global(out, "torch._utils", "_rebuild_tensor_v2");
    out.push(0x28); // MARK (reduce args)
    // persistent id: MARK 'storage' torch.FloatStorage <key> 'cpu' <numel> TUPLE BINPERSID
    out.push(0x28); // MARK
    op_str(out, "storage");
    op_global(out, "torch", "FloatStorage");
    op_str(out, key);
    op_str(out, "cpu");
    op_int(out, numel as i64);
    out.push(0x74); // TUPLE
    out.push(0x51); // BINPERSID
    op_int(out, 0); // storage_offset
    op_usize_tuple(out, shape); // size
    op_usize_tuple(out, &contiguous_stride(shape)); // stride
    out.push(0x89); // NEWFALSE (requires_grad)
    out.push(0x7d); // EMPTY_DICT (backward_hooks)
    out.push(0x74); // TUPLE (6 reduce args)
    out.push(0x52); // REDUCE
}

/// Write a flat `{name: f32 tensor}` state_dict as a `torch.save`-compatible
/// `.pt`/`.pth` file at `path`, readable back byte-identically by
/// [`crate::torchpt::read`]/[`crate::torchpt::parse_shapes`].
pub fn write(path: &str, tensors: &[TensorOut]) -> Result<(), String> {
    let mut pickle = vec![0x80, 0x02]; // PROTO 2
    pickle.push(0x7d); // EMPTY_DICT
    if !tensors.is_empty() {
        pickle.push(0x28); // MARK (dict items)
        for (i, t) in tensors.iter().enumerate() {
            op_str(&mut pickle, &t.name);
            op_tensor(&mut pickle, &i.to_string(), &t.shape);
        }
        pickle.push(0x75); // SETITEMS
    }
    pickle.push(0x2e); // STOP

    let mut zip = ZipWriter::new();
    zip.add("archive/version", b"3\n");
    zip.add("archive/data.pkl", &pickle);
    for (i, t) in tensors.iter().enumerate() {
        let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        zip.add(&format!("archive/data/{i}"), &bytes);
    }
    std::fs::write(path, zip.finish()).map_err(|e| format!("torchpt_write: {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-torchpt-write-test-{tag}-{}-{n}.pth", std::process::id()))
    }

    /// A written fixture round-trips byte-identically through the full
    /// (data-materializing) reader.
    #[test]
    fn round_trips_through_the_full_reader() {
        let path = tmp("round-trip");
        write(
            path.to_str().unwrap(),
            &[
                TensorOut { name: "token_embedding.weight".to_string(), shape: vec![4, 3], data: (0..12).map(|i| i as f32).collect() },
                TensorOut { name: "blocks.0.attn.q.weight".to_string(), shape: vec![3, 3], data: vec![1.0; 9] },
            ],
        )
        .unwrap();

        let r = crate::torchpt::read(path.to_str().unwrap()).unwrap();
        assert_eq!(r.len(), 2);
        let a = r.iter().find(|t| t.name == "token_embedding.weight").unwrap();
        assert_eq!(a.shape, vec![4, 3]);
        assert_eq!(a.data, (0..12).map(|i| i as f32).collect::<Vec<_>>());
        let b = r.iter().find(|t| t.name == "blocks.0.attn.q.weight").unwrap();
        assert_eq!(b.shape, vec![3, 3]);
        assert_eq!(b.data, vec![1.0; 9]);

        std::fs::remove_file(&path).ok();
    }

    /// The shapes-only reader agrees with the full one on every name/shape,
    /// and (its whole point) never needed the storage bytes to get there.
    #[test]
    fn round_trips_through_the_shapes_only_reader() {
        let path = tmp("shapes-only");
        write(path.to_str().unwrap(), &[TensorOut { name: "encoder.conv1.weight".to_string(), shape: vec![2, 3, 3, 3, 3], data: vec![0.0; 2 * 3 * 3 * 3 * 3] }]).unwrap();

        let shapes = crate::torchpt::read_shapes(path.to_str().unwrap()).unwrap();
        assert_eq!(shapes, vec![("encoder.conv1.weight".to_string(), vec![2, 3, 3, 3, 3])]);

        std::fs::remove_file(&path).ok();
    }

    /// A 0-tensor state_dict (the EMPTY_DICT-only path, no MARK/SETITEMS)
    /// still parses to an empty tensor list rather than erroring.
    #[test]
    fn an_empty_state_dict_writes_and_reads_back_empty() {
        let path = tmp("empty");
        write(path.to_str().unwrap(), &[]).unwrap();
        let r = crate::torchpt::read(path.to_str().unwrap()).unwrap();
        assert!(r.is_empty(), "{r:?}");
        std::fs::remove_file(&path).ok();
    }
}
