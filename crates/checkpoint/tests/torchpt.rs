// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tests for the pure-Rust `torch.save` (.pt) reader. All fixtures are
//! synthetic byte streams constructed in-code (no committed binaries): a
//! minimal STORED-zip builder plus a pickle-opcode builder emit exactly what
//! torch's zip writer + protocol-2 pickler produce for state_dicts. The flat
//! two-tensor case is additionally written out as a hand-computed literal
//! opcode sequence (commented byte-by-byte) so the test does not merely
//! round-trip through our own builders.

use checkpoint::torchpt;

// ---------------------------------------------------------------------------
// zip builder (STORED entries, torch container layout)
// ---------------------------------------------------------------------------

/// Build a ZIP from (name, data, method, local_extra_pad) entries. torch
/// writes method 0 (STORED) and pads *local* extra fields for tensor-data
/// alignment; `local_extra_pad` simulates that padding. CRCs are zeroed (the
/// reader slices data by offsets and does not checksum).
fn zip_with(entries: &[(&str, &[u8], u16, usize)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cd = Vec::new();
    for (name, data, method, pad) in entries {
        let lho = out.len() as u32;
        // local file header
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]); // mod time + date
        out.extend_from_slice(&[0u8; 4]); // crc32 (unchecked)
        out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // comp size
        out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncomp size
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&(*pad as u16).to_le_bytes()); // extra len
        out.extend_from_slice(name.as_bytes());
        out.extend(std::iter::repeat_n(0xAAu8, *pad)); // alignment padding
        out.extend_from_slice(data);
        // central directory record
        cd.extend_from_slice(b"PK\x01\x02");
        cd.extend_from_slice(&20u16.to_le_bytes()); // version made by
        cd.extend_from_slice(&20u16.to_le_bytes()); // version needed
        cd.extend_from_slice(&0u16.to_le_bytes()); // flags
        cd.extend_from_slice(&method.to_le_bytes());
        cd.extend_from_slice(&[0u8; 4]); // mod time + date
        cd.extend_from_slice(&[0u8; 4]); // crc32
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes()); // extra len (central)
        cd.extend_from_slice(&0u16.to_le_bytes()); // comment len
        cd.extend_from_slice(&0u16.to_le_bytes()); // disk number
        cd.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        cd.extend_from_slice(&[0u8; 4]); // external attrs
        cd.extend_from_slice(&lho.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
    }
    let cd_off = out.len() as u32;
    let cd_size = cd.len() as u32;
    let n = entries.len() as u16;
    out.extend_from_slice(&cd);
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes()); // this disk
    out.extend_from_slice(&0u16.to_le_bytes()); // cd disk
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    out
}

/// Standard torch archive: `archive/version`, `archive/data.pkl`, and one
/// `archive/data/<key>` blob per storage. All STORED.
fn pt_archive(pickle: &[u8], storages: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut names: Vec<String> = Vec::new();
    for (key, _) in storages {
        names.push(format!("archive/data/{key}"));
    }
    let mut entries: Vec<(&str, &[u8], u16, usize)> =
        vec![("archive/version", b"3\n", 0, 0), ("archive/data.pkl", pickle, 0, 0)];
    for (i, (_, data)) in storages.iter().enumerate() {
        entries.push((&names[i], data, 0, 0));
    }
    zip_with(&entries)
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// pickle builder (protocol 2, the opcodes torch's pickler emits)
// ---------------------------------------------------------------------------

struct P(Vec<u8>);

impl P {
    fn new() -> Self {
        P(vec![0x80, 0x02]) // PROTO 2
    }
    fn op(mut self, b: u8) -> Self {
        self.0.push(b);
        self
    }
    /// BINUNICODE <u32 LE len> <utf8>
    fn s(mut self, s: &str) -> Self {
        self.0.push(0x58);
        self.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.0.extend_from_slice(s.as_bytes());
        self
    }
    /// BININT1 / BININT2 / BININT depending on magnitude
    fn int(mut self, v: i64) -> Self {
        if (0..256).contains(&v) {
            self.0.push(0x4b); // BININT1
            self.0.push(v as u8);
        } else if (256..65536).contains(&v) {
            self.0.push(0x4d); // BININT2
            self.0.extend_from_slice(&(v as u16).to_le_bytes());
        } else {
            self.0.push(0x4a); // BININT
            self.0.extend_from_slice(&(v as i32).to_le_bytes());
        }
        self
    }
    /// GLOBAL <module>\n<name>\n
    fn global(mut self, module: &str, name: &str) -> Self {
        self.0.push(0x63);
        self.0.extend_from_slice(module.as_bytes());
        self.0.push(b'\n');
        self.0.extend_from_slice(name.as_bytes());
        self.0.push(b'\n');
        self
    }
    /// MARK <ints...> TUPLE — a size/stride tuple of any rank (incl. 0-dim)
    fn usize_tuple(self, dims: &[usize]) -> Self {
        let mut p = self.op(0x28); // MARK
        for d in dims {
            p = p.int(*d as i64);
        }
        p.op(0x74) // TUPLE
    }
    /// Persistent id: MARK 'storage' <StorageType> <key> 'cpu' <numel> TUPLE BINPERSID
    fn storage(self, stype: &str, key: &str, numel: usize) -> Self {
        self.op(0x28) // MARK
            .s("storage")
            .global("torch", stype)
            .s(key)
            .s("cpu")
            .int(numel as i64)
            .op(0x74) // TUPLE
            .op(0x51) // BINPERSID
    }
    /// `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, False, {})`
    fn tensor(self, stype: &str, key: &str, numel: usize, offset: usize, size: &[usize], stride: &[usize]) -> Self {
        self.global("torch._utils", "_rebuild_tensor_v2")
            .op(0x28) // MARK (reduce args)
            .storage(stype, key, numel)
            .int(offset as i64)
            .usize_tuple(size)
            .usize_tuple(stride)
            .op(0x89) // NEWFALSE (requires_grad)
            .op(0x7d) // EMPTY_DICT (backward_hooks)
            .op(0x74) // TUPLE
            .op(0x52) // REDUCE
    }
    fn stop(mut self) -> Vec<u8> {
        self.0.push(0x2e); // STOP
        self.0
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// Flat dict of two f32 tensors where the pickle is a hand-computed literal
/// byte sequence (each opcode commented), independent of the P builder.
#[test]
fn flat_dict_hand_built_pickle() {
    let mut p: Vec<u8> = Vec::new();
    p.extend_from_slice(&[0x80, 0x02]); // PROTO 2
    p.push(0x7d); //                       EMPTY_DICT
    p.extend_from_slice(&[0x71, 0x00]); // BINPUT 0
    p.push(0x28); //                       MARK (dict items)
    // key "a"
    p.push(0x58); //                       BINUNICODE
    p.extend_from_slice(&1u32.to_le_bytes());
    p.push(b'a');
    // value: _rebuild_tensor_v2(storage('0', f32, 2), 0, (2,), (1,), False, {})
    p.push(0x63); //                       GLOBAL torch._utils _rebuild_tensor_v2
    p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
    p.extend_from_slice(&[0x71, 0x01]); // BINPUT 1
    p.push(0x28); //                       MARK (reduce args)
    p.push(0x28); //                       MARK (persistent id tuple)
    p.extend_from_slice(&[0x8c, 7]); //    SHORT_BINUNICODE "storage"
    p.extend_from_slice(b"storage");
    p.push(0x63); //                       GLOBAL torch FloatStorage
    p.extend_from_slice(b"torch\nFloatStorage\n");
    p.extend_from_slice(&[0x8c, 1, b'0']); // SHORT_BINUNICODE "0" (storage key)
    p.extend_from_slice(&[0x8c, 3]); //    SHORT_BINUNICODE "cpu"
    p.extend_from_slice(b"cpu");
    p.extend_from_slice(&[0x4b, 0x02]); // BININT1 2 (numel)
    p.push(0x74); //                       TUPLE -> ('storage', FloatStorage, '0', 'cpu', 2)
    p.push(0x51); //                       BINPERSID -> storage ref
    p.extend_from_slice(&[0x4b, 0x00]); // BININT1 0 (storage_offset)
    p.extend_from_slice(&[0x4b, 0x02, 0x85]); // BININT1 2; TUPLE1 -> size (2,)
    p.extend_from_slice(&[0x4b, 0x01, 0x85]); // BININT1 1; TUPLE1 -> stride (1,)
    p.push(0x89); //                       NEWFALSE (requires_grad)
    p.push(0x7d); //                       EMPTY_DICT (backward_hooks)
    p.push(0x74); //                       TUPLE (6 reduce args)
    p.push(0x52); //                       REDUCE -> tensor "a"
    // key "b"
    p.push(0x58); //                       BINUNICODE "b"
    p.extend_from_slice(&1u32.to_le_bytes());
    p.push(b'b');
    // value: _rebuild_tensor_v2(storage('1', f32, 3), 0, (3,), (1,), False, {})
    p.extend_from_slice(&[0x68, 0x01]); // BINGET 1 (memoized _rebuild_tensor_v2)
    p.push(0x28); //                       MARK (reduce args)
    p.push(0x28); //                       MARK (persistent id tuple)
    p.extend_from_slice(&[0x8c, 7]); //    SHORT_BINUNICODE "storage"
    p.extend_from_slice(b"storage");
    p.push(0x63); //                       GLOBAL torch FloatStorage
    p.extend_from_slice(b"torch\nFloatStorage\n");
    p.extend_from_slice(&[0x8c, 1, b'1']); // SHORT_BINUNICODE "1" (storage key)
    p.extend_from_slice(&[0x8c, 3]); //    SHORT_BINUNICODE "cpu"
    p.extend_from_slice(b"cpu");
    p.extend_from_slice(&[0x4b, 0x03]); // BININT1 3 (numel)
    p.push(0x74); //                       TUPLE
    p.push(0x51); //                       BINPERSID
    p.extend_from_slice(&[0x4b, 0x00]); // BININT1 0 (storage_offset)
    p.extend_from_slice(&[0x4b, 0x03, 0x85]); // size (3,)
    p.extend_from_slice(&[0x4b, 0x01, 0x85]); // stride (1,)
    p.push(0x89); //                       NEWFALSE
    p.push(0x7d); //                       EMPTY_DICT
    p.push(0x74); //                       TUPLE
    p.push(0x52); //                       REDUCE -> tensor "b"
    p.push(0x75); //                       SETITEMS (into the dict at memo 0)
    p.push(0x2e); //                       STOP

    let file = pt_archive(
        &p,
        &[("0", f32_bytes(&[1.5, -2.0])), ("1", f32_bytes(&[3.0, 4.0, 5.0]))],
    );
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.skipped_non_tensor, 0);
    assert_eq!(r.tensors.len(), 2);
    assert_eq!(r.tensors[0].name, "a");
    assert_eq!(r.tensors[0].shape, vec![2]);
    assert_eq!(r.tensors[0].data, vec![1.5, -2.0]);
    assert_eq!(r.tensors[1].name, "b");
    assert_eq!(r.tensors[1].shape, vec![3]);
    assert_eq!(r.tensors[1].data, vec![3.0, 4.0, 5.0]);
}

/// Nested OrderedDicts (GLOBAL collections OrderedDict + REDUCE + SETITEMS,
/// the exact shape torch's pickler emits) flatten with '.' joins; the scalar
/// leaf is counted, and a memoized class is re-fetched with BINGET.
#[test]
fn nested_dict_flattening() {
    let p = P::new()
        .global("collections", "OrderedDict")
        .op(0x71).op(0x01) // BINPUT 1 (the class)
        .op(0x29) // EMPTY_TUPLE
        .op(0x52) // REDUCE -> outer OrderedDict
        .op(0x71).op(0x00) // BINPUT 0
        .op(0x28) // MARK (outer items)
        .s("denoiser")
        .op(0x68).op(0x01) // BINGET 1 (OrderedDict class again)
        .op(0x29) // EMPTY_TUPLE
        .op(0x52) // REDUCE -> inner OrderedDict
        .op(0x28) // MARK (inner items)
        .s("inner_model.conv_in.weight")
        .tensor("FloatStorage", "0", 4, 0, &[2, 2], &[2, 1])
        .op(0x75) // SETITEMS (inner)
        .s("step")
        .int(500)
        .op(0x75) // SETITEMS (outer)
        .stop();

    let file = pt_archive(&p, &[("0", f32_bytes(&[1.0, 2.0, 3.0, 4.0]))]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 1);
    assert_eq!(r.tensors[0].name, "denoiser.inner_model.conv_in.weight");
    assert_eq!(r.tensors[0].shape, vec![2, 2]);
    assert_eq!(r.tensors[0].data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(r.skipped_non_tensor, 1); // "step"
}

/// F16, BF16, and F64 storages decode to the exact f32 values of their
/// hand-written bit patterns.
#[test]
fn f16_bf16_f64_storages() {
    let p = P::new()
        .op(0x7d) // EMPTY_DICT
        .op(0x28) // MARK
        .s("h")
        .tensor("HalfStorage", "0", 3, 0, &[3], &[1])
        .s("bf")
        .tensor("BFloat16Storage", "1", 2, 0, &[2], &[1])
        .s("d")
        .tensor("DoubleStorage", "2", 2, 0, &[2], &[1])
        .op(0x75) // SETITEMS
        .stop();

    // f16 bit patterns: 0x3C00 = 1.0, 0xC000 = -2.0, 0x3800 = 0.5
    let h: Vec<u8> = [0x3C00u16, 0xC000, 0x3800].iter().flat_map(|v| v.to_le_bytes()).collect();
    // bf16 bit patterns: 0x3F80 = 1.0, 0xC080 = -4.0
    let bf: Vec<u8> = [0x3F80u16, 0xC080].iter().flat_map(|v| v.to_le_bytes()).collect();
    let d: Vec<u8> = [2.5f64, -0.125].iter().flat_map(|v| v.to_le_bytes()).collect();

    let file = pt_archive(&p, &[("0", h), ("1", bf), ("2", d)]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 3);
    assert_eq!(r.tensors[0].name, "h");
    assert_eq!(r.tensors[0].data, vec![1.0, -2.0, 0.5]);
    assert_eq!(r.tensors[1].name, "bf");
    assert_eq!(r.tensors[1].data, vec![1.0, -4.0]);
    assert_eq!(r.tensors[2].name, "d");
    assert_eq!(r.tensors[2].data, vec![2.5, -0.125]);
    assert_eq!(r.skipped_non_tensor, 0);
}

/// A stride-swapped 2x3 view with storage_offset 1 materializes contiguous:
/// element [i][j] = storage[1 + i*1 + j*2].
#[test]
fn noncontiguous_strided_view() {
    let p = P::new()
        .op(0x7d)
        .op(0x28)
        .s("v")
        .tensor("FloatStorage", "0", 7, 1, &[2, 3], &[1, 2])
        .op(0x75)
        .stop();
    // storage: [0, 10, 20, 30, 40, 50, 60]
    let file = pt_archive(&p, &[("0", f32_bytes(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 1);
    assert_eq!(r.tensors[0].shape, vec![2, 3]);
    // row 0 (i=0): s[1], s[3], s[5]; row 1 (i=1): s[2], s[4], s[6]
    assert_eq!(r.tensors[0].data, vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0]);
}

/// A 0-dim scalar tensor (empty size/stride) yields shape [] and one value;
/// exercises the singular SETITEM opcode too.
#[test]
fn zero_dim_scalar_tensor() {
    let p = P::new()
        .op(0x7d) // EMPTY_DICT
        .s("count")
        .tensor("FloatStorage", "0", 3, 2, &[], &[])
        .op(0x73) // SETITEM
        .stop();
    let file = pt_archive(&p, &[("0", f32_bytes(&[7.0, 8.0, 9.0]))]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 1);
    assert_eq!(r.tensors[0].name, "count");
    assert!(r.tensors[0].shape.is_empty());
    assert_eq!(r.tensors[0].data, vec![9.0]); // storage[2]
}

/// Two tensors sharing one storage key with different offsets both resolve
/// (the storage is decoded once and sliced per view).
#[test]
fn shared_storage_two_views() {
    let p = P::new()
        .op(0x7d)
        .op(0x28)
        .s("x")
        .tensor("FloatStorage", "0", 10, 2, &[3], &[1])
        .s("y")
        .tensor("FloatStorage", "0", 10, 4, &[5], &[1])
        .op(0x75)
        .stop();
    let storage: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let file = pt_archive(&p, &[("0", f32_bytes(&storage))]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 2);
    assert_eq!(r.tensors[0].name, "x");
    assert_eq!(r.tensors[0].data, vec![2.0, 3.0, 4.0]);
    assert_eq!(r.tensors[1].name, "y");
    assert_eq!(r.tensors[1].data, vec![4.0, 5.0, 6.0, 7.0, 8.0]);
}

/// An int64 storage (LongStorage) is an unsupported dtype: hard error naming
/// the type, never a silent skip.
#[test]
fn unknown_storage_dtype_errors() {
    let p = P::new()
        .op(0x7d)
        .op(0x28)
        .s("idx")
        .tensor("LongStorage", "0", 2, 0, &[2], &[1])
        .op(0x75)
        .stop();
    let file = pt_archive(&p, &[("0", vec![0u8; 16])]);
    let err = torchpt::parse(&file).unwrap_err();
    assert!(err.contains("LongStorage"), "error should name the dtype: {err}");
}

/// A DEFLATE-compressed entry is rejected with a clear error (torch containers
/// are STORED-only; we never mis-slice compressed bytes).
#[test]
fn compressed_entry_errors() {
    let p = P::new()
        .op(0x7d)
        .op(0x28)
        .s("w")
        .tensor("FloatStorage", "0", 1, 0, &[1], &[1])
        .op(0x75)
        .stop();
    let blob = f32_bytes(&[1.0]);
    let file = zip_with(&[
        ("archive/data.pkl", &p, 0, 0),
        ("archive/data/0", &blob, 8, 0), // method 8 = DEFLATE
    ]);
    let err = torchpt::parse(&file).unwrap_err();
    assert!(err.contains("compression method 8"), "unexpected error: {err}");
}

/// Non-tensor leaves (int, float via BINFLOAT, big int via LONG1, str, None,
/// bool, list elements) are skipped silently but each is counted.
#[test]
fn non_tensor_leaves_counted() {
    let p = P::new()
        .op(0x7d) // EMPTY_DICT
        .op(0x28) // MARK
        .s("w")
        .tensor("FloatStorage", "0", 2, 0, &[2], &[1])
        .s("epoch")
        .int(3)
        .s("lr")
        // BINFLOAT 0.5 (f64 big-endian 0x3FE0000000000000)
        .op(0x47).op(0x3F).op(0xE0).op(0x00).op(0x00).op(0x00).op(0x00).op(0x00).op(0x00)
        .s("global_step")
        // LONG1, 5 bytes LE = 0x0100000000 = 4294967296
        .op(0x8a).op(0x05).op(0x00).op(0x00).op(0x00).op(0x00).op(0x01)
        .s("name")
        .s("run-1")
        .s("sched")
        .op(0x4e) // NONE
        .s("amp")
        .op(0x88) // NEWTRUE
        .s("betas")
        .op(0x5d) // EMPTY_LIST
        .op(0x28) // MARK
        .int(9)
        .int(999)
        .op(0x65) // APPENDS
        .op(0x75) // SETITEMS
        .stop();
    let file = pt_archive(&p, &[("0", f32_bytes(&[1.0, 2.0]))]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 1);
    assert_eq!(r.tensors[0].data, vec![1.0, 2.0]);
    // epoch, lr, global_step, name, sched, amp + 2 list elements = 8
    assert_eq!(r.skipped_non_tensor, 8);
}

/// Tensor data offsets follow the *local* header extra length (torch pads
/// local extra fields to 64-byte-align tensor data; the central directory
/// records no extra bytes).
#[test]
fn local_header_extra_padding() {
    let p = P::new()
        .op(0x7d)
        .op(0x28)
        .s("w")
        .tensor("FloatStorage", "0", 2, 0, &[2], &[1])
        .op(0x75)
        .stop();
    let blob = f32_bytes(&[6.25, -7.5]);
    let file = zip_with(&[
        ("archive/data.pkl", &p, 0, 0),
        ("archive/data/0", &blob, 0, 21), // 21 bytes of local extra padding
    ]);
    let r = torchpt::parse(&file).unwrap();
    assert_eq!(r.tensors.len(), 1);
    assert_eq!(r.tensors[0].data, vec![6.25, -7.5]);
}

/// BUILD sets dict *instance attributes* (a real `nn.Module.state_dict()`
/// carries a `_metadata` version table this way). Pure-metadata attributes
/// are dropped (matching torch.load's item view), but an attribute subtree
/// containing a tensor is kept so no tensor is ever lost.
#[test]
fn build_state_metadata_dropped_tensor_kept() {
    let p = P::new()
        .global("collections", "OrderedDict")
        .op(0x29) // EMPTY_TUPLE
        .op(0x52) // REDUCE -> state_dict
        .op(0x28) // MARK
        .s("w")
        .tensor("FloatStorage", "0", 1, 0, &[1], &[1])
        .op(0x75) // SETITEMS
        // BUILD state: {"_metadata": {"": {"version": 1}}, "extra": tensor}
        .op(0x7d) // EMPTY_DICT (state)
        .op(0x28) // MARK
        .s("_metadata")
        .op(0x7d) // EMPTY_DICT
        .s("")
        .op(0x7d) // EMPTY_DICT
        .s("version")
        .int(1)
        .op(0x73) // SETITEM
        .op(0x73) // SETITEM
        .s("extra")
        .tensor("FloatStorage", "1", 1, 0, &[1], &[1])
        .op(0x75) // SETITEMS (into state dict)
        .op(0x62) // BUILD
        .stop();
    let file = pt_archive(&p, &[("0", f32_bytes(&[1.0])), ("1", f32_bytes(&[2.0]))]);
    let r = torchpt::parse(&file).unwrap();
    let names: Vec<&str> = r.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["w", "extra"]);
    assert_eq!(r.tensors[1].data, vec![2.0]);
    assert_eq!(r.skipped_non_tensor, 0); // _metadata version int not counted
}

/// A zip without a data.pkl entry is not a torch checkpoint.
#[test]
fn missing_data_pkl_errors() {
    let file = zip_with(&[("archive/version", b"3\n", 0, 0)]);
    let err = torchpt::parse(&file).unwrap_err();
    assert!(err.contains("data.pkl"), "unexpected error: {err}");
}

/// Garbage bytes are rejected at the zip layer.
#[test]
fn garbage_bytes_error() {
    assert!(torchpt::parse(b"not a zip at all").is_err());
    assert!(torchpt::parse(&[]).is_err());
}
