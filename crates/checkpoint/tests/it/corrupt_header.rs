// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A safetensors file's first 8 bytes are a length the reader then trusts to
//! size a buffer or a slice. A corrupt or foreign file makes that length
//! arbitrary, and trusting it turns "this file is bad" into an allocation of
//! up to 2^64 bytes, which the process cannot survive. Every reader must
//! refuse a length longer than the file or than the format's header cap
//! with [`HeaderLenError`], naming the file, the claimed length and the
//! file's size, and must still read a valid file.

use checkpoint::safetensors::{HeaderLenError, MAX_HEADER_BYTES};
use checkpoint::{mmap::MmapSafetensors, safetensors as stf, st};
use serde_json::json;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

fn valid_file(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-corrupt-header-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let tensors = [("w".to_string(), vec![2u64, 2], vec![1.0f32, 2.0, 3.0, 4.0])];
    st::save_safetensors(path.to_str().unwrap(), &tensors, &json!({"k": 1}), None).unwrap();
    path
}

/// Overwrite the length prefix of `path` with `claimed`, optionally growing the
/// file (sparsely) to `grow_to` bytes first.
fn corrupt(path: &PathBuf, claimed: u64, grow_to: Option<u64>) {
    let mut f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    if let Some(len) = grow_to {
        f.set_len(len).unwrap();
    }
    f.seek(SeekFrom::Start(0)).unwrap();
    f.write_all(&claimed.to_le_bytes()).unwrap();
}

fn assert_io_refused(r: Result<impl std::fmt::Debug, std::io::Error>, what: &str, want: &HeaderLenError) {
    let e = r.expect_err(&format!("{what}: a corrupt header length must be refused"));
    let got = e.get_ref().and_then(|inner| inner.downcast_ref::<HeaderLenError>());
    assert_eq!(got, Some(want), "{what}: expected the typed header-length error, got {e}");
}

fn assert_str_refused(r: Result<impl std::fmt::Debug, String>, what: &str, want: &HeaderLenError) {
    let e = r.expect_err(&format!("{what}: a corrupt header length must be refused"));
    assert!(e.contains(&want.to_string()), "{what}: expected `{want}`, got `{e}`");
}

#[test]
fn every_reader_refuses_a_header_length_the_file_cannot_hold() {
    let base = valid_file("sizes");
    let size = std::fs::metadata(&base).unwrap().len();
    // The cap case needs a file big enough that only the cap is violated; a
    // sparse file costs no disk.
    let big = size.max(MAX_HEADER_BYTES + 16);
    let cases = [("u64::MAX", u64::MAX, None, size), ("file_size+1", size + 1, None, size), ("cap+1", MAX_HEADER_BYTES + 1, Some(big), big)];
    for (label, claimed, grow_to, file_len) in cases {
        let path = valid_file(label.replace(['+', ':'], "_").as_str());
        corrupt(&path, claimed, grow_to);
        let p = path.to_str().unwrap();
        let want = HeaderLenError { file: p.to_string(), claimed, file_len };
        assert!(want.to_string().contains(p), "{label}: the error names the file");
        assert!(want.to_string().contains(&claimed.to_string()), "{label}: the error names the claimed length");
        assert!(want.to_string().contains(&file_len.to_string()), "{label}: the error names the file size");

        assert_io_refused(st::read_metadata(p), &format!("{label} read_metadata"), &want);
        assert_io_refused(st::read_card(p), &format!("{label} read_card"), &want);
        assert_io_refused(st::param_count_from_header(p), &format!("{label} param_count_from_header"), &want);
        assert_io_refused(st::declared_data_extent(p), &format!("{label} declared_data_extent"), &want);
        // The mapped readers report through `String` errors; the message is
        // the same typed error's.
        assert_str_refused(st::load_safetensors(p).map(|_| ()).map_err(|e| e.to_string()), &format!("{label} load_safetensors"), &want);
        assert_str_refused(MmapSafetensors::open(p).map(|_| ()), &format!("{label} MmapSafetensors::open"), &want);
        assert_str_refused(stf::read(p).map(|_| ()), &format!("{label} safetensors::read"), &want);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
    // An in-memory buffer has no file name; it is still refused, not sliced.
    let mut bytes = std::fs::read(&base).unwrap();
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let e = stf::parse(&bytes).map(|_| ()).expect_err("a corrupt buffer is refused");
    assert!(e.contains(&u64::MAX.to_string()), "the buffer error names the claimed length: {e}");

    // The valid file still reads through every path.
    let p = base.to_str().unwrap();
    assert_eq!(st::param_count_from_header(p).unwrap(), 4);
    assert!(st::read_metadata(p).unwrap().contains_key("brain.config"));
    assert_eq!(st::declared_data_extent(p).unwrap(), size);
    assert_eq!(st::load_safetensors(p).unwrap().tensors["w"], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(MmapSafetensors::open(p).unwrap().names().len(), 1);
    assert_eq!(stf::read(p).unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}
