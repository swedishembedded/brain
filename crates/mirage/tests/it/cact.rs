// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `.cact` container reader and its Cactus-Quant decoder, on synthetic
//! files whose expected values are computed here by a deliberately naive
//! route (an explicit Walsh matrix, a bit-by-bit packer), plus hostile-input
//! behaviour: no byte sequence may make the reader panic or read out of range.

use mirage::cact::{CactError, CactFile, EntryDtype};

use crate::cactgen::{build, codebooks, f16_bits, pack_bits, Blob};

/// `(codebook[idx] * norm) @ H` with `H` the natural-order Walsh matrix over
/// `group`, normalised by `1/sqrt(group)`; rows are padded to whole groups.
fn expected_rows(rows: usize, cols: usize, bits: u32, group: usize, indices: &[u8], norms: &[u16]) -> Vec<f32> {
    let book: Vec<f32> = match bits {
        2 => codebooks()[0..4].to_vec(),
        3 => codebooks()[4..12].to_vec(),
        _ => codebooks()[12..28].to_vec(),
    };
    let groups = cols.div_ceil(group);
    let padded = groups * group;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for g in 0..groups {
            let norm = half::f16::from_bits(norms[r * groups + g]).to_f32() as f64;
            for j in 0..group {
                let mut acc = 0f64;
                for k in 0..group {
                    let sign = if (k & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                    acc += book[indices[r * padded + g * group + k] as usize] as f64 * norm * sign;
                }
                let col = g * group + j;
                if col < cols {
                    out[r * cols + col] = (acc / (group as f64).sqrt()) as f32;
                }
            }
        }
    }
    out
}

fn cq_blob(rows: usize, cols: usize, bits: u32, group: usize, seed: u32) -> Blob {
    let groups = cols.div_ceil(group);
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s >> 8
    };
    let indices = (0..rows * groups * group).map(|_| (next() % (1 << bits)) as u8).collect();
    let norms = (0..rows * groups).map(|_| f16_bits(0.25 + (next() % 100) as f32 / 50.0)).collect();
    Blob::Cq { rows, cols, bits, group, indices, norms }
}

#[test]
fn small_tensors_and_raw_blobs_read_back() {
    let bytes = build(&[
        Blob::F16(vec![f16_bits(1.5), f16_bits(-2.0), f16_bits(0.25)], vec![3]),
        Blob::F32(vec![0.5, -1.0, 7.0, 8.0], vec![2, 2]),
        Blob::Raw(b"opaque".to_vec()),
    ]);
    let f = CactFile::parse(bytes).unwrap();
    assert_eq!(f.entries().len(), 3);
    assert_eq!(f.entries()[0].dtype, EntryDtype::Fp16);
    assert_eq!(f.f32_tensor(0).unwrap(), vec![1.5, -2.0, 0.25]);
    assert_eq!(f.entries()[1].shape, vec![2, 2]);
    assert_eq!(f.f32_tensor(1).unwrap(), vec![0.5, -1.0, 7.0, 8.0]);
    assert_eq!(f.raw(2).unwrap(), b"opaque");
    assert!(f.raw(0).is_err() && f.f32_tensor(2).is_err(), "a blob is not a tensor and vice versa");
    assert_eq!(f.header().rope_theta, 100000.0);
    assert_eq!(f.header().num_tensors(), 3);
}

#[test]
fn cq_matrices_decode_to_the_naive_reference_at_every_width() {
    for (bits, rows, cols, group) in [(2u32, 3usize, 128usize, 128usize), (3, 2, 130, 128), (4, 4, 256, 128), (2, 2, 40, 16), (4, 1, 8, 8)] {
        let blob = cq_blob(rows, cols, bits, group, bits * 31 + cols as u32);
        let Blob::Cq { indices, norms, .. } = &blob else { unreachable!() };
        let want = expected_rows(rows, cols, bits, group, indices, norms);
        let f = CactFile::parse(build(&[blob])).unwrap();
        let got = f.f32_tensor(0).unwrap();
        assert_eq!(got.len(), rows * cols);
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "bits {bits} cols {cols} group {group}: worst error {worst}");
    }
}

#[test]
fn the_packed_form_stays_available_for_native_kernels() {
    let blob = cq_blob(3, 130, 2, 128, 5);
    let Blob::Cq { indices, norms, .. } = &blob else { unreachable!() };
    let (indices, norms) = (indices.clone(), norms.clone());
    let f = CactFile::parse(build(&[blob])).unwrap();
    let m = f.cq(0).unwrap();
    assert_eq!((m.rows, m.cols, m.padded_cols, m.bits, m.group), (3, 130, 256, 2, 128));
    assert_eq!(m.row_packed(1), &pack_bits(&indices[256..512], 2)[..]);
    assert_eq!(m.norm(2, 1), half::f16::from_bits(norms[5]).to_f32());
    assert_eq!(m.codebook.len(), 4);
    let mut row = vec![0u8; 256];
    m.unpack_row(1, &mut row);
    assert_eq!(row, indices[256..512]);
    let mut decoded = vec![0f32; 130];
    m.decode_row(1, &mut decoded);
    assert_eq!(decoded, f.f32_tensor(0).unwrap()[130..260]);
}

#[test]
fn a_cq_tensor_is_not_accepted_as_anything_else() {
    let f = CactFile::parse(build(&[Blob::F32(vec![1.0], vec![1])])).unwrap();
    assert!(f.cq(0).is_err());
}

/// Parse `bytes`, then try every accessor on every entry; only errors may result.
fn exercise(bytes: Vec<u8>) {
    if let Ok(f) = CactFile::parse(bytes) {
        for i in 0..f.entries().len() {
            let _ = f.f32_tensor(i);
            let _ = f.raw(i);
            let _ = f.cq(i);
        }
    }
}

fn sample() -> Vec<u8> {
    build(&[Blob::F16(vec![f16_bits(1.0); 5], vec![5]), cq_blob(2, 130, 3, 128, 9), Blob::F32(vec![1.0; 6], vec![3, 2]), Blob::Raw(vec![7; 33])])
}

#[test]
fn no_truncation_of_a_valid_file_panics_or_parses_wrongly() {
    let full = sample();
    assert!(CactFile::parse(full.clone()).is_ok());
    for len in 0..full.len() {
        exercise(full[..len].to_vec());
    }
    // A file cut inside any blob is refused, not read short. The last blob is
    // 33 bytes followed by 31 bytes of alignment padding, which may be missing.
    let last_blob_end = full.len() - 31;
    for len in 0..last_blob_end {
        assert!(CactFile::parse(full[..len].to_vec()).is_err(), "a file cut at {len} of {} parsed", full.len());
    }
    assert!(CactFile::parse(full[..last_blob_end].to_vec()).is_ok());
}

#[test]
fn single_byte_corruption_never_panics() {
    let full = sample();
    for at in 0..(308 + 44 * 4 + 16) {
        for value in [0u8, 1, 0x7f, 0x80, 0xff] {
            let mut b = full.clone();
            b[at] = value;
            exercise(b);
        }
    }
}

fn mutate_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn mutate_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn expect_err(bytes: Vec<u8>, needle: &str) {
    let err = CactFile::parse(bytes).map(|_| ()).unwrap_err();
    assert!(err.to_string().contains(needle), "expected `{needle}` in `{err}`");
}

#[test]
fn structural_lies_are_refused_with_a_reason() {
    let ok = sample();
    let rec = |i: usize| 308 + 44 * i;

    let mut b = ok.clone();
    mutate_u32(&mut b, 0, 0xdeadbeef);
    expect_err(b, "tag");

    let mut b = ok.clone();
    mutate_u32(&mut b, 8, 12);
    expect_err(b, "codebook");

    let mut b = ok.clone();
    mutate_u32(&mut b, 4, 0xffff_ffff);
    expect_err(b, "tensors");

    // Directory record layout: dtype u8, ndim u8, pad u16, shape[4] u32 at +4,
    // offset u64 at +20, nbytes u64 at +28, group u32 at +36, bits u32 at +40.
    let mut b = ok.clone();
    b[rec(0)] = 9;
    expect_err(b, "dtype");

    let mut b = ok.clone();
    b[rec(0) + 1] = 5;
    expect_err(b, "ndim");

    let mut b = ok.clone();
    mutate_u32(&mut b, rec(0) + 8, 3);
    expect_err(b, "shape");

    let mut b = ok.clone();
    mutate_u64(&mut b, rec(0) + 20, u64::MAX - 3);
    expect_err(b, "outside");

    let mut b = ok.clone();
    mutate_u64(&mut b, rec(0) + 20, 0);
    expect_err(b, "directory");

    let mut b = ok.clone();
    mutate_u64(&mut b, rec(0) + 28, 3);
    expect_err(b, "bytes");

    let mut b = ok.clone();
    mutate_u64(&mut b, rec(1) + 28, 1 << 40);
    expect_err(b, "outside");

    for bits in [0u32, 1, 5, 7, 32] {
        let mut b = ok.clone();
        mutate_u32(&mut b, rec(1) + 40, bits);
        expect_err(b, "bits");
    }
    for group in [0u32, 7, 100, 1 << 30] {
        let mut b = ok.clone();
        mutate_u32(&mut b, rec(1) + 36, group);
        expect_err(b, "group");
    }

    // A shape whose element count overflows is refused rather than wrapped.
    let mut b = ok.clone();
    mutate_u32(&mut b, rec(1) + 4, u32::MAX);
    mutate_u32(&mut b, rec(1) + 8, u32::MAX);
    assert!(CactFile::parse(b).is_err());
}

#[test]
fn a_file_too_large_to_be_a_model_is_refused_before_it_is_read() {
    let dir = std::env::temp_dir().join(format!("brain-mirage-cact-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("huge.cact");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(mirage::cact::MAX_FILE_BYTES + 1).unwrap();
    let err = CactFile::open(&path).map(|_| ()).unwrap_err();
    assert!(matches!(err, CactError::Header(_)) && err.to_string().contains("too large"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}
