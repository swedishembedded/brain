// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A writer for synthetic `.cact` files, so reader tests do not depend on the
//! released weights. Deliberately naive: indices are packed one bit at a time.

use mirage::cact::TAG;

/// Lloyd-Max-like unit-sphere codebooks: 2-bit (4), 3-bit (8), 4-bit (16) levels.
pub fn codebooks() -> Vec<f32> {
    let mut v = Vec::new();
    for levels in [4usize, 8, 16] {
        v.extend((0..levels).map(|i| (i as f32 - (levels as f32 - 1.0) / 2.0) * 0.1));
    }
    v
}

/// One tensor to place in a synthetic file.
pub enum Blob {
    F16(Vec<u16>, Vec<u32>),
    F32(Vec<f32>, Vec<u32>),
    Raw(Vec<u8>),
    /// rows, cols, bits, group, indices (rows x padded cols), norms (rows x groups)
    Cq { rows: usize, cols: usize, bits: u32, group: usize, indices: Vec<u8>, norms: Vec<u16> },
}

/// Pack indices LSB-first, one bit at a time.
pub fn pack_bits(indices: &[u8], bits: u32) -> Vec<u8> {
    let mut out = vec![0u8; indices.len() * bits as usize / 8];
    for (i, &idx) in indices.iter().enumerate() {
        for b in 0..bits as usize {
            if (idx >> b) & 1 == 1 {
                let bit = i * bits as usize + b;
                out[bit / 8] |= 1 << (bit % 8);
            }
        }
    }
    out
}

pub fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

pub fn build(blobs: &[Blob]) -> Vec<u8> {
    build_with_header(blobs, &[], 100000.0)
}

/// Like [`build`], with header `u32` fields set (index, value) and a rotary base.
pub fn build_with_header(blobs: &[Blob], fields: &[(usize, u32)], rope_theta: f32) -> Vec<u8> {
    let mut header = [0u32; 48];
    header[0] = TAG;
    header[1] = blobs.len() as u32;
    header[2] = 28;
    for &(i, v) in fields {
        header[i] = v;
    }
    let mut file: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
    file.extend(rope_theta.to_le_bytes());
    file.extend(codebooks().iter().flat_map(|v| v.to_le_bytes()));
    assert_eq!(file.len(), 308);
    let dir_end = 308 + 44 * blobs.len();
    let mut offset = (dir_end + 63) & !63;
    let mut directory = Vec::new();
    let mut data = Vec::new();
    for blob in blobs {
        let (dtype, ndim, shape, bytes, group, bits): (u8, u8, [u32; 4], Vec<u8>, u32, u32) = match blob {
            Blob::F16(v, shape) => (1, shape.len() as u8, pad_shape(shape), v.iter().flat_map(|x| x.to_le_bytes()).collect(), 0, 0),
            Blob::F32(v, shape) => (2, shape.len() as u8, pad_shape(shape), v.iter().flat_map(|x| x.to_le_bytes()).collect(), 0, 0),
            Blob::Raw(b) => (4, 0, [0; 4], b.clone(), 0, 0),
            Blob::Cq { rows, cols, bits, group, indices, norms } => {
                let mut bytes = pack_bits(indices, *bits);
                bytes.extend(norms.iter().flat_map(|x| x.to_le_bytes()));
                (3, 2, pad_shape(&[*rows as u32, *cols as u32]), bytes, *group as u32, *bits)
            }
        };
        directory.push(dtype);
        directory.push(ndim);
        directory.extend(0u16.to_le_bytes());
        for d in shape {
            directory.extend(d.to_le_bytes());
        }
        directory.extend((offset as u64).to_le_bytes());
        directory.extend((bytes.len() as u64).to_le_bytes());
        directory.extend(group.to_le_bytes());
        directory.extend(bits.to_le_bytes());
        data.extend(&bytes);
        let padded = (bytes.len() + 63) & !63;
        data.extend(vec![0u8; padded - bytes.len()]);
        offset += padded;
    }
    file.extend(directory);
    file.resize((dir_end + 63) & !63, 0);
    file.extend(data);
    file
}

fn pad_shape(shape: &[u32]) -> [u32; 4] {
    let mut out = [0u32; 4];
    out[..shape.len()].copy_from_slice(shape);
    out
}

