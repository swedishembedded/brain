// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A minimal, dependency-free PNG encoder: 8-bit truecolor RGB, one *stored*
//! (uncompressed) DEFLATE block stream. brain has no image-codec dependency at the
//! API layer, but OpenAI's `b64_json` must be a real PNG — so this turns the raw
//! HWC image a model returns (see `capability::blob`) into a valid, if uncompressed,
//! PNG. Correctness over size; the served images are modest and the bytes are
//! base64'd once. If a compact codec is ever needed here, swap `zlib_stored` for a
//! real deflate — the chunk framing stays the same.

/// Standard PNG CRC-32 (poly `0xEDB88320`, init/final `0xFFFFFFFF`).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Adler-32 of the uncompressed data (the zlib stream's trailer).
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b): (u32, u32) = (1, 0);
    for &x in bytes {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Append one PNG chunk (`len | type | data | crc(type+data)`).
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = Vec::with_capacity(4 + data.len());
    crc_in.extend_from_slice(kind);
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

/// Wrap `raw` in a zlib stream of *stored* DEFLATE blocks (no compression): a
/// 2-byte zlib header, one block per ≤65535-byte run, then the Adler-32 trailer.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // CMF/FLG: deflate, 32K window, check ok (0x7801 % 31 == 0)
    let mut i = 0;
    loop {
        let end = (i + 0xFFFF).min(raw.len());
        let block = &raw[i..end];
        let last = end == raw.len();
        out.push(if last { 1 } else { 0 }); // BFINAL bit, BTYPE=00 (stored)
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes()); // NLEN = ones-complement of LEN
        out.extend_from_slice(block);
        i = end;
        if last {
            break;
        }
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// Encode interleaved 8-bit RGB pixels (`w·h·3` bytes, row-major) as a PNG.
pub fn encode_rgb8(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    debug_assert_eq!(rgb.len(), w as usize * h as usize * 3);
    // PNG scanlines: each row is a filter-type byte (0 = none) then w·3 sample bytes.
    let stride = w as usize * 3;
    let mut raw = Vec::with_capacity(h as usize * (1 + stride));
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * stride..y * stride + stride]);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]); // PNG signature
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // bit depth 8, color type 2 (RGB), deflate/adaptive/no-interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// The 8-byte PNG signature — used to detect a blob that is already a PNG.
pub const SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_and_adler_match_known_vectors() {
        // Known CRC-32 of "123456789" is 0xCBF43926; Adler-32 is 0x091E01DE.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"123456789"), 0x091E_01DE);
    }

    #[test]
    fn encodes_a_structurally_valid_png() {
        // 2×2 RGB.
        let rgb: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let png = encode_rgb8(&rgb, 2, 2);
        assert_eq!(&png[0..8], &SIGNATURE);
        // IHDR immediately follows the signature: len=13 then "IHDR" then w,h.
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes([png[16], png[17], png[18], png[19]]), 2); // width
        assert_eq!(u32::from_be_bytes([png[20], png[21], png[22], png[23]]), 2); // height
        // The stream ends with IEND.
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }
}
