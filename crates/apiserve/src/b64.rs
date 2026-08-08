// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal, dependency-free standard base64 (RFC 4648, with padding)
//! decoder — what OpenAI's `image_url` data URLs and `input_audio.data`,
//! and Anthropic's `image.source.data`, all carry. No workspace crate
//! already depends on a `base64` crate (checked: `grep base64 Cargo.lock`
//! finds nothing), and decoding is a small, bounded, well-defined
//! operation — matches this codebase's general preference for a minimal
//! dependency footprint over pulling in a crate for one function (see
//! `audio::wav`'s own "dependency-light" doc for the same convention
//! elsewhere).

/// Decode a standard base64 string (with or without `=` padding; whitespace
/// and newlines tolerated, matching real-world data URLs that sometimes wrap
/// long lines). Returns `Err` on an invalid character or truncated input —
/// never silently drops or truncates bytes.
pub fn decode(s: &str) -> Result<Vec<u8>, String> {
    let mut vals: Vec<u8> = Vec::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_whitespace() {
            continue;
        }
        if b == b'=' {
            break; // padding: only ever trails real data
        }
        vals.push(match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("base64: invalid character {:?}", b as char)),
        });
    }
    let mut out = Vec::with_capacity(vals.len() * 3 / 4 + 3);
    for chunk in vals.chunks(4) {
        let n = chunk.len();
        if n == 1 {
            return Err("base64: truncated input (a single trailing symbol can't decode to a byte)".to_string());
        }
        let b0 = chunk[0] as u32;
        let b1 = chunk[1] as u32;
        out.push(((b0 << 2) | (b1 >> 4)) as u8);
        if n >= 3 {
            let b2 = chunk[2] as u32;
            out.push((((b1 & 0xF) << 4) | (b2 >> 2)) as u8);
        }
        if n == 4 {
            let b2 = chunk[2] as u32;
            let b3 = chunk[3] as u32;
            out.push((((b2 & 0x3) << 6) | b3) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_rfc_4648_test_vectors() {
        let cases: &[(&str, &str)] = &[("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")];
        for (want, b64) in cases {
            let got = decode(b64).unwrap();
            assert_eq!(got, want.as_bytes(), "decoding {b64:?}");
        }
    }

    #[test]
    fn tolerates_embedded_whitespace_and_missing_padding() {
        assert_eq!(decode("Zm9v\nYmFy").unwrap(), b"foobar");
        assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar"); // no trailing = at all
    }

    #[test]
    fn rejects_invalid_characters_and_truncation() {
        assert!(decode("not!valid").is_err());
        assert_eq!(decode("Zg").unwrap(), b"f"); // 2 symbols decode to 1 byte, even unpadded
        assert!(decode("Z").is_err()); // 1 trailing symbol alone can never decode to a whole byte
    }

    #[test]
    fn round_trips_real_binary_data() {
        // A few PNG magic bytes, base64-encoded by hand against a known-good
        // encoder -- catches a byte-order/shift bug an all-ASCII test vector
        // could miss.
        let png_magic = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let b64 = "iVBORw0KGgo=";
        assert_eq!(decode(b64).unwrap(), png_magic);
    }
}
