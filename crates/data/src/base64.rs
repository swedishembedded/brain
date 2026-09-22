// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dependency-free standard base64 (RFC 4648, `+/` alphabet, `=` padding).
//!
//! Lives here, not in `crates/events` (which re-exports this module
//! unchanged as `events::base64` for its own existing callers), because
//! `unigram`'s SentencePiece `Precompiled` normalizer needs it to decode a
//! tokenizer.json's `precompiled_charsmap` field, and `brain-data` is a
//! layer-3 training-substrate crate - depending on `brain-events` for one
//! codec function pulled `brain-forecast` (a layer-4 model crate)
//! transitively into every leaf crate that depends on `brain-data`
//! (`brain-promote`, `brain-rlcd`), which `scripts/gates/check-crate-layers.sh`
//! exists specifically to catch.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes to a standard base64 string (with padding).
pub fn encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { ALPHABET[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[(n & 63) as usize] as char } else { '=' });
    }
    out
}

/// Decode a standard base64 string. Whitespace is ignored; invalid characters
/// or bad padding return `Err` (never panics).
pub fn decode(input: &str) -> Result<Vec<u8>, String> {
    // Reverse map: byte value -> 6-bit sextet, 255 = invalid.
    let val = |c: u8| -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let mut sextets: Vec<u8> = Vec::with_capacity(input.len());
    let mut pad = 0usize;
    for &c in input.as_bytes() {
        match c {
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            b'=' => pad += 1,
            _ => {
                let v = val(c).ok_or_else(|| format!("base64: invalid char {:?}", c as char))?;
                if pad > 0 {
                    return Err("base64: data character after padding".to_string());
                }
                sextets.push(v);
            }
        }
    }
    let mut out = Vec::with_capacity(sextets.len() / 4 * 3);
    for group in sextets.chunks(4) {
        let n = match group.len() {
            4 => ((group[0] as u32) << 18) | ((group[1] as u32) << 12) | ((group[2] as u32) << 6) | (group[3] as u32),
            3 => ((group[0] as u32) << 18) | ((group[1] as u32) << 12) | ((group[2] as u32) << 6),
            2 => ((group[0] as u32) << 18) | ((group[1] as u32) << 12),
            _ => return Err("base64: truncated input (orphan sextet)".to_string()),
        };
        out.push((n >> 16) as u8);
        if group.len() >= 3 {
            out.push((n >> 8) as u8);
        }
        if group.len() == 4 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip_and_known_vectors() {
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"f"), "Zg==");
        assert_eq!(encode(b"fo"), "Zm8=");
        assert_eq!(encode(b"foo"), "Zm9v");
        assert_eq!(encode(b"foob"), "Zm9vYg==");
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
        for s in [b"".to_vec(), b"f".to_vec(), b"foobar".to_vec(), (0u8..=255).collect()] {
            assert_eq!(decode(&encode(&s)).unwrap(), s);
        }
    }

    #[test]
    fn base64_decode_errors() {
        assert!(decode("****").is_err()); // invalid chars
        assert!(decode("Zg=v").is_err()); // data after padding
        assert!(decode("Z").is_err()); // orphan sextet
    }
}
