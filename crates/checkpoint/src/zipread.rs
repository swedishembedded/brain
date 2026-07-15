// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal ZIP-container reader for PyTorch `torch.save` archives.
//!
//! Torch (>= 1.6) writes checkpoints as an *uncompressed* ZIP: every entry is
//! STORED (method 0), so tensor bytes can be sliced straight out of the file.
//! We locate the end-of-central-directory record, walk the central directory,
//! and resolve each entry's data range via its *local* header — torch pads
//! local extra fields to 64-byte-align tensor data, so the local name/extra
//! lengths (not the central ones) determine where the data starts.
//!
//! Compressed entries are rejected with a clear error rather than skipped:
//! callers rely on a full-coverage guarantee.

/// One archive entry: `bytes[offset..offset + len]` is its (stored) data.
pub struct ZipEntry {
    pub name: String,
    pub offset: usize,
    pub len: usize,
}

/// Read `len` bytes at `at`, erroring (never panicking) on truncation.
fn take<'a>(bytes: &'a [u8], at: usize, len: usize, what: &str) -> Result<&'a [u8], String> {
    bytes
        .get(at..at + len)
        .ok_or_else(|| format!("zip: truncated file reading {what} at offset {at}"))
}

fn rd16(bytes: &[u8], at: usize, what: &str) -> Result<u16, String> {
    let b = take(bytes, at, 2, what)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn rd32(bytes: &[u8], at: usize, what: &str) -> Result<u32, String> {
    let b = take(bytes, at, 4, what)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Parse the central directory of a ZIP archive into named data ranges.
///
/// Errors on: missing/garbled end-of-central-directory, zip64 archives
/// (torch only emits those beyond 4 GiB), compressed (non-STORED) entries,
/// and any truncation.
pub fn parse(bytes: &[u8]) -> Result<Vec<ZipEntry>, String> {
    // End-of-central-directory: fixed 22 bytes + variable comment, found by
    // scanning backwards for its signature within the last 64 KiB + 22 bytes.
    if bytes.len() < 22 {
        return Err("zip: file too short for an end-of-central-directory record".into());
    }
    let scan_floor = bytes.len().saturating_sub(22 + 65535);
    let mut eocd = None;
    for i in (scan_floor..=bytes.len() - 22).rev() {
        if bytes[i..i + 4] == *b"PK\x05\x06" {
            eocd = Some(i);
            break;
        }
    }
    let e = eocd.ok_or("zip: end-of-central-directory record not found")?;
    let n_entries = rd16(bytes, e + 10, "EOCD entry count")?;
    let cd_size = rd32(bytes, e + 12, "EOCD central directory size")?;
    let cd_off = rd32(bytes, e + 16, "EOCD central directory offset")?;
    if n_entries == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        return Err("zip: zip64 archives are not supported".into());
    }

    let mut out = Vec::with_capacity(n_entries as usize);
    let mut pos = cd_off as usize;
    for i in 0..n_entries {
        if take(bytes, pos, 4, "central directory signature")? != b"PK\x01\x02" {
            return Err(format!("zip: bad central directory signature for entry {i} at offset {pos}"));
        }
        let method = rd16(bytes, pos + 10, "entry method")?;
        let comp_size = rd32(bytes, pos + 20, "entry compressed size")? as usize;
        let uncomp_size = rd32(bytes, pos + 24, "entry uncompressed size")? as usize;
        let name_len = rd16(bytes, pos + 28, "entry name length")? as usize;
        let extra_len = rd16(bytes, pos + 30, "entry extra length")? as usize;
        let comment_len = rd16(bytes, pos + 32, "entry comment length")? as usize;
        let local_off = rd32(bytes, pos + 42, "entry local header offset")? as usize;
        let name = String::from_utf8(take(bytes, pos + 46, name_len, "entry name")?.to_vec())
            .map_err(|_| format!("zip: non-utf8 entry name at offset {pos}"))?;
        if comp_size == 0xFFFF_FFFF || uncomp_size == 0xFFFF_FFFF || local_off == 0xFFFF_FFFF {
            return Err(format!("zip: zip64 entry '{name}' is not supported"));
        }
        if method != 0 {
            return Err(format!(
                "zip: entry '{name}' uses compression method {method}; torch containers use STORED (0) only"
            ));
        }
        if comp_size != uncomp_size {
            return Err(format!(
                "zip: stored entry '{name}' has mismatched sizes ({comp_size} != {uncomp_size})"
            ));
        }

        // Data offset comes from the *local* header lengths (torch alignment
        // padding lives in the local extra field, not the central one).
        if take(bytes, local_off, 4, "local header signature")? != b"PK\x03\x04" {
            return Err(format!("zip: bad local header signature for entry '{name}'"));
        }
        let l_name_len = rd16(bytes, local_off + 26, "local name length")? as usize;
        let l_extra_len = rd16(bytes, local_off + 28, "local extra length")? as usize;
        let data_off = local_off + 30 + l_name_len + l_extra_len;
        if data_off + comp_size > bytes.len() {
            return Err(format!("zip: entry '{name}' data range extends past end of file"));
        }
        out.push(ZipEntry { name, offset: data_off, len: comp_size });
        pos += 46 + name_len + extra_len + comment_len;
    }
    Ok(out)
}
