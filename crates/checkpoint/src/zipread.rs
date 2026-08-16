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
//!
//! **ZIP64.** A container whose central directory starts past 4 GiB, or whose
//! entries do, replaces the overflowing 32-bit fields with the sentinel
//! `0xFFFF_FFFF` and carries the real 64-bit values in a zip64
//! end-of-central-directory record (`PK\x06\x06`, reached through the locator
//! `PK\x06\x07` that sits immediately before the ordinary EOCD) and in a
//! per-entry "zip64 extended information" extra field (header id `0x0001`).
//! Both are read here. This is not an exotic case: **every** torch checkpoint
//! over 4 GB is one, which is most modern text encoders - Wan2.1's 11.4 GB
//! umT5-XXL encoder among them.

/// One archive entry: `bytes[offset..offset + len]` is its (stored) data.
#[derive(Debug)]
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

fn rd64(bytes: &[u8], at: usize, what: &str) -> Result<u64, String> {
    let b = take(bytes, at, 8, what)?;
    Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

/// The 32-bit field value that means "the real one is 64 bits, elsewhere".
const Z64: u32 = 0xFFFF_FFFF;

/// Pull the 64-bit replacements for whichever of `(uncompressed, compressed,
/// local header offset)` are the `0xFFFF_FFFF` sentinel out of a central
/// directory entry's extra field.
///
/// The zip64 extended-information field packs only the overflowing members, in
/// this fixed order, so which u64 sits where depends on which of the three were
/// sentinels - reading them positionally without that check is the classic way
/// to get a plausible-looking wrong offset.
fn zip64_extra(
    extra: &[u8],
    name: &str,
    uncomp: &mut usize,
    comp: &mut usize,
    local_off: &mut usize,
) -> Result<(), String> {
    let want = (*uncomp as u32 == Z64) as usize
        + (*comp as u32 == Z64) as usize
        + (*local_off as u32 == Z64) as usize;
    if want == 0 {
        return Ok(());
    }
    let mut p = 0usize;
    while p + 4 <= extra.len() {
        let id = rd16(extra, p, "extra header id")?;
        let size = rd16(extra, p + 2, "extra data size")? as usize;
        let body = extra
            .get(p + 4..p + 4 + size)
            .ok_or_else(|| format!("zip: entry '{name}' has a truncated extra field"))?;
        if id == 0x0001 {
            if body.len() < want * 8 {
                return Err(format!(
                    "zip: entry '{name}' needs {want} zip64 field(s) but its extended \
                     information block holds only {} bytes",
                    body.len()
                ));
            }
            let mut q = 0usize;
            for (field, what) in [
                (&mut *uncomp, "uncompressed size"),
                (&mut *comp, "compressed size"),
                (&mut *local_off, "local header offset"),
            ] {
                if *field as u32 == Z64 {
                    let v = rd64(body, q, what)?;
                    *field = usize::try_from(v)
                        .map_err(|_| format!("zip: entry '{name}' {what} {v} exceeds usize"))?;
                    q += 8;
                }
            }
            return Ok(());
        }
        p += 4 + size;
    }
    Err(format!("zip: entry '{name}' uses zip64 sentinels but carries no 0x0001 extra field"))
}

/// Parse the central directory of a ZIP archive into named data ranges.
///
/// Errors on: missing/garbled end-of-central-directory, compressed
/// (non-STORED) entries, and any truncation. ZIP64 containers (every torch
/// checkpoint beyond 4 GiB) are read, not rejected.
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
    let mut n_entries = rd16(bytes, e + 10, "EOCD entry count")? as u64;
    let mut cd_off = rd32(bytes, e + 16, "EOCD central directory offset")? as usize;
    let overflowed = n_entries == 0xFFFF || cd_off as u32 == Z64;
    // The zip64 locator sits immediately BEFORE the EOCD when one exists. Look
    // for it whenever a field overflowed; a container that overflowed without
    // one is corrupt, and reading the sentinel as an offset would seek to
    // 4 GiB - 1 and fail somewhere far less obvious.
    if overflowed {
        let loc = e
            .checked_sub(20)
            .filter(|&l| bytes[l..l + 4] == *b"PK\x06\x07")
            .ok_or("zip: zip64 sentinels in the EOCD but no zip64 locator before it")?;
        let z64 = usize::try_from(rd64(bytes, loc + 8, "zip64 EOCD offset")?)
            .map_err(|_| "zip: zip64 EOCD offset exceeds usize".to_string())?;
        if take(bytes, z64, 4, "zip64 EOCD signature")? != b"PK\x06\x06" {
            return Err("zip: zip64 end-of-central-directory record not found at the locator".into());
        }
        n_entries = rd64(bytes, z64 + 32, "zip64 EOCD entry count")?;
        cd_off = usize::try_from(rd64(bytes, z64 + 48, "zip64 central directory offset")?)
            .map_err(|_| "zip: zip64 central directory offset exceeds usize".to_string())?;
    }

    let mut out = Vec::with_capacity(n_entries.min(1 << 20) as usize);
    let mut pos = cd_off;
    for i in 0..n_entries {
        if take(bytes, pos, 4, "central directory signature")? != b"PK\x01\x02" {
            return Err(format!("zip: bad central directory signature for entry {i} at offset {pos}"));
        }
        let method = rd16(bytes, pos + 10, "entry method")?;
        let mut comp_size = rd32(bytes, pos + 20, "entry compressed size")? as usize;
        let mut uncomp_size = rd32(bytes, pos + 24, "entry uncompressed size")? as usize;
        let name_len = rd16(bytes, pos + 28, "entry name length")? as usize;
        let extra_len = rd16(bytes, pos + 30, "entry extra length")? as usize;
        let comment_len = rd16(bytes, pos + 32, "entry comment length")? as usize;
        let mut local_off = rd32(bytes, pos + 42, "entry local header offset")? as usize;
        let name = String::from_utf8(take(bytes, pos + 46, name_len, "entry name")?.to_vec())
            .map_err(|_| format!("zip: non-utf8 entry name at offset {pos}"))?;
        let extra = take(bytes, pos + 46 + name_len, extra_len, "entry extra field")?;
        zip64_extra(extra, &name, &mut uncomp_size, &mut comp_size, &mut local_off)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn le16(v: u16) -> [u8; 2] {
        v.to_le_bytes()
    }
    fn le32(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn le64(v: u64) -> [u8; 8] {
        v.to_le_bytes()
    }

    /// One STORED entry, with the central directory's local-header offset
    /// optionally written as the zip64 sentinel + a 0x0001 extra field. Nothing
    /// here is 4 GiB long: the point is the ENCODING, and a real 11 GB
    /// checkpoint cannot be a unit test.
    fn archive(name: &str, data: &[u8], zip64: bool) -> Vec<u8> {
        let mut b = Vec::new();
        let local_off = b.len() as u32;
        b.extend_from_slice(b"PK\x03\x04");
        b.extend_from_slice(&le16(20)); // version needed
        b.extend_from_slice(&le16(0)); // flags
        b.extend_from_slice(&le16(0)); // method: STORED
        b.extend_from_slice(&le16(0)); // time
        b.extend_from_slice(&le16(0)); // date
        b.extend_from_slice(&le32(0)); // crc
        b.extend_from_slice(&le32(data.len() as u32));
        b.extend_from_slice(&le32(data.len() as u32));
        b.extend_from_slice(&le16(name.len() as u16));
        b.extend_from_slice(&le16(0)); // local extra
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(data);

        let cd_off = b.len() as u32;
        let extra: Vec<u8> = if zip64 {
            let mut e = Vec::new();
            e.extend_from_slice(&le16(0x0001));
            e.extend_from_slice(&le16(8));
            e.extend_from_slice(&le64(local_off as u64));
            e
        } else {
            Vec::new()
        };
        b.extend_from_slice(b"PK\x01\x02");
        b.extend_from_slice(&le16(20)); // version made by
        b.extend_from_slice(&le16(20)); // version needed
        b.extend_from_slice(&le16(0));
        b.extend_from_slice(&le16(0)); // method
        b.extend_from_slice(&le16(0));
        b.extend_from_slice(&le16(0));
        b.extend_from_slice(&le32(0)); // crc
        b.extend_from_slice(&le32(data.len() as u32));
        b.extend_from_slice(&le32(data.len() as u32));
        b.extend_from_slice(&le16(name.len() as u16));
        b.extend_from_slice(&le16(extra.len() as u16));
        b.extend_from_slice(&le16(0)); // comment
        b.extend_from_slice(&le16(0)); // disk
        b.extend_from_slice(&le16(0)); // internal attrs
        b.extend_from_slice(&le32(0)); // external attrs
        b.extend_from_slice(&le32(if zip64 { Z64 } else { local_off }));
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&extra);
        let cd_size = b.len() as u32 - cd_off;

        if zip64 {
            let z64 = b.len() as u64;
            b.extend_from_slice(b"PK\x06\x06");
            b.extend_from_slice(&le64(44)); // size of the record that follows
            b.extend_from_slice(&le16(45));
            b.extend_from_slice(&le16(45));
            b.extend_from_slice(&le32(0)); // this disk
            b.extend_from_slice(&le32(0)); // cd start disk
            b.extend_from_slice(&le64(1)); // entries on this disk
            b.extend_from_slice(&le64(1)); // entries total
            b.extend_from_slice(&le64(cd_size as u64));
            b.extend_from_slice(&le64(cd_off as u64));
            b.extend_from_slice(b"PK\x06\x07");
            b.extend_from_slice(&le32(0));
            b.extend_from_slice(&le64(z64));
            b.extend_from_slice(&le32(1));
        }

        b.extend_from_slice(b"PK\x05\x06");
        b.extend_from_slice(&le16(0));
        b.extend_from_slice(&le16(0));
        b.extend_from_slice(&le16(1));
        b.extend_from_slice(&le16(1));
        b.extend_from_slice(&le32(cd_size));
        b.extend_from_slice(&le32(if zip64 { Z64 } else { cd_off }));
        b.extend_from_slice(&le16(0));
        b
    }

    #[test]
    fn plain_and_zip64_archives_resolve_to_the_same_data_range() {
        let data = b"tensor bytes".as_slice();
        for zip64 in [false, true] {
            let a = archive("archive/data/0", data, zip64);
            let e = parse(&a).unwrap_or_else(|err| panic!("zip64={zip64}: {err}"));
            assert_eq!(e.len(), 1, "zip64={zip64}");
            assert_eq!(e[0].name, "archive/data/0");
            assert_eq!(&a[e[0].offset..e[0].offset + e[0].len], data, "zip64={zip64}");
        }
    }

    /// A container that says "zip64" and then does not carry the records is
    /// corrupt; reading the sentinel as a real offset would seek to 4 GiB - 1
    /// and fail somewhere far less obvious.
    #[test]
    fn zip64_sentinels_without_the_records_error_clearly() {
        let mut a = archive("archive/data/0", b"xy", true);
        let n = a.len();
        // Break the locator signature, leaving the EOCD sentinels in place.
        let loc = n - 22 - 20;
        a[loc + 3] = b'X';
        let err = parse(&a).unwrap_err();
        assert!(err.contains("zip64 locator"), "{err}");

        // ...and an entry whose offset is a sentinel with no 0x0001 field.
        let mut a = archive("archive/data/0", b"xy", true);
        let at = a.windows(4).position(|w| w == b"PK\x01\x02").expect("central directory");
        a[at + 30] = 0; // extra length low byte -> 0
        let err = parse(&a).unwrap_err();
        assert!(err.contains("0x0001"), "{err}");
    }
}
