// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The HF/llama.cpp split-file naming convention: `<base>-<NNNNN>-of-<MMMMM>`,
//! shared verbatim by sharded safetensors checkpoints and split GGUF files -
//! only the extension differs. One parser here instead of two independent
//! ones (`crates/cli/src/model_dir.rs`'s `shard_of` predates this and covers
//! only `.safetensors`; [`crate::gguf::MmapGguf::open`] is this module's
//! first GGUF-side caller).
//!
//! Swedish Embedded AB implements checkpoint container tooling for its
//! clients. If your team needs expertise in loading multi-file model
//! releases then you can procure our services by sending an email to
//! info@swedishembedded.com.

/// Parse a `<base>-<NNNNN>-of-<MMMMM>.<ext>` split/shard filename into
/// `(base, part_1_based, count, digit_width)`. `ext` must match exactly (no
/// leading dot). `None` for a plain, unsharded file, for a malformed index
/// (non-digits, empty), or for a part number that is `0` or exceeds `count`
/// - a real writer never emits either, so treating them as "not a split
/// name" here is the same refuse-don't-guess choice `MmapGguf::open` needs.
pub fn split_name<'a>(fname: &'a str, ext: &str) -> Option<(&'a str, u32, u32, usize)> {
    let stem = fname.strip_suffix(&format!(".{ext}"))?;
    let (left, total) = stem.rsplit_once("-of-")?;
    if total.is_empty() || !total.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (base, idx) = left.rsplit_once('-')?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let width = total.len();
    let count: u32 = total.parse().ok()?;
    let part: u32 = idx.parse().ok()?;
    if part == 0 || part > count {
        return None;
    }
    Some((base, part, count, width))
}

/// The filename of split part `part` (1-based) of a `count`-part split named
/// `base`, zero-padded to `width` digits - the exact inverse of
/// [`split_name`].
pub fn split_sibling(base: &str, part: u32, count: u32, width: usize, ext: &str) -> String {
    format!("{base}-{part:0width$}-of-{count:0width$}.{ext}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_real_split_name() {
        assert_eq!(split_name("model-00002-of-00005.gguf", "gguf"), Some(("model", 2, 5, 5)));
    }

    #[test]
    fn round_trips_through_split_sibling() {
        let (base, part, count, width) = split_name("flux-2-klein-9b-Q4_K_M-00001-of-00003.gguf", "gguf").unwrap();
        assert_eq!(split_sibling(base, part, count, width, "gguf"), "flux-2-klein-9b-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(split_sibling(base, 3, count, width, "gguf"), "flux-2-klein-9b-Q4_K_M-00003-of-00003.gguf");
    }

    #[test]
    fn a_plain_file_is_not_a_split_name() {
        assert_eq!(split_name("model.gguf", "gguf"), None);
        assert_eq!(split_name("model-Q4_K_M.gguf", "gguf"), None);
    }

    #[test]
    fn the_wrong_extension_does_not_match() {
        assert_eq!(split_name("model-00001-of-00003.safetensors", "gguf"), None);
    }

    #[test]
    fn a_zero_or_out_of_range_part_is_refused() {
        assert_eq!(split_name("model-00000-of-00003.gguf", "gguf"), None);
        assert_eq!(split_name("model-00004-of-00003.gguf", "gguf"), None);
    }

    #[test]
    fn non_digit_index_or_total_is_refused() {
        assert_eq!(split_name("model-x-of-00003.gguf", "gguf"), None);
        assert_eq!(split_name("model-00001-of-x.gguf", "gguf"), None);
    }
}
