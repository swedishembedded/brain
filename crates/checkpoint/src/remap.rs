// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A [`TensorSource`] that renames and reslices an inner source's tensors on
//! the fly, so a name-mapping/qkv-splitting import (`s3dit::import::import_comfy`,
//! `qwen3::import::brain_init_from_hf`) can be expressed as a *plan* over a
//! streaming source instead of an eager pass that materializes every renamed
//! tensor into a second, owned `HashMap<String, Vec<f32>>`.
//!
//! [`Fetch`] says how one destination (brain) tensor name resolves against the
//! source: the whole of one source tensor, a contiguous element range of one
//! (a fused `qkv.weight` split into `to_q`/`to_k`/`to_v`), or the ordered
//! concatenation of several. [`RemapSource::validate`] does the same
//! "every destination produced, no source tensor unused" check
//! `brain_init_from_hf` does today, but from shapes alone — no tensor data is
//! read.

use std::collections::HashMap;

use crate::TensorSource;

/// How one destination tensor's bytes resolve against the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fetch {
    /// The whole of source tensor `.0`, unchanged.
    Whole(String),
    /// Elements `[start, start+len)` of source tensor `name` — the shape a
    /// fused `qkv.weight` split into `to_q`/`to_k`/`to_v` needs.
    Slice { name: String, start: usize, len: usize },
    /// The ordered concatenation of several fetches. Unlike `Whole`/`Slice`
    /// (which can stream and zero-copy straight from the source), producing
    /// one contiguous destination tensor from several source pieces is a
    /// real materialization — bounded by the concatenated tensor's own
    /// size, never by the whole model's.
    Concat(Vec<Fetch>),
}

/// A [`TensorSource`] that wraps an inner one with a rename/reslice [`Fetch`]
/// plan. `Whole` and `Slice` fetches are zero-copy through
/// [`TensorSource::raw_words`] when the inner source is, and stream through
/// [`TensorSource::with_tensor_chunks`] otherwise — a `RemapSource` adds no
/// materialization of its own for either. `Concat` is the one case that must
/// build a real (bounded, per-destination-tensor) output buffer.
pub struct RemapSource<'a> {
    inner: &'a dyn TensorSource,
    plan: HashMap<String, Fetch>,
}

impl<'a> RemapSource<'a> {
    pub fn new(inner: &'a dyn TensorSource, plan: HashMap<String, Fetch>) -> RemapSource<'a> {
        RemapSource { inner, plan }
    }

    /// Names-and-shapes-only coverage check, mirroring what
    /// `qwen3::import::brain_init_from_hf` validates today (every destination
    /// name has a source, every size matches) but reading no tensor data —
    /// `numel` alone answers it. Fails loudly, naming every mismatch found
    /// (not just the first), so a broken plan is one error message, not one
    /// panic per parameter as the build proceeds.
    pub fn validate(&self, want: &[(String, usize)]) -> Result<(), String> {
        let mut problems = Vec::new();
        for (name, expect_numel) in want {
            match self.plan.get(name) {
                None => problems.push(format!("'{name}': no fetch plan")),
                Some(fetch) => match self.fetch_numel(fetch) {
                    Some(n) if n != *expect_numel => {
                        problems.push(format!("'{name}': plan yields {n} elements, want {expect_numel}"))
                    }
                    Some(_) => {}
                    None => problems.push(format!("'{name}': source tensor missing or slice out of range")),
                },
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(format!("RemapSource::validate: {} problem(s): {}", problems.len(), problems.join("; ")))
        }
    }

    /// [`Fetch`]'s element count, resolving `Whole` (and `Concat` of `Whole`s)
    /// against the inner source's `numel` — still no tensor data read.
    ///
    /// `None` for a missing source tensor AND for a `Slice` whose range does
    /// not fit inside its source. That single rule is what keeps every access
    /// path (`with_tensor`, `with_tensor_chunks`, `raw_words`, `numel`,
    /// `validate`) refusing an out-of-range slice the same way: a clean "not
    /// available", never a panic and never silently truncated partial data —
    /// the worst possible failure for an import (a config-vs-checkpoint dim
    /// mismatch would otherwise yield silently-wrong tail weights).
    fn fetch_numel(&self, fetch: &Fetch) -> Option<usize> {
        match fetch {
            Fetch::Whole(src) => self.inner.numel(src),
            Fetch::Slice { name, start, len } => {
                let n = self.inner.numel(name)?;
                let end = start.checked_add(*len)?;
                (end <= n).then_some(*len)
            }
            Fetch::Concat(parts) => parts.iter().map(|p| self.fetch_numel(p)).sum(),
        }
    }
}

impl TensorSource for RemapSource<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let Some(fetch) = self.plan.get(name) else { return false };
        match fetch {
            Fetch::Whole(src) => self.inner.with_tensor(src, f),
            Fetch::Slice { name: src, start, len } => {
                // An out-of-range slice is a refusal (`false`), exactly like
                // `raw_words`/`with_tensor_chunks`/`numel` — see fetch_numel.
                if self.fetch_numel(fetch).is_none() {
                    return false;
                }
                let (start, len) = (*start, *len);
                self.inner.with_tensor(src, &mut |data| {
                    // In range by the fetch_numel check above; a violation here
                    // means the SOURCE's numel/with_tensor disagree (a source
                    // impl bug, not user data), which deserves the loud panic.
                    f(&data[start..start + len]);
                })
            }
            Fetch::Concat(parts) => {
                let Some(total) = self.fetch_numel(fetch) else { return false };
                let mut buf = vec![0.0f32; total];
                let mut off = 0usize;
                for p in parts {
                    let Some(n) = self.fetch_numel(p) else { return false };
                    if !self.fetch_into(p, &mut buf[off..off + n]) {
                        return false;
                    }
                    off += n;
                }
                f(&buf);
                true
            }
        }
    }

    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        match self.plan.get(name)? {
            Fetch::Whole(src) => self.inner.raw_words(src),
            Fetch::Slice { name: src, start, len } => {
                let words = self.inner.raw_words(src)?;
                let end = start.checked_add(*len)?;
                words.get(*start..end)
            }
            // A contiguous destination built from several source pieces has
            // no single borrowed slice to hand back.
            Fetch::Concat(_) => None,
        }
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(fetch) = self.plan.get(name) else { return false };
        match fetch {
            Fetch::Whole(src) => self.inner.with_tensor_chunks(src, max_elems, f),
            Fetch::Slice { name: src, start, len } => {
                // Refuse an out-of-range slice up front. Without this check the
                // overlap-clip below would deliver only the overlapping PREFIX
                // and still return `true` — silently-wrong tail weights.
                if self.fetch_numel(fetch).is_none() {
                    return false;
                }
                let (start, len) = (*start, *len);
                self.inner.with_tensor_chunks(src, max_elems, &mut |chunk_off, chunk| {
                    let chunk_off = chunk_off as usize;
                    let chunk_end = chunk_off + chunk.len();
                    // Overlap of [chunk_off, chunk_end) with [start, start+len),
                    // re-based to the destination's own [0, len) coordinates.
                    let lo = chunk_off.max(start);
                    let hi = chunk_end.min(start + len);
                    if lo < hi {
                        f((lo - start) as u64, &chunk[lo - chunk_off..hi - chunk_off]);
                    }
                })
            }
            // Default (materialize once, hand over as one chunk) — bounded by
            // this destination tensor's own size, which is the same cost
            // `with_tensor` above already pays for Concat.
            Fetch::Concat(_) => self.with_tensor(name, &mut |d| f(0, d)),
        }
    }

    fn numel(&self, name: &str) -> Option<usize> {
        let fetch = self.plan.get(name)?;
        self.fetch_numel(fetch)
    }
}

impl RemapSource<'_> {
    /// Fill `out` with `fetch`'s data. `out.len()` must equal `fetch`'s numel
    /// (the caller, `Concat`'s assembly loop, already sized it that way).
    fn fetch_into(&self, fetch: &Fetch, out: &mut [f32]) -> bool {
        match fetch {
            Fetch::Whole(src) => {
                // `out` was sized from fetch_numel == the source's numel; a
                // length mismatch means the source's numel/with_tensor
                // disagree — refuse rather than panic in copy_from_slice.
                let mut ok = false;
                let found = self.inner.with_tensor(src, &mut |d| {
                    if d.len() == out.len() {
                        out.copy_from_slice(d);
                        ok = true;
                    }
                });
                found && ok
            }
            Fetch::Slice { name: src, start, len } => {
                if self.fetch_numel(fetch).is_none() {
                    return false; // out of range: same refusal as every other path
                }
                let (start, len) = (*start, *len);
                self.inner.with_tensor(src, &mut |d| out.copy_from_slice(&d[start..start + len]))
            }
            Fetch::Concat(parts) => {
                let mut off = 0usize;
                for p in parts {
                    let Some(n) = self.fetch_numel(p) else { return false };
                    if !self.fetch_into(p, &mut out[off..off + n]) {
                        return false;
                    }
                    off += n;
                }
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;

    fn src(entries: &[(&str, &[f32])]) -> Map<String, Vec<f32>> {
        entries.iter().map(|(n, v)| (n.to_string(), v.to_vec())).collect()
    }

    #[test]
    fn whole_passes_through_unchanged() {
        let inner = src(&[("a", &[1.0, 2.0, 3.0])]);
        let mut plan = HashMap::new();
        plan.insert("dst".to_string(), Fetch::Whole("a".to_string()));
        let r = RemapSource::new(&inner, plan);
        let mut got = None;
        assert!(r.with_tensor("dst", &mut |d| got = Some(d.to_vec())));
        assert_eq!(got.unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(r.numel("dst"), Some(3));
        // Zero-copy passthrough: a HashMap<String,Vec<f32>> is already a
        // TensorSource with a real raw_words impl (bit-cast, no allocation).
        assert!(r.raw_words("dst").is_some());
    }

    /// The real-world case this exists for: a fused `qkv.weight` split into
    /// three destination tensors, matching `s3dit::import::import_comfy`'s
    /// `to_q`/`to_k`/`to_v` split byte-for-byte but via a plan instead of an
    /// eager rewrite of the whole tensor map.
    #[test]
    fn slice_splits_a_fused_qkv_tensor() {
        let dim = 2usize;
        let dd = dim * dim;
        let qkv: Vec<f32> = (0..3 * dd).map(|i| i as f32).collect();
        let inner = src(&[("attn.qkv.weight", &qkv)]);
        let mut plan = HashMap::new();
        plan.insert("attn.to_q.weight".to_string(), Fetch::Slice { name: "attn.qkv.weight".to_string(), start: 0, len: dd });
        plan.insert("attn.to_k.weight".to_string(), Fetch::Slice { name: "attn.qkv.weight".to_string(), start: dd, len: dd });
        plan.insert("attn.to_v.weight".to_string(), Fetch::Slice { name: "attn.qkv.weight".to_string(), start: 2 * dd, len: dd });
        let r = RemapSource::new(&inner, plan);

        let mut q = None;
        assert!(r.with_tensor("attn.to_q.weight", &mut |d| q = Some(d.to_vec())));
        assert_eq!(q.unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
        let mut v = None;
        assert!(r.with_tensor("attn.to_v.weight", &mut |d| v = Some(d.to_vec())));
        assert_eq!(v.unwrap(), vec![8.0, 9.0, 10.0, 11.0]);

        // Zero-copy: slicing a raw_words borrow is still a borrow, no allocation.
        let raw = r.raw_words("attn.to_k.weight").expect("f32 source, must be zero-copyable");
        let want: Vec<u32> = qkv[dd..2 * dd].iter().map(|v| v.to_bits()).collect();
        assert_eq!(raw, want.as_slice());
    }

    #[test]
    fn slice_streams_in_bounded_chunks_correctly_offset() {
        let n = 1000usize;
        let full: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let inner = src(&[("big", &full)]);
        let mut plan = HashMap::new();
        plan.insert("dst".to_string(), Fetch::Slice { name: "big".to_string(), start: 137, len: 300 });
        let r = RemapSource::new(&inner, plan);

        let mut reassembled = vec![0.0f32; 300];
        r.with_tensor_chunks("dst", 64, &mut |off, chunk| {
            reassembled[off as usize..off as usize + chunk.len()].copy_from_slice(chunk);
        });
        assert_eq!(reassembled, full[137..437]);
    }

    #[test]
    fn concat_builds_one_contiguous_tensor_from_several_pieces() {
        let inner = src(&[("a", &[1.0, 2.0]), ("b", &[3.0, 4.0, 5.0])]);
        let mut plan = HashMap::new();
        plan.insert(
            "dst".to_string(),
            Fetch::Concat(vec![Fetch::Whole("a".to_string()), Fetch::Slice { name: "b".to_string(), start: 1, len: 2 }]),
        );
        let r = RemapSource::new(&inner, plan);
        let mut got = None;
        assert!(r.with_tensor("dst", &mut |d| got = Some(d.to_vec())));
        assert_eq!(got.unwrap(), vec![1.0, 2.0, 4.0, 5.0]);
        assert_eq!(r.numel("dst"), Some(4));
        assert!(r.raw_words("dst").is_none(), "concat has no single borrowed slice to hand back");
    }

    #[test]
    fn validate_catches_a_missing_tensor_and_a_size_mismatch() {
        let inner = src(&[("a", &[1.0, 2.0, 3.0])]);
        let mut plan = HashMap::new();
        plan.insert("ok".to_string(), Fetch::Whole("a".to_string()));
        plan.insert("wrong_size".to_string(), Fetch::Slice { name: "a".to_string(), start: 0, len: 2 });
        // "missing" has no plan entry at all.
        let r = RemapSource::new(&inner, plan);
        let err = r.validate(&[("ok".to_string(), 3), ("wrong_size".to_string(), 5), ("missing".to_string(), 1)]).unwrap_err();
        assert!(err.contains("wrong_size"), "{err}");
        assert!(err.contains("missing"), "{err}");
        assert!(!err.contains("'ok'"), "a correctly-sized entry must not be reported: {err}");
    }

    /// An out-of-range Slice (the config-vs-checkpoint dim-mismatch case) must
    /// be refused IDENTICALLY by every access path: `false`/`None`, never a
    /// panic (the old eager path) and never silently-truncated partial data
    /// (the old chunked path, which delivered only the overlapping prefix and
    /// still returned `true`).
    #[test]
    fn out_of_range_slice_is_refused_consistently_by_every_access_path() {
        let inner = src(&[("a", &[1.0, 2.0, 3.0, 4.0])]);
        let mut plan = HashMap::new();
        plan.insert("oob".to_string(), Fetch::Slice { name: "a".to_string(), start: 2, len: 5 });
        plan.insert("cat".to_string(), Fetch::Concat(vec![Fetch::Slice { name: "a".to_string(), start: 2, len: 5 }]));
        let r = RemapSource::new(&inner, plan);

        assert_eq!(r.numel("oob"), None);
        assert!(r.raw_words("oob").is_none());
        assert!(!r.with_tensor("oob", &mut |_| panic!("no data may be delivered for an OOB slice")));
        assert!(!r.with_tensor_chunks("oob", 2, &mut |_, _| panic!("no partial data may be delivered for an OOB slice")));
        assert!(!r.with_tensor("cat", &mut |_| panic!("no data may be delivered for an OOB concat part")));
        let err = r.validate(&[("oob".to_string(), 5)]).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn validate_passes_a_correct_plan() {
        let inner = src(&[("a", &[1.0, 2.0, 3.0, 4.0])]);
        let mut plan = HashMap::new();
        plan.insert("q".to_string(), Fetch::Slice { name: "a".to_string(), start: 0, len: 2 });
        plan.insert("k".to_string(), Fetch::Slice { name: "a".to_string(), start: 2, len: 2 });
        let r = RemapSource::new(&inner, plan);
        assert!(r.validate(&[("q".to_string(), 2), ("k".to_string(), 2)]).is_ok());
    }
}
