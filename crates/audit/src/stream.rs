// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The unit of reading, and the order it arrives in.
//!
//! An **episode** is one bounded stretch of one document. Everything else in
//! this crate keys on it, so four properties are decided here and nowhere
//! else:
//!
//! 1. **A document is read start to end, and an episode never spans two of
//!    them.** A document has an order, and a unit that begins mid-sentence in
//!    one file and ends in another is not something a probe can be frozen
//!    against.
//! 2. **Document order is a seeded permutation of the SORTED paths.**
//!    Sorting first is what makes the seed the only source of order:
//!    `read_dir` order is not stable across filesystems and not guaranteed
//!    stable across runs on one, so a stream that took it as given would be
//!    irreproducible for a reason nothing in the run recorded. Permuting
//!    afterwards is what lets the order-permutation control arm be "the same
//!    corpus at a different seed" rather than a second code path.
//! 3. **Binary is decided by content, never by extension.** Extensions lie in
//!    both directions, and a reader pointed at a real directory meets both: a
//!    `.txt` holding a core dump, a `.bin` holding a manual page.
//! 4. **An episode's identity is the digest of its own bytes.** Not of
//!    `(path, ordinal)`: the questions that identity has to answer later are
//!    "have I read this content before" (so a re-stated document costs one
//!    forward pass instead of a training run) and "is this row already in the
//!    rehearsal reservoir". Two identical episodes in different files share
//!    an id deliberately; their `source`/`ordinal` still separate them.
//!
//! ## What this deliberately does not do yet
//!
//! **The corpus is fixed for the life of a stream.** [`Cursor`] is a document
//! index plus an ordinal, so adding or removing a file shifts what an index
//! means; [`EpisodeStream::resume`] therefore refuses a cursor whose
//! fingerprint does not match, by name, rather than silently reading a
//! different stream. A reader that follows a GROWING directory needs an
//! append-only document identity rather than an index, which is a different
//! mechanism and not this one.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use data::rng::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How the stream is cut and ordered.
///
/// Every field is part of [`Cursor`]'s fingerprint, because every one of them
/// changes which episodes exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Permutes the document order. The same seed over the same corpus is the
    /// same sequence; a different seed is the order-permutation control arm.
    pub seed: u64,
    /// Upper bound on an episode in CHARACTERS, not bytes, so the same corpus
    /// cuts the same way regardless of how wide its code points are.
    pub episode_chars: usize,
    /// How much of a file's head is sampled to decide whether it is text.
    pub sniff_bytes: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig { seed: 0, episode_chars: 4096, sniff_bytes: 8192 }
    }
}

/// The digest of an episode's own text. See this module's rule 4.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EpisodeId(String);

impl EpisodeId {
    /// The id of a given text, so a caller holding content but not an
    /// [`Episode`] can ask whether the stream has already produced it.
    pub fn of(text: &str) -> EpisodeId {
        let mut h = Sha256::new();
        h.update(text.as_bytes());
        EpisodeId(format!("{:x}", h.finalize()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One bounded stretch of one document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Episode {
    /// Digest of [`Episode::text`]. Equal ids mean equal content, including
    /// across different files.
    pub id: EpisodeId,
    /// Where it came from, relative to the stream's root, so a ledger row
    /// stays meaningful if the corpus is moved.
    pub source: PathBuf,
    /// Which episode of that document this is, counting from zero.
    pub ordinal: usize,
    pub text: String,
}

/// Why a file in the corpus produced no episodes. Recorded rather than
/// silently dropped: a reader that quietly skipped half a directory and a
/// reader that read all of it look identical from the outside otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Skipped {
    /// Its head is not text. See this module's rule 3.
    NotText(PathBuf),
    /// It holds no characters.
    Empty(PathBuf),
    /// It could not be read at all. The reason is the OS's own.
    Unreadable(PathBuf, String),
}

impl Skipped {
    pub fn path(&self) -> &Path {
        match self {
            Skipped::NotText(p) | Skipped::Empty(p) | Skipped::Unreadable(p, _) => p,
        }
    }
}

/// Where a stream had got to. Valid only for the corpus and config that
/// produced it, which is what `fingerprint` exists to enforce.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cursor {
    /// Digest of the config and the ordered document list.
    pub fingerprint: String,
    /// Index into that document list.
    pub doc: usize,
    /// Which episode of that document comes next.
    pub ordinal: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("stream root {0} is not a directory")]
    NotADirectory(PathBuf),
    #[error("cannot read stream root {0}: {1}")]
    Root(PathBuf, String),
    #[error(
        "this cursor belongs to a different stream (fingerprint {found}, this corpus and config are {expected}) - the corpus or the StreamConfig changed, and a document index does not mean the same thing across that change"
    )]
    CursorMismatch { expected: String, found: String },
}

type Result<T> = std::result::Result<T, StreamError>;

/// A deterministic, resumable stream of episodes over a directory.
#[derive(Debug)]
pub struct EpisodeStream {
    root: PathBuf,
    cfg: StreamConfig,
    docs: Vec<PathBuf>,
    fingerprint: String,
    skipped: Vec<Skipped>,
    doc: usize,
    ordinal: usize,
    /// Which document is currently cut into `pending`. Without it an
    /// exhausted document is read a second time just to discover it is
    /// exhausted, and every document in the corpus costs two reads.
    loaded: Option<usize>,
    pending: VecDeque<Episode>,
}

impl EpisodeStream {
    /// Walk `root`, keep the files that are text, order them by the config's
    /// seed, and position at the first episode.
    ///
    /// Every candidate file is opened here rather than lazily, because the
    /// document list has to be complete before it can be ordered, and the
    /// order has to be fixed before a [`Cursor`] into it means anything.
    pub fn open(root: &Path, cfg: StreamConfig) -> Result<EpisodeStream> {
        if !root.is_dir() {
            return Err(StreamError::NotADirectory(root.to_path_buf()));
        }
        let mut found = Vec::new();
        walk(root, root, &mut found)?;
        // Sorted BEFORE the seed touches it - see this module's rule 2.
        found.sort();

        let mut docs = Vec::new();
        let mut skipped = Vec::new();
        for rel in found {
            match std::fs::read(root.join(&rel)) {
                Err(e) => skipped.push(Skipped::Unreadable(rel, e.to_string())),
                Ok(bytes) => {
                    let head = &bytes[..bytes.len().min(cfg.sniff_bytes)];
                    if !is_text(head) {
                        skipped.push(Skipped::NotText(rel));
                    } else if bytes.is_empty() {
                        skipped.push(Skipped::Empty(rel));
                    } else {
                        docs.push(rel);
                    }
                }
            }
        }

        // Fisher-Yates over the sorted list, from the workspace's own
        // SplitMix64 rather than a fresh local generator.
        let mut rng = Rng::new(cfg.seed);
        for i in (1..docs.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            docs.swap(i, j);
        }

        let fingerprint = fingerprint(&cfg, &docs);
        Ok(EpisodeStream { root: root.to_path_buf(), cfg, docs, fingerprint, skipped, doc: 0, ordinal: 0, loaded: None, pending: VecDeque::new() })
    }

    /// Reopen the same corpus and continue at `cursor`.
    pub fn resume(root: &Path, cfg: StreamConfig, cursor: &Cursor) -> Result<EpisodeStream> {
        let mut s = EpisodeStream::open(root, cfg)?;
        if s.fingerprint != cursor.fingerprint {
            return Err(StreamError::CursorMismatch { expected: s.fingerprint, found: cursor.fingerprint.clone() });
        }
        s.doc = cursor.doc;
        s.ordinal = cursor.ordinal;
        Ok(s)
    }

    /// Where the NEXT episode will come from.
    ///
    /// Taken from the position the stream has ADVANCED to, never from the
    /// last episode handed out, so a cursor saved after `k` episodes and
    /// resumed produces episode `k` exactly once.
    pub fn cursor(&self) -> Cursor {
        Cursor { fingerprint: self.fingerprint.clone(), doc: self.doc, ordinal: self.ordinal }
    }

    /// The ordered document list, relative to the root.
    pub fn documents(&self) -> &[PathBuf] {
        &self.docs
    }

    /// Files the corpus contained that produced no episodes, and why.
    pub fn skipped(&self) -> &[Skipped] {
        &self.skipped
    }
}

impl Iterator for EpisodeStream {
    type Item = Episode;

    fn next(&mut self) -> Option<Episode> {
        loop {
            if let Some(e) = self.pending.pop_front() {
                self.ordinal += 1;
                return Some(e);
            }
            if self.loaded == Some(self.doc) {
                // Cut, handed out, and empty: this document is done.
                self.doc += 1;
                self.ordinal = 0;
                self.loaded = None;
                continue;
            }
            let rel = self.docs.get(self.doc)?.clone();
            // A document is cut whole, then the episodes before `ordinal`
            // are dropped: a resumed stream re-derives the same cut from the
            // same bytes, so the ordinal it was given still names the same
            // episode.
            let text = match std::fs::read_to_string(self.root.join(&rel)) {
                Ok(t) => t,
                // `open` already read this file, so reaching here means the
                // corpus changed underneath a running reader. Recording it
                // is the point: a reader that quietly produced nothing for
                // half a directory looks identical to one that read it all.
                Err(e) => {
                    self.skipped.push(Skipped::Unreadable(rel, e.to_string()));
                    self.doc += 1;
                    self.ordinal = 0;
                    self.loaded = None;
                    continue;
                }
            };
            let mut rest = text.as_str();
            let mut ordinal = 0usize;
            while !rest.is_empty() {
                let n = cut(rest, self.cfg.episode_chars);
                let (piece, tail) = rest.split_at(n);
                if ordinal >= self.ordinal {
                    self.pending.push_back(Episode { id: EpisodeId::of(piece), source: rel.clone(), ordinal, text: piece.to_string() });
                }
                ordinal += 1;
                rest = tail;
            }
            self.loaded = Some(self.doc);
        }
    }
}

/// Byte offset at which to cut `text` so the piece is at most `max_chars`
/// characters and, where possible, ends on a line boundary.
///
/// Lines are kept whole because a probe frozen against an episode is frozen
/// against lines of it; a cut through the middle of one makes the two halves
/// individually unprobeable. A single line longer than the window is cut at a
/// character boundary instead, which always makes progress.
fn cut(text: &str, max_chars: usize) -> usize {
    let end = text.char_indices().nth(max_chars).map(|(i, _)| i).unwrap_or(text.len());
    if end == text.len() {
        return end;
    }
    // `nl + 1` keeps the newline with the line it ends, and is always at
    // least 1, so the cut cannot stall on a leading newline.
    match text[..end].rfind('\n') {
        Some(nl) => nl + 1,
        None => end,
    }
}

/// Whether the head of a file looks like text. See this module's rule 3.
fn is_text(head: &[u8]) -> bool {
    // A NUL is the one byte no text encoding this reader accepts ever
    // contains, and it is what separates a manual page from a core dump
    // faster than any statistical test. Past that the head has to decode:
    // a file brain cannot read as UTF-8 is not a document it can learn
    // from, whatever its extension claims. `from_utf8` may legitimately
    // fail on a multi-byte character straddling the sniff boundary, so the
    // trailing partial character is not held against the file.
    if head.contains(&0) {
        return false;
    }
    match std::str::from_utf8(head) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none() && e.valid_up_to() > 0,
    }
}

/// Every file under `dir`, as paths relative to `root`.
fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| StreamError::Root(dir.to_path_buf(), e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| StreamError::Root(dir.to_path_buf(), e.to_string()))?;
        let path = entry.path();
        if path.is_dir() {
            walk(root, &path, out)?;
        } else {
            out.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
        }
    }
    Ok(())
}

/// Digest of everything that decides which episodes exist and in what order.
/// A [`Cursor`] carries it so that resuming across a change to either is
/// refused rather than silently reinterpreted.
fn fingerprint(cfg: &StreamConfig, docs: &[PathBuf]) -> String {
    let mut h = Sha256::new();
    h.update(cfg.seed.to_le_bytes());
    h.update(cfg.episode_chars.to_le_bytes());
    h.update(cfg.sniff_bytes.to_le_bytes());
    for d in docs {
        h.update(d.to_string_lossy().as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static N: AtomicUsize = AtomicUsize::new(0);

    /// A private corpus per test. Samples and fixtures are not committed, so
    /// every test here writes the files it reads.
    struct Corpus(PathBuf);

    impl Corpus {
        fn new(name: &str) -> Corpus {
            let d = std::env::temp_dir().join(format!(
                "brain-audit-{name}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).expect("temp corpus");
            Corpus(d)
        }
        fn file(&self, name: &str, bytes: &[u8]) -> &Corpus {
            let p = self.0.join(name);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).expect("corpus subdir");
            }
            std::fs::write(p, bytes).expect("corpus file");
            self
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Corpus {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// `n` lines of distinct, recognisable text.
    fn lines(tag: &str, n: usize) -> Vec<u8> {
        (0..n).map(|i| format!("{tag} line {i}\n")).collect::<String>().into_bytes()
    }

    fn cfg(seed: u64, episode_chars: usize) -> StreamConfig {
        StreamConfig { seed, episode_chars, ..StreamConfig::default() }
    }

    fn shape(s: EpisodeStream) -> Vec<(PathBuf, usize, EpisodeId)> {
        s.map(|e| (e.source, e.ordinal, e.id)).collect()
    }

    /// The whole point of the seed. Same seed, same sequence - and a
    /// different seed must actually reorder, or the seed is decoration and
    /// the order-permutation control arm measures nothing.
    #[test]
    fn the_same_seed_reads_the_same_episode_sequence() {
        let c = Corpus::new("seed");
        for i in 0..8 {
            c.file(&format!("doc{i}.txt"), &lines(&format!("doc{i}"), 40));
        }

        let a = shape(EpisodeStream::open(c.path(), cfg(7, 200)).expect("open"));
        let b = shape(EpisodeStream::open(c.path(), cfg(7, 200)).expect("open"));
        assert_eq!(a, b, "the same seed over the same corpus must be the same sequence");
        assert!(a.len() > 8, "the corpus should cut into more episodes than documents, got {}", a.len());

        let other = shape(EpisodeStream::open(c.path(), cfg(8, 200)).expect("open"));
        assert_eq!(
            a.len(),
            other.len(),
            "a different seed reorders documents; it must not change how many episodes exist"
        );
        assert_ne!(a, other, "a different seed must actually produce a different order");
    }

    /// Both halves, because either one alone is consistent with an extension
    /// test: a text file named `.bin` is READ, and a binary file named `.txt`
    /// is SKIPPED.
    #[test]
    fn a_binary_file_is_skipped_by_content_not_by_extension() {
        let c = Corpus::new("sniff");
        c.file("manual.bin", &lines("manual", 20));
        c.file("core.txt", &[0u8, 1, 2, 0, 255, 254, 0, 7, 0, 0, 3]);

        let s = EpisodeStream::open(c.path(), cfg(0, 4096)).expect("open");
        let docs: Vec<PathBuf> = s.documents().to_vec();
        let skipped: Vec<Skipped> = s.skipped().to_vec();

        assert_eq!(docs, vec![PathBuf::from("manual.bin")], "a text file must be read whatever it is called");
        assert_eq!(
            skipped,
            vec![Skipped::NotText(PathBuf::from("core.txt"))],
            "a binary file must be skipped whatever it is called, and the skip must be recorded"
        );
    }

    /// Rule 1. Every episode's text must be a contiguous piece of exactly one
    /// document, and the episodes of one document must reassemble it exactly.
    #[test]
    fn an_episode_never_spans_two_documents() {
        let c = Corpus::new("span");
        let a = lines("alpha", 30);
        let b = lines("beta", 30);
        c.file("a.txt", &a).file("b.txt", &b);

        let mut got: std::collections::BTreeMap<PathBuf, String> = Default::default();
        for e in EpisodeStream::open(c.path(), cfg(3, 64)).expect("open") {
            let whole = if e.source == Path::new("a.txt") { &a } else { &b };
            let whole = std::str::from_utf8(whole).unwrap();
            assert!(
                whole.contains(&e.text),
                "episode {} of {:?} is not a piece of its own document",
                e.ordinal,
                e.source
            );
            got.entry(e.source).or_default().push_str(&e.text);
        }

        assert_eq!(got.len(), 2);
        assert_eq!(got[Path::new("a.txt")], std::str::from_utf8(&a).unwrap());
        assert_eq!(got[Path::new("b.txt")], std::str::from_utf8(&b).unwrap());
    }

    /// A reader is left running for days; the run it resumes has to be the
    /// run it stopped, not a fresh one that happens to share a directory.
    #[test]
    fn a_resumed_stream_continues_at_the_next_unread_episode() {
        let c = Corpus::new("resume");
        for i in 0..5 {
            c.file(&format!("d{i}.txt"), &lines(&format!("d{i}"), 25));
        }

        let whole = shape(EpisodeStream::open(c.path(), cfg(11, 100)).expect("open"));
        assert!(whole.len() > 6, "need enough episodes to stop in the middle of one");

        let mut s = EpisodeStream::open(c.path(), cfg(11, 100)).expect("open");
        let head: Vec<_> = (&mut s).take(5).map(|e| (e.source, e.ordinal, e.id)).collect();
        let saved = s.cursor();
        drop(s);

        let resumed = EpisodeStream::resume(c.path(), cfg(11, 100), &saved).expect("resume");
        let mut rejoined = head;
        rejoined.extend(shape(resumed));
        assert_eq!(rejoined, whole, "a stop and a resume must produce exactly the uninterrupted sequence");
    }

    /// The other half of the pair. A cursor is a document INDEX, so it means
    /// something different the moment the corpus or the cut changes, and
    /// silently continuing would read a stream nobody asked for.
    #[test]
    fn a_resume_under_a_changed_config_is_refused_rather_than_silently_reading_a_different_stream() {
        let c = Corpus::new("mismatch");
        for i in 0..4 {
            c.file(&format!("d{i}.txt"), &lines(&format!("d{i}"), 25));
        }
        let mut s = EpisodeStream::open(c.path(), cfg(2, 100)).expect("open");
        let _ = s.next();
        let saved = s.cursor();
        drop(s);

        // Same corpus, a different cut: different episodes exist.
        match EpisodeStream::resume(c.path(), cfg(2, 250), &saved) {
            Err(StreamError::CursorMismatch { .. }) => {}
            other => panic!("expected CursorMismatch on a changed episode_chars, got {other:?}"),
        }

        // Same config, a corpus that grew: index 3 is no longer document 3.
        c.file("d4.txt", &lines("d4", 25));
        match EpisodeStream::resume(c.path(), cfg(2, 100), &saved) {
            Err(StreamError::CursorMismatch { .. }) => {}
            other => panic!("expected CursorMismatch on a grown corpus, got {other:?}"),
        }
    }

    /// Identity is content, not location: the dedup question asked later is
    /// "have I read this before", and two copies of one manual page are the
    /// same reading whatever they are filed under.
    #[test]
    fn identical_content_in_two_files_shares_an_episode_id() {
        let c = Corpus::new("dedup");
        c.file("here/page.txt", &lines("page", 10));
        c.file("there/page.txt", &lines("page", 10));
        c.file("other.txt", &lines("other", 10));

        let all: Vec<Episode> = EpisodeStream::open(c.path(), cfg(0, 4096)).expect("open").collect();
        let ids: std::collections::BTreeSet<&str> = all.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(all.len(), 3, "three documents, one episode each at this cut");
        assert_eq!(ids.len(), 2, "the two identical pages must share one id, the third must differ");
    }

    /// An empty file is a real thing to find in a directory and must not
    /// become a zero-length episode that every later stage has to special-case.
    #[test]
    fn an_empty_file_yields_no_episodes_and_is_recorded_as_skipped() {
        let c = Corpus::new("empty");
        c.file("nothing.txt", b"").file("something.txt", &lines("s", 5));

        let s = EpisodeStream::open(c.path(), cfg(0, 4096)).expect("open");
        assert_eq!(s.skipped(), &[Skipped::Empty(PathBuf::from("nothing.txt"))]);
        assert_eq!(s.count(), 1);
    }

    /// A line longer than the window must still make progress rather than
    /// loop forever looking for a boundary that is not there.
    #[test]
    fn a_line_longer_than_the_window_is_cut_at_a_character_boundary() {
        let long = "\u{e9}".repeat(500);
        let n = cut(&long, 100);
        assert!(long.is_char_boundary(n), "the cut must land on a character boundary");
        assert_eq!(long[..n].chars().count(), 100, "and must take the whole window when no line break exists");
    }
}
