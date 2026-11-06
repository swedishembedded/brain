// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What [`super::external::ExternalPolicyGradient`] measures about an example the first time a step draws it, kept for
//! every later draw.
//!
//! The measurements are one `f32` per row position (and a flag per position for the teacher's clip), so a dataset of
//! long agent calls would hold gigabytes of them. They live in a file next to the prepared split instead, appended once
//! per example and read back whole when a step lays the example out; only a fixed-size slot per example stays in
//! memory, whatever the example's length.

use std::fs::File;
use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The teacher advantage of every row position of one example (NaN where the position is not supervised), and which
/// positions the bound clipped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenAdvantages {
    pub value: Vec<f32>,
    pub clipped: Vec<bool>,
}

/// What was measured for one example. Each part exists only when the example needs it: `behavior` when a supervised
/// token of an example with an advantage has no recorded log-probability, `reference` when the KL is wanted and the
/// example has an advantage, `teacher` when the record carries a teacher context.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExampleScores {
    /// Per row position: the log-probability of the policy the run started from (the behaviour proxy).
    pub behavior: Option<Vec<f32>>,
    /// Per row position: the log-probability of the reference policy.
    pub reference: Option<Vec<f32>>,
    pub teacher: Option<TokenAdvantages>,
}

const NONE: u32 = u32::MAX;

impl ExampleScores {
    fn encode(&self) -> Vec<u8> {
        let lens = [self.behavior.as_ref(), self.reference.as_ref(), self.teacher.as_ref().map(|t| &t.value)];
        let mut out = Vec::new();
        for len in lens.iter().map(|v| v.map_or(NONE, |v| v.len() as u32)) {
            out.extend_from_slice(&len.to_le_bytes());
        }
        for v in lens.iter().flatten() {
            out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        if let Some(t) = &self.teacher {
            out.extend(t.clipped.iter().map(|&c| u8::from(c)));
        }
        out
    }

    fn decode(bytes: &[u8]) -> Result<ExampleScores> {
        let corrupt = || Error::new(ErrorKind::InvalidData, "a score cache entry is shorter than its header says");
        let len_at = |i: usize| bytes.get(4 * i..4 * i + 4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).ok_or_else(corrupt);
        let lens = [len_at(0)?, len_at(1)?, len_at(2)?];
        let mut at = 12;
        let mut floats = |len: u32| -> Result<Option<Vec<f32>>> {
            if len == NONE {
                return Ok(None);
            }
            let end = at + 4 * len as usize;
            let chunk = bytes.get(at..end).ok_or_else(corrupt)?;
            at = end;
            Ok(Some(chunk.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()))
        };
        let (behavior, reference, value) = (floats(lens[0])?, floats(lens[1])?, floats(lens[2])?);
        let teacher = match value {
            Some(value) => {
                let clipped = bytes.get(at..at + value.len()).ok_or_else(corrupt)?.iter().map(|&b| b != 0).collect();
                Some(TokenAdvantages { value, clipped })
            }
            None => None,
        };
        Ok(ExampleScores { behavior, reference, teacher })
    }
}

/// The scores of the examples a run has drawn, on disk, indexed by example.
pub struct ScoreCache {
    file: File,
    path: PathBuf,
    /// Per example: where its entry starts in the file and how long it is; `None` until the example was scored.
    slots: Vec<Option<(u64, u64)>>,
    end: u64,
}

static NEXT: AtomicU64 = AtomicU64::new(0);

impl ScoreCache {
    /// A cache for `examples` examples in a new file in `dir`, named for the process so that runs sharing a prepared
    /// split never share a cache: the scores are those of one run's start-of-run policy.
    ///
    /// # Errors
    /// The file cannot be created in `dir`.
    pub fn create(dir: &Path, examples: usize) -> Result<ScoreCache> {
        let path = dir.join(format!("train.pg-scores.{}.{}.bin", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).map_err(|e| Error::new(e.kind(), format!("{}: the score cache cannot be created ({e}); the prepared directory must be writable", path.display())))?;
        Ok(ScoreCache { file, path, slots: vec![None; examples], end: 0 })
    }

    pub fn contains(&self, example: usize) -> bool {
        self.slots[example].is_some()
    }

    /// Keep the scores of `example`; it must not have been scored before.
    pub fn put(&mut self, example: usize, scores: &ExampleScores) -> Result<()> {
        assert!(self.slots[example].is_none(), "example {example} is scored once");
        let bytes = scores.encode();
        self.file.seek(SeekFrom::Start(self.end))?;
        self.file.write_all(&bytes)?;
        self.slots[example] = Some((self.end, bytes.len() as u64));
        self.end += bytes.len() as u64;
        Ok(())
    }

    /// The scores kept for `example`.
    ///
    /// # Errors
    /// The example was never scored, or the file cannot be read.
    pub fn get(&self, example: usize) -> Result<ExampleScores> {
        let (offset, len) = self.slots[example].ok_or_else(|| Error::new(ErrorKind::NotFound, format!("example {example} was never scored")))?;
        let mut bytes = vec![0u8; len as usize];
        let mut file = &self.file;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut bytes)?;
        ExampleScores::decode(&bytes)
    }

    /// Bytes held in memory: the slot per example, independent of how long the examples are.
    pub fn resident_bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<Option<(u64, u64)>>()
    }

    /// Bytes written to disk so far.
    pub fn disk_bytes(&self) -> u64 {
        self.end
    }

    /// How many examples are scored.
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for ScoreCache {
    fn drop(&mut self) {
        // The scores belong to this run alone; a leftover file only costs disk, so a failed removal is not an error.
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_round_trip_through_the_file_whatever_parts_they_hold() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = ScoreCache::create(dir.path(), 3).unwrap();
        let full = ExampleScores { behavior: Some(vec![-1.0, -2.0]), reference: Some(vec![-0.5]), teacher: Some(TokenAdvantages { value: vec![0.75, 0.25], clipped: vec![false, true] }) };
        cache.put(2, &full).unwrap();
        cache.put(0, &ExampleScores::default()).unwrap();
        assert_eq!(cache.get(2).unwrap(), full);
        assert_eq!(cache.get(0).unwrap(), ExampleScores::default());
        assert!(cache.get(1).is_err() && !cache.contains(1) && cache.len() == 2);
        let path = cache.path.clone();
        drop(cache);
        assert!(!path.exists(), "the cache file goes with the run");
    }
}
