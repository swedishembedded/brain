// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Episode dataset: the on-disk format for recorded / generated world-model
//! interaction data (frames + actions + optional rewards, grouped into
//! episodes), plus the boundary-safe window sampler that training consumes.
//!
//! ## On-disk layout (one dataset per directory)
//!   * `frames.u8`   — raw CHW `u8` frames, `N * (C*H*W)` bytes, values `[0,255]`.
//!   * `actions.bin` — `N` little-endian `u32`: `actions[i]` is the action
//!     CONSUMED by the step that produced frame `i`.
//!   * `rewards.f32` — `N` little-endian `f32`. OPTIONAL: absent when the
//!     source had no reward signal (e.g. play-session recordings).
//!   * `meta.json`   — `{"version":1,"n":N,"c":C,"h":H,"w":W,"num_actions":A,
//!     "fps":F,"episodes":[{"start":s,"len":l},...]}`.
//!
//! ## Atomicity
//! [`EpisodeWriter`] streams everything into a sibling `<dir>.tmp` directory
//! and only [`EpisodeWriter::finalize`] renames it to `<dir>` (the same
//! approach as `checkpoint::save`): readers see either no dataset or a
//! complete one. Dropping a writer without finalizing removes the temp
//! directory and leaves NO dataset directory behind.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::rng::Rng;

/// One episode's extent inside the flat frame arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Episode {
    /// Index of the episode's first frame.
    pub start: usize,
    /// Number of frames in the episode.
    pub len: usize,
}

/// Streaming writer for an episode dataset. See the module docs for the
/// layout and the atomicity contract.
pub struct EpisodeWriter {
    final_dir: PathBuf,
    tmp_dir: PathBuf,
    frames: Option<BufWriter<fs::File>>,
    actions: Option<BufWriter<fs::File>>,
    rewards: Option<BufWriter<fs::File>>,
    /// Whether pushes carry rewards; fixed by the FIRST push.
    has_rewards: Option<bool>,
    c: u32,
    h: u32,
    w: u32,
    num_actions: u32,
    fps: u32,
    n: usize,
    episodes: Vec<Episode>,
    /// First frame index of the episode currently being written.
    ep_start: usize,
    finalized: bool,
}

impl EpisodeWriter {
    /// Start a new dataset at `dir` (which must not exist yet). Data streams
    /// into `<dir>.tmp`; nothing appears at `dir` until [`Self::finalize`].
    pub fn create(
        dir: &Path,
        c: u32,
        h: u32,
        w: u32,
        num_actions: u32,
        fps: u32,
    ) -> Result<EpisodeWriter, String> {
        if c == 0 || h == 0 || w == 0 {
            return Err(format!("episode: bad frame shape {c}x{h}x{w}"));
        }
        if dir.exists() {
            return Err(format!("episode: {} already exists", dir.display()));
        }
        let mut tmp = dir.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp_dir = PathBuf::from(tmp);
        if tmp_dir.exists() {
            // A stale temp from a crashed earlier run: safe to replace.
            fs::remove_dir_all(&tmp_dir)
                .map_err(|e| format!("episode: cannot clear stale {}: {e}", tmp_dir.display()))?;
        }
        fs::create_dir_all(&tmp_dir)
            .map_err(|e| format!("episode: cannot create {}: {e}", tmp_dir.display()))?;
        let open = |name: &str| -> Result<BufWriter<fs::File>, String> {
            fs::File::create(tmp_dir.join(name))
                .map(BufWriter::new)
                .map_err(|e| format!("episode: cannot create {name}: {e}"))
        };
        Ok(EpisodeWriter {
            frames: Some(open("frames.u8")?),
            actions: Some(open("actions.bin")?),
            rewards: None,
            has_rewards: None,
            final_dir: dir.to_path_buf(),
            tmp_dir,
            c,
            h,
            w,
            num_actions,
            fps,
            n: 0,
            episodes: Vec::new(),
            ep_start: 0,
            finalized: false,
        })
    }

    /// Bytes in one frame (`C*H*W`).
    pub fn frame_len(&self) -> usize {
        (self.c * self.h * self.w) as usize
    }

    /// Frames pushed so far.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Append one step: the CHW `u8` frame it produced, the action it
    /// consumed, and the reward it yielded (all pushes must agree on whether
    /// a reward is present — the optional `rewards.f32` file is all-or-none).
    pub fn push(&mut self, frame_u8: &[u8], action: u32, reward: Option<f32>) -> Result<(), String> {
        if frame_u8.len() != self.frame_len() {
            return Err(format!(
                "episode: frame has {} bytes, expected c*h*w = {}",
                frame_u8.len(),
                self.frame_len()
            ));
        }
        match (self.has_rewards, reward.is_some()) {
            (None, has) => {
                self.has_rewards = Some(has);
                if has {
                    self.rewards = Some(
                        fs::File::create(self.tmp_dir.join("rewards.f32"))
                            .map(BufWriter::new)
                            .map_err(|e| format!("episode: cannot create rewards.f32: {e}"))?,
                    );
                }
            }
            (Some(expect), has) if expect != has => {
                return Err(format!(
                    "episode: push {} a reward after earlier pushes {} one",
                    if has { "carried" } else { "omitted" },
                    if expect { "carried" } else { "omitted" }
                ));
            }
            _ => {}
        }
        let werr = |e: std::io::Error| format!("episode: write failed: {e}");
        self.frames.as_mut().unwrap().write_all(frame_u8).map_err(werr)?;
        self.actions.as_mut().unwrap().write_all(&action.to_le_bytes()).map_err(werr)?;
        if let Some(r) = reward {
            self.rewards.as_mut().unwrap().write_all(&r.to_le_bytes()).map_err(werr)?;
        }
        self.n += 1;
        Ok(())
    }

    /// Close the current episode (no-op when it is empty). The next `push`
    /// starts a new episode. `finalize` closes the last episode implicitly.
    pub fn end_episode(&mut self) {
        if self.n > self.ep_start {
            self.episodes.push(Episode { start: self.ep_start, len: self.n - self.ep_start });
            self.ep_start = self.n;
        }
    }

    fn meta_json(&self) -> String {
        let eps: Vec<serde_json::Value> = self
            .episodes
            .iter()
            .map(|e| serde_json::json!({ "start": e.start, "len": e.len }))
            .collect();
        serde_json::json!({
            "version": 1,
            "n": self.n,
            "c": self.c,
            "h": self.h,
            "w": self.w,
            "num_actions": self.num_actions,
            "fps": self.fps,
            "episodes": eps,
        })
        .to_string()
    }

    /// Flush everything, write `meta.json`, and atomically rename
    /// `<dir>.tmp` -> `<dir>` (readers see the old state or the complete
    /// new dataset, never a partial one — same contract as `checkpoint::save`).
    pub fn finalize(mut self) -> Result<(), String> {
        self.end_episode();
        for w in [self.frames.take(), self.actions.take(), self.rewards.take()].into_iter().flatten() {
            let mut w = w;
            w.flush().map_err(|e| format!("episode: flush failed: {e}"))?;
            // Dropping the BufWriter closes the file before the rename.
        }
        fs::write(self.tmp_dir.join("meta.json"), self.meta_json())
            .map_err(|e| format!("episode: cannot write meta.json: {e}"))?;
        fs::rename(&self.tmp_dir, &self.final_dir)
            .map_err(|e| format!("episode: cannot finalize {}: {e}", self.final_dir.display()))?;
        self.finalized = true;
        Ok(())
    }
}

impl Drop for EpisodeWriter {
    fn drop(&mut self) {
        if !self.finalized {
            // Abandoned writer: leave no dataset dir and no temp litter.
            self.frames.take();
            self.actions.take();
            self.rewards.take();
            let _ = fs::remove_dir_all(&self.tmp_dir);
        }
    }
}

/// One training window drawn from a dataset: `t_len` consecutive frames that
/// all belong to a single episode.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    /// `t_len * C*H*W` values in `[0,1]` (u8 / 255), frames oldest-first.
    pub frames_f32: Vec<f32>,
    /// `t_len` actions, aligned with the frames.
    pub actions: Vec<u32>,
    /// `t_len` rewards when the dataset has them.
    pub rewards: Option<Vec<f32>>,
    /// Index of the window's first frame in the flat dataset.
    pub start_index: usize,
}

/// A validated, opened episode dataset. Frames are read on demand from
/// `frames.u8` (one shared file handle, positional reads — no mmap); actions
/// and rewards are small and held in memory.
#[derive(Debug)]
pub struct EpisodeDataset {
    file: fs::File,
    pub n: usize,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub num_actions: u32,
    pub fps: u32,
    pub episodes: Vec<Episode>,
    actions: Vec<u32>,
    rewards: Option<Vec<f32>>,
}

impl EpisodeDataset {
    /// Open `dir`, hard-validating `meta.json` against the actual file sizes
    /// and the episode table against `n` (contiguous, covering all frames).
    pub fn open(dir: &Path) -> Result<EpisodeDataset, String> {
        let meta_path = dir.join("meta.json");
        let meta: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&meta_path)
                .map_err(|e| format!("episode: cannot read {}: {e}", meta_path.display()))?,
        )
        .map_err(|e| format!("episode: bad meta.json: {e}"))?;
        let field = |k: &str| -> Result<u64, String> {
            meta[k].as_u64().ok_or_else(|| format!("episode: meta.json missing '{k}'"))
        };
        let version = field("version")?;
        if version != 1 {
            return Err(format!("episode: unsupported version {version} (expected 1)"));
        }
        let n = field("n")? as usize;
        let (c, h, w) = (field("c")? as u32, field("h")? as u32, field("w")? as u32);
        let num_actions = field("num_actions")? as u32;
        let fps = field("fps")? as u32;
        let eps_json = meta["episodes"].as_array().ok_or("episode: meta.json missing 'episodes'")?;
        let mut episodes = Vec::with_capacity(eps_json.len());
        let mut expect_start = 0usize;
        for e in eps_json {
            let start = e["start"].as_u64().ok_or("episode: episode missing 'start'")? as usize;
            let len = e["len"].as_u64().ok_or("episode: episode missing 'len'")? as usize;
            if start != expect_start || len == 0 {
                return Err(format!(
                    "episode: bad episode table: start={start} len={len}, expected start {expect_start}, len > 0"
                ));
            }
            expect_start = start + len;
            episodes.push(Episode { start, len });
        }
        if expect_start != n {
            return Err(format!("episode: episodes cover {expect_start} frames but n={n}"));
        }

        let frame_len = (c as usize) * (h as usize) * (w as usize);
        if frame_len == 0 {
            return Err(format!("episode: bad frame shape {c}x{h}x{w}"));
        }
        let frames_path = dir.join("frames.u8");
        let file = fs::File::open(&frames_path)
            .map_err(|e| format!("episode: cannot open {}: {e}", frames_path.display()))?;
        let fsize = file
            .metadata()
            .map_err(|e| format!("episode: cannot stat frames.u8: {e}"))?
            .len();
        if fsize != (n * frame_len) as u64 {
            return Err(format!(
                "episode: frames.u8 is {fsize} bytes, expected n*c*h*w = {}",
                n * frame_len
            ));
        }
        let actions = crate::binio::read_u32_bin(&dir.join("actions.bin"))
            .map_err(|e| format!("episode: cannot read actions.bin: {e}"))?;
        if actions.len() != n {
            return Err(format!("episode: actions.bin has {} entries, expected n = {n}", actions.len()));
        }
        let rewards_path = dir.join("rewards.f32");
        let rewards = if rewards_path.exists() {
            let r = crate::binio::read_f32_bin(&rewards_path)
                .map_err(|e| format!("episode: cannot read rewards.f32: {e}"))?;
            if r.len() != n {
                return Err(format!("episode: rewards.f32 has {} entries, expected n = {n}", r.len()));
            }
            Some(r)
        } else {
            None
        };
        Ok(EpisodeDataset { file, n, c, h, w, num_actions, fps, episodes, actions, rewards })
    }

    /// Bytes / elements in one frame (`C*H*W`).
    pub fn frame_len(&self) -> usize {
        (self.c * self.h * self.w) as usize
    }

    /// All actions, `actions[i]` consumed by the step that produced frame `i`.
    pub fn actions(&self) -> &[u32] {
        &self.actions
    }

    /// All rewards, when the dataset recorded them.
    pub fn rewards(&self) -> Option<&[f32]> {
        self.rewards.as_deref()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file
                .read_exact_at(buf, offset)
                .map_err(|e| format!("episode: read at {offset} failed: {e}"))
        }
        #[cfg(not(unix))]
        {
            // Fallback: positional reads through a fresh seek. `&File` cannot
            // seek without unix's pread, so clone the handle.
            use std::io::{Read, Seek, SeekFrom};
            let mut f = self.file.try_clone().map_err(|e| format!("episode: {e}"))?;
            f.seek(SeekFrom::Start(offset)).map_err(|e| format!("episode: {e}"))?;
            f.read_exact(buf).map_err(|e| format!("episode: read at {offset} failed: {e}"))
        }
    }

    /// Frame `i` as raw CHW `u8`.
    pub fn frame(&self, i: usize) -> Result<Vec<u8>, String> {
        if i >= self.n {
            return Err(format!("episode: frame {i} out of range (n = {})", self.n));
        }
        let fl = self.frame_len();
        let mut buf = vec![0u8; fl];
        self.read_at((i * fl) as u64, &mut buf)?;
        Ok(buf)
    }

    /// Frame `i` as CHW `f32` in `[0,1]` (u8 / 255).
    pub fn frame_f32(&self, i: usize) -> Result<Vec<f32>, String> {
        Ok(self.frame(i)?.iter().map(|&b| b as f32 / 255.0).collect())
    }

    /// The window starting at flat frame index `start` (must lie entirely
    /// inside one episode — callers get starts from `sample_window` /
    /// `iter_windows`, which guarantee that).
    fn window_at(&self, start: usize, t_len: usize) -> Result<Window, String> {
        let fl = self.frame_len();
        let mut bytes = vec![0u8; t_len * fl];
        self.read_at((start * fl) as u64, &mut bytes)?;
        Ok(Window {
            frames_f32: bytes.iter().map(|&b| b as f32 / 255.0).collect(),
            actions: self.actions[start..start + t_len].to_vec(),
            rewards: self.rewards.as_ref().map(|r| r[start..start + t_len].to_vec()),
            start_index: start,
        })
    }

    /// Draw one training window of `t_len` consecutive frames, uniform over
    /// ALL valid starts across ALL episodes. A window NEVER crosses an
    /// episode boundary (frames from different episodes are causally
    /// unrelated — the model must not learn across a reset).
    /// `Err` when no episode has at least `t_len` frames.
    pub fn sample_window(&self, rng: &mut Rng, t_len: usize) -> Result<Window, String> {
        if t_len == 0 {
            return Err("episode: t_len must be > 0".into());
        }
        let total: usize = self.episodes.iter().map(|e| (e.len + 1).saturating_sub(t_len)).sum();
        if total == 0 {
            return Err(format!("episode: no episode has >= {t_len} frames"));
        }
        let mut k = rng.gen_range_inclusive(0, total as i64 - 1) as usize;
        for e in &self.episodes {
            let starts = (e.len + 1).saturating_sub(t_len);
            if k < starts {
                return self.window_at(e.start + k, t_len);
            }
            k -= starts;
        }
        unreachable!("k < total by construction");
    }

    /// Deterministic sweep for eval: every window of `t_len` frames at
    /// `stride`-spaced starts, per episode (never crossing a boundary),
    /// episodes in order. Panics on an I/O error mid-iteration (the sizes
    /// were validated at `open`, so that indicates the file changed under us).
    pub fn iter_windows(&self, t_len: usize, stride: usize) -> impl Iterator<Item = Window> + '_ {
        assert!(t_len > 0 && stride > 0, "episode: t_len and stride must be > 0");
        let mut starts = Vec::new();
        for e in &self.episodes {
            let mut s = e.start;
            while s + t_len <= e.start + e.len {
                starts.push(s);
                s += stride;
            }
        }
        starts
            .into_iter()
            .map(move |s| self.window_at(s, t_len).unwrap_or_else(|e| panic!("{e}")))
    }

    /// Split episode INDICES into (train, val) with ~`train_frac` of the
    /// episodes in train. The split is BY EPISODE, never by frame: adjacent
    /// frames within an episode are nearly identical, so a frame-level split
    /// would leak train content into val and inflate eval scores.
    pub fn split(&self, train_frac: f64) -> (Vec<usize>, Vec<usize>) {
        let n_ep = self.episodes.len();
        let k = ((n_ep as f64) * train_frac.clamp(0.0, 1.0)).round() as usize;
        let k = k.min(n_ep);
        ((0..k).collect(), (k..n_ep).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("brain_episode_{name}_{}", std::process::id()))
    }

    /// A recognizable frame: every byte = (i*31 + j) & 0xff for frame i.
    fn frame_bytes(i: usize, fl: usize) -> Vec<u8> {
        (0..fl).map(|j| ((i * 31 + j) & 0xff) as u8).collect()
    }

    /// Write a dataset with the given episode lengths; frame i carries
    /// pattern i, action i, reward i as f32 (when `rewards`).
    fn write_ds(dir: &Path, ep_lens: &[usize], rewards: bool, c: u32, h: u32, w: u32) {
        let mut wtr = EpisodeWriter::create(dir, c, h, w, 5, 15).unwrap();
        let fl = wtr.frame_len();
        let mut i = 0usize;
        for &l in ep_lens {
            for _ in 0..l {
                let r = rewards.then_some(i as f32);
                wtr.push(&frame_bytes(i, fl), i as u32, r).unwrap();
                i += 1;
            }
            wtr.end_episode();
        }
        wtr.finalize().unwrap();
    }

    #[test]
    fn episode_roundtrip_exact_bytes() {
        let dir = tmp("roundtrip");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[3, 2], true, 3, 4, 5);
        let ds = EpisodeDataset::open(&dir).unwrap();
        assert_eq!((ds.n, ds.c, ds.h, ds.w), (5, 3, 4, 5));
        assert_eq!((ds.num_actions, ds.fps), (5, 15));
        assert_eq!(ds.episodes, vec![Episode { start: 0, len: 3 }, Episode { start: 3, len: 2 }]);
        assert_eq!(ds.actions(), &[0, 1, 2, 3, 4]);
        assert_eq!(ds.rewards().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        for i in 0..5 {
            assert_eq!(ds.frame(i).unwrap(), frame_bytes(i, ds.frame_len()), "frame {i}");
            let f32s = ds.frame_f32(i).unwrap();
            assert_eq!(f32s.len(), ds.frame_len());
            assert!(f32s.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
        assert!(ds.frame(5).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_roundtrip_without_rewards() {
        let dir = tmp("no_rewards");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[4], false, 1, 2, 2);
        let ds = EpisodeDataset::open(&dir).unwrap();
        assert!(ds.rewards().is_none());
        assert!(!dir.join("rewards.f32").exists());
        let w = ds.sample_window(&mut Rng::new(1), 2).unwrap();
        assert!(w.rewards.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_finalize_is_atomic_and_drop_cleans_up() {
        let dir = tmp("atomic");
        let tmp_dir = PathBuf::from(format!("{}.tmp", dir.display()));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&tmp_dir);

        // Dropped without finalize: neither the dataset nor the temp remain.
        {
            let mut w = EpisodeWriter::create(&dir, 1, 2, 2, 3, 10).unwrap();
            w.push(&[0; 4], 0, None).unwrap();
            assert!(!dir.exists(), "dataset dir must not exist before finalize");
            assert!(tmp_dir.exists(), "writer streams into <dir>.tmp");
        }
        assert!(!dir.exists(), "abandoned writer must leave no dataset dir");
        assert!(!tmp_dir.exists(), "abandoned writer must clean its temp dir");

        // Finalized: dataset appears, temp is gone.
        let mut w = EpisodeWriter::create(&dir, 1, 2, 2, 3, 10).unwrap();
        w.push(&[7; 4], 1, None).unwrap();
        assert!(!dir.exists());
        w.finalize().unwrap();
        assert!(dir.exists());
        assert!(!tmp_dir.exists());
        EpisodeDataset::open(&dir).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_push_validates_frame_len_and_reward_consistency() {
        let dir = tmp("validate_push");
        let _ = fs::remove_dir_all(&dir);
        let mut w = EpisodeWriter::create(&dir, 1, 2, 2, 3, 10).unwrap();
        assert!(w.push(&[0; 3], 0, None).is_err(), "wrong frame len must be rejected");
        w.push(&[0; 4], 0, Some(1.0)).unwrap();
        assert!(w.push(&[0; 4], 0, None).is_err(), "reward presence must be consistent");
        drop(w);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_meta_file_size_mismatch_errors() {
        // Truncated frames.u8.
        let dir = tmp("mismatch_frames");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[3], true, 1, 2, 2);
        let frames = fs::read(dir.join("frames.u8")).unwrap();
        fs::write(dir.join("frames.u8"), &frames[..frames.len() - 1]).unwrap();
        assert!(EpisodeDataset::open(&dir).unwrap_err().contains("frames.u8"));
        let _ = fs::remove_dir_all(&dir);

        // Oversized actions.bin.
        let dir = tmp("mismatch_actions");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[3], true, 1, 2, 2);
        let mut acts = fs::read(dir.join("actions.bin")).unwrap();
        acts.extend_from_slice(&9u32.to_le_bytes());
        fs::write(dir.join("actions.bin"), acts).unwrap();
        assert!(EpisodeDataset::open(&dir).unwrap_err().contains("actions.bin"));
        let _ = fs::remove_dir_all(&dir);

        // Wrong-length rewards.f32.
        let dir = tmp("mismatch_rewards");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[3], true, 1, 2, 2);
        fs::write(dir.join("rewards.f32"), 0f32.to_le_bytes()).unwrap();
        assert!(EpisodeDataset::open(&dir).unwrap_err().contains("rewards.f32"));
        let _ = fs::remove_dir_all(&dir);

        // Episode table not covering n.
        let dir = tmp("mismatch_episodes");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[3], true, 1, 2, 2);
        let meta = fs::read_to_string(dir.join("meta.json")).unwrap();
        fs::write(dir.join("meta.json"), meta.replace("\"len\":3", "\"len\":2")).unwrap();
        assert!(EpisodeDataset::open(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_sample_window_never_crosses_boundaries() {
        let dir = tmp("window_bounds");
        let _ = fs::remove_dir_all(&dir);
        // Episode 1 (len 3) is too short for t_len 4 and must never be hit.
        write_ds(&dir, &[8, 3, 6], true, 1, 2, 2);
        let ds = EpisodeDataset::open(&dir).unwrap();
        let t_len = 4;
        let mut rng = Rng::new(42);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            let w = ds.sample_window(&mut rng, t_len).unwrap();
            let s = w.start_index;
            let ep = ds
                .episodes
                .iter()
                .find(|e| s >= e.start && s + t_len <= e.start + e.len)
                .unwrap_or_else(|| panic!("window [{s}, {}) crosses an episode boundary", s + t_len));
            assert!(ep.len >= t_len);
            // Actions encode the flat frame index: the window content must
            // be the t_len consecutive steps at start_index.
            let want: Vec<u32> = (s as u32..(s + t_len) as u32).collect();
            assert_eq!(w.actions, want);
            assert_eq!(w.frames_f32.len(), t_len * ds.frame_len());
            seen.insert(s);
        }
        // Uniform over valid starts: every one of the (8-4+1)+(6-4+1)=8 valid
        // starts appears within 500 draws.
        assert_eq!(seen.len(), 8, "expected all 8 valid starts, saw {seen:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_sample_window_deterministic_per_seed_and_errs_when_too_long() {
        let dir = tmp("window_seed");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[6, 5], false, 1, 2, 2);
        let ds = EpisodeDataset::open(&dir).unwrap();
        let draw = |seed: u64| -> Vec<usize> {
            let mut rng = Rng::new(seed);
            (0..50).map(|_| ds.sample_window(&mut rng, 3).unwrap().start_index).collect()
        };
        assert_eq!(draw(7), draw(7), "same seed must draw the same windows");
        assert_ne!(draw(7), draw(8), "different seeds should differ");
        assert!(ds.sample_window(&mut Rng::new(1), 7).is_err(), "no episode has 7 frames");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_iter_windows_covers_episodes_without_crossing() {
        let dir = tmp("iter");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[5, 4], false, 1, 2, 2);
        let ds = EpisodeDataset::open(&dir).unwrap();
        let starts: Vec<usize> = ds.iter_windows(3, 2).map(|w| w.start_index).collect();
        // Episode 0 (frames 0..5): starts 0, 2; episode 1 (frames 5..9): 5.
        // Start 7 would need frames 7..10 and 9 is out; start 4 would cross.
        assert_eq!(starts, vec![0, 2, 5]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn episode_split_by_episode_disjoint_and_covers_all() {
        let dir = tmp("split");
        let _ = fs::remove_dir_all(&dir);
        write_ds(&dir, &[2, 2, 2, 2, 2], false, 1, 2, 2);
        let ds = EpisodeDataset::open(&dir).unwrap();
        let (train, val) = ds.split(0.8);
        assert_eq!(train, vec![0, 1, 2, 3]);
        assert_eq!(val, vec![4]);
        let mut all = train.clone();
        all.extend(&val);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all, (0..5).collect::<Vec<_>>(), "split must be disjoint and cover all episodes");
        let (t0, v0) = ds.split(0.0);
        assert!(t0.is_empty());
        assert_eq!(v0.len(), 5);
        let (t1, v1) = ds.split(1.0);
        assert_eq!(t1.len(), 5);
        assert!(v1.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
