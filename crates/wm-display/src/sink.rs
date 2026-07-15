// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Frame sinks: where rendered RGB8 frames go. The window is one sink; CI
//! uses the headless ones (hashes for golden tests, PPM dumps for artifacts).

use std::io::Write;

/// Per-frame HUD state shown by sinks that can display it.
#[derive(Clone, Debug, Default)]
pub struct Hud {
    pub model: String,
    pub fps: f32,
    pub target_fps: u32,
    pub step: u64,
    pub paused: bool,
    pub quality: u32,
    /// The action CONSUMED by the step that produced this frame (recorders
    /// pair it with the frame; see `record::RecorderSink`).
    pub action: u32,
    /// True on the first frame emitted after a `UxKey::Reset` — i.e. this
    /// frame starts a new episode. Recorders close the previous episode when
    /// they see it.
    pub reset: bool,
}

/// A consumer of interleaved RGB8 frames (`w*h*3` bytes).
pub trait FrameSink {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud);
}

/// Discards frames (pure pacing/bench runs).
pub struct HeadlessSink;

impl FrameSink for HeadlessSink {
    fn frame(&mut self, _rgb: &[u8], _w: u32, _h: u32, _hud: &Hud) {}
}

/// FNV-1a hash of every frame, for golden-rollout tests.
#[derive(Default)]
pub struct HashSink {
    pub hashes: Vec<u64>,
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl FrameSink for HashSink {
    fn frame(&mut self, rgb: &[u8], _w: u32, _h: u32, _hud: &Hud) {
        self.hashes.push(fnv1a(rgb));
    }
}

/// Writes each frame as `frame_NNNNNN.ppm` (binary P6) into a directory.
pub struct PpmDirSink {
    dir: std::path::PathBuf,
    n: u64,
}

impl PpmDirSink {
    pub fn new(dir: impl Into<std::path::PathBuf>) -> std::io::Result<PpmDirSink> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(PpmDirSink { dir, n: 0 })
    }
}

impl FrameSink for PpmDirSink {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, _hud: &Hud) {
        let path = self.dir.join(format!("frame_{:06}.ppm", self.n));
        self.n += 1;
        if let Ok(mut f) = std::fs::File::create(path) {
            let _ = write!(f, "P6\n{w} {h}\n255\n");
            let _ = f.write_all(rgb);
        }
    }
}

/// Fans a frame out to two sinks (e.g. window + recorder).
pub struct TeeSink<A: FrameSink, B: FrameSink>(pub A, pub B);

impl<A: FrameSink, B: FrameSink> FrameSink for TeeSink<A, B> {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        self.0.frame(rgb, w, h, hud);
        self.1.frame(rgb, w, h, hud);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sink_fnv1a_known_vector() {
        // FNV-1a 64-bit of "a" is 0xaf63dc4c8601ec8c.
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn sink_hash_collects_per_frame_and_tee_orders() {
        let mut tee = TeeSink(HashSink::default(), HashSink::default());
        let hud = Hud::default();
        tee.frame(&[1, 2, 3], 1, 1, &hud);
        tee.frame(&[4, 5, 6], 1, 1, &hud);
        assert_eq!(tee.0.hashes, tee.1.hashes);
        assert_eq!(tee.0.hashes.len(), 2);
        assert_ne!(tee.0.hashes[0], tee.0.hashes[1]);
    }

    #[test]
    fn sink_ppm_writes_decodable_p6() {
        let dir = std::env::temp_dir().join(format!("wm_ppm_test_{}", std::process::id()));
        let mut s = PpmDirSink::new(&dir).unwrap();
        s.frame(&[255, 0, 0, 0, 255, 0], 2, 1, &Hud::default());
        let bytes = std::fs::read(dir.join("frame_000000.ppm")).unwrap();
        assert!(bytes.starts_with(b"P6\n2 1\n255\n"));
        assert_eq!(&bytes[bytes.len() - 6..], &[255, 0, 0, 0, 255, 0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
