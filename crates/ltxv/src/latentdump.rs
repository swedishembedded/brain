// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A raw `[C, T, H, W]` f32 latent on disk, so the DENOISE half of a
//! generation and the DECODE half can be run, and re-run, independently.
//!
//! Swedish Embedded AB implements reproducible inference-pipeline
//! instrumentation for generative video models for its clients. If your team
//! needs a way to bisect a quality regression between a diffusion transformer
//! and its decoder without re-running either, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # Why this exists
//!
//! A real 1080p clip costs ~35 minutes end to end and the two stages that
//! produce it fail in visually similar ways: a bad latent and a bad decode
//! both come out as a smeared, warped image. Telling them apart needs the
//! SAME latent decoded more than once - through the whole-clip path and the
//! tiled one, at different crops, or not at all (statistics only). Dumping
//! the final latent once and decoding it N times turns a 35-minute A/B into a
//! one-minute one, and removes sampler nondeterminism from the comparison
//! entirely: both arms are literally the same bytes.
//!
//! The format is deliberately the dumbest thing that round-trips: a 20-byte
//! little-endian header (`magic`, `c`, `t`, `h`, `w`) then `c*t*h*w` `f32`s in
//! `[C, T, H, W]` row-major order - the exact layout
//! [`crate::vae3d::LtxVaeDecoder::decode`] takes. No compression, no version
//! negotiation, no dependency.

use std::io::{Read, Write};

/// `"LTXL"` little-endian - a four-byte guard so a truncated or unrelated file
/// is rejected with a message rather than decoded as noise.
const MAGIC: u32 = 0x4c58_544c;

/// The shape of a dumped latent: `[channels, latent_frames, latent_h, latent_w]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentShape {
    pub c: u32,
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl LatentShape {
    pub fn len(&self) -> usize {
        self.c as usize * self.t as usize * self.h as usize * self.w as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Write `data` (`[c, t, h, w]` row-major) to `path`.
pub fn write(path: &str, shape: LatentShape, data: &[f32]) -> Result<(), String> {
    assert_eq!(data.len(), shape.len(), "latent dump: {shape:?} needs {} values, got {}", shape.len(), data.len());
    let mut f = std::fs::File::create(path).map_err(|e| format!("creating {path}: {e}"))?;
    let mut head = Vec::with_capacity(20);
    for v in [MAGIC, shape.c, shape.t, shape.h, shape.w] {
        head.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&head).map_err(|e| format!("writing {path}: {e}"))?;
    let mut buf = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf).map_err(|e| format!("writing {path}: {e}"))?;
    Ok(())
}

/// Read back what [`write`] produced.
pub fn read(path: &str) -> Result<(LatentShape, Vec<f32>), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("opening {path}: {e}"))?;
    let mut head = [0u8; 20];
    f.read_exact(&mut head).map_err(|e| format!("reading {path} header: {e}"))?;
    let g = |i: usize| u32::from_le_bytes([head[i * 4], head[i * 4 + 1], head[i * 4 + 2], head[i * 4 + 3]]);
    if g(0) != MAGIC {
        return Err(format!("{path} is not a latent dump (bad magic)"));
    }
    let shape = LatentShape { c: g(1), t: g(2), h: g(3), w: g(4) };
    let mut raw = Vec::new();
    f.read_to_end(&mut raw).map_err(|e| format!("reading {path}: {e}"))?;
    if raw.len() != shape.len() * 4 {
        return Err(format!("{path}: {shape:?} needs {} bytes of payload, file has {}", shape.len() * 4, raw.len()));
    }
    let data = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    Ok((shape, data))
}

/// If `BRAIN_LTXV_LATENT_DUMP` is set, write the latent about to be decoded to
/// that path. A dump failure is traced, never fatal: this is diagnostic
/// plumbing on the path of a 35-minute generation and must not be able to
/// destroy one.
pub fn dump_if_requested(shape: LatentShape, data: &[f32]) {
    let Ok(path) = std::env::var("BRAIN_LTXV_LATENT_DUMP") else { return };
    match write(&path, shape, data) {
        Ok(()) => tracing::info!(path, ?shape, "wrote final latent"),
        Err(e) => tracing::warn!(error = %e, "latent dump failed (generation continues)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_latent_round_trips_through_the_file_bit_for_bit() {
        let shape = LatentShape { c: 3, t: 2, h: 4, w: 5 };
        let data: Vec<f32> = (0..shape.len()).map(|i| (i as f32) * 0.5 - 7.25).collect();
        let path = std::env::temp_dir().join("brain_ltxv_latentdump_roundtrip.bin");
        let path = path.to_str().unwrap();
        write(path, shape, &data).unwrap();
        let (got_shape, got) = read(path).unwrap();
        assert_eq!(got_shape, shape);
        assert_eq!(got.len(), data.len());
        for (i, (a, b)) in got.iter().zip(&data).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "value {i} changed across the round trip");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_file_that_is_not_a_latent_dump_is_rejected() {
        let path = std::env::temp_dir().join("brain_ltxv_latentdump_bogus.bin");
        let path = path.to_str().unwrap();
        std::fs::write(path, vec![0u8; 64]).unwrap();
        assert!(read(path).is_err(), "a file with the wrong magic must not decode as a latent");
        let _ = std::fs::remove_file(path);
    }
}
