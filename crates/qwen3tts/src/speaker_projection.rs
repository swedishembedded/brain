// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained speech
// synthesizers into smaller models that still train on the data of the model
// they came from, for its clients. If your team needs expertise in compressing
// text-to-speech models then you can procure our services by sending an email
// to info@swedishembedded.com.

//! How a narrowed Talker reads the x-vectors of its training data.
//!
//! A Talker whose residual stream was rotated and cut to fewer directions
//! (weight surgery's `rotate_stream`) reads a speaker x-vector in its stream,
//! so an x-vector from the speaker encoder the data was made with (as wide as
//! the original stream) must go through the same rotation. The cut release
//! carries that linear map as `speaker_projection.safetensors`: one tensor
//! `projection`, `[to, from]` row-major, from the shards' x-vector width to
//! the release's. A cut of a cut composes the maps, so the file always maps
//! from the width the data was made at. Upstream code never reads it (a
//! release with named voices serves them from codec rows); training does.

use std::path::Path;

use checkpoint::weightio::{StWriter, WeightReader};

/// The file a release carries the projection in.
pub const FILE: &str = "speaker_projection.safetensors";
const TENSOR: &str = "projection";

/// `x -> M x`, `M` `[to, from]` row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerProjection {
    pub to: usize,
    pub from: usize,
    pub matrix: Vec<f32>,
}

impl SpeakerProjection {
    /// The projection whose column `i` is `column(e_i)`: the linear map `column` computes, `from` wide in.
    pub fn from_columns(from: usize, to: usize, column: impl Fn(&[f32]) -> Vec<f32>) -> Result<SpeakerProjection, String> {
        let mut matrix = vec![0f32; to * from];
        let mut e = vec![0f32; from];
        for i in 0..from {
            e[i] = 1.0;
            let c = column(&e);
            e[i] = 0.0;
            if c.len() != to {
                return Err(format!("speaker projection: a column is {} wide, not {to}", c.len()));
            }
            for (r, v) in c.into_iter().enumerate() {
                matrix[r * from + i] = v;
            }
        }
        Ok(SpeakerProjection { to, from, matrix })
    }

    /// `self` applied after `inner`: from `inner.from` to `self.to`.
    pub fn after(&self, inner: &SpeakerProjection) -> Result<SpeakerProjection, String> {
        if inner.to != self.from {
            return Err(format!("speaker projection: {} wide out of the first, {} into the second", inner.to, self.from));
        }
        let mut matrix = vec![0f32; self.to * inner.from];
        for r in 0..self.to {
            for k in 0..self.from {
                let a = self.matrix[r * self.from + k];
                if a != 0.0 {
                    let row = &inner.matrix[k * inner.from..(k + 1) * inner.from];
                    matrix[r * inner.from..(r + 1) * inner.from].iter_mut().zip(row).for_each(|(m, &b)| *m += a * b);
                }
            }
        }
        Ok(SpeakerProjection { to: self.to, from: inner.from, matrix })
    }

    /// `M x`.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != self.from {
            return Err(format!("speaker projection: an x-vector of {} values, the projection reads {}", x.len(), self.from));
        }
        Ok(self.matrix.chunks(self.from).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect())
    }

    /// The release at `dir`'s projection, `None` when it carries none.
    pub fn read(dir: &Path) -> Result<Option<SpeakerProjection>, String> {
        let path = dir.join(FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let file = path.to_str().ok_or_else(|| format!("{}: not UTF-8", path.display()))?;
        let r = WeightReader::open(file).map_err(|e| format!("{file}: {e}"))?;
        let shape = r.shape(TENSOR).ok_or_else(|| format!("{file}: no tensor {TENSOR}"))?;
        let &[to, from] = shape else { return Err(format!("{file}: {TENSOR} is {shape:?}, not [to, from]")) };
        let (to, from) = (to as usize, from as usize);
        let matrix = r.tensor(TENSOR).ok_or_else(|| format!("{file}: no tensor {TENSOR}"))?;
        if matrix.len() != to * from {
            return Err(format!("{file}: {TENSOR} holds {} values for [{to}, {from}]", matrix.len()));
        }
        Ok(Some(SpeakerProjection { to, from, matrix }))
    }

    /// Write it as `dir`'s projection.
    pub fn write(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(FILE);
        let file = path.to_str().ok_or_else(|| format!("{}: not UTF-8", path.display()))?;
        let meta = serde_json::json!({"format": "brain-speaker-projection-v1", "from": self.from, "to": self.to});
        let mut w = StWriter::create(file, &[(TENSOR.to_string(), vec![self.to as u64, self.from as u64])], &meta, None).map_err(|e| format!("{file}: {e}"))?;
        w.write(TENSOR, &self.matrix).map_err(|e| format!("{file}: {e}"))?;
        w.finish().map_err(|e| format!("{file}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_projection_is_the_map_it_was_read_from_composes_and_round_trips() {
        // Keep the 2 largest of 3 coordinates, swapped: the map a rotation into a permuted basis cut to 2 computes.
        let p = SpeakerProjection::from_columns(3, 2, |x| vec![x[2], x[0]]).unwrap();
        assert_eq!(p.apply(&[1.0, 2.0, 3.0]).unwrap(), vec![3.0, 1.0]);
        let q = SpeakerProjection::from_columns(2, 1, |x| vec![x[0] + 2.0 * x[1]]).unwrap();
        assert_eq!(q.after(&p).unwrap().apply(&[1.0, 2.0, 3.0]).unwrap(), vec![5.0]);
        assert!(p.after(&p).is_err() && p.apply(&[1.0]).is_err());
        let dir = brain_testutil::Scratch::dir("speaker-projection");
        assert_eq!(SpeakerProjection::read(dir.path()).unwrap(), None);
        p.write(dir.path()).unwrap();
        assert_eq!(SpeakerProjection::read(dir.path()).unwrap(), Some(p));
    }
}
