// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The host-held codec tables a speech fine-tune trains, and the frozen text
//! path beside them.
//!
//! The Talker's `codec_embedding` (`[vocab, d]`) and the code predictor's
//! residual tables (`num_code_groups - 1` of `[codebook, d]`, in the Talker's
//! width) are gathered row by row into every input the two decoders read, so
//! they live on the host, where a prompt plan and a frame are materialised,
//! and their gradients are scattered back there.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::prompt::{MtpHost, TalkerHost};
use crate::talker::TextProjection;

/// The tables, as [`crate::prompt::TalkerHost`] and [`crate::prompt::MtpHost`]
/// read them.
pub struct HostTables {
    /// The Talker width: of every codec row and of the text projection's output.
    pub d: usize,
    /// The frozen text path. Every token is projected once and kept
    /// ([`TalkerHost::project_text`]): it never changes in this release.
    pub text: TextProjection,
    /// `talker.model.codec_embedding`, `[vocab, d]`.
    pub codec: Vec<f32>,
    /// `talker.code_predictor.model.codec_embedding.{i}`, each `[codebook, d]`.
    pub residual: Vec<Vec<f32>>,
    projected: RefCell<HashMap<u32, Vec<f32>>>,
}

impl HostTables {
    pub fn new(d: usize, text: TextProjection, codec: Vec<f32>, residual: Vec<Vec<f32>>) -> HostTables {
        HostTables { d, text, codec, residual, projected: RefCell::new(HashMap::new()) }
    }
}

impl TalkerHost for HostTables {
    fn d(&self) -> usize {
        self.d
    }

    fn text(&self) -> &TextProjection {
        &self.text
    }

    fn codec_embed(&self, id: u32) -> &[f32] {
        &self.codec[id as usize * self.d..(id as usize + 1) * self.d]
    }

    fn project_text(&self, ids: &[u32]) -> Vec<f32> {
        let mut cache = self.projected.borrow_mut();
        let mut out = Vec::with_capacity(ids.len() * self.d);
        for &id in ids {
            let row = cache.entry(id).or_insert_with(|| self.text.project(&self.text.embed_text(&[id])));
            out.extend_from_slice(row);
        }
        out
    }
}

impl MtpHost for HostTables {
    fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        &self.residual[residual_idx][code as usize * self.d..(code as usize + 1) * self.d]
    }
}

/// The gradients of [`HostTables`]' trained tables, the same shapes.
pub struct HostGrads {
    pub d: usize,
    pub codec: Vec<f32>,
    pub residual: Vec<Vec<f32>>,
}

impl HostGrads {
    pub fn zeros_like(t: &HostTables) -> HostGrads {
        HostGrads { d: t.d, codec: vec![0.0; t.codec.len()], residual: t.residual.iter().map(|r| vec![0.0; r.len()]).collect() }
    }

    pub fn clear(&mut self) {
        self.codec.fill(0.0);
        self.residual.iter_mut().for_each(|r| r.fill(0.0));
    }

    /// `codec[id] += g`.
    pub fn add_codec(&mut self, id: u32, g: &[f32]) {
        add(&mut self.codec[id as usize * self.d..(id as usize + 1) * self.d], g);
    }

    /// `residual[table][code] += g`.
    pub fn add_residual(&mut self, table: usize, code: u32, g: &[f32]) {
        add(&mut self.residual[table][code as usize * self.d..(code as usize + 1) * self.d], g);
    }

    /// The sum of squares of every gradient, in f64.
    pub fn sum_sq(&self) -> f64 {
        self.codec.iter().chain(self.residual.iter().flatten()).map(|&g| f64::from(g) * f64::from(g)).sum()
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}
