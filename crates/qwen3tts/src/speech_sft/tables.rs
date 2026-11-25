// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The codec tables and the frozen text path of a speech fine-tune as host
//! generation reads them ([`crate::prompt::TalkerHost`], [`crate::prompt::MtpHost`]):
//! the reference the device-assembled training rows are checked against. The
//! run itself keeps its tables on the device ([`super::model`]).

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
