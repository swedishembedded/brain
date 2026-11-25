// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a loaded batch's forward predicted and passed through, read back for
//! calibration and parity: the logits of every supervised position of both
//! decoders and the residual stream at every layer boundary.

use model::hostmath::linear_rows;

use super::SpeechSftModel;

/// The logits of every supervised position of the latest forward, each with
/// its target: the Talker's `[talker_targets.len(), talker_vocab]` (row
/// order), the code predictor's `[cp_targets.len(), cp_vocab]` (frame by
/// frame, codebooks `1..` in order).
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    pub talker: Vec<f32>,
    pub talker_targets: Vec<u32>,
    pub talker_vocab: usize,
    pub cp: Vec<f32>,
    pub cp_targets: Vec<u32>,
    pub cp_vocab: usize,
}

/// Which decoder a residual state is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderStack {
    Talker,
    CodePredictor,
}

impl SpeechSftModel {
    /// The logits of the latest forward's supervised positions (run
    /// [`Self::score`] or the forward first), from the decoders' final hidden
    /// states and their heads, on the host.
    pub fn logits(&self) -> Logits {
        let loaded = self.loaded.borrow();
        let l = loaded.as_ref().expect("speech sft: load a batch and run the forward before reading its logits");
        let s = &self.shape;
        let (d, v) = (s.d(), s.talker.vocab as usize);
        let hidden = self.talker.final_hidden();
        let rows: Vec<f32> = l.talker_targets.iter().flat_map(|&(r, _)| hidden[r as usize * d..(r as usize + 1) * d].iter().copied()).collect();
        let talker = linear_rows(&rows, &self.talker.read_weight("lm_head.weight"), l.talker_targets.len(), d, v);

        let (g, dc, cv) = (s.mtp.num_code_groups as usize, s.mtp.d_model as usize, s.mtp.vocab as usize);
        let frames = l.frames as usize;
        let cp_hidden = self.cp.final_hidden();
        let heads = self.cp.read_weight("lm_head.weight");
        let mut per_head: Vec<Vec<f32>> = Vec::with_capacity(g - 1);
        for p in 1..g {
            let rows: Vec<f32> = (0..frames).flat_map(|f| cp_hidden[(f * g + p) * dc..(f * g + p + 1) * dc].iter().copied()).collect();
            per_head.push(linear_rows(&rows, &heads[(p - 1) * cv * dc..p * cv * dc], frames, dc, cv));
        }
        let mut cp = Vec::with_capacity(frames * (g - 1) * cv);
        let mut cp_targets = Vec::with_capacity(frames * (g - 1));
        for f in 0..frames {
            for p in 1..g {
                cp.extend_from_slice(&per_head[p - 1][f * cv..(f + 1) * cv]);
                cp_targets.push(l.cp_codes[f * g + p]);
            }
        }
        Logits { talker, talker_targets: l.talker_targets.iter().map(|&(_, y)| y).collect(), talker_vocab: v, cp, cp_targets, cp_vocab: cv }
    }

    /// Residual stream points of `which` decoder after the latest forward
    /// (`2l` enters layer `l`, `2l + 1` is after its attention, `2 * layers`
    /// leaves the last), each as `[rows, width]` over the rows that hold data
    /// (no padding), handed to `f(point, rows)`.
    pub fn for_each_residual(&self, which: DecoderStack, f: &mut dyn FnMut(usize, &[f32])) {
        let loaded = self.loaded.borrow();
        let l = loaded.as_ref().expect("speech sft: load a batch and run the forward before reading its states");
        let s = &self.shape;
        let (q, width, layers) = match which {
            DecoderStack::Talker => (&self.talker, s.d(), s.talker.n_layers as usize),
            DecoderStack::CodePredictor => (&self.cp, s.mtp.d_model as usize, s.mtp.n_layers as usize),
        };
        for point in 0..=2 * layers {
            let all = q.read_residual(point);
            let rows: Vec<f32> = match which {
                DecoderStack::Talker => l.talker_valid.iter().flat_map(|&r| all[r as usize * width..(r as usize + 1) * width].iter().copied()).collect(),
                DecoderStack::CodePredictor => all[..l.frames as usize * s.mtp.num_code_groups as usize * width].to_vec(),
            };
            f(point, &rows);
        }
    }
}
