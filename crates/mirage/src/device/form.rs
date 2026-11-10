// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Form scoring on the device, for a batch of images that answer the same form.
//!
//! The decoder steps one token per sequence at a position the whole batch shares, so the images of a form are one
//! batch while their committed answers have the same token length, and a cohort that splits (two images commit
//! answers of different lengths) continues as one cohort per length. Within a cohort every candidate of a field
//! is scored by rewinding the state to the end of the field head, feeding the candidate's tokens as the batch's
//! next positions and reading the logits of every step: the prefix is computed once per field and the cache
//! rows beyond it are simply overwritten by the next candidate. Free text and box values are generated one image
//! at a time (their lengths differ per image by nature).
//!
//! The arithmetic of a score, the candidate list, the terminators and the settling into probabilities are the
//! host scorer's own functions (`crate::form`), so both scorers answer the same question.

use super::batch_model::DecoderMemory;
use super::decoder::{BatchState, DeviceCross, DeviceDecoder};
use super::reader::MemoryBatch;
use super::Device;
use crate::form::{self, Answer, Confidence, Extract, FormError, FormField, Terminators};
use crate::tokenizer::Tokenizer;
use serde_json::Value;

/// Images that continue together: the same committed length, so the same position.
struct Cohort {
    /// Index of each row's image in the caller's list.
    images: Vec<usize>,
    state: BatchState,
    cross: DeviceCross,
    /// Every row's tokens so far (prompt, spec, heads and committed answers), the first `state.position()` fed.
    seqs: Vec<Vec<u32>>,
}

impl DeviceDecoder {
    /// Answer `fields` for every memory in `memories` (one image each): the device twin of
    /// [`crate::form::extract`], image for image.
    pub fn extract_forms(&self, dev: &Device, tok: &Tokenizer, prompt: &[u32], memories: &[&DecoderMemory], fields: &[FormField], max_tokens: usize) -> Result<Vec<Extract>, FormError> {
        let mut base = prompt.to_vec();
        base.extend(form::spec_tokens(tok, fields)?);
        let ends = Terminators::new(tok)?;
        // Room for the prompt and, per field, the head, the longest answer and its terminator.
        let mut room = base.len() + 2;
        for (i, f) in fields.iter().enumerate() {
            let head = form::head_tokens(tok, f)?.len();
            let answer = match form::generated_length(&f.kind) {
                Some(length) => length.min(max_tokens).max(1) + 1,
                None => form::candidates(tok, &f.kind)?.iter().map(|c| c.sequence(ends.after(i, fields.len())).len()).max().unwrap_or(1),
            };
            room += head + answer + 1;
        }

        let mut answers: Vec<Vec<(String, Answer)>> = vec![Vec::new(); memories.len()];
        let all: Vec<usize> = (0..memories.len()).collect();
        let mut root = self.cohort(dev, memories, &all, room, vec![base; memories.len()]);
        self.feed(dev, &mut root, false);
        let mut work = vec![(root, 0usize)];
        while let Some((mut co, f)) = work.pop() {
            let Some(field) = fields.get(f) else { continue };
            let terminator = ends.after(f, fields.len());
            co.seqs.iter_mut().try_for_each(|s| form::head_tokens(tok, field).map(|h| s.extend(h)))?;
            let head_logits = self.feed(dev, &mut co, true).expect("logits were asked for");
            let vocab = head_logits.len() / co.images.len();

            if let Some(length) = form::generated_length(&field.kind) {
                for row in 0..co.images.len() {
                    let mut single = self.cohort_of(dev, memories, &co, &[row]);
                    let (text, probability) = self.generate(dev, tok, &mut single, head_logits[row * vocab..(row + 1) * vocab].to_vec(), terminator, length.min(max_tokens))?;
                    answers[single.images[0]].push((field.name.clone(), Answer { value: Value::String(text), confidence: Confidence::Generated(probability) }));
                    work.push((single, f + 1));
                }
                continue;
            }

            let candidates = form::candidates(tok, &field.kind)?;
            let rows = co.images.len();
            let saved = co.state.position();
            let before = co.seqs[0].len();
            let first: Vec<Vec<f32>> = head_logits.chunks_exact(vocab).map(form::log_softmax).collect();
            let mut scores = vec![vec![0f32; candidates.len()]; rows];
            for (ci, c) in candidates.iter().enumerate() {
                let seq = c.sequence(terminator);
                for (row, score) in first.iter().zip(scores.iter_mut()) {
                    score[ci] = row[seq[0] as usize];
                }
                if seq.len() > 1 {
                    co.state.seek(dev, saved);
                    co.seqs.iter_mut().for_each(|s| s.extend_from_slice(&seq[..seq.len() - 1]));
                    for (t, logits) in self.feed_each(dev, &mut co).iter().enumerate() {
                        for (r, row) in logits.chunks_exact(vocab).enumerate() {
                            scores[r][ci] += form::log_softmax(row)[seq[t + 1] as usize];
                        }
                    }
                    co.seqs.iter_mut().for_each(|s| s.truncate(before));
                }
            }
            // Settle every row, then group the rows by the length of the answer they commit.
            let mut groups: std::collections::BTreeMap<usize, Vec<(usize, Vec<u32>)>> = std::collections::BTreeMap::new();
            for (row, row_scores) in scores.iter().enumerate() {
                let (best, probabilities) = form::settle(&candidates, row_scores);
                answers[co.images[row]].push((field.name.clone(), Answer { value: candidates[best].value().clone(), confidence: Confidence::Candidates(probabilities) }));
                let commit = candidates[best].sequence(terminator);
                groups.entry(commit.len()).or_default().push((row, commit));
            }
            co.state.seek(dev, saved);
            for (_, members) in groups {
                let keep: Vec<usize> = members.iter().map(|(r, _)| *r).collect();
                let mut next = self.cohort_of(dev, memories, &co, &keep);
                for (s, (_, commit)) in next.seqs.iter_mut().zip(&members) {
                    s.extend_from_slice(commit);
                }
                self.feed(dev, &mut next, false);
                work.push((next, f + 1));
            }
        }
        Ok(answers.into_iter().map(|fields| Extract { fields }).collect())
    }

    /// A cohort of the memories `images` with `seqs`, nothing fed yet.
    fn cohort(&self, dev: &Device, memories: &[&DecoderMemory], images: &[usize], room: usize, seqs: Vec<Vec<u32>>) -> Cohort {
        let batch = upload_memories(dev, images.iter().map(|&i| memories[i]));
        let cross = self.cross(dev, &batch);
        Cohort { images: images.to_vec(), state: self.new_state(dev, images.len(), room), cross, seqs }
    }

    /// The rows `keep` of `from` as a cohort of their own (state copied, cross-attention rebuilt for them).
    fn cohort_of(&self, dev: &Device, memories: &[&DecoderMemory], from: &Cohort, keep: &[usize]) -> Cohort {
        let images: Vec<usize> = keep.iter().map(|&r| from.images[r]).collect();
        let batch = upload_memories(dev, images.iter().map(|&i| memories[i]));
        Cohort { cross: self.cross(dev, &batch), state: self.select_state(dev, &from.state, keep), seqs: keep.iter().map(|&r| from.seqs[r].clone()).collect(), images }
    }

    /// Feed the tokens every row has beyond the fed positions; the logits `[rows, vocab]` of the last step when asked.
    fn feed(&self, dev: &Device, co: &mut Cohort, want_logits: bool) -> Option<Vec<f32>> {
        let target = co.seqs[0].len();
        co.state.set_tokens(dev, &co.seqs);
        let mut last = None;
        while co.state.position() < target {
            let want = want_logits && co.state.position() + 1 == target;
            last = self.step(dev, &mut co.state, &co.cross, want, None);
        }
        last.map(|b| dev.read_at(&b, 0, co.images.len() * self.vocabulary()))
    }

    /// Like [`Self::feed`] with the logits of every step.
    fn feed_each(&self, dev: &Device, co: &mut Cohort) -> Vec<Vec<f32>> {
        let target = co.seqs[0].len();
        co.state.set_tokens(dev, &co.seqs);
        let mut out = Vec::new();
        while co.state.position() < target {
            let logits = self.step(dev, &mut co.state, &co.cross, true, None).expect("logits were asked for");
            out.push(dev.read_at(&logits, 0, co.images.len() * self.vocabulary()));
        }
        out
    }

    /// Greedy text for the single row of `co`, whose head has just been fed (`logits` follow it).
    fn generate(&self, dev: &Device, tok: &Tokenizer, co: &mut Cohort, mut logits: Vec<f32>, terminator: &[u32], budget: usize) -> Result<(String, f64), FormError> {
        let mut out = Vec::new();
        let mut probability = 1.0f64;
        for _ in 0..budget.max(1) {
            let lp = form::log_softmax(&logits);
            let next = lp.iter().enumerate().fold(0, |b, (i, v)| if *v > lp[b] { i } else { b }) as u32;
            probability *= f64::from(lp[next as usize]).exp();
            if terminator.first() == Some(&next) {
                co.seqs[0].extend_from_slice(terminator);
                self.feed(dev, co, false);
                return Ok((tok.decode(&out)?, probability));
            }
            out.push(next);
            co.seqs[0].push(next);
            logits = self.feed(dev, co, true).expect("logits were asked for");
        }
        co.seqs[0].extend_from_slice(terminator);
        self.feed(dev, co, false);
        Ok((tok.decode(&out)?, probability))
    }
}

/// The memories as one device batch, rows beyond an image's own masked out.
fn upload_memories<'a>(dev: &Device, memories: impl Iterator<Item = &'a DecoderMemory> + Clone) -> MemoryBatch {
    let batch = memories.clone().count();
    let dim = memories.clone().next().map_or(0, |m| m.dim);
    let stride = memories.clone().map(|m| m.rows.len() / dim).max().unwrap_or(0);
    let mut rows = vec![0f32; batch * stride * dim];
    let mut mask = vec![0f32; batch * stride];
    for (k, m) in memories.enumerate() {
        rows[k * stride * dim..k * stride * dim + m.rows.len()].copy_from_slice(&m.rows);
        mask[k * stride..k * stride + m.rows.len() / dim].fill(1.0);
    }
    MemoryBatch { rows: dev.buf(&rows), mask: dev.buf(&mask), stride, batch }
}
