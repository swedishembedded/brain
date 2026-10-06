// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training through the engine's one step loop (`model::train::fit_controlled`
//! owns the schedule, accumulation, clipping, early stopping and resume), and
//! batched prediction.

use data::rng::Rng;
use model::Objective;

use crate::batch::assemble;
use crate::encode::{Encoded, Forecast};
use crate::model::Horizon;

/// The timeline objective: event NLL plus masked-value NLL on random training
/// batches; the held-out event NLL as the evaluation the loop early-stops on.
pub struct TimelineObjective<'a> {
    train: &'a [Encoded],
    held_out: Option<&'a [Encoded]>,
    mask_rate: f64,
    last_eval: Option<f32>,
}

impl<'a> TimelineObjective<'a> {
    /// `mask_rate` is the share of each subject's numeric values hidden for
    /// the value objective on every training batch.
    pub fn new(
        train: &'a [Encoded],
        held_out: Option<&'a [Encoded]>,
        mask_rate: f64,
    ) -> TimelineObjective<'a> {
        assert!(!train.is_empty(), "no training subjects");
        TimelineObjective {
            train,
            held_out,
            mask_rate,
            last_eval: None,
        }
    }
}

impl Objective<Horizon> for TimelineObjective<'_> {
    fn regime(&self) -> &'static str {
        "timeline"
    }

    fn micro_step(&mut self, model: &Horizon, rng: &mut Rng) -> f32 {
        let b = model.batch_size() as usize;
        let picks: Vec<&Encoded> = (0..b)
            .map(|_| &self.train[(rng.next_u64() % self.train.len() as u64) as usize])
            .collect();
        let hb = assemble(&model.cfg, &picks, b, self.mask_rate, rng);
        model.set_batch(&hb);
        let loss = model.forward();
        model.backward();
        loss
    }

    /// The whole held-out set's weighted event NLL (batches are not sampled:
    /// an early-stopping decision on a noisy estimate stops on noise).
    fn eval(&mut self, model: &Horizon, _rng: &mut Rng, _batches: u32) -> Option<f32> {
        let held_out = self.held_out?;
        let loss = event_nll(model, held_out);
        self.last_eval = Some(loss);
        Some(loss)
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        self.last_eval
            .map(|l| vec![("held_out_event_nll", l)])
            .unwrap_or_default()
    }
}

/// Run `subjects` through the model in batches, calling `each` with every
/// batch's subjects and the model state after its forward.
fn for_each_batch(
    model: &Horizon,
    subjects: &[Encoded],
    mut each: impl FnMut(&[Encoded], &Horizon),
) {
    let b = model.batch_size() as usize;
    let mut rng = Rng::new(0); // no masking: the rng is never consulted
    for chunk in subjects.chunks(b) {
        let refs: Vec<&Encoded> = chunk.iter().collect();
        model.set_batch(&assemble(&model.cfg, &refs, b, 0.0, &mut rng));
        model.forward_submit();
        each(chunk, model);
    }
}

/// The weighted mean event NLL of the OUTCOME codes over all of `subjects`
/// (per unit of weight): the next-event group, a training signal, is not part
/// of it.
pub fn event_nll(model: &Horizon, subjects: &[Encoded]) -> f32 {
    let (mut total, mut weight) = (0.0f64, 0.0f64);
    for_each_batch(model, subjects, |chunk, m| {
        let w: f64 = chunk.iter().map(|s| s.weight as f64).sum();
        total += m.loss_split().0 as f64 * w;
        weight += w;
    });
    (total / weight) as f32
}

/// Every subject's own event NLL of the OUTCOME codes (per unit of its
/// weight), in order, from one pass over the batches: what [`event_nll`]
/// averages. A subject's value does not depend on its batch neighbours.
pub fn event_nll_each(model: &Horizon, subjects: &[Encoded]) -> Vec<f32> {
    let mut out = Vec::with_capacity(subjects.len());
    for_each_batch(model, subjects, |chunk, m| {
        let batch_weight: f64 = chunk.iter().map(|s| s.weight as f64).sum();
        let by_slot = m.outcome_loss_by_slot();
        // The kernel wrote w_i / sum(w) times the subject's NLL.
        out.extend(chunk.iter().zip(by_slot).map(|(s, loss)| (loss * batch_weight / s.weight as f64) as f32));
    });
    out
}

/// Log-hazards `[pieces * codes]` for every subject, in order.
pub fn predict_log_hazards(model: &Horizon, subjects: &[Encoded]) -> Vec<Vec<f32>> {
    let per = (model.cfg.pieces() * model.cfg.n_codes) as usize;
    let mut out = Vec::with_capacity(subjects.len());
    for_each_batch(model, subjects, |chunk, m| {
        let lh = m.read_log_hazards();
        out.extend((0..chunk.len()).map(|i| lh[i * per..(i + 1) * per].to_vec()));
    });
    out
}

/// The log-hazards and the summary state of every subject, in order, from ONE
/// forward pass.
pub fn predict_hazards_and_states(
    model: &Horizon,
    subjects: &[Encoded],
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let per = (model.cfg.pieces() * model.cfg.n_codes) as usize;
    let d = model.cfg.d_model as usize;
    let mut out = Vec::with_capacity(subjects.len());
    for_each_batch(model, subjects, |chunk, m| {
        let (lh, z) = (m.read_log_hazards(), m.read_state());
        out.extend((0..chunk.len()).map(|i| {
            (
                lh[i * per..(i + 1) * per].to_vec(),
                z[i * d..(i + 1) * d].to_vec(),
            )
        }));
    });
    out
}

/// The summary state `[d_model]` of every subject, in order.
pub fn predict_states(model: &Horizon, subjects: &[Encoded]) -> Vec<Vec<f32>> {
    let d = model.cfg.d_model as usize;
    let mut out = Vec::with_capacity(subjects.len());
    for_each_batch(model, subjects, |chunk, m| {
        let z = m.read_state();
        out.extend((0..chunk.len()).map(|i| z[i * d..(i + 1) * d].to_vec()));
    });
    out
}

/// `(mu, sigma)` on the normal-score scale for each subject's forecast
/// queries (at most `cfg.forecasts` per subject, built with
/// [`crate::encode::forecast_query`]), in order.
pub fn predict_forecasts(
    model: &Horizon,
    subjects: &[Encoded],
    queries: &[Vec<Forecast>],
) -> Result<Vec<Vec<(f32, f32)>>, String> {
    let nf = model.cfg.forecasts as usize;
    if subjects.len() != queries.len() {
        return Err(format!(
            "{} subjects but {} query lists",
            subjects.len(),
            queries.len()
        ));
    }
    if let Some(q) = queries.iter().find(|q| q.len() > nf) {
        return Err(format!(
            "{} forecast queries for one subject; the model takes at most {nf}",
            q.len()
        ));
    }
    let asked: Vec<Encoded> = subjects
        .iter()
        .zip(queries)
        .map(|(s, q)| Encoded {
            forecasts: q.clone(),
            ..s.clone()
        })
        .collect();
    let mut out = Vec::with_capacity(subjects.len());
    for_each_batch(model, &asked, |chunk, m| {
        let p = m.read_forecasts();
        for (i, s) in chunk.iter().enumerate() {
            out.push(
                (0..s.forecasts.len())
                    .map(|j| (p[2 * (i * nf + j)], p[2 * (i * nf + j) + 1].exp()))
                    .collect(),
            );
        }
    });
    Ok(out)
}
