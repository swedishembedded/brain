// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of models whose parameters live in host memory for its
// clients. If your team needs expertise in holding optimizer state in narrow formats, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The optimizer of a model whose parameters live in host memory ([`HostAdam`]): AdamW through the streamed optimizer
//! every other host-side update in the tree runs (`optim::streamed`), with the moments in the state format the run's
//! precision names (f32, bf16 or blockwise int8/log8), so a host trainer's optimizer state costs 8, 4 or about 2 bytes
//! per parameter instead of a hand-rolled pair of f32 vectors.
//!
//! A trainer hands each step its parameters as `(value, gradient)` slice pairs, in the order it built the optimizer
//! with; the update rewrites the values in place.
//!
//! `optim::NamedMoments` is the f32, lazily keyed form for a trainer that discovers its parameters as it steps them;
//! this one knows its tensors up front, which is what a state held in blocks needs.

use std::io;

use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, ParamTiles, StreamedOptimizer, TensorSpec};
use paramstore::precision::{MasterPolicy, ParamPrecision, ValueDtype};

use crate::resident_ft::StepArgs;

/// Elements one tile of the host update covers.
const TILE: usize = 1 << 20;

/// Host-side AdamW over named host tensors, its moments in one state format.
pub struct HostAdam {
    opt: StreamedOptimizer<BlockStateStorage<MemBacking>>,
    names: Vec<String>,
    numels: Vec<usize>,
}

/// One tensor's `(value, gradient)`, read and written by the streamed optimizer in tiles.
struct HostTensor<'a> {
    value: &'a mut [f32],
    grad: &'a [f32],
}

impl ParamTiles for HostTensor<'_> {
    fn read_value(&mut self, _ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        out.copy_from_slice(&self.value[start..start + out.len()]);
        Ok(())
    }

    fn write_value(&mut self, _ti: usize, start: usize, vals: &[f32], _rnd: Rounding) -> io::Result<()> {
        self.value[start..start + vals.len()].copy_from_slice(vals);
        Ok(())
    }

    fn read_grad(&mut self, _ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        out.copy_from_slice(&self.grad[start..start + out.len()]);
        Ok(())
    }
}

impl HostAdam {
    /// AdamW over `params` (`(name, elements)`, the order every step hands them in) with each tensor at `precision`:
    /// an f32 value with no master copy (the host value is the parameter) and any state format. Anything else is
    /// refused by name.
    pub fn new(params: &[(String, usize)], precision: ParamPrecision) -> Result<HostAdam, String> {
        if precision.value != ValueDtype::F32 || precision.master != MasterPolicy::None {
            return Err(format!("a host optimizer updates the f32 host value with no master copy; got {precision:?}"));
        }
        let specs: Vec<TensorSpec> = params.iter().map(|(name, numel)| TensorSpec { name: name.clone(), numel: *numel, precision, row_len: None }).collect();
        let rule = Rule::AdamW { adam: crate::Adam::default(), wd: 0.0 };
        let layout = state_layout(&rule, &specs);
        let state = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), TILE).map_err(|e| e.to_string())?;
        let seed = data::rng::random_seed();
        let opt = StreamedOptimizer::new(rule, specs, state, seed, TILE).map_err(|e| e.to_string())?;
        Ok(HostAdam { opt, names: params.iter().map(|(n, _)| n.clone()).collect(), numels: params.iter().map(|(_, n)| *n).collect() })
    }

    /// The AdamW step `args` describes over `tensors`, every gradient multiplied by the accumulation scale and the
    /// global clip over exactly these tensors ([`crate::grad_multiplier`]).
    pub fn step(&mut self, tensors: &mut [(&mut [f32], &[f32])], args: &StepArgs) -> io::Result<()> {
        if tensors.len() != self.numels.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "host optimizer: the step's tensors differ from the ones it was built for"));
        }
        let sum_sq = if args.clip.is_some() { tensors.iter().flat_map(|(_, g)| g.iter()).map(|&g| (g as f64).powi(2)).sum() } else { 0.0 };
        let multiplier = self.begin_step(args, sum_sq);
        for (ti, (value, grad)) in tensors.iter_mut().enumerate() {
            self.update_tensor(ti, value, grad, args.lr, multiplier)?;
        }
        Ok(())
    }

    /// Open the step `args` describes, for a trainer that holds one tensor at a time: `grad_sum_sq` is the sum of
    /// squares of every raw gradient (0 without a clip). Returns the multiplier [`Self::update_tensor`] applies.
    pub fn begin_step(&mut self, args: &StepArgs, grad_sum_sq: f64) -> f32 {
        self.opt.set_rule(Rule::AdamW { adam: args.adam, wd: args.wd });
        let seed = self.opt.seed();
        self.opt.set_position(args.step - 1, seed);
        self.opt.begin_step();
        crate::grad_multiplier(grad_sum_sq, args.clip, args.extra_scale)
    }

    /// Update tensor `ti` (in the order the optimizer was built with) for the step [`Self::begin_step`] opened; every
    /// tensor once per step.
    pub fn update_tensor(&mut self, ti: usize, value: &mut [f32], grad: &[f32], lr: f32, multiplier: f32) -> io::Result<()> {
        if self.numels.get(ti) != Some(&value.len()) || grad.len() != value.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("host optimizer: tensor {ti} differs from the one it was built for")));
        }
        self.opt.update_tensor(&mut HostTensor { value, grad }, ti, lr, multiplier)
    }

    /// [`Self::step`] over a trainer's `name -> values` maps: every tensor the optimizer was built with must have its
    /// value and gradient there.
    pub fn step_named(&mut self, values: &mut std::collections::HashMap<String, Vec<f32>>, grads: &std::collections::HashMap<String, Vec<f32>>, args: &StepArgs) -> io::Result<()> {
        let order: std::collections::HashMap<&str, usize> = self.names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
        let mut pairs: Vec<(usize, &mut [f32], &[f32])> = values
            .iter_mut()
            .filter_map(|(name, v)| Some((*order.get(name.as_str())?, v.as_mut_slice(), grads.get(name)?.as_slice())))
            .collect();
        if pairs.len() != self.names.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "host optimizer: a tensor it was built for has no value or gradient"));
        }
        pairs.sort_by_key(|(i, _, _)| *i);
        let mut tensors: Vec<(&mut [f32], &[f32])> = pairs.into_iter().map(|(_, v, g)| (v, g)).collect();
        self.step(&mut tensors, args)
    }

    /// Host bytes of the optimizer state.
    pub fn state_bytes(&self) -> u64 {
        self.opt.state_physical_bytes()
    }

    /// The tensors, `(name, elements)`, in the order the optimizer was built with.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, usize)> {
        self.names.iter().map(String::as_str).zip(self.numels.iter().copied())
    }

    /// Key the stochastic rounding of narrow moment formats on `seed` (the run's), so a run resumed under the same
    /// seed rounds as the original did. f32 moments round to nearest and do not read it.
    pub fn pin_seed(&mut self, seed: u64) {
        let step = self.opt.step_count();
        self.opt.set_position(step, seed);
    }

    /// Tensor `ti`'s AdamW moments `(m, v)`, decoded to f32: what an exact resume saves beside the value.
    pub fn read_moments(&mut self, ti: usize) -> io::Result<(Vec<f32>, Vec<f32>)> {
        let numel = self.numel(ti)?;
        let mut m = vec![0.0f32; numel];
        let mut v = vec![0.0f32; numel];
        let tile = self.opt.tile_elems();
        for start in (0..numel).step_by(tile) {
            let n = tile.min(numel - start);
            self.opt.state_mut().read_tile(ti, 0, start, &mut m[start..start + n])?;
            self.opt.state_mut().read_tile(ti, 1, start, &mut v[start..start + n])?;
        }
        Ok((m, v))
    }

    /// Put back tensor `ti`'s moments as [`Self::read_moments`] returned them, encoded into the state format (round to
    /// nearest; f32 moments are restored exactly).
    pub fn write_moments(&mut self, ti: usize, m: &[f32], v: &[f32]) -> io::Result<()> {
        let numel = self.numel(ti)?;
        if m.len() != numel || v.len() != numel {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("host optimizer: tensor {ti} holds {numel} values, the moments {} and {}", m.len(), v.len())));
        }
        let tile = self.opt.tile_elems();
        for start in (0..numel).step_by(tile) {
            let n = tile.min(numel - start);
            self.opt.state_mut().write_tile(ti, 0, start, &m[start..start + n], optim::codec::Rounding::NEAREST)?;
            self.opt.state_mut().write_tile(ti, 1, start, &v[start..start + n], optim::codec::Rounding::NEAREST)?;
        }
        Ok(())
    }

    fn numel(&self, ti: usize) -> io::Result<usize> {
        self.numels.get(ti).copied().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("host optimizer: no tensor {ti} (it holds {})", self.numels.len())))
    }
}

/// The host-optimizer-state contract every model with [`crate::Model::set_host_optimizer`] honours, for its own test
/// to run: `build` makes a fresh trainable model (the same initial weights every call) and `step` runs one optimizer
/// step on a fixed batch (`zero_grads`, the batch, `forward`, `backward`, `adamw_step` at learning rate `lr`).
///
/// * the run's precision reaches the model through the generic path (`state=bf16`, `state=int8`);
/// * the first update in each format equals the f32 run's within the format's quantization: a first AdamW step moves
///   every weight by about `lr * sign(g)` whatever holds the moments, so the two may differ by 5% of `lr`;
/// * a precision the host optimizer does not run (a bf16 value) is refused by name.
pub fn check_host_state_formats<M: crate::Model>(name: &str, build: impl Fn() -> M, step: impl Fn(&M), lr: f32) -> Result<(), String> {
    let values = |m: &M| -> Vec<(String, Vec<f32>)> {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), m.read_weight(&n))).collect()
    };
    let reference = build();
    step(&reference);
    let want = values(&reference);
    for spec in ["state=bf16", "state=int8"] {
        let mut m = build();
        crate::trainprec::apply_train_precision(&mut m, &crate::TrainPrecision::parse(spec)?, name)?;
        step(&m);
        for ((n, a), (_, b)) in want.iter().zip(values(&m)) {
            if let Some(i) = a.iter().zip(&b).position(|(x, y)| (x - y).abs() > 0.05 * lr) {
                return Err(format!("{name}: {spec}: {n}[{i}] is {} after the first update, {} with f32 moments", b[i], a[i]));
            }
        }
    }
    let mut m = build();
    match crate::trainprec::apply_train_precision(&mut m, &crate::TrainPrecision::parse("value=bf16")?, name) {
        Ok(()) => Err(format!("{name}: a bf16 value was accepted by a host optimizer that updates the f32 value")),
        Err(e) if !e.contains(name) => Err(format!("{name}: the refusal does not name the model: {e}")),
        Err(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paramstore::precision::StateFormat;

    fn args(step: u32) -> StepArgs {
        StepArgs { step, lr: 1e-2, wd: 0.01, adam: crate::Adam::default(), clip: None, extra_scale: 1.0 }
    }

    /// Spec: with f32 moments the host update is the optimizer crate's reference AdamW (`Adam::update_slice`, what a
    /// hand-rolled host loop computes) to the last few ulps, step after step.
    #[test]
    fn f32_moments_follow_the_reference_adamw() {
        let n = 1000;
        let grads: Vec<Vec<f32>> = (0..3).map(|s| (0..n).map(|i| ((i * 7 + s * 13) % 17) as f32 / 17.0 - 0.5).collect()).collect();
        let mut w = vec![0.25f32; n];
        let (mut want, mut m, mut v) = (w.clone(), vec![0.0f32; n], vec![0.0f32; n]);
        let mut opt = HostAdam::new(&[("w".to_string(), n)], ParamPrecision::FP32).unwrap();
        for (t, g) in grads.iter().enumerate() {
            let a = args(t as u32 + 1);
            a.adam.update_slice(a.step, a.lr, a.wd, 1.0, &mut want, &mut m, &mut v, g);
            opt.step(&mut [(&mut w[..], &g[..])], &a).unwrap();
        }
        let worst = w.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst <= 4.0 * f32::EPSILON, "host update differs from the reference by {worst}");
    }

    /// Spec: a run that saves the moments after a step and restores them into a fresh optimizer continues exactly as
    /// the uninterrupted one does (f32 moments).
    #[test]
    fn restored_moments_continue_the_run_exactly() {
        let n = 3000;
        let grad = |s: usize| -> Vec<f32> { (0..n).map(|i| ((i * 5 + s * 11) % 13) as f32 / 13.0 - 0.4).collect() };
        let params = [("w".to_string(), n)];
        let mut straight = HostAdam::new(&params, ParamPrecision::FP32).unwrap();
        let mut w_straight = vec![0.5f32; n];
        for t in 1..=3 {
            straight.step(&mut [(&mut w_straight[..], &grad(t)[..])], &args(t as u32)).unwrap();
        }
        let mut first = HostAdam::new(&params, ParamPrecision::FP32).unwrap();
        let mut w = vec![0.5f32; n];
        for t in 1..=2 {
            first.step(&mut [(&mut w[..], &grad(t)[..])], &args(t as u32)).unwrap();
        }
        let (m, v) = first.read_moments(0).unwrap();
        assert!(m.iter().any(|x| *x != 0.0) && v.iter().all(|x| *x >= 0.0));
        let mut resumed = HostAdam::new(&params, ParamPrecision::FP32).unwrap();
        resumed.write_moments(0, &m, &v).unwrap();
        resumed.step(&mut [(&mut w[..], &grad(3)[..])], &args(3)).unwrap();
        assert_eq!(w, w_straight, "a resumed host update differs from the straight run");
        assert!(resumed.write_moments(0, &m[1..], &v).is_err() && resumed.read_moments(1).is_err());
    }

    /// Spec: narrower moments hold fewer host bytes, in order; a precision with a master copy is refused.
    #[test]
    fn narrow_moments_hold_fewer_host_bytes() {
        let params = [("w".to_string(), 1 << 16)];
        let bytes = |state| HostAdam::new(&params, ParamPrecision { state, ..ParamPrecision::FP32 }).unwrap().state_bytes();
        let (f32_b, bf16_b, int8_b) = (bytes(StateFormat::F32), bytes(StateFormat::Bf16), bytes(StateFormat::Int8Block { block: 64 }));
        assert!(int8_b < bf16_b && bf16_b < f32_b, "f32 {f32_b}, bf16 {bf16_b}, int8 {int8_b}");
        let master = ParamPrecision { master: MasterPolicy::F32, ..ParamPrecision::FP32 };
        assert!(HostAdam::new(&params, master).err().unwrap().contains("master"));
    }
}
