// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! PID event/effect data pipeline (port of the Python module's plant, oracle,
//! CBOR tokenizer, DAgger trajectory generator, dataset/windowing, and the
//! closed-loop rollout). The control loop is scalar CPU code; only the
//! transformer runs on the GPU.

use std::collections::HashMap;

use crate::model::{PidConfig, BOS, DECIDE, EV_END, EV_START, FX_END, FX_START, IGNORE, PAD, U_BINS};
// `Pid` (the decoder) is only used by the native blocking closed-loop rollout.
#[cfg(not(target_arch = "wasm32"))]
use crate::model::Pid;

pub const DT: f32 = 0.05;
const VALUE_BINS: i64 = 101;
const VALUE_MIN: f32 = -1.5;
const VALUE_MAX: f32 = 1.5;
const WN: f32 = 3.5;
const ZETA: f32 = 1.0;
const EV_PLANT_STATE: u32 = 1;
const FX_SET_ACTUATOR: u32 = 1;

// ---- quantization (matches Python quantize_range/dequantize_range) ----
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}
fn quantize_range(x: f32, lo: f32, hi: f32, bins: i64) -> i64 {
    let x = clamp(x, lo, hi);
    ((x - lo) / (hi - lo) * (bins - 1) as f32).round() as i64
}
fn dequantize_range(q: i64, lo: f32, hi: f32, bins: i64) -> f32 {
    let q = q.clamp(0, bins - 1);
    lo + (hi - lo) * (q as f32 / (bins - 1) as f32)
}
pub fn quantize_u(u: f32) -> u32 {
    quantize_range(u, -1.0, 1.0, U_BINS as i64) as u32
}
pub fn dequantize_u(q: u32) -> f32 {
    dequantize_range(q as i64, -1.0, 1.0, U_BINS as i64)
}
fn quantize_value(x: f32) -> i64 {
    quantize_range(x, VALUE_MIN, VALUE_MAX, VALUE_BINS)
}

// ---- CBOR (matches the Python fallback for small non-negative ints/arrays) ----
fn cbor_uint(n: i64, out: &mut Vec<u32>) {
    let n = n as u64;
    if n < 24 {
        out.push(n as u32);
    } else if n <= 0xFF {
        out.push(0x18);
        out.push(n as u32);
    } else {
        out.push(0x19);
        out.push((n >> 8) as u32);
        out.push((n & 0xFF) as u32);
    }
}
fn cbor_array(items: &[i64]) -> Vec<u32> {
    let mut out = Vec::new();
    out.push(0x80 | items.len() as u32); // len < 24 for our schemas
    for &v in items {
        cbor_uint(v, &mut out);
    }
    out
}

pub fn encode_event(setpoint: f32, y: f32, error: f32) -> Vec<u32> {
    let obj = [
        EV_PLANT_STATE as i64,
        quantize_value(setpoint),
        quantize_value(y),
        quantize_value(error),
    ];
    let mut v = vec![EV_START];
    v.extend(cbor_array(&obj));
    v.push(EV_END);
    v
}
pub fn encode_effect_bin(u_bin: u32) -> Vec<u32> {
    let obj = [FX_SET_ACTUATOR as i64, u_bin as i64];
    let mut v = vec![FX_START];
    v.extend(cbor_array(&obj));
    v.push(FX_END);
    v
}

// ---- plant + controllers ----
#[derive(Clone, Copy)]
pub struct PlantSpec {
    pub tau: f32,
    pub gain: f32,
    pub disturbance: f32,
}

pub struct Plant {
    pub y: f32,
    spec: PlantSpec,
}
impl Plant {
    pub fn new(spec: PlantSpec) -> Plant {
        Plant { y: 0.0, spec }
    }
    pub fn step(&mut self, u: f32) -> f32 {
        let u = clamp(u, -1.0, 1.0);
        self.y += DT * (-self.y + self.spec.gain * u + self.spec.disturbance) / self.spec.tau;
        self.y
    }
}

pub fn training_plants() -> Vec<PlantSpec> {
    let taus = [0.45, 0.65, 0.85];
    let gains = [1.00, 1.25, 1.50];
    let mut v = Vec::new();
    for &t in &taus {
        for &g in &gains {
            v.push(PlantSpec { tau: t, gain: g, disturbance: 0.0 });
        }
    }
    v
}
pub fn validation_plants() -> Vec<PlantSpec> {
    let taus = [0.55, 0.75];
    let gains = [1.125, 1.375];
    let mut v = Vec::new();
    for &t in &taus {
        for &g in &gains {
            v.push(PlantSpec { tau: t, gain: g, disturbance: 0.0 });
        }
    }
    v
}
pub fn pole_place_pi(spec: &PlantSpec) -> (f32, f32) {
    let kp = (2.0 * ZETA * WN * spec.tau - 1.0) / spec.gain;
    let ki = spec.tau * WN * WN / spec.gain;
    (kp, ki)
}
pub fn velocity_pi_bin(kp: f32, ki: f32, u_prev: f32, error: f32, prev_error: f32) -> u32 {
    let du = kp * (error - prev_error) + ki * DT * error;
    quantize_u(clamp(u_prev + du, -1.0, 1.0))
}

// ---- RNG (xorshift, matches train.rs style) ----
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }
    fn gauss(&mut self, std: f32) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
    }
}

/// One trajectory step: the encoded event, the applied-effect tokens fed back,
/// and the expert label bin for that (real, possibly off-policy) state.
pub struct TrajStep {
    pub event: Vec<u32>,
    pub effect: Vec<u32>,
    pub expert_bin: u32,
}

/// DAgger trajectory: an exploration policy drives the plant; each state is
/// relabeled with the perfect per-plant velocity-PI action.
pub fn generate_trajectory(spec: &PlantSpec, length: usize, rng: &mut Rng) -> Vec<TrajStep> {
    let mut plant = Plant::new(*spec);
    plant.y = rng.range(-0.6, 0.6);
    let (kp, ki) = pole_place_pi(spec);
    let mut sp = rng.range(-1.0, 1.0);
    let mut u_prev = 0.0f32;
    let mut prev_error = 0.0f32;
    let mut traj = Vec::with_capacity(length);
    for t in 0..length {
        if t == 0 || rng.uniform() < 0.10 {
            sp = rng.range(-1.0, 1.0);
        }
        let measured = plant.y;
        let error = sp - measured;
        let expert_bin = velocity_pi_bin(kp, ki, u_prev, error, prev_error);
        let u_expert = dequantize_u(expert_bin);
        let r = rng.uniform();
        let u_applied = if r < 0.3 {
            clamp(u_expert + rng.gauss(0.25), -1.0, 1.0)
        } else if r < 0.38 {
            rng.range(-1.0, 1.0)
        } else {
            u_expert
        };
        let u_applied = dequantize_u(quantize_u(u_applied));
        traj.push(TrajStep {
            event: encode_event(sp, measured, error),
            effect: encode_effect_bin(quantize_u(u_applied)),
            expert_bin,
        });
        prev_error = error;
        u_prev = u_applied;
        plant.step(u_applied);
    }
    traj
}

/// Build one training window (tokens, labels) of length `seq_steps`, padded to
/// `t_pad`. Mirrors EventEffectDataset.__getitem__ incl. the cold-start mask.
fn build_window(traj: &[TrajStep], start: usize, seq_steps: usize, t_pad: usize) -> (Vec<u32>, Vec<u32>) {
    let mut tokens = vec![BOS];
    let mut labels = vec![IGNORE];
    for offset in 0..seq_steps {
        let s = &traj[start + offset];
        tokens.extend_from_slice(&s.event);
        labels.extend(std::iter::repeat(IGNORE).take(s.event.len()));
        tokens.push(DECIDE);
        if offset == 0 && start != 0 {
            labels.push(IGNORE);
        } else {
            labels.push(s.expert_bin);
        }
        tokens.extend_from_slice(&s.effect);
        labels.extend(std::iter::repeat(IGNORE).take(s.effect.len()));
    }
    assert!(tokens.len() <= t_pad, "window {} exceeds T={}", tokens.len(), t_pad);
    while tokens.len() < t_pad {
        tokens.push(PAD);
        labels.push(IGNORE);
    }
    (tokens, labels)
}

/// A flat corpus of trajectories spread across the training plants.
pub struct PidDataset {
    pub trajectories: Vec<Vec<TrajStep>>,
    pub traj_len: usize,
    pub seq_steps: usize,
    pub t_pad: usize,
}
impl PidDataset {
    pub fn new(plants: &[PlantSpec], n_traj: usize, traj_len: usize, seq_steps: usize, t_pad: usize, seed: u64) -> PidDataset {
        let mut rng = Rng::new(seed);
        let trajectories = (0..n_traj)
            .map(|i| generate_trajectory(&plants[i % plants.len()], traj_len, &mut rng))
            .collect();
        PidDataset { trajectories, traj_len, seq_steps, t_pad }
    }
    /// Assemble a batch of `b` random windows -> (tokens, labels), each length t_pad.
    pub fn batch(&self, b: usize, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        let max_start = (self.traj_len - self.seq_steps - 1).max(1);
        let mut xs = Vec::with_capacity(b * self.t_pad);
        let mut ys = Vec::with_capacity(b * self.t_pad);
        for _ in 0..b {
            let ti = (rng.next() as usize) % self.trajectories.len();
            let start = (rng.next() as usize) % max_start;
            let (tok, lab) = build_window(&self.trajectories[ti], start, self.seq_steps, self.t_pad);
            xs.extend_from_slice(&tok);
            ys.extend_from_slice(&lab);
        }
        (xs, ys)
    }
}

// ---- evaluation / rollout ----
pub fn eval_step_schedule(t: usize) -> f32 {
    let table = [(0usize, 0.70f32), (45, -0.60), (90, 0.40), (135, -0.85)];
    let mut sp = table[0].1;
    for &(ts, val) in &table {
        if t >= ts {
            sp = val;
        }
    }
    sp
}

// Used only by `rollout_on_plant`, which is native-only.
#[cfg(not(target_arch = "wasm32"))]
fn steady_state_error(rows: &[(usize, f32, f32)]) -> f32 {
    // rows: (t, setpoint, y); mean |sp-y| over the last 10 of each 45-step hold.
    let mut errs = 0.0;
    let mut n = 0;
    for hold in [0usize, 45, 90, 135] {
        for &(t, sp, y) in rows {
            if t >= hold + 35 && t < hold + 45 {
                errs += (sp - y).abs();
                n += 1;
            }
        }
    }
    errs / (n.max(1) as f32)
}

pub fn run_oracle_closed_loop(spec: &PlantSpec, steps: usize) -> Vec<(usize, f32, f32)> {
    let mut plant = Plant::new(*spec);
    let (kp, ki) = pole_place_pi(spec);
    let (mut u_prev, mut prev_error) = (0.0f32, 0.0f32);
    let mut rows = Vec::new();
    for t in 0..steps {
        let sp = eval_step_schedule(t);
        let measured = plant.y;
        let error = sp - measured;
        let u = dequantize_u(velocity_pi_bin(kp, ki, u_prev, error, prev_error));
        plant.step(u);
        rows.push((t, sp, measured));
        prev_error = error;
        u_prev = u;
    }
    rows
}

/// Closed-loop rollout with the model controlling the plant (no PID). Returns
/// (rows, mse, ss_err). `dec` must be a Pid built with b=1, t>=block_size.
/// Native only: drives the model with the blocking `logits_last` readback. The
/// browser build performs single inferences via `Pid::logits_last_async`.
#[cfg(not(target_arch = "wasm32"))]
pub fn rollout_on_plant(dec: &Pid, cfg: &PidConfig, spec: &PlantSpec, steps: usize) -> (f32, f32) {
    let mut plant = Plant::new(*spec);
    let mut context: Vec<u32> = vec![BOS];
    let block = cfg.block_size as usize;
    let mut rows = Vec::new();
    for t in 0..steps {
        let sp = eval_step_schedule(t);
        let measured = plant.y;
        let error = sp - measured;
        context.extend(encode_event(sp, measured, error));
        context.push(DECIDE);
        let window_start = context.len().saturating_sub(block);
        let logits = dec.logits_last(&context[window_start..]);
        let u_bin = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        context.extend(encode_effect_bin(u_bin));
        plant.step(dequantize_u(u_bin));
        rows.push((t, sp, measured));
        if context.len() > block {
            let mut nc = vec![BOS];
            nc.extend_from_slice(&context[context.len() - (block - 1)..]);
            context = nc;
        }
    }
    let mse = rows.iter().map(|&(_, sp, y)| (sp - y).powi(2)).sum::<f32>() / rows.len().max(1) as f32;
    (mse, steady_state_error(&rows))
}

// ---- from-scratch init (Rust training; PyTorch-trained nets load from file) ----
pub fn init_weights(cfg: &PidConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut map = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let vals: Vec<f32> = if name.ends_with("ln1.weight")
            || name.ends_with("ln2.weight")
            || name == "ln.weight"
        {
            vec![1.0; numel]
        } else if name.ends_with(".bias") || name.ends_with("ln1.bias") || name == "ln.bias" {
            vec![0.0; numel]
        } else {
            (0..numel).map(|_| 0.02 * rng.gauss(1.0)).collect()
        };
        map.insert(name, vals);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_value_center_and_ends() {
        assert_eq!(quantize_value(0.0), 50); // midpoint of [-1.5,1.5], 101 bins
        assert_eq!(quantize_value(-1.5), 0);
        assert_eq!(quantize_value(1.5), 100);
        assert_eq!(quantize_value(-10.0), 0); // clamped
        assert_eq!(quantize_value(10.0), 100);
    }

    #[test]
    fn quantize_u_roundtrip_and_known() {
        assert_eq!(quantize_u(0.0), 40); // midpoint of [-1,1], 81 bins
        assert_eq!(quantize_u(-1.0), 0);
        assert_eq!(quantize_u(1.0), 80);
        assert!((dequantize_u(0) + 1.0).abs() < 1e-6);
        assert!((dequantize_u(80) - 1.0).abs() < 1e-6);
        assert!(dequantize_u(40).abs() < 1e-6);
        // round-trip is idempotent on the bin grid
        for b in 0..U_BINS {
            assert_eq!(quantize_u(dequantize_u(b)), b);
        }
    }

    #[test]
    fn cbor_uint_encoding() {
        let enc = |n: i64| {
            let mut v = Vec::new();
            cbor_uint(n, &mut v);
            v
        };
        assert_eq!(enc(0), vec![0]);
        assert_eq!(enc(23), vec![23]);
        assert_eq!(enc(24), vec![0x18, 24]);
        assert_eq!(enc(100), vec![0x18, 100]);
        assert_eq!(enc(255), vec![0x18, 255]);
        assert_eq!(enc(256), vec![0x19, 1, 0]);
    }

    #[test]
    fn cbor_array_and_event_effect_framing() {
        assert_eq!(cbor_array(&[1, 50]), vec![0x82, 1, 0x18, 50]);
        let ev = encode_event(0.0, 0.0, 0.0);
        assert_eq!(*ev.first().unwrap(), EV_START);
        assert_eq!(*ev.last().unwrap(), EV_END);
        assert_eq!(ev[1], 0x84); // 4-element array
        // [type=1, sp=50, y=50, err=50] -> 1, 0x18,50, 0x18,50, 0x18,50
        assert_eq!(&ev[2..], &[1, 0x18, 50, 0x18, 50, 0x18, 50, EV_END]);
        let fx = encode_effect_bin(40);
        assert_eq!(fx, vec![FX_START, 0x82, 1, 0x18, 40, FX_END]);
        let fx0 = encode_effect_bin(5); // small u_bin -> single-byte int
        assert_eq!(fx0, vec![FX_START, 0x82, 1, 5, FX_END]);
    }

    #[test]
    fn pole_placement_matches_formula() {
        let s = PlantSpec { tau: 0.65, gain: 1.0, disturbance: 0.0 };
        let (kp, ki) = pole_place_pi(&s);
        assert!((kp - (2.0 * 1.0 * 3.5 * 0.65 - 1.0)).abs() < 1e-5);
        assert!((ki - 0.65 * 3.5 * 3.5).abs() < 1e-5);
    }

    #[test]
    fn velocity_pi_zero_state_is_neutral() {
        let s = PlantSpec { tau: 0.65, gain: 1.0, disturbance: 0.0 };
        let (kp, ki) = pole_place_pi(&s);
        assert_eq!(velocity_pi_bin(kp, ki, 0.0, 0.0, 0.0), quantize_u(0.0));
        // positive error increases u
        assert!(velocity_pi_bin(kp, ki, 0.0, 0.5, 0.0) > quantize_u(0.0));
    }

    #[test]
    fn plant_first_order_step() {
        let mut p = Plant::new(PlantSpec { tau: 0.5, gain: 2.0, disturbance: 0.0 });
        let y1 = p.step(1.0);
        // dy = (-0 + 2*1)/0.5 = 4; y = 0 + 0.05*4 = 0.2
        assert!((y1 - 0.2).abs() < 1e-6);
        // saturates the actuator
        let mut q = Plant::new(PlantSpec { tau: 0.5, gain: 2.0, disturbance: 0.0 });
        assert!((q.step(5.0) - 0.2).abs() < 1e-6); // u clamped to 1
    }

    #[test]
    fn eval_schedule_holds() {
        assert_eq!(eval_step_schedule(0), 0.70);
        assert_eq!(eval_step_schedule(44), 0.70);
        assert_eq!(eval_step_schedule(45), -0.60);
        assert_eq!(eval_step_schedule(90), 0.40);
        assert_eq!(eval_step_schedule(135), -0.85);
        assert_eq!(eval_step_schedule(179), -0.85);
    }

    #[test]
    fn plant_grids_disjoint_and_interpolated() {
        let tr = training_plants();
        let va = validation_plants();
        assert_eq!(tr.len(), 9);
        assert_eq!(va.len(), 4);
        // every validation plant lies strictly between training nodes (no overlap)
        for v in &va {
            assert!(!tr.iter().any(|t| (t.tau - v.tau).abs() < 1e-6 && (t.gain - v.gain).abs() < 1e-6));
            assert!(v.tau > 0.45 && v.tau < 0.85);
            assert!(v.gain > 1.0 && v.gain < 1.5);
        }
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
        let mut c = Rng::new(7);
        for _ in 0..1000 {
            let u = c.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn window_masks_cold_start_only_when_midstream() {
        let mut rng = Rng::new(1);
        let traj = generate_trajectory(&training_plants()[0], 30, &mut rng);
        // start==0 window: the first DECIDE IS supervised
        let (tok0, lab0) = build_window(&traj, 0, 3, 256);
        let first_decide = tok0.iter().position(|&t| t == DECIDE).unwrap();
        assert_ne!(lab0[first_decide], IGNORE);
        // start>0 window: the first DECIDE is masked (no in-window predecessor)
        let (tok1, lab1) = build_window(&traj, 5, 3, 256);
        let fd1 = tok1.iter().position(|&t| t == DECIDE).unwrap();
        assert_eq!(lab1[fd1], IGNORE);
        // both windows padded to T with PAD/IGNORE
        assert_eq!(tok0.len(), 256);
        assert!(tok0.iter().rev().take(1).all(|&t| t == PAD));
        assert_eq!(lab0.len(), 256);
    }

    #[test]
    fn dataset_batch_shape() {
        let plants = training_plants();
        let ds = PidDataset::new(&plants, 9, 40, 6, 128, 5);
        let mut rng = Rng::new(9);
        let (xs, ys) = ds.batch(4, &mut rng);
        assert_eq!(xs.len(), 4 * 128);
        assert_eq!(ys.len(), 4 * 128);
        // at least one supervised label and one PAD present
        assert!(ys.iter().any(|&v| v != IGNORE));
        assert!(xs.iter().any(|&v| v == PAD));
    }
}
