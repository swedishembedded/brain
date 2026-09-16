// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A leaky integrate-and-fire population over a CSC connectome.

use gpu_core::{BufUsage, DeviceBuffer, Gpu};

use crate::csc::Csc;
use crate::seam::{DynamicalSystem, Port, State, StepStats};

/// Kernel indices into [`crate::KERNELS`]. The order is this crate's own;
/// `Gpu::step` takes the index the device was built with.
const K_GATHER: usize = 0;
const K_LIF: usize = 1;

/// Membrane dynamics, in the discretised form the kernel actually integrates.
///
/// `dt_over_tau` rather than `dt` and `tau` separately because that is the
/// only combination the update uses, and carrying both would let a caller set
/// a pair the kernel cannot honour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LifParams {
    /// `dt / tau_m`. Must be in `(0, 1]`: at `a > 1` the Euler update
    /// overshoots the resting potential and the membrane oscillates, which is
    /// a numerical artefact rather than a neuron.
    pub dt_over_tau: f32,
    pub v_rest: f32,
    pub v_reset: f32,
    pub v_th: f32,
    /// Input resistance: the scale from current to membrane volts.
    pub r: f32,
    /// Absolute refractory period, in ticks.
    pub refrac_ticks: u32,
}

impl Default for LifParams {
    /// Dimensionless defaults in the usual millivolt-shaped range, chosen so
    /// a unit-current input is comfortably suprathreshold.
    fn default() -> Self {
        LifParams { dt_over_tau: 0.1, v_rest: 0.0, v_reset: 0.0, v_th: 1.0, r: 1.0, refrac_ticks: 0 }
    }
}

impl LifParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.dt_over_tau > 0.0 && self.dt_over_tau <= 1.0) {
            return Err(format!("dt_over_tau must be in (0, 1], got {}", self.dt_over_tau));
        }
        if self.v_th <= self.v_reset {
            return Err(format!("v_th ({}) must exceed v_reset ({})", self.v_th, self.v_reset));
        }
        Ok(())
    }

    /// The exact discrete membrane trajectory for a CONSTANT input current,
    /// starting from `v0`, ignoring threshold.
    ///
    /// The Euler update `v <- v + a(v_rest - v + r*I)` is a geometric
    /// recurrence, so it has a closed form rather than merely an
    /// approximation:
    ///
    /// ```text
    /// v_k = v_inf + (v0 - v_inf)(1 - a)^k,   v_inf = v_rest + r*I
    /// ```
    ///
    /// This is the runtime's correctness oracle. `gradcheck` cannot gate a
    /// system whose learning rule is local rather than differentiated, so the
    /// substitute is an independently derived closed form for the forward
    /// dynamics -- and it exists only because the discretisation was chosen to
    /// have one.
    pub fn analytic_v(&self, v0: f32, current: f32, ticks: u32) -> f32 {
        let v_inf = self.v_rest + self.r * current;
        v_inf + (v0 - v_inf) * (1.0 - self.dt_over_tau).powi(ticks as i32)
    }
}

/// A spiking network: a connectome plus the state that makes it run.
pub struct SpikingNet {
    gpu: Gpu,
    n: u32,
    params: LifParams,
    tick: u64,

    // Graph (immutable for as long as the connectome is).
    indptr: DeviceBuffer,
    pre: DeviceBuffer,
    w: DeviceBuffer,

    // State.
    v: DeviceBuffer,
    refrac: DeviceBuffer,
    spike: DeviceBuffer,
    isyn: DeviceBuffer,
    drive: DeviceBuffer,

    // The connectome's own weights, kept so `reset` can restore them after
    // plasticity has moved them.
    w0: Vec<f32>,
}

impl SpikingNet {
    /// Build on an existing device. The caller owns device creation because
    /// this workspace allows ONE device per process (`Gpu::share` /
    /// `gpu_core::testgpu` for tests) and a crate that built its own would be
    /// the second one.
    pub fn new(gpu: Gpu, csc: &Csc, params: LifParams) -> Result<SpikingNet, String> {
        csc.validate()?;
        params.validate()?;
        let n = csc.n;
        if n == 0 {
            return Err("a connectome with no neurons cannot be stepped".to_string());
        }
        let bytes = |k: usize| (k.max(1) * 4) as u64;
        // The graph is written once and never read back; the state is both.
        let fixed = BufUsage::STORAGE | BufUsage::COPY_DST;
        let live = BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC;

        let indptr = gpu.buffer("neuro.indptr", bytes(csc.indptr.len()), fixed);
        let pre = gpu.buffer("neuro.pre", bytes(csc.pre.len()), fixed);
        let w = gpu.buffer("neuro.w", bytes(csc.w.len()), live);
        gpu.write(&indptr, &csc.indptr);
        gpu.write(&pre, &csc.pre);
        gpu.write_f32(&w, &csc.w);

        let nb = bytes(n as usize);
        let v = gpu.buffer("neuro.v", nb, live);
        let refrac = gpu.buffer("neuro.refrac", nb, live);
        let spike = gpu.buffer("neuro.spike", nb, live);
        let isyn = gpu.buffer("neuro.isyn", nb, live);
        let drive = gpu.buffer("neuro.drive", nb, live);

        let mut net = SpikingNet {
            gpu,
            n,
            params,
            tick: 0,
            indptr,
            pre,
            w,
            v,
            refrac,
            spike,
            isyn,
            drive,
            w0: csc.w.clone(),
        };
        net.reset(0);
        Ok(net)
    }

    pub fn neurons(&self) -> u32 {
        self.n
    }

    pub fn params(&self) -> LifParams {
        self.params
    }

    /// The LIF kernel's `Params` block: two counts then five f32 bit patterns,
    /// in declaration order. A mismatched param list here is silently wrong
    /// rather than a crash, which is why the order is written once.
    fn lif_params(&self) -> [u32; 7] {
        [
            self.n,
            self.params.refrac_ticks,
            self.params.dt_over_tau.to_bits(),
            self.params.v_rest.to_bits(),
            self.params.v_reset.to_bits(),
            self.params.v_th.to_bits(),
            self.params.r.to_bits(),
        ]
    }
}

impl DynamicalSystem for SpikingNet {
    fn reset(&mut self, _seed: u64) {
        // Deterministic and seed-independent today: every neuron starts at
        // rest with no spike history. The seed is in the signature because a
        // stochastic initialisation is the obvious next variant and callers
        // should already be passing one.
        let n = self.n as usize;
        self.gpu.write_f32(&self.v, &vec![self.params.v_rest; n]);
        self.gpu.write(&self.refrac, &vec![0u32; n]);
        self.gpu.write_f32(&self.spike, &vec![0.0; n]);
        self.gpu.write_f32(&self.isyn, &vec![0.0; n]);
        self.gpu.write_f32(&self.drive, &vec![0.0; n]);
        self.gpu.write_f32(&self.w, &self.w0.clone());
        self.tick = 0;
    }

    fn step(&mut self) -> StepStats {
        let gather = self.gpu.step(
            K_GATHER,
            &[&self.indptr, &self.pre, &self.w, &self.spike, &self.isyn],
            &[self.n],
            self.n * 64,
        );
        let lif = self.gpu.step(
            K_LIF,
            &[&self.v, &self.refrac, &self.isyn, &self.drive, &self.spike],
            &self.lif_params(),
            self.n,
        );
        // One submission, two steps: the gather reads last tick's spikes and
        // the LIF overwrites them, so they must not be reordered.
        self.gpu.submit(&[], &[gather, lif]);
        self.tick += 1;
        StepStats { tick: self.tick }
    }

    fn drive(&mut self, port: Port, values: &[f32]) -> Result<(), String> {
        if port != Port::Drive {
            return Err(format!("{port:?} is not writable"));
        }
        if values.len() != self.n as usize {
            return Err(format!("drive expects {} values, got {}", self.n, values.len()));
        }
        self.gpu.write_f32(&self.drive, values);
        Ok(())
    }

    fn read(&self, port: Port, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.n as usize {
            return Err(format!("read expects {} values, got {}", self.n, out.len()));
        }
        let buf = match port {
            Port::Spike => &self.spike,
            Port::Membrane => &self.v,
            Port::Current => &self.isyn,
            Port::Drive => &self.drive,
        };
        out.copy_from_slice(&self.gpu.read(buf, self.n as usize));
        Ok(())
    }

    fn port_len(&self, _port: Port) -> usize {
        self.n as usize
    }

    fn snapshot(&self) -> State {
        let n = self.n as usize;
        State {
            v: self.gpu.read(&self.v, n),
            // `Gpu::read` returns f32 words; a u32 buffer's values are those
            // words' BIT PATTERNS, not their numeric value (`model::paged`
            // reads its own u32 buffers the same way). `as u32` here would
            // silently round a refractory counter to garbage.
            refrac: self.gpu.read(&self.refrac, n).iter().map(|x| x.to_bits()).collect(),
            spike: self.gpu.read(&self.spike, n),
            w: self.gpu.read(&self.w, self.w0.len()),
            tick: self.tick,
        }
    }

    fn restore(&mut self, state: &State) -> Result<(), String> {
        let n = self.n as usize;
        if state.v.len() != n || state.refrac.len() != n || state.spike.len() != n {
            return Err(format!("snapshot is for {} neurons, this network has {n}", state.v.len()));
        }
        if !state.w.is_empty() && state.w.len() != self.w0.len() {
            return Err(format!("snapshot has {} weights, this connectome has {}", state.w.len(), self.w0.len()));
        }
        self.gpu.write_f32(&self.v, &state.v);
        self.gpu.write(&self.refrac, &state.refrac);
        self.gpu.write_f32(&self.spike, &state.spike);
        if !state.w.is_empty() {
            self.gpu.write_f32(&self.w, &state.w);
        }
        self.tick = state.tick;
        Ok(())
    }
}
