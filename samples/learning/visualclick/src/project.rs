// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The trainable layer between the screen and the decision head - and the
//! reason this sample answers correctly.
//!
//! `Decide`'s head scores option `i` as
//!
//! ```text
//! ctx_i   = crossattn(q_i -> state rows)
//! score_i = LN(ctx_i + q_i) . w + b
//! ```
//!
//! so BOTH sides of that attention have to carry something the head can
//! select on. The query `q_i` is the part that decides which option is being
//! scored, and it is the part worth being careful about: a measurement on
//! this repository's own MiniLM checkpoint found that feeding a frozen
//! sentence encoder's `[CLS]` in as the query directly leaves the options
//! nearly indistinguishable. Scoring sixteen options whose slot text differed
//! only in a name, the sixteen queries sat at **cosine 0.908** to each other,
//! and the attribute the instruction actually asks about moved the vector even
//! less - **0.936** between two whole different instructions. A
//! cross-attention asked to pick one of sixteen rows using queries that are
//! 91% the same vector has almost nothing to work with, and no amount of
//! training the head alone recovers it, because the head's query projection
//! sees only what the frozen encoder already collapsed.
//!
//! The fix is a trainable projection on each side, which is what this module
//! is:
//!
//! ```text
//! state row i = LN( Wstate . control_i )                    <- what is on screen
//! slot row  i = LN( s(cls) * Wopt . control_i + b(cls) )    <- which control, and what was asked
//!   where     s(cls) = 1 + 2 tanh( Wscale . cls )
//! ```
//!
//! The instruction conditions the query MULTIPLICATIVELY as well as
//! additively. With a purely additive `Wopt . control_i + Winstr . cls`, the
//! instruction contributes the same vector to every option, so the weight the
//! query puts on any feature of the control - its position, say - is the same
//! whatever was asked. An instruction that has to REVERSE how a feature is
//! read, rather than just shift it, has no way to say so. A per-channel scale
//! (FiLM-style conditioning) supplies exactly that degree of freedom, and both
//! the bound on `s` and its zero initialisation are load bearing - see
//! `gamma_lin` for what an unbounded one cost.
//!
//! `b` is the load-bearing offset. The instruction's colour/shape/state
//! content survives in the frozen encoder's output as a small residual on a
//! large shared direction; a learned linear map can project that residual out
//! and scale it up, which is exactly what a fixed identity path cannot do.
//! `Wopt` gives every option a distinct, learned identity instead of a
//! sentence-encoding of a name, and `Wstate` turns a control's appearance into
//! something the query can match against.
//!
//! The STATE rows are deliberately not modulated: they are what is on the
//! screen, which does not change with what was asked about it.
//!
//! Everything downstream - the cross-attention, its optimizer, the softmax,
//! the loss - is `crates/decide`'s existing kept-features path, unmodified.
//! The rows go in through `Features::from_parts` and the gradient comes back
//! out of the encoder's own seed buffer.

use crate::screen::Rng;

const LN_EPS: f32 = 1e-5;

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    lr: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, lr }
    }

    fn step(&mut self, params: &mut [f32], grad: &mut [f32]) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            params[i] -= self.lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + eps);
            grad[i] = 0.0;
        }
    }
}

/// `y = W x + b`, row-major `W` as `[out, in]`.
struct Linear {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    adam_w: Adam,
    adam_b: Adam,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, rng: &mut Rng, lr: f32) -> Linear {
        // Fan-in scaled, though the LayerNorm downstream makes the exact
        // scale far less critical than the symmetry breaking.
        let scale = (1.0 / in_dim as f32).sqrt();
        let w = (0..out_dim * in_dim).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect();
        Linear {
            in_dim,
            out_dim,
            w,
            b: vec![0.0; out_dim],
            gw: vec![0.0; out_dim * in_dim],
            gb: vec![0.0; out_dim],
            adam_w: Adam::new(out_dim * in_dim, lr),
            adam_b: Adam::new(out_dim, lr),
        }
    }

    fn forward_into(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        for o in 0..self.out_dim {
            let row = o * self.in_dim;
            let mut acc = self.b[o];
            for i in 0..self.in_dim {
                acc += self.w[row + i] * x[i];
            }
            out[o] = acc;
        }
    }

    /// Accumulate parameter gradients. The input gradient is not produced:
    /// nothing upstream of these projections is trainable (the pixels are
    /// pixels, the encoder is frozen), so computing it would be work whose
    /// result is discarded.
    fn accumulate(&mut self, x: &[f32], dy: &[f32]) {
        debug_assert_eq!(dy.len(), self.out_dim);
        for o in 0..self.out_dim {
            let row = o * self.in_dim;
            self.gb[o] += dy[o];
            for i in 0..self.in_dim {
                self.gw[row + i] += dy[o] * x[i];
            }
        }
    }

    fn step(&mut self) {
        self.adam_w.step(&mut self.w, &mut self.gw);
        self.adam_b.step(&mut self.b, &mut self.gb);
    }
}

/// Every row the head reads is LayerNorm'd by the encoder that normally
/// produces it (`crates/decide/src/model.rs` is post-LayerNorm throughout), so
/// a row spliced in from outside is normalised the same way rather than
/// arriving at whatever scale a linear layer happened to emit.
struct LayerNorm {
    dim: usize,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    g_gamma: Vec<f32>,
    g_beta: Vec<f32>,
    adam_gamma: Adam,
    adam_beta: Adam,
}

impl LayerNorm {
    fn new(dim: usize, lr: f32) -> LayerNorm {
        LayerNorm {
            dim,
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            g_gamma: vec![0.0; dim],
            g_beta: vec![0.0; dim],
            adam_gamma: Adam::new(dim, lr),
            adam_beta: Adam::new(dim, lr),
        }
    }

    /// Writes the normalised output and returns `(xhat, inv_std)` for the
    /// backward pass.
    fn forward_into(&self, x: &[f32], out: &mut [f32]) -> (Vec<f32>, f32) {
        let n = self.dim as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + LN_EPS).sqrt();
        let mut xhat = vec![0.0f32; self.dim];
        for h in 0..self.dim {
            xhat[h] = (x[h] - mean) * inv_std;
            out[h] = self.gamma[h] * xhat[h] + self.beta[h];
        }
        (xhat, inv_std)
    }

    /// Accumulates `gamma`/`beta` gradients and writes `dL/dx` into `dx`.
    fn backward(&mut self, xhat: &[f32], inv_std: f32, dy: &[f32], dx: &mut [f32]) {
        let n = self.dim as f32;
        let mut dxhat = vec![0.0f32; self.dim];
        for h in 0..self.dim {
            self.g_beta[h] += dy[h];
            self.g_gamma[h] += dy[h] * xhat[h];
            dxhat[h] = dy[h] * self.gamma[h];
        }
        let sum_dxhat: f32 = dxhat.iter().sum();
        let sum_dxhat_xhat: f32 = dxhat.iter().zip(xhat).map(|(a, b)| a * b).sum();
        for h in 0..self.dim {
            dx[h] = inv_std / n * (n * dxhat[h] - sum_dxhat - xhat[h] * sum_dxhat_xhat);
        }
    }

    fn step(&mut self) {
        self.adam_gamma.step(&mut self.gamma, &mut self.g_gamma);
        self.adam_beta.step(&mut self.beta, &mut self.g_beta);
    }
}

/// What `forward` cached for `backward`. Kept as one struct so the two can
/// only ever be called in the right order with the right shapes - the pairing
/// `crates/decide`'s own forward/backward already relies on.
#[derive(Default)]
struct Cache {
    n: usize,
    state_xhat: Vec<f32>,
    state_inv: Vec<f32>,
    slot_xhat: Vec<f32>,
    slot_inv: Vec<f32>,
    /// `Wopt . control_i` before the instruction modulates it - what the
    /// scale's own gradient is measured against.
    opt_out: Vec<f32>,
    /// The per-channel scale this decision's instruction produced, and the
    /// derivative of that scale with respect to what `gamma_lin` emitted.
    scale: Vec<f32>,
    d_scale: Vec<f32>,
}

pub struct Grounder {
    d_model: usize,
    /// Where the appearance block ends and the position block begins.
    split: usize,
    /// Position gets its OWN projection rather than sharing one with
    /// appearance, and that is what makes a purely positional reference
    /// answerable.
    ///
    /// One projection over a concatenated feature vector spreads its output
    /// across the dimensions it is given, so 26 position dimensions against
    /// 192 of colour and shape arrive at roughly a tenth the amplitude however
    /// wide the position block is made. Measured, that is exactly the shape of
    /// the failure: regions a control's APPEARANCE also predicts were answered
    /// (in the toolbar 91.7%, in the top row 96.4% - both correlate with a
    /// control's size and the panel behind it) while the two regions that are
    /// nothing but an x coordinate were not (on the left 25.0%, on the right
    /// 47.4%). Two projections summed give each block its own full-rank map
    /// into `d_model`, so their relative weight is something the optimizer
    /// sets rather than something the input layout fixes in advance.
    state_pos: Linear,
    state_lin: Linear,
    state_ln: LayerNorm,
    opt_lin: Linear,
    opt_pos: Linear,
    /// The instruction's per-channel SCALE on the option projection, as
    /// `1 + 2 tanh(gamma_lin(cls))`.
    ///
    /// Bounded, and that is not decoration. An unbounded `1 + gamma_lin(cls)`
    /// was measured: it starts identical to the additive layer and still lost
    /// ten points overall (84.4% -> 74.2%) and fourteen on attribute
    /// instructions, with single-example losses spiking above chance
    /// mid-run - a multiplicative scale free to grow rescales the query by
    /// whatever the last few steps pushed it to, and the head is re-fitting a
    /// moving target. `tanh` keeps the scale in `(-1, 3)`: wide enough to
    /// INVERT a channel, which is the whole point, and narrow enough that it
    /// cannot run away. Zero-initialised, so the scale is exactly 1 on the
    /// first step and training starts from the additive layer it replaces.
    gamma_lin: Linear,
    /// The instruction's per-channel SHIFT.
    beta_lin: Linear,
    slot_ln: LayerNorm,
    cache: Cache,
}

impl Grounder {
    /// `split` is where the appearance block ends and the position block
    /// begins - see [`Grounder::state_pos`] for why the two are projected
    /// separately.
    pub fn new(feat_dim: usize, split: usize, d_model: usize, seed: u64, lr: f32) -> Grounder {
        assert!(split > 0 && split < feat_dim, "the feature vector must hold a position block after the appearance block");
        let mut rng = Rng::new(seed);
        let mut gamma_lin = Linear::new(d_model, d_model, &mut rng, lr);
        // Starts as the identity modulation: every scale is exactly 1 on the
        // first step, so the layer is initially the additive version this
        // replaced and cannot be worse than it at init.
        gamma_lin.w.iter_mut().for_each(|v| *v = 0.0);
        Grounder {
            d_model,
            split,
            state_lin: Linear::new(split, d_model, &mut rng, lr),
            state_pos: Linear::new(feat_dim - split, d_model, &mut rng, lr),
            state_ln: LayerNorm::new(d_model, lr),
            opt_lin: Linear::new(split, d_model, &mut rng, lr),
            opt_pos: Linear::new(feat_dim - split, d_model, &mut rng, lr),
            gamma_lin,
            beta_lin: Linear::new(d_model, d_model, &mut rng, lr),
            slot_ln: LayerNorm::new(d_model, lr),
            cache: Cache::default(),
        }
    }

    /// `[state rows; slot rows]`, flattened - exactly the layout
    /// `Features::from_parts` reads, with one state row and one slot row per
    /// control on screen.
    pub fn forward(&mut self, feats: &[Vec<f32>], instr_cls: &[f32]) -> Vec<f32> {
        let n = feats.len();
        let d = self.d_model;
        assert_eq!(instr_cls.len(), d, "the instruction row must be one encoder row wide");

        let mut rows = vec![0.0f32; 2 * n * d];
        let mut c = Cache {
            n,
            state_xhat: vec![0.0; n * d],
            state_inv: vec![0.0; n],
            slot_xhat: vec![0.0; n * d],
            slot_inv: vec![0.0; n],
            opt_out: vec![0.0; n * d],
            scale: vec![0.0; d],
            d_scale: vec![0.0; d],
        };

        // The instruction is projected ONCE and shared by every option's
        // query - one encoder pass and two projections per decision, not one
        // per option.
        let mut beta = vec![0.0f32; d];
        let mut raw = vec![0.0f32; d];
        self.gamma_lin.forward_into(instr_cls, &mut raw);
        self.beta_lin.forward_into(instr_cls, &mut beta);
        for h in 0..d {
            let t = raw[h].tanh();
            c.scale[h] = 1.0 + 2.0 * t;
            c.d_scale[h] = 2.0 * (1.0 - t * t);
        }

        let mut pre = vec![0.0f32; d];
        let mut opt = vec![0.0f32; d];
        let mut posbuf = vec![0.0f32; d];
        for i in 0..n {
            assert_eq!(feats[i].len(), self.split + self.state_pos.in_dim, "control {i} has the wrong feature width");
            let (look, at) = feats[i].split_at(self.split);

            self.state_lin.forward_into(look, &mut pre);
            self.state_pos.forward_into(at, &mut posbuf);
            for h in 0..d {
                pre[h] += posbuf[h];
            }
            let (xhat, inv) = self.state_ln.forward_into(&pre, &mut rows[i * d..(i + 1) * d]);
            c.state_xhat[i * d..(i + 1) * d].copy_from_slice(&xhat);
            c.state_inv[i] = inv;

            self.opt_lin.forward_into(look, &mut opt);
            self.opt_pos.forward_into(at, &mut posbuf);
            for h in 0..d {
                opt[h] += posbuf[h];
            }
            c.opt_out[i * d..(i + 1) * d].copy_from_slice(&opt);
            for h in 0..d {
                pre[h] = c.scale[h] * opt[h] + beta[h];
            }
            let at = (n + i) * d;
            let (xhat, inv) = self.slot_ln.forward_into(&pre, &mut rows[at..at + d]);
            c.slot_xhat[i * d..(i + 1) * d].copy_from_slice(&xhat);
            c.slot_inv[i] = inv;
        }

        self.cache = c;
        debug_assert!(rows.iter().all(|v| v.is_finite()), "a projected row is not finite");
        rows
    }

    /// `d_rows` is `dL/d(forward's output)`, same layout - the head's own
    /// gradient on the rows it was given, read back out of the encoder's seed
    /// buffer.
    pub fn backward(&mut self, feats: &[Vec<f32>], instr_cls: &[f32], d_rows: &[f32]) {
        let (n, d) = (self.cache.n, self.d_model);
        assert_eq!(feats.len(), n, "backward must see the same controls forward did");
        assert_eq!(d_rows.len(), 2 * n * d, "one gradient row per row produced");

        let mut dpre = vec![0.0f32; d];
        let mut d_opt = vec![0.0f32; d];
        // The scale and shift are shared across every option on this screen,
        // so their gradients are SUMS over options - accumulated here and
        // applied to their producing projections once.
        let mut d_gamma = vec![0.0f32; d];
        let mut d_beta = vec![0.0f32; d];

        for i in 0..n {
            let sx = &self.cache.state_xhat[i * d..(i + 1) * d];
            let (look, at) = feats[i].split_at(self.split);
            self.state_ln.backward(sx, self.cache.state_inv[i], &d_rows[i * d..(i + 1) * d], &mut dpre);
            // Summed, so each projection sees the same upstream gradient.
            self.state_lin.accumulate(look, &dpre);
            self.state_pos.accumulate(at, &dpre);

            let ox = &self.cache.slot_xhat[i * d..(i + 1) * d];
            let slot = (n + i) * d;
            self.slot_ln.backward(ox, self.cache.slot_inv[i], &d_rows[slot..slot + d], &mut dpre);

            // pre = scale * opt + beta, so the scale's gradient is measured
            // against what it scaled and the option projection's is
            // attenuated by the scale it was multiplied by.
            let opt = &self.cache.opt_out[i * d..(i + 1) * d];
            for h in 0..d {
                d_opt[h] = dpre[h] * self.cache.scale[h];
                // Through the tanh that bounds the scale.
                d_gamma[h] += dpre[h] * opt[h] * self.cache.d_scale[h];
                d_beta[h] += dpre[h];
            }
            self.opt_lin.accumulate(look, &d_opt);
            self.opt_pos.accumulate(at, &d_opt);
        }

        self.gamma_lin.accumulate(instr_cls, &d_gamma);
        self.beta_lin.accumulate(instr_cls, &d_beta);
    }

    pub fn step(&mut self) {
        self.state_lin.step();
        self.state_pos.step();
        self.state_ln.step();
        self.opt_lin.step();
        self.opt_pos.step();
        self.gamma_lin.step();
        self.beta_lin.step();
        self.slot_ln.step();
    }

    /// Parameter count - what this sample actually trains, next to the 22M
    /// frozen encoder it reads through.
    pub fn parameters(&self) -> usize {
        let lin = |l: &Linear| l.w.len() + l.b.len();
        lin(&self.state_lin) + lin(&self.state_pos) + lin(&self.opt_lin) + lin(&self.opt_pos) + lin(&self.gamma_lin) + lin(&self.beta_lin) + 4 * self.d_model
    }

    fn arrays(&self) -> Vec<&Vec<f32>> {
        vec![
            &self.state_lin.w,
            &self.state_lin.b,
            &self.state_pos.w,
            &self.state_pos.b,
            &self.state_ln.gamma,
            &self.state_ln.beta,
            &self.opt_lin.w,
            &self.opt_lin.b,
            &self.opt_pos.w,
            &self.opt_pos.b,
            &self.gamma_lin.w,
            &self.gamma_lin.b,
            &self.beta_lin.w,
            &self.beta_lin.b,
            &self.slot_ln.gamma,
            &self.slot_ln.beta,
        ]
    }

    fn arrays_mut(&mut self) -> Vec<&mut Vec<f32>> {
        vec![
            &mut self.state_lin.w,
            &mut self.state_lin.b,
            &mut self.state_pos.w,
            &mut self.state_pos.b,
            &mut self.state_ln.gamma,
            &mut self.state_ln.beta,
            &mut self.opt_lin.w,
            &mut self.opt_lin.b,
            &mut self.opt_pos.w,
            &mut self.opt_pos.b,
            &mut self.gamma_lin.w,
            &mut self.gamma_lin.b,
            &mut self.beta_lin.w,
            &mut self.beta_lin.b,
            &mut self.slot_ln.gamma,
            &mut self.slot_ln.beta,
        ]
    }

    /// Little-endian f32, one length-prefixed array per parameter block.
    /// Adam's moments are not written: a loaded grounder answers, it does not
    /// resume training, and a fresh run's zeroed moments are the correct
    /// start regardless.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.d_model as u64).to_le_bytes());
        for a in self.arrays() {
            out.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for v in a.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, out)
    }

    pub fn load(path: &str, feat_dim: usize, split: usize, lr: f32) -> Result<Grounder, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let u64_at = |b: &[u8], at: usize| -> Result<u64, String> {
            b.get(at..at + 8)
                .ok_or_else(|| format!("{path}: truncated at byte {at}"))
                .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
        };
        let d_model = u64_at(&bytes, 0)? as usize;
        let mut g = Grounder::new(feat_dim, split, d_model, 0, lr);
        let mut at = 8;
        // Shapes are checked rather than trusted: this file is a build
        // artifact under `out/` and can go stale against a code change that
        // resizes a projection.
        for (k, a) in g.arrays_mut().into_iter().enumerate() {
            let len = u64_at(&bytes, at)? as usize;
            at += 8;
            if len != a.len() {
                return Err(format!("{path}: array {k} has {len} values, this build expects {}", a.len()));
            }
            for v in a.iter_mut() {
                let raw = bytes.get(at..at + 4).ok_or_else(|| format!("{path}: truncated at byte {at}"))?;
                *v = f32::from_le_bytes(raw.try_into().expect("4 bytes"));
                at += 4;
            }
        }
        if at != bytes.len() {
            return Err(format!("{path}: {} trailing bytes", bytes.len() - at));
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| (0..dim).map(|_| rng.f32()).collect()).collect()
    }

    /// Every parameter's analytic gradient against a central difference of the
    /// same scalar loss.
    ///
    /// The loss is a dot product against a FIXED random target, not
    /// `sum(rows^2)`: a LayerNorm'd row has `sum(xhat^2) = dim` by
    /// construction, so a squared loss is nearly invariant to exactly the
    /// perturbations a gradient check makes and both the true gradient and any
    /// finite-difference estimate of it drown in f32 rounding. A fixed target
    /// makes the loss linear in the output, which has no such degeneracy.
    #[test]
    fn every_projection_matches_a_numerical_gradient() {
        let (feat_dim, split, d, n) = (9usize, 5usize, 6usize, 4usize);
        let mut g = Grounder::new(feat_dim, split, d, 3, 0.1);
        let f = feats(n, feat_dim, 5);
        let mut rng = Rng::new(17);
        let cls: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let target: Vec<f32> = (0..2 * n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();

        // dL/d(rows) = target, for L = rows . target
        g.forward(&f, &cls);
        g.backward(&f, &cls, &target);

        let analytic: Vec<Vec<f32>> = vec![
            g.state_lin.gw.clone(),
            g.state_lin.gb.clone(),
            g.state_pos.gw.clone(),
            g.state_pos.gb.clone(),
            g.state_ln.g_gamma.clone(),
            g.state_ln.g_beta.clone(),
            g.opt_lin.gw.clone(),
            g.opt_lin.gb.clone(),
            g.opt_pos.gw.clone(),
            g.opt_pos.gb.clone(),
            g.gamma_lin.gw.clone(),
            g.gamma_lin.gb.clone(),
            g.beta_lin.gw.clone(),
            g.beta_lin.gb.clone(),
            g.slot_ln.g_gamma.clone(),
            g.slot_ln.g_beta.clone(),
        ];
        let names = ["state_lin.w", "state_lin.b", "state_pos.w", "state_pos.b", "state_ln.gamma", "state_ln.beta", "opt_lin.w", "opt_lin.b", "opt_pos.w", "opt_pos.b", "gamma_lin.w", "gamma_lin.b", "beta_lin.w", "beta_lin.b", "slot_ln.gamma", "slot_ln.beta"];

        let eps = 1e-3f32;
        for (k, name) in names.iter().enumerate() {
            let len = analytic[k].len();
            for i in 0..len {
                let orig = g.arrays_mut()[k][i];

                g.arrays_mut()[k][i] = orig + eps;
                let plus: f32 = g.forward(&f, &cls).iter().zip(&target).map(|(v, t)| v * t).sum();
                g.arrays_mut()[k][i] = orig - eps;
                let minus: f32 = g.forward(&f, &cls).iter().zip(&target).map(|(v, t)| v * t).sum();
                g.arrays_mut()[k][i] = orig;

                let numeric = (plus - minus) / (2.0 * eps);
                let a = analytic[k][i];
                assert!(
                    (a - numeric).abs() < 1e-2 * (numeric.abs() + 1.0),
                    "{name}[{i}]: analytic {a} vs numeric {numeric}"
                );
            }
        }
    }

    /// Every row the head reads arrives at the scale the encoder's own rows
    /// have - the property that makes an attention dot product comparable
    /// between a spliced row and a real one.
    #[test]
    fn every_row_is_unit_scale_at_init() {
        let (feat_dim, d, n) = (20usize, 32usize, 5usize);
        let mut g = Grounder::new(feat_dim, 14, d, 1, 0.01);
        let f = feats(n, feat_dim, 2);
        let cls: Vec<f32> = vec![0.2; d];
        let rows = g.forward(&f, &cls);
        for (i, row) in rows.chunks(d).enumerate() {
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
            assert!((0.5..2.0).contains(&rms), "row {i}: rms {rms} is not unit scale");
        }
    }

    /// The instruction has to actually reach the slot rows and no further: two
    /// different instructions must move every slot row and leave every state
    /// row untouched. A wiring mistake that dropped the conditioning would train to
    /// a plausible-looking chance-level result instead of failing loudly.
    #[test]
    fn the_instruction_moves_slot_rows_and_only_slot_rows() {
        let (feat_dim, d, n) = (12usize, 16usize, 4usize);
        let mut g = Grounder::new(feat_dim, 8, d, 8, 0.01);
        let f = feats(n, feat_dim, 4);
        let a = g.forward(&f, &vec![0.3; d]);
        let b = g.forward(&f, &vec![-0.4; d]);
        for i in 0..n {
            let state_moved: f32 = (0..d).map(|h| (a[i * d + h] - b[i * d + h]).abs()).sum();
            assert!(state_moved < 1e-6, "state row {i} changed with the instruction");
            let at = (n + i) * d;
            let slot_moved: f32 = (0..d).map(|h| (a[at + h] - b[at + h]).abs()).sum();
            assert!(slot_moved > 1e-3, "slot row {i} did not change with the instruction");
        }
    }

    /// Two controls that look different must produce different rows - if they
    /// did not, no head could tell them apart however well it was trained.
    #[test]
    fn different_controls_produce_different_rows() {
        let (feat_dim, d, n) = (12usize, 16usize, 3usize);
        let mut g = Grounder::new(feat_dim, 8, d, 12, 0.01);
        let f = feats(n, feat_dim, 77);
        let rows = g.forward(&f, &vec![0.1; d]);
        for i in 0..n {
            for j in (i + 1)..n {
                let state: f32 = (0..d).map(|h| (rows[i * d + h] - rows[j * d + h]).abs()).sum();
                assert!(state > 1e-3, "state rows {i} and {j} are identical");
            }
        }
    }

    pub(super) fn round_trip_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("visualclick-grounder-{tag}-{}.bin", std::process::id()))
    }

    #[test]
    fn save_then_load_reproduces_the_same_rows() {
        let (feat_dim, d, n) = (14usize, 12usize, 3usize);
        let mut g = Grounder::new(feat_dim, 9, d, 5, 0.05);
        let f = feats(n, feat_dim, 6);
        let cls: Vec<f32> = vec![0.25; d];
        let target: Vec<f32> = (0..2 * n * d).map(|i| (i as f32 * 0.01).sin()).collect();
        for _ in 0..25 {
            g.forward(&f, &cls);
            g.backward(&f, &cls, &target);
            g.step();
        }
        let before = g.forward(&f, &cls);

        let path = round_trip_path("roundtrip");
        let p = path.to_str().expect("temp path is UTF-8");
        g.save(p).expect("save");
        let mut loaded = Grounder::load(p, feat_dim, 9, 0.05).expect("load");
        std::fs::remove_file(&path).ok();

        assert_eq!(before, loaded.forward(&f, &cls), "a loaded grounder must reproduce the saved one exactly");
    }

    /// A checkpoint from a different build is refused by name, not silently
    /// misread into rows that mean nothing.
    #[test]
    fn a_checkpoint_with_the_wrong_shape_is_refused() {
        let path = round_trip_path("badshape");
        let p = path.to_str().expect("temp path is UTF-8");
        Grounder::new(9, 5, 6, 1, 0.01).save(p).expect("save");
        let err = match Grounder::load(p, 11, 5, 0.01) {
            Ok(_) => panic!("a different feature width must be refused"),
            Err(e) => e,
        };
        std::fs::remove_file(&path).ok();
        assert!(err.contains("expects"), "the error should name the mismatch: {err}");
    }
}
