// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Loss primitives, each a scalar with its own backward: weighted token cross-entropy, CenterNet focal loss,
//! weighted L1, weighted binary cross-entropy with Dice, and CTC.
//!
//! These are brain's own objective terms; the vendor did not publish a training recipe (see the crate ledger).

use super::ops::rows;
use super::real::{r, sigmoid, Real};
use super::tape::{Id, Tape};

/// Focal-loss exponent on the prediction error (CenterNet).
const FOCAL_GAMMA: i32 = 2;
/// Focal-loss exponent that discounts negatives near a positive (CenterNet).
const FOCAL_BETA: i32 = 4;
/// Floor inside logarithms.
const LOG_FLOOR: f64 = 1e-12;

fn softplus<R: Real>(x: R) -> R {
    // log(1 + e^x), stable for either sign.
    x.max(R::ZERO) + (R::ONE + (-(x.abs())).exp()).ln()
}

impl<R: Real> Tape<R> {
    /// `sum_r weight[r] * (logsumexp(logits[r]) - logits[r][target[r]])` over rows of width `v`. A row with weight
    /// zero is ignored (its target is not read). The caller normalises the weights.
    pub fn softmax_ce(&mut self, logits: Id, v: usize, targets: &[u32], weights: &[f64]) -> Id {
        let n = targets.len();
        assert_eq!(self.len(logits), n * v, "softmax_ce: logits do not match the targets");
        assert_eq!(weights.len(), n, "softmax_ce: one weight per row");
        let lse_row = |row: &[R]| {
            let max = row.iter().copied().fold(row[0], |a, b| a.max(b));
            max + row.iter().map(|x| (*x - max).exp()).sum::<R>().ln()
        };
        let total: R = self
            .val(logits)
            .chunks_exact(v)
            .zip(targets.iter().zip(weights))
            .filter(|(_, (_, w))| **w != 0.0)
            .map(|(row, (&t, &w))| r::<R>(w) * (lse_row(row) - row[t as usize]))
            .sum();
        let (targets, weights) = (targets.to_vec(), weights.to_vec());
        self.op(vec![total], move |vals, gy, g| {
            let mut dl = vec![R::ZERO; n * v];
            rows(&mut dl, v, |ri, out| {
                if weights[ri] == 0.0 {
                    return;
                }
                let row = &vals[logits.0][ri * v..(ri + 1) * v];
                let max = row.iter().copied().fold(row[0], |a, b| a.max(b));
                let denom = row.iter().map(|x| (*x - max).exp()).sum::<R>();
                let w = r::<R>(weights[ri]) * gy[0];
                for (j, o) in out.iter_mut().enumerate() {
                    *o = w * (row[j] - max).exp() / denom;
                }
                out[targets[ri] as usize] -= w;
            });
            g.add(logits, &dl);
        })
    }

    /// CenterNet focal loss of heat logits against a soft target in `[0, 1]`, summed over cells with
    /// `weight[i]` (0 excludes a cell) and divided by `norm`. A cell is positive where `target == 1`. `alpha`
    /// weights positives and `1 - alpha` negatives.
    pub fn focal_heat(&mut self, logits: Id, target: &[f64], weight: &[f64], alpha: f64, norm: f64) -> Id {
        let n = target.len();
        assert_eq!(self.len(logits), n, "focal_heat: logits do not match the target");
        assert_eq!(weight.len(), n, "focal_heat: one weight per cell");
        let floor = r::<R>(LOG_FLOOR);
        let term = move |x: R, t: f64, w: f64| -> R {
            if w == 0.0 {
                return R::ZERO;
            }
            let p = sigmoid(x);
            if t >= 1.0 {
                -r::<R>(alpha * w) * (R::ONE - p).powi_(FOCAL_GAMMA) * (p + floor).ln()
            } else {
                -r::<R>((1.0 - alpha) * w * (1.0 - t).powi(FOCAL_BETA)) * p.powi_(FOCAL_GAMMA) * (R::ONE - p + floor).ln()
            }
        };
        let total: R = self.val(logits).iter().enumerate().map(|(i, x)| term(*x, target[i], weight[i])).sum::<R>() / r::<R>(norm);
        let (target, weight) = (target.to_vec(), weight.to_vec());
        self.op(vec![total], move |vals, gy, g| {
            let x = &vals[logits.0];
            let mut dl = vec![R::ZERO; n];
            rows(&mut dl, 1, |i, o| {
                if weight[i] == 0.0 {
                    return;
                }
                let p = sigmoid(x[i]);
                let one = R::ONE;
                // d p / d x = p (1 - p)
                let dp = p * (one - p);
                let d = if target[i] >= 1.0 {
                    // -a w [ -2 (1-p) log(p+f) + (1-p)^2 / (p+f) ] dp
                    -r::<R>(alpha * weight[i]) * (-r::<R>(f64::from(FOCAL_GAMMA)) * (one - p) * (p + floor).ln() + (one - p).powi_(FOCAL_GAMMA) / (p + floor)) * dp
                } else {
                    // -b w (1-t)^4 [ 2 p log(1-p+f) - p^2 / (1-p+f) ] dp
                    -r::<R>((1.0 - alpha) * weight[i] * (1.0 - target[i]).powi(FOCAL_BETA))
                        * (r::<R>(f64::from(FOCAL_GAMMA)) * p * (one - p + floor).ln() - p.powi_(FOCAL_GAMMA) / (one - p + floor))
                        * dp
                };
                o[0] = d * gy[0] / r::<R>(norm);
            });
            g.add(logits, &dl);
        })
    }

    /// `sum_i weight[i] * |x[i] - target[i]| / norm` (subgradient zero at equality).
    pub fn l1(&mut self, x: Id, target: &[f64], weight: &[f64], norm: f64) -> Id {
        let n = target.len();
        assert_eq!(self.len(x), n, "l1: prediction does not match the target");
        let total: R = self.val(x).iter().enumerate().map(|(i, v)| r::<R>(weight[i]) * (*v - r::<R>(target[i])).abs()).sum::<R>() / r::<R>(norm);
        let (target, weight) = (target.to_vec(), weight.to_vec());
        self.op(vec![total], move |vals, gy, g| {
            let dx: Vec<R> = vals[x.0]
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let diff = *v - r::<R>(target[i]);
                    let sign = if diff > R::ZERO { R::ONE } else if diff < R::ZERO { -R::ONE } else { R::ZERO };
                    r::<R>(weight[i]) * sign * gy[0] / r::<R>(norm)
                })
                .collect();
            g.add(x, &dx);
        })
    }

    /// Mask loss over one object's region: weighted mean binary cross-entropy of `logits` plus the Dice loss
    /// `1 - (2 sum(p t) + 1) / (sum(p) + sum(t) + 1)`, both against the 0/1 `target`.
    pub fn bce_dice(&mut self, logits: Id, target: &[f64], bce_weight: f64, dice_weight: f64) -> Id {
        let n = target.len();
        assert_eq!(self.len(logits), n, "bce_dice: logits do not match the target");
        assert!(n > 0, "bce_dice: an empty region");
        let nf = r::<R>(n as f64);
        let one = R::ONE;
        let (bce, sp, spt, st) = dice_parts(self.val(logits), target);
        let dice = one - (r::<R>(2.0) * spt + one) / (sp + st + one);
        let total = r::<R>(bce_weight) * bce / nf + r::<R>(dice_weight) * dice;
        let target = target.to_vec();
        self.op(vec![total], move |vals, gy, g| {
            let xs = &vals[logits.0];
            let (_, sp, spt, st) = dice_parts(xs, &target);
            let den = sp + st + one;
            let num = r::<R>(2.0) * spt + one;
            let dx: Vec<R> = xs
                .iter()
                .zip(&target)
                .map(|(x, t)| {
                    let t = r::<R>(*t);
                    let p = sigmoid(*x);
                    let dp = p * (one - p);
                    // d dice / d p = -(2 t * den - num) / den^2
                    let ddice = -(r::<R>(2.0) * t * den - num) / (den * den);
                    (r::<R>(bce_weight) * (p - t) / nf + r::<R>(dice_weight) * ddice * dp) * gy[0]
                })
                .collect();
            g.add(logits, &dx);
        })
    }

    /// Connectionist temporal classification loss `-log p(labels | logits)` for `logits: [frames, classes]` with
    /// blank class 0 and `labels` in `1..classes`. An alignment cannot exist when the frames are too few for the
    /// labels and their forced blanks; that sample has loss zero and no gradient (reported by `feasible`).
    pub fn ctc(&mut self, logits: Id, classes: usize, labels: &[u32]) -> Id {
        let frames = self.len(logits) / classes;
        if !ctc_feasible(frames, labels) {
            return self.op(vec![R::ZERO], |_, _, _| {});
        }
        let (loss, grad) = ctc_forward_backward(self.val(logits), frames, classes, labels);
        self.op(vec![loss], move |_, gy, g| g.add(logits, &grad.iter().map(|d| *d * gy[0]).collect::<Vec<_>>()))
    }
}

/// `(sum of binary cross-entropy terms, sum p, sum p t, sum t)` of logits against a 0/1 target.
fn dice_parts<R: Real>(xs: &[R], target: &[f64]) -> (R, R, R, R) {
    let (mut bce, mut sp, mut spt, mut st) = (R::ZERO, R::ZERO, R::ZERO, R::ZERO);
    for (x, t) in xs.iter().zip(target) {
        let t = r::<R>(*t);
        bce += softplus(*x) - t * *x;
        let p = sigmoid(*x);
        sp += p;
        spt += p * t;
        st += t;
    }
    (bce, sp, spt, st)
}

/// Whether a CTC alignment of `labels` into `frames` frames exists (repeated labels need a blank between them).
pub fn ctc_feasible(frames: usize, labels: &[u32]) -> bool {
    let repeats = labels.windows(2).filter(|w| w[0] == w[1]).count();
    !labels.is_empty() && labels.len() + repeats <= frames
}

trait PowI: Sized {
    fn powi_(self, e: i32) -> Self;
}

impl<R: Real> PowI for R {
    fn powi_(self, e: i32) -> R {
        (0..e).fold(R::ONE, |acc, _| acc * self)
    }
}

fn log_add<R: Real>(a: R, b: R) -> R {
    let neg_inf = r::<R>(-1e300);
    if a <= neg_inf {
        return b;
    }
    if b <= neg_inf {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// `(-log p(labels|x), d loss / d logits)` in log space.
fn ctc_forward_backward<R: Real>(logits: &[R], frames: usize, classes: usize, labels: &[u32]) -> (R, Vec<R>) {
    let neg_inf = r::<R>(-1e300);
    // Log-softmax per frame.
    let mut lp = logits.to_vec();
    for row in lp.chunks_exact_mut(classes) {
        let max = row.iter().copied().fold(row[0], |a, b| a.max(b));
        let lse = max + row.iter().map(|x| (*x - max).exp()).sum::<R>().ln();
        row.iter_mut().for_each(|x| *x -= lse);
    }
    // Extended label sequence: blank, l1, blank, l2, ..., blank.
    let ext: Vec<usize> = std::iter::once(0).chain(labels.iter().flat_map(|&l| [l as usize, 0])).collect();
    let s_len = ext.len();
    let skip_ok = |s: usize| s >= 2 && ext[s] != 0 && ext[s] != ext[s - 2];
    let at = |t: usize, s: usize| lp[t * classes + ext[s]];
    let mut alpha = vec![neg_inf; frames * s_len];
    alpha[0] = at(0, 0);
    if s_len > 1 {
        alpha[1] = at(0, 1);
    }
    for t in 1..frames {
        for s in 0..s_len {
            let mut a = alpha[(t - 1) * s_len + s];
            if s >= 1 {
                a = log_add(a, alpha[(t - 1) * s_len + s - 1]);
            }
            if skip_ok(s) {
                a = log_add(a, alpha[(t - 1) * s_len + s - 2]);
            }
            alpha[t * s_len + s] = if a <= neg_inf { neg_inf } else { a + at(t, s) };
        }
    }
    let last = (frames - 1) * s_len;
    let log_p = if s_len > 1 { log_add(alpha[last + s_len - 1], alpha[last + s_len - 2]) } else { alpha[last] };

    let mut beta = vec![neg_inf; frames * s_len];
    beta[last + s_len - 1] = at(frames - 1, s_len - 1);
    if s_len > 1 {
        beta[last + s_len - 2] = at(frames - 1, s_len - 2);
    }
    for t in (0..frames - 1).rev() {
        for s in 0..s_len {
            let mut b = beta[(t + 1) * s_len + s];
            if s + 1 < s_len {
                b = log_add(b, beta[(t + 1) * s_len + s + 1]);
            }
            if s + 2 < s_len && skip_ok(s + 2) {
                b = log_add(b, beta[(t + 1) * s_len + s + 2]);
            }
            beta[t * s_len + s] = if b <= neg_inf { neg_inf } else { b + at(t, s) };
        }
    }
    // d loss / d logits = softmax - occupancy, occupancy_k = sum_{s: ext[s]=k} alpha beta / (y_k * P).
    let mut grad = vec![R::ZERO; frames * classes];
    for t in 0..frames {
        let mut occ = vec![neg_inf; classes];
        for s in 0..s_len {
            let ab = alpha[t * s_len + s] + beta[t * s_len + s];
            if ab > neg_inf / r::<R>(2.0) {
                occ[ext[s]] = log_add(occ[ext[s]], ab);
            }
        }
        for k in 0..classes {
            let softmax = lp[t * classes + k].exp();
            let occupancy = if occ[k] <= neg_inf / r::<R>(2.0) { R::ZERO } else { (occ[k] - lp[t * classes + k] - log_p).exp() };
            grad[t * classes + k] = softmax - occupancy;
        }
    }
    (-log_p, grad)
}
