// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Hadamard MLP of a decoder layer on the device: `y += D4 K3 D3 P1 K2 silu(...) ...` with the rank-8
//! conditioning between the first two rotations.
//!
//! Two resident forms are gated against each other, as for the HConvGLU mixers: the structured form runs the
//! three Kronecker rotations (`16x16` and `32x32` factors, the permutations and diagonals) with
//! `mirage_kron_rows`, the folded form multiplies each rotation out into a dense `n x n` matrix and runs
//! GEMMs.

use gpu_core::{DeviceBuffer, Step};
use model::ops::Weight;

use super::kron::{Kron, KronParams};
use super::{Device, WeightForm};
use crate::config::HADAMARD_COND_RANK;
use crate::decoder::hadamard::HadamardMlp;

/// One layer's resident Hadamard MLP.
pub struct DeviceHadamard {
    n: u32,
    form: Form,
}

enum Form {
    Folded { cond_rot: Weight, rot: [Weight; 2], cond_u: DeviceBuffer, d2: DeviceBuffer, b2: DeviceBuffer },
    /// `hp` is `[cond_u | d2 | b2]` for the fused first rotation and middle stage.
    ///
    /// `packed` holds the three rotations back to back for the single-dispatch kernel of devices with multi-barrier workgroups.
    Structured { cond_v: Weight, rot: [Kron; 3], hp: DeviceBuffer, packed: DeviceBuffer },
}

impl DeviceHadamard {
    /// Upload `mlp` with the permutations `perms` in the device's weight form.
    pub fn upload(dev: &Device, mlp: &HadamardMlp, perms: &(Vec<u32>, Vec<u32>)) -> DeviceHadamard {
        let n = mlp.n;
        let form = match dev.form() {
            WeightForm::Folded => {
                let fold = mlp.fold(perms);
                Form::Folded {
                    cond_rot: dev.weight("cond_rot", &[mlp.cond_v.wt.clone(), fold[0].clone()].concat(), HADAMARD_COND_RANK + n, n),
                    rot: [dev.weight("w2", &fold[1], n, n), dev.weight("w3", &fold[2], n, n)],
                    cond_u: dev.buf(&mlp.cond_u),
                    d2: dev.buf(&mlp.d[1]),
                    b2: dev.buf(&mlp.b2),
                }
            }
            WeightForm::Structured => {
                let spec = |i: usize| -> KronParams<'_> {
                    match i {
                        0 => KronParams { a: &mlp.w[0].0, b: &mlp.w[0].1, pre: Some(&mlp.d[0]), perm: Some(&perms.0), post: None },
                        1 => KronParams { a: &mlp.w[1].0, b: &mlp.w[1].1, pre: None, perm: Some(&perms.1), post: None },
                        _ => KronParams { a: &mlp.w[2].0, b: &mlp.w[2].1, pre: Some(&mlp.d[2]), perm: None, post: Some(&mlp.d[3]) },
                    }
                };
                let n = mlp.n;
                Form::Structured {
                    cond_v: dev.linear_weight("cond_v", &mlp.cond_v),
                    rot: [0, 1, 2].map(|i| Kron::upload(dev, mlp.factors, &spec(i))),
                    hp: dev.buf(&[mlp.cond_u.as_slice(), &mlp.d[1], &mlp.b2].concat()),
                    packed: dev.buf(&(0..3).flat_map(|i| spec(i).packed(n)).collect::<Vec<f32>>()),
                }
            }
        };
        DeviceHadamard { n: n as u32, form }
    }

    /// Append the steps adding the MLP of the normalised rows `h` to the residual rows `y`.
    pub fn encode_add(&self, dev: &Device, s: &mut Vec<Step>, h: &DeviceBuffer, rows: u32, y: &DeviceBuffer) {
        let (n, rank) = (self.n, HADAMARD_COND_RANK as u32);
        match &self.form {
            Form::Folded { cond_rot, rot, cond_u, d2, b2 } => {
                let z1 = dev.alloc((rows * n) as usize);
                let zc = dev.linear_new(s, cond_rot, h, rows);
                dev.hada_mid(s, (&zc, rank + n, 0), (&zc, rank + n, rank), cond_u, (d2, b2), &z1, (rows, n, rank));
                let z2 = dev.linear_new(s, &rot[0], &z1, rows);
                let z3 = dev.linear_new(s, &rot[1], &z2, rows);
                dev.add(s, &z3, y, rows * n);
            }
            Form::Structured { cond_v, rot, hp, packed } => {
                let logits = dev.linear_new(s, cond_v, h, rows);
                let (fa, fb) = rot[0].factors();
                if dev.cooperative() && n <= 512 && fa <= 32 && fb <= 32 && rank <= 32 {
                    let bufs = [h, packed, &logits, hp, y];
                    s.push(dev.gpu().dispatch(dev.k("mirage_hada_mlp_wg"), &bufs, &[rows, fa, fb, rank], gpu_core::Dispatch::Workgroups(rows)));
                    return;
                }
                let z1 = dev.alloc((rows * n) as usize);
                dev.kron_hada(s, &rot[0], h, rows, (&logits, hp), rank, &z1);
                let z2 = dev.alloc((rows * n) as usize);
                dev.kron(s, &rot[1], &z1, rows, &z2, false);
                dev.kron(s, &rot[2], &z2, rows, y, true);
            }
        }
    }
}
