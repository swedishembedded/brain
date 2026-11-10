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
    Structured { cond_v: Weight, rot: [Kron; 3], hp: DeviceBuffer },
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
                let rotation = |i: usize, pre: Option<&[f32]>, perm: Option<&[u32]>, post: Option<&[f32]>| {
                    Kron::upload(dev, mlp.factors, &KronParams { a: &mlp.w[i].0, b: &mlp.w[i].1, pre, perm, post })
                };
                Form::Structured {
                    cond_v: dev.linear_weight("cond_v", &mlp.cond_v),
                    rot: [
                        rotation(0, Some(&mlp.d[0]), Some(&perms.0), None),
                        rotation(1, None, Some(&perms.1), None),
                        rotation(2, Some(&mlp.d[2]), None, Some(&mlp.d[3])),
                    ],
                    hp: dev.buf(&[mlp.cond_u.as_slice(), &mlp.d[1], &mlp.b2].concat()),
                }
            }
        };
        DeviceHadamard { n: n as u32, form }
    }

    /// Append the steps adding the MLP of the normalised rows `h` to the residual rows `y`.
    pub fn encode_add(&self, dev: &Device, s: &mut Vec<Step>, h: &DeviceBuffer, rows: u32, y: &DeviceBuffer) {
        let (n, rank) = (self.n, HADAMARD_COND_RANK as u32);
        let z1 = dev.alloc((rows * n) as usize);
        match &self.form {
            Form::Folded { cond_rot, rot, cond_u, d2, b2 } => {
                let zc = dev.linear_new(s, cond_rot, h, rows);
                dev.hada_mid(s, (&zc, rank + n, 0), (&zc, rank + n, rank), cond_u, (d2, b2), &z1, (rows, n, rank));
                let z2 = dev.linear_new(s, &rot[0], &z1, rows);
                let z3 = dev.linear_new(s, &rot[1], &z2, rows);
                dev.add(s, &z3, y, rows * n);
            }
            Form::Structured { cond_v, rot, hp } => {
                let logits = dev.linear_new(s, cond_v, h, rows);
                dev.kron_hada(s, &rot[0], h, rows, (&logits, hp), rank, &z1);
                let z2 = dev.alloc((rows * n) as usize);
                dev.kron(s, &rot[1], &z1, rows, &z2, false);
                dev.kron(s, &rot[2], &z2, rows, y, true);
            }
        }
    }
}
