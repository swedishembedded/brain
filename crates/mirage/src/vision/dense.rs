// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The dense head: a feature pyramid from the encoder grid up to stride 8,
//! the ten-channel detection map, and the mask prototype / coefficient
//! branches that share its input.
//!
//! Channels of the detection map: object logit, text logit, object width and
//! height, text width and height, object centre offsets (x, y), text centre
//! offsets (x, y); sizes are in cells.

use super::encoder::EncoderOutput;
use super::ops::{self, Linear, Map};
use super::weights::Weights;
use crate::config::{MirageConfig, DENSE_BLOCKS, DENSE_OUT_CHANNELS, MASK_BRANCH_BLOCKS, MASK_RANK};

/// Which proposal family a detection channel set describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Objects.
    Object,
    /// Text lines.
    Text,
}

impl Kind {
    /// `(logit, width, height, dx, dy)` channel indices.
    fn channels(self) -> [usize; 5] {
        match self {
            Kind::Object => [0, 2, 3, 6, 7],
            Kind::Text => [1, 4, 5, 8, 9],
        }
    }
}

/// One detected proposal, boxes canvas-normalised in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// Sigmoid of the peak logit.
    pub score: f32,
    /// Peak cell row on the stride-8 map.
    pub y: usize,
    /// Peak cell column.
    pub x: usize,
    /// `[x0, y0, x1, y1]`.
    pub bbox: [f32; 4],
}

/// Default confidence threshold for proposals.
pub const DEFAULT_THRESHOLD: f32 = 0.3;

/// Most object / text proposals kept.
pub const MAX_OBJECTS: usize = 64;
/// Most text proposals kept.
pub const MAX_TEXT: usize = 32;

struct Block {
    norm: Vec<f32>,
    dw: Vec<f32>,
    pw: Linear,
}

struct Branch {
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    head: Linear,
}

impl Branch {
    fn new(w: &Weights, pre: &str, out: &str, n: usize) -> Branch {
        Branch {
            blocks: (0..n)
                .map(|i| Block {
                    norm: w.vec(&format!("dense/{pre}_norm_{i}/scale")),
                    dw: w.vec(&format!("dense/{pre}_dw_{i}")),
                    pw: w.linear(&format!("dense/{pre}_pw_{i}/kernel")),
                })
                .collect(),
            final_norm: w.vec(&format!("dense/{pre}_final_norm/scale")),
            head: w.linear(&format!("dense/{out}/kernel")),
        }
    }

    /// Residual blocks re-masked after every block, then the activated linear head.
    fn forward(&self, mut y: Vec<f32>, m: Map, valid: &[bool]) -> Vec<f32> {
        for b in &self.blocks {
            let mut a = ops::zc_norm(&y, m.c, &b.norm);
            ops::silu_inplace(&mut a);
            let (d, _) = ops::depthwise(&a, m, &b.dw, 3, 3, 1, 1);
            let mut r = b.pw.apply(&d, m.h * m.w);
            ops::add_inplace(&mut r, &y);
            ops::mask_rows(&mut r, m.c, valid);
            y = r;
        }
        let mut a = ops::zc_norm(&y, m.c, &self.final_norm);
        ops::silu_inplace(&mut a);
        self.head.apply(&a, m.h * m.w)
    }
}

/// The detection head and the mask branches.
pub struct DenseHead {
    cfg: MirageConfig,
    coarse: Linear,
    mid: Linear,
    fine: Linear,
    neck_norm: Vec<f32>,
    neck_dw: Vec<f32>,
    neck_pw: Linear,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    out: Linear,
    bias: Vec<f32>,
    edge: Linear,
    proto: Branch,
    coef: Branch,
}

/// Dense-head outputs.
pub struct DenseOutput {
    /// Stride-8 working map before the head, `[64², 128]`.
    pub d64: Vec<f32>,
    /// The head input `silu(ZC(d64))`, padding zeroed, `[64², 128]`.
    pub head_in: Vec<f32>,
    /// The detection map, `[64², 10]`.
    pub head: Vec<f32>,
}

impl DenseHead {
    /// Prepare from the checkpoint tensors.
    pub fn new(cfg: &MirageConfig, w: &Weights) -> DenseHead {
        DenseHead {
            cfg: cfg.clone(),
            coarse: w.linear("dense/coarse/kernel"),
            mid: w.linear("dense/mid/kernel"),
            fine: w.linear("dense/fine/kernel"),
            neck_norm: w.vec("dense/neck_norm/scale"),
            neck_dw: w.vec("dense/neck_dw"),
            neck_pw: w.linear("dense/neck_pw/kernel"),
            blocks: (0..DENSE_BLOCKS)
                .map(|i| Block {
                    norm: w.vec(&format!("dense/block_norm_{i}/scale")),
                    dw: w.vec(&format!("dense/dw_{i}")),
                    pw: w.linear(&format!("dense/pw_{i}/kernel")),
                })
                .collect(),
            final_norm: w.vec("dense/final_norm/scale"),
            out: w.linear("dense/out/kernel"),
            bias: w.vec("dense/bias"),
            edge: w.linear("dense/edge/kernel"),
            proto: Branch::new(w, "proto", "protos", MASK_BRANCH_BLOCKS),
            coef: Branch::new(w, "coef", "coef", MASK_BRANCH_BLOCKS),
        }
    }

    fn side(&self) -> usize {
        self.cfg.canvas / self.cfg.stride
    }

    /// The pyramid and detection map for a resized image of `h x w`.
    pub fn forward(&self, enc: &EncoderOutput, h: usize, w: usize) -> DenseOutput {
        let d = self.cfg.dense_dim;
        let (_, fine_level, mid_level) = self.cfg.dense_levels();
        let grid = self.cfg.grid();
        let coarse = self.coarse.apply(&enc.encoder, grid * grid);
        let (mut d32, m32) = ops::upsample2(&coarse, Map { h: grid, w: grid, c: d });
        let (mid_map, mid_x) = &enc.stages[mid_level];
        let skip = self.mid.apply(mid_x, mid_map.h * mid_map.w);
        ops::add_inplace(&mut d32, &skip);
        let v32 = ops::valid_cells(h, w, m32.h);
        ops::mask_rows(&mut d32, d, &v32);
        let mut n = ops::zc_norm(&d32, d, &self.neck_norm);
        ops::silu_inplace(&mut n);
        let (nd, _) = ops::depthwise(&n, m32, &self.neck_dw, 3, 3, 1, 1);
        let mut neck = self.neck_pw.apply(&nd, m32.h * m32.w);
        ops::mask_rows(&mut neck, d, &v32);
        ops::add_inplace(&mut d32, &neck);

        let (mut d64, m64) = ops::upsample2(&d32, m32);
        let (fine_map, fine_x) = &enc.stages[fine_level];
        let skip = self.fine.apply(fine_x, fine_map.h * fine_map.w);
        ops::add_inplace(&mut d64, &skip);
        for b in &self.blocks {
            let mut a = ops::zc_norm(&d64, d, &b.norm);
            ops::silu_inplace(&mut a);
            let (dw, _) = ops::depthwise(&a, m64, &b.dw, 3, 3, 1, 1);
            let r = b.pw.apply(&dw, m64.h * m64.w);
            ops::add_inplace(&mut d64, &r);
        }
        let mut head_in = ops::zc_norm(&d64, d, &self.final_norm);
        ops::silu_inplace(&mut head_in);
        ops::mask_rows(&mut head_in, d, &ops::valid_cells(h, w, m64.h));
        let head = self.out.apply_bias(&head_in, m64.h * m64.w, &self.bias);
        debug_assert_eq!(head.len(), self.side() * self.side() * DENSE_OUT_CHANNELS);
        DenseOutput { d64, head_in, head }
    }

    /// Mask prototypes `[128², 32]` and the coefficient map `[64², 32]`.
    pub fn mask_maps(&self, head_in: &[f32], enc: &EncoderOutput, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.dense_dim;
        let side = self.side();
        let (edge_level, _, _) = self.cfg.dense_levels();
        let (edge_map, edge_x) = &enc.stages[edge_level];
        let (mut u, um) = ops::upsample2(head_in, Map { h: side, w: side, c: d });
        let skip = self.edge.apply(edge_x, edge_map.h * edge_map.w);
        ops::add_inplace(&mut u, &skip);
        let v_proto = ops::valid_cells(h, w, um.h);
        ops::mask_rows(&mut u, d, &v_proto);
        let protos = self.proto.forward(u, um, &v_proto);

        let v_coef = ops::valid_cells(h, w, side);
        let mut c_in = head_in.to_vec();
        ops::mask_rows(&mut c_in, d, &v_coef);
        let coefs = self.coef.forward(c_in, Map { h: side, w: side, c: d }, &v_coef);
        debug_assert_eq!(protos.len(), um.h * um.w * MASK_RANK);
        (protos, coefs)
    }
}

/// Local-maximum proposals of one kind from the detection map.
///
/// A cell is kept when its probability equals the maximum of its 3x3
/// neighbourhood (outside the map counts as absent) and exceeds `threshold`.
/// The `cap` best by score survive and are ordered by 16 horizontal bands of
/// the box centre, then by centre x (stable). `valid` optionally restricts
/// candidates (and neighbours) to cells inside the resized image.
pub fn find_proposals(head: &[f32], side: usize, kind: Kind, threshold: f32, cap: usize, valid: Option<&[bool]>) -> Vec<Proposal> {
    let [lc, wc, hc, dxc, dyc] = kind.channels();
    let prob: Vec<f32> = head.as_chunks::<DENSE_OUT_CHANNELS>().0.iter().map(|c| ops::sigmoid(c[lc])).collect();
    let usable = |y: isize, x: isize| {
        (0..side as isize).contains(&y) && (0..side as isize).contains(&x) && valid.is_none_or(|v| v[y as usize * side + x as usize])
    };
    let mut found = Vec::new();
    for y in 0..side {
        for x in 0..side {
            if !usable(y as isize, x as isize) {
                continue;
            }
            let p = prob[y * side + x];
            let is_peak = (-1..=1).all(|dy| (-1..=1).all(|dx| !usable(y as isize + dy, x as isize + dx) || prob[(y as isize + dy) as usize * side + (x as isize + dx) as usize] <= p));
            if !(is_peak && p > threshold) {
                continue;
            }
            let v = &head[(y * side + x) * DENSE_OUT_CHANNELS..][..DENSE_OUT_CHANNELS];
            let (cx, cy) = (x as f32 + v[dxc], y as f32 + v[dyc]);
            let (bw, bh) = (v[wc].abs(), v[hc].abs());
            let s = side as f32;
            let bbox = [(cx - bw / 2.0) / s, (cy - bh / 2.0) / s, (cx + bw / 2.0) / s, (cy + bh / 2.0) / s].map(|c| c.clamp(0.0, 1.0));
            found.push(Proposal { score: p, y, x, bbox });
        }
    }
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    found.truncate(cap);
    let key = |p: &Proposal| ((16.0 * f64::from(p.bbox[1] + p.bbox[3]) / 2.0).floor()) * 10000.0 + f64::from(p.bbox[0] + p.bbox[2]) / 2.0 * 1000.0;
    found.sort_by(|a, b| key(a).total_cmp(&key(b)));
    found
}
