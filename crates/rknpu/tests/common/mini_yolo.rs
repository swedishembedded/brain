// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A YOLO-shaped test graph shared by the host test and the board check:
//! stride-2 stem with an even kernel, a residual bottleneck, a split into two
//! branches, concatenation, chained max pools, upsampling and two heads.

use data::rng::Rng;
use std::sync::Arc;

use rknpu::graph::{Graph, OwnedConvParams, TensorId};
use rknpu::layout::Shape;

fn conv(g: &mut Graph, rng: &mut Rng, x: TensorId, oc: usize, k: usize, s: usize, relu: bool) -> TensorId {
    let ic = g.shape(x).c;
    let bound = (1.0 / (ic * k * k) as f64).sqrt() * 1.7;
    let w = (0..oc * ic * k * k).map(|_| rng.uniform(-bound, bound) as f32).collect();
    let b = (0..oc).map(|_| rng.uniform(-0.1, 0.1) as f32).collect();
    let p = if k % 2 == 1 { k / 2 } else { 0 };
    g.conv(x, Arc::new(OwnedConvParams { weights: w, bias: b }), oc, k, s, [p; 4], relu).unwrap()
}

pub fn mini_yolo(rng: &mut Rng, size: usize) -> Graph {
    let mut g = Graph::new(Shape::new(size, size, 3));
    let x = g.input();
    let x = conv(&mut g, rng, x, 12, 2, 2, false); // space-to-depth stem
    let x = conv(&mut g, rng, x, 32, 3, 1, true);
    let x = conv(&mut g, rng, x, 32, 3, 2, true);
    // C3-style block: two branches, a residual bottleneck, concat, fuse.
    let a = conv(&mut g, rng, x, 16, 1, 1, true);
    let b = conv(&mut g, rng, x, 16, 1, 1, true);
    let m = conv(&mut g, rng, a, 16, 1, 1, true);
    let m = conv(&mut g, rng, m, 16, 3, 1, true);
    let m = g.add(a, m).unwrap();
    let cat = g.concat(&[m, b]).unwrap();
    let p3 = conv(&mut g, rng, cat, 32, 1, 1, true);
    // SPPF-style pooling pyramid.
    let y = conv(&mut g, rng, p3, 32, 3, 2, true);
    let y1 = g.max_pool(y, 3, 1, 1).unwrap();
    let y2 = g.max_pool(y1, 3, 1, 1).unwrap();
    let cat = g.concat(&[y, y1, y2]).unwrap();
    let p4 = conv(&mut g, rng, cat, 32, 1, 1, true);
    // Neck: upsample and merge with the higher-resolution branch.
    let up = g.upsample2x(p4);
    let cat = g.concat(&[up, p3]).unwrap();
    let n3 = conv(&mut g, rng, cat, 32, 3, 1, true);
    let h3 = conv(&mut g, rng, n3, 21, 1, 1, false);
    let h4 = conv(&mut g, rng, p4, 21, 1, 1, false);
    g.mark_output(h3);
    g.mark_output(h4);
    g
}
