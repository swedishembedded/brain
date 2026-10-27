// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The YOLOv5s layer topology (Rockchip's ReLU export: a 2x2 stride-2 stem,
//! C3 blocks with 1/2/3/1 bottlenecks, SPPF with chained 3x3 pools, the PAN
//! neck and three 255-channel heads) with random weights. Real size, real
//! schedule: what the board check times. The detector's own graph comes from
//! its checkpoint, not from here.

use std::borrow::Cow;
use std::sync::Arc;

use data::rng::Rng;
use rknpu::graph::{ConvParams, Graph, TensorId};
use rknpu::layout::Shape;

/// Uniform random parameters regenerated from a seed on every request, so
/// even a full-size network costs no memory until a layer is quantized.
struct SeededParams {
    seed: u64,
    weights: usize,
    bias: usize,
    bound: f64,
}

impl ConvParams for SeededParams {
    fn weights(&self) -> Cow<'_, [f32]> {
        let mut rng = Rng::new(self.seed);
        Cow::Owned((0..self.weights).map(|_| rng.uniform(-self.bound, self.bound) as f32).collect())
    }

    fn bias(&self) -> Cow<'_, [f32]> {
        let mut rng = Rng::new(self.seed ^ 0x9e37_79b9_7f4a_7c15);
        Cow::Owned((0..self.bias).map(|_| rng.uniform(-0.1, 0.1) as f32).collect())
    }
}

struct B<'a> {
    g: Graph,
    rng: &'a mut Rng,
}

impl B<'_> {
    fn conv(&mut self, x: TensorId, oc: usize, k: usize, s: usize, relu: bool) -> TensorId {
        let ic = self.g.shape(x).c;
        let params = SeededParams { seed: self.rng.next_u64(), weights: oc * ic * k * k, bias: oc, bound: (1.0 / (ic * k * k) as f64).sqrt() * 1.7 };
        let p = if k % 2 == 1 { k / 2 } else { 0 };
        self.g.conv(x, Arc::new(params), oc, k, s, [p; 4], relu).unwrap()
    }

    fn c3(&mut self, x: TensorId, oc: usize, n: usize, shortcut: bool) -> TensorId {
        let h = oc / 2;
        let mut m = self.conv(x, h, 1, 1, true);
        let b = self.conv(x, h, 1, 1, true);
        for _ in 0..n {
            let y = self.conv(m, h, 1, 1, true);
            let y = self.conv(y, h, 3, 1, true);
            m = if shortcut { self.g.add(m, y).unwrap() } else { y };
        }
        let cat = self.g.concat(&[m, b]).unwrap();
        self.conv(cat, oc, 1, 1, true)
    }

    fn sppf(&mut self, x: TensorId, oc: usize) -> TensorId {
        let x = self.conv(x, oc / 2, 1, 1, true);
        let mut ys = vec![x];
        let mut y = x;
        for _ in 0..3 {
            // Rockchip's export replaces each 5x5 pool with two 3x3 pools.
            y = self.g.max_pool(y, 3, 1, 1).unwrap();
            y = self.g.max_pool(y, 3, 1, 1).unwrap();
            ys.push(y);
        }
        let cat = self.g.concat(&ys).unwrap();
        self.conv(cat, oc, 1, 1, true)
    }
}

pub fn yolov5s(rng: &mut Rng, size: usize) -> Graph {
    let mut b = B { g: Graph::new(Shape::new(size, size, 3)), rng };
    let x = b.g.input();
    let x = b.conv(x, 12, 2, 2, false);
    let x = b.conv(x, 32, 3, 1, true);
    let x = b.conv(x, 64, 3, 2, true);
    let x = b.c3(x, 64, 1, true);
    let x = b.conv(x, 128, 3, 2, true);
    let p3 = b.c3(x, 128, 2, true);
    let x = b.conv(p3, 256, 3, 2, true);
    let p4 = b.c3(x, 256, 3, true);
    let x = b.conv(p4, 512, 3, 2, true);
    let x = b.c3(x, 512, 1, true);
    let x = b.sppf(x, 512);
    let h10 = b.conv(x, 256, 1, 1, true);
    let up = b.g.upsample2x(h10);
    let cat = b.g.concat(&[up, p4]).unwrap();
    let x = b.c3(cat, 256, 1, false);
    let h14 = b.conv(x, 128, 1, 1, true);
    let up = b.g.upsample2x(h14);
    let cat = b.g.concat(&[up, p3]).unwrap();
    let n3 = b.c3(cat, 128, 1, false);
    let x = b.conv(n3, 128, 3, 2, true);
    let cat = b.g.concat(&[x, h14]).unwrap();
    let n4 = b.c3(cat, 256, 1, false);
    let x = b.conv(n4, 256, 3, 2, true);
    let cat = b.g.concat(&[x, h10]).unwrap();
    let n5 = b.c3(cat, 512, 1, false);
    for n in [n3, n4, n5] {
        let head = b.conv(n, 255, 1, 1, false);
        b.g.mark_output(head);
    }
    b.g
}
