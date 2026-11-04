// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P2 per-block gradient-check gate.
//!
//! For each YOLOv8 conv block ({Conv(s1,K3), Conv(s2,K3), Conv(K1),
//! Bottleneck(shortcut on/off), C2f, SPPF}) and head branch ({cls, reg}) we
//! build a tiny `CheckModel` harness whose:
//!   * parameters  = the block's params (so they are perturbed),
//!   * forward     = run the block on a fixed small random NCHW input, then a
//!     PROXY scalar loss `L = <r, out>` for a fixed random vector `r` (the same
//!     `L = <dy, y>` trick the P1 kernel micro-checks used — it exercises the
//!     whole block's forward+backward), and
//!   * backward    = seed the output-grad buffer with `r` and run the block's
//!     reverse steps, leaving param grads in the ParamStore.
//!
//! BatchNorm runs in TRAIN mode with batch N=4 (well-conditioned batch stats).
//! Each block asserts `directional_check(&h, 5e-3, 3, seed).all_within(4e-3,
//! 8e-2)` — the same combined tolerance the other models' gradchecks use — plus
//! a forward-output SHAPE assertion. No GPU: `Gpu::new_cpu`.

use std::cell::RefCell;

use gpu_core::{DeviceBuffer, Gpu};
use gradcheck::{directional_check, CheckModel};
use paramstore::ParamStore;
use yolov8::blocks::{Bottleneck, Conv, C2f, SPPF};
use yolov8::head::Branch;
use yolov8::net::{Ctx, Shape, PIPELINES};

// ---------------------------------------------------------------------------
// Generic harness: owns a block, a fixed input, and a fixed proxy vector r.
// ---------------------------------------------------------------------------

/// A block the harness can drive: forward writes its output buffer, backward
/// consumes an output-grad and writes an input-grad + accumulates param grads.
trait TestBlock {
    fn param_list(&self) -> Vec<(String, usize)>;
    fn out_numel(&self) -> u32;
    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer);
    fn out(&self) -> &DeviceBuffer;
    fn backward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer);
}

struct Harness {
    gpu: Gpu,
    ps: ParamStore,
    block: Box<dyn TestBlock>,
    x: DeviceBuffer,
    d_in: DeviceBuffer,
    d_out: DeviceBuffer,
    r: Vec<f32>,
    fwd_done: RefCell<bool>,
}

impl Harness {
    fn new(block: Box<dyn TestBlock>, in_shape: Shape, init_std: f32, seed: u64, gpu: Gpu) -> Harness {
        let params = block.param_list();
        let init = yolov8::init::init_params(&params, seed, init_std);
        let ps = ParamStore::new(&gpu, params, &init);

        // Fixed random input.
        let xin = randvec(seed ^ 0xA5A5, in_shape.numel() as usize);
        let x = gpu.storage_init("x", &xin);
        let d_in = gpu.storage(in_shape.numel() as u64);
        let d_out = gpu.storage(block.out_numel() as u64);
        let r = randvec(seed ^ 0x5A5A, block.out_numel() as usize);
        Harness { gpu, ps, block, x, d_in, d_out, r, fwd_done: RefCell::new(false) }
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, yolov8::net::ids())
    }
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.ps.w(name), bytemuck::cast_slice(data));
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    fn loss(&self) -> f32 {
        let ctx = self.ctx();
        self.block.forward(&ctx, &self.ps, &self.x);
        *self.fwd_done.borrow_mut() = true;
        let out = self.gpu.read(self.block.out(), self.block.out_numel() as usize);
        out.iter().zip(&self.r).map(|(o, r)| o * r).sum()
    }
    fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    fn backward(&self) {
        // Forward must have run so activation caches are populated.
        if !*self.fwd_done.borrow() {
            let _ = self.loss();
        }
        // Seed d_out with r.
        self.gpu.write(&self.d_out, bytemuck::cast_slice(&self.r));
        let ctx = self.ctx();
        self.block.backward(&ctx, &self.ps, &self.x, &self.d_out, &self.d_in);
    }
}

// Deterministic LCG -> values in (-1, 1).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1) | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.next_f32()).collect()
}

fn gpu() -> Gpu {
    Gpu::new_cpu(PIPELINES)
}

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

fn assert_grads(h: &Harness, seed: u64, label: &str) {
    let report = directional_check(h, 5e-3, 3, seed ^ 0x1234);
    let fails = report.failures(ATOL, RTOL);
    if !fails.is_empty() {
        report.print();
    }
    assert!(
        fails.is_empty(),
        "[{label}] gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// TestBlock impls (thin adapters over the block methods).
// ---------------------------------------------------------------------------

macro_rules! impl_block {
    ($ty:ty) => {
        impl TestBlock for $ty {
            fn param_list(&self) -> Vec<(String, usize)> {
                <$ty>::param_list(self)
            }
            fn out_numel(&self) -> u32 {
                self.out_shape.numel()
            }
            fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
                <$ty>::forward(self, ctx, ps, x)
            }
            fn out(&self) -> &DeviceBuffer {
                <$ty>::out(self)
            }
            fn backward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
                <$ty>::backward(self, ctx, ps, x, d_out, d_in)
            }
        }
    };
}
impl_block!(Conv);
impl_block!(Bottleneck);
impl_block!(C2f);
impl_block!(SPPF);
impl_block!(Branch);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const N: u32 = 4; // batch >= 4 for well-conditioned BN batch stats

#[test]
fn conv_stride1_k3() {
    let g = gpu();
    let in_shape = Shape::new(N, 3, 8, 8);
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Conv::new(&ctx, "conv", in_shape, 5, 3, 1, 1, true))
    };
    assert_eq!(block.out_shape, Shape::new(N, 5, 8, 8));
    let h = Harness::new(block, in_shape, 0.3, 101, g);
    assert_grads(&h, 101, "conv_s1_k3");
}

#[test]
fn conv_stride2_k3() {
    let g = gpu();
    let in_shape = Shape::new(N, 3, 8, 8);
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Conv::new(&ctx, "conv", in_shape, 5, 3, 2, 1, true))
    };
    assert_eq!(block.out_shape, Shape::new(N, 5, 4, 4));
    let h = Harness::new(block, in_shape, 0.3, 202, g);
    assert_grads(&h, 202, "conv_s2_k3");
}

#[test]
fn conv_k1() {
    let g = gpu();
    let in_shape = Shape::new(N, 4, 6, 6);
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Conv::new(&ctx, "conv", in_shape, 6, 1, 1, 0, true))
    };
    assert_eq!(block.out_shape, Shape::new(N, 6, 6, 6));
    let h = Harness::new(block, in_shape, 0.3, 303, g);
    assert_grads(&h, 303, "conv_k1");
}

#[test]
fn bottleneck_shortcut_on() {
    let g = gpu();
    let in_shape = Shape::new(N, 6, 6, 6);
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Bottleneck::new(&ctx, "b", in_shape, 6, true, true)) // c_in==c_out -> residual
    };
    assert!(block.shortcut, "shortcut should be active for c_in==c_out");
    assert_eq!(block.out_shape, Shape::new(N, 6, 6, 6));
    let h = Harness::new(block, in_shape, 0.25, 404, g);
    assert_grads(&h, 404, "bottleneck_shortcut_on");
}

#[test]
fn bottleneck_shortcut_off() {
    let g = gpu();
    let in_shape = Shape::new(N, 4, 6, 6);
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Bottleneck::new(&ctx, "b", in_shape, 6, false, true)) // c_in!=c_out anyway
    };
    assert!(!block.shortcut);
    assert_eq!(block.out_shape, Shape::new(N, 6, 6, 6));
    let h = Harness::new(block, in_shape, 0.25, 505, g);
    assert_grads(&h, 505, "bottleneck_shortcut_off");
}

#[test]
fn c2f_block() {
    let g = gpu();
    let in_shape = Shape::new(N, 6, 6, 6);
    // C_out=8 (c=4), n=2 bottlenecks, shortcut on (c_in==c_out==4 inside).
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(C2f::new(&ctx, "c2f", in_shape, 8, 2, true, true))
    };
    assert_eq!(block.out_shape, Shape::new(N, 8, 6, 6));
    let h = Harness::new(block, in_shape, 0.25, 606, g);
    assert_grads(&h, 606, "c2f");
}

#[test]
fn sppf_block() {
    let g = gpu();
    let in_shape = Shape::new(N, 8, 6, 6);
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(SPPF::new(&ctx, "sppf", in_shape, 8, true)) // c=4 inner, 4c=16 concat
    };
    assert_eq!(block.out_shape, Shape::new(N, 8, 6, 6));
    let h = Harness::new(block, in_shape, 0.25, 707, g);
    assert_grads(&h, 707, "sppf");
}

#[test]
fn head_cls_branch() {
    let g = gpu();
    let in_shape = Shape::new(N, 8, 5, 5);
    let nc = 3;
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Branch::new(&ctx, "head.0.cls", in_shape, 6, nc, true))
    };
    assert_eq!(block.out_shape, Shape::new(N, nc, 5, 5));
    let h = Harness::new(block, in_shape, 0.25, 808, g);
    assert_grads(&h, 808, "head_cls");
}

#[test]
fn head_reg_branch() {
    let g = gpu();
    let in_shape = Shape::new(N, 8, 5, 5);
    let reg_max = 4;
    let block = {
        let ctx = Ctx::new(&g, yolov8::net::ids());
        Box::new(Branch::new(&ctx, "head.0.reg", in_shape, 6, 4 * reg_max, true))
    };
    assert_eq!(block.out_shape, Shape::new(N, 4 * reg_max, 5, 5));
    let h = Harness::new(block, in_shape, 0.25, 909, g);
    assert_grads(&h, 909, "head_reg");
}
