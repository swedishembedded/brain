// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Camera head (`camera_head.py` parity): iterative refinement of a 9-vector
//! `[t(3), quat_xyzw(4), fov_v, fov_u]` per frame from the cam token (row 0)
//! of the last trunk tap.
//!
//! 4 recorded iterations, each: param-embed(prev raw pred) → SiLU →
//! adaptive-LN modulation (`gate·(LNnoaffine(x)·(1+scale)+shift) + x`) →
//! 4 plain 2048-dim ViT blocks over the S frame tokens → out-norm →
//! MLP(2048→1024→9, GELU-erf) → delta accumulated into the RAW prediction
//! (activations - relu on fov - are applied host-side on readback).

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_fwd, VitBlockWeights, VitKernelIds, VitScratch, VitShape};
use paramstore::ParamStore;

/// Kernel ids beyond [`VitKernelIds`] the camera head needs.
#[derive(Clone, Copy)]
pub struct CamKernels {
    pub vit: VitKernelIds,
    pub silu: usize,
    pub mul: usize,
    pub axpy: usize,
}

pub struct CamBufs {
    pub cam_tok: DeviceBuffer, // [S, 2048] raw gathered tokens
    pub cam_n: DeviceBuffer,   // token_norm output
    pub xn: DeviceBuffer,      // adapt_norm output
    pub net_in: DeviceBuffer,  // param embed [S, 2048]
    pub sil: DeviceBuffer,
    pub shift: DeviceBuffer,
    pub scale: DeviceBuffer,
    pub gate: DeviceBuffer,
    pub m: [DeviceBuffer; 3],  // modulation temps
    pub x: DeviceBuffer,       // refine-net working tokens
    pub on: DeviceBuffer,      // out_norm output
    pub h1: DeviceBuffer,      // [S, 1024]
    pub h1g: DeviceBuffer,
    pub delta: DeviceBuffer,   // [S, 9]
    pub pred: DeviceBuffer,    // [S, 9] RAW accumulated prediction
    pub init9: DeviceBuffer,   // [S, 9] init_token tiled
    pub ones: DeviceBuffer,    // [2048] (no-affine LN gamma)
    pub zerosc: DeviceBuffer,  // [2048] (no-affine LN beta)
    pub scr: VitScratch,
}

impl CamBufs {
    /// `sh` is the refine-net block shape - it MUST be [`cam_shape`] for the
    /// same config the weights were laid out for, or the block's MLP dispatch
    /// runs past the end of `mlp.fc1.weight`.
    pub fn new(gpu: &Gpu, s: usize, sh: &VitShape, init_token: &[f32]) -> CamBufs {
        let dim2 = sh.dim as usize;
        let sd = (s * dim2) as u64;
        let mut init9 = Vec::with_capacity(s * 9);
        for _ in 0..s {
            init9.extend_from_slice(init_token);
        }
        CamBufs {
            cam_tok: gpu.storage(sd),
            cam_n: gpu.storage(sd),
            xn: gpu.storage(sd),
            net_in: gpu.storage(sd),
            sil: gpu.storage(sd),
            shift: gpu.storage(sd),
            scale: gpu.storage(sd),
            gate: gpu.storage(sd),
            m: [gpu.storage(sd), gpu.storage(sd), gpu.storage(sd)],
            x: gpu.storage(sd),
            on: gpu.storage(sd),
            h1: gpu.storage((s * dim2 / 2) as u64),
            h1g: gpu.storage((s * dim2 / 2) as u64),
            delta: gpu.storage((s * 9) as u64),
            pred: gpu.storage((s * 9) as u64),
            init9: gpu.storage_init("cam.init9", &init9),
            ones: gpu.storage_init("cam.ones", &vec![1.0f32; dim2]),
            zerosc: gpu.storage_init("cam.zeros", &vec![0.0f32; dim2]),
            scr: VitScratch::new(gpu, sh, s as u32, s as u32, s as u32),
        }
    }
}

/// The refine-net block shape for a config: width `2*dim` (the frame‖global
/// tap), the config's own head count and MLP ratio.
///
/// It has to be derived rather than assumed, because it is also what
/// `MirrorConfig::param_list` sizes `cam_head.refine_net.*` from - the two read
/// the SAME numbers or the block dispatches past the end of its own weights.
pub fn cam_shape(dim: u32, heads: u32, mlp_ratio: u32) -> VitShape {
    let d2 = 2 * dim;
    VitShape { dim: d2, heads, mlp: mlp_ratio * d2, eps: 1e-5 }
}

/// Weight accessor bound to the `cam_head.` prefix.
pub struct CamWeights<'a> {
    pub ps: &'a ParamStore,
}

impl<'a> CamWeights<'a> {
    fn get(&self, name: &str) -> &'a DeviceBuffer {
        self.ps.w(&format!("cam_head.{name}"))
    }
}

/// Record the full camera head. `last_tap` = `[s*td, 2C]`; `pred` must be in
/// the submit clears list (deltas accumulate with axpy). `sh` is the refine-net
/// block shape - [`cam_shape`] for the config the weights came from.
#[allow(clippy::too_many_arguments)]
pub fn record_cam_head(
    gpu: &Gpu,
    k: &CamKernels,
    cw: &CamWeights,
    b: &CamBufs,
    last_tap: &DeviceBuffer,
    s: usize,
    td: usize,
    sh: &VitShape,
    iters: usize,
    blocks: usize,
    steps: &mut Vec<Step>,
) {
    let su = s as u32;
    let d2 = sh.dim;
    let dim2 = d2 as usize;
    let n2 = su * d2;
    // gather cam tokens (row 0 of each frame) - cam_tok is in the clears list
    for fi in 0..s {
        steps.push(gpu.step_sliced(
            k.axpy,
            &[&b.cam_tok, last_tap],
            &[((fi * dim2) as u64, 0), ((fi * td * dim2) as u64, dim2 as u64)],
            &[d2, f(1.0)],
            d2,
        ));
    }
    steps.push(gpu.step(
        k.vit.layernorm,
        &[&b.cam_tok, cw.get("token_norm.weight"), cw.get("token_norm.bias"), &b.cam_n],
        &[d2, su, f(1e-5)],
        su,
    ));
    // adapt_norm: LayerNorm without affine, eps 1e-6 (constant once - the
    // input cam_n never changes across iterations).
    steps.push(gpu.step(
        k.vit.layernorm,
        &[&b.cam_n, &b.ones, &b.zerosc, &b.xn],
        &[d2, su, f(1e-6)],
        su,
    ));

    for it in 0..iters {
        // net_in = param_embed(init | prev raw pred)
        let src9 = if it == 0 { &b.init9 } else { &b.pred };
        steps.push(gpu.step(
            k.vit.matmul,
            &[src9, cw.get("param_embed.weight"), &b.net_in],
            &[su, 9, d2],
            n2,
        ));
        steps.push(gpu.step(k.vit.bias_add, &[&b.net_in, cw.get("param_embed.bias")], &[su, d2], n2));
        steps.push(gpu.step(k.silu, &[&b.net_in, &b.sil], &[n2], n2));
        // shift/scale/gate: one 2048→6144 linear, dispatched as 3 sliced matmuls
        let agw = cw.get("adapt_norm_gen.1.weight");
        let agb = cw.get("adapt_norm_gen.1.bias");
        for (i, out) in [(0u64, &b.shift), (1, &b.scale), (2, &b.gate)] {
            steps.push(gpu.step_sliced(
                k.vit.matmul,
                &[&b.sil, agw, out],
                &[(0, 0), (i * dim2 as u64 * dim2 as u64, (dim2 * dim2) as u64), (0, 0)],
                &[su, d2, d2],
                n2,
            ));
            steps.push(gpu.step_sliced(
                k.vit.bias_add,
                &[out, agb],
                &[(0, 0), (i * dim2 as u64, dim2 as u64)],
                &[su, d2],
                n2,
            ));
        }
        // mod = gate*(xn*(1+scale)+shift) + cam_n
        steps.push(gpu.step(k.mul, &[&b.xn, &b.scale, &b.m[0]], &[n2], n2));
        steps.push(gpu.step(k.vit.add2, &[&b.m[0], &b.xn, &b.m[1]], &[n2], n2));
        steps.push(gpu.step(k.vit.add2, &[&b.m[1], &b.shift, &b.m[2]], &[n2], n2));
        steps.push(gpu.step(k.mul, &[&b.m[2], &b.gate, &b.m[0]], &[n2], n2));
        steps.push(gpu.step(k.vit.add2, &[&b.m[0], &b.cam_n, &b.x], &[n2], n2));
        // cfg.cam_blocks plain 2048-dim blocks over the S frame tokens
        for blk in 0..blocks {
            let p = |n: &str| format!("refine_net.{blk}.{n}");
            let w = VitBlockWeights {
                norm1_w: cw.get(&p("norm1.weight")),
                norm1_b: cw.get(&p("norm1.bias")),
                qkv_w: cw.get(&p("attn.qkv.weight")),
                qkv_b: cw.get(&p("attn.qkv.bias")),
                qk_norm: None,
                rope: None,
                proj_w: cw.get(&p("attn.proj.weight")),
                proj_b: cw.get(&p("attn.proj.bias")),
                ls1: Some(cw.get(&p("ls1.gamma"))),
                norm2_w: cw.get(&p("norm2.weight")),
                norm2_b: cw.get(&p("norm2.bias")),
                fc1_w: cw.get(&p("mlp.fc1.weight")),
                fc1_b: cw.get(&p("mlp.fc1.bias")),
                fc2_w: cw.get(&p("mlp.fc2.weight")),
                fc2_b: cw.get(&p("mlp.fc2.bias")),
                ls2: Some(cw.get(&p("ls2.gamma"))),
            };
            vit_block_fwd(gpu, &k.vit, sh, &w, &b.x, su, &[(0, su)], su, &b.scr, steps);
        }
        // delta = fc2(gelu(fc1(out_norm(x))))
        steps.push(gpu.step(
            k.vit.layernorm,
            &[&b.x, cw.get("out_norm.weight"), cw.get("out_norm.bias"), &b.on],
            &[d2, su, f(1e-5)],
            su,
        ));
        steps.push(gpu.step(
            k.vit.matmul,
            &[&b.on, cw.get("param_predictor.fc1.weight"), &b.h1],
            &[su, d2, d2 / 2],
            su * d2 / 2,
        ));
        steps.push(gpu.step(k.vit.bias_add, &[&b.h1, cw.get("param_predictor.fc1.bias")], &[su, d2 / 2], su * d2 / 2));
        steps.push(gpu.step(k.vit.mlp_act, &[&b.h1, &b.h1g], &[su * d2 / 2], su * d2 / 2));
        steps.push(gpu.step(
            k.vit.matmul,
            &[&b.h1g, cw.get("param_predictor.fc2.weight"), &b.delta],
            &[su, d2 / 2, 9],
            su * 9,
        ));
        steps.push(gpu.step(k.vit.bias_add, &[&b.delta, cw.get("param_predictor.fc2.bias")], &[su, 9], su * 9));
        steps.push(gpu.step(k.axpy, &[&b.pred, &b.delta], &[su * 9, f(1.0)], su * 9));
    }
}
