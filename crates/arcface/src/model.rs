// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ArcFace IResNet-100 embedding graph.
//!
//! Composed from the SHARED conv-net blocks in `crates/vision` (`Conv`,
//! `BatchNorm`, `PReLU`) over the shared kernel-id seam
//! (`ConvKernelIds::resolve`, by NAME) - this crate adds no conv, no norm and no
//! activation of its own.
//!
//! SSA discipline (`AGENTS.md`): every stage writes a FRESH buffer, which is
//! also the activation cache the deferred backward reads. Nothing is computed in
//! place.
//!
//! # The network is BatchNorm-FOLDED
//!
//! The released ONNX graph has its BatchNorms folded into the preceding
//! convolutions, so nearly every conv here is `Norm::None` **with a bias**, and
//! the activation follows directly. Only three BatchNorms survive as real nodes
//! (each block's entry `bn1`, the final `bn2`, and `features` after `fc`).
//! Modelling this from its *torch* module definition would need weights that are
//! not in the file.

use std::sync::OnceLock;

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use vision::{
    Act, BatchNorm, BnNames, Conv, ConvKernelIds, ConvNames, ConvSpec, Ctx, Norm, PReLU, Shape,
};

use crate::config::ArcFaceConfig;
use onnx::walk::Tensors;

/// Every kernel the forward graph, the alignment warp and the trainer dispatch,
/// by name.
///
/// The blocks resolve their own ids from this list via
/// [`ConvKernelIds::resolve`], so the ORDER here is free - that is the whole
/// point of the name-resolved seam. The backward-only kernels (`*_dx`, `*_dw`,
/// `prelu_bwd*`, the loss head) are registered in ONE list rather than a
/// separate training one: `gpu_core::testgpu::dev` keys its device pool on the
/// slice's address, so a second list would build a second device in every test
/// binary that touches both.
pub const PIPELINES: &[(&str, &str)] = &[
    // conv (dense; every conv in the model has groups=1, dilation=1)
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("bias_grad", kernels::BIAS_GRAD),
    // batchnorm (bn1 / bn2 / features)
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    // PReLU: a LEARNED per-channel slope. Both backward variants - selecting
    // between them on `DeviceCaps::workgroup_reductions` is a correctness gate.
    ("prelu", kernels::PRELU),
    ("prelu_bwd", kernels::PRELU_BWD),
    ("prelu_bwd_wg", kernels::PRELU_BWD_WG),
    // the 5-point alignment warp (see `crate::align`)
    ("grid_sample", kernels::GRID_SAMPLE),
    // elementwise / linear
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_add", kernels::BIAS_ADD),
    // the additive-angular-margin head (see `crate::train`): row-L2-normalise
    // both the embedding and the class centres, cosine table, angular margin, CE.
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
    ("arcface_margin", kernels::ARCFACE_MARGIN),
    ("arcface_margin_bwd", kernels::ARCFACE_MARGIN_BWD),
    ("ce_value", kernels::CE_VALUE),
    // `ce_grad` recomputes the row softmax on EVERY output element -
    // O(rows*classes^2). `ce_stats`+`ce_grad_stats` precompute the per-row
    // max/sum once and reuse it - O(rows*classes) - the same migration
    // `gpt2`/`lfm2`/`qwen3` already made for their (much larger) vocab. Kept
    // alongside `ce_grad` rather than removing it: `ce_grad`'s contract
    // (unconditional division by `n_rows`, no ignore mask) is still the
    // simplest correct kernel for a caller with no masking need, and other
    // crates still dispatch it directly by name.
    ("ce_grad", kernels::CE_GRAD),
    ("ce_stats", kernels::CE_STATS),
    ("ce_grad_stats", kernels::CE_GRAD_STATS),
    // preprocessing: brain's one per-channel affine, dispatched via `imaging::Ctx`.
    ("film_chan", kernels::FILM_CHAN),
];

/// Kernel ids resolved once against [`PIPELINES`].
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

/// The pipeline index of a kernel this crate dispatches directly (i.e. one with
/// no block in `crates/vision` behind it: `matmul`, `bias_add`, `axpy`,
/// `grid_sample`). Resolved by NAME against [`PIPELINES`] for the same reason
/// [`ConvKernelIds`] is - a bare index means nothing outside the list that
/// declared it.
pub fn kernel(name: &str) -> usize {
    PIPELINES
        .iter()
        .position(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("kernel `{name}` is not in arcface::PIPELINES"))
}

/// Build a frozen (inference) ParamStore from an imported tensor map, checking
/// every tensor's element count against the manifest that produced it.
fn frozen_store(gpu: &Gpu, t: &Tensors) -> ParamStore {
    let mut init = std::collections::HashMap::with_capacity(t.len());
    let mut roles = Vec::with_capacity(t.len());
    for (name, (shape, data)) in t {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "tensor {name}: {} values for shape {shape:?}", data.len());
        roles.push((name.clone(), n, Role::Frozen));
        init.insert(name.clone(), data.clone());
    }
    roles.sort_by(|a, b| a.0.cmp(&b.0));
    ParamStore::new_with_roles(gpu, roles, &init)
}

/// A norm-free conv spec with a learned bias - the shape EVERY conv in the
/// folded graph takes.
pub(crate) fn folded(cout: u32, k: u32, stride: u32, pad: u32, act: Act) -> ConvSpec {
    ConvSpec { cout, k, stride, pad, groups: 1, dilation: 1, norm: Norm::None, act, bias: true }
}

/// One IResNet residual block:
/// `y = conv2(prelu(conv1(bn1(x)))) + (downsample(x) or x)`.
///
/// Note that the shortcut reads the block's **input**, not `bn1`'s output - a
/// pre-activation ("v3") residual. Feeding it `bn1(x)` instead still runs, still
/// produces plausible features, and is wrong; it is visible only against the
/// `s{n}b0_branch` golden.
pub(crate) struct ResBlock {
    pub(crate) bn1: BatchNorm,
    pub(crate) conv1: Conv,
    pub(crate) prelu: PReLU,
    pub(crate) conv2: Conv,
    pub(crate) downsample: Option<Conv>,
    out: DeviceBuffer,
    pub(crate) out_shape: Shape,
}

impl ResBlock {
    pub(crate) fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, stride: u32) -> ResBlock {
        let bn1 = BatchNorm::new(ctx, BnNames::torch(&format!("{prefix}.bn1")), in_shape, false);
        bn1.set_eval(true);
        let conv1 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv1"),
            ConvNames::torch_flat(&format!("{prefix}.conv1")),
            in_shape,
            folded(cout, 3, 1, 1, Act::None),
            false,
        );
        let prelu = PReLU::new(ctx, &format!("{prefix}.prelu"), conv1.out_shape);
        let conv2 = Conv::with_names(
            ctx,
            &format!("{prefix}.conv2"),
            ConvNames::torch_flat(&format!("{prefix}.conv2")),
            conv1.out_shape,
            folded(cout, 3, stride, 1, Act::None),
            false,
        );
        let downsample = if stride != 1 || in_shape.c != cout {
            Some(Conv::with_names(
                ctx,
                &format!("{prefix}.downsample"),
                ConvNames::torch_flat(&format!("{prefix}.downsample")),
                in_shape,
                folded(cout, 1, stride, 0, Act::None),
                false,
            ))
        } else {
            None
        };
        let out_shape = conv2.out_shape;
        ResBlock { bn1, conv1, prelu, conv2, downsample, out: ctx.act(out_shape.numel()), out_shape }
    }

    /// This block's output buffer - the next block's input, and therefore the
    /// activation the next block's backward needs as its `x`.
    pub(crate) fn out_ref(&self) -> &DeviceBuffer {
        &self.out
    }

    pub(crate) fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.bn1.param_list();
        v.extend(self.conv1.param_list());
        v.extend(self.prelu.param_list());
        v.extend(self.conv2.param_list());
        if let Some(d) = &self.downsample {
            v.extend(d.param_list());
        }
        v
    }

    pub(crate) fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) -> &DeviceBuffer {
        self.bn1.forward(ctx, ps, x);
        self.conv1.forward(ctx, ps, self.bn1.out());
        self.prelu.forward(ctx, ps, self.conv1.out());
        self.conv2.forward(ctx, ps, self.prelu.out());
        let ident: &DeviceBuffer = match &self.downsample {
            Some(d) => {
                d.forward(ctx, ps, x);
                d.out()
            }
            None => x,
        };
        let n = self.out_shape.numel();
        let s = ctx.step(ctx.ids.need(ctx.ids.add2, "add2"), &[self.conv2.out(), ident, &self.out], &[n], n);
        ctx.gpu.submit(&[], &[s]);
        &self.out
    }
}

/// Per-stage taps the parity test replays against. Named for the goldens
/// (`stem`, `layer1..4`, `bn2`, `flatten`, `fc`, `embedding`) so the test is a
/// pure lookup rather than a translation table.
#[derive(Clone, Debug, Default)]
pub struct ArcFaceTaps {
    pub stem: Vec<f32>,
    pub layers: [Vec<f32>; 4],
    /// Every residual `Add` output, in block order - the bisection ladder.
    pub blocks: Vec<Vec<f32>>,
    pub bn2: Vec<f32>,
    pub fc: Vec<f32>,
    /// The graph's raw 512-d output (the `features` BN). NOT L2-normalised.
    pub embedding: Vec<f32>,
    /// First-block internals of each stage, in `[bn_in, conv1, prelu, conv2,
    /// branch]` order - the goldens' `s{n}b0_*` taps.
    pub stage_b0: [[Vec<f32>; 5]; 4],
}

/// Inference-only ArcFace IResNet-100 embedder.
pub struct ArcFace {
    gpu: Gpu,
    cfg: ArcFaceConfig,
    ps: ParamStore,
    input: DeviceBuffer,
    stem_conv: Conv,
    stem_prelu: PReLU,
    blocks: Vec<ResBlock>,
    /// `(stage, index of the stage's LAST block in `blocks`)`.
    stage_last: [usize; 4],
    bn2: BatchNorm,
    fc_out: DeviceBuffer,
    features: BatchNorm,
}

impl ArcFace {
    /// Build on a shared device handle (`Gpu::share` / `testgpu::dev`) from an
    /// imported tensor map.
    pub fn new(gpu: Gpu, cfg: ArcFaceConfig, weights: &Tensors) -> ArcFace {
        let ids = ids();
        let ctx = Ctx::new(&gpu, ids);
        let sz = cfg.image_size;
        let in_shape = Shape::new(1, 3, sz, sz);

        let stem_conv = Conv::with_names(
            &ctx,
            "stem.conv",
            ConvNames::torch_flat("stem.conv"),
            in_shape,
            folded(cfg.stem_channels, 3, 1, 1, Act::None),
            false,
        );
        let stem_prelu = PReLU::new(&ctx, "stem.prelu", stem_conv.out_shape);

        let mut shape = stem_conv.out_shape;
        let mut blocks: Vec<ResBlock> = Vec::new();
        let mut stage_last = [0usize; 4];
        for (s, last) in stage_last.iter_mut().enumerate() {
            for b in 0..cfg.layers[s] as usize {
                let stride = if b == 0 { 2 } else { 1 };
                let blk = ResBlock::new(&ctx, &format!("layer{}.{}", s + 1, b), shape, cfg.channels[s], stride);
                shape = blk.out_shape;
                blocks.push(blk);
            }
            *last = blocks.len() - 1;
        }

        let bn2 = BatchNorm::new(&ctx, BnNames::torch("bn2"), shape, false);
        bn2.set_eval(true);
        let e = cfg.embedding;
        let fc_out = ctx.act(e);
        // The `features` BN runs over a flat [1, 512] vector: N=1, C=512, H=W=1.
        let features = BatchNorm::new(&ctx, BnNames::torch("features"), Shape::new(1, e, 1, 1), false);
        features.set_eval(true);

        let mut params: Vec<(String, usize)> = stem_conv.param_list();
        params.extend(stem_prelu.param_list());
        for b in &blocks {
            params.extend(b.param_list());
        }
        params.extend(bn2.param_list());
        params.push(("fc.weight".into(), (e * cfg.flatten()) as usize));
        params.push(("fc.bias".into(), e as usize));
        params.extend(features.param_list());

        // The graph's own parameter list vs the import manifest: two independently
        // derived lists of the same 462 tensors. Comparing them here catches a
        // block wired at the wrong width before a single kernel runs.
        assert_eq!(
            params.len(),
            weights.len(),
            "arcface: the graph reads {} tensors but the checkpoint has {}",
            params.len(),
            weights.len()
        );
        for (name, n) in &params {
            match weights.get(name) {
                None => panic!("arcface: checkpoint is missing {name}"),
                Some((shape, _)) => {
                    let have: usize = shape.iter().product();
                    assert_eq!(have, *n, "arcface: {name} has {have} values, the graph wants {n}");
                }
            }
        }

        let ps = frozen_store(&gpu, weights);
        let input = gpu.storage(in_shape.numel() as u64);
        ArcFace { gpu, cfg, ps, input, stem_conv, stem_prelu, blocks, stage_last, bn2, fc_out, features }
    }

    pub fn config(&self) -> &ArcFaceConfig {
        &self.cfg
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx::new(&self.gpu, ids())
    }

    /// Run the network on an NCHW `[1, 3, 112, 112]` blob and return the raw
    /// 512-d embedding (not normalised - the graph has no normalisation).
    ///
    /// Captures no taps: reading them back is ~1.9 M floats and 60 device syncs
    /// per image, and computing them only to drop them would put the parity
    /// ladder's instrumentation in the production path.
    pub fn embed_blob(&self, blob: &[f32]) -> Vec<f32> {
        self.run(blob, false).embedding
    }

    /// Forward with every stage tap captured - what the parity test replays.
    pub fn forward(&self, blob: &[f32]) -> ArcFaceTaps {
        self.run(blob, true)
    }

    /// The graph. `capture` gates ONLY the host readbacks, never a dispatch, so
    /// the two paths compute bit-identical values.
    fn run(&self, blob: &[f32], capture: bool) -> ArcFaceTaps {
        let ctx = self.ctx();
        let want = (3 * self.cfg.image_size * self.cfg.image_size) as usize;
        assert_eq!(blob.len(), want, "arcface: blob must be [3,{0},{0}]", self.cfg.image_size);
        self.gpu.write(&self.input, bytemuck::cast_slice(blob));

        self.stem_conv.forward(&ctx, &self.ps, &self.input);
        self.stem_prelu.forward(&ctx, &self.ps, self.stem_conv.out());

        let mut taps = ArcFaceTaps::default();
        if capture {
            taps.stem = self.read(self.stem_prelu.out(), self.stem_conv.out_shape.numel());
        }

        let mut x: &DeviceBuffer = self.stem_prelu.out();
        let mut stage = 0usize;
        let mut first_of_stage = true;
        for (i, blk) in self.blocks.iter().enumerate() {
            let y = blk.forward(&ctx, &self.ps, x);
            if capture && first_of_stage {
                taps.stage_b0[stage] = [
                    self.read(blk.bn1.out(), blk.bn1.shape.numel()),
                    self.read(blk.conv1.out(), blk.conv1.out_shape.numel()),
                    self.read(blk.prelu.out(), blk.prelu.shape.numel()),
                    self.read(blk.conv2.out(), blk.conv2.out_shape.numel()),
                    // Stage-first blocks always stride, so the shortcut conv
                    // always exists; `None` cannot occur here and an empty tap
                    // would be silently skipped by the parity test.
                    match &blk.downsample {
                        Some(d) => self.read(d.out(), d.out_shape.numel()),
                        None => panic!("layer{}.0 has no downsample conv", stage + 1),
                    },
                ];
            }
            first_of_stage = false;
            if capture {
                taps.blocks.push(self.read(y, blk.out_shape.numel()));
            }
            if i == self.stage_last[stage] {
                if capture {
                    taps.layers[stage] = taps.blocks.last().cloned().unwrap_or_default();
                }
                stage = (stage + 1).min(3);
                first_of_stage = true;
            }
            x = y;
        }

        self.bn2.forward(&ctx, &self.ps, x);
        if capture {
            taps.bn2 = self.read(self.bn2.out(), self.bn2.shape.numel());
        }

        // Flatten is a no-op: `bn2`'s NCHW buffer IS the [1, 25088] row.
        // fc: `matmul` is out = x · Wᵀ with x [m,k], W [n,k] - exactly the ONNX
        // Gemm's transB=1 layout. Params [m, k, n]; one invocation per output.
        let (e, fl) = (self.cfg.embedding, self.cfg.flatten());
        let s_mm = ctx.step(
            kernel("matmul"),
            &[self.bn2.out(), self.ps.w("fc.weight"), &self.fc_out],
            &[1, fl, e],
            e,
        );
        // `bias_add` is the [M, N] LINEAR bias (`out[idx] += bias[idx % N]`), which
        // is right for a flat [1, 512] row - NOT the NCHW `add_chan_inplace`.
        let s_b = ctx.step(kernel("bias_add"), &[&self.fc_out, self.ps.w("fc.bias")], &[1, e], e);
        ctx.gpu.submit(&[], &[s_mm, s_b]);
        if capture {
            taps.fc = self.read(&self.fc_out, e);
        }

        // The embedding is the RESULT, not a tap - always read.
        self.features.forward(&ctx, &self.ps, &self.fc_out);
        taps.embedding = self.read(self.features.out(), e);
        taps
    }

    fn read(&self, b: &DeviceBuffer, n: u32) -> Vec<f32> {
        self.gpu.read(b, n as usize)
    }
}

// The embedding's consumer-side math - L2-normalisation and cosine - is
// `model::hostmath::{l2_normalize, cosine}`. It is not face-specific (the ECAPA
// speaker embedding wants the identical pair), so it lives in host math's one
// home and is called directly; a local wrapper here is how a shared function
// becomes a private copy at the next edit (`AGENTS.md`).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_lookup_is_by_name_and_panics_on_an_unregistered_one() {
        assert_eq!(PIPELINES[kernel("matmul")].0, "matmul");
        assert_eq!(PIPELINES[kernel("grid_sample")].0, "grid_sample");
        assert!(std::panic::catch_unwind(|| kernel("no_such_kernel")).is_err());
    }

    /// Every kernel the shared blocks will look up must actually be registered,
    /// or the first forward panics at a dispatch instead of at build time.
    #[test]
    fn every_kernel_the_blocks_need_is_registered() {
        let i = ids();
        for (id, name) in [
            (i.conv_bias_reg, "conv_bias_reg"),
            (i.conv2d_dx, "conv2d_dx"),
            (i.conv2d_dw, "conv2d_dw"),
            (i.bn_eval, "bn_eval"),
            (i.bn_train, "bn_train"),
            (i.prelu, "prelu"),
            (i.prelu_bwd, "prelu_bwd"),
            (i.prelu_bwd_wg, "prelu_bwd_wg"),
            (i.add2, "add2"),
        ] {
            assert_ne!(id, vision::NONE, "{name} missing from arcface::PIPELINES");
        }
    }

    /// A SHRUNKEN embedder of the same shape, built from manifest-derived
    /// weights and run end to end.
    ///
    /// The released 112² graph needs a 261 MB checkpoint that is not in the
    /// repo, so without this the first thing to dispatch the EVAL path's kernels
    /// (eval-mode BatchNorm, `matmul`, `bias_add`) would be the skippable parity
    /// test - the trainer's own tests cover the TRAIN path only. A kernel
    /// missing from [`PIPELINES`] panics here by NAME.
    #[test]
    fn the_whole_graph_runs_with_every_kernel_it_dispatches_registered() {
        let cfg = ArcFaceConfig {
            image_size: 32,
            layers: [2, 1, 1, 1],
            channels: [4, 6, 6, 8],
            stem_channels: 4,
            embedding: 8,
            ..ArcFaceConfig::iresnet100()
        };
        let weights: Tensors = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, shape)| {
                let k: usize = shape.iter().product();
                // BN gamma at 1 and running_var at 1: a zero variance is a
                // division by sqrt(eps) and a zero gamma erases the signal, and
                // either would make a wrongly-wired graph look the same as a
                // right one.
                let v = if n.ends_with(".running_var") || n.ends_with("bn1.weight") || n == "bn2.weight" || n == "features.weight" {
                    1.0
                } else if n.ends_with(".running_mean") {
                    0.0
                } else {
                    0.05
                };
                (n, (shape, vec![v; k]))
            })
            .collect();

        let gpu = gpu_core::testgpu::dev(PIPELINES);
        let m = ArcFace::new(gpu, cfg.clone(), &weights);
        let t = m.forward(&vec![0.5f32; (3 * cfg.image_size * cfg.image_size) as usize]);

        assert_eq!(t.embedding.len(), cfg.embedding as usize);
        assert_eq!(t.blocks.len(), cfg.n_blocks() as usize);
        assert_eq!(t.fc.len(), cfg.embedding as usize);
        assert!(t.embedding.iter().all(|v| v.is_finite()), "{:?}", t.embedding);
        assert!(t.embedding.iter().any(|v| v.abs() > 0.0), "the embedding is all zero");
        // `embed_blob` is the same graph without the readbacks.
        let e = m.embed_blob(&vec![0.5f32; (3 * cfg.image_size * cfg.image_size) as usize]);
        assert_eq!(e, t.embedding);
    }
}
