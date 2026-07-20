// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The WorldMirror-2 model on the brain engine.
//!
//! P1 scope: per-frame DINOv2 ViT-L/14-reg encoding (frames → 1369 normed
//! patch tokens each), built on `model::vit`. The trunk (P2) and heads (P3)
//! extend this file's `Mirror` with further recorded stages on the same
//! `ParamStore`/scratch.
//!
//! Follows the depth `Predictor` pattern: borrows the `Gpu`, holds the frozen
//! `ParamStore`, lazily (re)builds buffers when the input shape changes, and
//! records the whole forward as `Step`s.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_fwd, VitBlockWeights, VitKernelIds, VitScratch, VitShape};
use paramstore::{ParamStore, Role};

use crate::cam::{record_cam_head, CamBufs, CamKernels, CamWeights};
use crate::config::MirrorConfig;
use crate::dpt::{DptCtx, DptKernels, DptScratch, GsBranch, HeadWeights};
use crate::preprocess::{IMAGENET_MEAN, IMAGENET_STD};

// ---- kernel indices (order matches PIPELINES) ----
const K_LAYERNORM: usize = 0;
const K_MATMUL: usize = 1;
const K_BIAS_ADD: usize = 2;
const K_GELU_ERF: usize = 3;
const K_SCALE_CHAN: usize = 4;
const K_ADD2: usize = 5;
const K_ATTN_SCORES_CROSS: usize = 6;
const K_ATTN_SOFTMAX_CROSS: usize = 7;
const K_ATTN_APPLY_CROSS: usize = 8;
const K_LN_HEAD: usize = 9;
const K_ROPE2D: usize = 10;
const K_CONV2D: usize = 11;
const K_ADD_CHAN_BCAST: usize = 12;
const K_NCHW_NLC: usize = 13;
const K_AXPY: usize = 14;
const K_CONCAT2: usize = 15;
const K_CONV2D_DX: usize = 16;
const K_ADD_CHAN_INPLACE: usize = 17;
const K_RESIZE_BILINEAR: usize = 18;
const K_LEAKY_RELU: usize = 19;
const K_RELU_INPLACE: usize = 20;
const K_SILU: usize = 21;
const K_MUL: usize = 22;
const K_NLC_NCHW: usize = 23;
const K_ADD_INPLACE: usize = 24;

pub const PIPELINES: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add2", kernels::ADD2),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("ln_head", kernels::LN_HEAD),
    ("rope2d", kernels::ROPE2D),
    ("conv2d", kernels::CONV2D),
    ("add_chan_bcast", kernels::ADD_CHAN_BCAST),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("axpy", kernels::AXPY),
    ("concat2", kernels::CONCAT2),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("silu", kernels::SILU),
    ("mul", kernels::MUL),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("add_inplace", kernels::ADD_INPLACE),
];

fn vit_ids(base: usize) -> VitKernelIds {
    VitKernelIds {
        layernorm: base + K_LAYERNORM,
        matmul: base + K_MATMUL,
        bias_add: base + K_BIAS_ADD,
        gelu_erf: base + K_GELU_ERF,
        scale_chan: base + K_SCALE_CHAN,
        add2: base + K_ADD2,
        attn_scores_cross: base + K_ATTN_SCORES_CROSS,
        attn_softmax_cross: base + K_ATTN_SOFTMAX_CROSS,
        attn_apply_cross: base + K_ATTN_APPLY_CROSS,
        ln_head: base + K_LN_HEAD,
        rope2d: base + K_ROPE2D,
    }
}

const VGT: &str = "visual_geometry_transformer";
/// LayerNorm eps: DINOv2 uses 1e-6 (partial(nn.LayerNorm, eps=1e-6)); the
/// trunk blocks use torch's default 1e-5.
const DINO_EPS: f32 = 1e-6;
const TRUNK_EPS: f32 = 1e-5;
/// Per-frame special tokens: cam(1) + reg(4) + pose(1) + ray(1); pose/ray are
/// ZERO in the no-prior path but the layout keeps them (patch_start_idx = 7).
pub const PATCH_START: usize = 7;
/// Global-attention score-slab budget (auto-chunks queries under it).
const ATTN_BUDGET: u64 = 1 << 30;

/// Per-shape buffers + the recorded forward (DINOv2 encode + trunk).
struct Built {
    s: usize,
    hp: usize,
    wp: usize,
    frames_in: DeviceBuffer,
    #[allow(dead_code)]
    conv_raw: DeviceBuffer,
    #[allow(dead_code)]
    conv_out: DeviceBuffer,
    tokens: DeviceBuffer,
    scr: VitScratch,
    patch_out: DeviceBuffer,
    trunk_tokens: DeviceBuffer,
    zeros: DeviceBuffer,
    /// Frame‖global concat at the 4 tap levels, each `[s*td_trunk, 2*C]`.
    taps: Vec<DeviceBuffer>,
    /// Per-frame raw [0,1] CHW frames (GS input-merger input).
    rgb_frames: Vec<DeviceBuffer>,
    /// Per-frame head outputs, pre-activation NCHW at full res.
    depth_out: Vec<DeviceBuffer>,    // [3,H,W]: depth, conf, mask
    pts_out: Vec<DeviceBuffer>,      // [4,H,W]
    norm_out: Vec<DeviceBuffer>,     // [4,H,W]
    gs_depth_out: Vec<DeviceBuffer>, // [3,H,W]
    gs_params: Vec<DeviceBuffer>,    // [12,H,W]
    #[allow(dead_code)]
    dscr: DptScratch,
    cam: CamBufs,
    steps: Vec<Step>,
}

pub struct Mirror<'g> {
    gpu: &'g Gpu,
    pub cfg: MirrorConfig,
    pub ps: ParamStore,
    base: usize,
    /// Host copies of the tiny token/pos constants (assembled on the host).
    head_rows: Vec<f32>, // [1+reg, C]: cls+pos[0], registers (no pos)
    pos_patch: Vec<f32>, // [1369, C]: pos_embed[1..]
    /// Trunk special-token rows [PATCH_START, C] for frame 0 / frames 1+
    /// (cam+reg variants; pose/ray rows zero).
    trunk_head_f0: Vec<f32>,
    trunk_head_fr: Vec<f32>,
    rope_periods: Vec<f32>,
    cam_init9: Vec<f32>,
    built: Option<Built>,
}

impl<'g> Mirror<'g> {
    /// `base` = offset of [`PIPELINES`] inside the `Gpu`'s kernel list.
    pub fn new(
        gpu: &'g Gpu,
        cfg: MirrorConfig,
        init: &HashMap<String, Vec<f32>>,
        base: usize,
    ) -> Mirror<'g> {
        let c = cfg.dim;
        let reg = cfg.reg_tokens;
        // Assemble the constant per-frame head rows host-side: cls gets
        // pos_embed[0]; registers get no positional embedding.
        let cls = &init[&format!("{VGT}.patch_embed.cls_token")];
        let regs = &init[&format!("{VGT}.patch_embed.register_tokens")];
        let pos = &init[&format!("{VGT}.patch_embed.pos_embed")];
        let mut head_rows = Vec::with_capacity((1 + reg) * c);
        for d in 0..c {
            head_rows.push(cls[d] + pos[d]);
        }
        head_rows.extend_from_slice(&regs[..reg * c]);
        let pos_patch = pos[c..].to_vec();

        // Trunk specials: cam_token [1,2,1,C] / reg_token [1,2,4,C] — variant
        // 0 for frame 0, variant 1 for every later frame; pose/ray rows zero.
        let cam_t = &init[&format!("{VGT}.cam_token")];
        let reg_t = &init[&format!("{VGT}.reg_token")];
        let head = |variant: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(PATCH_START * c);
            v.extend_from_slice(&cam_t[variant * c..(variant + 1) * c]);
            v.extend_from_slice(&reg_t[variant * reg * c..(variant + 1) * reg * c]);
            v.resize(PATCH_START * c, 0.0); // pose + ray
            v
        };
        let trunk_head_f0 = head(0);
        let trunk_head_fr = head(1);
        let rope_periods = init[&format!("{VGT}.frame_blocks.0.attn.rope.periods")].clone();
        let cam_init9 = init["cam_head.init_token"].clone();

        let roles: Vec<(String, usize, Role)> = cfg
            .param_list()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(gpu, roles, init);
        Mirror {
            gpu,
            cfg,
            ps,
            base,
            head_rows,
            pos_patch,
            trunk_head_f0,
            trunk_head_fr,
            rope_periods,
            cam_init9,
            built: None,
        }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn dino_block_weights(&self, b: usize) -> VitBlockWeights<'_> {
        let p = |s: &str| format!("{VGT}.patch_embed.blocks.{b}.{s}");
        VitBlockWeights {
            norm1_w: self.w(&p("norm1.weight")),
            norm1_b: self.w(&p("norm1.bias")),
            qkv_w: self.w(&p("attn.qkv.weight")),
            qkv_b: self.w(&p("attn.qkv.bias")),
            qk_norm: None,
            rope: None,
            proj_w: self.w(&p("attn.proj.weight")),
            proj_b: self.w(&p("attn.proj.bias")),
            ls1: Some(self.w(&p("ls1.gamma"))),
            norm2_w: self.w(&p("norm2.weight")),
            norm2_b: self.w(&p("norm2.bias")),
            fc1_w: self.w(&p("mlp.fc1.weight")),
            fc1_b: self.w(&p("mlp.fc1.bias")),
            fc2_w: self.w(&p("mlp.fc2.weight")),
            fc2_b: self.w(&p("mlp.fc2.bias")),
            ls2: Some(self.w(&p("ls2.gamma"))),
        }
    }

    fn trunk_block_weights<'a>(
        &'a self,
        kind: &str,
        b: usize,
        rope: Option<model::vit::RopeTables<'a>>,
    ) -> VitBlockWeights<'a> {
        let p = |s: &str| format!("{VGT}.{kind}.{b}.{s}");
        VitBlockWeights {
            norm1_w: self.w(&p("norm1.weight")),
            norm1_b: self.w(&p("norm1.bias")),
            qkv_w: self.w(&p("attn.qkv.weight")),
            qkv_b: self.w(&p("attn.qkv.bias")),
            qk_norm: Some(model::vit::QkNorm {
                q_w: self.w(&p("attn.q_norm.weight")),
                q_b: self.w(&p("attn.q_norm.bias")),
                k_w: self.w(&p("attn.k_norm.weight")),
                k_b: self.w(&p("attn.k_norm.bias")),
            }),
            rope,
            proj_w: self.w(&p("attn.proj.weight")),
            proj_b: self.w(&p("attn.proj.bias")),
            ls1: Some(self.w(&p("ls1.gamma"))),
            norm2_w: self.w(&p("norm2.weight")),
            norm2_b: self.w(&p("norm2.bias")),
            fc1_w: self.w(&p("mlp.fc1.weight")),
            fc1_b: self.w(&p("mlp.fc1.bias")),
            fc2_w: self.w(&p("mlp.fc2.weight")),
            fc2_b: self.w(&p("mlp.fc2.bias")),
            ls2: Some(self.w(&p("ls2.gamma"))),
        }
    }

    /// (Re)build buffers + record the full forward (DINOv2 encode + trunk)
    /// for `s` frames of `hp*wp` patches (native 37×37 grid only for now —
    /// pos-embed interpolation for other grids lands with the infer CLI).
    fn build(&mut self, s: usize, hp: usize, wp: usize) {
        let cfg = &self.cfg;
        let c = cfg.dim;
        let reg = cfg.reg_tokens;
        let patches = hp * wp;
        assert_eq!(
            patches,
            cfg.patches(),
            "pos-embed interpolation for non-native grids is not wired yet"
        );
        let td = 1 + reg + patches; // per-frame DINOv2 tokens
        let td_t = PATCH_START + patches; // per-frame trunk tokens
        let rows = (s * td) as u32;
        let rows_t = (s * td_t) as u32;
        let (h, w) = (hp * cfg.patch, wp * cfg.patch);
        let gpu = self.gpu;
        let ids = vit_ids(self.base);
        let sh = VitShape {
            dim: c as u32,
            heads: cfg.heads as u32,
            mlp: (cfg.mlp_ratio * c) as u32,
            eps: DINO_EPS,
        };
        let sh_t = VitShape { eps: TRUNK_EPS, ..sh };

        let frames_in = gpu.storage((s * 3 * h * w) as u64);
        let conv_raw = gpu.storage((s * c * patches) as u64);
        let conv_out = gpu.storage((s * c * patches) as u64);
        let tokens = gpu.storage(rows as u64 * c as u64);
        let patch_out = gpu.storage((s * patches * c) as u64);
        let chunk = td as u32; // per-frame spans: slab = heads*td*td
        // One scratch serves DINOv2 (per-frame spans) and the trunk (frame
        // spans + query-chunked global attention): size to the largest need.
        let chunk_g = model::vit::attn_chunk_for(&sh_t, rows_t, ATTN_BUDGET).min(rows_t);
        let slab = sh.heads as u64
            * (td as u64 * td as u64)
                .max(td_t as u64 * td_t as u64)
                .max(chunk_g as u64 * rows_t as u64);
        let scr = VitScratch {
            ln: gpu.storage(rows_t as u64 * c as u64),
            qkv: gpu.storage(3 * rows_t as u64 * c as u64),
            ctx: gpu.storage(rows_t as u64 * c as u64),
            h: gpu.storage(rows_t as u64 * sh.mlp as u64),
            h2: gpu.storage(rows_t as u64 * sh.mlp as u64),
            res: gpu.storage(rows_t as u64 * c as u64),
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
        };
        let head_rows_buf = gpu.storage_init("mirror.head_rows", &self.head_rows);
        let pos_patch_buf = gpu.storage_init("mirror.pos_patch", &self.pos_patch);

        let mut steps: Vec<Step> = Vec::new();
        // patch conv [S,3,H,W] -> [S,C,hp,wp] + channel bias
        let pe = |n: &str| format!("{VGT}.patch_embed.patch_embed.proj.{n}");
        steps.push(gpu.step(
            self.base + K_CONV2D,
            &[&frames_in, self.w(&pe("weight")), &conv_raw],
            &[
                s as u32, 3, h as u32, w as u32, c as u32, cfg.patch as u32, cfg.patch as u32, 0,
                hp as u32, wp as u32,
            ],
            (s * c * patches) as u32,
        ));
        steps.push(gpu.step(
            self.base + K_ADD_CHAN_BCAST,
            &[&conv_raw, self.w(&pe("bias")), &conv_out],
            &[s as u32, c as u32, patches as u32],
            (s * c * patches) as u32,
        ));
        // per frame: tokens = [cls+pos0, regs, patches+pos]
        for fi in 0..s {
            let row0 = (fi * td) as u64;
            steps.push(gpu.step_sliced(
                self.base + K_NCHW_NLC,
                &[&conv_out, &tokens],
                &[((fi * c * patches) as u64, 0), ((row0 + 1 + reg as u64) * c as u64, 0)],
                &[(c * patches) as u32, c as u32, patches as u32],
                (c * patches) as u32,
            ));
            steps.push(gpu.step_sliced(
                self.base + K_AXPY,
                &[&tokens, &head_rows_buf],
                &[(row0 * c as u64, 0), (0, 0)],
                &[((1 + reg) * c) as u32, f(1.0)],
                ((1 + reg) * c) as u32,
            ));
            steps.push(gpu.step_sliced(
                self.base + K_AXPY,
                &[&tokens, &pos_patch_buf],
                &[((row0 + 1 + reg as u64) * c as u64, 0), (0, 0)],
                &[(patches * c) as u32, f(1.0)],
                (patches * c) as u32,
            ));
        }
        // 24 pre-LN blocks, per-frame attention spans
        let spans: Vec<(u32, u32)> = (0..s).map(|fi| ((fi * td) as u32, td as u32)).collect();
        for b in 0..cfg.depth {
            let bw = self.dino_block_weights(b);
            vit_block_fwd(gpu, &ids, &sh, &bw, &tokens, rows, &spans, chunk, &scr, &mut steps);
        }
        // final DINOv2 LayerNorm, then extract the patch rows per frame
        steps.push(gpu.step(
            self.base + K_LAYERNORM,
            &[
                &tokens,
                self.w(&format!("{VGT}.patch_embed.norm.weight")),
                self.w(&format!("{VGT}.patch_embed.norm.bias")),
                &scr.ln,
            ],
            &[c as u32, rows, f(DINO_EPS)],
            rows,
        ));
        for fi in 0..s {
            steps.push(gpu.step_sliced(
                self.base + K_AXPY,
                &[&patch_out, &scr.ln],
                &[
                    ((fi * patches * c) as u64, 0),
                    (((fi * td + 1 + reg) * c) as u64, (patches * c) as u64),
                ],
                &[(patches * c) as u32, f(1.0)],
                (patches * c) as u32,
            ));
        }

        // ---- trunk: token assembly ----
        let trunk_tokens = gpu.storage(rows_t as u64 * c as u64);
        let zeros = gpu.storage(rows_t as u64 * c as u64);
        let tap_tmp = gpu.storage(rows_t as u64 * c as u64);
        let head_f0 = gpu.storage_init("mirror.trunk_head_f0", &self.trunk_head_f0);
        let head_fr = gpu.storage_init("mirror.trunk_head_fr", &self.trunk_head_fr);
        let (cos, sin) = crate::rope2d::rope_tables(&self.rope_periods, hp, wp, PATCH_START);
        let rope_cos = gpu.storage_init("mirror.rope_cos", &cos);
        let rope_sin = gpu.storage_init("mirror.rope_sin", &sin);
        for fi in 0..s {
            let row0 = (fi * td_t) as u64;
            let head = if fi == 0 { &head_f0 } else { &head_fr };
            steps.push(gpu.step_sliced(
                self.base + K_AXPY,
                &[&trunk_tokens, head],
                &[(row0 * c as u64, 0), (0, 0)],
                &[(PATCH_START * c) as u32, f(1.0)],
                (PATCH_START * c) as u32,
            ));
            steps.push(gpu.step_sliced(
                self.base + K_AXPY,
                &[&trunk_tokens, &patch_out],
                &[
                    ((row0 + PATCH_START as u64) * c as u64, 0),
                    ((fi * patches * c) as u64, (patches * c) as u64),
                ],
                &[(patches * c) as u32, f(1.0)],
                (patches * c) as u32,
            ));
        }

        // ---- trunk: 24 × (frame block, global block), taps at cfg.tap_levels ----
        let spans_t: Vec<(u32, u32)> =
            (0..s).map(|fi| ((fi * td_t) as u32, td_t as u32)).collect();
        let span_all = [(0u32, rows_t)];
        let mut taps: Vec<DeviceBuffer> = Vec::new();
        for l in 0..cfg.depth {
            let rope = || model::vit::RopeTables { cos: &rope_cos, sin: &rope_sin, tmod: td_t as u32 };
            let fw = self.trunk_block_weights("frame_blocks", l, Some(rope()));
            vit_block_fwd(gpu, &ids, &sh_t, &fw, &trunk_tokens, rows_t, &spans_t, td_t as u32, &scr, &mut steps);
            let is_tap = cfg.tap_levels.contains(&l);
            if is_tap {
                steps.push(gpu.step(
                    self.base + K_ADD2,
                    &[&trunk_tokens, &zeros, &tap_tmp],
                    &[rows_t * c as u32],
                    rows_t * c as u32,
                ));
            }
            let gw = self.trunk_block_weights("global_blocks", l, Some(rope()));
            vit_block_fwd(gpu, &ids, &sh_t, &gw, &trunk_tokens, rows_t, &span_all, chunk_g, &scr, &mut steps);
            if is_tap {
                let tap = gpu.storage(rows_t as u64 * 2 * c as u64);
                steps.push(gpu.step(
                    self.base + K_CONCAT2,
                    &[&tap_tmp, &trunk_tokens, &tap],
                    &[rows_t, c as u32, c as u32, 1, 1],
                    rows_t * 2 * c as u32,
                ));
                taps.push(tap);
            }
        }

        // ---- dense heads (per frame) + camera head ----
        let dscr = DptScratch::new(gpu, cfg, hp, wp);
        let dk = dpt_kernels(self.base);
        let dctx = DptCtx { gpu, k: dk, cfg, scr: &dscr, eps: TRUNK_EPS };
        let hw_px = h * w;
        let mut rgb_frames = Vec::new();
        let mut depth_out = Vec::new();
        let mut pts_out = Vec::new();
        let mut norm_out = Vec::new();
        let mut gs_depth_out = Vec::new();
        let mut gs_params = Vec::new();
        for _ in 0..s {
            rgb_frames.push(gpu.storage((3 * hw_px) as u64));
            depth_out.push(gpu.storage((3 * hw_px) as u64));
            pts_out.push(gpu.storage((4 * hw_px) as u64));
            norm_out.push(gpu.storage((4 * hw_px) as u64));
            gs_depth_out.push(gpu.storage((3 * hw_px) as u64));
            gs_params.push(gpu.storage((12 * hw_px) as u64));
        }
        for fi in 0..s {
            for (prefix, out_ch, outs) in [
                ("depth_head", 3usize, &depth_out),
                ("pts_head", 4, &pts_out),
                ("norm_head", 4, &norm_out),
            ] {
                let hwt = HeadWeights { ps: &self.ps, prefix };
                dctx.head_frame(&hwt, &taps, fi, td_t, (hp, wp), out_ch, &outs[fi], None, &mut steps);
            }
            let hwt = HeadWeights { ps: &self.ps, prefix: "gs_head" };
            let gsb = GsBranch {
                rgb: &rgb_frames[fi],
                im_w: self.w("gs_head.input_merger.0.weight"),
                im_b: self.w("gs_head.input_merger.0.bias"),
                g0_w: self.w("gs_renderer.gs_head.0.weight"),
                g2_w: self.w("gs_renderer.gs_head.2.weight"),
                g2_b: self.w("gs_renderer.gs_head.2.bias"),
                out: &gs_params[fi],
            };
            dctx.head_frame(&hwt, &taps, fi, td_t, (hp, wp), 3, &gs_depth_out[fi], Some(&gsb), &mut steps);
        }
        let cam = CamBufs::new(gpu, s, 2 * c, &self.cam_init9);
        let ck = CamKernels {
            vit: ids,
            silu: self.base + K_SILU,
            mul: self.base + K_MUL,
            axpy: self.base + K_AXPY,
        };
        let cwt = CamWeights { ps: &self.ps };
        record_cam_head(gpu, &ck, &cwt, &cam, &taps[cfg.tap_levels.len() - 1], s, td_t, 2 * c, 4, &mut steps);

        self.built = Some(Built {
            s,
            hp,
            wp,
            frames_in,
            conv_raw,
            conv_out,
            tokens,
            scr,
            patch_out,
            trunk_tokens,
            zeros,
            taps,
            rgb_frames,
            depth_out,
            pts_out,
            norm_out,
            gs_depth_out,
            gs_params,
            dscr,
            cam,
            steps,
        });
    }

    /// Run the recorded forward on `s` frames of RAW [0,1] CHW pixels
    /// (concatenated): ImageNet normalization happens here (reference parity —
    /// the VGT normalizes internally), then DINOv2 → trunk → heads → camera.
    pub fn forward(&mut self, frames_chw: &[f32], s: usize, hp: usize, wp: usize) {
        let rebuild = match &self.built {
            Some(b) => b.s != s || b.hp != hp || b.wp != wp,
            None => true,
        };
        if rebuild {
            self.build(s, hp, wp);
        }
        let b = self.built.as_ref().unwrap();
        let hw = hp * self.cfg.patch * wp * self.cfg.patch;
        assert_eq!(frames_chw.len(), s * 3 * hw);
        let mut norm = Vec::with_capacity(frames_chw.len());
        for fr in 0..s {
            for ch in 0..3 {
                let base = (fr * 3 + ch) * hw;
                norm.extend(
                    frames_chw[base..base + hw]
                        .iter()
                        .map(|&v| (v - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch]),
                );
            }
            self.gpu.write(&b.rgb_frames[fr], bytemuck_cast(&frames_chw[fr * 3 * hw..(fr + 1) * 3 * hw]));
        }
        self.gpu.write(&b.frames_in, bytemuck_cast(&norm));
        // axpy-assembled buffers must start zeroed each run.
        self.gpu.submit(
            &[
                &b.tokens,
                &b.patch_out,
                &b.trunk_tokens,
                &b.zeros,
                &b.cam.cam_tok,
                &b.cam.pred,
            ],
            &b.steps,
        );
    }

    /// Raw camera 9-vectors `[s, 9]` (activations NOT applied — see
    /// `gaussians::decode_cameras`).
    pub fn cam_pred_raw(&self) -> Vec<f32> {
        let b = self.built.as_ref().unwrap();
        self.gpu.read(&b.cam.pred, b.s * 9)
    }

    /// Per-frame pre-activation head outputs (NCHW at full res).
    pub fn head_out(&self, which: Head, frame: usize) -> &DeviceBuffer {
        let b = self.built.as_ref().unwrap();
        match which {
            Head::Depth => &b.depth_out[frame],
            Head::Points => &b.pts_out[frame],
            Head::Normals => &b.norm_out[frame],
            Head::GsDepth => &b.gs_depth_out[frame],
            Head::GsParams => &b.gs_params[frame],
        }
    }

    /// DINOv2 patch tokens `[s*hp*wp, C]` (valid after [`Self::forward`]).
    pub fn patch_tokens(&self) -> &DeviceBuffer {
        &self.built.as_ref().unwrap().patch_out
    }

    /// The 4 tap buffers `[s*(7+hp*wp), 2C]` (frame‖global concat).
    pub fn taps(&self) -> &[DeviceBuffer] {
        &self.built.as_ref().unwrap().taps
    }

    /// Back-compat P1 entry: forward + return the DINOv2 patch tokens.
    pub fn encode_frames(&mut self, frames_chw: &[f32], s: usize, hp: usize, wp: usize) -> &DeviceBuffer {
        self.forward(frames_chw, s, hp, wp);
        self.patch_tokens()
    }
}

/// DPT kernel indices for a `Gpu` whose pipeline list contains [`PIPELINES`]
/// at offset `base` (public for the stage-isolation tests).
pub fn dpt_kernels(base: usize) -> DptKernels {
    DptKernels {
        layernorm: base + K_LAYERNORM,
        nlc_nchw: base + K_NLC_NCHW,
        conv2d: base + K_CONV2D,
        conv2d_dx: base + K_CONV2D_DX,
        add_chan_inplace: base + K_ADD_CHAN_INPLACE,
        resize_bilinear: base + K_RESIZE_BILINEAR,
        leaky_relu: base + K_LEAKY_RELU,
        relu_inplace: base + K_RELU_INPLACE,
        add2: base + K_ADD2,
        add_inplace: base + K_ADD_INPLACE,
        axpy: base + K_AXPY,
    }
}

/// Which dense-head output to fetch.
#[derive(Clone, Copy)]
pub enum Head {
    Depth,
    Points,
    Normals,
    GsDepth,
    GsParams,
}

fn bytemuck_cast(v: &[f32]) -> &[u32] {
    // safe: same size/alignment, plain-old-data
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) }
}
