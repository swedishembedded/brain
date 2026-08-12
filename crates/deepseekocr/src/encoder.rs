// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The **DeepEncoder**: image in, `[image_tokens, decoder d_model]` out.
//!
//! ```text
//! sam1::SamEncoder(image)               -> [1, c_out, gh/4, gw/4]  NCHW
//!   NCHW -> NLC                         -> compressor_flat [N, c_out]
//!   (fixture only: Linear(c_out, W))    -> patch tokens    [N, W]
//! clip::ClipVision(PatchSource::Tokens) -> [1 + N, W]      (patch embed BYPASSED)
//!   drop row 0 (the class token)        -> clip_spatial    [N, W]
//! concat2([clip_spatial, compressor_flat])                 [N, W + c_out]
//! Linear(W + c_out, d_model)            -> projector_out   [N, d_model]
//! ```
//!
//! ## Layout facts, each read off a reference rather than assumed
//!
//! * **The flatten is NCHW -> NLC, row-major over the compressor grid**: token
//!   `y*gw + x` is channel-vector `[:, y, x]`. That is `flatten(2).transpose(1,2)`
//!   in the reference dump and `permute + reshape` in llama.cpp, and it is why
//!   the fixture's compressor grid is `4x2` and not square -- a transposed
//!   flatten cannot survive distinct extents.
//! * **The class-token row is DROPPED, not pooled**, and it is row 0.
//! * **The concat is `[clip_spatial, compressor_flat]`** -- see this crate's lib
//!   header for the two independent sources that pin it.
//! * **The concat's high half is the compressor output PRE-bridge.** In the
//!   fixture the bridge widens `c_out` to `clip_width` for CLIP's benefit only;
//!   the projector still sees the raw `c_out`-wide features. (At real scale
//!   there is no bridge and the question does not arise.)
//!
//! ## Training vs inference builds
//!
//! [`DeepEncoder::new`]'s `train` flag is threaded, unchanged, into all three
//! stages: `sam1::SamEncoder::new_on`, `ClipVision::new_on_src` and this
//! module's own glue [`ParamStore`]. `false` makes every parameter
//! [`Role::Frozen`] (weight buffer only - no gradient, no AdamW moments) and
//! skips both towers' backward scratch. It is not a tuning knob: at the real
//! 1024x1024 shape a trainable build of this encoder costs several GB more
//! than a frozen one, and [`DeepEncoder::backward`] is unavailable on a frozen
//! one by construction.
//!
//! ## Why the stages exchange host buffers
//!
//! `sam1`, `clip` and this glue stage each register a different kernel set, so
//! each owns a different `gpu_core::Gpu`; `ClipVision`'s injected-token seam is
//! a host `&[f32]` on both the forward (`set_tokens`) and the backward
//! (`read_token_grad`) side for exactly that reason. The round trips are
//! therefore where the seams already are, not extra ones, and the tensors
//! involved are `[N, width]` -- the smallest things in the pipeline. A
//! same-device fast path is additive.

use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};
use gpu_core::{DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};
use sam1::model::SamEncoder;

use crate::config::{DeepseekOcrConfig, BYPASS_B, BYPASS_W, IMAGE_NEWLINE, PROJECTOR_B, PROJECTOR_W, VIEW_SEPARATOR};
use crate::layout::{RowGather, RowGatherIds};
use crate::rows::Src;
use crate::DeviceFactory;

// ---- glue kernel indices (order matches GLUE_PIPELINES) ----
const G_MATMUL: usize = 0;
const G_BIAS_ADD: usize = 1;
const G_CONCAT2: usize = 2;
const G_MATMUL_DX: usize = 3;
const G_MATMUL_DW: usize = 4;
const G_BIAS_GRAD: usize = 5;

/// The concat + projector stage's kernels. Forward first, backward appended,
/// then [`crate::layout::LAYOUT_PIPELINES`] appended after that - so every index
/// above stays stable and the row-layout path adds no separate device.
pub const GLUE_PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("concat2", kernels::CONCAT2),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("splice", kernels::SPLICE),
    ("embed", kernels::EMBED),
    ("emb_bwd", kernels::EMB_BWD),
];

/// Where [`crate::layout::LAYOUT_PIPELINES`] sits inside [`GLUE_PIPELINES`].
const G_ROWS: RowGatherIds = RowGatherIds { splice: 6, embed: 7, emb_bwd: 8 };

/// Buffers and parameters of the concat + projector (+ optional bridge) stage.
struct Glue {
    gpu: Gpu,
    ps: ParamStore,
    /// `[N, c_out]` -- the compressor output, flattened.
    comp_flat: DeviceBuffer,
    /// `[N, W]` -- CLIP's output with the class-token row dropped.
    clip_spatial: DeviceBuffer,
    /// `[N, W + c_out]`.
    concat: DeviceBuffer,
    /// `[N, d_model]`.
    proj_out: DeviceBuffer,
    /// `[N, W]` -- the bridge's output; absent unless `patch_bypass`.
    bridged: Option<DeviceBuffer>,

    d_proj_out: DeviceBuffer,
    d_concat: DeviceBuffer,
    d_bridged: Option<DeviceBuffer>,
    d_bridge_in: Option<DeviceBuffer>,
}

/// SAM + compressor + CLIP + concat + projector, forward and backward.
pub struct DeepEncoder {
    pub cfg: DeepseekOcrConfig,
    sam: SamEncoder,
    clip: ClipVision,
    glue: Glue,
    /// `image_tokens` -- rows of every `[N, *]` tensor above.
    n: u32,
}

impl DeepEncoder {
    /// Build the whole encoder. `init` must name every tensor of
    /// [`DeepseekOcrConfig::sam`]`.param_list()`,
    /// [`DeepseekOcrConfig::clip`]`.tensor_manifest()` and
    /// [`DeepseekOcrConfig::glue_param_list`] -- one flat source, because the
    /// three name spaces are already disjoint (`vision.sam.*`, CLIP's bare
    /// `blocks.*`/`pos_embed`/…, and `projector.*`/`patch_bypass.*`).
    ///
    /// `init` is a [`checkpoint::TensorSource`]: an eager
    /// `&HashMap<String, Vec<f32>>` coerces, and an mmap-backed
    /// `WeightReader`/`MmapGguf` streams one tensor at a time instead of
    /// materializing the tower as a second host copy.
    ///
    /// `train = false` builds every stage frozen and forward-only; see this
    /// module's header.
    pub fn new(
        dev: DeviceFactory<'_>,
        cfg: DeepseekOcrConfig,
        init: &dyn checkpoint::TensorSource,
        seed: u64,
        train: bool,
    ) -> DeepEncoder {
        cfg.check();
        let (gh, gw) = cfg.token_grid();
        let n = gh * gw;

        let sam = SamEncoder::new_on(dev(sam1::model::PIPELINES), cfg.sam.clone(), init, seed, train);
        assert_eq!(sam.out_len(), (n * cfg.compressor_out()) as usize, "compressor output disagrees with the token grid");

        let clip = ClipVision::new_on_src(
            dev(CLIP_VISION_PIPELINES),
            cfg.clip.clone(),
            1,
            PatchSource::Tokens { grid: (gh, gw) },
            init,
            train,
        );
        assert_eq!(clip.seq_len(), n + 1, "CLIP must run at 1 + image_tokens rows");

        let gpu = dev(GLUE_PIPELINES);
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> =
            cfg.glue_param_list().into_iter().map(|(name, numel)| (name, numel, role)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, init);
        let (w, cout, dm) = (cfg.clip_width() as u64, cfg.compressor_out() as u64, cfg.projector_out() as u64);
        let nl = n as u64;
        let glue = Glue {
            comp_flat: gpu.storage(nl * cout),
            clip_spatial: gpu.storage(nl * w),
            concat: gpu.storage(nl * (w + cout)),
            proj_out: gpu.storage(nl * dm),
            bridged: cfg.patch_bypass.then(|| gpu.storage(nl * w)),
            d_proj_out: gpu.storage(nl * dm),
            d_concat: gpu.storage(nl * (w + cout)),
            d_bridged: cfg.patch_bypass.then(|| gpu.storage(nl * w)),
            d_bridge_in: cfg.patch_bypass.then(|| gpu.storage(nl * cout)),
            gpu,
            ps,
        };
        DeepEncoder { cfg, sam, clip, glue, n }
    }

    /// Image tokens one view produces.
    pub fn tokens(&self) -> u32 {
        self.n
    }

    /// `[image_tokens * d_model]` -- the projector output's element count.
    pub fn out_len(&self) -> usize {
        (self.n * self.cfg.projector_out()) as usize
    }

    pub fn sam(&self) -> &SamEncoder {
        &self.sam
    }
    pub fn clip(&self) -> &ClipVision {
        &self.clip
    }

    // -----------------------------------------------------------------------
    // forward
    // -----------------------------------------------------------------------

    /// Run the encoder on one `[3, image_h, image_w]` NCHW image and return the
    /// projector output `[image_tokens, d_model]` (row-major).
    ///
    /// This is the **contiguous** path: `image_tokens` rows, in projector order,
    /// no newline or view-separator rows. It is what the checkpoint-free golden
    /// fixture covers and what every parity test in this crate runs.
    /// [`Self::forward_rows`] is the real interleaved layout on top of it.
    pub fn forward(&self, image: &[f32]) -> Vec<f32> {
        self.run_forward(image);
        self.glue.gpu.read(&self.glue.proj_out, self.out_len())
    }

    /// [`Self::forward`] without the readback - everything up to and including
    /// the projector's submit, leaving the result in `glue.proj_out`.
    fn run_forward(&self, image: &[f32]) {
        self.sam.write_image(image);
        let _ = self.sam.forward();

        // NCHW -> NLC. The one layout the whole fixture's non-square compressor
        // grid exists to pin: token (y, x) is channel-vector [:, y, x].
        let comp = self.sam.gpu.read(self.sam.output(), self.sam.out_len());
        let comp_flat = nchw_to_nlc(&comp, self.cfg.compressor_out() as usize, self.n as usize);
        self.glue.gpu.write_f32(&self.glue.comp_flat, &comp_flat);

        // The fixture's widening bridge, if any: the compressor output becomes
        // CLIP's patch token. At real scale it IS the patch token, verbatim.
        let tokens = match &self.glue.bridged {
            Some(bridged) => {
                let (n, cin, w) = (self.n, self.cfg.compressor_out(), self.cfg.clip_width());
                let g = &self.glue.gpu;
                let steps = vec![
                    g.step(G_MATMUL, &[&self.glue.comp_flat, self.glue.ps.w(BYPASS_W), bridged], &[n, cin, w], n * w),
                    g.step(G_BIAS_ADD, &[bridged, self.glue.ps.w(BYPASS_B)], &[n, w], n * w),
                ];
                g.submit(&[], &steps);
                g.read(bridged, (n * w) as usize)
            }
            None => comp_flat.clone(),
        };
        self.clip.set_tokens(&tokens);
        self.clip.forward();

        // Drop the class-token row (row 0) -- the tower's spatial output.
        let w = self.cfg.clip_width() as usize;
        let clip_spatial = self.clip.read_output()[w..].to_vec();
        assert_eq!(clip_spatial.len(), self.n as usize * w);
        self.glue.gpu.write_f32(&self.glue.clip_spatial, &clip_spatial);

        let (n, cout, dm) = (self.n, self.cfg.compressor_out(), self.cfg.projector_out());
        let pin = self.cfg.projector_in();
        let g = &self.glue.gpu;
        // `concat2` Params: [N, Ca, Cb, H, W] over NCHW -- a row-wise concat is
        // that with H = W = 1, so "channels" are the two halves' widths.
        let steps = vec![
            g.step(G_CONCAT2, &[&self.glue.clip_spatial, &self.glue.comp_flat, &self.glue.concat], &[n, self.cfg.clip_width(), cout, 1, 1], n * pin),
            g.step(G_MATMUL, &[&self.glue.concat, self.glue.ps.w(PROJECTOR_W), &self.glue.proj_out], &[n, pin, dm], n * dm),
            g.step(G_BIAS_ADD, &[&self.glue.proj_out, self.glue.ps.w(PROJECTOR_B)], &[n, dm], n * dm),
        ];
        g.submit(&[], &steps);
    }

    // -----------------------------------------------------------------------
    // the real interleaved row layout
    // -----------------------------------------------------------------------

    /// Build the device state for one image-block layout, on this encoder's own
    /// glue device.
    ///
    /// `rows` is `crate::rows::RowPlan::rows` - the `Src` sequence the decoder's
    /// image run follows. The projector row count is this encoder's own
    /// [`Self::tokens`], so a layout built for a different view geometry is
    /// refused here rather than producing a plausible wrong block.
    pub fn row_gather(&self, rows: &[Src]) -> RowGather {
        RowGather::new(&self.glue.gpu, rows, self.n, self.cfg.projector_out())
    }

    /// [`Self::forward`] followed by the row gather: returns the FULL image
    /// block `[rows.len(), d_model]`, with the projector's output at the
    /// `Src::Projector` rows and the two learned vectors at the others.
    ///
    /// 273 rows at the real geometry (256 projector + 16 `image_newline` + 1
    /// `view_separator`), versus [`Self::forward`]'s 256. This is what the
    /// reference feeds its decoder.
    pub fn forward_rows(&self, image: &[f32], rg: &RowGather) -> Vec<f32> {
        assert_eq!(rg.projector_rows(), self.n, "this RowGather was built for a different encoder");
        self.run_forward(image);
        let g = &self.glue.gpu;
        let mut steps = Vec::new();
        rg.build_fwd(g, G_ROWS, &self.glue.proj_out, self.glue.ps.w(IMAGE_NEWLINE), self.glue.ps.w(VIEW_SEPARATOR), &mut steps);
        g.submit(&[], &steps);
        g.read(rg.block(), rg.block_len())
    }

    /// Adjoint of [`Self::forward_rows`]. `d_block` is `[rows.len(), d_model]`
    /// - the gradient the decoder's splice seam produced over the FULL block.
    ///
    /// De-interleaves it: the `Src::Projector` rows go on into
    /// [`Self::backward`], and the newline/separator rows are summed onto the
    /// two learned vectors' parameter gradients (16 terms and 1 at the real
    /// geometry - a shared parameter's gradient is the sum over every use).
    /// Returns the gradient w.r.t. the input image, as [`Self::backward`] does.
    pub fn backward_rows(&self, d_block: &[f32], rg: &RowGather) -> Vec<f32> {
        assert!(
            self.sam.is_trainable(),
            "DeepEncoder: this is an inference build (train = false); it allocated no gradient buffers"
        );
        assert_eq!(rg.projector_rows(), self.n, "this RowGather was built for a different encoder");
        assert_eq!(d_block.len(), rg.block_len(), "d_block must be [layout rows, d_model]");
        let g = &self.glue.gpu;
        g.write_f32(rg.d_block(), d_block);
        let mut steps = Vec::new();
        rg.build_bwd(g, G_ROWS, self.glue.ps.g(IMAGE_NEWLINE), self.glue.ps.g(VIEW_SEPARATOR), &mut steps);
        // The two shared gradients ACCUMULATE (`zero_grads` owns clearing them);
        // `d_proj` is fully overwritten by the inverse gather.
        g.submit(&[], &steps);
        let d_proj = g.read(rg.d_proj(), rg.proj_len());
        self.backward(&d_proj)
    }

    // -----------------------------------------------------------------------
    // backward
    // -----------------------------------------------------------------------

    /// Adjoint of [`Self::forward`]. `d_out` is `[image_tokens, d_model]` -- the
    /// gradient the decoder's splice seam produced. Parameter gradients land in
    /// each stage's own ParamStore; the returned value is the gradient w.r.t.
    /// the input image (`[3, image_h, image_w]`).
    pub fn backward(&self, d_out: &[f32]) -> Vec<f32> {
        assert!(
            self.sam.is_trainable(),
            "DeepEncoder: this is an inference build (train = false); it allocated no gradient buffers"
        );
        assert_eq!(d_out.len(), self.out_len(), "d_out must be [image_tokens, d_model]");
        let (n, cout, dm) = (self.n, self.cfg.compressor_out(), self.cfg.projector_out());
        let (w, pin) = (self.cfg.clip_width(), self.cfg.projector_in());
        let g = &self.glue.gpu;
        let ps = &self.glue.ps;

        // ---- projector ----
        g.write_f32(&self.glue.d_proj_out, d_out);
        let steps = vec![
            g.step(G_BIAS_GRAD, &[&self.glue.d_proj_out, ps.g(PROJECTOR_B)], &[n, dm], dm),
            g.step(G_MATMUL_DW, &[&self.glue.d_proj_out, &self.glue.concat, ps.g(PROJECTOR_W)], &[n, pin, dm], dm * pin),
            g.step(G_MATMUL_DX, &[&self.glue.d_proj_out, ps.w(PROJECTOR_W), &self.glue.d_concat], &[n, pin, dm, 0], n * pin),
        ];
        g.submit(&[], &steps);

        // ---- concat: the adjoint is the two half-slices, in the same order ----
        let d_concat = g.read(&self.glue.d_concat, (n * pin) as usize);
        let (wu, coutu, nu) = (w as usize, cout as usize, n as usize);
        let mut d_clip_spatial = Vec::with_capacity(nu * wu);
        let mut d_comp = Vec::with_capacity(nu * coutu);
        for r in 0..nu {
            let row = &d_concat[r * (wu + coutu)..(r + 1) * (wu + coutu)];
            d_clip_spatial.extend_from_slice(&row[..wu]);
            d_comp.extend_from_slice(&row[wu..]);
        }

        // ---- CLIP: re-insert the dropped class row as an exact zero ----
        let mut d_clip_out = vec![0f32; (self.n as usize + 1) * wu];
        d_clip_out[wu..].copy_from_slice(&d_clip_spatial);
        self.clip.backward(&d_clip_out);
        self.clip.poll_wait();
        let d_tokens = self.clip.read_token_grad();

        // ---- the bridge (fixture only), then the compressor's two consumers ----
        // `compressor_flat` feeds BOTH the concat's high half and (through the
        // bridge, or verbatim) CLIP's patch tokens, so its gradient is the SUM.
        let d_from_clip = match (&self.glue.bridged, &self.glue.d_bridged, &self.glue.d_bridge_in) {
            (Some(bridged), Some(d_bridged), Some(d_bridge_in)) => {
                g.write_f32(d_bridged, &d_tokens);
                let steps = vec![
                    g.step(G_BIAS_GRAD, &[d_bridged, ps.g(BYPASS_B)], &[n, w], w),
                    g.step(G_MATMUL_DW, &[d_bridged, &self.glue.comp_flat, ps.g(BYPASS_W)], &[n, cout, w], w * cout),
                    g.step(G_MATMUL_DX, &[d_bridged, ps.w(BYPASS_W), d_bridge_in], &[n, cout, w, 0], n * cout),
                ];
                g.submit(&[], &steps);
                let _ = bridged;
                g.read(d_bridge_in, nu * coutu)
            }
            _ => d_tokens,
        };
        for (a, b) in d_comp.iter_mut().zip(&d_from_clip) {
            *a += *b;
        }

        // ---- NLC -> NCHW, then the whole SAM tower ----
        let d_comp_nchw = nlc_to_nchw(&d_comp, coutu, nu);
        self.sam.gpu.write_f32(self.sam.d_out(), &d_comp_nchw);
        self.sam.backward();
        self.sam.gpu.read(self.sam.d_image(), (3 * self.cfg.sam.image_h() * self.cfg.sam.image_w()) as usize)
    }

    pub fn zero_grads(&self) {
        self.sam.zero_grads();
        self.clip.zero_grads();
        self.glue.ps.zero_grads(&self.glue.gpu);
    }

    // -----------------------------------------------------------------------
    // taps and parameter access
    // -----------------------------------------------------------------------

    /// `[image_tokens, c_out]` -- the compressor output, flattened.
    pub fn read_compressor_flat(&self) -> Vec<f32> {
        self.glue.gpu.read(&self.glue.comp_flat, (self.n * self.cfg.compressor_out()) as usize)
    }
    /// `[image_tokens, clip_width]` -- CLIP's output, class token dropped.
    pub fn read_clip_spatial(&self) -> Vec<f32> {
        self.glue.gpu.read(&self.glue.clip_spatial, (self.n * self.cfg.clip_width()) as usize)
    }
    /// `[image_tokens, d_model]` -- the projector output [`Self::forward`]
    /// returned, re-read without re-running the encoder.
    pub fn read_projector_out(&self) -> Vec<f32> {
        self.glue.gpu.read(&self.glue.proj_out, self.out_len())
    }
    /// `[image_tokens, clip_width + c_out]`.
    pub fn read_vision_concat(&self) -> Vec<f32> {
        self.glue.gpu.read(&self.glue.concat, (self.n * self.cfg.projector_in()) as usize)
    }
    /// `[image_tokens, clip_width]` -- the bridged patch tokens; `None` without
    /// the fixture bridge (there the compressor output IS the patch token).
    pub fn read_patch_tokens(&self) -> Option<Vec<f32>> {
        self.glue.bridged.as_ref().map(|b| self.glue.gpu.read(b, (self.n * self.cfg.clip_width()) as usize))
    }
    pub fn glue_param_names(&self) -> Vec<String> {
        self.cfg.glue_param_list().into_iter().map(|(n, _)| n).collect()
    }
    pub fn read_glue_weight(&self, name: &str) -> Vec<f32> {
        self.glue.ps.read_weight(&self.glue.gpu, name)
    }
    pub fn write_glue_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.glue.ps.numel(name), "{name}: size mismatch");
        self.glue.gpu.write_f32(self.glue.ps.w(name), data);
    }
    pub fn read_glue_grad(&self, name: &str) -> Vec<f32> {
        self.glue.ps.read_grad(&self.glue.gpu, name)
    }
}

/// `[C, H, W]` -> `[H*W, C]`. `n == H*W`.
fn nchw_to_nlc(src: &[f32], c: usize, n: usize) -> Vec<f32> {
    assert_eq!(src.len(), c * n);
    let mut out = vec![0f32; c * n];
    for ch in 0..c {
        for i in 0..n {
            out[i * c + ch] = src[ch * n + i];
        }
    }
    out
}

/// `[H*W, C]` -> `[C, H, W]` -- the exact inverse of [`nchw_to_nlc`].
fn nlc_to_nchw(src: &[f32], c: usize, n: usize) -> Vec<f32> {
    assert_eq!(src.len(), c * n);
    let mut out = vec![0f32; c * n];
    for i in 0..n {
        for ch in 0..c {
            out[ch * n + i] = src[i * c + ch];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two permutations are exact inverses, and the forward one really is
    /// channel-last (a transposed grid would still round-trip, so the explicit
    /// element check is the one that matters).
    #[test]
    fn nchw_nlc_round_trips_and_is_channel_last() {
        // C = 2, H = 3, W = 2 -> n = 6.
        let src: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let nlc = nchw_to_nlc(&src, 2, 6);
        assert_eq!(nlc, vec![0., 6., 1., 7., 2., 8., 3., 9., 4., 10., 5., 11.]);
        assert_eq!(nlc_to_nchw(&nlc, 2, 6), src);
    }
}
