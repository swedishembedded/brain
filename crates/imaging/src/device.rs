// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device path: every per-pixel operation, as a dispatch of an existing
//! kernel.
//!
//! ## The rule this module implements
//!
//! `AGENTS.md`: *host math does not run on the accelerator* — a host loop is
//! invisible to `--device` and a benchmark of it reports host numbers under a
//! device label. Resizing a 4K frame is 8 M elements; that is a kernel's job.
//! The kernels already exist, and this module dispatches them. It re-implements
//! none of them.
//!
//! ## Kernel indices are resolved BY NAME
//!
//! `Gpu::step(kind, ..)` takes a position into whatever `PIPELINES` array the
//! *owning model* handed to `Gpu::new`, so a bare index means nothing outside
//! that model. [`ImagingKernelIds`] resolves each kernel by name against the
//! live handle, exactly as `vision::ConvKernelIds` does for the conv blocks —
//! two families, two seams, one mechanism. A caller that never registered
//! `pad2d` gets a panic naming `pad2d` at the dispatch, not a silently wrong
//! kernel producing plausible pixels.
//!
//! Use [`PIPELINES`] to register the whole set (`Gpu::new(imaging::PIPELINES)`),
//! or append the names you need to a model's own list — order is irrelevant.
//!
//! ## Params were read before dispatching, and are written out here
//!
//! A mismatched param list is silently wrong,
//! not a crash (`silu_mul` cost a forward cosine of 0.504). Every dispatch
//! below states its kernel's `Params` field order in a comment immediately
//! above it. Keep those comments truthful; they are the contract.
//!
//! ## Eager, not recorded
//!
//! Each operation records one or a few `Step`s and submits them. This is a
//! preprocessing/postprocessing façade, not a training graph: it allocates its
//! own output and returns it. Models that need the ops fused into a recorded
//! graph should dispatch the kernels directly with the `Params` documented here
//! rather than have this module grow a second, recording API.

use gpu_core::{DeviceBuffer, Gpu};
use vision::Shape;

use crate::color::Normalization;
use crate::pixels::Rect;

/// A kernel this handle did not register. Re-exported from `vision` — one
/// sentinel, not two.
pub use vision::ids::NONE;

/// Every kernel `imaging` dispatches, ready for `Gpu::new(imaging::PIPELINES)`.
///
/// A model with its own pipeline list does not need this: it can append the
/// names it wants and [`ImagingKernelIds::resolve_on`] will find them wherever
/// they sit.
///
/// `region_copy` is deliberately absent. It copies `dst[i] = src[i]` at
/// *identical* indices in both buffers, so it can neither compact a tile out of
/// an image nor place one back — `crop2d` and `pad2d` are the pair that can.
pub const PIPELINES: &[(&str, &str)] = &[
    // ---- resampling. Same Params, same bindings, different tap stencil.
    ("resize_nearest", kernels::RESIZE_NEAREST),
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("resize_bicubic", kernels::RESIZE_BICUBIC),
    // ---- geometry. crop2d is pad2d's exact adjoint with the same Params.
    ("pad2d", kernels::PAD2D),
    ("crop2d", kernels::CROP2D),
    // ---- layout. NCHW <-> NLC, where L = H*W and C is trailing: NLC *is* the
    // interleaved HWC layout, so these are the device twins of
    // `pixels::{chw_to_hwc, hwc_to_chw}`.
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    // ---- per-channel affine `y = x*(1+s) + b`. brain's one affine kernel:
    // normalise, denormalise, negate, invert a mask, remap a value range.
    ("film_chan", kernels::FILM_CHAN),
    // ---- mask algebra (see `crate::mask`).
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
    ("axpy", kernels::AXPY),
    ("maxpool2d", kernels::MAXPOOL2D),
    ("avgpool2d", kernels::AVGPOOL2D),
    ("conv2d_gd", kernels::CONV2D_GD),
    // brain's only `sign()`. The name is Kronos-historical (Binary Spherical
    // Quantization); with `inv_sqrt_k = 1` it is exactly
    // `torch.where(x > 0, +1, -1)`, which is what a hard threshold needs.
    ("bsq_quantize", kernels::BSQ_QUANTIZE),
];

/// Pipeline indices for the kernels this crate dispatches, resolved by name.
#[derive(Clone, Copy, Debug)]
pub struct ImagingKernelIds {
    pub resize_nearest: usize,
    pub resize_bilinear: usize,
    pub resize_bicubic: usize,
    pub pad2d: usize,
    pub crop2d: usize,
    pub nchw_nlc: usize,
    pub nlc_nchw: usize,
    pub film_chan: usize,
    pub mul: usize,
    pub add2: usize,
    pub add_inplace: usize,
    pub axpy: usize,
    pub maxpool2d: usize,
    pub avgpool2d: usize,
    pub conv2d_gd: usize,
    pub bsq_quantize: usize,
}

impl ImagingKernelIds {
    /// Resolve against a pipeline list. Absent kernels become [`NONE`].
    pub fn resolve(pipelines: &[(&str, &str)]) -> ImagingKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).unwrap_or(NONE);
        ImagingKernelIds::build(k)
    }

    /// Resolve against a live handle's registered pipelines — the usual entry,
    /// because it works for a model that mixed these kernels into its own list.
    pub fn resolve_on(gpu: &Gpu) -> ImagingKernelIds {
        ImagingKernelIds::build(|name| gpu.kernel_index(name).unwrap_or(NONE))
    }

    fn build(k: impl Fn(&str) -> usize) -> ImagingKernelIds {
        ImagingKernelIds {
            resize_nearest: k("resize_nearest"),
            resize_bilinear: k("resize_bilinear"),
            resize_bicubic: k("resize_bicubic"),
            pad2d: k("pad2d"),
            crop2d: k("crop2d"),
            nchw_nlc: k("nchw_nlc"),
            nlc_nchw: k("nlc_nchw"),
            film_chan: k("film_chan"),
            mul: k("mul"),
            add2: k("add2"),
            add_inplace: k("add_inplace"),
            axpy: k("axpy"),
            maxpool2d: k("maxpool2d"),
            avgpool2d: k("avgpool2d"),
            conv2d_gd: k("conv2d_gd"),
            bsq_quantize: k("bsq_quantize"),
        }
    }

    /// Assert a kernel was registered, panicking with its NAME if not.
    #[inline]
    pub fn need(&self, id: usize, what: &str) -> usize {
        assert_ne!(id, NONE, "kernel `{what}` is not registered — add it to this handle's PIPELINES");
        id
    }
}

/// Which resampling function a resize uses.
///
/// Each variant names the exact reference function it reproduces. They are not
/// quality tiers to be swapped freely: a model imported against one of them
/// produces different numbers under another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Filter {
    /// `resize_nearest.wgsl` — torch `nearest`, i.e. ONNX `asymmetric` +
    /// `nearest_mode = floor`, `src = floor(dst*in/out)`. Integer arithmetic;
    /// [`AlignCorners`] does not apply and is ignored.
    ///
    /// This is **not** the half-pixel nearest that
    /// [`crate::letterbox::letterbox_rgb`] uses. See that module.
    Nearest,
    /// `resize_bilinear.wgsl`. The default, and bit-equivalent (under
    /// [`AlignCorners::HalfPixel`]) to the three host bilinear copies in
    /// `depth::predict` / `cli::depth_cli` / `cli::resident_depth`.
    #[default]
    Bilinear,
    /// `resize_bicubic.wgsl` — PyTorch's **non-antialiased** cubic convolution,
    /// `a = -0.75`, fixed 4x4 support, clamp-to-edge taps.
    ///
    /// NOT `mirror::preprocess::resize_bicubic` (PIL fixed-point, antialiased,
    /// `a = -0.5`, bit-exact against a PIL golden) and NOT
    /// `mirror::preprocess::resize_bicubic_torch` (antialiased, f64 accumulate).
    /// Those are different mathematical functions, not other implementations of
    /// this one, and antialiased downsampling still has no kernel. Pointing
    /// mirror at this variant would break `t1_pil_bicubic_exact`.
    Bicubic,
}

/// The source-coordinate convention. The two differ by half a pixel, both look
/// plausible, and no gradient check can tell them apart — only a numeric parity
/// test against the reference can. So it is always spelled out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignCorners {
    /// `src = (dst + 0.5) * in/out - 0.5`. PyTorch's default; ONNX
    /// `coordinate_transformation_mode = "half_pixel"`. **The default here.**
    #[default]
    HalfPixel,
    /// `src = dst * (in-1)/(out-1)`. Corner samples land on corner pixels; ONNX
    /// `"align_corners"`. `mirror::dpt`'s upsampler and ZipDepth's final
    /// upsample back to source resolution use this one.
    Corners,
}

impl AlignCorners {
    /// The `align_corners` uniform word the resize kernels take.
    fn word(self) -> u32 {
        match self {
            AlignCorners::HalfPixel => 0,
            AlignCorners::Corners => 1,
        }
    }
}

/// Asymmetric border amounts in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Border {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

impl Border {
    pub fn uniform(n: u32) -> Border {
        Border { left: n, right: n, top: n, bottom: n }
    }
    /// Centre an `inner` extent inside `outer`, extra pixel to the right/bottom.
    /// The letterbox geometry, expressed as a border.
    pub fn centre(inner_w: u32, inner_h: u32, outer_w: u32, outer_h: u32) -> Border {
        assert!(inner_w <= outer_w && inner_h <= outer_h, "centre: inner must fit inside outer");
        let (dw, dh) = (outer_w - inner_w, outer_h - inner_h);
        Border { left: dw / 2, right: dw - dw / 2, top: dh / 2, bottom: dh - dh / 2 }
    }
}

/// The device-side image context: a `&Gpu` plus the resolved kernel indices.
///
/// Cheap to construct (one name lookup per kernel) but not free — hold one per
/// pipeline rather than building one per frame.
pub struct Ctx<'g> {
    pub gpu: &'g Gpu,
    pub ids: ImagingKernelIds,
}

impl<'g> Ctx<'g> {
    /// Resolve this crate's kernels against `gpu`'s registered pipelines.
    pub fn new(gpu: &'g Gpu) -> Ctx<'g> {
        Ctx { gpu, ids: ImagingKernelIds::resolve_on(gpu) }
    }

    /// A fresh device buffer of `n` f32 elements.
    pub fn buf(&self, n: u32) -> DeviceBuffer {
        self.gpu.storage(n as u64)
    }

    /// Upload host f32 as a device buffer.
    pub fn upload(&self, name: &str, data: &[f32]) -> DeviceBuffer {
        self.gpu.storage_init(name, data)
    }

    /// Read `n` f32 back.
    pub fn download(&self, buf: &DeviceBuffer, n: u32) -> Vec<f32> {
        self.gpu.read(buf, n as usize)
    }

    /// Resample `x` (NCHW) to `ho x wo`.
    ///
    /// `resize_{nearest,bilinear,bicubic}` Params — bilinear and bicubic:
    /// `[N, C, H, W, Ho, Wo, align_corners]`; nearest: the same list **without**
    /// `align_corners` (6 words). One thread per OUTPUT element.
    pub fn resize(
        &self,
        x: &DeviceBuffer,
        s: Shape,
        ho: u32,
        wo: u32,
        filter: Filter,
        align: AlignCorners,
    ) -> (DeviceBuffer, Shape) {
        assert!(ho > 0 && wo > 0, "resize: target must be non-empty");
        let out_shape = Shape::new(s.n, s.c, ho, wo);
        let out = self.buf(out_shape.numel());
        let threads = out_shape.numel();
        let step = match filter {
            Filter::Nearest => self.gpu.step(
                self.ids.need(self.ids.resize_nearest, "resize_nearest"),
                &[x, &out],
                &[s.n, s.c, s.h, s.w, ho, wo],
                threads,
            ),
            Filter::Bilinear => self.gpu.step(
                self.ids.need(self.ids.resize_bilinear, "resize_bilinear"),
                &[x, &out],
                &[s.n, s.c, s.h, s.w, ho, wo, align.word()],
                threads,
            ),
            Filter::Bicubic => self.gpu.step(
                self.ids.need(self.ids.resize_bicubic, "resize_bicubic"),
                &[x, &out],
                &[s.n, s.c, s.h, s.w, ho, wo, align.word()],
                threads,
            ),
        };
        self.gpu.submit(&[], &[step]);
        (out, out_shape)
    }

    /// Zero-pad `x` by `b`.
    ///
    /// `pad2d` Params `[total, h, w, l, r, t, b]` where `h`/`w` are the
    /// **unpadded** dims and `total` counts the **padded** output
    /// (`N*C*(h+t+b)*(w+l+r)`). N and C are folded into one image index by the
    /// kernel, so this works for any batch/channel count.
    ///
    /// Zero is the only fill this kernel can produce. A grey letterbox border or
    /// an edge-replicated outpaint canvas (`zimage::caps::build_outpaint_canvas`)
    /// needs a `pad_value` / `pad_mode` word added to `pad2d.wgsl` — one uniform
    /// word, not a second kernel. Until then, do not pretend: pad with zero and
    /// composite the fill, or stay on the host.
    pub fn pad_zero(&self, x: &DeviceBuffer, s: Shape, b: Border) -> (DeviceBuffer, Shape) {
        let out_shape = Shape::new(s.n, s.c, s.h + b.top + b.bottom, s.w + b.left + b.right);
        let out = self.buf(out_shape.numel());
        let total = out_shape.numel();
        let step = self.gpu.step(
            self.ids.need(self.ids.pad2d, "pad2d"),
            &[x, &out],
            &[total, s.h, s.w, b.left, b.right, b.top, b.bottom],
            total,
        );
        self.gpu.submit(&[], &[step]);
        (out, out_shape)
    }

    /// Crop `rect` out of `x`.
    ///
    /// `crop2d` Params `[total, h, w, l, r, t, b]` — **`h`/`w` are the CROPPED
    /// (output) dims** and `total` counts the output (`N*C*h*w`); the input is
    /// the larger `[NC, h+t+b, w+l+r]` tensor. Identical field list to `pad2d`
    /// with the opposite meaning of `h`/`w`, which is exactly the kind of
    /// mismatch that is silently wrong rather than a crash — hence the
    /// rect-shaped signature here, so no caller computes those words by hand.
    pub fn crop(&self, x: &DeviceBuffer, s: Shape, rect: Rect) -> (DeviceBuffer, Shape) {
        assert!(
            rect.right() <= s.w && rect.bottom() <= s.h && rect.w > 0 && rect.h > 0,
            "crop: {rect:?} does not fit inside {}x{}",
            s.w,
            s.h
        );
        let out_shape = Shape::new(s.n, s.c, rect.h, rect.w);
        let out = self.buf(out_shape.numel());
        let total = out_shape.numel();
        let step = self.gpu.step(
            self.ids.need(self.ids.crop2d, "crop2d"),
            &[x, &out],
            &[
                total,
                rect.h,
                rect.w,
                rect.x,
                s.w - rect.right(),
                rect.y,
                s.h - rect.bottom(),
            ],
            total,
        );
        self.gpu.submit(&[], &[step]);
        (out, out_shape)
    }

    /// Add `src` into `dst` at `at`, in place. `src` must be `at.w x at.h`.
    ///
    /// Two dispatches: `pad2d` grows `src` to the canvas size (zero elsewhere),
    /// then `add_inplace` (`Params [total]`) accumulates it. Exact when the
    /// destination region is zero or the regions are disjoint — which is the
    /// contract [`crate::tiling`] provides.
    pub fn add_region(&self, dst: &DeviceBuffer, dst_shape: Shape, src: &DeviceBuffer, at: Rect) {
        assert!(
            at.right() <= dst_shape.w && at.bottom() <= dst_shape.h,
            "add_region: {at:?} does not fit inside {}x{}",
            dst_shape.w,
            dst_shape.h
        );
        let src_shape = Shape::new(dst_shape.n, dst_shape.c, at.h, at.w);
        let border = Border {
            left: at.x,
            right: dst_shape.w - at.right(),
            top: at.y,
            bottom: dst_shape.h - at.bottom(),
        };
        let (grown, grown_shape) = self.pad_zero(src, src_shape, border);
        debug_assert_eq!(grown_shape.numel(), dst_shape.numel());
        let total = dst_shape.numel();
        let step = self.gpu.step(
            self.ids.need(self.ids.add_inplace, "add_inplace"),
            &[dst, &grown],
            &[total],
            total,
        );
        self.gpu.submit(&[], &[step]);
    }

    /// NCHW -> interleaved HWC (`nchw_nlc`, since NLC with `L = H*W` **is** the
    /// interleaved layout). Params `[total, c, hw]`, one thread per element.
    ///
    /// The device twin of [`crate::pixels::chw_to_hwc`]. Use this one whenever a
    /// device buffer is in scope; the host function is for bytes that never
    /// reach the device.
    pub fn to_hwc(&self, x: &DeviceBuffer, s: Shape) -> DeviceBuffer {
        let total = s.numel();
        let out = self.buf(total);
        let step = self.gpu.step(
            self.ids.need(self.ids.nchw_nlc, "nchw_nlc"),
            &[x, &out],
            &[total, s.c, s.h * s.w],
            total,
        );
        self.gpu.submit(&[], &[step]);
        out
    }

    /// Interleaved HWC -> NCHW (`nlc_nchw`), the exact inverse of
    /// [`Ctx::to_hwc`]. Params `[total, c, hw]`. `s` describes the NCHW form.
    pub fn to_chw(&self, x: &DeviceBuffer, s: Shape) -> DeviceBuffer {
        let total = s.numel();
        let out = self.buf(total);
        let step = self.gpu.step(
            self.ids.need(self.ids.nlc_nchw, "nlc_nchw"),
            &[x, &out],
            &[total, s.c, s.h * s.w],
            total,
        );
        self.gpu.submit(&[], &[step]);
        out
    }

    /// Per-channel affine `y = x * scale[c] + shift[c]`.
    ///
    /// `film_chan` Params `[N, C, H, W]`, one thread per element, computing
    /// `y = x * (1 + s) + b` from a **single** `sb` buffer laid out `[N, 2C]`:
    /// row `n` is `[s_0..s_{C-1}, b_0..b_{C-1}]`. The `-1` on the scale is
    /// applied here, in the one place that knows the kernel's convention.
    ///
    /// This one kernel covers normalisation, denormalisation, negation,
    /// `1 - mask`, and every `[0,1] <-> [-1,1]` remap in the workspace.
    pub fn affine(&self, x: &DeviceBuffer, s: Shape, scale: &[f32], shift: &[f32]) -> DeviceBuffer {
        assert_eq!(scale.len(), s.c as usize, "affine: one scale per channel");
        assert_eq!(shift.len(), s.c as usize, "affine: one shift per channel");
        let c = s.c as usize;
        let mut sb = vec![0f32; s.n as usize * 2 * c];
        for n in 0..s.n as usize {
            for ch in 0..c {
                sb[n * 2 * c + ch] = scale[ch] - 1.0;
                sb[n * 2 * c + c + ch] = shift[ch];
            }
        }
        let sb = self.upload("imaging.affine.sb", &sb);
        let total = s.numel();
        let out = self.buf(total);
        let step = self.gpu.step(
            self.ids.need(self.ids.film_chan, "film_chan"),
            &[x, &sb, &out],
            &[s.n, s.c, s.h, s.w],
            total,
        );
        self.gpu.submit(&[], &[step]);
        out
    }

    /// `(x - mean) / std` per channel. Requires `s.c == 3`.
    pub fn normalize(&self, x: &DeviceBuffer, s: Shape, n: &Normalization) -> DeviceBuffer {
        assert_eq!(s.c, 3, "normalize: Normalization carries three channels");
        let (scale, shift) = n.scale_shift();
        self.affine(x, s, &scale, &shift)
    }

    /// `x * std + mean` per channel — the inverse of [`Ctx::normalize`].
    pub fn denormalize(&self, x: &DeviceBuffer, s: Shape, n: &Normalization) -> DeviceBuffer {
        assert_eq!(s.c, 3, "denormalize: Normalization carries three channels");
        let (scale, shift) = n.inverse_scale_shift();
        self.affine(x, s, &scale, &shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_resolve_by_name_not_position() {
        let a: &[(&str, &str)] = &[("pad2d", ""), ("crop2d", ""), ("mul", "")];
        let b: &[(&str, &str)] = &[("mul", ""), ("pad2d", ""), ("crop2d", "")];
        let ia = ImagingKernelIds::resolve(a);
        let ib = ImagingKernelIds::resolve(b);
        assert_eq!((ia.pad2d, ia.crop2d, ia.mul), (0, 1, 2));
        assert_eq!((ib.pad2d, ib.crop2d, ib.mul), (1, 2, 0));
        assert_eq!(a[ia.pad2d].0, "pad2d");
        assert_eq!(b[ib.pad2d].0, "pad2d");
    }

    #[test]
    fn unregistered_kernels_resolve_to_none() {
        let ids = ImagingKernelIds::resolve(&[("mul", "")]);
        assert_eq!(ids.resize_bicubic, NONE);
        assert_eq!(ids.need(ids.mul, "mul"), 0);
    }

    #[test]
    #[should_panic(expected = "kernel `resize_bicubic` is not registered")]
    fn need_panics_with_the_kernel_name() {
        let ids = ImagingKernelIds::resolve(&[("mul", "")]);
        ids.need(ids.resize_bicubic, "resize_bicubic");
    }

    /// Every name in `PIPELINES` must exist in the kernel registry, and the
    /// whole set must resolve. A typo here would otherwise surface as a
    /// `need()` panic at some caller's first frame.
    #[test]
    fn pipelines_names_are_all_real_kernels() {
        for (name, src) in PIPELINES {
            assert_eq!(kernels::src(name), *src, "PIPELINES entry `{name}` is not that kernel");
        }
        let ids = ImagingKernelIds::resolve(PIPELINES);
        for (i, (name, _)) in PIPELINES.iter().enumerate() {
            let got = match *name {
                "resize_nearest" => ids.resize_nearest,
                "resize_bilinear" => ids.resize_bilinear,
                "resize_bicubic" => ids.resize_bicubic,
                "pad2d" => ids.pad2d,
                "crop2d" => ids.crop2d,
                "nchw_nlc" => ids.nchw_nlc,
                "nlc_nchw" => ids.nlc_nchw,
                "film_chan" => ids.film_chan,
                "mul" => ids.mul,
                "add2" => ids.add2,
                "add_inplace" => ids.add_inplace,
                "axpy" => ids.axpy,
                "maxpool2d" => ids.maxpool2d,
                "avgpool2d" => ids.avgpool2d,
                "conv2d_gd" => ids.conv2d_gd,
                "bsq_quantize" => ids.bsq_quantize,
                other => panic!("PIPELINES has `{other}` but ImagingKernelIds has no field for it"),
            };
            assert_eq!(got, i, "`{name}` resolved to the wrong slot");
        }
    }

    #[test]
    fn align_corners_words_match_the_kernel_header() {
        assert_eq!(AlignCorners::HalfPixel.word(), 0);
        assert_eq!(AlignCorners::Corners.word(), 1);
        assert_eq!(AlignCorners::default(), AlignCorners::HalfPixel);
        assert_eq!(Filter::default(), Filter::Bilinear);
    }

    #[test]
    fn centre_border_puts_the_extra_pixel_right_and_bottom() {
        // 479 content in a 640 square: 80 above, 81 below — the odd-pad case.
        let b = Border::centre(640, 479, 640, 640);
        assert_eq!((b.top, b.bottom, b.left, b.right), (80, 81, 0, 0));
        let u = Border::uniform(3);
        assert_eq!((u.left, u.right, u.top, u.bottom), (3, 3, 3, 3));
    }
}
