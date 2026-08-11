// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bicubic resize forward (Catmull-Rom / cubic convolution, a = -0.75), NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Bicubic resize forward (Catmull-Rom / cubic convolution, a = -0.75), NCHW.
//   x : [N, C, H,  W ]   idx = ((n*C + c)*H  + hi)*W  + wi
//   y : [N, C, Ho, Wo]   idx = ((n*C + c)*Ho + ho)*Wo + wo
//
// One invocation per OUTPUT element. Ho/Wo are caller-computed and passed in.
//
// Sibling of resize_bilinear.wgsl / resize_nearest.wgsl — SAME Params struct,
// SAME binding order, SAME align_corners semantics. Only the tap stencil differs:
// 4x4 cubic instead of 2x2 linear. Needed because SAM 2 interpolates its
// windowed-attention pos_embed with F.interpolate(mode='bicubic'), and bicubic
// is NOT a cosmetic upgrade over bilinear there — the embedding is resampled
// once at load and every downstream feature inherits the error.
//
// --- the cubic kernel, and why a = -0.75 ---------------------------------------
// Keys' cubic convolution kernel, parameterised by `a`:
//   |s| <= 1 : ((a + 2)|s| - (a + 3))|s|^2 + 1
//   1 < |s| < 2 : ((a|s| - 5a)|s| + 8a)|s| - 4a
//   otherwise: 0
// PyTorch pins a = -0.75 in ATen: `get_cubic_upsample_coefficients` in
// aten/src/ATen/native/UpSample.h declares `scalar_t A = -0.75;` and builds the
// four taps as cubic_convolution2(t+1), cubic_convolution1(t),
// cubic_convolution1(1-t), cubic_convolution2(2-t) — which is exactly the two
// branches above evaluated at |s| = t+1, t, 1-t, 2-t. The CUDA path
// (UpSampleBicubic2d.cu) and the CPU vectorised path use the same -0.75. OpenCV's
// INTER_CUBIC uses -0.75 too; TensorFlow/`tf.image.resize(method='bicubic')` uses
// -0.5, so a model exported from PyTorch resampled with -0.5 is quietly wrong.
// ONNX Resize exposes this as the `cubic_coeff_a` attribute: emit -0.75 to match.
//
// NOT a GPU replacement for `mirror::preprocess::resize_bicubic_torch`. That host
// function is `F.interpolate(mode="bicubic", antialias=True)` — torch's
// ANTIALIASED resampler is a port of PIL's, so it uses a = -0.5, a scale-widened
// support, and weights RENORMALISED per output pixel. This kernel is the non-AA
// path (a = -0.75, fixed 4-tap support, no renormalisation). The two are different
// functions, not two implementations of one: retargeting DINOv2's pos-embed
// interpolation at this kernel would silently change mirror's outputs. Antialiased
// downsampling still has no kernel — it needs a scale-dependent tap count.
// The four weights sum to 1 for every t, so a constant image is reproduced
// exactly; that is the cheapest smoke test of the polynomial.
//
// --- align_corners --------------------------------------------------------------
// Identical to resize_bilinear.wgsl except for ONE deliberate difference, which
// is a real PyTorch behaviour and not an oversight:
//
//   align_corners = 1:  src = o * (in - 1) / (out - 1)        (out==1 -> 0)
//   align_corners = 0:  src = (o + 0.5) * (in / out) - 0.5    (NOT clamped >= 0)
//
// resize_bilinear clamps the half_pixel coordinate to >= 0. Bicubic must NOT:
// `area_pixel_compute_source_index` in UpSample.h takes a `bool cubic` argument
// and applies the `src_idx < 0 -> 0` clamp only when `!cubic` ("Note[Follow
// OpenCV resize logic]"). For bicubic the negative coordinate is kept and the
// out-of-range taps are handled by the tap clamp below instead. Clamping the
// coordinate as well would double-clamp the first output pixel and shift it by
// up to half a source pixel — plausible-looking, and invisible to a gradient
// check, because the kernel stays perfectly self-consistent while resampling the
// wrong grid. Only a parity test against the reference catches it.
//
// ONNX correspondence, so the engine and an exported graph compute the SAME
// function:
//   align_corners = 1  <->  coordinate_transformation_mode = "align_corners"
//   align_corners = 0  <->  coordinate_transformation_mode = "half_pixel"
//   both               <->  mode = "cubic", cubic_coeff_a = -0.75
//
// --- borders ---------------------------------------------------------------------
// Taps land at floor(src) + {-1, 0, +1, +2}, so up to two of them fall outside the
// image at every edge. PyTorch's `upsample_get_value_bounded` clamps the ACCESS
// (replicate/clamp-to-edge), it does not drop the tap or renormalise the weights.
// Reproduced here with clamp(., 0, dim-1). A dropped tap would renormalise the
// row implicitly and disagree with torch at every border pixel.
//
// --- structure -------------------------------------------------------------------
// The 4x4 stencil is written out flat: no var<function> array, no loop, no helper
// function. Three reasons, all load-bearing:
//   * a thread-private array indexed by a loop lands in local (global-backed)
//     memory unless the compiler can unroll every index — a documented
//     `flash_attn_bidir` trap;
//   * the wgsl-cpu Cranelift JIT inlines a single entry point and rejects
//     user-defined function calls outright, so a `cubic_w`/`src_coord` helper
//     would compile on wgpu and hard-fail on the CPU backend (same reason
//     resize_bilinear.wgsl writes its coordinate map out twice and gelu_erf.wgsl
//     inlines its erf);
//   * 16 named loads keep the tap set obvious to the reader of the adjoint.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
    align_corners: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, c, ho, wo).
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // --- source coordinate, y axis (inlined; see header). No max(.,0): cubic. ---
    var sy = 0.0;
    if (p.align_corners == 1u) {
        if (p.Ho > 1u) { sy = f32(ho) * f32(p.H - 1u) / f32(p.Ho - 1u); }
    } else {
        sy = (f32(ho) + 0.5) * (f32(p.H) / f32(p.Ho)) - 0.5;
    }
    // --- source coordinate, x axis ---
    var sx = 0.0;
    if (p.align_corners == 1u) {
        if (p.Wo > 1u) { sx = f32(wo) * f32(p.W - 1u) / f32(p.Wo - 1u); }
    } else {
        sx = (f32(wo) + 0.5) * (f32(p.W) / f32(p.Wo)) - 0.5;
    }

    // Base tap and fraction. floor() first, then i32: sy/sx can be negative under
    // half_pixel, and u32(floor(-0.25)) is not -1.
    let byf = floor(sy);
    let bxf = floor(sx);
    let by = i32(byf);
    let bx = i32(bxf);
    let ty = sy - byf;
    let tx = sx - bxf;

    // --- cubic convolution weights, a = -0.75 (PyTorch's A) ---
    // wy0..wy3 weight source rows by-1, by, by+1, by+2.
    let a = -0.75;
    let qy0 = ty + 1.0;                // |s| for tap -1, in (1, 2)
    let qy2 = 1.0 - ty;                // |s| for tap +1, in [0, 1]
    let qy3 = 2.0 - ty;                // |s| for tap +2, in (1, 2]
    let wy0 = ((a * qy0 - 5.0 * a) * qy0 + 8.0 * a) * qy0 - 4.0 * a;
    let wy1 = ((a + 2.0) * ty - (a + 3.0)) * ty * ty + 1.0;
    let wy2 = ((a + 2.0) * qy2 - (a + 3.0)) * qy2 * qy2 + 1.0;
    let wy3 = ((a * qy3 - 5.0 * a) * qy3 + 8.0 * a) * qy3 - 4.0 * a;

    let qx0 = tx + 1.0;
    let qx2 = 1.0 - tx;
    let qx3 = 2.0 - tx;
    let wx0 = ((a * qx0 - 5.0 * a) * qx0 + 8.0 * a) * qx0 - 4.0 * a;
    let wx1 = ((a + 2.0) * tx - (a + 3.0)) * tx * tx + 1.0;
    let wx2 = ((a + 2.0) * qx2 - (a + 3.0)) * qx2 * qx2 + 1.0;
    let wx3 = ((a * qx3 - 5.0 * a) * qx3 + 8.0 * a) * qx3 - 4.0 * a;

    // Clamp-to-edge taps (replicate border), matching upsample_get_value_bounded.
    let hm = i32(p.H) - 1;
    let wm = i32(p.W) - 1;
    let r0 = u32(clamp(by - 1, 0, hm));
    let r1 = u32(clamp(by,     0, hm));
    let r2 = u32(clamp(by + 1, 0, hm));
    let r3 = u32(clamp(by + 2, 0, hm));
    let c0 = u32(clamp(bx - 1, 0, wm));
    let c1 = u32(clamp(bx,     0, wm));
    let c2 = u32(clamp(bx + 1, 0, wm));
    let c3 = u32(clamp(bx + 2, 0, wm));

    let base = (n * p.C + c) * p.H;
    let b0 = (base + r0) * p.W;
    let b1 = (base + r1) * p.W;
    let b2 = (base + r2) * p.W;
    let b3 = (base + r3) * p.W;

    // Horizontal pass per row, then vertical — separable, exactly as ATen does it.
    let v0 = x[b0 + c0] * wx0 + x[b0 + c1] * wx1 + x[b0 + c2] * wx2 + x[b0 + c3] * wx3;
    let v1 = x[b1 + c0] * wx0 + x[b1 + c1] * wx1 + x[b1 + c2] * wx2 + x[b1 + c3] * wx3;
    let v2 = x[b2 + c0] * wx0 + x[b2 + c1] * wx1 + x[b2 + c2] * wx2 + x[b2 + c3] * wx3;
    let v3 = x[b3 + c0] * wx0 + x[b3 + c1] * wx1 + x[b3 + c2] * wx2 + x[b3 + c3] * wx3;

    y[idx] = v0 * wy0 + v1 * wy1 + v2 * wy2 + v3 * wy3;
}
