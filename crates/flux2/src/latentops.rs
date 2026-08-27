// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side surgery on a FLUX.2 VAE latent, plus the decoded-image metrics
//! that make an edit's effect a number rather than an impression.
//!
//! The object every operation here takes is the VAE posterior **mean**
//! `[32, H/8, W/8]` - what [`vae::VaeEncoder::encode_mean`] produces and what
//! [`vae::VaeDecoder::decode`] consumes. The DiT never sees that tensor
//! directly: it sees [`vae::latent::pack`]'s 2x2 pixel-unshuffle followed by a
//! frozen per-channel BatchNorm, `[128, H/16, W/16]`. That map is a *reshape
//! composed with a per-channel affine*, so every **linear** operation in this
//! module (blend, splice, masked mixing) commutes with it exactly - mixing two
//! latents here and mixing the same two in DiT-packed space give the same
//! image. The spatial operations do not commute quite so freely: a horizontal
//! flip here is a flip *plus a swap of the two column sub-channels* there, so
//! it is representable but is not the naive packed-space flip.
//!
//! Everything is host math on a cold path - one pass over a tensor that is
//! four orders of magnitude smaller than the activations the VAE graph moves -
//! so there is nothing here for a kernel to do.
//!
//! Swedish Embedded AB implements latent-space analysis and editing tooling for
//! generative-imaging pipelines for its clients. If your team needs expertise in
//! diffusion latent spaces then you can procure our services by sending an email
//! to info@swedishembedded.com.

use std::io::{Read, Write};

/// What a stored latent file holds, so a reader never has to guess from the
/// channel count alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// VAE posterior mean, `[32, H/8, W/8]`, un-normalized - the encoder's
    /// output and the decoder's input.
    VaeMean,
    /// RGB samples in `[0, 255]`, `[3, H, W]` - a decoded image kept in
    /// latent-file form so that pixel-space and latent-space edits can be
    /// driven by exactly the same code and cannot differ by an implementation
    /// detail.
    ///
    /// The units are 8-bit levels rather than the VAE's `[-1, 1]` on purpose:
    /// `v/127.5 − 1` is **not** losslessly invertible in f32 (the subtraction
    /// costs a mantissa bit, and every level in 1..=63 comes back one lower),
    /// which would give the pixel-space control arm of an experiment a
    /// systematic darkening of its own. Levels are what the metrics measure, so
    /// levels are what this container holds.
    Rgb,
}

impl Kind {
    fn tag(self) -> u32 {
        match self {
            Kind::VaeMean => 0,
            Kind::Rgb => 1,
        }
    }

    fn from_tag(t: u32) -> Result<Kind, String> {
        match t {
            0 => Ok(Kind::VaeMean),
            1 => Ok(Kind::Rgb),
            _ => Err(format!("unknown latent kind tag {t}")),
        }
    }

    /// Pixels per cell along each axis: 8 for the VAE latent grid (the encoder
    /// downsamples 3 times), 1 for an image.
    pub fn stride(self) -> usize {
        match self {
            Kind::VaeMean => 8,
            Kind::Rgb => 1,
        }
    }
}

/// A `[c, h, w]` f32 tensor with enough header to be read back without a
/// side-channel: magic, kind, dims, then `c*h*w` little-endian f32.
#[derive(Clone, Debug, PartialEq)]
pub struct Latent {
    pub kind: Kind,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

const MAGIC: &[u8; 8] = b"BRNLAT\x00\x01";

impl Latent {
    pub fn new(kind: Kind, c: usize, h: usize, w: usize, data: Vec<f32>) -> Result<Latent, String> {
        if data.len() != c * h * w {
            return Err(format!("latent: {} values for {c}x{h}x{w}", data.len()));
        }
        Ok(Latent { kind, c, h, w, data })
    }

    /// Serialize to `magic ‖ kind ‖ c ‖ h ‖ w ‖ data`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + 4 * self.data.len());
        out.extend_from_slice(MAGIC);
        for v in [self.kind.tag(), self.c as u32, self.h as u32, self.w as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Latent, String> {
        if bytes.len() < 24 || &bytes[..8] != MAGIC {
            return Err("not a brain latent file (bad magic)".into());
        }
        let u = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let kind = Kind::from_tag(u(8) as u32)?;
        let (c, h, w) = (u(12), u(16), u(20));
        let n = c * h * w;
        if bytes.len() != 24 + 4 * n {
            return Err(format!("latent: {} payload bytes for {c}x{h}x{w}", bytes.len() - 24));
        }
        let data =
            (0..n).map(|i| f32::from_le_bytes(bytes[24 + 4 * i..28 + 4 * i].try_into().unwrap())).collect();
        Latent::new(kind, c, h, w, data)
    }

    pub fn load(path: &str) -> Result<Latent, String> {
        let mut buf = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .map_err(|e| format!("reading {path}: {e}"))?;
        Latent::from_bytes(&buf).map_err(|e| format!("{path}: {e}"))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        if let Some(d) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| format!("creating {}: {e}", d.display()))?;
        }
        std::fs::File::create(path)
            .and_then(|mut f| f.write_all(&self.to_bytes()))
            .map_err(|e| format!("writing {path}: {e}"))
    }

    /// Per-channel `(mean, std)` - the population std, `sqrt(E[x²] − E[x]²)`.
    pub fn channel_stats(&self) -> Vec<(f32, f32)> {
        let plane = self.h * self.w;
        (0..self.c)
            .map(|ci| {
                let s = &self.data[ci * plane..(ci + 1) * plane];
                let m = s.iter().map(|&v| v as f64).sum::<f64>() / plane as f64;
                let var = s.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>() / plane as f64;
                (m as f32, var.sqrt() as f32)
            })
            .collect()
    }

    fn same_shape(&self, other: &Latent, what: &str) -> Result<(), String> {
        if (self.kind, self.c, self.h, self.w) != (other.kind, other.c, other.h, other.w) {
            return Err(format!(
                "{what}: shape mismatch {:?}{}x{}x{} vs {:?}{}x{}x{}",
                self.kind, self.c, self.h, self.w, other.kind, other.c, other.h, other.w
            ));
        }
        Ok(())
    }
}

/// `(1 − alpha)·a + alpha·b`, elementwise. `alpha = 0` is `a`, `1` is `b`.
pub fn blend(a: &Latent, b: &Latent, alpha: f32) -> Result<Latent, String> {
    a.same_shape(b, "blend")?;
    let data = a.data.iter().zip(&b.data).map(|(&x, &y)| lerp(x, y, alpha)).collect();
    Latent::new(a.kind, a.c, a.h, a.w, data)
}

/// Linear interpolation that is **bit-exact at the endpoints**. Without the
/// branch, `x + 1.0·(y − x)` is only `y` to within a rounding error, and an
/// unmixed region of a splice would then differ from the untouched latent in
/// the last bit - which decodes to a nonzero MAD and makes a control run look
/// like an effect.
fn lerp(x: f32, y: f32, t: f32) -> f32 {
    if t == 0.0 {
        x
    } else if t == 1.0 {
        y
    } else {
        x + t * (y - x)
    }
}

/// `(1 − m)·a + m·b` with a per-cell mask `m` of length `h*w`, broadcast over
/// channels - the general form [`blend`] and [`splice`] are both special cases
/// of.
pub fn mix(a: &Latent, b: &Latent, mask: &[f32]) -> Result<Latent, String> {
    a.same_shape(b, "mix")?;
    if mask.len() != a.h * a.w {
        return Err(format!("mix: mask has {} cells, latent has {}", mask.len(), a.h * a.w));
    }
    let plane = a.h * a.w;
    let mut data = vec![0.0f32; a.data.len()];
    for ci in 0..a.c {
        for i in 0..plane {
            let (x, y) = (a.data[ci * plane + i], b.data[ci * plane + i]);
            data[ci * plane + i] = lerp(x, y, mask[i]);
        }
    }
    Latent::new(a.kind, a.c, a.h, a.w, data)
}

/// A rectangle in **pixel** coordinates of the full-resolution image, which is
/// the only frame in which "misaligned by half a cell" means anything.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
}

/// The soft splice mask for `rect`, evaluated at pixel resolution and then
/// **box-averaged down** to the latent grid.
///
/// The averaging is the whole point: a cell only partly covered by the
/// rectangle comes out with a fractional weight, so the spliced latent holds a
/// linear blend of two unrelated latents in exactly that cell. That is the
/// mechanism a half-cell-misaligned splice is supposed to expose, so it is
/// modelled rather than rounded away.
///
/// `feather` is a full ramp width in pixels: the mask goes 0 → 1 linearly over
/// `feather` pixels centred on the rectangle's boundary. `feather = 0` is a
/// hard edge.
pub fn splice_mask(kind: Kind, h: usize, w: usize, rect: Rect, feather: f32) -> Vec<f32> {
    let s = kind.stride();
    let (ph, pw) = (h * s, w * s);
    let mut acc = vec![0.0f32; h * w];
    for py in 0..ph {
        for px in 0..pw {
            // Signed distance to the rectangle boundary, positive inside,
            // on the separable (per-axis minimum) convention.
            let cx = px as f64 + 0.5;
            let cy = py as f64 + 0.5;
            let dx = (cx - rect.x as f64).min(rect.x as f64 + rect.w as f64 - cx);
            let dy = (cy - rect.y as f64).min(rect.y as f64 + rect.h as f64 - cy);
            let d = dx.min(dy);
            let m = if feather > 0.0 {
                (0.5 + d / feather as f64).clamp(0.0, 1.0) as f32
            } else {
                f32::from(d > 0.0)
            };
            acc[(py / s) * w + px / s] += m;
        }
    }
    let inv = 1.0 / (s * s) as f32;
    for v in &mut acc {
        *v *= inv;
    }
    acc
}

/// Paste `b`'s content into `a` over `rect`, with an optional feather. Both
/// arguments must share a shape; the rectangle is in pixels.
pub fn splice(a: &Latent, b: &Latent, rect: Rect, feather: f32) -> Result<Latent, String> {
    a.same_shape(b, "splice")?;
    let m = splice_mask(a.kind, a.h, a.w, rect, feather);
    mix(a, b, &m)
}

/// Mirror left-right.
pub fn flip_h(x: &Latent) -> Latent {
    let mut out = x.clone();
    for ci in 0..x.c {
        for y in 0..x.h {
            for j in 0..x.w {
                out.data[(ci * x.h + y) * x.w + j] = x.data[(ci * x.h + y) * x.w + (x.w - 1 - j)];
            }
        }
    }
    out
}

/// Mirror top-bottom.
pub fn flip_v(x: &Latent) -> Latent {
    let mut out = x.clone();
    for ci in 0..x.c {
        for y in 0..x.h {
            for j in 0..x.w {
                out.data[(ci * x.h + y) * x.w + j] = x.data[(ci * x.h + (x.h - 1 - y)) * x.w + j];
            }
        }
    }
    out
}

/// `k` quarter-turns **clockwise**. The result's `h`/`w` swap for odd `k`.
pub fn rot90(x: &Latent, k: i64) -> Latent {
    let k = k.rem_euclid(4);
    let mut cur = x.clone();
    for _ in 0..k {
        let (h, w) = (cur.h, cur.w);
        let mut data = vec![0.0f32; cur.data.len()];
        // clockwise: out[j, h-1-i] = in[i, j]
        for ci in 0..cur.c {
            for i in 0..h {
                for j in 0..w {
                    data[(ci * w + j) * h + (h - 1 - i)] = cur.data[(ci * h + i) * w + j];
                }
            }
        }
        cur = Latent { kind: cur.kind, c: cur.c, h: w, w: h, data };
    }
    cur
}

/// Circular shift by `(dy, dx)` cells (positive = down / right).
pub fn roll(x: &Latent, dy: i64, dx: i64) -> Latent {
    let (h, w) = (x.h as i64, x.w as i64);
    let mut out = x.clone();
    for ci in 0..x.c {
        for y in 0..x.h {
            for j in 0..x.w {
                let sy = (y as i64 - dy).rem_euclid(h) as usize;
                let sx = (j as i64 - dx).rem_euclid(w) as usize;
                out.data[(ci * x.h + y) * x.w + j] = x.data[(ci * x.h + sy) * x.w + sx];
            }
        }
    }
    out
}

/// Rotate `deg` degrees clockwise about the centre, bilinear, edge-clamped,
/// output the same size as the input.
pub fn rotate(x: &Latent, deg: f32) -> Latent {
    let (h, w) = (x.h as f64, x.w as f64);
    let (cy, cx) = ((h - 1.0) / 2.0, (w - 1.0) / 2.0);
    let t = (deg as f64).to_radians();
    let (s, c) = (t.sin(), t.cos());
    let mut out = x.clone();
    for ci in 0..x.c {
        let plane = &x.data[ci * x.h * x.w..(ci + 1) * x.h * x.w];
        for y in 0..x.h {
            for j in 0..x.w {
                // Inverse map: rotating the image clockwise by t samples the
                // source at the counter-clockwise-rotated coordinate.
                let (dy, dx) = (y as f64 - cy, j as f64 - cx);
                let sy = cy + (-s * dx + c * dy);
                let sx = cx + (c * dx + s * dy);
                out.data[(ci * x.h + y) * x.w + j] = sample_bilinear(plane, x.h, x.w, sy, sx);
            }
        }
    }
    out
}

fn sample_bilinear(plane: &[f32], h: usize, w: usize, y: f64, x: f64) -> f32 {
    let y = y.clamp(0.0, h as f64 - 1.0);
    let x = x.clamp(0.0, w as f64 - 1.0);
    let (y0, x0) = (y.floor() as usize, x.floor() as usize);
    let (y1, x1) = ((y0 + 1).min(h - 1), (x0 + 1).min(w - 1));
    let (fy, fx) = ((y - y0 as f64) as f32, (x - x0 as f64) as f32);
    let a = plane[y0 * w + x0];
    let b = plane[y0 * w + x1];
    let c = plane[y1 * w + x0];
    let d = plane[y1 * w + x1];
    let top = a + fx * (b - a);
    let bot = c + fx * (d - c);
    top + fy * (bot - top)
}

/// What to do to one channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChanOp {
    /// Set every cell to 0.0 - the *raw* zero, which is not the channel's
    /// neutral value: the posterior mean has a per-channel DC offset.
    Zero,
    /// Set every cell to the channel's own spatial mean - removes the
    /// channel's spatial information while keeping its DC level.
    Mean,
    /// Multiply raw values by `f` (scales the DC offset too).
    Scale(f32),
    /// Multiply only the deviation from the channel mean by `f`.
    ScaleCentred(f32),
}

/// Apply `op` to channel `ci`.
pub fn channel_op(x: &Latent, ci: usize, op: ChanOp) -> Result<Latent, String> {
    if ci >= x.c {
        return Err(format!("channel {ci} out of range (c = {})", x.c));
    }
    let plane = x.h * x.w;
    let mut out = x.clone();
    let s = &mut out.data[ci * plane..(ci + 1) * plane];
    let mean = s.iter().map(|&v| v as f64).sum::<f64>() as f32 / plane as f32;
    for v in s.iter_mut() {
        *v = match op {
            ChanOp::Zero => 0.0,
            ChanOp::Mean => mean,
            ChanOp::Scale(f) => *v * f,
            ChanOp::ScaleCentred(f) => mean + (*v - mean) * f,
        };
    }
    Ok(out)
}

/// How `sigma` is interpreted by [`add_noise`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseUnits {
    /// `sigma` multiplies each channel's **own** measured spatial std, so the
    /// same `sigma` is the same relative perturbation in every channel - the
    /// only unit in which "which channel is most noise-sensitive" is a fair
    /// question.
    PerChannelStd,
    /// `sigma` multiplies the std measured over the whole latent.
    GlobalStd,
    /// `sigma` is in raw latent units.
    Raw,
}

/// Separable `k`x`k` box blur of one plane, wrapping at the edges, then
/// renormalized to unit variance. This turns white noise into noise with a
/// correlation length of roughly `k` cells while leaving its amplitude
/// comparable - the control that separates "how far the latent moved" from
/// "in what kind of direction it moved".
fn smooth_unit(plane: &mut [f32], h: usize, w: usize, k: usize) {
    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0f32;
            for d in 0..k {
                s += plane[y * w + (x + d + w - k / 2) % w];
            }
            tmp[y * w + x] = s / k as f32;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0f32;
            for d in 0..k {
                s += tmp[((y + d + h - k / 2) % h) * w + x];
            }
            plane[y * w + x] = s / k as f32;
        }
    }
    let n = (h * w) as f64;
    let m = plane.iter().map(|&v| v as f64).sum::<f64>() / n;
    let sd = (plane.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>() / n).sqrt();
    if sd > 0.0 {
        for v in plane.iter_mut() {
            *v = ((*v as f64 - m) / sd) as f32;
        }
    }
}

/// Add zero-mean Gaussian noise. `only` restricts it to a single channel;
/// `smooth` (> 1) gives the noise a spatial correlation length of that many
/// latent cells at the same amplitude.
pub fn add_noise(
    x: &Latent,
    sigma: f32,
    units: NoiseUnits,
    seed: u64,
    only: Option<usize>,
    smooth: usize,
) -> Latent {
    let plane = x.h * x.w;
    let stats = x.channel_stats();
    let global = {
        let n = x.data.len() as f64;
        let m = x.data.iter().map(|&v| v as f64).sum::<f64>() / n;
        (x.data.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>() / n).sqrt() as f32
    };
    let mut z = model::hostmath::randn(x.data.len(), seed);
    if smooth > 1 {
        for ci in 0..x.c {
            smooth_unit(&mut z[ci * plane..(ci + 1) * plane], x.h, x.w, smooth);
        }
    }
    let mut out = x.clone();
    for ci in 0..x.c {
        if only.is_some_and(|k| k != ci) {
            continue;
        }
        let scale = sigma
            * match units {
                NoiseUnits::PerChannelStd => stats[ci].1,
                NoiseUnits::GlobalStd => global,
                NoiseUnits::Raw => 1.0,
            };
        for i in 0..plane {
            out.data[ci * plane + i] += scale * z[ci * plane + i];
        }
    }
    out
}

// ===================== decoded-image metrics =====================

/// Comparison of two same-size RGB images, all four numbers defined here so a
/// reported figure never has to be reverse-engineered from the code.
///
/// Every number is computed on the **u8 sample values, 0..255**, of the two
/// images, over all `h*w*3` samples unless said otherwise:
///
/// * `mad` - mean of `|a − b|`. Units: 8-bit levels.
/// * `rel_l2` - `‖a − b‖₂ / ‖b‖₂`, i.e. relative to the *second* argument,
///   which is by convention the reference.
/// * `cosine` - `⟨a, b⟩ / (‖a‖·‖b‖)`, **uncentred**. On non-negative pixel data
///   this saturates near 1 for anything remotely similar, which is exactly why
///   it is never reported alone.
/// * `edge_corr` - Pearson correlation between the two images' Sobel gradient
///   magnitudes, computed on Rec.601 luma (`0.299R + 0.587G + 0.114B`) over the
///   interior (the 1-pixel border, where the 3x3 Sobel window would hang off
///   the image, is excluded). This is the structural metric: it ignores a
///   global brightness or contrast shift and reports whether the *edges* still
///   land in the same places with the same relative strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageMetrics {
    pub mad: f64,
    pub rel_l2: f64,
    pub cosine: f64,
    pub edge_corr: f64,
}

/// Compare `a` against reference `b`, both interleaved RGB u8 of size `w`x`h`.
pub fn compare(a: &[u8], b: &[u8], h: usize, w: usize) -> Result<ImageMetrics, String> {
    if a.len() != h * w * 3 || b.len() != h * w * 3 {
        return Err(format!("compare: {} / {} bytes for {w}x{h} RGB", a.len(), b.len()));
    }
    let mut sad = 0.0f64;
    let (mut d2, mut bb, mut aa, mut ab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        sad += (x - y).abs();
        d2 += (x - y) * (x - y);
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    let n = a.len() as f64;
    Ok(ImageMetrics {
        mad: sad / n,
        rel_l2: if bb > 0.0 { d2.sqrt() / bb.sqrt() } else { f64::NAN },
        cosine: if aa > 0.0 && bb > 0.0 { ab / (aa.sqrt() * bb.sqrt()) } else { f64::NAN },
        edge_corr: pearson(&sobel_mag(a, h, w), &sobel_mag(b, h, w)),
    })
}

/// Rec.601 luma Sobel gradient magnitude over the interior pixels.
fn sobel_mag(px: &[u8], h: usize, w: usize) -> Vec<f64> {
    let luma: Vec<f64> = (0..h * w)
        .map(|i| 0.299 * px[3 * i] as f64 + 0.587 * px[3 * i + 1] as f64 + 0.114 * px[3 * i + 2] as f64)
        .collect();
    if h < 3 || w < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((h - 2) * (w - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |dy: usize, dx: usize| luma[(y + dy - 1) * w + (x + dx - 1)];
            let gx = -p(0, 0) - 2.0 * p(1, 0) - p(2, 0) + p(0, 2) + 2.0 * p(1, 2) + p(2, 2);
            let gy = -p(0, 0) - 2.0 * p(0, 1) - p(0, 2) + p(2, 0) + 2.0 * p(2, 1) + p(2, 2);
            out.push((gx * gx + gy * gy).sqrt());
        }
    }
    out
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::NAN;
    }
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return f64::NAN;
    }
    sab / (saa.sqrt() * sbb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(kind: Kind, c: usize, h: usize, w: usize, seed: u64) -> Latent {
        Latent::new(kind, c, h, w, model::hostmath::randn(c * h * w, seed)).unwrap()
    }

    #[test]
    fn a_latent_file_round_trips_through_its_own_header() {
        let x = fixture(Kind::VaeMean, 32, 6, 4, 7);
        let back = Latent::from_bytes(&x.to_bytes()).unwrap();
        assert_eq!(back, x);
        // Self-describing: the header alone, not the caller, decides the shape.
        assert!(Latent::from_bytes(&x.to_bytes()[..40]).is_err());
        assert!(Latent::from_bytes(b"not a latent file at all").is_err());
    }

    #[test]
    fn blend_reaches_both_endpoints_exactly() {
        let (a, b) = (fixture(Kind::VaeMean, 4, 3, 5, 1), fixture(Kind::VaeMean, 4, 3, 5, 2));
        assert_eq!(blend(&a, &b, 0.0).unwrap().data, a.data);
        assert_eq!(blend(&a, &b, 1.0).unwrap().data, b.data);
        let half = blend(&a, &b, 0.5).unwrap();
        for i in 0..a.data.len() {
            assert!((half.data[i] - 0.5 * (a.data[i] + b.data[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn spatial_ops_are_invertible() {
        let x = fixture(Kind::VaeMean, 3, 6, 4, 3);
        assert_eq!(flip_h(&flip_h(&x)).data, x.data);
        assert_eq!(flip_v(&flip_v(&x)).data, x.data);
        assert_eq!(rot90(&rot90(&x, 1), 3).data, x.data);
        assert_eq!(rot90(&x, 4).data, x.data);
        assert_eq!(roll(&x, 6, 4).data, x.data);
        assert_eq!(roll(&roll(&x, 2, -3), -2, 3).data, x.data);
        // An odd number of quarter turns transposes the grid.
        let r = rot90(&x, 1);
        assert_eq!((r.h, r.w), (x.w, x.h));
        // Clockwise, pinned on a corner: the top-left cell ends up top-right.
        assert_eq!(r.data[r.w - 1], x.data[0]);
        // 90 degrees of `rotate` on a square grid agrees with `rot90`.
        let sq = fixture(Kind::VaeMean, 2, 5, 5, 4);
        let (bilinear, exact) = (rotate(&sq, 90.0), rot90(&sq, 1));
        for i in 0..sq.data.len() {
            assert!((bilinear.data[i] - exact.data[i]).abs() < 1e-5, "at {i}");
        }
    }

    #[test]
    fn a_splice_mask_records_fractional_cell_coverage() {
        // A hard-edged rectangle on exact 8px cell boundaries: every cell is
        // fully in or fully out, no cell holds a mixture.
        let m = splice_mask(Kind::VaeMean, 4, 4, Rect { x: 8, y: 8, w: 16, h: 16 }, 0.0);
        for v in &m {
            assert!(*v == 0.0 || *v == 1.0, "aligned splice produced partial cell {v}");
        }
        assert_eq!(m.iter().sum::<f32>(), 4.0);

        // The same rectangle shifted half a cell: the boundary column and row
        // now hold half of each latent.
        let m = splice_mask(Kind::VaeMean, 4, 4, Rect { x: 4, y: 4, w: 16, h: 16 }, 0.0);
        assert!(m.iter().any(|v| *v > 0.0 && *v < 1.0), "misaligned splice produced no partial cell");
        assert!((m.iter().sum::<f32>() - 4.0).abs() < 1e-5, "coverage area must be preserved");

        // Feathering widens the transition band. Coverage is broadly preserved
        // (the per-axis-minimum convention loses a little at the corners), and
        // no cell is left fully saturated on the boundary.
        let hard = splice_mask(Kind::VaeMean, 8, 8, Rect { x: 16, y: 16, w: 32, h: 32 }, 0.0);
        let f = splice_mask(Kind::VaeMean, 8, 8, Rect { x: 16, y: 16, w: 32, h: 32 }, 16.0);
        assert_eq!(hard.iter().filter(|v| **v > 0.01 && **v < 0.99).count(), 0);
        assert!((f.iter().sum::<f32>() - hard.iter().sum::<f32>()).abs() < 0.2 * hard.iter().sum::<f32>());
        assert!(f.iter().filter(|v| **v > 0.01 && **v < 0.99).count() > 8);
    }

    #[test]
    fn splice_degenerates_to_its_endpoints() {
        let (a, b) = (fixture(Kind::VaeMean, 4, 4, 4, 11), fixture(Kind::VaeMean, 4, 4, 4, 12));
        let all = splice(&a, &b, Rect { x: 0, y: 0, w: 32, h: 32 }, 0.0).unwrap();
        assert_eq!(all.data, b.data);
        let none = splice(&a, &b, Rect { x: 0, y: 0, w: 0, h: 0 }, 0.0).unwrap();
        assert_eq!(none.data, a.data);
    }

    #[test]
    fn channel_ops_touch_exactly_one_channel() {
        let x = fixture(Kind::VaeMean, 5, 3, 3, 21);
        let plane = 9;
        for (op, name) in [
            (ChanOp::Zero, "zero"),
            (ChanOp::Mean, "mean"),
            (ChanOp::Scale(2.0), "scale"),
            (ChanOp::ScaleCentred(0.0), "cscale"),
        ] {
            let y = channel_op(&x, 2, op).unwrap();
            assert_eq!(y.data[..2 * plane], x.data[..2 * plane], "{name} leaked below");
            assert_eq!(y.data[3 * plane..], x.data[3 * plane..], "{name} leaked above");
            assert_ne!(y.data[2 * plane..3 * plane], x.data[2 * plane..3 * plane], "{name} was a no-op");
        }
        // `Mean` and `ScaleCentred(0)` are the same edit; `Zero` is not, because
        // the channel's DC level is not zero.
        let m = channel_op(&x, 2, ChanOp::Mean).unwrap();
        let cs = channel_op(&x, 2, ChanOp::ScaleCentred(0.0)).unwrap();
        for i in 0..x.data.len() {
            assert!((m.data[i] - cs.data[i]).abs() < 1e-6);
        }
        assert!(channel_op(&x, 5, ChanOp::Zero).is_err());
    }

    #[test]
    fn noise_is_seeded_scaled_and_channel_scoped() {
        let mut x = fixture(Kind::VaeMean, 4, 16, 16, 31);
        // Give channel 3 ten times the spread of the others.
        for v in &mut x.data[3 * 256..] {
            *v *= 10.0;
        }
        let a = add_noise(&x, 0.25, NoiseUnits::PerChannelStd, 5, None, 1);
        assert_eq!(a.data, add_noise(&x, 0.25, NoiseUnits::PerChannelStd, 5, None, 1).data);
        assert_ne!(a.data, add_noise(&x, 0.25, NoiseUnits::PerChannelStd, 6, None, 1).data);

        // Per-channel units make the *relative* perturbation equal despite the
        // 10x scale difference; raw units do not.
        let rel = |y: &Latent, ci: usize| {
            let (s, e) = (ci * 256, (ci + 1) * 256);
            let d: f64 = (s..e).map(|i| ((y.data[i] - x.data[i]) as f64).powi(2)).sum();
            let n: f64 = (s..e).map(|i| (x.data[i] as f64).powi(2)).sum();
            (d / n).sqrt()
        };
        assert!((rel(&a, 0) / rel(&a, 3) - 1.0).abs() < 0.35, "per-channel units are not relative");
        let raw = add_noise(&x, 0.25, NoiseUnits::Raw, 5, None, 1);
        assert!(rel(&raw, 0) / rel(&raw, 3) > 5.0, "raw units should hit the small channel far harder");

        let one = add_noise(&x, 0.25, NoiseUnits::PerChannelStd, 5, Some(1), 1);
        assert_eq!(one.data[..256], x.data[..256]);
        assert_ne!(one.data[256..512], x.data[256..512]);
        assert_eq!(one.data[512..], x.data[512..]);
    }

    /// Smoothing must change the noise's spatial *correlation* while leaving
    /// its *amplitude* alone - otherwise "same displacement, different
    /// direction" is not a controlled comparison but a weaker perturbation.
    #[test]
    fn smoothed_noise_keeps_its_amplitude_and_gains_correlation() {
        let zero = Latent::new(Kind::VaeMean, 2, 64, 64, vec![0.0; 2 * 64 * 64]).unwrap();
        // A zero latent has zero std, so per-channel units would scale to
        // nothing; raw units make the amplitude claim checkable.
        let white = add_noise(&zero, 1.0, NoiseUnits::Raw, 9, None, 1);
        let corr = add_noise(&zero, 1.0, NoiseUnits::Raw, 9, None, 4);
        let rms = |l: &Latent| (l.data.iter().map(|&v| (v as f64) * v as f64).sum::<f64>() / l.data.len() as f64).sqrt();
        assert!((rms(&white) / rms(&corr) - 1.0).abs() < 0.05, "smoothing changed the amplitude");
        let lag1 = |l: &Latent| {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for ci in 0..l.c {
                for y in 0..l.h {
                    for x in 1..l.w {
                        let (a, b) = (l.data[(ci * l.h + y) * l.w + x], l.data[(ci * l.h + y) * l.w + x - 1]);
                        num += a as f64 * b as f64;
                        den += a as f64 * a as f64;
                    }
                }
            }
            num / den
        };
        assert!(lag1(&white).abs() < 0.1, "white noise should be uncorrelated, got {}", lag1(&white));
        assert!(lag1(&corr) > 0.5, "smoothed noise should be correlated, got {}", lag1(&corr));
    }

    /// The metric gate itself is under test: identical images must score
    /// perfectly, and each metric must actually move for the kind of damage it
    /// exists to catch.
    #[test]
    fn the_metrics_agree_on_identity_and_disagree_on_damage() {
        let (h, w) = (24usize, 32usize);
        // Deliberately asymmetric: a mirror of this image must not look like
        // the original to any metric here.
        let base: Vec<u8> = (0..h * w * 3)
            .map(|i| {
                let (y, x, k) = (i / 3 / w, (i / 3) % w, i % 3);
                let block = usize::from(x < w / 3 && y < h / 2) * 90;
                ((x * 5 + y * 11 + k * 7 + block) % 200 + 20) as u8
            })
            .collect();
        let same = compare(&base, &base, h, w).unwrap();
        assert_eq!(same.mad, 0.0);
        assert_eq!(same.rel_l2, 0.0);
        assert!((same.cosine - 1.0).abs() < 1e-12);
        assert!((same.edge_corr - 1.0).abs() < 1e-12);

        // A pure brightness lift: MAD and rel_l2 move, cosine barely does, and
        // edge_corr does not move at all - that is the metric split this
        // struct exists to make visible.
        let lifted: Vec<u8> = base.iter().map(|&v| v.saturating_add(10)).collect();
        let m = compare(&lifted, &base, h, w).unwrap();
        assert!(m.mad > 5.0 && m.rel_l2 > 0.02);
        assert!(m.cosine > 0.999, "uncentred cosine is nearly blind to a DC lift");
        assert!(m.edge_corr > 0.999, "a DC lift leaves every edge where it was");

        // Structure moved but the histogram kept exactly: edge_corr collapses.
        let mut flipped = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for k in 0..3 {
                    flipped[(y * w + x) * 3 + k] = base[(y * w + (w - 1 - x)) * 3 + k];
                }
            }
        }
        let m = compare(&flipped, &base, h, w).unwrap();
        assert!(m.edge_corr.abs() < 0.9, "edge_corr {} should not survive a flip", m.edge_corr);

        assert!(compare(&base, &base[..10], h, w).is_err());
    }
}
