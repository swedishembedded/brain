// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A photograph as a MEASUREMENT: its pixels at the precision they were
//! recorded, in the colour encoding they were recorded in, the right way up,
//! and what the camera said about taking it.
//!
//! [`crate::load`] hands back 8-bit RGB, which is right for a model that
//! wants a picture. A reconstruction wants more: a 16-bit capture's extra
//! precision, the transfer curve and primaries its values are encoded in (so
//! filtering can happen in linear light and a Display-P3 photograph is not
//! mistaken for sRGB), the EXIF orientation applied rather than ignored, and
//! the metadata that constrains the geometry and photometry it will be
//! explained by:
//!
//! * focal length - through [`Exif::focal_px`], the prior that keeps
//!   structure from motion out of the wrong focal-length basin;
//! * make, model, lens and focal length - which photographs share one
//!   physical camera and so one calibration ([`Photo::sensor_key`]);
//! * shutter, aperture and ISO - each exposure's brightness relative to the
//!   others ([`Exif::exposure_log2`]), the known part of what a photometric
//!   camera model otherwise has to discover;
//! * GPS - a metric scale and a world frame where the capture spans enough
//!   ground for it to mean anything.
//!
//! EXIF is parsed here from its TIFF structure (CIPA DC-008, "Exchangeable
//! image file format"); ICC colour from ICC.1:2022 - the matrix/TRC form every
//! camera and phone profile uses (`rXYZ gXYZ bXYZ` and `rTRC gTRC bTRC`), which
//! is what converts a wide-gamut photograph into this crate's Rec.709 primaries.
//!
//! Swedish Embedded AB implements camera-faithful image pipelines for
//! photogrammetry and computer vision. If your team needs expertise in image
//! metadata and colour management then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::io::Cursor;
use std::path::Path;

use image::{DynamicImage, ImageDecoder};

use crate::pixels::Rgb8;

/// GPS position of a capture, WGS84.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gps {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    /// Metres above sea level, when recorded.
    pub altitude_m: Option<f64>,
}

/// What a photograph's EXIF says about how it was taken. Every field is
/// optional because every field is, in practice, sometimes missing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exif {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    /// EXIF orientation 1..8 as recorded (the pixels of a [`Photo`] already
    /// have it applied).
    pub orientation: Option<u16>,
    pub focal_mm: Option<f64>,
    pub focal_35mm: Option<f64>,
    pub exposure_s: Option<f64>,
    pub f_number: Option<f64>,
    pub iso: Option<f64>,
    /// `YYYY:MM:DD HH:MM:SS` as recorded, sub-seconds appended when present.
    pub datetime: Option<String>,
    pub gps: Option<Gps>,
}

impl Exif {
    /// The focal length in pixels of a `width x height` image (as oriented),
    /// from the 35 mm-equivalent focal length: 35 mm film's diagonal is
    /// 43.27 mm, and "equivalent" is defined on the diagonal.
    pub fn focal_px(&self, width: u32, height: u32) -> Option<f64> {
        let f35 = self.focal_35mm.filter(|v| *v > 0.0)?;
        let diag = ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
        Some(f35 * diag / (36.0f64.hypot(24.0)))
    }

    /// The exposure's light gathering in log2 stops: `log2(t * ISO / N²)`.
    /// Only differences between photographs mean anything - a photograph one
    /// stop higher than another recorded twice the light from the same
    /// radiance.
    pub fn exposure_log2(&self) -> Option<f64> {
        let (t, n, iso) = (self.exposure_s?, self.f_number?, self.iso?);
        (t > 0.0 && n > 0.0 && iso > 0.0).then(|| (t * iso / (n * n)).log2())
    }
}

/// How a photograph's stored values map to light.
#[derive(Clone, Debug, PartialEq)]
pub enum Transfer {
    /// The sRGB curve (IEC 61966-2-1): 8-bit JPEGs and PNGs with no profile,
    /// or with an sRGB / Display-P3 profile.
    Srgb,
    /// Values are proportional to light.
    Linear,
    /// A profile's own tone curve, per channel: encoded values sampled on a
    /// uniform grid of `len` points in [0,1], mapped to linear light.
    Table([Vec<f32>; 3]),
}

impl Transfer {
    /// Linear light of an encoded value in [0,1], channel `c`.
    pub fn decode(&self, c: usize, v: f32) -> f32 {
        match self {
            Transfer::Srgb => srgb_to_linear(v),
            Transfer::Linear => v,
            Transfer::Table(t) => {
                let lut = &t[c];
                let x = v.clamp(0.0, 1.0) * (lut.len() - 1) as f32;
                let i = (x as usize).min(lut.len() - 2);
                let f = x - i as f32;
                lut[i] * (1.0 - f) + lut[i + 1] * f
            }
        }
    }
}

/// The sRGB transfer, encoded -> linear.
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// The sRGB transfer, linear -> encoded.
pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.max(0.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// One photograph, oriented, at its recorded precision.
#[derive(Clone, Debug)]
pub struct Photo {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB as STORED, in [0,1] - 8- or 16-bit values at full
    /// precision, not yet linearized.
    pub encoded: Vec<f32>,
    /// Bits per sample of the source.
    pub bits: u8,
    pub transfer: Transfer,
    /// Linear RGB in the photograph's own primaries -> linear Rec.709 (sRGB
    /// primaries, D65); `None` when they already are.
    pub to_rec709: Option<[f32; 9]>,
    pub exif: Exif,
}

impl Photo {
    /// Linear-light Rec.709 RGB: what filtering, resampling and a
    /// scene-linear fit work in. Averaging ENCODED values is not averaging
    /// light - a box filter over an sRGB edge darkens it.
    pub fn linear(&self) -> Vec<f32> {
        let m = self.to_rec709;
        let mut out = Vec::with_capacity(self.encoded.len());
        for p in self.encoded.chunks_exact(3) {
            let l = [self.transfer.decode(0, p[0]), self.transfer.decode(1, p[1]), self.transfer.decode(2, p[2])];
            match m {
                None => out.extend_from_slice(&l),
                Some(m) => {
                    for r in 0..3 {
                        out.push(m[r * 3] * l[0] + m[r * 3 + 1] * l[1] + m[r * 3 + 2] * l[2]);
                    }
                }
            }
        }
        out
    }

    /// sRGB-encoded Rec.709 RGB in [0,1] (out-of-gamut values clamped): the
    /// display-referred form a default fit and every splat viewer use.
    pub fn display(&self) -> Vec<f32> {
        if self.transfer == Transfer::Srgb && self.to_rec709.is_none() {
            return self.encoded.clone();
        }
        self.linear().into_iter().map(|v| linear_to_srgb(v).clamp(0.0, 1.0)).collect()
    }

    /// 8-bit sRGB, for the stages that look at a picture (feature detection).
    pub fn rgb8(&self) -> Rgb8 {
        let px = self.display().iter().map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect();
        Rgb8 { w: self.width, h: self.height, px }
    }

    /// Which physical camera took it, as far as the metadata can tell:
    /// photographs with the same key share one calibration. Make, model,
    /// lens, focal length and image size - a phone's lenses are different
    /// sensors, and so is one zoom setting from another.
    pub fn sensor_key(&self) -> String {
        let e = &self.exif;
        format!(
            "{}|{}|{}|{:.2}|{}x{}",
            e.make.as_deref().unwrap_or("?"),
            e.model.as_deref().unwrap_or("?"),
            e.lens.as_deref().unwrap_or("?"),
            e.focal_mm.unwrap_or(0.0),
            self.width,
            self.height
        )
    }
}

/// Read a photograph from a file: JPEG, PNG (8 or 16 bit) or TIFF.
pub fn load_photo(path: impl AsRef<Path>) -> Result<Photo, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    decode_photo(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// [`load_photo`] from memory.
pub fn decode_photo(bytes: &[u8]) -> Result<Photo, String> {
    let reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| format!("sniffing: {e}"))?;
    let mut dec = reader.into_decoder().map_err(|e| format!("decoding: {e}"))?;
    let exif_raw = dec.exif_metadata().ok().flatten();
    let icc = dec.icc_profile().ok().flatten();
    let orientation = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let bits = (dec.original_color_type().bits_per_pixel() / dec.original_color_type().channel_count().max(1) as u16) as u8;
    let mut img = DynamicImage::from_decoder(dec).map_err(|e| format!("decoding: {e}"))?;
    img.apply_orientation(orientation);
    let rgb = img.to_rgb32f();
    let (width, height) = (rgb.width(), rgb.height());
    let (transfer, to_rec709) = match icc.as_deref().and_then(|p| icc_colour(p).ok()) {
        Some(c) => c,
        None => (Transfer::Srgb, None),
    };
    let mut exif = exif_raw.as_deref().map(parse_exif).unwrap_or_default();
    if exif.orientation.is_none() && exif_raw.is_some() {
        exif.orientation = Some(1);
    }
    Ok(Photo { width, height, encoded: rgb.into_raw(), bits, transfer, to_rec709, exif })
}

// ------------------------------------------------------------------- EXIF

struct Tiff<'a> {
    b: &'a [u8],
    le: bool,
}

impl<'a> Tiff<'a> {
    fn u16(&self, at: usize) -> Option<u16> {
        let s = self.b.get(at..at + 2)?;
        Some(if self.le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    }
    fn u32(&self, at: usize) -> Option<u32> {
        let s = self.b.get(at..at + 4)?;
        let a = [s[0], s[1], s[2], s[3]];
        Some(if self.le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    }

    /// `(tag, type, count, value-or-offset position)` of every entry of the
    /// IFD at `at`.
    fn entries(&self, at: usize) -> Vec<(u16, u16, u32, usize)> {
        let Some(n) = self.u16(at) else { return Vec::new() };
        (0..n as usize)
            .filter_map(|i| {
                let e = at + 2 + 12 * i;
                let (tag, ty, count) = (self.u16(e)?, self.u16(e + 2)?, self.u32(e + 4)?);
                let size = match ty {
                    1 | 2 | 6 | 7 => 1,
                    3 | 8 => 2,
                    4 | 9 | 11 => 4,
                    5 | 10 | 12 => 8,
                    _ => return None,
                } * count as usize;
                let pos = if size <= 4 { e + 8 } else { self.u32(e + 8)? as usize };
                Some((tag, ty, count, pos))
            })
            .collect()
    }

    fn ascii(&self, pos: usize, count: u32) -> Option<String> {
        let s = self.b.get(pos..pos + count as usize)?;
        let s = String::from_utf8_lossy(s).trim_end_matches('\0').trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    fn number(&self, ty: u16, pos: usize, i: usize) -> Option<f64> {
        match ty {
            3 => self.u16(pos + 2 * i).map(f64::from),
            4 => self.u32(pos + 4 * i).map(f64::from),
            9 => self.u32(pos + 4 * i).map(|v| v as i32 as f64),
            5 => {
                let (n, d) = (self.u32(pos + 8 * i)?, self.u32(pos + 8 * i + 4)?);
                (d != 0).then(|| n as f64 / d as f64)
            }
            10 => {
                let (n, d) = (self.u32(pos + 8 * i)? as i32, self.u32(pos + 8 * i + 4)? as i32);
                (d != 0).then(|| n as f64 / d as f64)
            }
            1 | 7 => self.b.get(pos + i).map(|v| *v as f64),
            _ => None,
        }
    }
}

/// Parse a raw EXIF chunk (a TIFF structure: `II*\0` or `MM\0*`, then
/// IFD0 with its EXIF and GPS sub-IFDs). Anything unreadable is simply
/// absent.
pub fn parse_exif(chunk: &[u8]) -> Exif {
    let chunk = chunk.strip_prefix(b"Exif\0\0").unwrap_or(chunk);
    let le = match chunk.get(0..4) {
        Some([0x49, 0x49, 42, 0]) => true,
        Some([0x4d, 0x4d, 0, 42]) => false,
        _ => return Exif::default(),
    };
    let t = Tiff { b: chunk, le };
    let mut e = Exif::default();
    let Some(ifd0) = t.u32(4) else { return e };
    let mut sub = Vec::new();
    let mut gps_at = None;
    let mut subsec = None;
    for (tag, ty, count, pos) in t.entries(ifd0 as usize) {
        match tag {
            0x010F => e.make = t.ascii(pos, count),
            0x0110 => e.model = t.ascii(pos, count),
            0x0112 => e.orientation = t.number(ty, pos, 0).map(|v| v as u16),
            0x0132 if e.datetime.is_none() => e.datetime = t.ascii(pos, count),
            0x8769 => sub.extend(t.number(ty, pos, 0).map(|v| v as usize)),
            0x8825 => gps_at = t.number(ty, pos, 0).map(|v| v as usize),
            _ => {}
        }
    }
    for at in sub {
        for (tag, ty, count, pos) in t.entries(at) {
            match tag {
                0x829A => e.exposure_s = t.number(ty, pos, 0),
                0x829D => e.f_number = t.number(ty, pos, 0),
                0x8827 => e.iso = t.number(ty, pos, 0),
                0x9003 => e.datetime = t.ascii(pos, count),
                0x9291 => subsec = t.ascii(pos, count),
                0x920A => e.focal_mm = t.number(ty, pos, 0),
                0xA405 => e.focal_35mm = t.number(ty, pos, 0).filter(|v| *v > 0.0),
                0xA434 => e.lens = t.ascii(pos, count),
                _ => {}
            }
        }
    }
    if let (Some(d), Some(s)) = (&mut e.datetime, subsec) {
        d.push('.');
        d.push_str(&s);
    }
    if let Some(at) = gps_at {
        let (mut lat, mut lon, mut alt) = (None, None, None);
        let (mut lat_ref, mut lon_ref, mut alt_below) = ('N', 'E', false);
        let dms = |ty: u16, pos: usize| -> Option<f64> { Some(t.number(ty, pos, 0)? + t.number(ty, pos, 1)? / 60.0 + t.number(ty, pos, 2)? / 3600.0) };
        for (tag, ty, count, pos) in t.entries(at) {
            match tag {
                1 => lat_ref = t.ascii(pos, count).and_then(|s| s.chars().next()).unwrap_or('N'),
                2 => lat = dms(ty, pos),
                3 => lon_ref = t.ascii(pos, count).and_then(|s| s.chars().next()).unwrap_or('E'),
                4 => lon = dms(ty, pos),
                5 => alt_below = t.number(ty, pos, 0) == Some(1.0),
                6 => alt = t.number(ty, pos, 0),
                _ => {}
            }
        }
        if let (Some(la), Some(lo)) = (lat, lon) {
            e.gps = Some(Gps {
                latitude_deg: if lat_ref == 'S' { -la } else { la },
                longitude_deg: if lon_ref == 'W' { -lo } else { lo },
                altitude_m: alt.map(|a| if alt_below { -a } else { a }),
            });
        }
    }
    e
}

// -------------------------------------------------------------------- ICC

/// Linear sRGB/Rec.709 (D65) from CIE XYZ (D65), IEC 61966-2-1.
const XYZ_TO_REC709: [f64; 9] = [3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252];

/// Bradford chromatic adaptation D50 -> D65, the matrix ICC.1 prescribes for
/// leaving its D50 connection space.
const BRADFORD_D50_TO_D65: [f64; 9] = [0.9555766, -0.0230393, 0.0631636, -0.0282895, 1.0099416, 0.0210077, 0.0122982, -0.0204830, 1.3299098];

fn mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    std::array::from_fn(|i| (0..3).map(|k| a[(i / 3) * 3 + k] * b[k * 3 + i % 3]).sum())
}

/// The tone curve and the Rec.709 conversion of a matrix/TRC ICC profile.
/// A profile that is NOT that form (a LUT-based output profile) is an error,
/// and the caller treats the photograph as plain sRGB.
pub fn icc_colour(p: &[u8]) -> Result<(Transfer, Option<[f32; 9]>), String> {
    let be32 = |at: usize| -> Option<u32> { p.get(at..at + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]])) };
    let s15 = |at: usize| -> Option<f64> { be32(at).map(|v| v as i32 as f64 / 65536.0) };
    let count = be32(128).ok_or("no tag table")? as usize;
    let mut tags = std::collections::HashMap::new();
    for i in 0..count.min(256) {
        let at = 132 + 12 * i;
        let (sig, off, len) = (p.get(at..at + 4).ok_or("truncated tag table")?, be32(at + 4).ok_or("tag")?, be32(at + 8).ok_or("tag")?);
        tags.insert(sig.to_vec(), (off as usize, len as usize));
    }
    let xyz = |sig: &[u8]| -> Result<[f64; 3], String> {
        let (off, _) = *tags.get(sig).ok_or("not a matrix/TRC profile")?;
        Ok([s15(off + 8).ok_or("XYZ")?, s15(off + 12).ok_or("XYZ")?, s15(off + 16).ok_or("XYZ")?])
    };
    let (r, g, b) = (xyz(b"rXYZ")?, xyz(b"gXYZ")?, xyz(b"bXYZ")?);
    // columns are the primaries in the D50 connection space
    let to_xyz50 = [r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2]];
    let m = mat3(&XYZ_TO_REC709, &mat3(&BRADFORD_D50_TO_D65, &to_xyz50));
    let identity = (0..9).all(|i| (m[i] - if i % 4 == 0 { 1.0 } else { 0.0 }).abs() < 2e-3);
    let curve = |sig: &[u8]| -> Result<Vec<f32>, String> {
        let (off, _) = *tags.get(sig).ok_or("no tone curve")?;
        let ty = p.get(off..off + 4).ok_or("curve")?;
        let n = 1024usize;
        let eval: Box<dyn Fn(f64) -> f64> = match ty {
            b"curv" => {
                let cnt = be32(off + 8).ok_or("curv")? as usize;
                match cnt {
                    0 => Box::new(|x| x),
                    1 => {
                        let g = p.get(off + 12..off + 14).map(|s| u16::from_be_bytes([s[0], s[1]]) as f64 / 256.0).ok_or("curv")?;
                        Box::new(move |x| x.powf(g))
                    }
                    _ => {
                        let pts: Vec<f64> = (0..cnt)
                            .map(|i| p.get(off + 12 + 2 * i..off + 14 + 2 * i).map(|s| u16::from_be_bytes([s[0], s[1]]) as f64 / 65535.0))
                            .collect::<Option<_>>()
                            .ok_or("curv table")?;
                        Box::new(move |x| {
                            let f = x.clamp(0.0, 1.0) * (pts.len() - 1) as f64;
                            let i = (f as usize).min(pts.len() - 2);
                            pts[i] + (pts[i + 1] - pts[i]) * (f - i as f64)
                        })
                    }
                }
            }
            b"para" => {
                let func = p.get(off + 8..off + 10).map(|s| u16::from_be_bytes([s[0], s[1]])).ok_or("para")?;
                let k = |i: usize| s15(off + 12 + 4 * i).unwrap_or(0.0);
                let (g, a, bb, c, d, e, f) = (k(0), k(1), k(2), k(3), k(4), k(5), k(6));
                match func {
                    0 => Box::new(move |x| x.powf(g)),
                    1 => Box::new(move |x| if x >= -bb / a { (a * x + bb).powf(g) } else { 0.0 }),
                    2 => Box::new(move |x| if x >= -bb / a { (a * x + bb).powf(g) + c } else { c }),
                    3 => Box::new(move |x| if x >= d { (a * x + bb).powf(g) } else { c * x }),
                    4 => Box::new(move |x| if x >= d { (a * x + bb).powf(g) + e } else { c * x + f }),
                    other => return Err(format!("parametric curve type {other}")),
                }
            }
            _ => return Err("unknown tone curve type".into()),
        };
        Ok((0..n).map(|i| eval(i as f64 / (n - 1) as f64) as f32).collect())
    };
    let tables = [curve(b"rTRC")?, curve(b"gTRC")?, curve(b"bTRC")?];
    // A profile whose curves ARE the sRGB curve is kept symbolic: exact, and
    // the common case (sRGB and Display P3 both use it).
    let is_srgb = tables.iter().all(|t| t.iter().enumerate().all(|(i, v)| (v - srgb_to_linear(i as f32 / (t.len() - 1) as f32)).abs() < 2e-3));
    let is_linear = tables.iter().all(|t| t.iter().enumerate().all(|(i, v)| (v - i as f32 / (t.len() - 1) as f32).abs() < 1e-4));
    let transfer = if is_srgb {
        Transfer::Srgb
    } else if is_linear {
        Transfer::Linear
    } else {
        Transfer::Table(tables)
    };
    Ok((transfer, (!identity).then(|| m.map(|v| v as f32))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A little-endian EXIF chunk with an EXIF sub-IFD and a GPS IFD, built
    /// the way cameras lay them out.
    fn chunk() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"II*\0");
        b.extend_from_slice(&8u32.to_le_bytes());
        let entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        // IFD0 at 8: Make (ascii, offset), Orientation, ExifIFD, GPSIFD
        let ifd0 = 8usize;
        let n0 = 4u16;
        let data0 = ifd0 + 2 + 12 * n0 as usize + 4;
        b.extend_from_slice(&n0.to_le_bytes());
        entry(&mut b, 0x010F, 2, 7, data0 as u32);
        entry(&mut b, 0x0112, 3, 1, 6);
        let exif_at = data0 + 8;
        entry(&mut b, 0x8769, 4, 1, exif_at as u32);
        let gps_at = exif_at + 2 + 12 * 5 + 4 + 8 * 3 + 4;
        entry(&mut b, 0x8825, 4, 1, gps_at as u32);
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(b"Xiaomi\0\0");
        // EXIF IFD: ExposureTime 1/100, FNumber 2.2, ISO 100, FocalLength 1.65, 35mm 15
        let n1 = 5u16;
        let rat = exif_at + 2 + 12 * n1 as usize + 4;
        b.extend_from_slice(&n1.to_le_bytes());
        entry(&mut b, 0x829A, 5, 1, rat as u32);
        entry(&mut b, 0x829D, 5, 1, (rat + 8) as u32);
        entry(&mut b, 0x8827, 3, 1, 100);
        entry(&mut b, 0x920A, 5, 1, (rat + 16) as u32);
        entry(&mut b, 0xA405, 3, 1, 15);
        b.extend_from_slice(&0u32.to_le_bytes());
        for (n, d) in [(1u32, 100u32), (22, 10), (165, 100)] {
            b.extend_from_slice(&n.to_le_bytes());
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&[0u8; 4]);
        assert_eq!(b.len(), gps_at);
        // GPS IFD: 59 deg 20' 0" N, 18 deg 3' 0" E, 12 m
        let n2 = 5u16;
        let gd = gps_at + 2 + 12 * n2 as usize + 4;
        b.extend_from_slice(&n2.to_le_bytes());
        entry(&mut b, 1, 2, 2, u32::from_le_bytes([b'N', 0, 0, 0]));
        entry(&mut b, 2, 5, 3, gd as u32);
        entry(&mut b, 3, 2, 2, u32::from_le_bytes([b'E', 0, 0, 0]));
        entry(&mut b, 4, 5, 3, (gd + 24) as u32);
        entry(&mut b, 6, 5, 1, (gd + 48) as u32);
        b.extend_from_slice(&0u32.to_le_bytes());
        for (n, d) in [(59u32, 1u32), (20, 1), (0, 1), (18, 1), (3, 1), (0, 1), (12, 1)] {
            b.extend_from_slice(&n.to_le_bytes());
            b.extend_from_slice(&d.to_le_bytes());
        }
        b
    }

    #[test]
    fn exif_fields_come_out_of_the_tiff_structure() {
        let e = parse_exif(&chunk());
        assert_eq!(e.make.as_deref(), Some("Xiaomi"));
        assert_eq!(e.orientation, Some(6));
        assert!((e.exposure_s.unwrap() - 0.01).abs() < 1e-12);
        assert!((e.f_number.unwrap() - 2.2).abs() < 1e-12);
        assert_eq!(e.iso, Some(100.0));
        assert!((e.focal_mm.unwrap() - 1.65).abs() < 1e-12);
        assert_eq!(e.focal_35mm, Some(15.0));
        let g = e.gps.expect("gps");
        assert!((g.latitude_deg - (59.0 + 20.0 / 60.0)).abs() < 1e-9);
        assert!((g.longitude_deg - 18.05).abs() < 1e-9);
        assert_eq!(g.altitude_m, Some(12.0));
        // the same chunk with the JPEG APP1 prefix still parses
        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&chunk());
        assert_eq!(parse_exif(&app1), e);
    }

    /// 15 mm equivalent on a 4:3 frame: the diagonal is what "equivalent"
    /// is defined on.
    #[test]
    fn a_35mm_equivalent_focal_length_becomes_pixels_on_the_diagonal() {
        let e = Exif { focal_35mm: Some(15.0), ..Default::default() };
        let f = e.focal_px(3264, 2448).unwrap();
        assert!((f - 15.0 * 4080.0 / 43.266615).abs() < 1e-3, "{f}");
        // one stop brighter = double the light: t * ISO / N^2
        let a = Exif { exposure_s: Some(0.01), f_number: Some(2.0), iso: Some(100.0), ..Default::default() };
        let b = Exif { exposure_s: Some(0.02), ..a.clone() };
        assert!((b.exposure_log2().unwrap() - a.exposure_log2().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_srgb_transfer_round_trips() {
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }
}
