// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! AV1 in and out: [`Encoder`] over rav1e, [`Decoder`] over rav1d.
//!
//! Both work in temporal units, the unit a container stores as one sample:
//! the encoder returns them with temporal-delimiter OBUs removed (the AV1
//! ISOBMFF and Matroska mappings both ask for that), and the decoder takes
//! them as a container hands them over.

use std::ptr::NonNull;

use rav1e::prelude::{ChromaSampling, ColorDescription, ColorPrimaries, Config, Context, EncoderConfig, EncoderStatus, FrameType, MatrixCoefficients, PixelRange, Rational, TransferCharacteristics};

use crate::color::{self, Colour, Matrix, Planes, Yuv420};
use crate::Error;

/// How an [`Encoder`] trades size, quality and time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodeSettings {
    /// rav1e's base quantizer, 0 (lossless) ..= 255; lower is larger and closer.
    pub quantizer: u8,
    /// rav1e's speed preset, 0 (slowest, smallest) ..= 10 (fastest).
    pub speed: u8,
    /// Frames between forced key frames, where a reader can start decoding;
    /// `None` forces none after the first frame. rav1e still starts one at
    /// every scene cut it detects either way.
    ///
    /// `None` by default because a forced key frame in the middle of a shot
    /// is visible: rav1e's fixed quantizer codes a key frame about a quarter
    /// finer than the frames between, which settle a few dB lower within a
    /// few dozen frames, so every forced key frame reads as the picture
    /// sharpening all at once (a jump of several dB at each forced key frame of a
    /// real 313-frame clip at the old interval of 120, against a fraction of a dB between any two frames
    /// without them, in a few percent fewer bytes).
    pub key_frame_interval: Option<u64>,
    /// Tiles per frame (0: one tile per 64K pixels, at least one): the unit
    /// both coders spread over threads, at a small cost in compression.
    pub tiles: usize,
}

impl Default for EncodeSettings {
    /// Measured on a panning 1024x512 photograph (48 frames, release build,
    /// no assembly, a busy host): quantizer 80 holds a high PSNR at a modest
    /// bitrate; speed 10 codes as well as 6-9 there at a multiple of the
    /// rate; eight tiles (one per 64K pixels) halve the time again for a few
    /// percent more bytes.
    fn default() -> EncodeSettings {
        EncodeSettings { quantizer: 80, speed: 10, key_frame_interval: None, tiles: 0 }
    }
}

/// rav1e's longest key-frame interval (its own `MAX_MAX_KEY_FRAME_INTERVAL`,
/// which it does not export): past any clip this crate writes, so only the
/// first frame and the scene cuts rav1e detects are key frames.
const NO_FORCED_KEY_FRAMES: u64 = i32::MAX as u64 / 3;

/// The frame sides rav1e codes.
const MIN_SIDE: usize = 16;
const MAX_SIDE: usize = 65535;

/// One coded temporal unit.
pub struct Packet {
    pub data: Vec<u8>,
    /// The input frame it codes (frames are numbered from 0 in send order).
    pub frame: u64,
    pub key: bool,
}

/// An 8-bit 4:2:0 AV1 encoder at a fixed size and frame rate.
pub struct Encoder {
    ctx: Context<u8>,
    w: usize,
    h: usize,
}

impl Encoder {
    /// `fps` as the exact rational `num / den` frames per second.
    pub fn new(w: usize, h: usize, fps: (u64, u64), settings: EncodeSettings) -> Result<Encoder, Error> {
        if w < MIN_SIDE || h < MIN_SIDE || w > MAX_SIDE || h > MAX_SIDE {
            return Err(Error::Invalid(format!("AV1 frames here are {MIN_SIDE} to {MAX_SIDE} pixels a side, not {w}x{h}")));
        }
        let enc = EncoderConfig {
            width: w,
            height: h,
            time_base: Rational::new(fps.1, fps.0),
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            pixel_range: PixelRange::Limited,
            color_description: Some(ColorDescription {
                color_primaries: ColorPrimaries::BT709,
                transfer_characteristics: TransferCharacteristics::BT709,
                matrix_coefficients: MatrixCoefficients::BT709,
            }),
            quantizer: settings.quantizer as usize,
            min_key_frame_interval: 1,
            max_key_frame_interval: settings.key_frame_interval.map_or(NO_FORCED_KEY_FRAMES, |n| n.max(1)),
            tiles: if settings.tiles == 0 { (w * h).div_ceil(1 << 16) } else { settings.tiles },
            ..EncoderConfig::with_speed_preset(settings.speed.min(10))
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let ctx = Config::new().with_encoder_config(enc).with_threads(threads).new_context().map_err(|e| Error::Invalid(format!("AV1 encoder: {e}")))?;
        Ok(Encoder { ctx, w, h })
    }

    /// The `av1C` record (AV1 codec configuration with the sequence header)
    /// both containers carry: MP4's `av1C` box payload, Matroska's CodecPrivate.
    pub fn codec_config(&self) -> Vec<u8> {
        self.ctx.container_sequence_header()
    }

    /// Queue one RGB frame (`w * h * 3` bytes) and return the units now coded.
    pub fn push(&mut self, rgb: &[u8]) -> Result<Vec<Packet>, Error> {
        let yuv = color::rgb_to_yuv420(rgb, self.w, self.h);
        let mut frame = self.ctx.new_frame();
        let (cw, _) = Yuv420::chroma_size(self.w, self.h);
        frame.planes[0].copy_from_raw_u8(&yuv.y, self.w, 1);
        frame.planes[1].copy_from_raw_u8(&yuv.u, cw, 1);
        frame.planes[2].copy_from_raw_u8(&yuv.v, cw, 1);
        self.ctx.send_frame(frame).map_err(|e| Error::Invalid(format!("AV1 encoder: {e:?}")))?;
        self.drain(false)
    }

    /// Code every queued frame and return the last units.
    pub fn finish(mut self) -> Result<Vec<Packet>, Error> {
        self.ctx.flush();
        self.drain(true)
    }

    fn drain(&mut self, flushing: bool) -> Result<Vec<Packet>, Error> {
        let mut out = Vec::new();
        loop {
            match self.ctx.receive_packet() {
                Ok(p) => out.push(Packet { data: strip_temporal_delimiters(&p.data), frame: p.input_frameno, key: p.frame_type == FrameType::KEY }),
                Err(EncoderStatus::Encoded) => continue,
                Err(EncoderStatus::NeedMoreData) if !flushing => return Ok(out),
                Err(EncoderStatus::LimitReached) => return Ok(out),
                Err(e) => return Err(Error::Invalid(format!("AV1 encoder: {e:?}"))),
            }
        }
    }
}

/// `tu` without its temporal-delimiter OBUs (type 2). Every OBU rav1e writes
/// carries its size field, which is what lets the walk find the next one.
fn strip_temporal_delimiters(tu: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tu.len());
    let mut at = 0;
    while at < tu.len() {
        let header = tu[at];
        let obu_type = (header >> 3) & 0xf;
        let has_ext = header & 0x4 != 0;
        let has_size = header & 0x2 != 0;
        let mut pos = at + 1 + has_ext as usize;
        if !has_size {
            // An OBU without a size runs to the end of the unit.
            if obu_type != 2 {
                out.extend_from_slice(&tu[at..]);
            }
            break;
        }
        let Some((size, len)) = leb128(&tu[pos..]) else { break };
        pos += len;
        let end = (pos + size as usize).min(tu.len());
        if obu_type != 2 {
            out.extend_from_slice(&tu[at..end]);
        }
        at = end;
    }
    out
}

/// An unsigned LEB128 value and the bytes it took (at most 8).
pub(crate) fn leb128(b: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for (i, &byte) in b.iter().take(8).enumerate() {
        v |= ((byte & 0x7f) as u64) << (7 * i);
        if byte & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}

/// One decoded picture as 8-bit RGB.
pub struct Picture {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

/// An AV1 decoder (rav1d), fed temporal units in decode order.
pub struct Decoder {
    ctx: Option<rav1d::include::dav1d::dav1d::Dav1dContext>,
}

// SAFETY: a rav1d context is internally synchronized (its state sits behind a mutex) and owns no thread-affine
// resource; the decoder is used by one thread at a time through `&mut self`.
unsafe impl Send for Decoder {}

fn check(r: rav1d::Dav1dResult) -> Result<(), i32> {
    if r.0 < 0 {
        Err(-r.0)
    } else {
        Ok(())
    }
}

const EAGAIN: i32 = 11;

impl Decoder {
    pub fn new() -> Result<Decoder, Error> {
        use rav1d::include::dav1d::dav1d::Dav1dSettings;
        use rav1d::src::lib::{dav1d_default_settings, dav1d_open};
        let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
        let mut ctx: Option<rav1d::include::dav1d::dav1d::Dav1dContext> = None;
        // SAFETY: `dav1d_default_settings` initializes every field of the settings it is given; `dav1d_open` reads
        // them and writes the context handle, which `Drop` closes.
        unsafe {
            dav1d_default_settings(NonNull::new_unchecked(settings.as_mut_ptr()));
            let mut settings = settings.assume_init();
            settings.n_threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16) as i32;
            // A sequence header may declare a 65536 x 65536 frame from a few
            // bytes; the decoder refuses one over the bound before allocating it.
            settings.frame_size_limit = crate::MAX_FRAME_PIXELS as std::ffi::c_uint;
            check(dav1d_open(Some(NonNull::from(&mut ctx)), Some(NonNull::from(&mut settings)))).map_err(|e| Error::Malformed(format!("AV1 decoder: open failed ({e})")))?;
        }
        Ok(Decoder { ctx })
    }

    /// Decode one temporal unit; returns the pictures it completes (zero or more).
    pub fn push(&mut self, tu: &[u8]) -> Result<Vec<Picture>, Error> {
        use rav1d::include::dav1d::data::Dav1dData;
        use rav1d::src::lib::{dav1d_data_create, dav1d_data_unref, dav1d_send_data};
        let mut out = Vec::new();
        if tu.is_empty() {
            return Ok(out);
        }
        let mut data = Dav1dData::default();
        // SAFETY: `dav1d_data_create` allocates `tu.len()` bytes owned by `data` and returns them; they are
        // filled before the data is sent, and `data` is released by `dav1d_data_unref` on every path out.
        unsafe {
            let buf = dav1d_data_create(Some(NonNull::from(&mut data)), tu.len());
            if buf.is_null() {
                return Err(Error::Malformed("AV1 decoder: allocation failed".into()));
            }
            std::ptr::copy_nonoverlapping(tu.as_ptr(), buf, tu.len());
            // The decoder takes the unit whole, or answers EAGAIN while it holds pictures that must be handed out
            // first; then it takes the rest.
            while data.sz > 0 {
                match check(dav1d_send_data(self.ctx, Some(NonNull::from(&mut data)))) {
                    Ok(()) | Err(EAGAIN) => {}
                    Err(e) => {
                        dav1d_data_unref(Some(NonNull::from(&mut data)));
                        return Err(Error::Malformed(format!("AV1 decoder: the stream is invalid (error {e})")));
                    }
                }
                if let Err(e) = self.drain_into(&mut out) {
                    dav1d_data_unref(Some(NonNull::from(&mut data)));
                    return Err(e);
                }
            }
            dav1d_data_unref(Some(NonNull::from(&mut data)));
        }
        self.drain_into(&mut out)?;
        Ok(out)
    }

    /// The pictures still held after the last unit.
    pub fn finish(&mut self) -> Result<Vec<Picture>, Error> {
        let mut out = Vec::new();
        while self.drain_into(&mut out)? {}
        Ok(out)
    }

    /// Move every ready picture into `out`; whether any was ready.
    fn drain_into(&mut self, out: &mut Vec<Picture>) -> Result<bool, Error> {
        use rav1d::include::dav1d::picture::Dav1dPicture;
        use rav1d::src::lib::{dav1d_get_picture, dav1d_picture_unref};
        let mut any = false;
        loop {
            let mut pic = std::mem::MaybeUninit::<Dav1dPicture>::zeroed();
            // SAFETY: `dav1d_get_picture` writes the whole picture on every return; a picture it produced is read
            // through its own plane pointers and strides while it is referenced, then released.
            unsafe {
                match check(dav1d_get_picture(self.ctx, Some(NonNull::new_unchecked(pic.as_mut_ptr())))) {
                    Ok(()) => {
                        let mut pic = pic.assume_init();
                        let rgb = to_rgb(&pic);
                        dav1d_picture_unref(Some(NonNull::from(&mut pic)));
                        out.push(rgb?);
                        any = true;
                    }
                    Err(EAGAIN) => return Ok(any),
                    Err(e) => return Err(Error::Malformed(format!("AV1 decoder: decoding failed (error {e})"))),
                }
            }
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the context came from `dav1d_open` and is closed exactly once.
        unsafe { rav1d::src::lib::dav1d_close(Some(NonNull::from(&mut self.ctx))) }
    }
}

/// The picture's planes as RGB, in the colour its sequence header declares.
///
/// # Safety
/// `pic` must be a picture `dav1d_get_picture` returned and not yet released.
unsafe fn to_rgb(pic: &rav1d::include::dav1d::picture::Dav1dPicture) -> Result<Picture, Error> {
    use rav1d::include::dav1d::headers::*;
    let (w, h) = (pic.p.w as usize, pic.p.h as usize);
    if (w as u64).saturating_mul(h as u64) > crate::MAX_FRAME_PIXELS {
        return Err(Error::Malformed(format!("AV1 decoder: a {w} x {h} frame is over {} pixels", crate::MAX_FRAME_PIXELS)));
    }
    let bits = pic.p.bpc as u32;
    let seq = pic.seq_hdr.ok_or_else(|| Error::Malformed("AV1 decoder: a picture without a sequence header".into()))?;
    // SAFETY: the sequence header lives as long as the picture references it.
    let seq = unsafe { seq.as_ref() };
    let matrix = match seq.mtrx {
        DAV1D_MC_BT709 => Matrix::BT709,
        DAV1D_MC_BT470BG | DAV1D_MC_BT601 | DAV1D_MC_SMPTE240 => Matrix::BT601,
        DAV1D_MC_BT2020_NCL | DAV1D_MC_BT2020_CL => Matrix::BT2020,
        _ if h >= 720 => Matrix::BT709,
        _ => Matrix::BT601,
    };
    let colour = Colour { matrix, full_range: seq.color_range != 0 };
    let chroma_shift = match pic.p.layout {
        DAV1D_PIXEL_LAYOUT_I420 => Some((1, 1)),
        DAV1D_PIXEL_LAYOUT_I422 => Some((1, 0)),
        DAV1D_PIXEL_LAYOUT_I444 => Some((0, 0)),
        _ => None,
    };
    let (cw, ch) = match chroma_shift {
        Some((sx, sy)) => ((w + sx as usize) >> sx, (h + sy as usize) >> sy),
        None => (0, 0),
    };
    // Each plane widened to u16 samples with its stride in samples.
    let plane = |i: usize, pw: usize, ph: usize, stride_bytes: isize| -> Result<(Vec<u16>, usize), Error> {
        let ptr = pic.data[i].ok_or_else(|| Error::Malformed("AV1 decoder: a picture plane is missing".into()))?.as_ptr() as *const u8;
        let mut out = Vec::with_capacity(pw * ph);
        for row in 0..ph {
            // SAFETY: rav1d allocates every row of the plane, `stride_bytes` apart, at least `pw` samples wide.
            let line = unsafe { std::slice::from_raw_parts(ptr.offset(row as isize * stride_bytes), if bits > 8 { pw * 2 } else { pw }) };
            if bits > 8 {
                out.extend(line.as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes(*b)));
            } else {
                out.extend(line.iter().map(|&b| b as u16));
            }
        }
        Ok((out, pw))
    };
    let y = plane(0, w, h, pic.stride[0])?;
    let uv = match chroma_shift {
        Some(_) => Some((plane(1, cw, ch, pic.stride[1])?, plane(2, cw, ch, pic.stride[1])?)),
        None => None,
    };
    let planes = Planes {
        w,
        h,
        bits,
        y: (&y.0, y.1),
        uv: uv.as_ref().map(|(u, v)| ((u.0.as_slice(), u.1), (v.0.as_slice(), v.1))),
        chroma_shift: chroma_shift.unwrap_or((0, 0)),
    };
    Ok(Picture { w, h, rgb: color::yuv_to_rgb(&planes, colour) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_delimiters_are_removed_and_other_obus_kept() {
        // TD (type 2, size 0), then a padding OBU (type 15) of 2 bytes.
        let tu = [0x12, 0x00, 0x7a, 0x02, 0xaa, 0xbb];
        assert_eq!(strip_temporal_delimiters(&tu), [0x7a, 0x02, 0xaa, 0xbb]);
    }
}
