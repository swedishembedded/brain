// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a video action promises about the clips it makes, declared by the model
//! that owns it, so a generic video request (`duration`, `resolution`,
//! `aspect_ratio`, `size`, `generate_audio`, `seed`) can be mapped onto the
//! action's own params without guessing from their names.
//!
//! Video models share no vocabulary: one counts `frames` on a `1 + 8k` grid at
//! a caller-chosen rate, another counts `num_frames` at a fixed frame rate, a third
//! uses a `1 + 4k` grid; sizes snap to different multiples; sound is optional,
//! always on, or absent. A [`VideoContract`] states those facts once, and
//! [`VideoContract::map`] and [`VideoContract::listing`] derive everything a
//! caller sees from them. A request the contract cannot honour is refused with
//! the field named, never silently adjusted beyond the documented snapping.
//!
//! Swedish Embedded AB implements solutions for serving generative video behind
//! one stable API for its clients. If your team needs expertise in video model
//! serving then you can procure our services by sending an email to
//! info@swedishembedded.com.

use serde_json::{json, Map, Value};

use crate::Media;

/// The longest clip an edit takes as its input, in frames.
pub const MAX_EDIT_CLIP_FRAMES: u32 = 2_000;

/// The most pixels all frames of an edit's input clip may hold together, the
/// same budget as all images of one request ([`crate::blob::MAX_REQUEST_PIXELS`]):
/// 121 frames at 768 x 512, or 30 at 1080p. A decoded pixel is twelve bytes
/// of float, and a few kilobytes of video can declare far more.
pub const MAX_EDIT_CLIP_PIXELS: u64 = crate::blob::MAX_REQUEST_PIXELS;

/// Named resolution tiers and the length of the frame's SHORT side they mean.
pub const RESOLUTIONS: [(&str, u32); 4] = [("480p", 480), ("720p", 720), ("1080p", 1080), ("2K", 1440)];

/// Aspect ratios a request may name, as `(name, width, height)`.
pub const ASPECT_RATIOS: [(&str, u32, u32); 9] =
    [("16:9", 16, 9), ("9:16", 9, 16), ("1:1", 1, 1), ("4:3", 4, 3), ("3:4", 3, 4), ("3:2", 3, 2), ("2:3", 2, 3), ("21:9", 21, 9), ("9:21", 9, 21)];

/// Where an action's frame rate comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Fps {
    /// A param the request may set, and the rate used when it does not.
    Param { name: String, default: u32 },
    /// The model always renders at this rate.
    Fixed(u32),
}

/// Whether an action makes sound.
#[derive(Clone, Debug, PartialEq)]
pub enum Audio {
    None,
    /// Every clip has a sound track.
    Always,
    /// A boolean param turns it on, off by `default`.
    Param { name: String, default: bool },
}

/// The frame image types a request may anchor a clip with, in OpenRouter's names.
pub const FRAME_TYPES: [&str; 2] = ["first_frame", "last_frame"];

/// The param a request's reference media fill (see
/// [`VideoContract::references`]): the server writes each reference to a
/// private file of its own and lists the paths, in the request's order,
/// joined by `separator`.
#[derive(Clone, Debug, PartialEq)]
pub struct References {
    pub param: String,
    pub separator: char,
    /// The media kinds the action takes as references.
    pub media: Vec<Media>,
    /// The most references one request may carry.
    pub max: usize,
}

/// The name a model listing gives a reference media kind.
pub fn reference_type(media: Media) -> &'static str {
    match media {
        Media::Image => "image",
        Media::Video => "video",
        Media::Audio => "audio",
        _ => "other",
    }
}

/// The facts a video action declares. See the module doc.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoContract {
    /// The param holding the frame count.
    pub frames_param: String,
    /// Legal frame counts are `base + k * step`, `k >= 0`.
    pub frame_grid: (u32, u32),
    pub fps: Fps,
    /// The width and height params.
    pub size_params: (String, String),
    /// Both sides must be a multiple of this.
    pub size_multiple: u32,
    /// The largest frame, `w * h`, the action renders.
    pub max_pixels: u64,
    /// The shortest and longest clip, in seconds, the action renders.
    pub seconds: (f64, f64),
    pub audio: Audio,
    pub seed_param: Option<String>,
    /// The `video` blob input an edit action takes (`None` for generation).
    pub clip_input: Option<String>,
    /// The host-resolved still params a request's `first_frame` and
    /// `last_frame` images fill. The server writes each image to a private
    /// file and sets the param to its path; a caller never names a path.
    pub first_frame: Option<String>,
    pub last_frame: Option<String>,
    /// The host-resolved param a request's reference media fill.
    pub references: Option<References>,
    /// False for a configuration that is only a smoke test (random weights):
    /// it may be run, but must not be offered as a product.
    pub sellable: bool,
}

/// A generic video request, as the public API carries it. Every field is
/// optional; an absent one leaves the action's own default in place.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoRequest {
    pub duration: Option<f64>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    /// `WIDTHxHEIGHT`.
    pub size: Option<String>,
    pub generate_audio: Option<bool>,
    pub seed: Option<i64>,
}

impl VideoContract {
    /// A generation contract with the given frame param, grid, rate, size params
    /// and bounds; sound, seed and editing are added with the builder methods.
    pub fn new(frames_param: &str, frame_grid: (u32, u32), fps: Fps, size_params: (&str, &str), size_multiple: u32, max_pixels: u64, seconds: (f64, f64)) -> VideoContract {
        VideoContract {
            frames_param: frames_param.into(),
            frame_grid,
            fps,
            size_params: (size_params.0.into(), size_params.1.into()),
            size_multiple,
            max_pixels,
            seconds,
            audio: Audio::None,
            seed_param: None,
            clip_input: None,
            first_frame: None,
            last_frame: None,
            references: None,
            sellable: true,
        }
    }
    pub fn audio(mut self, audio: Audio) -> VideoContract {
        self.audio = audio;
        self
    }
    pub fn seed(mut self, param: &str) -> VideoContract {
        self.seed_param = Some(param.into());
        self
    }
    pub fn clip_input(mut self, blob: &str) -> VideoContract {
        self.clip_input = Some(blob.into());
        self
    }
    /// The host-resolved still params a request's first and last frame images fill.
    pub fn frame_images(mut self, first: Option<&str>, last: Option<&str>) -> VideoContract {
        self.first_frame = first.map(Into::into);
        self.last_frame = last.map(Into::into);
        self
    }
    /// The host-resolved param a request's reference media fill; see [`References`].
    pub fn references(mut self, param: &str, separator: char, media: &[Media], max: usize) -> VideoContract {
        self.references = Some(References { param: param.into(), separator, media: media.to_vec(), max });
        self
    }
    /// The param a frame image of `frame_type` fills, if the action takes one.
    pub fn frame_param(&self, frame_type: &str) -> Option<&str> {
        match frame_type {
            "first_frame" => self.first_frame.as_deref(),
            "last_frame" => self.last_frame.as_deref(),
            _ => None,
        }
    }
    /// The frame image types the action takes, in [`FRAME_TYPES`] order.
    pub fn frame_types(&self) -> Vec<&'static str> {
        FRAME_TYPES.into_iter().filter(|t| self.frame_param(t).is_some()).collect()
    }
    /// The reference media kinds the action takes, by their listing names.
    pub fn reference_types(&self) -> Vec<&'static str> {
        self.references.as_ref().map(|r| r.media.iter().map(|m| reference_type(*m)).collect()).unwrap_or_default()
    }
    pub fn sellable(mut self, sellable: bool) -> VideoContract {
        self.sellable = sellable;
        self
    }

    /// The rate a request renders at, given the params it already maps to.
    pub fn fps_of(&self, params: &Map<String, Value>) -> u32 {
        match &self.fps {
            Fps::Fixed(n) => *n,
            Fps::Param { name, default } => params.get(name).and_then(Value::as_u64).map_or(*default, |v| v as u32),
        }
    }

    /// The smallest legal frame count holding at least `seconds` at `fps`.
    pub fn frames_for(&self, seconds: f64, fps: u32) -> u32 {
        let (base, step) = self.frame_grid;
        let wanted = (seconds * f64::from(fps)).ceil().max(f64::from(base)) as u32;
        let step = step.max(1);
        base + (wanted - base).div_ceil(step) * step
    }

    /// [`Self::frames_for`], stepped back one grid step when snapping up
    /// would carry the clip past the longest the action renders; `Err` when
    /// no legal count lands inside the bounds.
    fn frames_in_bounds(&self, seconds: f64, fps: u32) -> Result<u32, String> {
        let (lo, hi) = self.seconds;
        let length = |frames: u32| f64::from(frames) / f64::from(fps.max(1));
        let up = self.frames_for(seconds, fps);
        let frames = if length(up) > hi + 1e-9 { up.checked_sub(self.frame_grid.1).filter(|f| *f >= self.frame_grid.0).unwrap_or(up) } else { up };
        if length(frames) < lo - 1e-9 || length(frames) > hi + 1e-9 {
            return Err(format!("duration {seconds} s has no frame count this model renders between {lo} and {hi} s at {fps} fps"));
        }
        Ok(frames)
    }

    /// The `(width, height)` a resolution tier and aspect ratio mean for this
    /// action: the short side is the tier's, the long side follows the ratio,
    /// and both are rounded to the nearest multiple the action accepts.
    pub fn size_for(&self, resolution: &str, aspect_ratio: &str) -> Result<(u32, u32), String> {
        let short = RESOLUTIONS.iter().find(|(name, _)| *name == resolution).map(|(_, s)| *s).ok_or_else(|| format!("resolution '{resolution}' is not one of {}", names(RESOLUTIONS.iter().map(|r| r.0))))?;
        let (aw, ah) = ASPECT_RATIOS
            .iter()
            .find(|(name, ..)| *name == aspect_ratio)
            .map(|(_, w, h)| (*w, *h))
            .ok_or_else(|| format!("aspect_ratio '{aspect_ratio}' is not one of {}", names(ASPECT_RATIOS.iter().map(|r| r.0))))?;
        let long = f64::from(short) * f64::from(aw.max(ah)) / f64::from(aw.min(ah));
        let round = |v: f64| -> u32 {
            let m = f64::from(self.size_multiple.max(1));
            ((v / m).round().max(1.0) * m) as u32
        };
        let (short, long) = (round(f64::from(short)), round(long));
        Ok(if aw >= ah { (long, short) } else { (short, long) })
    }

    fn check_size(&self, w: u32, h: u32) -> Result<(), String> {
        let m = self.size_multiple.max(1);
        if w == 0 || h == 0 || !w.is_multiple_of(m) || !h.is_multiple_of(m) {
            return Err(format!("size {w}x{h}: both sides must be positive multiples of {m}"));
        }
        if u64::from(w) * u64::from(h) > self.max_pixels {
            return Err(format!("size {w}x{h} is larger than this model renders ({} pixels at most)", self.max_pixels));
        }
        Ok(())
    }

    /// Map `req` onto the action's params. Only fields the request names are
    /// set; the rest keep the action's defaults. An edit action
    /// ([`Self::clip_input`]) takes its length and size from the clip, so a
    /// request naming either is refused.
    pub fn map(&self, req: &VideoRequest) -> Result<Map<String, Value>, String> {
        let mut params = Map::new();
        let editing = self.clip_input.is_some();
        if editing && (req.duration.is_some() || req.size.is_some() || req.resolution.is_some() || req.aspect_ratio.is_some()) {
            return Err("an edit takes its length and size from the input video; set them through the action's own params".into());
        }

        let size = match (&req.size, &req.resolution, &req.aspect_ratio) {
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => return Err("give size, or resolution and aspect_ratio, not both".into()),
            (Some(size), None, None) => Some(parse_size(size)?),
            (None, Some(res), ar) => Some(self.size_for(res, ar.as_deref().unwrap_or("16:9"))?),
            (None, None, Some(ar)) => Some(self.size_for(RESOLUTIONS[1].0, ar)?),
            (None, None, None) => None,
        };
        if let Some((w, h)) = size {
            self.check_size(w, h)?;
            params.insert(self.size_params.0.clone(), json!(w));
            params.insert(self.size_params.1.clone(), json!(h));
        }

        if let Some(seconds) = req.duration {
            let (lo, hi) = self.seconds;
            if !seconds.is_finite() || seconds < lo || seconds > hi {
                return Err(format!("duration {seconds} s is outside what this model renders ({lo} to {hi} s)"));
            }
            let fps = self.fps_of(&params);
            params.insert(self.frames_param.clone(), json!(self.frames_in_bounds(seconds, fps)?));
        }

        if let Some(on) = req.generate_audio {
            match (&self.audio, on) {
                (Audio::Param { name, .. }, on) => {
                    params.insert(name.clone(), json!(on));
                }
                (Audio::Always, true) | (Audio::None, false) => {}
                (Audio::Always, false) => return Err("this model always generates sound; generate_audio cannot be false".into()),
                (Audio::None, true) => return Err("this model does not generate sound".into()),
            }
        }

        if let Some(seed) = req.seed {
            let name = self.seed_param.as_ref().ok_or("this model does not take a seed")?;
            params.insert(name.clone(), json!(seed));
        }
        Ok(params)
    }

    /// Whole-second durations this action renders at its default rate,
    /// ascending: exactly those [`Self::map`] accepts.
    pub fn durations(&self) -> Vec<u32> {
        let (lo, hi) = self.seconds;
        let fps = self.fps_of(&Map::new());
        (lo.ceil().max(1.0) as u32..=hi.floor() as u32).filter(|s| self.frames_in_bounds(f64::from(*s), fps).is_ok()).collect()
    }

    /// The `(resolution, aspect_ratio, width, height)` combinations that fit.
    pub fn sizes(&self) -> Vec<(&'static str, &'static str, u32, u32)> {
        let mut out = Vec::new();
        for (res, _) in RESOLUTIONS {
            for (ar, ..) in ASPECT_RATIOS {
                if let Ok((w, h)) = self.size_for(res, ar) {
                    if self.check_size(w, h).is_ok() {
                        out.push((res, ar, w, h));
                    }
                }
            }
        }
        out
    }

    /// What a model listing says about this action, in the field names of the
    /// OpenRouter video model listing.
    pub fn listing(&self) -> Value {
        let sizes = self.sizes();
        let mut resolutions: Vec<&str> = Vec::new();
        let mut ratios: Vec<&str> = Vec::new();
        for (res, ar, ..) in &sizes {
            if !resolutions.contains(res) {
                resolutions.push(res);
            }
            if !ratios.contains(ar) {
                ratios.push(ar);
            }
        }
        let editing = self.clip_input.is_some();
        json!({
            "supported_durations": if editing { json!(null) } else { json!(self.durations()) },
            "supported_resolutions": if editing { json!(null) } else { json!(resolutions) },
            "supported_aspect_ratios": if editing { json!(null) } else { json!(ratios) },
            "supported_sizes": if editing { json!(null) } else { json!(sizes.iter().map(|(.., w, h)| format!("{w}x{h}")).collect::<Vec<_>>()) },
            "generate_audio": match self.audio { Audio::None => json!(false), Audio::Always => json!(true), Audio::Param { default, .. } => json!(default) },
            "seed": self.seed_param.is_some(),
            "supported_frame_images": self.frame_types(),
            "supported_reference_types": self.reference_types(),
        })
    }

    /// What in this contract `spec` does not have: a param or input it names
    /// that the action does not declare. Empty when the two agree; a
    /// generation names its frame and size params, an edit only its clip.
    pub fn problems(&self, spec: &crate::ActionSpec) -> Vec<String> {
        let mut named: Vec<&str> = Vec::new();
        if self.clip_input.is_none() {
            named.extend([self.frames_param.as_str(), self.size_params.0.as_str(), self.size_params.1.as_str()]);
            if let Fps::Param { name, .. } = &self.fps {
                named.push(name);
            }
        }
        if let Audio::Param { name, .. } = &self.audio {
            named.push(name);
        }
        named.extend(self.seed_param.as_deref());
        let mut out: Vec<String> = named.into_iter().filter(|n| !spec.params.iter().any(|p| p.name == *n)).map(|n| format!("{}: no param '{n}'", spec.name)).collect();
        // A param the server fills with a path it wrote must be a string the
        // host resolves: a remote caller can never set it directly.
        let host_paths = self.first_frame.iter().chain(&self.last_frame).chain(self.references.as_ref().map(|r| &r.param));
        for name in host_paths {
            match spec.params.iter().find(|p| &p.name == name) {
                None => out.push(format!("{}: no param '{name}'", spec.name)),
                Some(p) if !p.host_resolved || p.ty != crate::ParamType::Str => out.push(format!("{}: '{name}' is not a host-resolved string param", spec.name)),
                Some(_) => {}
            }
        }
        if let Some(clip) = &self.clip_input {
            if !spec.inputs.iter().any(|b| &b.name == clip && b.media == crate::Media::Video) {
                out.push(format!("{}: no video input '{clip}'", spec.name));
            }
        }
        out
    }

    pub fn to_json(&self) -> Value {
        json!({
            "frames_param": self.frames_param,
            "frame_grid": [self.frame_grid.0, self.frame_grid.1],
            "fps": match &self.fps { Fps::Fixed(n) => json!(n), Fps::Param { name, default } => json!({"param": name, "default": default}) },
            "size_params": [self.size_params.0, self.size_params.1],
            "size_multiple": self.size_multiple,
            "max_pixels": self.max_pixels,
            "seconds": [self.seconds.0, self.seconds.1],
            "audio": match &self.audio { Audio::None => json!(false), Audio::Always => json!(true), Audio::Param { name, default } => json!({"param": name, "default": default}) },
            "seed_param": self.seed_param,
            "clip_input": self.clip_input,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "references": self.references.as_ref().map(|r| json!({"param": r.param, "separator": r.separator.to_string(), "media": r.media.iter().map(|m| reference_type(*m)).collect::<Vec<_>>(), "max": r.max})),
            "sellable": self.sellable,
        })
    }
}

/// `WIDTHxHEIGHT` into its two numbers.
pub fn parse_size(size: &str) -> Result<(u32, u32), String> {
    let bad = || format!("size '{size}' is not WIDTHxHEIGHT");
    let (w, h) = size.split_once(['x', 'X']).ok_or_else(bad)?;
    Ok((w.trim().parse().map_err(|_| bad())?, h.trim().parse().map_err(|_| bad())?))
}

fn names<'a>(it: impl Iterator<Item = &'a str>) -> String {
    it.collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LTX-2.5's shape: `frames` on 1 + 8k at a param rate, multiples of 32.
    fn ltx() -> VideoContract {
        VideoContract::new("frames", (1, 8), Fps::Param { name: "fps".into(), default: 24 }, ("width", "height"), 32, 1920 * 1088, (1.0, 20.0))
            .audio(Audio::Param { name: "audio".into(), default: false })
            .seed("seed")
    }

    // perf-number: MiniMax-H3's released frame rate and duration bounds, a contract constant
    /// MiniMax-H3's shape: `num_frames` at a fixed 24 fps, 5 to 15 s, sound always.
    fn minimax() -> VideoContract {
        VideoContract::new("num_frames", (1, 8), Fps::Fixed(24), ("width", "height"), 32, 1280 * 736, (5.0, 15.0)).audio(Audio::Always).seed("seed")
    }

    fn req() -> VideoRequest {
        VideoRequest::default()
    }

    #[test]
    fn an_empty_request_keeps_every_default() {
        assert!(ltx().map(&req()).unwrap().is_empty());
    }

    #[test]
    fn duration_snaps_up_to_the_frame_grid_at_the_requested_rate() {
        // perf-number: frame-grid arithmetic of the requested rate and duration, not a measurement
        // 5 s at 24 fps is 120 frames; the next 1 + 8k count is 121.
        let p = ltx().map(&VideoRequest { duration: Some(5.0), ..req() }).unwrap();
        assert_eq!(p["frames"], json!(121));
        // A fixed rate uses its own frame param name.
        let p = minimax().map(&VideoRequest { duration: Some(5.0), ..req() }).unwrap();
        assert_eq!(p["num_frames"], json!(121));
        assert!(p.get("frames").is_none());
    }

    #[test]
    fn duration_outside_the_models_bounds_is_refused_naming_them() {
        let e = minimax().map(&VideoRequest { duration: Some(3.0), ..req() }).unwrap_err();
        assert!(e.contains("5") && e.contains("15"), "{e}");
    }

    #[test]
    fn resolution_and_aspect_ratio_become_a_size_on_the_models_grid() {
        let p = ltx().map(&VideoRequest { resolution: Some("720p".into()), aspect_ratio: Some("16:9".into()), ..req() }).unwrap();
        // 1280 is a multiple of 32; 720 is not, and rounds to 736.
        assert_eq!((p["width"].as_u64(), p["height"].as_u64()), (Some(1280), Some(736)));
        let p = ltx().map(&VideoRequest { resolution: Some("480p".into()), aspect_ratio: Some("9:16".into()), ..req() }).unwrap();
        assert_eq!((p["width"].as_u64(), p["height"].as_u64()), (Some(480), Some(864)));
    }

    #[test]
    fn an_explicit_size_off_the_grid_or_too_large_is_refused() {
        assert!(ltx().map(&VideoRequest { size: Some("1000x500".into()), ..req() }).unwrap_err().contains("multiples of 32"));
        assert!(minimax().map(&VideoRequest { size: Some("1920x1088".into()), ..req() }).unwrap_err().contains("larger"));
        assert!(ltx().map(&VideoRequest { size: Some("1280x720".into()), resolution: Some("720p".into()), ..req() }).is_err());
        assert!(ltx().map(&VideoRequest { size: Some("wide".into()), ..req() }).unwrap_err().contains("WIDTHxHEIGHT"));
    }

    #[test]
    fn sound_requests_follow_what_the_model_can_do() {
        assert_eq!(ltx().map(&VideoRequest { generate_audio: Some(true), ..req() }).unwrap()["audio"], json!(true));
        assert!(minimax().map(&VideoRequest { generate_audio: Some(false), ..req() }).is_err());
        assert!(minimax().map(&VideoRequest { generate_audio: Some(true), ..req() }).unwrap().is_empty());
        let silent = VideoContract::new("frames", (1, 4), Fps::Fixed(16), ("width", "height"), 16, 832 * 480, (1.0, 5.0));
        assert!(silent.map(&VideoRequest { generate_audio: Some(true), ..req() }).is_err());
        assert!(silent.map(&VideoRequest { seed: Some(1), ..req() }).is_err());
    }

    #[test]
    fn an_edit_refuses_length_and_size_but_takes_a_seed() {
        let edit = ltx().clip_input("video");
        assert!(edit.map(&VideoRequest { duration: Some(2.0), ..req() }).is_err());
        assert_eq!(edit.map(&VideoRequest { seed: Some(7), ..req() }).unwrap()["seed"], json!(7));
    }

    #[test]
    fn snapping_never_carries_a_clip_past_the_longest_the_model_renders() {
        // A 17n + 5 grid at a fixed rate: the longest duration snaps up to 362 frames, past the bound, so it is stepped back
        // to 345 frames, still inside the bounds.
        let c = VideoContract::new("num_frames", (5, 17), Fps::Fixed(24), ("width", "height"), 32, 1344 * 768, (5.0, 15.0));
        assert_eq!(c.map(&VideoRequest { duration: Some(15.0), ..req() }).unwrap()["num_frames"], json!(345));
        assert_eq!(c.map(&VideoRequest { duration: Some(5.0), ..req() }).unwrap()["num_frames"], json!(124));
        assert_eq!(c.durations(), (5..=15).collect::<Vec<u32>>());
    }

    #[test]
    fn the_listing_offers_only_what_map_accepts() {
        let c = minimax();
        let l = c.listing();
        assert_eq!(l["supported_durations"], json!([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]));
        assert_eq!(l["generate_audio"], json!(true));
        for size in l["supported_sizes"].as_array().unwrap() {
            assert!(c.map(&VideoRequest { size: Some(size.as_str().unwrap().into()), ..req() }).is_ok(), "{size}");
        }
        for d in l["supported_durations"].as_array().unwrap() {
            assert!(c.map(&VideoRequest { duration: Some(d.as_f64().unwrap()), ..req() }).is_ok(), "{d}");
        }
        assert!(!l["supported_resolutions"].as_array().unwrap().contains(&json!("1080p")));
    }

    #[test]
    fn frame_images_and_references_are_listed_and_must_be_host_paths() {
        use crate::{ActionSpec, ParamSpec, ParamType};
        let c = ltx().frame_images(Some("start"), None).references("refs", ',', &[Media::Image, Media::Audio], 4);
        assert_eq!(c.frame_types(), ["first_frame"]);
        assert_eq!(c.frame_param("last_frame"), None);
        assert_eq!(c.listing()["supported_frame_images"], json!(["first_frame"]));
        assert_eq!(c.listing()["supported_reference_types"], json!(["image", "audio"]));
        let base = |start: ParamSpec| {
            ActionSpec::new("t2v", "")
                .param(ParamSpec::new("frames", ParamType::Int, ""))
                .param(ParamSpec::new("width", ParamType::Int, ""))
                .param(ParamSpec::new("height", ParamType::Int, ""))
                .param(ParamSpec::new("fps", ParamType::Int, ""))
                .param(ParamSpec::new("audio", ParamType::Bool, ""))
                .param(ParamSpec::new("seed", ParamType::Int, ""))
                .param(start)
        };
        // A param a remote caller could set itself would let it name a path.
        let open = base(ParamSpec::new("start", ParamType::Str, ""));
        assert_eq!(c.problems(&open), ["t2v: 'start' is not a host-resolved string param", "t2v: no param 'refs'"]);
        let host = base(ParamSpec::new("start", ParamType::Str, "").host_resolved()).param(ParamSpec::new("refs", ParamType::Str, "").host_resolved());
        assert!(c.problems(&host).is_empty(), "{:?}", c.problems(&host));
    }

}
