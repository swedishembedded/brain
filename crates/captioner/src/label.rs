// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reusable workflow: walk a folder of images, caption each one with any
//! [`Captioner`](crate::Captioner), and write the caption file the image
//! trainers already read.
//!
//! Two properties make this worth having as a capability rather than a script:
//!
//! * **Resumable.** The caption file is re-read before the run and rewritten
//!   after every image, so an interrupted run - a crash, a cancelled job, a
//!   machine that got busy - resumes where it stopped instead of paying for the
//!   whole folder again. Re-running a finished folder is a no-op.
//! * **Idempotent, and editable in between.** An image that already has a
//!   caption is left exactly as it is, so a human can correct a caption by hand
//!   and a later run will not overwrite the correction. [`LabelOpts::overwrite`]
//!   is the explicit opt-out.
//!
//! The output goes through `data::imageset`'s writer, so a labeled folder is by
//! construction a folder `flux2::finetune` (and every other captioned-image
//! trainer) can train on, with the captions as editable multi-line blocks.
//!
//! Swedish Embedded AB implements dataset labeling pipelines for its clients.
//! If your team needs expertise in curating training data for image and video
//! models then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::{Path, PathBuf};

use crate::{CaptionRequest, Captioner, Clip, Frame};

/// The image extensions the folder walk considers. Deliberately the set
/// `data::imageset` can actually decode: an extension listed here and not
/// decodable would be captioned and then silently dropped by the trainer, which
/// is the worst of both.
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "ppm", "pnm", "pgm", "pbm"];

/// How to label a folder.
pub struct LabelOpts {
    /// The instruction handed to the model for every image. This is where
    /// caption quality is decided - see [`crate::CaptionRequest::instruction`].
    pub instruction: String,
    /// Where to write the captions. Relative paths resolve inside the dataset
    /// folder, which is where a trainer looks for them.
    pub out: PathBuf,
    /// Token budget per caption.
    pub max_new: u32,
    /// Re-caption images that already have a caption, discarding what is there
    /// (including any hand edits). Off by default: that is what makes a re-run
    /// safe.
    pub overwrite: bool,
}

impl LabelOpts {
    /// Defaults for captioning an image folder for LoRA training.
    pub fn new(instruction: impl Into<String>) -> LabelOpts {
        LabelOpts { instruction: instruction.into(), out: PathBuf::from("captions.yaml"), max_new: 320, overwrite: false }
    }
}

/// What a labeling run did.
#[derive(Default, PartialEq, Eq, Debug)]
pub struct LabelReport {
    /// Images captioned by the model on this run.
    pub captioned: usize,
    /// Images left alone because they already had a caption.
    pub skipped: usize,
    /// Images that could not be decoded or captioned. Each was reported through
    /// `warn` and left uncaptioned, so a re-run retries exactly these.
    pub failed: usize,
}

/// Caption every image in `dir` that does not already have a caption, writing
/// the result through `data::imageset::write_captions_yaml`.
///
/// `progress(done, total, file)` is called before each image is captioned;
/// `warn` receives per-image failures. Neither a failed decode nor a failed
/// caption aborts the run - one bad file in a folder of hundreds must not cost
/// the rest, and the failures are re-tried by simply running again.
pub fn label_dir(
    model: &mut dyn Captioner,
    dir: &Path,
    opts: &LabelOpts,
    mut progress: impl FnMut(usize, usize, &str),
    mut warn: impl FnMut(&str),
) -> Result<LabelReport, String> {
    if !dir.is_dir() {
        return Err(format!("label: {} is not a directory", dir.display()));
    }
    let caps = model.capabilities();
    if caps.max_frames == 0 {
        return Err(format!("label: {} reports max_frames 0", caps.model));
    }
    let out = if opts.out.is_absolute() { opts.out.clone() } else { dir.join(&opts.out) };

    let mut captions = data::imageset::read_captions_yaml(&out, &mut |w| warn(w));
    let files = image_files(dir)?;
    if files.is_empty() {
        return Err(format!("label: no images in {} (looked for {})", dir.display(), IMAGE_EXTENSIONS.join(", ")));
    }

    // "Not done" is no entry OR an empty one. A failed image is written with an
    // empty caption (below) so the file inventories the whole folder; treating
    // that entry as finished would make every failure permanent on the next
    // run, which is the opposite of what listing it is for.
    let todo: Vec<&String> =
        files.iter().filter(|f| opts.overwrite || captions.get(*f).is_none_or(|c| c.trim().is_empty())).collect();
    let mut report = LabelReport { skipped: files.len() - todo.len(), ..LabelReport::default() };
    let total = todo.len();
    for (i, file) in todo.into_iter().enumerate() {
        progress(i, total, file);
        match caption_one(model, &dir.join(file), opts) {
            Ok(text) => {
                captions.insert(file.clone(), text);
                report.captioned += 1;
                // Write after EVERY image, not at the end: a run that dies on
                // image 40 of 50 must not throw away the first 39 captions.
                data::imageset::write_captions_yaml(&out, &captions)?;
            }
            Err(e) => {
                warn(&format!("{file}: {e}"));
                report.failed += 1;
                // List it anyway, with an empty caption: the captions file is
                // then a complete inventory of the folder, so a consumer never
                // has to guess whether a missing name failed or was never seen.
                // The resume rule above treats an empty caption as outstanding,
                // so this does not cost the retry.
                captions.entry(file.clone()).or_default();
                data::imageset::write_captions_yaml(&out, &captions)?;
            }
        }
    }
    // Always leave the file on disk, even for a fully-resumed no-op run, so the
    // caller can rely on it existing afterwards.
    data::imageset::write_captions_yaml(&out, &captions)?;
    Ok(report)
}

/// Decode one image and caption it.
fn caption_one(model: &mut dyn Captioner, path: &Path, opts: &LabelOpts) -> Result<String, String> {
    let clip = Clip::still(decode_frame(path)?);
    let max_new = opts.max_new.min(model.capabilities().max_new_limit).max(1);
    let req = CaptionRequest { clip: &clip, instruction: &opts.instruction, max_new };
    model.validate(&req)?;
    let text = model.caption(&req, &mut |_| {})?;
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("the model returned an empty caption".into());
    }
    Ok(text)
}

/// Decode an image file to the HWC f32 `[0,1]` frame a captioner consumes.
fn decode_frame(path: &Path) -> Result<Frame, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("decode: {e}"))?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Frame::new(img.iter().map(|&b| b as f32 / 255.0).collect(), w, h)
}

/// Every image file directly in `dir`, by file name, sorted. Sorted so a run is
/// deterministic and a resumed run visits the folder in the same order.
fn image_files(dir: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("label: {}: {e}", dir.display()))? {
        let e = e.map_err(|e| format!("label: {}: {e}", dir.display()))?;
        if !e.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let name = e.file_name().to_string_lossy().to_string();
        let ext = Path::new(&name).extension().unwrap_or_default().to_string_lossy().to_lowercase();
        if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
            out.push(name);
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Capabilities;

    /// A captioner that answers with the image's own mean brightness, so a test
    /// can tell which image produced which caption. A stub that ignored its
    /// input would make every resume/overwrite assertion below vacuous.
    struct MeanCaptioner {
        calls: usize,
        tag: &'static str,
    }

    impl Captioner for MeanCaptioner {
        fn capabilities(&self) -> Capabilities {
            Capabilities { model: "test/mean".into(), max_frames: 1, max_new_limit: 64 }
        }
        fn caption(&mut self, req: &CaptionRequest<'_>, on_token: &mut dyn FnMut(&str)) -> Result<String, String> {
            self.calls += 1;
            let f = req.clip.first();
            let mean = f.hwc.iter().sum::<f32>() / f.hwc.len() as f32;
            let text = format!("{} {:.3}\n{} x {}\n{}", self.tag, mean, f.w, f.h, req.instruction);
            on_token(&text);
            Ok(text)
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("captioner_label_{}_{name}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Write a solid-grey PNG whose value identifies it.
    fn png(dir: &Path, name: &str, level: u8, w: u32, h: u32) {
        let mut img = image::RgbImage::new(w, h);
        for p in img.pixels_mut() {
            *p = image::Rgb([level, level, level]);
        }
        img.save(dir.join(name)).unwrap();
    }

    #[test]
    fn captions_every_image_and_writes_a_loadable_caption_file() {
        let d = scratch("basic");
        png(&d, "a.png", 0, 4, 4);
        png(&d, "b.png", 255, 6, 2);
        std::fs::write(d.join("notes.txt"), "not an image").unwrap();

        let mut m = MeanCaptioner { calls: 0, tag: "first" };
        let opts = LabelOpts::new("describe the room in bohemian style");
        let r = label_dir(&mut m, &d, &opts, |_, _, _| {}, |w| panic!("unexpected warning: {w}")).unwrap();
        assert_eq!(r, LabelReport { captioned: 2, skipped: 0, failed: 0 });
        assert_eq!(m.calls, 2);

        // The captions come back through the trainer's own loader, multi-line
        // and intact - the property the whole format change exists for.
        let back = data::imageset::read_captions_yaml(&d.join("captions.yaml"), &mut |_| {});
        assert_eq!(back.len(), 2);
        assert_eq!(back["a.png"], "first 0.000\n4 x 4\ndescribe the room in bohemian style");
        assert_eq!(back["b.png"], "first 1.000\n6 x 2\ndescribe the room in bohemian style");
        std::fs::remove_dir_all(&d).ok();
    }

    /// Resume: a second run captions only what is missing, and does not disturb
    /// - or even re-read through the model - what is already there.
    #[test]
    fn a_second_run_captions_only_the_new_images() {
        let d = scratch("resume");
        png(&d, "a.png", 0, 4, 4);
        let opts = LabelOpts::new("describe");

        let mut first = MeanCaptioner { calls: 0, tag: "first" };
        let r1 = label_dir(&mut first, &d, &opts, |_, _, _| {}, |_| {}).unwrap();
        assert_eq!(r1, LabelReport { captioned: 1, skipped: 0, failed: 0 });

        // A human edits a.png's caption, then a new image arrives.
        let mut edited = data::imageset::read_captions_yaml(&d.join("captions.yaml"), &mut |_| {});
        edited.insert("a.png".into(), "HAND EDITED\nsecond line".into());
        data::imageset::write_captions_yaml(&d.join("captions.yaml"), &edited).unwrap();
        png(&d, "b.png", 255, 4, 4);

        let mut second = MeanCaptioner { calls: 0, tag: "second" };
        let r2 = label_dir(&mut second, &d, &opts, |_, _, _| {}, |_| {}).unwrap();
        assert_eq!(r2, LabelReport { captioned: 1, skipped: 1, failed: 0 });
        assert_eq!(second.calls, 1, "the already-captioned image must not reach the model");

        let back = data::imageset::read_captions_yaml(&d.join("captions.yaml"), &mut |_| {});
        assert_eq!(back["a.png"], "HAND EDITED\nsecond line", "a hand edit must survive a re-run");
        assert!(back["b.png"].starts_with("second "), "the new image got the new run's caption");
        std::fs::remove_dir_all(&d).ok();
    }

    /// `--overwrite` is the explicit opt-out, and it must actually replace the
    /// text rather than merely re-running the model.
    #[test]
    fn overwrite_replaces_existing_captions() {
        let d = scratch("overwrite");
        png(&d, "a.png", 0, 4, 4);
        let mut first = MeanCaptioner { calls: 0, tag: "first" };
        label_dir(&mut first, &d, &LabelOpts::new("describe"), |_, _, _| {}, |_| {}).unwrap();

        let mut second = MeanCaptioner { calls: 0, tag: "second" };
        let opts = LabelOpts { overwrite: true, ..LabelOpts::new("describe") };
        let r = label_dir(&mut second, &d, &opts, |_, _, _| {}, |_| {}).unwrap();
        assert_eq!(r, LabelReport { captioned: 1, skipped: 0, failed: 0 });
        let back = data::imageset::read_captions_yaml(&d.join("captions.yaml"), &mut |_| {});
        assert!(back["a.png"].starts_with("second "), "overwrite must replace, not keep: {}", back["a.png"]);
        std::fs::remove_dir_all(&d).ok();
    }

    /// One unreadable file must not cost the rest of the folder. It IS listed
    /// in the captions file, with an empty caption, so the file is a complete
    /// inventory of the folder rather than a list of the lucky images - a
    /// caller diffing the two no longer has to ask which images are absent
    /// because they failed and which because they were never seen.
    ///
    /// An empty caption is nevertheless NOT a caption: a re-run must retry
    /// exactly those, so the resume rule is "no caption or an empty one",
    /// not "no key". Both halves are asserted here, because writing the key
    /// without relaxing the resume rule would silently make every failure
    /// permanent, which is the more expensive half of the bug.
    #[test]
    fn a_bad_image_is_listed_with_an_empty_caption_and_still_retried() {
        let d = scratch("bad");
        png(&d, "good.png", 128, 4, 4);
        std::fs::write(d.join("broken.png"), b"not a png at all").unwrap();

        let mut m = MeanCaptioner { calls: 0, tag: "t" };
        let mut warnings = Vec::new();
        let r = label_dir(&mut m, &d, &LabelOpts::new("describe"), |_, _, _| {}, |w| warnings.push(w.to_string())).unwrap();
        assert_eq!(r, LabelReport { captioned: 1, skipped: 0, failed: 1 });
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].starts_with("broken.png:"), "{:?}", warnings);

        let back = data::imageset::read_captions_yaml(&d.join("captions.yaml"), &mut |_| {});
        assert!(back.contains_key("good.png"));
        assert_eq!(back.get("broken.png").map(String::as_str), Some(""), "a failed image is listed with an empty caption");

        // Re-run: the empty entry must be retried, not counted as done.
        let mut m2 = MeanCaptioner { calls: 0, tag: "t" };
        let r2 = label_dir(&mut m2, &d, &LabelOpts::new("describe"), |_, _, _| {}, |_| {}).unwrap();
        assert_eq!(r2, LabelReport { captioned: 0, skipped: 1, failed: 1 }, "the empty entry is retried; the captioned one is skipped");
        std::fs::remove_dir_all(&d).ok();
    }

    /// Progress must be reported against the work actually left to do, not the
    /// folder size - on a resumed run those differ, and a caller showing 1/50
    /// when there is one image left is lying to the operator.
    #[test]
    fn progress_totals_count_only_the_outstanding_images() {
        let d = scratch("progress");
        png(&d, "a.png", 0, 4, 4);
        png(&d, "b.png", 255, 4, 4);
        let opts = LabelOpts::new("describe");
        let mut m = MeanCaptioner { calls: 0, tag: "t" };
        let mut seen = Vec::new();
        label_dir(&mut m, &d, &opts, |i, tot, f| seen.push((i, tot, f.to_string())), |_| {}).unwrap();
        assert_eq!(seen, vec![(0, 2, "a.png".into()), (1, 2, "b.png".into())]);

        png(&d, "c.png", 64, 4, 4);
        let mut m2 = MeanCaptioner { calls: 0, tag: "t" };
        let mut seen2 = Vec::new();
        label_dir(&mut m2, &d, &opts, |i, tot, f| seen2.push((i, tot, f.to_string())), |_| {}).unwrap();
        assert_eq!(seen2, vec![(0, 1, "c.png".into())]);
        std::fs::remove_dir_all(&d).ok();
    }
}
