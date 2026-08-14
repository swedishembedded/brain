// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer face restoration behind the generalized [`capability`] interface
//! - what makes `brain caps restore` / `brain do restore restore_face …`, the
//! D-Bus `Run` method and `brain perf`'s `CapabilityTarget` work with no
//! restoration-specific plumbing in the CLI or the transports.
//!
//! One action, `restore_face`: a face image in, the restored face out, with the
//! identity-fidelity dial `w` as a plain float param. `w` is a one-element
//! device buffer read by `scale_add` ([`crate::model`]), so sweeping it across
//! calls is a buffer write, not a graph rebuild - the resident instance is built
//! once and answers every `w`.
//!
//! # Geometry: 512² in, 512² out
//!
//! The graph is built for `cfg.img_size()` (512²) and every buffer in it is
//! sized from that, so the action resizes the input to 512² on the device and
//! returns the restored 512² face. Pasting it back into a full photo is the
//! caller's job (`examples/restore/`), exactly as in the reference CLI where
//! `cropped_faces/` and `restored_faces/` are 512² and the paste-back is a
//! separate step.
//!
//! # Scope this action inherits from the model
//!
//! `crates/restore` is forward-only and takes an **aligned** face:
//! detection + 5-point alignment live in
//! `crates/arcface` and are not chained in here, because CodeFormer's alignment
//! template is facexlib's 512² one and not `arcface::ARCFACE_DST_112` rescaled -
//! wiring the wrong template would silently degrade every restoration. An
//! unaligned photo still restores; it just is not the reference recipe.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Ctx, Filter, Shape};
use serde_json::json;

use crate::config::CodeFormerConfig;
use crate::model::CodeFormer;

/// The model id used on the CLI (`brain do restore …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/codeformer";

/// [`crate::model::KERNELS`] plus the two kernels only the *serving* path
/// dispatches: `resize_bilinear` (the input is any size, the graph is 512²) and
/// `film_chan` (the `[0,1] <-> [-1,1]` value-range affine, via `imaging::Ctx`).
///
/// Appended, never reordered - `crate::model` and `vae::blocks::Builder` address
/// their kernel slots positionally, so extending at the tail keeps ONE kernel
/// index space and leaves every existing slot valid. The test below pins that.
///
/// A `const` array (not a `Vec`) so it is `'static`: `gpu_core::testgpu::dev`
/// and the weak device pool key on a `&'static` kernel set.
const N_MODEL: usize = crate::model::KERNELS.len();
pub const SERVING_PIPELINES: [(&str, &str); N_MODEL + 2] = serving_set();

const fn serving_set() -> [(&'static str, &'static str); N_MODEL + 2] {
    let mut k = [("", ""); N_MODEL + 2];
    let mut i = 0;
    while i < N_MODEL {
        k[i] = crate::model::KERNELS[i];
        i += 1;
    }
    k[N_MODEL] = ("resize_bilinear", kernels::RESIZE_BILINEAR);
    k[N_MODEL + 1] = ("film_chan", kernels::FILM_CHAN);
    k
}

/// The `restore_face` schema. One function, so the [`Provider`], the residency
/// adapter and the static manifest cannot drift apart.
pub fn restore_spec() -> ActionSpec {
    ActionSpec::new("restore_face", "blind face restoration with a fidelity dial (CodeFormer)")
        .param(
            ParamSpec::new("w", ParamType::Float, "identity-fidelity dial: 0 = maximum quality, 1 = maximum fidelity to the input")
                .default(json!(0.5)),
        )
        .input(BlobSpec::new("image", Media::Image, "the (ideally aligned) face to restore; resized to 512x512").required())
        .output(BlobSpec::new("image", Media::Image, "the restored 512x512 face, RGB in [0,1]"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "CodeFormer blind face restoration: a degraded face -> a restored one, with the fidelity dial w.", vec![restore_spec()])
}

/// Resolve the checkpoint: `path` may be `codeformer.pth` itself or the
/// directory holding it (the layout `tests/parity.rs` and
/// `BRAIN_CODEFORMER_WEIGHTS` already use).
pub fn checkpoint_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        p.join("codeformer.pth").to_string_lossy().into_owned()
    } else {
        path.to_string()
    }
}

/// Import `codeformer.pth` and build the model on `gpu`.
pub fn load(path: &str, gpu: Gpu) -> Result<CodeFormer, String> {
    let cfg = CodeFormerConfig::codeformer();
    let im = crate::import::load(&checkpoint_path(path), &cfg)?;
    // `taps = false`: the parity ladder needs the recorded intermediates, a
    // served call does not, and each tap is a live device buffer.
    Ok(CodeFormer::new(cfg, &im.tensors, gpu, false))
}

// ===================== the shared work =====================

/// A built CodeFormer - the single implementation of `restore_face`, shared by
/// the [`RestoreProvider`] and the residency adapter
/// (`crates/cli/src/resident_restore.rs`).
pub struct Session {
    model: CodeFormer,
}

impl Session {
    pub fn new(model: CodeFormer) -> Session {
        Session { model }
    }

    /// Run one `restore_face` invocation (already validated against
    /// [`restore_spec`]).
    pub fn restore_face(&self, inv: &Invocation) -> ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let side = self.model.config().img_size();
        let fidelity = inv.get_f64("w").unwrap_or(0.5) as f32;
        if !(0.0..=1.0).contains(&fidelity) {
            return Err(format!("restore: w must be in [0, 1], got {fidelity}"));
        }

        // Resize to the graph's square and map [0,1] -> [-1,1], both on the
        // device (`resize_bilinear` + the one `film_chan` affine). The layout
        // permutation around them is host glue by the `crates/imaging` rule.
        let ctx = Ctx::new(self.model.gpu());
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);
        let src = ctx.upload("restore.caps.src", &chw);
        let (small, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), side, side, Filter::Bilinear, AlignCorners::HalfPixel);
        let signed = ctx.affine(&small, shape, &[2.0; 3], &[-1.0; 3]);
        let input = ctx.download(&signed, shape.numel());

        let out = self.model.restore(&input, fidelity);

        // Back to the wire format: [-1,1] -> [0,1] on the device, then HWC.
        let dev = ctx.upload("restore.caps.out", &out.image);
        let unit = ctx.affine(&dev, shape, &[0.5; 3], &[0.5; 3]);
        let restored = ctx.download(&unit, shape.numel());
        let hwc_out = imaging::pixels::chw_to_hwc(&restored, 3, side as usize, side as usize);

        Ok(Outcome::new()
            .set("width", json!(side))
            .set("height", json!(side))
            .set("source_width", json!(w))
            .set("source_height", json!(h))
            .set("w", json!(fidelity))
            .set("codes", json!(out.indices.len()))
            .blob("image", capability::blob::image_blob(&hwc_out, side, side, 3)))
    }
}

// ===================== the provider =====================

/// The executable CodeFormer behind the manifest. Construction is free - the
/// checkpoint imports lazily on the first call and stays resident.
pub struct RestoreProvider {
    weights: String,
    hot: Arc<Mutex<Option<(String, Session)>>>,
}

impl RestoreProvider {
    /// `weights` is `codeformer.pth` or the directory holding it; it comes from
    /// a CLI flag or `BRAIN_CODEFORMER_WEIGHTS`, never a baked-in path.
    pub fn new(weights: impl Into<String>) -> RestoreProvider {
        RestoreProvider { weights: weights.into(), hot: Arc::new(Mutex::new(None)) }
    }
    /// `BRAIN_CODEFORMER_WEIGHTS` - `None` when unset or when it does not resolve to
    /// a file that exists. Deliberately NOT falling back to `BRAIN_VQGAN_WEIGHTS`:
    /// that one commonly points at `vqgan_code1024.pth`, which has none of the
    /// CodeFormer tensors, so the fallback would register a model whose every
    /// call fails.
    pub fn from_env() -> Option<RestoreProvider> {
        let path = std::env::var("BRAIN_CODEFORMER_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        std::path::Path::new(&checkpoint_path(&path)).exists().then(|| RestoreProvider::new(path))
    }
}

impl Provider for RestoreProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "restore_face")
            .then(|| Arc::new(RestoreAction { weights: self.weights.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct RestoreAction {
    weights: String,
    hot: Arc<Mutex<Option<(String, Session)>>>,
}

impl Action for RestoreAction {
    fn spec(&self) -> ActionSpec {
        restore_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut guard = self.hot.lock().map_err(|_| "restore: hot model lock poisoned")?;
        if !matches!(&*guard, Some((p, _)) if *p == self.weights) {
            *guard = None; // free the old build before importing another checkpoint
            let gpu = Gpu::new(&SERVING_PIPELINES);
            *guard = Some((self.weights.clone(), Session::new(load(&self.weights, gpu)?)));
        }
        guard.as_ref().expect("built above").1.restore_face(inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::{Blob, Registry};

    #[test]
    fn manifest_declares_restore_face() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "restore_face");
        assert!(!a.streaming, "restoration is one-shot (the image is the single artifact)");
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.required));
        assert_eq!(a.outputs[0].name, "image");
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let inv = a.validate(Invocation::new().blob("image", img)).unwrap();
        assert_eq!(inv.get_f64("w"), Some(0.5), "the fidelity dial must default, not be required");
        assert!(a.validate(Invocation::new()).is_err(), "the image is required");
    }

    /// The serving pipelines must EXTEND the model's kernel set, never reorder
    /// it: `crate::model` and `vae::blocks::Builder` index slots positionally.
    #[test]
    fn serving_pipelines_only_append() {
        let base = crate::model::KERNELS;
        let ext = SERVING_PIPELINES;
        assert_eq!(ext[..base.len()], base[..], "the shared prefix must stay identical");
        for (name, _) in &ext[base.len()..] {
            assert!(!base.iter().any(|(n, _)| n == name), "{name} is already registered - appending it would duplicate a pipeline");
        }
    }

    #[test]
    fn a_directory_resolves_to_the_released_file_name() {
        assert_eq!(checkpoint_path("/x/codeformer.pth"), "/x/codeformer.pth");
        // a path that is not a directory on this box stays verbatim
        assert_eq!(checkpoint_path("/nonexistent/dir"), "/nonexistent/dir");
    }

    /// `codeformer.pth` is not on every box, and `w` outside `[0,1]` is a user
    /// error: both must surface as a clean `ActionResult`, never a panic.
    #[test]
    fn bad_w_and_missing_weights_are_clean_errors() {
        let mut reg = Registry::new();
        reg.register(Arc::new(RestoreProvider::new("/nonexistent/codeformer.pth")));
        let img = || Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let err = reg.run(MODEL, "restore_face", Invocation::new().blob("image", img()), &mut |_| {}).unwrap_err();
        assert!(!err.is_empty(), "expected a descriptive error, got: {err}");
        // out-of-range w is rejected by the action, not by the schema (the spec
        // has no range type) - assert the message names the bound.
        assert!(restore_spec().validate(Invocation::new().blob("image", img()).set("w", json!(2.0))).is_ok());
    }
}
