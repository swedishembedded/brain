// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The VQ autoencoder behind the generalized [`capability`] interface — what
//! makes `brain caps vqgan` / `brain do vqgan encode …`, the D-Bus `Run` method
//! and `brain perf`'s `CapabilityTarget` work with no VQ-specific plumbing in
//! the CLI or the transports.
//!
//! Two actions, the two halves of the autoencoder:
//!
//! * **`encode`** — image in, one codebook index per latent position out
//!   (`u32` little-endian, `Media::Bytes`, meta `{lh, lw, codebook_size}`), plus
//!   the mean squared assignment distance as a quantisation-error readout.
//! * **`decode`** — those indices in, the generated image out.
//!
//! Round-tripping them is `reconstruct`; it is deliberately NOT a third action,
//! because the whole point of a discrete latent is that the codes travel
//! (compressed, edited, predicted by another model) between the two calls. A
//! client that wants the round trip issues both — over D-Bus the codes come back
//! as an fd and go straight back in as one.
//!
//! # Value range and geometry
//!
//! The reference feeds RGB in `[-1, 1]`
//! (`tools/codeformer_dump_reference.py`), brain's wire format is RGB in
//! `[0, 1]`; the conversion is the one `film_chan` affine via `imaging::Ctx`,
//! never a host loop. The graph is built for a fixed square side (any multiple
//! of the 32× downscale), so `size` is part of the residency instance key.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use imaging::{AlignCorners, Ctx, Filter, Shape};
use serde_json::json;

use crate::config::VqganConfig;
use crate::model::Vqgan;

/// The model id used on the CLI (`brain do vqgan …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "vqgan";

/// The default square side the graph is built for — the released checkpoints'
/// training resolution.
pub const DEFAULT_SIZE: i64 = 512;

/// [`crate::model::KERNELS`] plus the two kernels only the *serving* path
/// dispatches: `resize_bilinear` (the input is any size, the graph is fixed) and
/// `film_chan` (the `[0,1] <-> [-1,1]` value-range affine, via `imaging::Ctx`).
///
/// Appended, never reordered — `vae::blocks::Builder` addresses its slots
/// positionally, so extending at the tail keeps ONE kernel index space and
/// leaves every existing slot valid. The test below pins that.
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

/// Reject a side the graph cannot be built for BEFORE the checkpoint is read —
/// `Vqgan::new` asserts (panics) on it.
pub fn check_size(size: u32) -> Result<(), String> {
    let scale = VqganConfig::codeformer().downscale();
    if size == 0 || !size.is_multiple_of(scale) {
        return Err(format!("vqgan: size {size} is not a positive multiple of the {scale}x downscale"));
    }
    Ok(())
}

fn size_param() -> ParamSpec {
    ParamSpec::new("size", ParamType::Int, "square side the graph is built for (a multiple of the 32x downscale)")
        .default(json!(DEFAULT_SIZE))
}

pub fn encode_spec() -> ActionSpec {
    ActionSpec::new("encode", "image -> one codebook index per latent position (VQ encoder + nearest-code assignment)")
        .param(size_param())
        .input(BlobSpec::new("image", Media::Image, "the image to encode; resized to size x size").required())
        .output(BlobSpec::new("codes", Media::Bytes, "u32 little-endian, row-major over the [lh, lw] latent grid"))
}

pub fn decode_spec() -> ActionSpec {
    ActionSpec::new("decode", "codebook indices -> an image (codebook gather + VQ generator)")
        .param(size_param())
        .input(BlobSpec::new("codes", Media::Bytes, "u32 little-endian, exactly lh*lw of them").required())
        .output(BlobSpec::new("image", Media::Image, "the generated image, RGB in [0,1]"))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "VQGAN discrete autoencoder: images <-> codebook indices (the VQ core CodeFormer is built on).", vec![encode_spec(), decode_spec()])
}

/// Resolve the checkpoint: `path` may be a `.pth`/`.safetensors` file or a
/// directory holding `codeformer.pth` / `vqgan_code1024.pth` (the layout
/// `BRAIN_VQGAN_WEIGHTS` already uses in the parity tests).
pub fn checkpoint_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if !p.is_dir() {
        return path.to_string();
    }
    for name in ["vqgan_code1024.pth", "codeformer.pth"] {
        let c = p.join(name);
        if c.exists() {
            return c.to_string_lossy().into_owned();
        }
    }
    p.join("vqgan_code1024.pth").to_string_lossy().into_owned()
}

/// Import a released checkpoint and build both graphs for a `size × size` input.
pub fn load(path: &str, size: u32, gpu: Gpu) -> Result<Vqgan, String> {
    check_size(size)?;
    let cfg = VqganConfig::codeformer();
    let file = checkpoint_path(path);
    // Say WHICH file a directory resolved to. `codeformer.pth` and
    // `vqgan_code1024.pth` sit in the same release directory, share every VQ
    // tensor name and produce visibly different codes — resolving silently is
    // how someone spends an afternoon comparing against the wrong goldens.
    if file != path {
        eprintln!("vqgan: {path} -> {file}");
    }
    let im = crate::import::load(&file, &cfg)?;
    // `taps = false`: the parity ladder needs the recorded intermediates, a
    // served call does not, and each tap pins a live device buffer.
    Ok(Vqgan::new(cfg, &im.tensors, size, size, gpu, false))
}

// ===================== the shared work =====================

/// A built VQ autoencoder — the single implementation of `encode`/`decode`,
/// shared by the [`VqganProvider`] and the residency adapter
/// (`crates/cli/src/resident_restore.rs`).
pub struct Session {
    model: Vqgan,
}

impl Session {
    pub fn new(model: Vqgan) -> Session {
        Session { model }
    }

    pub fn encode(&self, inv: &Invocation) -> ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let (lh, lw) = self.model.latent_size();
        let side = lh * self.model.config().downscale();

        // Resize to the graph's square and map [0,1] -> [-1,1], both on the
        // device; the layout permutation around them is host glue.
        let ctx = Ctx::new(self.model.gpu());
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);
        let src = ctx.upload("vqgan.caps.src", &chw);
        let (small, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), side, side, Filter::Bilinear, AlignCorners::HalfPixel);
        let signed = ctx.affine(&small, shape, &[2.0; 3], &[-1.0; 3]);

        let (indices, min_dist) = self.model.encode(&ctx.download(&signed, shape.numel()));
        // A reduction to one scalar: the mean squared distance to the chosen
        // code — the quantisation error, which is what makes the codes
        // interpretable to a caller.
        let mse = if min_dist.is_empty() { 0.0 } else { min_dist.iter().sum::<f32>() as f64 / min_dist.len() as f64 };
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        Ok(Outcome::new()
            .set("lh", json!(lh))
            .set("lw", json!(lw))
            .set("codes", json!(indices.len()))
            .set("codebook_size", json!(self.model.config().codebook_size))
            .set("quant_mse", json!(mse))
            .blob(
                "codes",
                Blob::new(Media::Bytes, bytes)
                    .with_meta(json!({"lh": lh, "lw": lw, "dtype": "u32", "codebook_size": self.model.config().codebook_size})),
            ))
    }

    pub fn decode(&self, inv: &Invocation) -> ActionResult {
        let b = inv.get_blob("codes").ok_or("vqgan decode: missing required input 'codes'")?;
        let (lh, lw) = self.model.latent_size();
        let t = (lh * lw) as usize;
        if b.bytes.len() != 4 * t {
            return Err(format!("vqgan decode: {} bytes of codes, expected {} (= 4 x {lh} x {lw})", b.bytes.len(), 4 * t));
        }
        let indices: Vec<u32> = b.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
        // `Vqgan::decode` PANICS on an out-of-range index (the `embed` gather has
        // no bounds check); over the wire the indices are caller-supplied, so
        // reject them here as an error instead.
        let k = self.model.config().codebook_size;
        if let Some(&bad) = indices.iter().find(|&&i| i >= k) {
            return Err(format!("vqgan decode: code index {bad} out of range for {k} codes"));
        }

        let out = self.model.decode(&indices);
        let side = lh * self.model.config().downscale();
        let shape = Shape::new(1, self.model.config().out_channels, side, side);
        let ctx = Ctx::new(self.model.gpu());
        let dev = ctx.upload("vqgan.caps.out", &out);
        let unit = ctx.affine(&dev, shape, &[0.5; 3], &[0.5; 3]);
        let hwc = imaging::pixels::chw_to_hwc(&ctx.download(&unit, shape.numel()), 3, side as usize, side as usize);
        Ok(Outcome::new()
            .set("width", json!(side))
            .set("height", json!(side))
            .blob("image", capability::blob::image_blob(&hwc, side, side, 3)))
    }

    /// Dispatch by action name — the seam the residency `Instance` uses.
    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "encode" => self.encode(inv),
            "decode" => self.decode(inv),
            other => Err(format!("vqgan: unknown action '{other}'")),
        }
    }
}

// ===================== the provider =====================

/// The executable VQ autoencoder behind the manifest. Construction is free — the
/// checkpoint imports lazily on the first call and stays resident per `size`.
pub struct VqganProvider {
    weights: String,
    hot: Arc<Mutex<Option<(String, Session)>>>,
}

impl VqganProvider {
    /// `weights` is a released checkpoint or the directory holding one; it comes
    /// from a CLI flag or `BRAIN_VQGAN_WEIGHTS`, never a baked-in path.
    pub fn new(weights: impl Into<String>) -> VqganProvider {
        VqganProvider { weights: weights.into(), hot: Arc::new(Mutex::new(None)) }
    }
    /// `BRAIN_VQGAN_WEIGHTS` — `None` when unset or when it does not resolve to
    /// a file that exists.
    pub fn from_env() -> Option<VqganProvider> {
        let path = std::env::var("BRAIN_VQGAN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        std::path::Path::new(&checkpoint_path(&path)).exists().then(|| VqganProvider::new(path))
    }
}

impl Provider for VqganProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        matches!(name, "encode" | "decode")
            .then(|| Arc::new(VqAction { name: name.to_string(), weights: self.weights.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct VqAction {
    name: String,
    weights: String,
    hot: Arc<Mutex<Option<(String, Session)>>>,
}

impl Action for VqAction {
    fn spec(&self) -> ActionSpec {
        if self.name == "encode" {
            encode_spec()
        } else {
            decode_spec()
        }
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let size = inv.get_i64("size").unwrap_or(DEFAULT_SIZE).max(0) as u32;
        let key = format!("{}:{size}", self.weights);
        let mut guard = self.hot.lock().map_err(|_| "vqgan: hot model lock poisoned")?;
        if !matches!(&*guard, Some((k, _)) if *k == key) {
            *guard = None; // free the old build before importing/rebuilding
            let gpu = Gpu::new(&SERVING_PIPELINES);
            *guard = Some((key, Session::new(load(&self.weights, size, gpu)?)));
        }
        guard.as_ref().expect("built above").1.run(&self.name, inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_encode_and_decode() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 2);
        let e = m.actions.iter().find(|a| a.name == "encode").expect("encode");
        assert_eq!(e.outputs[0].name, "codes");
        assert_eq!(e.outputs[0].media, Media::Bytes);
        let d = m.actions.iter().find(|a| a.name == "decode").expect("decode");
        assert!(d.inputs.iter().any(|b| b.name == "codes" && b.media == Media::Bytes && b.required));
        assert_eq!(d.outputs[0].media, Media::Image);
        // the codes blob a client gets back from `encode` is exactly the one
        // `decode` accepts — the round trip must validate without editing.
        let codes = Blob::new(Media::Bytes, vec![0u8; 4]).with_meta(json!({"lh":1,"lw":1}));
        assert!(d.validate(Invocation::new().blob("codes", codes)).is_ok());
        let inv = e.validate(Invocation::new().blob("image", Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3})))).unwrap();
        assert_eq!(inv.get_i64("size"), Some(DEFAULT_SIZE));
    }

    /// The serving pipelines must EXTEND the model's kernel set, never reorder
    /// it: `vae::blocks::Builder` indexes its slots positionally.
    #[test]
    fn serving_pipelines_only_append() {
        let base = crate::model::KERNELS;
        let ext = SERVING_PIPELINES;
        assert_eq!(ext[..base.len()], base[..], "the shared prefix must stay identical");
        for (name, _) in &ext[base.len()..] {
            assert!(!base.iter().any(|(n, _)| n == name), "{name} is already registered — appending it would duplicate a pipeline");
        }
    }

    /// A size that is not a multiple of the 32x downscale would panic deep in
    /// `Vqgan::new`; it must be an error at the surface instead.
    #[test]
    fn a_bad_size_is_rejected_before_the_graph_is_built() {
        let err = check_size(100).unwrap_err();
        assert!(err.contains("32x downscale"), "{err}");
        assert!(check_size(0).is_err(), "zero is not a valid side");
        assert!(check_size(512).is_ok() && check_size(256).is_ok());
    }

    /// Missing weights and an out-of-range code must both be clean
    /// `ActionResult` errors, never a panic.
    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut reg = Registry::new();
        reg.register(Arc::new(VqganProvider::new("/nonexistent/vqgan.pth")));
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w":1,"h":1,"c":3}));
        let err = reg.run(MODEL, "encode", Invocation::new().blob("image", img), &mut |_| {}).unwrap_err();
        assert!(!err.is_empty(), "expected a descriptive error, got: {err}");
    }
}
