// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL behind the generalized [`capability`] interface - what makes
//! `brain caps sdxl` / `brain do sdxl text2image ...`, the D-Bus `Run` method
//! and `brain perf`'s `CapabilityTarget` work with no SDXL-specific plumbing
//! in the CLI or the transports.
//!
//! One action: **`text2image`** - a prompt in, an HWC RGB image out
//! (`Sdxl::generate`, see its module docs for the full pipeline: two CLIP
//! towers, a discrete Euler scheduler, classifier-free guidance, and a VAE
//! decode). Everything the action does is param decoding + calling
//! [`pipeline::Sdxl`]; nothing here re-implements the sampling loop.
//!
//! # No batching
//!
//! Unlike CLIP/T5's text towers, `Sdxl::generate` is a full multi-step
//! diffusion sample - there is no `[B, ...]` axis to fill, every request runs
//! its own denoising loop, and grouping N of them would still be N loops. The
//! residency adapter (`crates/catalog/src/resident_sdxl.rs`) uses the serial
//! default and says so, the same way `resident_scrfd.rs` does for the face
//! stack.
//!
//! # Size is fixed at build time
//!
//! `pipeline::Sdxl::load` records the UNet graph at one `(h, w)`, so the
//! [`Session`] keeps one built pipeline per size actually requested rather
//! than rebuilding on every call at a size a caller already used.

use std::sync::{Arc, Mutex};

use capability::{hot::Hot, Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::pipeline::{GenerateOptions, Sdxl};

/// The model id used on the CLI (`brain do sdxl ...`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/sdxl";

fn text2image_spec() -> ActionSpec {
    ActionSpec::new("text2image", "Generate an image from a text prompt (SDXL base, classifier-free guidance).")
        .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("negative", ParamType::Str, "negative prompt (only used when guidance > 1.0)").default(json!("")))
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 8)").default(json!(1024)).min(256.0).max(2048.0).step(8.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 8)").default(json!(1024)).min(256.0).max(2048.0).step(8.0))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps").default(json!(30)).min(1.0).max(150.0).step(1.0))
        .param(ParamSpec::new("guidance", ParamType::Float, "classifier-free guidance scale; 1.0 disables CFG").default(json!(5.0)).min(1.0).max(30.0).step(0.1))
        .param(ParamSpec::new("vae_device", ParamType::Str, "where the VAE decode runs, a --device spec (default cpu: the UNet is still resident, and a 768px+ decode beside it can exceed a 24 GB card)").default(json!("cpu")))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random; the outcome reports the one used)"))
        .output(capability::BlobSpec::new("image", capability::Media::Image, "the generated image"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "SDXL text-to-image: UNet2DConditionModel + dual CLIP conditioning.", vec![text2image_spec()])
}

/// Decode the request. A `seed` that is not a non-negative integer is refused;
/// an absent one is drawn, and the outcome reports it.
fn opts_from(inv: &Invocation) -> Result<(String, GenerateOptions), String> {
    let prompt = inv.get_str("prompt").unwrap_or_default();
    let o = GenerateOptions {
        steps: inv.get_i64("steps").unwrap_or(30).max(1) as usize,
        guidance: inv.get_f64("guidance").unwrap_or(5.0) as f32,
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        height: inv.get_i64("height").unwrap_or(1024).max(8) as u32,
        width: inv.get_i64("width").unwrap_or(1024).max(8) as u32,
        negative: inv.get_str("negative").unwrap_or_default(),
        vae_device: inv.get_str("vae_device").unwrap_or_else(|| "cpu".into()),
    };
    Ok((prompt, o))
}

// ===================== the shared work =====================

/// The pipelines on one device, keyed by `(h, w)` - the single implementation
/// of `text2image`, shared by [`SdxlProvider`] and the residency adapter
/// (`crates/catalog/src/resident_sdxl.rs`).
pub struct Session {
    root: String,
    built: Mutex<std::collections::HashMap<(u32, u32), Sdxl>>,
}

impl Session {
    /// `root` is the released diffusers SDXL checkpoint directory.
    pub fn new(root: impl Into<String>) -> Session {
        Session { root: root.into(), built: Mutex::new(std::collections::HashMap::new()) }
    }

    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "text2image" => self.text2image(inv),
            other => Err(format!("sdxl: unknown action '{other}'")),
        }
    }

    fn text2image(&self, inv: &Invocation) -> ActionResult {
        let (prompt, o) = opts_from(inv)?;
        let (h, w) = (o.height, o.width);
        let mut guard = self.built.lock().map_err(|_| "sdxl: pipeline lock poisoned")?;
        let p = match guard.entry((h, w)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(Sdxl::load(&self.root, h, w)?),
        };
        let hwc = p.generate(&prompt, &o)?;
        Ok(Outcome::new().set("seed", json!(o.seed)).blob("image", capability::blob::image_blob(&hwc, w, h, 3)))
    }
}

// ===================== the provider =====================

type HotSession = Arc<Hot<String, Arc<Session>>>;

/// The executable SDXL stack behind the manifest. Construction is free -
/// pipelines import lazily on first use, per requested size.
pub struct SdxlProvider {
    root: String,
    hot: HotSession,
}

impl SdxlProvider {
    pub fn new(root: impl Into<String>) -> SdxlProvider {
        SdxlProvider { root: root.into(), hot: Arc::new(Hot::new()) }
    }

}

impl Provider for SdxlProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "text2image")
            .then(|| Arc::new(SdxlAction { root: self.root.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct SdxlAction {
    root: String,
    hot: HotSession,
}

impl Action for SdxlAction {
    fn spec(&self) -> ActionSpec {
        text2image_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let session = self.hot.get_cloned(self.root.clone(), || Ok(Arc::new(Session::new(self.root.clone()))))?;
        session.run("text2image", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;

    #[test]
    fn manifest_declares_text2image() {
        let m = manifest();
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["text2image"]);
    }

    #[test]
    fn an_unknown_action_is_named_not_ignored() {
        let p = SdxlProvider::new("/nonexistent");
        assert!(p.action("edit").is_none());
    }

    #[test]
    fn size_params_carry_ui_ranges() {
        let spec = text2image_spec();
        let w = spec.params.iter().find(|p| p.name == "width").expect("width param");
        assert_eq!(w.min, Some(256.0));
        assert_eq!(w.step, Some(8.0));
    }

    /// An unseeded request is valid and draws its own seed (a different one
    /// each time), a given seed is kept, and one that is not a non-negative
    /// integer is refused rather than clamped.
    #[test]
    fn an_unseeded_request_draws_a_seed_and_a_malformed_one_is_refused() {
        let unseeded = || Invocation::new().set("prompt", json!("a cat"));
        assert!(text2image_spec().validate(unseeded()).is_ok());
        let seed_of = |inv: Invocation| opts_from(&inv).map(|(_, o)| o.seed);
        assert_ne!(seed_of(unseeded()), seed_of(unseeded()), "two unseeded requests must not share a seed");
        assert_eq!(seed_of(unseeded().set("seed", json!(42))), Ok(42));
        assert!(seed_of(unseeded().set("seed", json!(-1))).is_err());
    }
}
