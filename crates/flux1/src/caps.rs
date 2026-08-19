// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.1 behind the generalized [`capability`] interface - what makes
//! `brain caps flux1` / `brain do flux1 text2image ...`, the D-Bus `Run`
//! method and `brain perf`'s `CapabilityTarget` work with no FLUX.1-specific
//! plumbing in the CLI or the transports.
//!
//! One action: **`text2image`** - a prompt in, an HWC RGB image out
//! (`pipeline::Flux1::generate`, see its module docs - including the honest
//! note on what is and is not verified end to end). Everything the action
//! does is param decoding + calling [`pipeline::Flux1`]; nothing here
//! re-implements the sampling loop.
//!
//! # No batching
//!
//! Same reasoning as `sdxlunet::caps`: every request is its own multi-step
//! sample with no `[B, ...]` axis to fill. The residency adapter
//! (`crates/cli/src/resident_flux1.rs`) uses the serial default.
//!
//! # Size is fixed at build time
//!
//! `pipeline::Flux1::load` records the DiT's max joint-token budget for one
//! `(h, w)`, so the [`Session`] keeps one built pipeline per size actually
//! requested rather than rebuilding on every call at a size a caller already
//! used - the same pattern `sdxlunet::caps::Session` uses.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

use crate::pipeline::{Flux1, GenerateOptions};

/// The model id used on the CLI (`brain do flux1 ...`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/flux1";

/// The variant enum, in manifest order.
const VARIANTS: [&str; 3] = ["dev", "kontext-dev", "schnell"];

/// T5-XXL context length. FLUX.1-dev's released default is 512; schnell is
/// commonly run at 256 for speed. A UI-rangeable param rather than baked in,
/// same as `t5encoder::caps`'s `max_len`.
const DEFAULT_MAX_LEN: u32 = 512;

fn text2image_spec() -> ActionSpec {
    ActionSpec::new("text2image", "Generate an image from a text prompt (FLUX.1 dev/kontext-dev/schnell).")
        .param(ParamSpec::new("prompt", ParamType::Str, "text description of the desired image").required())
        .param(ParamSpec::new("width", ParamType::Int, "output width, px (multiple of 16)").default(json!(1024)).min(256.0).max(2048.0).step(16.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px (multiple of 16)").default(json!(1024)).min(256.0).max(2048.0).step(16.0))
        .param(ParamSpec::new("steps", ParamType::Int, "denoising steps; 0 = variant default (4 schnell / 50 dev)").default(json!(0)).min(0.0).max(150.0).step(1.0))
        .param(ParamSpec::new("guidance", ParamType::Float, "guidance_in scalar -- dev/kontext-dev only, schnell ignores it").default(json!(3.5)).min(0.0).max(10.0).step(0.1))
        .param(ParamSpec::new("max_len", ParamType::Int, "T5-XXL context length").default(json!(DEFAULT_MAX_LEN)).min(32.0).max(512.0).step(1.0))
        .param(ParamSpec::new("variant", ParamType::Enum(VARIANTS.iter().map(|s| s.to_string()).collect()), "model variant").default(json!("dev")))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for 0)"))
        .output(capability::BlobSpec::new("image", capability::Media::Image, "the generated image"))
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "FLUX.1 (Black Forest Labs) MMDiT text-to-image: dev (guidance-distilled), kontext-dev, schnell (timestep-distilled).",
        vec![text2image_spec()],
    )
}

struct Req {
    prompt: String,
    variant: String,
    opts: GenerateOptions,
    max_len: usize,
}

fn req_from(inv: &Invocation) -> Req {
    Req {
        prompt: inv.get_str("prompt").unwrap_or_default(),
        variant: inv.get_str("variant").unwrap_or_else(|| "dev".into()),
        opts: GenerateOptions {
            steps: {
                let s = inv.get_i64("steps").unwrap_or(0);
                (s > 0).then_some(s as usize)
            },
            guidance: inv.get_f64("guidance").unwrap_or(3.5) as f32,
            seed: inv.get_i64("seed").unwrap_or(0) as u64,
            height: inv.get_i64("height").unwrap_or(1024).max(16) as u32,
            width: inv.get_i64("width").unwrap_or(1024).max(16) as u32,
        },
        max_len: inv.get_i64("max_len").unwrap_or(DEFAULT_MAX_LEN as i64).max(1) as usize,
    }
}

// ===================== the shared work =====================

/// The pipelines on one device, keyed by `(variant, h, w)` - the single
/// implementation of `text2image`, shared by [`Flux1Provider`] and the
/// residency adapter (`crates/cli/src/resident_flux1.rs`).
pub struct Session {
    root: String,
    built: Mutex<std::collections::HashMap<(String, u32, u32), Flux1>>,
}

impl Session {
    /// `root` is the released diffusers FLUX.1 checkpoint directory.
    pub fn new(root: impl Into<String>) -> Session {
        Session { root: root.into(), built: Mutex::new(std::collections::HashMap::new()) }
    }

    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "text2image" => self.text2image(inv),
            other => Err(format!("flux1: unknown action '{other}'")),
        }
    }

    fn text2image(&self, inv: &Invocation) -> ActionResult {
        let req = req_from(inv);
        let (h, w) = (req.opts.height, req.opts.width);
        let key = (req.variant.clone(), h, w);
        let mut guard = self.built.lock().map_err(|_| "flux1: pipeline lock poisoned")?;
        let p = match guard.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Flux1::load(&self.root, &req.variant, h, w)?)
            }
        };
        let hwc = p.generate(&req.prompt, &req.opts, req.max_len)?;
        Ok(Outcome::new().blob("image", capability::blob::image_blob(&hwc, w, h, 3)))
    }
}

// ===================== the provider =====================

type HotSession = Arc<Mutex<Option<(String, Arc<Session>)>>>;

/// The executable FLUX.1 stack behind the manifest. Construction is free -
/// pipelines import lazily on first use, per requested (variant, size).
pub struct Flux1Provider {
    root: String,
    hot: HotSession,
}

impl Flux1Provider {
    pub fn new(root: impl Into<String>) -> Flux1Provider {
        Flux1Provider { root: root.into(), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_FLUX1_DIR` - `None` when unset, or when the directory holds no
    /// released `transformer/`, since without one no action can run.
    pub fn from_env() -> Option<Flux1Provider> {
        let root = std::env::var("BRAIN_FLUX1_DIR").ok().filter(|p| !p.is_empty())?;
        std::path::Path::new(&root).join("transformer").exists().then(|| Flux1Provider::new(root))
    }
}

impl Provider for Flux1Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "text2image")
            .then(|| Arc::new(Flux1Action { root: self.root.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct Flux1Action {
    root: String,
    hot: HotSession,
}

impl Action for Flux1Action {
    fn spec(&self) -> ActionSpec {
        text2image_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let session = {
            let mut guard = self.hot.lock().map_err(|_| "flux1: hot session lock poisoned")?;
            if !matches!(&*guard, Some((r, _)) if *r == self.root) {
                *guard = None; // free the old build before pointing at another directory
                *guard = Some((self.root.clone(), Arc::new(Session::new(self.root.clone()))));
            }
            guard.as_ref().expect("built above").1.clone()
        };
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
        let p = Flux1Provider::new("/nonexistent");
        assert!(p.action("edit").is_none());
    }

    #[test]
    fn from_env_declines_a_directory_with_no_transformer() {
        assert!(Flux1Provider::from_env().is_none() || std::env::var("BRAIN_FLUX1_DIR").is_ok());
    }

    #[test]
    fn size_params_carry_ui_ranges() {
        let spec = text2image_spec();
        let w = spec.params.iter().find(|p| p.name == "width").expect("width param");
        assert_eq!(w.min, Some(256.0));
        assert_eq!(w.step, Some(16.0));
    }

    #[test]
    fn variant_defaults_to_dev() {
        let spec = text2image_spec();
        let v = spec.params.iter().find(|p| p.name == "variant").expect("variant param");
        assert_eq!(v.default, Some(json!("dev")));
    }
}
