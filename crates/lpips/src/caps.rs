// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS behind the generalized [`capability`] interface - what makes
//! `brain do brain/lpips distance` (and the served surfaces) work with no
//! LPIPS-specific plumbing in the CLI.
//!
//! One action, `distance`: two same-sized RGB images in, the scalar LPIPS
//! distance (lower is closer, identical images are exactly 0) and each AlexNet
//! tap's share of it out, as JSON. The weights are never a caller's to name:
//! the provider is handed the trunk and heads files the model-store resolver
//! found ([`crate::spec::LpipsSpec`]), reads them on the first call and keeps
//! the metric resident on its device for the calls after.
//!
//! Swedish Embedded AB implements image-quality evaluation for 3D
//! reconstruction and generative pipelines for its clients. If your team
//! needs expertise in perceptual metrics or GPU inference, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, Progress, Provider};
use gpu_core::Gpu;
use serde_json::json;

use crate::{Lpips, PIPELINES};

/// The model id used on the CLI (`brain do brain/lpips …`) and the event API.
pub const MODEL: &str = "brain/lpips";

const DISTANCE: &str = "distance";

fn distance_spec() -> ActionSpec {
    ActionSpec::new(DISTANCE, "LPIPS (AlexNet, v0.1) perceptual distance between two same-sized images")
        .input(BlobSpec::new("a", Media::Image, "the first RGB image").required())
        .input(BlobSpec::new("b", Media::Image, "the second RGB image, the same size as the first").required())
}

/// The static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "LPIPS perceptual image distance (AlexNet, v0.1) - lower is perceptually closer.", vec![distance_spec()])
}

/// The executable LPIPS model behind the manifest. Construction is free: the
/// two files are read, validated and put on the device by the first
/// `distance` call.
pub struct LpipsProvider {
    trunk: PathBuf,
    heads: PathBuf,
    metric: Arc<Mutex<Option<Lpips>>>,
}

impl LpipsProvider {
    /// `trunk` and `heads` are the files [`crate::spec::resolve`] (or the
    /// resolver's assembly) found.
    pub fn new(trunk: impl Into<PathBuf>, heads: impl Into<PathBuf>) -> LpipsProvider {
        LpipsProvider { trunk: trunk.into(), heads: heads.into(), metric: Arc::default() }
    }
}

impl Provider for LpipsProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }

    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == DISTANCE).then(|| Arc::new(DistanceAction { trunk: self.trunk.clone(), heads: self.heads.clone(), metric: self.metric.clone() }) as Arc<dyn Action>)
    }
}

struct DistanceAction {
    trunk: PathBuf,
    heads: PathBuf,
    metric: Arc<Mutex<Option<Lpips>>>,
}

impl Action for DistanceAction {
    fn spec(&self) -> ActionSpec {
        distance_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Decode before touching the weights: a bad request is refused without
        // reading the checkpoint.
        let (a, w, h) = capability::blob::decode_image(inv, "a")?;
        let (b, bw, bh) = capability::blob::decode_image(inv, "b")?;
        if (w, h) != (bw, bh) {
            return Err(format!("lpips distance: the images must be the same size (a is {w}x{h}, b is {bw}x{bh})"));
        }

        let mut guard = self.metric.lock().map_err(|_| "lpips: resident metric lock poisoned")?;
        if guard.is_none() {
            let weights = crate::import::read(&self.trunk, &self.heads)?;
            *guard = Some(Lpips::new(Gpu::new(PIPELINES), &weights)?);
        }
        let d = guard.as_mut().expect("built above").distance(&a, &b, w, h, None)?;
        Ok(Outcome::new().set("distance", json!(d.total)).set("layers", json!(d.layers)))
    }
}
