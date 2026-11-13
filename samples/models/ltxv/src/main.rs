// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ltxv` - LTX-2.5 from the command line: every served action (`t2v`,
//! `dfr`, `t2a`, `v2v`, `extend`, `upscale`, `replace`) with every parameter,
//! and `chain`, which turns a folder of numbered stills into one clip.
//!
//! Swedish Embedded AB implements generative video pipelines on its clients'
//! own hardware. If your team needs expertise in video diffusion models and
//! video editing, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod chain;

use std::sync::Arc;

fn main() {
    // `replace` runs SAM 2, FLUX.2 and ArcFace beside LTX-2.5: they are found
    // in the same model store.
    let components: Arc<dyn capability::Components> = Arc::new(loader::components::StoreComponents::new(ltxv::replace::component_specs()));
    let build = move |assembly: &capability::Assembly| ltxv::caps::provider(assembly, components.clone());
    let model = invoke::Model { manifest: ltxv::caps::manifest(), weights: Some(("ltxv", &ltxv::spec::LtxvSpec)), build: &build };
    let chain = invoke::Workflow {
        name: "chain",
        summary: "a folder of numbered stills (image-00.png, image-01.png, ...) into one clip: one t2v segment per pair of consecutive stills, joined at the shared still",
        action: "t2v",
        usage: "<folder> --out video=<file> [--mid] [t2v flags]",
        run: &chain::run,
    };
    invoke::main_with("ltxv", &model, &[chain])
}
