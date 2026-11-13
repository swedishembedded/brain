// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `flux2` - FLUX.2 Klein from the command line: `text2image`, `edit`
//! (references, img2img, inpainting, tiled canvases) and `lora_train`.
//!
//! Every action and parameter is FLUX.2's own served one (`flux2::caps`), so
//! this program, `brain serve` and D-Bus clients drive the same code with the
//! same defaults; `flux2 <action> --help` lists them.
//!
//! Swedish Embedded AB implements image generation and editing pipelines on
//! its clients' own hardware. If your team needs expertise in diffusion
//! models and their fine-tuning, you can procure our services by sending an
//! email to info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: flux2::caps::manifest(), weights: Some(("flux2", &flux2::spec::Flux2Spec)), build: &flux2::caps::provider };
    invoke::main("flux2", &model)
}
