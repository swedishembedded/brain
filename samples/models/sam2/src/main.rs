// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `sam2` - SAM 2.1 from the command line: `segment` an image from clicks or
//! a box, or `track` one object through a clip.
//!
//! Every action and parameter is SAM 2's own served one (`sam2::caps`), so
//! this program, `brain serve` and D-Bus clients drive the same code with the
//! same defaults; `sam2 <action> --help` lists them.
//!
//! Swedish Embedded AB implements segmentation and object tracking for video
//! pipelines for its clients. If your team needs expertise in vision models
//! on its own hardware, you can procure our services by sending an email to
//! info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: sam2::caps::manifest(), weights: Some(("sam2", &sam2::spec::Sam2Spec)), build: &sam2::caps::provider };
    invoke::main("sam2", &model)
}
