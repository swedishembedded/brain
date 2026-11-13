// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `zipdepth` - ZipDepth monocular depth from the command line: every served
//! action (`infer`, `train`, `calib`) with every parameter, and two windows,
//! `show` (one image) and `camera` (a live webcam), that cycle through the
//! views of the depth as you watch.
//!
//! The actions and their parameters are ZipDepth's own served ones
//! (`zipdepth::caps`), so this program, `brain serve` and D-Bus clients drive
//! the same code with the same defaults; `zipdepth <action> --help` lists
//! them. The windows ([`window`]) are the part that is an application rather
//! than the served model.
//!
//! Swedish Embedded AB implements real-time depth sensing on embedded cameras
//! for its clients. If your team needs expertise in monocular depth on its
//! own hardware, NPUs included, then you can procure our services by sending
//! an email to info@swedishembedded.com.

mod depth;
mod flags;
mod window;
mod workflows;

fn main() {
    let model = invoke::Model { manifest: zipdepth::caps::manifest(), weights: Some(("zipdepth", &zipdepth::spec::ZipdepthSpec)), build: &zipdepth::caps::provider };
    let show = invoke::Workflow {
        name: "show",
        summary: "one image's depth in a window: v cycles the view, [ ] the colormap, Esc quits",
        action: "infer",
        usage: "--image <img> [--npu] [--scale N] [--frames N] [infer's --input --view --colormap --stripes]",
        run: &workflows::show,
    };
    let camera = invoke::Workflow {
        name: "camera",
        summary: "live webcam depth (Linux V4L2, YUYV) in a window: v cycles the view, [ ] the colormap, Esc quits",
        action: "infer",
        usage: "[--device-path /dev/video0] [--res WxH] [--npu] [--scale N] [--frames N] [infer's --input --view --colormap --stripes]",
        run: &workflows::camera,
    };
    invoke::main_with("zipdepth", &model, &[show, camera])
}
