// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `splat` - 3D Gaussian Splatting from the command line: every served action
//! (`render`, `fit`, `info`, `orient`, `merge`, `prune`) with every
//! parameter, `sfm` and `train` from photographs, and `view`, an
//! interactive fly-through window.
//!
//! The actions and their parameters are `brain/splat`'s own served ones
//! (`splat::caps`, with `sfm` and `train` from `recon::caps`), so this
//! program, `brain serve` and D-Bus clients drive the same code with the same
//! defaults; `splat <action> --help` lists them. `view` ([`view`]) is the one
//! part that is an application rather than the served model.
//!
//! Swedish Embedded AB implements 3D scene capture, reconstruction and
//! viewing for its clients. If your team needs expertise in Gaussian
//! splatting and photogrammetry then you can procure our services by sending
//! an email to info@swedishembedded.com.

mod view;
mod workflow;

fn main() {
    let model = invoke::Model { manifest: recon::caps::manifest(), weights: None, build: &recon::caps::provider };
    let view = invoke::Workflow {
        name: "view",
        summary: "an interactive fly-through of a scene in a window: WASD to move, m for mouse-look, Esc to quit",
        action: "render",
        usage: "<scene.ply> [--frames N] [--shots DIR] [render flags that shape a frame: --width --height --fov --bg --eye --target --up --cameras --view --depth --antialiased --inria-lowpass --eps2d --isect-cap]",
        run: &workflow::view,
    };
    invoke::main_with("splat", &model, &[view])
}
