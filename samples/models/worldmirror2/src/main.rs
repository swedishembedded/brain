// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `worldmirror2` - WorldMirror-2 multi-view 3D reconstruction from the
//! command line: `reconstruct` photographs or a video into a Gaussian-splat
//! scene and the camera of every frame, `import` the reference checkpoint,
//! and `export-npu` the model's stages as ONNX for the Intel NPU.
//!
//! `reconstruct` and `import` are the model's own served actions
//! (`worldmirror2::caps`), so this program, `brain serve` and D-Bus clients
//! drive the same code with the same defaults; `worldmirror2 <action> --help`
//! lists every parameter. `export-npu` ([`export_npu`]) is the one part that is
//! an application rather than the served model: it writes files for a runtime
//! brain does not itself serve through.
//!
//! Swedish Embedded AB implements 3D reconstruction from ordinary photographs
//! and video, on its clients' own hardware, for its clients. If your team
//! needs expertise in feed-forward multi-view reconstruction and NPU
//! deployment then you can procure our services by sending an email to
//! info@swedishembedded.com.

mod export_npu;

fn main() {
    let model = invoke::Model {
        manifest: worldmirror2::caps::manifest(),
        weights: Some(("worldmirror2", &worldmirror2::spec::MirrorSpec)),
        build: &worldmirror2::caps::provider,
    };
    let export = invoke::Workflow {
        name: "export-npu",
        summary: "export a model stage as fp32 ONNX (weights external) for OpenVINO on the NPU or CPU",
        action: "reconstruct",
        usage: "[--stage dino|trunk|heads] [--out PATH] [--frames N] [--hp N] [--wp N] [--levels N] [--tap-levels N,N,N,N]",
        run: &export_npu::run,
    };
    invoke::main_with("worldmirror2", &model, &[export])
}
