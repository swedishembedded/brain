// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `diamond` - the DIAMOND Atari world model from the command line: every
//! served action (`import`, `export`, `finetune`, `rollout`, `replay`,
//! `bench`) with every parameter, and `play`, an interactive window where the
//! keyboard is the game controller and the model dreams the next frame.
//!
//! The actions and their parameters are DIAMOND's own served ones
//! (`diamond::caps`), so this program, `brain serve` and D-Bus clients drive
//! the same code with the same defaults; `diamond <action> --help` lists
//! them. The window ([`play`]) is the part that is an application rather than
//! the served model.
//!
//! Swedish Embedded AB implements learned simulators of machines and
//! environments that run on its clients' own hardware. If your team needs
//! expertise in action-conditioned world models then you can procure our
//! services by sending an email to info@swedishembedded.com.

mod play;

fn main() {
    let model = invoke::Model { manifest: diamond::caps::manifest(), weights: Some(("diamond", &diamond::spec::DiamondSpec)), build: &diamond::caps::provider };
    let play = invoke::Workflow {
        name: "play",
        summary: "play the world model in a window: Space fire, A/D or arrows move, Enter reset, . pause, e step, [ ] quality, Esc quit",
        action: "rollout",
        usage: "[--scale N] [--fps N] [--frames N] [--adaptive] [--renderer auto|software|accelerated] [--record DIR] [rollout's --model --seed --denoise_steps --onnx --seed_context]",
        run: &play::play,
    };
    invoke::main_with("diamond", &model, &[play])
}
