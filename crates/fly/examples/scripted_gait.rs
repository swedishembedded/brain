// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Can this body walk AT ALL, driven by anything?
//!
//! Every negative result so far has been about the nervous system: the local
//! rule does not learn, the reward prefers a corpse, the gain search finds
//! little. All of those share an assumption that has never been tested - that
//! the body, driven through this actuation path, is capable of locomotion in
//! the first place.
//!
//! This tests it with no connectome at all. A hand-written tripod gait writes
//! the actuators directly: alternating triangles of legs swing and stance at a
//! chosen frequency, which is what an insect does and what a central pattern
//! generator would produce. If a scripted gait walks, the body is fine and the
//! limit is upstream in the cord or the coupling. If it does not, no
//! controller was ever going to make it walk and the actuation path itself is
//! the thing to fix.
use flybody::{LegDof, Segment, Side};
use mujoco::{Data, Model, MuJoCo, StateSpec};

const BODY_LENGTH_CM: f64 = 0.25;
/// flybody's physics timestep is 1e-4 s and its walking tasks use 20 of them
/// per control tick, so a control tick is 2 ms.
const PHYSICS_PER_CONTROL: u32 = 20;
const CONTROL_DT: f64 = 0.002;

/// The six legs, with the tripod each belongs to. Tripod A is front-left,
/// middle-right, hind-left; the insect gait alternates the two.
const LEGS: [(Segment, Side, usize); 6] = [
    (Segment::T1, Side::Left, 0),
    (Segment::T2, Side::Right, 0),
    (Segment::T3, Side::Left, 0),
    (Segment::T1, Side::Right, 1),
    (Segment::T2, Side::Left, 1),
    (Segment::T3, Side::Right, 1),
];

fn main() {
    let mj = MuJoCo::load().unwrap();
    let xml = std::env::var("BRAIN_FLYBODY_XML").expect("$BRAIN_FLYBODY_XML");
    let model = Model::from_xml(&mj, &xml).unwrap();
    let names: Vec<String> = model.actuator_names().into_iter().map(|n| n.unwrap_or_default()).collect();
    let index = |dof: LegDof, seg: Segment, side: Side| -> Option<usize> {
        let want = dof.actuator(seg, side);
        names.iter().position(|n| *n == want)
    };

    let seconds: f64 = std::env::var("SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let ticks = (seconds / CONTROL_DT).round() as u32;
    println!("{} actuators, {ticks} control ticks = {seconds:.2} s simulated", model.nu());
    println!("{:>7}  {:>6}  {:>6}  {:>10}  {:>10}  {:>9}", "hz", "swing", "lift", "dx (cm)", "BL/s", "final z");

    // Settle first, so the starting pose is the one gravity puts the body in
    // rather than whatever the file declares.
    let settle = |data: &mut Data, model: &Model| {
        for _ in 0..500 {
            for _ in 0..PHYSICS_PER_CONTROL {
                data.step(model);
            }
        }
    };

    // Ranked on FORWARD travel with the body still upright. Ranking on the
    // magnitude of the displacement would crown a gait that tumbles
    // backwards, and ranking on displacement alone would crown one that
    // collapses onto its belly and slides - both of which this sweep produces.
    let mut best = (f64::MIN, String::new());
    for hz in [4.0f64, 8.0, 12.0, 20.0] {
        for swing in [0.3f32, 0.6, 1.0] {
            for lift in [0.3f32, 0.6, 1.0] {
                let mut data = Data::new(&model).unwrap();
                data.reset(&model);
                settle(&mut data, &model);
                let x0 = data.get(&model, StateSpec::QPOS)[0];

                for tick in 0..ticks {
                    let t = tick as f64 * CONTROL_DT;
                    let mut ctrl = vec![0.0f64; model.nu()];
                    for (seg, side, tripod) in LEGS {
                        let phase = 2.0 * std::f64::consts::PI * hz * t + if tripod == 0 { 0.0 } else { std::f64::consts::PI };
                        // Fore-aft swing on the coxa, levation on the femur a
                        // quarter cycle ahead - the leg lifts as it swings
                        // forward and plants as it pushes back, which is what
                        // makes a step rather than a scuff.
                        if let Some(i) = index(LegDof::Coxa, seg, side) {
                            ctrl[i] = (swing as f64) * phase.sin();
                        }
                        if let Some(i) = index(LegDof::Femur, seg, side) {
                            ctrl[i] = (lift as f64) * (phase + std::f64::consts::FRAC_PI_2).sin();
                        }
                    }
                    data.set(&model, StateSpec::CTRL, &ctrl).unwrap();
                    for _ in 0..PHYSICS_PER_CONTROL {
                        data.step(&model);
                    }
                }

                let q = data.get(&model, StateSpec::QPOS);
                let dx = q[0] - x0;
                let bls = dx / BODY_LENGTH_CM / seconds;
                println!("{hz:>7.1}  {swing:>6.1}  {lift:>6.1}  {dx:>10.4}  {bls:>10.3}  {:>9.4}", q[2]);
                let upright = q[2] > -0.02 && q[2] < 0.10;
                if upright && bls > best.0 {
                    best = (bls, format!("{hz:.1} Hz, swing {swing:.1}, lift {lift:.1}: {bls:+.3} BL/s, final z {:+.4}", q[2]));
                }
            }
        }
    }

    println!("\nbest forward gait with the body still upright: {}", best.1);
    println!("A real fly walks at 1 to 3 BL/s.");
    println!("\nThis is a BODY measurement, not a controller one: no connectome is involved.");
    println!("It bounds what any controller driving these actuators this way could reach.");
}
