// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! What does a REAL fly's walk score on this crate's gait analyser?
//!
//! Every gait number in this repository is produced by `gait::analyse`, and
//! nothing has ever checked that it recognises walking when it sees it. That
//! is the wrong way round: a metric used to decide whether a simulated animal
//! walks should first be shown a recorded animal that does.
//!
//! The recording is flybody's own walking reference - real fly kinematics,
//! a hundred snippets of a few hundred frames each, the whole body state per
//! frame. The coxa joints are read out of it exactly as the simulated fly's
//! are, and pushed through the same analyser.
//!
//! Three things this can tell us and nothing else can:
//!
//!   the ceiling      what score a real walk actually gets, which is the only
//!                    honest target for a simulated one
//!   the frequency    a real fly's step rate, against the 4-25 Hz band the
//!                    analyser assumes
//!   the sign         whether `tripod` is positive for a real tripod gait. If
//!                    it is negative the analyser has its leg triangles the
//!                    wrong way round, and every gait score ever reported here
//!                    has been upside down.
use fly::gait::Trace;
use fly::reference::Reference;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set");
        std::process::exit(2)
    })
}

fn main() {
    let reference = Reference::load(env("BRAIN_FLY_REFERENCE")).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    println!(
        "{} snippets, nq {}, nv {}, timestep {} s",
        reference.snippets(),
        reference.nq(),
        reference.nv(),
        reference.timestep()
    );

    // Which generalized coordinates the six coxa joints are. Taken from the
    // body model rather than guessed, by the same route the simulated fly
    // uses: `flybody::LegDof::Coxa` names the actuator and the model says
    // which coordinate it moves.
    let mj = mujoco::MuJoCo::load().unwrap();
    let model = mujoco::Model::from_xml(&mj, std::path::Path::new(&env("BRAIN_FLYBODY_XML"))).unwrap();
    let mut coxa = [usize::MAX; 6];
    for (i, (seg, side, _)) in flybody::LEGS.iter().enumerate() {
        let name = flybody::LegDof::Coxa.actuator(*seg, *side);
        match model.joint_qpos(&name) {
            Some(q) => coxa[i] = q,
            None => {
                eprintln!("the body has no joint for {name}");
                std::process::exit(1);
            }
        }
    }
    println!("coxa coordinates: {coxa:?}");

    let mut scores: Vec<(f64, f64, f64, f64)> = Vec::new();
    for s in 0..reference.snippets() {
        let n = reference.len(s);
        if n < 400 {
            continue;
        }
        let mut trace = Trace::new(reference.timestep());
        for i in 0..n {
            let Some((qpos, _)) = reference.frame(s, i) else { continue };
            let mut per_leg = [0.0f32; 6];
            for (slot, &q) in per_leg.iter_mut().zip(&coxa) {
                *slot = qpos.get(q).copied().unwrap_or(0.0);
            }
            trace.push(per_leg);
        }
        if let Some(g) = fly::analyse_gait(&trace) {
            scores.push((g.score(), g.step_hz, g.tripod, g.rhythmicity));
        }
    }

    if scores.is_empty() {
        eprintln!("no snippet was long enough to analyse; the reference may be the wrong shape");
        std::process::exit(1);
    }
    let mean = |f: fn(&(f64, f64, f64, f64)) -> f64| scores.iter().map(f).sum::<f64>() / scores.len() as f64;
    let best = scores.iter().cloned().fold((0.0, 0.0, 0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
    println!("\n{} snippets analysed", scores.len());
    println!("{:>14}  {:>9}  {:>9}  {:>9}  {:>11}", "", "score", "step Hz", "tripod", "rhythmicity");
    println!(
        "{:>14}  {:>9.3}  {:>9.2}  {:>9.3}  {:>11.3}",
        "mean",
        mean(|x| x.0),
        mean(|x| x.1),
        mean(|x| x.2),
        mean(|x| x.3)
    );
    println!("{:>14}  {:>9.3}  {:>9.2}  {:>9.3}  {:>11.3}", "best", best.0, best.1, best.2, best.3);

    println!("\nthis is the number a simulated fly has to be read against. A real walk");
    println!("scoring low means the analyser is wrong, not that the fly is.");
    if mean(|x| x.2) < 0.0 {
        println!("\nWARNING: tripod is NEGATIVE on real walking, so the analyser's two leg");
        println!("triangles are swapped and every gait score reported here is inverted.");
    }
}
