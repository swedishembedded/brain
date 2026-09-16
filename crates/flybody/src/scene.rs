// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Building a flight-capable model out of the published walking one.
//!
//! The distributed fruit-fly MJCF describes a body that can walk. It cannot
//! fly, and the reason is not a tuning matter: its wing geoms carry MuJoCo's
//! DEFAULT fluid model, which approximates a body by its inertia box and
//! produces almost no lift from a thin flapping plate. A wingbeat driven into
//! that model moves the wings correctly and generates nothing. Everything else
//! about the run looks healthy while it happens, which is what makes it worth
//! a module rather than a line.
//!
//! Four changes separate the two models, and all four are properties of the
//! MODEL rather than of the controller:
//!
//! 1. The wing aerodynamic surfaces switch to MuJoCo's ellipsoid fluid model
//!    with coefficients fitted for a fly's wing. This is the one that decides
//!    whether flight is possible at all.
//! 2. The wing actuators get a far higher gain. A wingbeat is a resonant
//!    oscillation driven near its natural frequency, and the walking model's
//!    gains cannot reach the stroke amplitudes flight needs.
//! 3. The wing hinge gets more damping, which is what keeps that resonance
//!    from running away.
//! 4. The integrator timestep drops. A 218 Hz wingbeat resolved at the walking
//!    model's timestep is 46 steps per cycle, which integrates the STROKE
//!    adequately and the fluid forces on a reversing wing badly.
//!
//! The rewrite is textual and every substitution is COUNTED. An upstream model
//! that renames a default class or reformats an attribute makes a pattern miss,
//! and a silent miss here is a fly that flaps and does not fly - so a miss is
//! an error naming the pattern instead.

use std::path::{Path, PathBuf};

/// How a flight model differs from the walking one. Defaults are the values
/// the published flight tasks use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flight {
    /// Integrator timestep, seconds.
    pub timestep: f64,
    /// Wing actuator gain, applied to all three wing degrees of freedom.
    pub wing_gain: f64,
    /// Wing hinge damping.
    pub wing_damping: f64,
    /// Ellipsoid-fluid-model coefficients for the wing surfaces:
    /// blunt drag, slender drag, angular drag, Kutta lift, Magnus lift.
    pub fluidcoef: [f64; 5],
}

impl Default for Flight {
    fn default() -> Self {
        Flight {
            timestep: 5e-5,
            wing_gain: 18.0,
            wing_damping: 0.007_769_230,
            fluidcoef: [1.0, 0.5, 1.5, 1.7, 1.0],
        }
    }
}

/// Replace exactly `want` occurrences of `from`, or say which pattern missed.
fn replace_exactly(text: &mut String, from: &str, to: &str, want: usize, what: &str) -> Result<(), String> {
    let found = text.matches(from).count();
    if found != want {
        return Err(format!(
            "the fruit-fly model does not look like the one this was written against: \
             expected {want} occurrence(s) of {from:?} while setting {what}, found {found}"
        ));
    }
    *text = text.replace(from, to);
    Ok(())
}

/// Write a flight-capable copy of `fruitfly_xml` into `dir`, and return its path.
///
/// The copy lives in a different directory from the model's 85 mesh and texture
/// files, so the compiler's search paths are made absolute rather than the
/// assets being duplicated. Nothing under the source tree is written to: the
/// published model may be read-only, and in the licence terms this workspace
/// operates under it is not ours to modify in place.
pub fn flight_model(fruitfly_xml: &Path, dir: &Path, cfg: Flight) -> Result<PathBuf, String> {
    let assets = fruitfly_xml
        .parent()
        .ok_or("the model path has no directory to resolve its meshes against")?
        .canonicalize()
        .map_err(|e| format!("{}: {e}", fruitfly_xml.display()))?;
    let mut text = std::fs::read_to_string(fruitfly_xml).map_err(|e| format!("{}: {e}", fruitfly_xml.display()))?;

    let abs = assets.display().to_string();
    replace_exactly(
        &mut text,
        "<compiler autolimits=\"true\" angle=\"radian\"/>",
        &format!("<compiler autolimits=\"true\" angle=\"radian\" meshdir=\"{abs}\" texturedir=\"{abs}\"/>"),
        1,
        "the asset search paths",
    )?;
    replace_exactly(&mut text, "timestep=\"0.0001\"", &format!("timestep=\"{}\"", cfg.timestep), 1, "the timestep")?;
    replace_exactly(
        &mut text,
        "<joint stiffness=\"0.01\" damping=\"0.0005\"/>",
        &format!("<joint stiffness=\"0.01\" damping=\"{}\"/>", cfg.wing_damping),
        1,
        "the wing hinge damping",
    )?;
    for (dof, gain) in [("yaw", "3"), ("roll", "2"), ("pitch", "1")] {
        replace_exactly(
            &mut text,
            &format!("<general gainprm=\"{gain}\"/>"),
            &format!("<general gainprm=\"{}\"/>", cfg.wing_gain),
            1,
            &format!("the {dof} actuator gain"),
        )?;
    }
    // Set on the wing-fluid DEFAULT rather than on the two geoms that use it.
    // The class name itself appears three times - once defining the class and
    // twice using it - so matching the name would also rewrite the definition
    // into nonsense; matching the default's own geom line touches exactly the
    // place the attribute belongs.
    let coef = cfg.fluidcoef.map(|c| c.to_string()).join(" ");
    replace_exactly(
        &mut text,
        "<geom type=\"ellipsoid\" group=\"3\" mass=\"0\"/>",
        &format!("<geom type=\"ellipsoid\" group=\"3\" mass=\"0\" fluidshape=\"ellipsoid\" fluidcoef=\"{coef}\"/>"),
        1,
        "the wing aerodynamic model",
    )?;

    let out = dir.join("fruitfly-flight.xml");
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(out)
}

/// The same, wrapped in a scene with a sky and no ground, which is what a
/// hovering fly wants: a floor at the model's own origin is a surface the wings
/// strike on the first downstroke.
pub fn flight_scene(fruitfly_xml: &Path, dir: &Path, cfg: Flight) -> Result<PathBuf, String> {
    let body = flight_model(fruitfly_xml, dir, cfg)?;
    let scene = dir.join("flight-scene.xml");
    let text = format!(
        r#"<mujoco model="brain-fly-flight">
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="100" height="100"/>
  </asset>
  <include file="{}"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
  </worldbody>
</mujoco>
"#,
        body.display()
    );
    std::fs::write(&scene, text).map_err(|e| format!("{}: {e}", scene.display()))?;
    Ok(scene)
}
