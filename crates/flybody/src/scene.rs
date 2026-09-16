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
fn model_text_with_absolute_assets(fruitfly_xml: &Path) -> Result<String, String> {
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
    Ok(text)
}

/// The published model's own integrator timestep.
pub const PUBLISHED_TIMESTEP: f64 = 1e-4;

/// Write a copy of `fruitfly_xml` into `dir` that integrates at `timestep`.
///
/// The one thing that decides whether a walking fly can be watched at natural
/// speed, and it is a FIDELITY dial, not a free one. The physics cost per
/// simulated second is `1/timestep` steps at a fixed cost each, so it scales
/// exactly. Measured with MuJoCo's own `testspeed` on this scene, one P-core,
/// three runs each:
///
/// | timestep | real time |
/// |---|---|
/// | 1e-4 (published) | 0.30x |
/// | 2e-4 | 0.62x |
/// | 4e-4 | 1.5x |
///
/// What is traded for it is contact accuracy: the floor's `solref` time
/// constant has to stay at or above twice the timestep or contacts go soft and
/// then unstable, which is why [`world`] scales the floor with this rather
/// than leaving it at the published 2e-4. A coarser step therefore means a
/// SOFTER floor, and the legs sink further into it before it pushes back.
/// Leave it at the published value for anything being measured.
pub fn ground_model(fruitfly_xml: &Path, dir: &Path, timestep: f64) -> Result<PathBuf, String> {
    // A SCENE fails here rather than three layers down, because the failure it
    // otherwise produces names a compiler attribute and not the mistake. A
    // caller who has been passing `floor.xml` - which works for every other
    // purpose, since it is included verbatim - gets told what is different
    // about this one.
    let mut text = model_text_with_absolute_assets(fruitfly_xml).map_err(|e| {
        format!(
            "{e}\n  a non-default timestep REWRITES the body model, so this needs the fruit-fly MJCF \
             itself rather than a scene that includes it"
        )
    })?;
    replace_exactly(&mut text, "timestep=\"0.0001\"", &format!("timestep=\"{timestep}\""), 1, "the timestep")?;
    let out = dir.join("fruitfly-ground.xml");
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(out)
}

pub fn flight_model(fruitfly_xml: &Path, dir: &Path, cfg: Flight) -> Result<PathBuf, String> {
    let mut text = model_text_with_absolute_assets(fruitfly_xml)?;
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

/// What kind of world the fly is put in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arena {
    /// A ground plane at the model's own standing height, with the walking
    /// model's parameters. What a walking fly needs.
    Ground,
    /// The flight model - ellipsoid wing aerodynamics, stiffer actuators, a
    /// shorter timestep - over the same ground, so the fly can take off from
    /// it and land back on it.
    ///
    /// The floor is kept rather than removed. A hovering fly does not need
    /// one, but a fly that can only hover is not doing anything you would
    /// watch; taking off and landing are the interesting parts and both need a
    /// surface.
    Air,
}

/// A world: a fly, a floor, a sky, and optionally something to fly towards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct World {
    pub arena: Arena,
    pub flight: Flight,
    /// How good the picture has to look, against how long a frame may take.
    pub look: Look,
    /// The GROUND body's integrator timestep. `None` is the published 1e-4;
    /// [`Arena::Air`] takes its own from [`Flight::timestep`] and ignores this.
    /// See [`ground_model`] - this is the one dial that decides whether a
    /// walking fly can be watched at natural speed, and what it costs.
    pub timestep: Option<f64>,
    /// Where to put a food marker, in the model's own centimetres, or `None`
    /// for an empty arena.
    ///
    /// It does not collide with anything. A fly that has to physically push
    /// into its target to reach it would be measuring contact resolution
    /// rather than navigation.
    pub food: Option<[f64; 3]>,
}

impl Default for World {
    fn default() -> Self {
        World { arena: Arena::Ground, flight: Flight::default(), look: Look::default(), timestep: None, food: None }
    }
}

/// What the scene costs to draw.
///
/// The published fruit-fly model asks for `shadowsize="8192"` and
/// `offsamples="24"` - a 67-megatexel shadow map and 24x multisampling. Those
/// are the right numbers for rendering a figure for a paper and the wrong ones
/// for a window somebody is watching, and the difference is not marginal.
/// MEASURED on an Intel Arc iGPU, one 960x720 frame of this scene, median of
/// twenty, timing `mjr_render` with a `glFinish` after it so that the GPU time
/// lands where it is spent rather than on the readback that waits for it:
///
/// | shadowsize / offsamples | GPU draw |
/// |---|---|
/// | 8192 / 24 (the model's own) | 316 ms |
/// | 1024 / 4 | 33 ms |
/// | 512 / 4 | 29 ms |
/// | 256 / 4 | 29 ms |
/// | **0 / 4 (default here)** | **4.6 ms** |
/// | 0 / 8 | 6.0 ms |
/// | 0 / 0 | 3.9 ms |
///
/// Read the middle rows before reaching for a smaller shadow map: below 1024
/// the size stops mattering, because what a shadow costs on this model is not
/// the map, it is drawing 272,550 triangles across 85 meshes a SECOND time
/// from the light. That is a flat ~25 ms whatever the resolution, which is
/// five times the entire rest of the frame - so shadows are off by default and
/// antialiasing, which is nearly free, is on.
///
/// The floor's reflectance is in the same struct because it is the same kind
/// of cost: a reflective floor makes MuJoCo draw the scene a second time,
/// mirrored. Measured at 4x/no-shadow it was inside the noise, and it is off
/// by default anyway because it buys a mirror nobody asked for.
///
/// `mjr_readPixels` costs a further 7 to 25 ms of its own for this 2 MB
/// frame and is NOT included above. Without the `glFinish` the whole render
/// appears inside the readback, which is how a GPU-bound frame gets
/// misdiagnosed as a slow copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Shadow map edge, in texels. `0` turns shadows off entirely.
    pub shadowsize: u32,
    /// Multisample count for the offscreen buffer. `0` turns antialiasing off.
    pub offsamples: u32,
    /// How much of the scene the floor mirrors, `0.0` for a matte floor.
    pub floor_reflectance: f64,
}

impl Default for Look {
    fn default() -> Self {
        Look { shadowsize: 0, offsamples: 4, floor_reflectance: 0.0 }
    }
}

impl Look {
    /// The published model's own visual settings: what to ask for when the
    /// output is a still image rather than a window.
    pub fn still() -> Look {
        Look { shadowsize: 8192, offsamples: 24, floor_reflectance: 0.2 }
    }
}

/// The floor's height in the published walking scene, and the contact softness
/// that goes with it. Both are the distributed model's own numbers: a fly
/// standing on a floor at the wrong height either hovers or sinks into it, and
/// neither looks like a bug until something is measured.
const FLOOR_Z: f64 = -0.132;
/// The contact time constant that goes with the published 1e-4 timestep, and
/// the ratio it stands in: MuJoCo needs `solref[0] >= 2 * timestep` or contacts
/// stop being solvable, and the published pair sits exactly at that ratio. So a
/// scene that integrates coarser gets a proportionally softer floor rather than
/// an unstable one - see [`ground_model`].
const FLOOR_SOLREF_PER_TIMESTEP: f64 = 2.0;

/// Write a complete scene - body, floor, sky, light, optional food - and return
/// its path.
///
/// The body is copied and rewritten into `dir` for [`Arena::Air`] and included
/// from where it lies for [`Arena::Ground`], so a walking run loads exactly the
/// published model and a flying one loads the published model plus the four
/// changes flight needs and nothing else.
pub fn world(fruitfly_xml: &Path, dir: &Path, w: World) -> Result<PathBuf, String> {
    world_extent(w);
    let (body, name) = match w.arena {
        Arena::Air => (flight_model(fruitfly_xml, dir, w.flight)?, "brain-fly-air"),
        Arena::Ground => match w.timestep {
            // Unmodified when nothing is asked for, so a default walking run
            // loads exactly the published model from where it lies.
            None => (
                fruitfly_xml
                    .canonicalize()
                    .map_err(|e| format!("{}: {e}", fruitfly_xml.display()))?,
                "brain-fly-ground",
            ),
            Some(dt) => (ground_model(fruitfly_xml, dir, dt)?, "brain-fly-ground"),
        },
    };
    let food = match w.food {
        Some([x, y, z]) => format!(
            // No collision, and lit from inside so it reads as a marker rather
            // than as an object the fly is expected to bump into.
            r#"    <body name="food" pos="{x} {y} {z}">
      <geom name="food" type="sphere" size="0.05" rgba="1 0.85 0.1 1" contype="0" conaffinity="0" mass="0"/>
      <light pos="0 0 0.2" diffuse=".3 .25 .05" specular="0 0 0"/>
    </body>
"#
        ),
        None => String::new(),
    };
    // What the free camera frames. MuJoCo derives the default camera's
    // distance from the model's own extent, and the extent is derived from
    // everything in the world - so a twenty-centimetre floor around a
    // quarter-centimetre animal produces a correct picture of an empty plain
    // with a speck in it. Stating the extent explicitly points the camera at
    // the fly instead, and the floor stays large enough to walk on.
    // Big enough to hold whatever the fly will be doing: the food if there is
    // any, and the height a flying one starts at. Too small and the animal
    // leaves the frame in the first second; too large and it is a speck.
    let extent = world_extent(w);
    // The `<visual>` block goes AFTER the include, which is what makes it an
    // override: MuJoCo merges every `<visual>` it parses and the last value
    // for an attribute wins.
    let Look { shadowsize, offsamples, floor_reflectance: reflectance } = w.look;
    let timestep = match w.arena {
        Arena::Air => w.flight.timestep,
        Arena::Ground => w.timestep.unwrap_or(PUBLISHED_TIMESTEP),
    };
    let solref = format!("{} 1", FLOOR_SOLREF_PER_TIMESTEP * timestep);
    let scene = dir.join("scene.xml");
    let text = format!(
        r#"<mujoco model="{name}">
  <asset>
    <texture name="brain_sky" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2=".05 .07 .12" width="200" height="200"/>
    <texture name="brain_grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="brain_grid" texture="brain_grid" texrepeat="4 4" texuniform="true" reflectance="{reflectance}"/>
  </asset>
  <include file="{}"/>
  <visual>
    <quality shadowsize="{shadowsize}" offsamples="{offsamples}"/>
  </visual>
  <statistic extent="{extent}" center="0 0 0"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="20 20 .1" material="brain_grid" pos="0 0 {FLOOR_Z}" solref="{solref}"/>
{food}  </worldbody>
</mujoco>
"#,
        body.display()
    );
    std::fs::write(&scene, text).map_err(|e| format!("{}: {e}", scene.display()))?;
    Ok(scene)
}

/// What [`world`] writes as the scene's `<statistic extent>`.
///
/// Public because a camera that follows the animal needs it: MuJoCo scales
/// every camera gesture by the model's extent, so a pan of a given size means
/// a different distance in a different world.
pub fn world_extent(w: World) -> f64 {
    let reach = w.food.map_or(0.0, |f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt());
    // Framed on the ANIMAL, not on the volume it will cross. The camera
    // follows it, so a frame wide enough to contain the whole flight would
    // only make the fly a speck for the entire run; two centimetres is eight
    // body lengths, which shows the animal and enough floor to judge its
    // height above.
    match w.arena {
        Arena::Ground => reach.max(1.0) * 1.2,
        Arena::Air => 2.0,
    }
}

/// A flight scene with no ground, for characterising the airframe alone.
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
