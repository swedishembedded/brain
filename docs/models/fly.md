# The fruit fly (connectome-driven digital animal)

Every other entry in this catalog takes a trained checkpoint and reproduces its
forward pass. This one has no checkpoint. A connectome is a **wiring diagram** -
which neuron connects to which, and how strongly - and the parameters that make
that wiring *function* exist in no file. They have to be found, in a body,
against consequences.

So what brain runs here is not a model in the usual sense. It is a real
*Drosophila* nervous system, reconstructed from electron microscopy by other
people's published work, executed as a spiking network on the GPU, wired to a
physics body through the animal's own motor neurons, and reading that body back
through its own proprioceptors and sense organs.

```text
  descending drive -> [ spiking nervous system ] -> motor neurons -> actuators
                                ^                                        |
                                |                                        v
                          proprioceptors <--------------------- joint state
```

## Support

| Capability | Supported |
|---|---|
| Runs the connectome as a spiking network | [x] |
| Closed sensorimotor loop in MuJoCo | [x] |
| Served over CLI / HTTP / D-Bus | [ ] not a servable architecture; reached through the sample and the examples |
| `gradcheck` | not applicable - see [Correctness](#how-correctness-works-without-a-gradient) |
| Learned walking | [ ] not achieved - see [What works and what does not](#what-works-and-what-does-not) |

## The substrate

Two connectomes joined into one nervous system:

| | |
|---|---|
| Brain | **BANC** (Harvard Dataverse, CC BY 4.0, no account needed) |
| Ventral nerve cord | **MANC** |
| Joined | **178,860 neurons, 15,902,235 synapses** |
| Crossing cells merged | 2,954, from the two datasets' own published correspondence |
| Cord motor neurons reachable from the optic lobe | 78% |
| Olfactory receptor neurons | 3,007 |

The join **identifies** the cells the two reconstructions agree are the same
cell, rather than wiring one dataset's output to the other's input, and drops
BANC's own cord so nothing is counted twice - BANC's cord is four to five times
more sparsely reconstructed than MANC's.

MANC alone (`Cns::Cord`, 23,188 neurons and 5.24M synapses) is what the leg
work sits on.

## The body

The nervous system drives [flybody](https://github.com/TuragaLab/flybody)'s
*Drosophila* model in MuJoCo. The map from neuron to actuator is built from the
connectome's own published `Sub Class` annotations rather than a hand-written
list: **330 leg motor neurons reach 44 actuators**, and every neuron that does
not reach one is reported with a reason.

Three clocks run at genuinely different rates - neural ticks, control ticks and
physics steps - and `fly::Timing` keeps them explicit, because pretending they
are one rate is how a sensorimotor loop ends up running the body faster than the
brain. The closed loop runs at **0.55x real time**.

MuJoCo is loaded at run time through `$BRAIN_MUJOCO_DIR` / `$MUJOCO_DIR`;
nothing in brain builds against it, so a machine without it still builds and
tests green.

## How correctness works without a gradient

`gradcheck` does not apply here. The learning rule is local and three-factor,
not differentiated, so there is no analytic gradient for finite differences to
check. The substitutes are:

- an **independently derived closed form** for the membrane equation, checked
  against the integrator;
- **cross-backend agreement** - the same network on the CPU and the GPU;
- **bit-for-bit snapshot replay**, so a run can be re-measured cold.

Behaviour is gated by objectives that carry their own controls
(`fly::learn::Objective`): a corpse, a **degree-matched shuffle** of the wiring,
and a no-stimulus row. A creature that behaves the same with the wiring
shuffled was not using the wiring, and the control is what makes that visible.

## What works and what does not

This is the honest state, and it is deliberately stricter than "it moves".

| | state |
|---|---|
| **Moves its legs** | **done.** 330 motor neurons on 44 actuators, loop closed at 0.55x real time. |
| **Walks** | **not achieved.** An earlier tuning was a two-second lunge, not a gait: 0.46 body-lengths/s over its own training episode but 0.02 BL/s over ten seconds, worse than the untouched connectome. A rhythm attributed to a specific descending neuron turned out to be an artefact of letting a membrane be driven arbitrarily below rest - it needed about twelve threshold-gaps of hyperpolarisation where a real fly has three, and at a physiological inhibitory reversal it does not oscillate in band at all. Both earlier claims are retracted. |
| **Flies** | **partial.** Two thirds of an episode airborne at 48 BL/s, upright - but a degree-matched shuffle reaches 77% of that, so this objective does not actually require the wiring. Steering is the task that would. |
| **Explores** | **partial.** The olfactory chain carries end to end and is lateralised. Steering is untrained. The mushroom body is wired for learning rather than searched: plasticity is confined to the Kenyon-cell output synapses and gated by identified dopaminergic cells per compartment. |

The reason "walks" is not simply declared done is the second column of that
table. Net forward displacement is a reward a single coordinated lunge scores
as well as a gait does, and searching directly against it suppressed the cord's
recurrent circuitry and drove sensory input straight to the muscles - which is a
reflex, and is exactly what that reward asks for.

## Running it

There is no `brain fly` subcommand; this is not a servable architecture. It is
reached through the interactive sample and the crate's examples.

```bash
make samples/fly/interactive/run ARGS="--connectome <dir>/manc-codex --body <dir>/floor.xml"
```

`W`/`S` change how hard the brain tells the cord to go, `A`/`D` bias the two
halves of the descending population against each other - nothing touches a
joint directly, so a turn has to come out of the cord's own circuitry. Holding
`C` lesions the proprioceptive channel and `--shuffled-connectome` swaps in the
degree-matched shuffle, so both controls are on the surface rather than buried
in a test.

It runs headless too, which is how it is exercised with no display attached:

```bash
SDL_VIDEODRIVER=dummy make samples/fly/interactive/run \
  ARGS="--connectome <dir>/manc-codex --body <dir>/floor.xml --frames 150 --shot fly.ppm"
```

The examples under `crates/fly/examples/` are the measurement surface - gait
analysis, ground and flight checks, reward-sensitivity sweeps, the learning
matrix with its controls, and `replay` for re-measuring a saved tuning cold:

```bash
make crates/fly/examples/walk_search/run ENV="OUT=/tmp/walk.txt"
make crates/fly/examples/replay/run ARGS=/tmp/walk.txt
```

## What it needs

- **MuJoCo**, found at run time (`$BRAIN_MUJOCO_DIR` / `$MUJOCO_DIR`).
- **The connectome export** - a directory containing `manc-codex/` with
  `neurons.csv.gz` and `connections_princeton.csv.gz`; BANC additionally for
  the joined nervous system.
- **The flybody MJCF, with a ground plane.** A body file with no floor
  simulates a fly falling forever, and every other reading still looks healthy
  while it does.
- A GPU for the spiking network, and EGL for the renderer (no display server
  required - rendering goes through an EGL device context, so this works over
  SSH).

## Related

- [`splat`](splat.md) and [`worldmirror2`](worldmirror2.md) - the other places
  brain produces something spatial rather than a tensor.
- [World models](world-models.md) - playable simulation learned from video,
  the other half of "a thing that acts".
