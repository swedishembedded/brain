# sample: fly/interactive

A real *Drosophila* connectome driving a real body, in a window.

```bash
make samples/fly/interactive/run ARGS="--connectome /path/to/resources/connectome \
                                       --body /path/to/flybody/floor.xml"
```

## What it demonstrates

* **A connectome is a model brain runs.** 23,188 neurons and 5.24 million
  synapses of MANC ventral nerve cord, stepped as a spiking network on the GPU,
  every control tick.
* **The body is driven through the animal's own motor neurons.** The map from
  neuron to actuator is built from the connectome's published `Sub Class`
  annotations, not from a hand-written list - 330 leg motor neurons reach 44
  flybody actuators, and every one that does not is reported with a reason.
* **The keyboard only gets the descending command.** W and S change how hard
  the brain is telling the cord to go; A and D bias the two halves of the
  descending population against each other. Nothing touches a joint directly,
  so a turn has to come out of the cord's own circuitry.
* **The controls are on the surface, not in a test.** `C` (held) lesions the
  proprioceptive channel and `--shuffled-connectome` replaces the wiring with a
  degree-matched shuffle of it. A creature that behaves the same either way was
  not using the thing you switched off, and you should be able to find that out
  without reaching past the SDK.

## What it needs

* **MuJoCo**, found through `$BRAIN_MUJOCO_DIR`, `$MUJOCO_DIR` or the loader's
  own search path. It is loaded at run time; nothing here builds against it.
* **The MANC connectome export**, a directory containing `manc-codex/` with
  `neurons.csv.gz` and `connections_princeton.csv.gz`.
* **The flybody MJCF** - the scene WITH a ground plane. A body file with no
  floor simulates a fly falling forever, and every other reading still looks
  healthy while it does.
* A GPU for the spiking network, and EGL for the renderer. There is no display
  server requirement: rendering goes through an EGL device context, so this
  works over SSH.

## Running with nobody watching

`--frames N` stops after N frames and `--shot out.ppm` writes the last one, so
the whole path - connectome, body, renderer, window blit - is exercised under
`SDL_VIDEODRIVER=dummy` on a machine with no display attached.

```bash
SDL_VIDEODRIVER=dummy make samples/fly/interactive/run \
  ARGS="--connectome ... --body ... --frames 150 --shot fly.ppm"
```

## Driving the camera

Drag to orbit around the fly, right-drag (or middle-drag) to pan, wheel to
zoom. The camera stays locked on the animal wherever it goes, so a pan is an
offset from it rather than a place: look at the ground beside the fly and you
keep looking beside the fly as it walks off.

## If it runs in slow motion

The title bar and the periodic frame line both report the ratio to real time,
and the frame line breaks a frame into the four things that can be slow. The
shape of it, with the numbers standing in for whatever your machine reports:

```
frame 150: 150 presented, 41 ms/frame, 0.81x realtime | cord 13.6 + body 22.1 + loop 0.9 + draw 4.3 ms over 17 ticks
```

* **cord** - the spiking network, and the device it is on. A control tick's
  whole budget is 2.00 ms at 500 Hz, so `cord` divided by the tick count is
  the number to compare against it.
* **body** - MuJoCo integrating 20 physics steps, on one thread.
* **loop** - this sample's own sensing and bookkeeping.
* **draw** - the offscreen render, the pixel readback, and the blit.

**If `body` dominates, it is MuJoCo and `--timestep` is the only real dial.**
The body costs `1/timestep` integrator steps per simulated second at a roughly
fixed cost each, so it does not optimise away. MuJoCo's own `testspeed` on this
scene, one P-core: 0.30x real time at the published 1e-4, 0.62x at 2e-4, 1.5x
at 4e-4. It is a FIDELITY trade, not a free one - the floor's contact time
constant scales with the step, so a coarser step means a softer floor and legs
that sink further into it. Measure gait at the published value; use this to
watch the animal move at its own speed.

**If `cord` dominates, try `BRAIN_DEVICE=cpu`.** The gather over the edge list
is pure streaming - about 11 MB per tick and almost no arithmetic - so it runs
at whatever memory bandwidth the device has. `brain roofline` measures both:
on a Meteor Lake laptop that is 23 GB/s for the integrated GPU against 68 GB/s
for the CPU, and the CPU backend also has no readback to wait on, which makes
it 8x faster there. On a discrete card the GPU wins by the same argument.

```bash
BRAIN_DEVICE=cpu make samples/fly/interactive/run ARGS="..."
```

## Options

| flag | meaning |
|---|---|
| `--connectome DIR` | directory holding `manc-codex/` |
| `--body FILE` | the flybody MJCF, the one with a floor |
| `--drive X` | starting descending command (default 1.5) |
| `--frames N` | stop after N frames |
| `--shot FILE` | write the last frame as a PPM |
| `--shuffled-connectome` | the structural control: same degrees, shuffled wiring |
| `--plastic` | let synapses change while it runs |
| `--brain` | join BANC's brain to the cord and run the whole animal |
| `--smell` | put the food's odour on the antennae; implies `--brain` |
| `--timestep X` | integrate the body at X seconds instead of the published 1e-4 |

## Two ways to send it after the food

`--seek` is the MISSING BRAIN, written by hand: it reads the food's true
position out of the simulator and pushes a turn into the descending
population. It works, and it is not the animal doing it.

`--smell` is the animal. With `--brain` the creature has BANC's 3,007
olfactory receptor neurons, and this puts a diffusive plume on them at each
antenna; what happens next is whatever the published wiring does with it.
Measured, the chain carries end to end and is lateralised - a smell on the
left moves the descending population differently from one on the right - and
nothing has trained it, so the animal does not yet steer towards the source.
The status line shows both concentrations and the receptor spikes they
produced, because a sensory channel that is connected and silent looks exactly
like a working one from every other reading.

## What it does NOT show

It does not walk. The connectome supplies structure, not synaptic strength,
time constants or intrinsic excitability, and a fly's gait is not in the
wiring diagram alone. What this sample demonstrates is that the loop is real
and closed: spikes become torque, the body moves, and the body is read back
into the same graph.

## SDK surfaces used

`creature` - `brain::Creature`, `brain::View`.
