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

## What it does NOT show

It does not walk. The connectome supplies structure, not synaptic strength,
time constants or intrinsic excitability, and a fly's gait is not in the
wiring diagram alone. What this sample demonstrates is that the loop is real
and closed: spikes become torque, the body moves, and the body is read back
into the same graph.

## SDK surfaces used

`creature` - `brain::Creature`, `brain::View`.
