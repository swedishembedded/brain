# fly - roadmap

Running a real *Drosophila* connectome in brain: real wiring, fetched weights, a
real body, learning to walk and to fly.

This is not a model port. Every other entry in this directory takes a trained
checkpoint and reproduces its forward pass. Here there is no trained checkpoint -
a connectome is a wiring diagram, and the parameters that make it *function* do
not exist in any file. They have to be learned, in a body, against consequences.
That inverts the usual ladder: import and parity come first as usual, but the
thing being gated at the end is a **behaviour**, not a tensor.

## Definition of done

`brain fly walk` fetches the MANC connectome, runs all 23,188 neurons and
5.24M synapses as a spiking network on the GPU, drives the flybody fly in
MuJoCo through its 369 leg motor neurons, and **learns to walk** - sustained
forward velocity with a tripod gait - with all four controls failing:

| condition | required outcome |
|---|---|
| plastic nervous system | learns |
| plasticity disabled | does not learn |
| connectome shuffled at matched degree | does not learn |
| reward shuffled | does not learn |
| weights frozen after training | retains, does not adapt |
| body perturbed (leg damped/inverted) + plasticity on | re-adapts |

Flight is the same statement over the 66 wing motor neurons with the wingbeat
entrained at 218 Hz.

The gate is `crates/promote`'s one-sided paired sign test over matched episode
pairs, not a screenshot and not a mean.

## Decisions taken

1. **Physics is MuJoCo, runtime-linked.** `mujoco-rs` (MIT OR Apache-2.0,
   tracking MuJoCo 3.12) rather than a hand-written articulated-body solver.
   flybody's walking depends on MuJoCo's adhesion model and its flight on
   MuJoCo's fluid-force model; reimplementing both and revalidating them is a
   project of its own, and doing so would also forfeit direct comparability
   against DeepMind's published policies. The crate goes behind a **non-default
   cargo feature and outside `default-members`**, exactly as `crates/vulkan`
   (coopmat) already is, so `make build` and `make test` stay green on a box
   with no MuJoCo installed.
   *Open:* `mujoco-rs` binds a **shared** C library. Whether a `cargo build`
   succeeds with no `libmujoco.so` present (dlopen) or fails at link time
   (`-lmujoco`) decides whether the feature gate alone is sufficient or whether
   the body crate needs its own `runtime-linking` shim like `crates/npu` has
   for OpenVINO. Resolve this in M3 by building it, not by reading about it.

2. **Local plasticity is the deliverable; gradients are an instrument.**
   The point is a digital animal, not a connectome-shaped RNN, so
   three-factor plasticity (eligibility trace x neuromodulator) over the real
   connectome is what has to produce walking. A surrogate-gradient path over
   the same graph is built **as a measuring instrument** - the ceiling that
   says how far the local rule is from the best that structure can do - and is
   never shipped in place of it. Without that ceiling there is no way to tell
   "the local rule is weak" from "this connectome cannot do the task", which
   is the distinction the whole exercise turns on. It is a control, not a
   fallback.

3. **Both data sources, no narrowing.** BANC (brain *and* nerve cord in one
   volume) is the end state and removes the brain/VNC seam; MANC v1.0 is the
   start because it is already public, CC-BY, and login-free. Real recorded
   fly kinematics are used as ground truth **and** connectome-independent gait
   criteria (stability, speed, duty factor, phase relations) are measured
   alongside - imitating a recording and generating a gait are different
   claims and both are wanted.

## The data, measured

Measured directly from the downloaded files, not quoted from a paper. The
resources tree lives outside this repo; its location comes from
`$BRAIN_FLY_DATA` (no default is baked in).

**MANC v1.0 - Male Adult Nerve Cord.** The fly's spinal-cord analogue: the
thing that actually drives legs and wings.

| | |
|---|---|
| neurons / edges / synapses | 23,188 / 5,243,574 / 30,698,527 |
| full CSR footprint | **41.9 MB** (u32 index + f32 weight); 24.6 MB at weight>=2 |
| in-degree | mean 226, median 175, p99 981, max 2464 |
| neurotransmitter predicted | 22,942 / 23,188 = 98.9% (ACh 11,454 / Glu 5,748 / GABA 5,740) |
| motor neurons | 709 - **369 leg**, 66 wing |
| leg MNs with a named muscle | **307 / 369 = 83.2%**, over 19 distinct muscles |
| leg MNs per (segment, side) | fl 65/67, ml 56/58, hl 58/65 |
| descending (brain -> VNC command) | 1,328 |
| sensory / proprioceptive | 6,282 / 1,347 |

The whole sensorimotor loop is 42 MB. That is the single most important number
here: it is small enough to sit resident on any card with the per-edge
plasticity state alongside it (weight + eligibility + index = 12 B/edge = 63 MB).

**Roofline estimate, to be measured not trusted:** one full sweep reads 41.9 MB,
so ~0.12 ms on a 347 GB/s card. Walking (500 Hz control) costs ~12% of one card.
Flight sets the real rate: 5 kHz control, 20 kHz physics, 218 Hz wingbeat - at a
10 kHz neural tick the full-density sweep is ~1.2x real time on one card, so
flight needs the weight>=2 sparsification, a second card, or the event-driven
path. Walking has an 8x margin; flight does not.

**BANC v888 - the whole CNS in one volume. Now on disk.** Brain *and* nerve
cord in a single reconstruction, so the seam between descending command and
executing cord is gone. Fetched by hand through an authenticated Codex session;
it is not scriptable, and the resources tree records the manual steps.

| | |
|---|---|
| neurons / edges / synapses | 158,262 / 3,037,361 / 23,556,214 |
| full CSR footprint | **24.3 MB** - the entire animal, smaller than MANC alone |
| in-degree | mean 19.2, median 8, p99 178, max 2624 (over ALL neurons, isolated ones included) |
| NT predicted | 94.5%, mean confidence 0.759, 48.7% above 0.8 |
| NT **verified** | **65,369 neurons (41%)** |
| motor neurons | 805: 391 leg, 62 wing, 172 abdomen, 49 neck, 25 haltere |
| descending / ascending / sensory | 1,316 / 1,849 / 16,557 |
| annotation coverage | 153,962 / 153,962 edge endpoints |

Three consequences. **Sign gets much better on BANC**: 41% of neurons carry a
VERIFIED neurotransmitter, which is ground truth rather than the
confidence-weighted prior the prediction gives - the fitted-sign design still
holds, but for two fifths of the animal it is fitted from a fixed point rather
than from a guess. **Residency stops being a question**: 24.3 MB means optic
lobe, central brain, cord and motor neurons are all resident at once, and the
per-edge plasticity state alongside it is another 24 MB. And **the importer
needs no Arrow**: Codex exports plain gzipped CSV with one schema shared across
every dataset, so the feather path M2 was going to need does not exist.

**BANC's nerve cord is NOT as densely reconstructed as MANC's, and that
decides which dataset the walking work sits on.** Measured, restricting to
BANC's own VNC-side neurons so the optic lobe cannot skew it: in-degree mean
40.0 / median 26 over VNC-internal edges, 51.9 / 32 counting every incoming
edge, and 110.1 / 80 onto motor neurons specifically. MANC is 224.2 / 173
overall. So BANC's cord is roughly four to five times sparser even on its own
territory, and the gap is not an artefact of the optic lobe.

The consequence, stated plainly so it is not rediscovered: **M3 to M6 sit on
MANC**, which has both the denser cord and the per-neuron muscle targets the
motor map needs. BANC is the substrate for brain-level work and for the
descending interface, and is worth revisiting as it is proofread further. The
seam this dataset was supposed to remove is therefore still there - the
measurement says the unified volume is not yet a replacement for the specialist
one, which is a better thing to know now than at M4.

The Codex and Janelia renderings of MANC cross-validate: 5,305,638 edges /
30,934,610 synapses / 23,665 neurons against 5,243,574 / 30,698,527 / 23,188,
with identical motor-neuron subclass counts. The small excess is a later patch
revision. Codex carries NT confidence inline; Janelia carries the per-neuron
muscle targets Codex omits, so M3's muscle map still reads the Janelia export.

### Sign is a learned parameter, not a given

Neurotransmitter is predicted for 98.9% of neurons, but mean confidence is
0.767 and **only 51% of predictions exceed 0.8**. Sign is also set by the
*postsynaptic receptor*, which EM cannot see at all. So the prediction enters
as a **prior whose strength is its own confidence**, and sign is fitted. A
model that hard-codes sign from `predictedNt` is asserting something the data
does not support, and will fail silently rather than loudly.

### What the connectome does not contain

Synaptic strength (count is a proxy, never a measurement), time constants,
intrinsic excitability, neuropeptide/volume transmission, gap junctions
(innexins are largely invisible to the EM pipeline), the sensory transduction
front end, and - categorically - the scanned animal's learned state. It is the
wiring of one dead fly, not its memories. Everything experience-dependent has
to be acquired in our environment, by our plasticity rule.

## Licence ledger

| artifact | licence | redistributable |
|---|---|---|
| MANC v1.0 (Janelia) | CC-BY 4.0 | yes, with attribution |
| FlyWire connectivity (Zenodo 10676866) | CC-BY 4.0 | yes - but the underlying FAFB **imagery** is CC-BY-**NC**; connectivity only |
| flygym / NeuroMechFly v2 | Apache-2.0 | yes |
| flybody code + body model | Apache-2.0 | yes |
| **flybody datasets (figshare)** | **GPL-3.0+** | **no - copyleft.** Fetch at use time, never vendor |
| flyvis | MIT | yes |
| Drosophila_brain_model (Shiu) | MIT | yes |

The Apache-2.0-code / GPL-3.0+-data split on flybody is the trap: the reference
walking and flight trajectories are *data*, and they are copyleft. They get a
non-redistributable tier in `checkpoint::license`, the same mechanism that
already refuses to publish TimesFM-3 derivatives.

## Architecture

Three new leaf crates plus a body crate. `model::Model` is **not** extended:
it is a static-graph, token-batch, differentiable seam (`ModelConfig::vocab()`,
`set_batch(Batch)`, `forward() -> f32`, ParamStore) whose docs promise every
implementor is gradient-checkable by construction. A creature has no vocab, no
block size, no scalar loss, and state that must survive across calls. brain's
established pattern for a new *kind* of model is a sibling trait unified at the
`capability` layer - `forecast::ForecastModel`, `wm_core::WorldModel`,
`model::serve::PagedDecoder`, `promote::Environment` - and that is what this
follows.

| crate | responsibility |
|---|---|
| `neuro` | the sparse stateful runtime: CSC gather, LIF, eligibility traces, neuromodulator fields, replay/consolidation. Knows nothing about flies |
| `connectome` | structure: CSV/feather -> CSR/CSC, NT prior, neuron metadata, named populations. Knows nothing about dynamics |
| `flybody` | the body seam: MuJoCo behind a feature gate, the MN -> muscle -> actuator map, proprioceptive readout |
| `fly` | the model: composes the three, owns `caps.rs`, the CLI verbs, the curriculum |

Seams:

```rust
pub trait DynamicalSystem {
    fn reset(&mut self, seed: u64);
    fn step(&mut self, dt: Micros) -> StepStats;
    fn drive(&mut self, port: PortId, values: &[f32]);
    fn read(&self, port: PortId, out: &mut [f32]);
    fn snapshot(&self) -> StateRef;
    fn restore(&mut self, s: &StateRef);
}

pub trait Plastic {
    fn set_plasticity(&mut self, on: bool);
    fn modulate(&mut self, field: NeuromodId, delta: f32);
    fn consolidate(&mut self, replay: &Trajectory);
}
```

`set_plasticity(false)` is in the trait rather than in a demo binary because it
is one of the six required controls - it has to be a capability of every
implementor, not something one experiment happens to support.

### The motor interface is constructible

flybody exposes 8 actuated DoF per leg (`coxa_abduct`, `coxa_twist`, `coxa`,
`femur_twist`, `femur`, `tibia`, `tarsus`, `tarsus2`) and 3 per wing
(`yaw`, `roll`, `pitch`); walking action dimension is 59. MANC's named leg
muscles map onto that chain anatomically:

| flybody DoF | MANC muscles (MN counts) |
|---|---|
| coxa / coxa_twist / coxa_abduct | Sternal ant. rotator 12, Sternal post. rotator 24, Pleural remotor/abductor 14, Tergopleural/Pleural promotor 8, Sternal adductor 6, Tergotr. 10, Sternotrochanter 14 |
| femur / femur_twist | Tr flexor 38, Tr extensor 11, Acc. tr flexor 13, Fe reductor 21 |
| tibia | Acc. ti flexor 45, Ti flexor 24, Ti extensor 12 |
| tarsus / tarsus2 | Ta depressor 9, Ta levator 5, ltm 20, ltm1-tibia 12, ltm2-femur 9 |

Flexor/extensor pairs give agonist/antagonist drive per DoF. The 62 leg MNs
carrying only a coarse label (`hind leg` 36, `middle leg` 24, `front leg` 2)
either get resolved from the literature muscle map or are left for the fit to
place - they are named honestly as unassigned either way, never silently
bucketed. Wing MNs cover the full canonical set: power muscles (DLM 10, DVM 14)
and steering muscles (b1/b2/b3, i1/i2, iii1/iii3, hg1-4, ps1/ps2, tp1/tp2).

### Kernels

5-6 new kernels, all small except one. The performance-critical one is
`syn_gather_csc`: one workgroup walks one postsynaptic neuron's contiguous
incoming-edge range, exactly **one** top-level `workgroupBarrier()` (the CPU
JIT splits at one barrier and no more). Thread-per-neuron would be the
coalescing bug `.agents/rules/kernels.md` already documents. Plus `lif_step`,
`elig_trace`, `plastic_update`, `spike_pack`. Compaction reuses `scan_add` /
`scan_block` / `sort_hist` / `sort_scatter`, which exist because the splat
rasterizer is atomic-free for the same reason - **no new scan or sort kernel.**

### On atomics

The gather formulation needs no scatter and therefore no atomics at all, so
nothing here requires amending the no-atomics rule. That rule's stated
rationale (portability) is nonetheless wrong as written, and this is the place
it was found:

* `atomic<u32>`/`atomic<i32>` are **core WGSL, mandatory in WebGPU**, present
  on every GPU this repo targets. Integer atomics are not a portability risk.
* WGSL/WebGPU core has **no `atomic<f32>`** at all, so the fp32 `atomicAdd` one
  would naively reach for is genuinely unavailable portably.
* The real blocker is verifiable in-repo and is stronger than either:
  **`crates/wgsl-cpu` has no atomic lowering whatsoever**, and neither does
  `crates/wgsl-cuda`. An atomic kernel is wgpu/Vulkan-only, which forfeits the
  CPU backend - the `make parity` oracle, and the thing that catches the
  silent-zero-gradient class this repo's own headline warning is about.

If the event-driven path later proves worth it, the amendment arrives as a
**measured** proposal, not an argument from first principles: add
`DeviceCaps::atomics` as a queried correctness gate in the shape of
`workgroup_reductions`, accumulate in **i32 fixed-point** so the atomic path is
bit-identical to the portable one regardless of arrival order (the same
integer-associativity argument `kernels.md` already makes for DP4A), and gate
it against the scan+sort reference at `max|d| == 0`.

## Milestones

* **M1 - sparse runtime. LANDED.** `crates/neuro` has the CSC connectome, five
  kernels (`syn_gather_csc`, `lif_step`, `neuro_trace`, `neuro_elig`,
  `neuro_learn`), the `DynamicalSystem`/`Plastic` seams with three-factor
  plasticity implemented, and 12 gates green on a real P40 against the
  48-thread Cranelift JIT.

  Two of those gates are entries in the definition-of-done control matrix
  rather than ordinary tests - "plasticity disabled does not learn" and "a
  zero neuromodulator changes nothing" - and both assert BIT-EXACT no-ops.
  That is only possible because `eta` and the modulator are premultiplied into
  one scalar before they reach the kernel, so a zero factor is an exact
  identity rather than a small number. Two controls the fly needs are
  therefore already mechanised.
  **Gate:** analytic-LIF closed-form parity for constant input; CPU == GPU on a
  random connectome; bit-exact replay from a restored snapshot; and the gather
  against a host sparse mat-vec. `gradcheck` does not apply to a local rule -
  that is a deliberate, recorded exception, and the substitutes are these four.

  The fourth gate was not in the original plan and is the one that mattered.
  The cross-backend check cannot catch a gather that is wrong the SAME way on
  both backends, and the analytic check runs on an empty connectome where the
  gather contributes nothing, so the gather started with no gate at all. The
  first version of its host oracle was itself vacuous - an unreachable
  threshold meant nothing ever spiked, so it compared zero against zero - and
  it passed with the gather's loop bound deliberately broken. It now asserts a
  real fraction of neurons fired before comparing, and catches both that
  mutation and a truncated partial fold. Mutation-verify a new gate: one that
  has never been seen to fail has not been tested.
* **M2 - connectome import. LANDED.** `crates/connectome` reads the Codex CSV
  schema (hand-rolled RFC 4180, because a quoted annotation field would
  otherwise shift every later column and a neuron would get a transmitter from
  its cell-type cell), aggregates per-neuropil rows into neuron-pair edges,
  carries the NT prior with its confidence and the verified type where there is
  one, selects populations by annotation predicate, and returns a coverage
  report that balances or refuses to return at all. Real-data gate green on
  both datasets with ZERO rejected rows, reproducing every published figure
  from an implementation independent of the one that first measured them.
  Original scope follows.

  the Codex CSV schema ->
  CSR/CSC, NT prior carrying its own confidence (and the verified type where
  BANC has one), named populations from `Super Class`/`Class`/`Nerve`, and
  `brain pull` integration. ONE reader serves BANC and MANC because Codex
  exports one schema; the Janelia MANC export is a second, smaller reader kept
  for the muscle targets Codex omits.
  **Gate:** two-way coverage over every neuron and every edge of whichever
  dataset is loaded - each row accounted for or explicitly rejected with a
  typed reason; the degree and class statistics in this file reproduced by the
  importer's own test, for BOTH datasets, which is also what settles the
  density question above.
* **M3 - body. LANDED except the kinematic replay.** `crates/mujoco` binds
  MuJoCo through its flat state API and validates its own mjModel layout
  against `mj_stateSize` rather than pinning a version - so no feature gate was
  needed after all, `dlopen` keeps it out of the build entirely and absence is
  a runtime skip. `crates/flybody` maps MANC's motor neurons onto flybody's
  actuators from the connectome's own `Sub Class` (`MN-LegNpT2-Ti_flexor`
  carries segment and muscle), so no Arrow dependency and no hand-written
  neuron list. The real fly loads at nq=109 / nv=108 / nu=78 and integrates
  stably; 330 of 396 leg motor neurons attach, the other 66 reported by name.

  **Three findings the walking milestone must handle, each pinned by test and
  each with a different cause:** (a) MANC annotates `Ta_depressor`,
  `Ta_levator` and the coxa promotor on the FRONT legs only, so `tarsus_T2/T3`
  have no motor neuron and `coxa_T2/T3` have a retractor with nothing to
  oppose it - an annotation gap; (b) `tarsus2` is one-way on every leg and that
  is CORRECT, since the long tendon muscle flexes the tarsus and an insect leg
  has no tarsal extensor, so inventing an antagonist would be wrong;
  (c) `femur_twist` has only the femur reductor, and whether that rotation has
  an antagonist or is merely unannotated is not something the map can settle.

  **The polarity question is now split into its dangerous and benign halves,
  and the dangerous one is closed by measurement.** Driving each leg actuator
  alone and reading which generalized coordinate moves shows flybody mirrors
  its own joint axes: positive control moves left and right the same
  anatomical way, agreeing to better than 0.01% on 21 of 24 DoF pairs, worst
  case 0.22% on T3 coxa abduction (the model comes from a real scan, not a
  mirrored idealisation). So one polarity per muscle is correct and no
  per-side flip is needed - had it been otherwise, a uniform convention would
  have driven the left legs forward and the right backward and the fly would
  have circled while every unit test passed.

  What remains is only the ABSOLUTE sense: whether a flexor's contraction is a
  positive or negative displacement in flybody's frame. Getting that backwards
  flips every leg identically, which a learning system can absorb and a gait
  metric will detect. Kinematic replay against the 16,252-snippet walking
  dataset would pin it outright and is worth doing, but it is no longer a
  blocker for M4.

  **The window is done, and the fly can be looked at.** `crates/mujoco` now
  binds `mjv_*`/`mjr_*` and renders offscreen through an EGL device context,
  so there is no display-server requirement and it works over SSH and in CI.
  The four visual structs are opaque OVER-ALLOCATED blobs with a canary tail
  verified after MuJoCo initialises them: their layouts depend on compile-time
  maxima, mirroring them would reintroduce the wrong-offset failure this
  binding avoids for `mjData`, and none of their fields need to be touched.
  The offscreen buffer is resized with `mjr_resizeOffscreen` and the result
  confirmed with `mjr_maxViewport`, because MuJoCo sizes it from the model's
  own `<visual><global offwidth/offheight>` (640x480 by default) and a larger
  viewport would otherwise read undefined pixels for every row past it.

  Two things measured while gating it, both recorded because they are
  surprising: a repeat render of an UNCHANGED state is not bit-identical
  across a thread change - it differs by one byte in 230,400 by one level, so
  the determinism control is bounded rather than exact - and MuJoCo reports
  `GL_INVALID_OPERATION` during `mjr_makeContext` even after the GL error
  queue is drained first, which establishes that the error is raised inside
  its own context creation rather than inherited.

  There is one GL context per process and at most one live `Renderer`, refused
  with a message rather than raced. `crates/fly`'s `watch` example runs the
  closed loop with the renderer hung off the body; measured over 150 frames
  (3 s simulated) at a standing descending command, the loop runs at 0.23x
  real time WITH rendering, the body stays on the ground, and 12% of the image
  changes between the third frame and the hundred-and-fiftieth.
* **M4 - closed loop, no learning. LANDED.** `crates/fly` composes the cord,
  the body and the wiring between them: 1,328 descending neurons as the command
  channel, 330 motor neurons driving 44 actuators, 304 leg proprioceptors
  reading joint state back. Lesioning the sensory channel changes all 109
  generalized coordinates, so the loop is closed rather than decorative.

  **The 500 Hz gate could not be met and should not have been written as one.**
  flybody runs at **2.69x slower than real time** on this hardware, so its own
  walking control rate is unreachable here whatever the nervous system costs;
  asserting it would assert something about the machine. What is brain's to
  answer is whether closing the loop is cheap against the body it closes
  around, and it is: **185 Hz closed against 189 Hz for the body alone**, a 2%
  cost. The connectome is not the bottleneck; MuJoCo is.

  **`BRAIN_FLYBODY_XML` must name a SCENE, not the bare body.** flybody ships
  `fruitfly.xml` with no worldbody geometry at all and `floor.xml` which
  includes it and adds ground. Everything before this ran the bare body, so the
  fly was in free fall: measured, the root reaches z = -70.6 still accelerating
  at -145, against -0.005 at rest with ground. The loop's closure was real
  either way, but a falling fly cannot walk. The M3 gate now asserts the root
  SETTLES rather than merely staying finite, which free fall also does.

  **Three defects found here, the first two the same shape - a measurement that
  looked healthy while the thing it measured was not happening:**

  1. *The sensory channel was connected and silent.* Proprioceptors are
     afferents with almost no incoming synapses, so injected current is the
     only thing that can fire them. At the first sensory gain the peak current
     was 0.45 against a threshold of 1.0: current flowed every tick, no
     proprioceptor ever spiked, and lesioning changed nothing - while the cord
     fired, the body moved, and every other check passed. Swept it; the
     transition sits exactly at threshold. `proprioceptor_spikes()` now exists
     so a test can assert the channel CARRIES something before asserting that
     removing it matters.
  2. *The sign prior was computed and never applied.* `Connectome::csc` holds
     raw synapse counts, which are unsigned. `signed_csc` applies the
     transmitter sign (52.1% excitatory, 47.7% inhibitory, 0.2% unknown
     contributing zero rather than a guess) and a scale, swept the same way: at
     1e-3 the cord is 1.1% active with ZERO motor spikes - alive, and driving
     nothing.

  Also measured here, and **wrong - see M8**: that a neural step without a
  readback only SUBMITS work (0.068 ms) while step-then-readback costs
  0.807 ms, therefore a neural tick costs one GPU round trip rather than
  kernel time. The first half is true and the conclusion does not follow. The
  loop that produced 0.068 ms left 500 submissions queued behind it, so the
  next loop measured paid for them, and the kernel time the round trip was
  supposed to dwarf was in fact most of the 0.807 ms. `neural_per_control`
  still defaults to 1, now for the ordinary reason that a neural tick is not
  free.
* **M5 - learning to walk. APPARATUS BUILT, NOT WALKING.** `crates/fly::learn`
  runs episodes under a control matrix (`Learning`, `Frozen`, `ShuffledReward`)
  with reward as forward displacement and a neuromodulator that is a reward
  PREDICTION ERROR against an EMA baseline, scored by `promote`'s paired sign
  test.

  **What is established:** reward-correlated plasticity changes behaviour
  differently from reward-SHUFFLED plasticity - same modulator distribution,
  destroyed correlation - at p = 0.0107 over 10 episodes against both controls,
  with `Frozen` byte-identical across every episode. That rules out "any
  plasticity produces drift", which is the thing this control matrix exists to
  rule out.

  **What is NOT established, and the numbers say so plainly.** flybody's
  `gravity="0 0 -981"` fixes the units as CENTIMETRES, so the best episode's
  +0.008 is 0.08 mm, about 1/30th of a body length in 0.6 s. A walking fly
  covers one to three body lengths per SECOND. This is drift, roughly 100x
  short of locomotion, and the experiment now prints body lengths so a raw
  figure cannot be read as success. There is also an uncontrolled confound:
  `Learning` ends the run firing 4.5x more than it started while
  `ShuffledReward` ends at a quarter, so the conditions differ in excitability
  and not only in behaviour, and a distance comparison between them is not
  clean. Controlling for activity is the next thing this needs.

  **Two apparatus bugs it found by being run:** `Fly::reset` called
  `DynamicalSystem::reset`, which restores the connectome's ORIGINAL weights -
  so every episode unlearned and all ten were byte-identical, a perfectly
  reproducible failure to learn. And the plasticity clamp was a hardcoded
  +/-0.2 against weights that run to +/-32, squashing the whole connectome to
  the bound on the first update. `SpikingNet::reset_state` and
  `Fly::initial_weight_scale` exist because of those.

  **Structural control added, and it is the most decisive of the four.**
  `Csc::shuffled_sources` reassigns every edge's source while preserving each
  neuron's in-degree and the weight multiset exactly, applied AFTER signing so
  the excitatory/inhibitory split is identical too. Over 12 episodes:
  Learning beats Frozen 11/12 (p = 0.0032), ShuffledReward 11/12 (p = 0.0032)
  and **ShuffledConnectome 12/12 (p = 0.0002)** - shuffling the wiring is worse
  than every other control, so the published structure is doing work. Learning
  is also the only condition that improves within its own run (+0.027 body
  lengths first half to second half); all three controls flatten or degrade.

  The control's honest limitation, recorded rather than omitted: in-degree is
  preserved exactly but out-degree becomes binomial where the real graph is
  heavy-tailed. A shuffle preserving both degree sequences is an edge-swap walk
  and a much more expensive object.

  **The activity confound is NOT resolved, and an earlier note here said it
  was.** Normalising distance by spike count reproduces the sign tests exactly
  (11/12, 11/12, 12/12), but that is arithmetic rather than corroboration:
  Learning's mean distance is positive and every control's is negative, and
  dividing both by a positive spike count cannot reorder them. The normalised
  test is therefore guaranteed to agree and is not independent evidence. What
  the normalised MEANS do show is still worth having - Learning moves +0.006 cm
  per million spikes while the controls move -0.024 to -0.049, so per unit of
  activity the controls go BACKWARDS and "Learning simply fires more" does not
  explain the direction. Actually controlling for activity needs conditions
  matched on firing rate, which this experiment does not yet do.

  **Retention is mechanised and gated.** After training moves ~934k weights,
  freezing holds them bit-identically and two frozen episodes are
  byte-identical in distance and spike count. Mutation-verified: leaving the
  membrane potential out of `reset_state` makes the two episodes diverge and
  the gate fails. Retention is a property of the MECHANISM and is therefore
  testable independently of whether what was learned is any good.

  **The ceiling is measured, and it says the learning rule is not the binding
  constraint.** `GainSearch` hill-climbs eleven per-presynaptic-cell-type gains
  with the wiring fixed - the standard connectome-constrained shape, and a
  stand-in for a surrogate-gradient method because differentiating through
  MuJoCo would need either a differentiable body or a policy-gradient
  estimator, both larger than the question. Converged after 66 evaluations:

  | | speed |
  |---|---|
  | connectome as imported (unit gains) | -0.013 BL/s |
  | local three-factor rule, best episode | 0.105 BL/s |
  | **gain search ceiling** | **0.201 BL/s** |
  | a real walking fly | 1 to 3 BL/s |

  So the local rule reaches about HALF of what a direct search over the same
  structure can find, and that ceiling is itself five to fifteen times short of
  walking. Improving the plasticity cannot get to locomotion from here: the
  binding constraint is the reward, the sensorimotor coupling or the
  parameterisation, not the rule. That is the whole reason this instrument
  exists - without it, "the local rule is weak" and "this setup cannot walk"
  are indistinguishable, and the M5 negative result would have been
  uninterpretable.

  **The shape of the solution it found is the more useful finding.** The search
  drives `motor` to 0.064, `intrinsic_neuron` to 0.130 and `sensory_ascending`
  to 0.000 while amplifying `ascending` to 3.57 and `sensory` to 2.29: it
  SUPPRESSES the cord's own recurrent circuitry and amplifies the sensory drive
  through to the muscles. That is a reflex, not a central pattern generator -
  and it is what a reward of net forward displacement over a short episode
  actually asks for, since a single coordinated lunge scores as well as a gait.
  The next experiment this points at is a reward that only a sustained periodic
  gait can earn.

  **The instrument's own limitation, stated:** eleven gains cannot express
  anything the cell-type partition cannot, so this is a LOWER bound on what the
  full 5.3M-weight space could reach. A per-synapse optimiser might do better,
  and a negative result from a gain search is weaker evidence than one from a
  per-synapse search.

  **The reward was redesigned, faithfully, and the answer got clearer rather
  than better.** flybody's own walking task is DeepMimic imitation, and the
  port initially got two things wrong that were read off its source and fixed:
  the factors MULTIPLY (`flybody/tasks/base.py` returns `np.prod`), not sum,
  and the task ends an episode once the centre of mass is more than 0.33 cm
  from the reference. Both halves of DeepMimic are now here - early
  termination AND reference-state initialisation from a random snippet and
  frame - and the control matrix scores the episode RETURN, since surviving
  longer is how a terminating episode earns more.

  Over 120 episodes per condition with random starts:

  | condition | mean return (of a perfect 6000) | first half to second |
  |---|---|---|
  | Learning | 49.3 | -14.6 |
  | Frozen | 39.6 | -12.1 |
  | ShuffledReward | 54.8 | -12.9 |
  | ShuffledConnectome | 49.8 | -15.5 |

  Learning beats Frozen 119/120 (p < 0.0001) and does NOT beat ShuffledReward
  (57/120, p = 0.74) or ShuffledConnectome (7/26). Nothing improves within its
  own run. **This is a negative result and it is reported as one:** the effect
  of plasticity here is not reward-correlated.

  One detail in it was the thread worth pulling. Learning and
  ShuffledConnectome returned IDENTICAL scores in 94 of 120 episodes while
  firing different numbers of spikes - which is what happens when a score is
  determined by something other than the animal.

  **So the reward was measured directly, and it pays a corpse.** With the start
  fixed and the descending command swept from silence to saturation, plus a
  PARALYSED control with every muscle severed:

  | condition | return | spikes |
  |---|---|---|
  | drive 0.0 to 1.0 (below threshold) | 31.79 | 415 |
  | drive 2.0 | 25.06 | 48,097 |
  | drive 3.0 | 22.85 | 69,671 |
  | drive 4.0 | 20.72 | 84,835 |
  | **PARALYSED** | **31.36** | 48,592 |

  A severed body scores **98.7%** of the best driven score, and driving the
  animal harder makes the score monotonically WORSE. This is not a bug in the
  reward - it is correct behaviour for an imitation reward applied to a body
  that cannot track: the reference walks away whatever happens, so any motion
  the creature makes is pure velocity error and the optimum within the
  reachable set is to hold still.

  **The ceiling instrument, re-run under imitation, agrees but is not flat.** A
  direct search over the eleven cell-type gains reaches a return of 50.0
  against 25.1 at unit gains and ~31 for a corpse - so there IS a reachable
  margin above immobility, but the entire dynamic range between a corpse and
  the best direct search is about 31 to 50 out of a perfect 4000. The local
  rule's own mean (49.3) sits at roughly that ceiling, which says again that
  the rule is not the binding constraint.

  **What all of this points at, and the measurement that would settle it:**
  every negative result so far shares an untested assumption - that this body,
  driven through this actuation path, is capable of locomotion at all.
  `examples/scripted_gait` tests exactly that with NO connectome involved: a
  hand-written alternating-tripod gait writes the actuators directly and sweeps
  frequency, swing and lift. If a scripted gait walks, the limit is in the cord
  or the coupling; if it does not, no controller was ever going to make this
  body walk and the actuation path is the thing to fix.

  **The body perturbation control is now mechanised.**
  `Fly::set_muscle_strength` scales one actuator's drive, and it is
  deliberately PERIPHERAL - the same spikes reach the same actuator and less
  force comes out, which is what a damaged muscle is. Perturbing a coupling
  constant or a motor-map polarity instead would perturb the CONTROLLER and
  then call the recovery re-adaptation, which is a different claim wearing this
  one's name. `examples/readapt` runs the three phases, the third of which is
  the control that makes the second mean anything: rewind to exactly the
  weights training ended with, apply the same lesion, and run the same episodes
  frozen - because a body that recovers on its own looks identical to a nervous
  system that re-adapted.
  **Then the walking body was tested WITHOUT a nervous system, and it walks.**
  A hand-written alternating tripod written straight to the actuators - fore-aft
  swing on the coxa, levation on the femur a quarter cycle ahead - reaches
  **+2.67 body lengths per second** at 12 Hz and finishes upright. A real fruit
  fly walks at one to three. The body, the actuators and the ground contact are
  not the limit, and every earlier negative result shared the untested
  assumption that they might be. What the cord is failing to produce is the
  RHYTHM.

  **So rhythm became the thing measured.** `fly::gait` scores a trace on
  stepping frequency, tripod antiphase, and how much of the power sits in that
  oscillation - connectome-independent, no reference recording, and validated
  on synthetic traces where the answer is known exactly. A perfect tripod
  scores above 0.9; the same frequency with all six legs in phase scores below
  0.02, which is what separates the phase measurement from the spectral one.

  **Three modelling corrections then came out of the literature on
  connectome-derived circuits, and the third was decisive.**

  *Neurons have sizes.* A leaky membrane obeys `C dV/dt = -g_L(V - V_rest) + I`
  and both terms scale with area, so input resistance goes as one over the
  area. A uniform threshold made the largest cells hundreds of times more
  excitable than the smallest. The correction was nearly a silent no-op: the
  MANC export HAS a `Surface area (nm^2)` column and leaves it EMPTY on all
  23,665 rows, so the lookup succeeds and every value parses as absent. Volume
  is what is populated, and the import gate now asserts the state of this per
  dataset in both directions.

  *A command goes to one cell type, not to the whole population.* A fly has
  1,328 descending neurons commanding different and sometimes opposing
  behaviours. Driving them together is not "go", it is every command at once.

  *And at the weight scale this crate had used since the loop was built, a
  single-cell-type command DIES before it arrives* - 0.007% of the cord active.
  Every earlier result about learning was measured on a network carrying no
  signal. Rhythm lives between 0.3 and 1.0, at a few percent active, and
  collapses into a fused burst above that. At 1.0 the whole cord reaches
  rhythmicity 0.432 with a POSITIVE tripod term of +0.228, at 3 Hz.

  *And one measurement artifact was reading as a result:* `gait::analyse`
  discarded nothing, so a run's onset sat at the bottom of the spectrum -
  subtracting the mean does not remove a ramp - and the dominant frequency
  pinned to the lowest band edge. The same run that now reads +0.228 previously
  read -0.218.

  **Restricting to one leg neuropil raises the frequency,** which is what the
  loop-delay argument predicts. `Connectome::subgraph` selects by the export's
  own `Top in/out region`; LEGNP_T1 plus the descending population is 5,752
  neurons and 4.8M synapses against a published front-leg model's 4,604 and
  3.8M. Step frequency goes from about 3 Hz on the whole cord to 9.75 Hz on
  T1.

  **And synapses had no time constant, which is the largest remaining thing
  that was wrong.** The gather wrote each tick's current straight over the last,
  so a spike's effect lasted exactly one tick. A recurrent loop through three
  neurons then closes in three ticks, and the only oscillation such a network
  can hold has a period the integration step chose. `LifParams` now carries
  `dt_over_tau_syn` and `dt_over_tau_inh` - SEPARATELY, because a
  reciprocal-inhibition oscillator's period is set by how long the inhibition
  takes to build and release relative to the excitation that provoked it, and
  equal time constants leave the loop no phase lag to turn into a rhythm. The
  default is the instantaneous synapse, so every earlier gate measures what it
  measured before and the same default is the control.

  **With a synaptic time constant, the T1 network produces alternating tripod
  coordination - as a TRANSIENT.** Driven from rest at a 5 ms excitatory
  constant, the two leg triangles reach a phase correlation of 0.94 measured on
  one side of each joint. Replicated at three trace lengths, it decays:

  | window analysed | agonist tripod | net tripod |
  |---|---|---|
  | 1.5 s | 0.941 | +0.495 |
  | 2.5 s | 0.777 | -0.069 |
  | 3.0 s | 0.593 | -0.116 |

  So the alternation is real and it is not sustained: the cord falls into it on
  being driven and drifts out of it, and it never becomes periodic
  (rhythmicity 0.001 to 0.003 throughout). Reporting the 0.94 without the decay
  would be reporting one cell of a sweep as a gait, which is the failure mode
  the rest of this file exists to avoid.

  That corner also sits at 14.6% of the cord active, which is the over-driven
  regime rather than the few-percent one a nerve cord should occupy.

  **What reproduces at the operating point adopted below,** across neighbouring
  cells of two independent sweeps at 1500 ticks: cord activity 1.4 to 6.2%,
  2.4 to 10.6 motor spikes per tick, net tripod +0.18 to +0.26, rhythmicity
  0.33 to 0.35, stepping frequency 2.5 to 6.75 Hz. Positive, repeatable, and
  well short of a fly's 10 to 15 Hz.

  Two instrument corrections were needed to see it. `Fly::leg_opposed` reports
  each joint's agonist and antagonist drive separately, because a cord
  producing a clean rhythm on BOTH sides of a joint in phase moves that joint
  nowhere and reads as no rhythm at all. And rhythmicity was scored on a single
  frequency bin, which penalises a real oscillation for being biological: a
  window of T seconds cannot resolve frequencies closer than 1/T, so it is now
  scored over one resolution element and a measured 0.006 against a noise floor
  of 0.007 is no longer how a genuine rhythm reads.

  **Still open:** the coordination is there and the PERIODICITY is not - the
  legs alternate, aperiodically. That is what the inhibitory time constant was
  added to address and what is being swept now.
* **M6 - flight. THE WINGS WORK.** The published body could not have flown, and
  not for want of tuning: its wing surfaces carry MuJoCo's DEFAULT fluid model,
  which approximates a body by its inertia box and makes almost no lift from a
  thin flapping plate. A wingbeat driven into it moves the wings correctly and
  produces nothing. `flybody::flight_model` generates the flight variant - the
  ellipsoid fluid model with a fly wing's coefficients, a far higher actuator
  gain, more hinge damping, a shorter timestep - as a textual rewrite where
  every substitution is COUNTED, so an upstream reformat is an error naming the
  pattern rather than a fly that flaps and does not fly.

  The wingbeat is GENERATED and the nervous system modulates it, because that
  is the anatomy. Power muscles are stretch-activated and drive a resonant
  thorax at a frequency the thorax sets: a fly beats near 218 Hz while those
  motor neurons fire at tens of hertz, so wiring a wing joint to their spike
  train would produce a wingbeat two orders of magnitude too slow. MANC's own
  annotations carry the split - **24 power motor neurons** across the dorsal
  longitudinal and dorsoventral groups, **28 amplitude** and **4
  angle-of-attack** steering neurons.

  Two implementation facts that are not details. The stroke is written every
  PHYSICS step: at 218 Hz a beat lasts 4.6 ms against a 2 ms control period, so
  writing it per control tick samples it twice and aliases it into a wobble.
  And feathering is `tanh(k cos phi)` rather than a sinusoid, because a real
  wing holds a nearly constant angle of attack through each half-stroke and
  flips it fast at the reversal - which is that shape with one parameter and no
  lookup table.

  **Measured:** stroke 2.4 rad peak, and the descent falls from **100.6 cm/s
  with the stroke off to 21.4 cm/s beating at 180 Hz**, a 79% reduction in sink
  rate. 180 rather than the animal's 218 because this airframe's hinge
  resonates lower than a real thorax, measured by sweep rather than assumed.

  Two things the gate got wrong first and now records. A fly with its wings out
  and STILL does not fall at g - those wings are large aerodynamic surfaces
  whether or not they beat, and passive drag alone halves the descent - so
  comparing against textbook free fall credits the wingbeat with lift the wings
  produce by existing. And the wings are not motionless with flight disabled:
  the sprung hinge rings at about half a radian, asserted as the baseline the
  stroke must clear rather than hidden behind a threshold this model does not
  meet.

  **Gate:** sink rate at least halved against the same body with the stroke
  off, which holds. **Still open:** the remaining sink, steering through the
  amplitude and angle-of-attack motor neurons, and a flight-imitation reward
  against the recorded saccade-evasion trajectories.
* **M7 - serving contract.** `fly::caps`, residency adapter, D-Bus, example,
  per `.agents/rules/serving-contract.md`.

* **M8 - real time. THE CORD IS NO LONGER THE BOTTLENECK; THE BODY IS.**
  Reported from a Meteor Lake laptop (Intel Arc iGPU + 22 CPU threads) where
  the sample ran at **0.05x real time, 754 ms per frame**. Profiled rather
  than guessed - `crates/fly/examples/loop_profile.rs` is the instrument, and
  it had to be fixed first (see M4 above: it measured a backlog).

  **Where the time went: the synaptic gather kernel, at 25.5 ms per neural
  tick against a 2.00 ms budget.** Not the round trip, not MuJoCo, not the
  window. `syn_gather_csc` ran one 64-thread WORK-GROUP per postsynaptic
  neuron, striding the neuron's edge range so the loads coalesce, then folding
  64 partials redundantly in every thread. Both halves of that are right for a
  matrix with long rows and wrong for a connectome: the fly's cord averages 58
  incoming edges per neuron at the synapse floor `Wiring` runs, so the fixed
  per-neuron cost (a barrier, a work-group allocation, 128 work-group-memory
  adds per thread) dwarfed the ~58 multiply-adds it existed to serve. The
  diagnostic that named it: the cost did not fall when the connectome was
  pruned - 24.4 ms over 1.37 M edges against 69.0 ms over 5.31 M, because the
  term that dominates scales with NEURONS.

  One neural tick of the cord at the floor (23,665 neurons / 1.37 M edges):

  | | Arc iGPU (wgpu) | 22-thread CPU JIT |
  |---|---|---|
  | work-group per neuron, redundant fold | 25.5 ms | 27.5 ms |
  | one thread per neuron | 8.9 ms | 1.70 ms |
  | + the sign split as a clamp pair | 6.3 ms | **0.81 ms** |

  **34x on the CPU backend, 4x on the iGPU**, and the CPU number is 2.48x real
  time for the cord alone. The second row is the shape change; the third is
  removing `if (w[k] < 0.0)`, whose condition is the transmitter labels in edge
  order - unpredictable per edge, and the CPU spent as long recovering from it
  as reading the graph. `spike` is never negative, so `max(c, 0)` / `min(c, 0)`
  produces the same two sums. `neuro_elig` had the same 64-invocations-per-
  neuron shape and got the same treatment.

  **The cord belongs on the CPU on this machine, and that is a measurement,
  not a preference.** `brain roofline` says 23 GB/s streaming for the iGPU
  against 68 GB/s for the CPU, and this kernel is pure streaming: 11 MB of
  edge list per tick, ~0 arithmetic intensity. The iGPU also charges ~3 ms per
  spike readback that the CPU backend does not charge at all. So
  `BRAIN_DEVICE=cpu` is the fast path here and the iGPU is better spent on the
  window. (On a discrete card with 347 GB/s the GPU is the right place; this is
  a property of integrated graphics, not of the backend.)

  **Then the window, which was the other half and was hiding behind the
  readback.** With the cord fixed the frame was 613 ms of `draw`, and
  `mjr_readPixels` appeared to be all of it. It was not: a `glFinish` after
  `mjr_render` moved 590 of those milliseconds onto the GPU where they were
  actually spent, and the readback is 7 to 25 ms. What the GPU was doing is
  the published model's `<visual>` block, inherited silently through the
  scene's `<include>`: `shadowsize="8192"` and `offsamples="24"`, figure
  quality, in a window somebody is watching. Shadows cost a flat ~25 ms at any
  map size below 1024 - the cost is re-drawing 272,550 triangles from the
  light, not the map - so the generated scene now states its own
  `flybody::Look`, shadows off and 4x antialiasing, and `Look::still()` hands
  the model's numbers back for a picture. **316 ms to 4.6 ms of GPU draw.**

  **What is left is the body, and it does not optimise away.** Physics cost
  per simulated second is `1/timestep` integrator steps at a roughly fixed
  cost each. MuJoCo's own profiler on this scene, one P-core: 437 us per step,
  42% position (half of that collision), 35% constraint - and its own verdict,
  0.30x real time, which is M4's 2.69x-slower measured again from the other
  side. The engine threadpool (`--nenginethread`) made no measurable
  difference; this model has too few islands. So the dial is the timestep, and
  it is now reachable (`--timestep`, `CreatureBuilder::timestep`): 0.62x at
  2e-4, 1.5x at 4e-4, against a floor that has to soften in step to stay
  solvable. **Measured end to end on the machine that reported 0.05x: 0.55x
  real time at 4e-4, steady over 150 frames, at cord 1.3 + body 2.4 ms per
  control tick and 15 ms of draw per frame.**

  Three things are known and not done:

  1. **Overlap the cord with the body.** They are independent within a tick if
     the motor path may lag one control tick (2 ms), which is a conduction
     delay rather than a fudge. `crates/mujoco` steps on the calling thread
     and the cord's readback blocks it; submitting the neural tick as an
     effect and integrating the body while it runs is this workspace's own
     event/effect model applied to the one loop that does not use it. It also
     needs the gait gates re-run, because it changes the closed loop.
  2. **The readback and the blit**, now the largest part of `draw`: a
     synchronous `mjr_readPixels` of 2 MB, a reallocated RGB buffer, and a
     SOFTWARE SDL renderer (the sample prints `"software" renderer` on this
     machine). A PBO readback one frame behind, or presenting the FBO
     directly, removes the GPU-to-CPU-to-GPU round trip entirely.
  3. **`mjOption` is not bound** - `crates/mujoco/src/sys.rs` binds `mj_step`
     and nothing else - so solver iterations, the friction cone and
     `noslip_iterations` cannot be traded against accuracy from here. The
     model sets an elliptic cone and 3 noslip iterations, and dropping the
     latter measured ~20% on its own.

M1-M4 are engineering. **M5 is the research milestone** and is where the
schedule is honestly uncertain: the published precedents (flyvis for vision,
Shiu for whole-brain LIF) constrain *parameters within fixed structure* by
gradient-based task optimisation - nobody has reached competent locomotion
from a real connectome by a local rule. That is the frontier this is aimed at,
and it is named as a frontier rather than as a delivery date.

## The sample application

The user-facing deliverable is a **sample**, not a CLI verb:
`samples/fly/interactive` - a standalone application that links the public
`brain` SDK and lets a person poke a fly. It follows `samples/README.md`'s
contract like every other sample, which means the SDK has to grow a creature
surface; that is the point, not a side effect.

**LANDED.** The SDK gained `brain::Creature` and `brain::View` behind a
`creature` surface, and `brain_arch::Domain` gained a `Creature` variant to
name it - the SDK's feature vocabulary is gated against that enum, so a
surface that is not a domain is refused, which is what stopped this from
quietly becoming a second vocabulary. Measured: the `creature` surface's
closure is 35 brain crates against 45 for the full SDK, and the sample links
nothing from the image surface.

The three controls are on the PUBLIC surface, not in an experiment binary:
`set_plasticity`, `set_proprioception` and `shuffled_connectome`. A creature
that behaves the same with its sensing lesioned was not using it, and anyone
embedding this should be able to find that out without reaching past the SDK.

The sample runs bounded and headless (`--frames N --shot out.ppm` under
`SDL_VIDEODRIVER=dummy`), which is what makes the whole path - connectome,
body, renderer, window blit - checkable on a machine with no display. Measured
end to end at 960x720: 23,665 neurons loaded in 8.8 s, 330 motor neurons on 44
actuators, ~7,000 spikes and ~90 motor spikes per rendered frame, 0.23x real
time.

Verified while scoping it:

* **MuJoCo 3.12.0 installs cleanly** (Apache-2.0, 32 MB, `libmujoco.so` +
  headers + the full `mjr_*` render API, 28 exported symbols) and ships its own
  `simulate` viewer to check a model against.
* **`mujoco-rs` is not usable here**: 6.x declares `rust-version = 1.95` and
  this toolchain is 1.94; the 5.x line that would build targets MuJoCo 3.9, and
  binding a 3.9 struct layout against a 3.12 library is a silent-corruption
  bug, not a version warning.
* So the binding is **hand-rolled over `dlopen`**, which is this repo's house
  style rather than a workaround: `crates/capture` hand-rolls V4L2 ioctl FFI,
  `crates/wm-display` hand-rolls minimal SDL2 FFI, and `backend-cuda` dlopens
  `libcuda.so.1` specifically so there is no build-time dependency. A dlopened
  `crates/mujoco` builds and tests green on a box with no MuJoCo installed,
  which means it can be a normal default member; absence is a runtime skip
  through `brain_testutil::skip_unavailable`, the same category as "no NPU".
  It pins the ABI with a `mj_version()` check that refuses a mismatch loudly
  instead of reading a struct at the wrong offset.
* **The window already exists.** `crates/wm-display` is a software-blit SDL2
  window with WASD chords, fixed-timestep pacing and headless sinks for CI -
  exactly the shape this sample needs. MuJoCo renders offscreen; the sample
  blits.

Staging, so the sample is runnable long before M5 lands: physics + the real
flybody model first (poke the fly, watch it fall over), then descending drive
from a connectome, then learning.

## Open questions

* **Model naming.** `crates/modelref`'s grammar is `<vendor>/<repo>` matching a
  real HuggingFace repo, with `brain/`, `local/`, `test/` reserved. A
  connectome is not an HF repo. Either a reserved vendor is added for public
  scientific datasets or the connectome is not a `ModelRef` at all and the
  weights that *are* fetched are the learned parameters over it. Decide before
  M2 writes a manifest.
* **Residency of a stateful creature.** Every existing brain model is
  request-scoped. A creature's accumulated learning lives in device state, so
  eviction destroys it and `run_batch` is meaningless. Needs a deliberate
  answer in M7, not a bolt-on.
* **Model naming and stateful residency** remain the two open questions; the
  BANC density question below is now closed by measurement.

## Sources

* MANC v1.0 - Janelia FlyEM, CC-BY 4.0, public bucket `flyem-manc-exports`
* FlyWire FAFB v783 connectivity - Zenodo 10676866, CC-BY 4.0
* flybody - Vaxenburg et al., *Whole-body physics simulation of fruit fly
  locomotion*, Nature 643:1312-1320 (2025); code Apache-2.0, datasets GPL-3.0+
* NeuroMechFly v2 - Wang-Chen et al., Nature Methods (2024); Apache-2.0
* Shiu et al., whole-brain LIF from the FlyWire connectome, Nature (2024)
* Lappalainen et al., connectome-constrained model of the fly visual system,
  Nature (2024)
* Bellec et al., *A solution to the learning dilemma for recurrent networks of
  spiking neurons*, Nat Commun 11:3625 (2020) - e-prop
