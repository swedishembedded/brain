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

  Also measured, counter to the obvious guess: a neural step without a readback
  only SUBMITS work (0.068 ms), while step-then-readback costs 0.807 ms. A
  neural tick costs one GPU round trip, not kernel time, which is why
  `neural_per_control` defaults to 1.
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

  **Still open in M5:** the last control - body perturbation followed by
  re-adaptation - and the reward redesign the ceiling's solution points at.
* **M6 - flight.** Wing MNs, power/steering split, wingbeat entrainment.
  **Gate:** 218 Hz entrainment; flight-imitation reward against the recorded
  saccade-evasion trajectories.
* **M7 - serving contract.** `fly::caps`, residency adapter, D-Bus, example,
  per `.agents/rules/serving-contract.md`.

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
