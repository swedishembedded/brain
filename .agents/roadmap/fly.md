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

* **M1 - sparse runtime.** PARTLY LANDED. `crates/neuro` has the CSC
  connectome, the `syn_gather_csc` + `lif_step` kernels, the
  `DynamicalSystem`/`Plastic` seams, and all four gates green on a real P40
  against the 48-thread Cranelift JIT. **Still open in M1:** the eligibility
  trace and neuromodulator kernels, and an implementation of `Plastic` for
  `SpikingNet` - the trait exists, nothing implements it yet.
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
* **M2 - connectome import.** `crates/connectome`: MANC CSV + feather -> CSR/CSC,
  NT prior with confidence, named populations, `brain pull` integration.
  **Gate:** two-way coverage over all 23,188 neurons and 5,243,574 edges -
  every row accounted for or explicitly rejected with a reason; degree
  statistics reproduce the table above.
* **M3 - body.** MuJoCo behind the feature gate; flybody model loaded; the
  MN -> muscle -> actuator map. **Gate:** the fly stands under gravity; passive
  replay of reference kinematics reproduces recorded joint angles. Also
  resolves the `mujoco-rs` link-time question above.
* **M4 - closed loop, no learning.** descending drive -> MNs -> torques ->
  proprioception -> back into the graph. **Gate:** the loop sustains 500 Hz
  in real time, *measured*; a lesioned proprioceptive channel changes the
  trajectory (i.e. the feedback is actually load-bearing, not decorative).
* **M5 - learning to walk.** Three-factor plasticity over 5.24M edges with sign
  as a fitted parameter; the surrogate-gradient ceiling alongside as the
  instrument. **Gate:** the definition-of-done table, via `promote`'s sign test.
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
* **BANC** requires a signed-in session at `codex.flywire.ai`; a human has to
  accept the terms once. MANC v1.0 covers M1-M6 without it.

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
