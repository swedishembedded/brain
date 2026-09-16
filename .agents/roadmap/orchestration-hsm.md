# orchestration: HSM + effects for residency/serving - roadmap

Not started. This is a research/design proposal for a future initiative,
recorded here so it survives past this conversation and so the sdk-design
sweep's remaining work (Phase 2's new pipelines, M10-full) can be shaped with
it in mind rather than drifting further from it first. Nothing in this
document is implemented; treat every claim about "target shape" as a
proposal, not a description of current code.

## The proposal, in one paragraph

Structure request lifecycle, model residency, device lifecycle, and training
orchestration as explicit state machines (flat where a component is simple,
hierarchical only where nested states genuinely share behavior - cancellation
handling across every "active" state is the textbook case). A machine's
`handle(event) -> effects` is pure and run-to-completion: it decides, it
never performs I/O, GPU submission, or blocking waits itself. An executor
runs the effects and reports completion back as new events. Every resource
(a model's residency, a device, an in-flight request) has exactly one
authoritative owner deciding its state; workers report facts, they don't
independently mutate policy. Every async operation carries an identity
(`owner_id + owner_generation + operation_id`) so a late completion, a
duplicate, or a stale result from before a device reset can be recognized
and discarded without leaking the resource it carries. The public SDK sits
on top of this as plain typed methods/futures/streams
(`model.generate(...).start()` / `.subscribe()` / `.cancel()` /
`.result()`), never exposing a state identifier or an event type to a
caller. Tensor ops, autograd, and kernel dispatch stay ordinary numerical
code - the effects system's boundary is at the serving/residency layer, not
inside a model's forward pass.

## Fit against brain's actual current architecture

Brain is closer to this shape than a green-field assessment would suggest -
this is evolution, not a competing runtime, which is the right frame for
proposing it here:

- **A single-owner dispatcher already exists.** `crates/residency`'s
  `Executor`/`ResidencyManager` already runs a message-passing core (a
  `Msg::Report` round-trip is how `Executor::residency` reads state back,
  per the stats-assembly wiring in `AGENTS.md`'s Serving stack table) rather
  than exposing shared mutable state to callers directly. This is most of
  "one authoritative owner" already, just not formalized as an explicit
  `enum State` + typed effect list a test can enumerate.
- **Typed request/response already exists.** `capability::Invocation`
  (params + blobs + a `CancelToken`), `Progress`, `Outcome`, and
  `ActionResult` are already the "typed description of work, not a bag of
  callbacks" the proposal asks for. `M5` of the sdk-design sweep
  (`ImagePipeline::generate_with_progress`) is a small, synchronous instance
  of exactly the `run.subscribe()/cancel()` shape point 6 proposes for the
  SDK - evidence the target SDK shape is reachable incrementally, not only
  via a rewrite.
- **A real generation-fencing gap already exists, independent of this
  proposal.** DFlash2's speculative decoding
  (`crates/qwen35/src/{dflash2,int8_gguf_resident}.rs`,
  `gdn_snapshot_xfer`) already has to roll back recurrent state on a
  rejected speculative round - exactly the "late completion vs. current
  attempt" race the proposal's point 4 names - handled ad hoc today, not
  through a named `owner_generation`. `backend_api::hardware`'s
  `BRAIN_GPU_WAIT_S` bound on a wedged submit is the device-lifecycle
  equivalent: today it panics naming the call site rather than fencing
  in-flight operations and recovering. Both are real, pre-existing
  candidates this pattern would actually fix, not hypothetical motivation.
- **M10 (this session) is a small case study of the "no single authoritative
  decision owner" failure mode the proposal is about**, one level down from
  residency: four flux2 call sites each re-derived the same
  variant/license/precision decision independently, and one of the four
  differences that drifted in was a real bug (two sites skipped the
  precision correction a `.gguf` DiT needs). `flux2::build::resolve` (landed
  in this session) is the same "make the decision once, in one place"
  instinct the proposal generalizes to whole resources - useful evidence for
  the pattern, at a scale small enough to have actually been finished in one
  sitting.
- **Multiple owners per resource type is already brain's convention, not a
  gap this proposal introduces.** `AGENTS.md`'s own `weightset` vs
  `residency::EvictionPolicy` split (documented as: eviction scores an
  *unpredictable* future request stream, `weightset` plans a denoise/decode
  loop's *known* future group traversal) is already "cooperating owners with
  explicit protocols" rather than one global machine - the proposal's
  warning against a single global HSM matches an invariant brain already
  holds elsewhere.

## Where it would NOT go

Tensor operations, autograd, kernel dispatch, and a model's forward/backward
stay ordinary numerical code, per the proposal's own table - this matches
`architecture.md`'s existing layering (`model`/`kernels`/`backend-*` sit
below `residency`/`capability`, and nothing in this proposal asks any of
those lower layers to become event-driven). The effects boundary belongs at
the same altitude `residency`/`capability` already occupy, not inside
`crates/model` or `crates/kernels`.

## What this changes about the sdk-design sweep's remaining work

- **Phase 2 (new SDK pipelines)**: design each new pipeline's progress/
  cancellation surface toward the `run.start()/subscribe()/cancel()/result()`
  shape from day one, rather than repeating M5's synchronous
  `generate_with_progress(cancel, on_progress)` pattern a second time. M5's
  shape is not wrong - it is the right SIMPLE default per rule 3's
  progressive disclosure - but a future pipeline's "advanced" tier should
  point at where this is heading, not invent its own third shape.
- **M10-full (the flux2 `build_resolved` unification)**: stays scoped
  exactly as already planned - a pure decision function, no state machine.
  It is a natural PILOT CANDIDATE for a later, narrower experiment (does
  `Flux2Resident`'s hot-cache/instance lifecycle read cleanly as a small
  state machine - loading, resident, evicting - with `resolve`/`build_sized`
  as effects?), but that pilot is separate future work, not a prerequisite
  for finishing M10-full.

## A phased pilot, if this is pursued (not scheduled)

1. Extract ONE narrow slice's scheduling decisions into a deterministic
   core first - a single resident model family's lifecycle (flux2's own
   `Flux2Resident`/`Flux2Instance`, already touched in this session, is a
   plausible first candidate precisely because it is small and already
   partly understood), not the whole `residency` dispatcher at once.
2. Make its loading/transfer/execution/cleanup explicit effects with
   completion events; keep the tokio (`apiserve`) and thread-dispatcher
   (`residency`) concurrency substrates both feeding the same core rather
   than picking one and rewriting the other first.
3. Establish the reservation/lease/generation invariants for that one slice;
   write the race tests the proposal names as highest-value BEFORE
   generalizing - cancellation during loading, eviction during execution, a
   late completion after a simulated device reset.
4. Only after that pilot proves out: decide whether to generalize to the
   rest of `residency`, and whether the SDK's public shape (point 6) changes
   as a result.

Every step above needs its own scoping and sign-off before code changes -
this document exists to make the proposal not get lost, not to authorize
starting it.
