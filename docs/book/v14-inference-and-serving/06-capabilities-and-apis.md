<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Capabilities, Providers and APIs

*How does an engine present hundreds of models through a few stable interfaces?*

An engine that serves one model through one interface needs one piece of glue. The second model and the second interface do not add to that; they multiply it. A command line, three web dialects, a message bus for desktop programs and a socket for a local controller are six ways in, and if every model is wired to every way in by hand, an engine of 100 models carries 600 pieces of glue, each of which has to agree with the model about what its parameters are called, which values they take and what comes back. Adding one parameter to one model means editing six places, and forgetting one of them produces no error: the parameter simply does not exist on that surface, and nobody finds out until a user asks why the same model behaves differently from a browser and from a terminal.

<!-- worked: capabilities_and_apis.adapter_counts -->
An engine with M models and S surfaces (a command line, an HTTP dialect, a message bus, ...). Without a shared description, every pair needs its own glue: M x S pieces. With one description per model that every surface reads, there are M descriptions and S readers. The last two columns count the places one new parameter of one model has to be written.

| models M | surfaces S | glue per pair: M x S | descriptions and readers: M + S | new param, glue per pair | new param, one description |
|---:|---:|---:|---:|---:|---:|
| 10 | 3 | 30 | 13 | 3 | 1 |
| 10 | 6 | 60 | 16 | 6 | 1 |
| 100 | 3 | 300 | 103 | 3 | 1 |
| 100 | 6 | 600 | 106 | 6 | 1 |
| 300 | 3 | 900 | 303 | 3 | 1 |
| 300 | 6 | 1,800 | 306 | 6 | 1 |
<!-- /worked -->

The way out is the oldest trick in interface design. Each model describes what it can do once, in a form that a program can read, and each surface is written once as a reader of that description. The 600 pieces become 106: 100 descriptions and 6 readers, and the new parameter is written in one place. The rest of the work is to make that description rich enough that every surface can be generated from it (types, defaults, ranges, binary inputs, which parameters only the serving machine may answer), to give the readers a single place where requests from all of them meet the device, and to make failures come back in a form a client can branch on rather than a sentence it has to parse.

Two ideas from earlier in this volume carry the weight of what follows, and they are worth bringing back first. A person types `Qwen/Qwen3-8B-Q4_K_M+maren`: into which parts does the grammar of a model reference cut it, in which order are the suffixes found, and why can no reference that parses name a directory outside the store (Chapter 1 of this volume, "Loading Models")? Which vendors never cause the network to be touched, and why does that make "is this fetched?" a question a test can answer without a network (the same chapter)? In continuous batching, at what moment does a waiting request join the running batch, and what happens to the slot of a request that finishes while the others are still generating (Chapter 4, "Batching and Scheduling Requests")? And why does a server that batches continuously need to see every request that wants the device, rather than a share of them (the same chapter)? Their answers close the chapter, after the exercises.

## An action that describes itself

Start with what a caller needs to know to use a model without reading its source. It needs the names of the things the model can do, and for each of them the inputs it takes and the outputs it gives. Some inputs are small values: a prompt, a number of steps, a seed, a choice among a few styles. Others are large binary payloads: an image, a mask, a clip of audio. A caller also needs to know which inputs it may leave out and what happens when it does, which numbers are allowed, and whether the result arrives at the end or in pieces.

A **capability** is one such thing a model can do, together with a description of it that a program can read: a name, a summary, a list of typed parameters (each with a type, a default or a mark that it is required, an optional inclusive range, a line of help), a list of named binary inputs and outputs (each with a kind of medium such as image, mask, audio, video, text or bytes), and a flag that says whether the action reports progress as it runs. A model's set of capabilities is its **manifest**. In everyday speech a capability is an ability; here it is the ability plus its contract, and it is the contract that makes it useful to a program.

brain's descriptor is `ActionSpec` in `crates/capability/src/lib.rs`: `ParamSpec` for a scalar parameter with the types `Str`, `Int`, `Float`, `Bool` and `Enum` (one of a fixed set of texts), `BlobSpec` for a binary input or output with its `Media` kind, and `Manifest` for the list of actions under a model's name. Every one of them serializes to JSON (JavaScript Object Notation, the text format of RFC 8259), so a host that has never seen the model can discover it at run time. Here is the action used as the running example of this chapter, a hypothetical image model's `paint`:

<!-- worked: capabilities_and_apis.spec_table -->
The action `paint` of a hypothetical model. Ranges are inclusive. A param marked `from $VAR` is filled from that environment variable of the machine that runs the action when the caller gives none; a host-resolved param is answered by the host in some other way. Input blobs: `init` (image, optional). Output blobs: `image` (image).

| param | type | required | default | range | host | in the served view |
|---|---|---|---|---|---|---|
| `prompt` | str | yes |  |  |  | yes |
| `steps` | int | no | `8` | [1, 1000] |  | yes |
| `guidance` | float | no | `3.5` | [0, 30] |  | yes |
| `style` | enum: photo, sketch | no | `"photo"` |  |  | yes |
| `seed` | int | no |  | [0, ] |  | yes |
| `weights` | str | yes |  |  | from \$BRAIN_TOY_WEIGHTS | no |
| `adapter` | str | no |  |  | host-resolved | no |
<!-- /worked -->

A description is only worth having if every call is held to it, and that is the job of validation. The rule `ActionSpec::validate` applies is short enough to state completely, and the order of its checks matters, because the first check that fails names the refusal:

1. Every param the caller named must be declared; the first undeclared name, in byte order of the names, is refused as unknown.
2. For each declared param in the order of the spec: if the caller gave a value, a text longer than 16,777,216 bytes is refused, then the value's type is checked (an `Int` must be a JSON integer, so `2.0` is not one; a `Float` accepts any JSON number; an `Enum` must be one of its texts), then a number is compared with the inclusive range. If the caller gave no value, the param is filled from the serving machine's environment when it declares a variable and that variable is set to a non-empty text, else from its default, and otherwise a required param is refused as missing.
3. Every declared input blob that the caller gave must have the declared medium, and a required one must be given.
4. Every blob the caller gave must be declared.

The result of a call that passes is the normalized call, with every default written in, so that the action never has to decide what a missing value means. A default is never range-checked: it is the action's own value, and a test over the whole catalog (`every_default_lies_within_its_declared_range` in `crates/catalog/src/lib.rs`) checks once that each one lies inside its range. The rule, as the book's pseudocode, is below. `Value` is a JSON value; `kind(v)` is `"int"` for a JSON integer, `"float"` for any other JSON number, `"bool"`, `"text"` or `"other"`; `number(v)` is a number as a `real`; `json(v)` is the compact JSON text of `v`; `type_ok(p, v)` is the type rule of step 2; and `missing_message(spec, p)` is the refusal of step 2's last case, which for a param with a variable names the variable (the exact texts are in the table that follows).

```text
record ParamSpec:
    name: text
    ty: text                      # "str", "int", "float", "bool" or "enum"
    values: list<text>            # the texts an "enum" allows; empty otherwise
    required: bool
    default: optional Value
    lo: optional real             # inclusive range
    hi: optional real
    host_env: optional text       # the variable the serving machine answers from
    host_resolved: bool           # answered by the serving machine in another way

record BlobSpec:
    name: text
    media: text
    required: bool

record ActionSpec:
    name: text
    params: list<ParamSpec>
    inputs: list<BlobSpec>

function validate(spec: ActionSpec, params: map<text, Value>, blobs: map<text, text>,
                  env: map<text, text>) -> (bool, map<text, Value>, text):
    for k in keys(params):
        if not any([p.name == k for p in spec.params]):
            return (false, params, "unknown param '" + k + "' for action '" + spec.name + "'")
    out = copy(params)
    for p in spec.params:
        if p.name in params:
            v = params[p.name]
            if kind(v) == "text" and len(bytes of v) > 16777216:
                return (false, params, "param '" + p.name + "' is longer than 16777216 bytes")
            if not type_ok(p, v):
                return (false, params, "param '" + p.name + "' must be " + p.ty + " (got " + json(v) + ")")
            if kind(v) == "int" or kind(v) == "float":
                if p.lo is present and number(v) < p.lo:
                    return (false, params, "param '" + p.name + "' must be at least " + p.lo + " (got " + json(v) + ")")
                if p.hi is present and number(v) > p.hi:
                    return (false, params, "param '" + p.name + "' must be at most " + p.hi + " (got " + json(v) + ")")
        else:
            h = get(env, p.host_env, "") if p.host_env is present else ""
            if h != "":
                out[p.name] = h
            else if p.default is present:
                out[p.name] = p.default
            else if p.required:
                return (false, params, missing_message(spec, p))
    for b in spec.inputs:
        if b.name in blobs and blobs[b.name] != b.media:
            return (false, params, "input '" + b.name + "' expects " + b.media + " but got " + blobs[b.name])
        if b.required and not (b.name in blobs):
            return (false, params, "missing required input '" + b.name + "' for action '" + spec.name + "'")
    for k in keys(blobs):
        if not any([b.name == k for b in spec.inputs]):
            return (false, params, "unknown input '" + k + "' for action '" + spec.name + "'")
    return (true, out, "")
```

A bound written into the refusal is printed as the engine prints a 64-bit float, with no fraction when it is whole (`1`, `30`, `0.5`). Ten local calls and four served ones, the second kind explained in the next section:

<!-- worked: capabilities_and_apis.validate_table -->
Each call is validated against `paint` by the local rule (`validate`, for a caller on the machine that holds the weights) or the served rule (`validate_served`, for every caller over a network or a bus). `BRAIN_TOY_WEIGHTS`, when set, is `/models/toy.safetensors`. Params print as JSON with the keys sorted and a space after each comma and colon; the first failing check names the refusal.

| call | rule | params | blobs | variable | result |
|---:|---|---|---|---|---|
| 1 | local | {"prompt": "a cat"} | none | set | accepted: {"guidance": 3.5, "prompt": "a cat", "steps": 8, "style": "photo", "weights": "/models/toy.safetensors"} |
| 2 | local | {"prompt": "a cat"} | none | unset | refused: action 'paint' needs 'weights', which this machine has not configured: set \$BRAIN_TOY_WEIGHTS (or pass 'weights' explicitly) |
| 3 | local | {"prompt": "a cat", "steps": 0} | none | set | refused: param 'steps' must be at least 1 (got 0) |
| 4 | local | {"prompt": "a cat", "steps": 2.0} | none | set | refused: param 'steps' must be int (got 2.0) |
| 5 | local | {"guidance": 30, "prompt": "a cat"} | none | set | accepted: {"guidance": 30, "prompt": "a cat", "steps": 8, "style": "photo", "weights": "/models/toy.safetensors"} |
| 6 | local | {"prompt": "a cat", "style": "oil"} | none | set | refused: param 'style' must be enum (got "oil") |
| 7 | local | {"colour": "red", "prompt": "a cat", "zoom": 2} | none | set | refused: unknown param 'colour' for action 'paint' |
| 8 | local | {} | none | set | refused: missing required param 'prompt' for action 'paint' |
| 9 | local | {"prompt": "a cat"} | `init` (mask) | set | refused: input 'init' expects image but got mask |
| 10 | local | {"prompt": "a cat", "weights": "/etc/passwd"} | none | set | accepted: {"guidance": 3.5, "prompt": "a cat", "steps": 8, "style": "photo", "weights": "/etc/passwd"} |
| 11 | served | {"prompt": "a cat"} | none | set | accepted: {"guidance": 3.5, "prompt": "a cat", "steps": 8, "style": "photo", "weights": "/models/toy.safetensors"} |
| 12 | served | {"prompt": "a cat", "weights": "/etc/passwd"} | none | set | refused: unknown param 'weights' for action 'paint' |
| 13 | served | {"adapter": "/srv/a.safetensors", "prompt": "a cat"} | none | set | refused: unknown param 'adapter' for action 'paint' |
| 14 | served | {"prompt": "a cat"} | none | unset | refused: action 'paint' needs 'weights', which this machine has not configured: set \$BRAIN_TOY_WEIGHTS (or pass 'weights' explicitly) |
<!-- /worked -->

Call 1 is the ordinary case: the caller names a prompt, and the normalized call that the action receives has the steps, the guidance and the style filled from their defaults and the checkpoint filled from the machine's variable. Calls 3 to 6 are the four ways a value can be wrong. A step count of 0 is outside `[1, 1000]`; `2.0` is a number but not an integer, so it is refused as a type error and never reaches the range check; a guidance of 30 is accepted because the range is inclusive; and `oil` is not one of the style's two texts. Call 7 has two unknown names and is refused for `colour`, the first in byte order. Call 8 names nothing and is refused for `prompt`, the first required param in the spec's order. Call 9 sends a mask where an image is declared.

The ceilings themselves are shared. A number a caller sets sizes the work a call does (tokens to generate, pixels to render, denoising steps), and if each model picked its own ceiling for a param of the same meaning, "how much may a caller ask for" would be decided once per model. `crates/capability/src/bounds.rs` holds one ceiling per meaning instead: 262,144 tokens for a generation (`TOKENS`), 8,192 pixels for one side of a rendered image (`SIDE_PX`), 1,000 denoising steps (`DENOISE_STEPS`), 2,048 frames of a clip, and a dozen more; a param that means what another model's means uses the same constant, and `crates/catalog/tests/it/served_bounds.rs` holds every served number to having one. A bound on the size of a request is also a bound on the work the server can be made to do, which is the subject of the next chapter.

## Answers only the serving machine can give

Look again at the `weights` param of `paint`. A checkpoint path is a perfectly good input for someone sitting at the machine that holds the weights. For a caller anywhere else it is a question with no correct answer: the caller does not share the serving machine's filesystem, a scheduler placing work on a machine it has never seen cannot know its directory layout, and a graph editor in a browser has no filesystem at all. Worse, a server that accepts a path from a remote caller lets that caller point the model at any file the server can read.

So a description needs a second fact about each param: whether the caller may set it at all. brain marks a param that only the serving machine can answer in one of two ways. `ParamSpec::host_env("BRAIN_...")` says that the machine publishes the answer in that environment variable, and validation fills it from there when the caller gives none (step 2 of the rule). `ParamSpec::host_resolved()` says the machine answers it some other way, typically by resolving the model store, and validation does not fill it; the provider supplies its own value. Both kinds are removed by a projection, `ActionSpec::for_serving` and `Manifest::for_serving`, which keeps every other param unchanged. The served view of `paint` is the spec with its last two rows gone, and the served check is validation run twice:

```text
function for_serving(spec: ActionSpec) -> ActionSpec:
    kept = []
    for p in spec.params:
        if not (p.host_env is present) and not p.host_resolved:
            append p to kept
    return ActionSpec(spec.name, kept, spec.inputs)

function validate_served(spec: ActionSpec, params: map<text, Value>, blobs: map<text, text>,
                         env: map<text, text>) -> (bool, map<text, Value>, text):
    (ok, out, why) = validate(for_serving(spec), params, blobs, env)
    if not ok:
        return (false, out, why)
    return validate(spec, out, blobs, env)
```

The first pass is against the served view, so a host-only param that a remote caller names is simply unknown: call 12 of the table names `weights` and is refused before anything else happens, and call 13 is refused for `adapter`. The second pass is against the full spec, which is where the host fills in its own answers: call 11 comes out exactly as call 1 did, with the checkpoint taken from the variable. Call 14 shows why the refusal for a missing host answer names the variable and not the param: on the served path there is no param the caller could have forgotten, and the message has to say which setting of the machine is missing. Call 10 is the local view of call 12: a caller standing on the serving machine may name a checkpoint, and an explicit value wins over the variable.

The order of the two passes is the point. Validating against the full spec alone would accept call 12; validating against the served view alone would leave call 11 without its checkpoint. Neither pass alone gives the right answers in both cases.

The history of this mechanism is a lesson in where a rule should live. In September 2026 thirteen models took their checkpoint location as an ordinary param, so every surface derived from their manifests advertised a required "path to a checkpoint" field to callers who could never fill it, and commit `edc7e0026` (4 September 2026) replaced the convention with the declaration: `host_env`, the fill in `validate` and the `for_serving` projection, with catalog tests in both directions (no served manifest carries a host-resolved param, and the projection removes nothing else). A month later commit `8bd620c45` (7 October 2026) found what the projection had not covered. Several models still took a file as a plain text param (an adapter to load, a dataset folder to read, a place to write a training result, a still frame to start a clip from), and the bus surface had used neither the projection nor a served check: its message says that D-Bus "had accepted and advertised every host-resolved param, `weights` included". The commit marked those params `host_resolved`, added `validate_served` as the one off-machine check, and made the bus list served manifests and refuse a host-only param in `Run`, `Subscribe` and `Plan`; `served_invocation` in `crates/dbus/src/service.rs` calls it. Its test, `crates/catalog/tests/it/served_paths.rs`, does not trust the declarations: it reads every catalog manifest and fails on any served text param whose name or help says path, file, folder or directory. The general lesson is that two surfaces with two copies of a security check have one check that is wrong.

## A provider, a registry, and a model that must outlive a bad request

A description has to be attached to something that runs it. A **provider** is that something: a loaded model, or a handle that will load it on first use, that hands out its manifest and, given the name of one of its actions, an object that runs that action. brain's two traits are short (`crates/capability/src/lib.rs`): `Provider` has `manifest()` and `action(name)`, and `Action` has `spec()` and `run(invocation, progress)`, where the invocation is the normalized call (params, named blobs, a cancellation flag and the caller's scope) and the result is an `Outcome` of scalar outputs and named blobs, or an error. The word has another meaning in brain's HTTP crate, where `Provider` names the vendor dialect a socket speaks (Anthropic, OpenAI or OpenRouter); in this chapter those are called dialects, and a provider is always a model's.

A `Registry` holds providers and dispatches by the pair (model, action): it finds the provider whose manifest has that model name, asks it for the action, validates the call against the action's spec and runs it. That is the whole of `Registry::run`. The dispatch is a linear scan that builds each provider's manifest to compare its name, so a call costs a pass over the registered models and the allocation of their manifests; that is work per call, not per token, and it holds no shared state.

What the registry does not do is just as important, and it took the project a while to write it down. Commit `96de1a070` (23 September 2026) recorded three promises of a long-lived registry that were true only because nothing had yet tried the other thing. The first is concurrency: `run` takes the registry by shared reference and serializes nothing, so two threads can enter the same action at once. `Send + Sync` on the `Action` trait is therefore a requirement on whoever implements it, and the contract test `the_registry_runs_two_callers_of_one_model_concurrently` (in `crates/capability/tests/contracts.rs`) observes two callers inside one `run` at the same time. In brain a model keeps its weights, caches and device queue in one slot behind a mutex that `run` holds for the whole call (the contract's own words for its models), so concurrent callers of one model are serialized and see each other only as waiting (`a_model_that_guards_its_resident_state_never_interleaves_two_runs`); a model that can serve several sequences at once does it by batching them inside, never by handing out its state.

The second promise is that one bad request must not kill a model. A panic inside a call that holds a standard mutex poisons it, and every later lock of a poisoned mutex fails, so with a plain lock a single request that hits a backend fault leaves that model answering with an error for the life of the process, although its checkpoint is still on disk and the process is otherwise healthy. `lock_resident` answers the poison the only way that is sound for a model: it discards what the slot holds (a panic can leave a device queue half-submitted and weight buffers half-rebound, and reusing that turns an error into wrong answers) and clears the poison, so the next call rebuilds the model through the path it already has for a slot that does not match its request. The pair of tests `a_panicking_request_bricks_a_plainly_locked_resident_for_the_whole_process` and `lock_resident_discards_the_resident_after_a_panic_and_serves_the_next_caller` shows both halves. The same file states the rebuild cost: the first call of a model reads and builds its checkpoint inside `run`, under the model's lock, so every other caller of that model waits behind it.

A provider that builds its model lazily needs one more piece, a cache of the built model keyed by what it was built from. `Hot` in `crates/capability/src/hot.rs` is that cache, written once instead of once per provider: it holds one value and its key, rebuilds when the key changes, and drops the old value before building the new one, so that replacing a model that fills most of a device frees the device first instead of holding two models at once (`the_previous_value_is_dropped_before_the_replacement_is_built`). Its acceptance rule can be a relation instead of equality: a cached model built for a context of 64 tokens serves a request for 32 and is rebuilt for 128 (`an_acceptance_rule_decides_a_hit`).

The third promise is about time, and it is where the chapter's rule about speed matters. A call is bounded by the work it was asked for (`max_new` tokens, `steps` denoising steps) and by nothing else. Cancellation is a flag that the invocation carries; the action polls it between its own steps, never inside one, so a request to stop is honoured at the next step boundary. The cost of stopping is therefore at most one step of work, and it is best counted in that work:

<!-- worked: capabilities_and_apis.cancel_work -->
An action polls its cancellation flag before each of its n steps and once after the last. Each step costs c FLOPs (floating-point operations); a decode step of a model of 8 x 10^9 parameters is counted as 2 FLOPs per parameter, one multiply and one add. The request arrives after w FLOPs of steps (or w bytes of a load) have been done; the steps completed are ceil(w / c), at most n, and the work after the request is the completed steps' work minus w.

| work | per step c | steps n | request at w | steps completed | work after the request | outcome |
|---|---:|---:|---:|---:|---:|---|
| decode, 8 x 10^9 params | 1.6 x 10^10 FLOPs | 512 | 0 FLOPs | 0 | 0 FLOPs | stopped at the next poll |
| decode, 8 x 10^9 params | 1.6 x 10^10 FLOPs | 512 | 10^11 FLOPs | 7 | 1.2 x 10^10 FLOPs | stopped at the next poll |
| decode, 8 x 10^9 params | 1.6 x 10^10 FLOPs | 512 | 10^13 FLOPs | 512 | 0 FLOPs | ran to the end before the request |
| denoise, hypothetical | 4 x 10^12 FLOPs | 28 | 10^13 FLOPs | 3 | 2 x 10^12 FLOPs | stopped at the next poll |
| first call: load a checkpoint | 1.6 x 10^10 bytes read | 512 | 10^9 bytes read | 0 | 1.5 x 10^10 bytes read | the load has no poll; the poll before step 0 stops it |
<!-- /worked -->

A request that arrives after $w$ FLOPs of steps that cost $c$ FLOPs each stops after $\lceil w/c \rceil$ steps, and the work done after it is less than one step, $1.2 \times 10^{10}$ FLOPs in the second row (seven decode steps of $1.6 \times 10^{10}$ FLOPs, against $10^{11}$ FLOPs done when the request came). The longest stretch with no poll is not a step at all: it is the first call, which reads the checkpoint before any step loop exists, so a request that arrives after $10^9$ of $1.6 \times 10^{10}$ bytes waits for the other $1.5 \times 10^{10}$ bytes to be read before the poll at step 0 can see it. What the flag returns is part of the contract too: an action that has given the caller nothing yet answers with the exact text `cancelled`, which brain's own clients map back to a typed cancellation (the SDK's `backend_err_or_cancelled` in `crates/sdk/src/pipeline.rs`), while a generation that has already streamed part of its answer completes with that part and the finish reason `cancelled`, never `stop`, so that a truncated answer is not billed, cached or stored as a complete one. A token rides on every invocation whether or not the action looks at it, and a manifest does not say which actions poll it, so a caller cannot learn from the interface that a particular model runs to its token limit regardless; a gate script (`scripts/gates/check-cancellable-actions.sh`) names those models and keeps the list from growing. A host that needs a bound in time arms the flag from its own timer; a host that must get its thread back whatever happens needs a separate process.

## A catalog that cannot be half-registered

With a description per model and a provider per model, something still has to say which models exist. The simplest answer is a list, and an engine soon has three: the manifests that a discovery command prints, the constructors that the run command looks a model up in, and the registrations that the server schedules. Each surface reads only its own list, so a model added to some of them and not the others is present on some surfaces and absent on others, with no error anywhere:

<!-- worked: capabilities_and_apis.catalog_drift -->
Three hand-kept lists: 1, the manifests a listing command prints; 2, the constructors the run command looks a model up in; 3, the registrations the server schedules. A model reaches a surface only through its list. The last row derives all three from one entry.

| lists the model was added to | listed | runnable by name | served | state |
|---|---|---|---|---|
| none | no | no | no | consistent |
| 1 | yes | no | no | silently partial |
| 2 | no | yes | no | silently partial |
| 1, 2 | yes | yes | no | silently partial |
| 3 | no | no | yes | silently partial |
| 1, 3 | yes | no | yes | silently partial |
| 2, 3 | no | yes | yes | silently partial |
| 1, 2, 3 | yes | yes | yes | consistent |
| one catalog entry | yes | yes | yes | consistent by construction |
<!-- /worked -->

Six of the eight ways of adding a model leave it silently partial. This happened in brain exactly as the table predicts. Commit `5075a67e7` (5 August 2026) records a super-resolution model that had been added with a manifest, a provider, a scheduler adapter and passing tests, and that `brain caps` still did not know, because only the third list had been edited. The commit replaced the three lists with one declaration, and that declaration is the idea this section is about. A **model catalog** is the single list of the models an engine offers, one entry per model, from which every surface derives what it lists, what it can construct by name and what it schedules, so that the three cannot disagree.

The catalog became a library of its own in commit `859427cf0` (25 August 2026), so that a program embedding brain can enumerate and construct every model without going through any transport. `ModelEntry` in `crates/catalog/src/lib.rs` has four fields: `manifest`, a function that builds the static manifest with no weights loaded; `provider`, a function that builds a runnable provider from a resolved set of checkpoint files (an `Assembly`, the resolver's output for the reference grammar and the store of Chapter 1); `spec`, the store architecture the entry's weights resolve through, if it has one; and `resident`, the constructor of the scheduler's adapter, if the server schedules the model through this list. `manifests()` and `provider(name)` are derived from the entries, and `residents` in `crates/catalog/src/serving.rs` reads the `resident` field of the same entries, so the three answers come from one entry and there is no second list in which a model can be forgotten.

A list that everything derives from needs tests over the whole list, and the catalog's are the ones that matter. `every_listed_model_is_constructible_by_name` builds every listed model by name against an empty model store of the test's own and fails if any answer is "unknown model" (the original failure) or a complaint about the internal assembly the caller never passed (a second failure the test once let through, when entries that had moved onto the resolver were handed an empty assembly); a model whose weights are absent may fail, but the test rejects any failure that blames the internal assembly or answers "unknown model". `catalog_ids_are_unique` fails on two entries with one name, since `provider` would then resolve by position. The other tests check the served projection over the real catalog in both directions. For construction to be safe to do for every entry, it has to be cheap, so an entry whose model loads gigabytes when constructed is wrapped in a `LazyProvider` that builds it on the first run and keeps it.

A catalog also makes it possible to test the whole contract with no weights at all. `crates/capability-mock` has two kinds of stand-in. `MockProvider::from_manifest` takes a real model's manifest and answers every one of its actions with deterministic synthetic content of the declared medium: a gradient image, a moving gradient for a clip, a sine tone for audio, an echo of the prompt for text. It ticks three progress steps and polls the cancellation flag before each and after the last, the discipline every real action must follow, so a client can be tested for progress and cancellation against it. Its content comes from a 32-bit seed folded from the model's name, the action's name and the request's seed, so two stand-ins never coincide and the same request always gives the same bytes:

<!-- worked: capabilities_and_apis.mock_seeds -->
The seed of a stand-in action folds the model's name, a `/`, the action's name and the request's `seed` (0 when absent) in 32-bit wrapping arithmetic: h = 0x811C9DC5; for each byte, h = (h xor byte) x 0x01000193; then h = h xor seed (its low 32 bits); h = h xor (h >> 15); h = h x 0x85EBCA6B; h = h xor (h >> 13). The pixel at column x, row y of a stand-in image is ((x + h) mod 256, (y + (h >> 8)) mod 256, (x + y + h) mod 256) / 255, and a stand-in tone has the frequency 220 + (h mod 440) Hz.

| model | action | seed | h | h in hex | pixel (0, 0) | pixel (1, 0) | tone (Hz) |
|---|---|---:|---:|---:|---|---|---:|
| `mock/vision` | `go` | 0 | 3,341,593,138 | 0xc72caa32 | (50, 170, 50) / 255 | (51, 170, 51) / 255 | 598 |
| `mock/vision` | `go` | 1 | 1,293,170,019 | 0x4d143563 | (99, 53, 99) / 255 | (100, 53, 100) / 255 | 559 |
| `mock/vision` | `go` | 2 | 3,143,050,247 | 0xbb572407 | (7, 36, 7) / 255 | (8, 36, 8) / 255 | 227 |
| `mock/vision` | `edit` | 0 | 1,747,600,282 | 0x682a439a | (154, 67, 154) / 255 | (155, 67, 155) / 255 | 582 |
| `real/model` | `text2image` | 0 | 3,819,083,734 | 0xe3a297d6 | (214, 151, 214) / 255 | (215, 151, 215) / 255 | 554 |
<!-- /worked -->

The fold is the 32-bit FNV-1a hash (Fowler, Noll and Vo's byte-at-a-time multiply and exclusive-or) of `model/action`, the seed's low 32 bits mixed in, and a finishing mix (an exclusive-or with the value shifted right 15 bits, a multiplication, an exclusive-or with the value shifted right 13 bits) that spreads the input bits over the result (`fold_seed` in `crates/capability-mock/src/synth.rs`). Changing the request's seed from 0 to 1 moves the first pixel of `mock/vision`'s image from (50, 170, 50)/255 to (99, 53, 99)/255 and its tone from 598 Hz to 559 Hz. A stand-in that only produces something shaped like the real output cannot show that a client handles a real record correctly, which is why the crate's second kind, `Mock`, is scripted: a test states exactly what a call to one (model, action) returns or fails with, optionally only when the prompt contains a text or on the third call, binds the script to real manifests so that a scripted output name or medium the manifest does not declare is refused before anything runs, and asks `verify()` at the end which expectations were never met.

## Dialects read from the shape of a manifest

The most common way a program asks a model for something is over HTTP (the Hypertext Transfer Protocol), and here the engine does not get to choose the shape of the conversation: programs already speak the dialects of a few large vendors. An **HTTP inference API** is the set of routes a server answers over HTTP to run models: a route lists the models, others take a request body in a vendor's shape (a chat's messages, a text to embed, a prompt for an image), run it and answer in the same shape, optionally as a stream of server-sent events (one `data:` line per piece). brain's `crates/apiserve` answers three such dialects, Anthropic's messages, OpenAI's and OpenRouter's, each on its own socket and behind its own key (Anthropic's `x-api-key` header, a bearer token for the others), and one route set of its own, `/v1/capabilities`, `/v1/run` and `/v1/jobs`, that reaches every action of every model in the catalog's own terms.

A dialect route cannot be generated from a manifest the way a command line can, because the dialect's shape is fixed by somebody else. What can be derived is which models each route may offer. `api_caps` in `crates/apiserve/src/catalog.rs` reads the shape of a manifest's actions, nothing written per model, and the rules are narrow on purpose, because each route dispatches one literal action with one fixed set of params. Chat is a streaming action named `generate` that takes `prompt`, `messages` or `text` and emits text; tool calling and image input are chat whose `generate` also declares `tools` or an `image` input. Embeddings is an action named `embed` that takes a `text` param, since the route always sends text and reads the mean vector back. Image generation is an action that emits an image, takes a `prompt` and needs no input blob. Transcription is `transcribe` with an `audio` input and a text output; speech is `speak` with a `text` param and an `audio` output.

<!-- worked: capabilities_and_apis.api_capabilities -->
Nine one-action manifests and what the shape rules read from them. The chat dialect lists a model only for chat; the other dialects list it for any capability.

| model | action | capabilities | listed by the chat-only dialect | listed by the others | most new tokens per request |
|---|---|---|---|---|---|
| `x/chat` | `generate` (streaming): params messages, max_new, tools; inputs image image; outputs text text | chat, tools, vision | yes | yes | 262,144 |
| `x/oneshot` | `generate`: params prompt; inputs none; outputs text text | none | no | no | not stated |
| `x/captioner` | `caption` (streaming): params prompt; inputs image image required; outputs text text | none | no | no | not stated |
| `x/textembed` | `embed`: params text; inputs none; outputs mean bytes | embeddings | no | yes | not stated |
| `x/faceembed` | `embed`: params none; inputs image image required; outputs embedding bytes | none | no | no | not stated |
| `x/painter` | `text2image`: params prompt, steps; inputs init image; outputs image image | images | no | yes | not stated |
| `x/restorer` | `restore`: params strength; inputs image image required; outputs image image | none | no | no | not stated |
| `x/listener` | `transcribe`: params language; inputs audio audio required; outputs text text | transcription | no | yes | not stated |
| `x/speaker` | `speak`: params text, voice; inputs none; outputs audio audio | speech | no | yes | not stated |
<!-- /worked -->

The rows that are refused are the instructive ones. `x/oneshot` has a `generate` that does not stream, and `x/captioner` streams text from a prompt under another name; listing either as a chat model would advertise a route that fails on the first request. `x/faceembed` is named `embed` but takes an image and no text, so `/v1/embeddings` would refuse every call for a missing input. `x/restorer` emits an image but needs one, so it is an editor, not a generator. The last column is a limit read from the manifest too: a model card states the most new tokens per request only when `generate` declares an upper bound on `max_new`, and says nothing rather than guess.

The name in a request is resolved with the same care, against the list of what is served. Each name is tried as written, then through the alias table that maps an old short name to its canonical reference (the reserved `brain/` vendor of Chapter 1), and, for the routing dialect only, without its first path segment, since a routing client writes `vendor/model`; the routing dialect also tries each name of a `models` fallback list in order.

<!-- worked: capabilities_and_apis.model_candidates -->
The served models are `brain/mock`, `brain/qwen3` and `Qwen/Qwen3-0.6B`; the alias table maps `mock` to `brain/mock` and `qwen` to `brain/qwen3`. Each name a request gives is tried as written, then its alias, and, for the routing dialect only, without its first path segment; the first candidate that is served wins.

| dialect | request body | candidates in order | answers |
|---|---|---|---|
| anthropic | {"model": "brain/mock"} | `brain/mock` | `brain/mock` |
| openai | {"model": "qwen"} | `qwen`, `brain/qwen3` | `brain/qwen3` |
| openai | {"model": "openrouter/brain/mock"} | `openrouter/brain/mock` | none: model not found |
| openrouter | {"model": "openrouter/brain/mock"} | `openrouter/brain/mock`, `brain/mock` | `brain/mock` |
| openrouter | {"model": "x/unknown", "models": ["qwen", "brain/mock"]} | `x/unknown`, `unknown`, `qwen`, `brain/qwen3`, `brain/mock`, `mock` | `brain/qwen3` |
| openrouter | {"model": "Qwen/Qwen3-0.6B"} | `Qwen/Qwen3-0.6B`, `Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |
<!-- /worked -->

The fifth row is the fallback list at work: `x/unknown` and `unknown` are not served, `qwen` is not served under that name, and its alias `brain/qwen3` is, so the request is answered by `brain/qwen3` even though `brain/mock` appears later in the list.

What happens after the request has a model matters more than how it was found, and it is the same for every route. A handler never runs a model. It builds an `Invocation` (for a chat, the messages as one JSON text param, `max_new`, the sampling params, `tools`), submits a job to the one shared executor of the residency and scheduling chapters, and waits for the reply or streams the progress (`submit` and `stream` in `crates/apiserve/src/bridge.rs`). The D-Bus surface below does the same. One executor for every surface is what makes continuous batching work across them: a chat request that arrived over HTTP and one that arrived over the bus wait in the same queue for the same model and join the same running batch at its next step, which they could not do if each surface ran models on its own. Two refusals happen before a request reaches the executor's queue and are kept apart because they mean different things. A request beyond the 256 requests the edge admits at once (`EDGE_CONCURRENCY` in `crates/residency/src/admission.rs`) is shed at once with status 503, "saturated, rejected at the edge"; a request that was accepted but could not be given a lane before its admission deadline is shed with 429. A client that drops the connection cancels its generation through a guard that rides in the response, instead of leaving it to run to its token limit for nobody. Bodies are bounded before any handler buffers them: 8,388,608 bytes for a dialect's chat, embeddings and image routes, and 67,108,864 bytes for `/v1/run` and for the video routes, which carry media inline as base64.

A dialect handler for chat, completions, images and transcription builds its invocation from its dialect's own fields, so a caller cannot name a param the route does not set; the embeddings, speech and video routes also pass theirs through the served check. The generic routes and the bus check every param a caller names with `validate_served`. The adapter that wraps a weight-free provider for the executor validates each job once more, against the manifest the model is served under, before the action runs (`ProviderInstance::run` in `crates/residency/src/bridge.rs`), so a param that manifest omits is unreachable even from a hand-made invocation; models that hold weights have adapters of their own. The whole arrangement fits one picture.

![The serving surfaces of the engine: the command line and the event socket check a call against an action's full spec; the generic HTTP routes and the bus check it against the served view first; the dialect routes build an invocation from their own fields; served calls go through the one executor; every path ends at the provider of one catalog entry.](img/one-description.svg)

The command line and the event socket, both used by callers on the serving machine, check a call against the full spec and run the provider directly. Everything that arrives from elsewhere reaches the provider only through the executor, which is why the scheduling of Chapter 4 sees all of it. The typed library of the last section is drawn beside the layer because it does not pass through it.

The generic routes are what let a model that no dialect has a shape for be served at all. `GET /v1/capabilities` lists every action of every model the caller may see, in its served view; `POST /v1/run` takes `{"model", "action", "params", "blobs"}`, validates it with `validate_served` and submits it through the same bridge, so a dialect's admission and cancellation apply unchanged; `POST /v1/jobs` starts one in the background for work too long to hold a request open, and the job is looked up by the caller's scope and its id, so another caller asking for an id that exists is told there is no such job.

## A failure a client can branch on

An action's error in brain is a text, and that is a weak contract. A client cannot tell "the weights are not on this machine" from "the request was malformed" from "the device ran out of memory" without matching on prose, and prose changes. Worse, an error text written for the operator routinely carries what a remote caller must not read: a weight path, a panic message with its source location, the text of an operating-system error.

brain's answer keeps the text and puts a class in front of it. `tag(class, message)` in `crates/capability/src/failure.rs` writes `[<slug>] message` with one of seven slugs, `not-installed`, `failed-to-load`, `unsupported-capability`, `timeout`, `invalid-request`, `backend-failure` and `unavailable`; an action that tags promises that its message names no path and no internals. `classify` reads the tag back, also after any `": "`, because a layer that passes an error on may prefix its own context. An error with no tag is sorted by a fixed sequence of rules, and the sequence decides both what the caller is told and what only the log sees:

```text
function classify(e: text) -> optional (text, text):
    rest = e
    loop:
        if starts_with(rest, "["):
            close = find(rest, "]")
            if close >= 0 and rest[1:close] in SLUGS:
                m = rest[close + 1:len(rest)]
                return (rest[1:close], m[1:len(m)] if starts_with(m, " ") else m)
        at = find(rest, ": ")
        if at < 0:
            return nothing
        rest = rest[at + 2:len(rest)]

function sort_reply(e: text) -> text:
    if classify(e) is present:                       return "classified"
    if contains(e, ": panicked while running: "):    return "faulted"
    if e == "cancelled" or ends_with(e, ": cancelled"): return "cancelled"
    if starts_with(e, "no model") or contains(e, "no action"): return "no model"
    if ends_with(e, "has no fill-in-the-middle tokens"): return "no fill-in-the-middle"
    if capacity_counts(e) is present:                return "capacity"
    return "failed"
```

`SLUGS` is the list of the seven slugs; `starts_with`, `ends_with`, `contains` and `find` are the text operations defined with this chapter's routines in the pseudocode reference; and `capacity_counts(e)` finds the two whole numbers of the one rejection whose text is safe to show, `needs <n> tokens, engine capacity is <m>`. The HTTP status of a classified failure follows the meaning of each status in HTTP's own specification (RFC 9110): `not-installed` is 404 Not Found, `failed-to-load` and `backend-failure` are 500, `unsupported-capability` and `invalid-request` are 422 Unprocessable Content (the request was understood and cannot be carried out), `timeout` is 504 and `unavailable` is 503 Service Unavailable, the status RFC 9110 gives to a temporary overload; both of those last two are sent with a `Retry-After` header. The admission refusal of the previous section is 429 Too Many Requests, from RFC 6585.

<!-- worked: capabilities_and_apis.failure_replies -->
What one action error becomes on each surface, for a request to the model named in the second column of each error (`x/z` for the sixth, `x/y` otherwise). The rules are tried in the order of the chapter; an error that hides detail is written to the server's log and never sent.

| error text the action returned | sorted as | message to the caller | detail logged only | HTTP status | bus reply |
|---|---|---|---|---:|---|
| [not-installed] weights for 'x/y' are not installed: pull them first | classified not-installed | weights for 'x/y' are not installed: pull them first | no | 404 | [not-installed] weights for 'x/y' are not installed: pull them first |
| activate x/y/default: [failed-to-load] the checkpoint is corrupt | classified failed-to-load | the checkpoint is corrupt | no | 500 | [failed-to-load] the checkpoint is corrupt |
| [timeout] | classified timeout | (empty) | no | 504 | [timeout] |
| x/y: panicked while running: index out of bounds at src/decode.rs:10 | faulted | the model faulted while running the request | yes | 500 | the model faulted while running the request |
| cancelled | cancelled | the request was cancelled | no | 503 | cancelled |
| no model 'x/z' | no model | no model 'x/z' | no | 404 | no model 'x/z' |
| qwen: needs 14995 tokens, engine capacity is 2048 | capacity | the request needs 14995 tokens but this model's serving capacity is 2048 tokens; reduce the prompt or lower max_tokens | yes | 400 | the request needs 14995 tokens but this model's serving capacity is 2048 tokens; reduce the prompt or lower max_tokens |
| backend error at /srv/m/w.gguf: kernel failed | failed | the model failed to process the request | yes | 400 | the model failed to process the request |
| clip says [timeout] in prose | failed | the model failed to process the request | yes | 400 | the model failed to process the request |
<!-- /worked -->

The second row shows why the tag is looked for after a prefix: the scheduler wrote `activate x/y/default:` in front of the model's own tagged error, and the class survives. The fourth is a panic that a lane of the scheduler isolated; its text has a source location in it, so the caller is told only that the model faulted, with status 500 as a backend failure, and the text goes to the log. The seventh is the capacity rejection, the one untagged error whose numbers are safe to show; it becomes a 400 that says how many tokens the request needs and how many the model can serve. The eighth is an error from a model that did not tag it: its path stays in the log, and the caller receives a generic sentence with status 400 and the code `invalid_request`. The last row is the reason `classify` looks only at the start of the text or right after `": "`: a bracketed word in the middle of prose is not a tag. The bus surface answers from the same sorting (`reply_message` in `crates/dbus/src/service.rs`), re-tagging a classified failure so that a bus client can branch on the class the way an HTTP client branches on the status. Six classes arrived in commit `a865395c3` (13 October 2026), whose message states the convention, and the statuses later that day in commit `e450da745`; the seventh, `unavailable`, came in commit `6207efc70` (6 November 2026). All were additive, so every action that does not tag keeps working with the generic answer.

## Lines, sockets and file descriptors between processes

Not every caller is on another machine. A desktop program, a supervisor that starts the engine as a child, or a test harness lives on the same machine as the server and wants something lighter than a web stack, with less to set up, and bound to the machine's own idea of who may call. An **IPC interface** (inter-process communication) is the protocol by which processes on one machine exchange requests and results: the messages, how they are framed on a stream, how a reply is matched to its request, and how large payloads travel. brain has two.

The first is a stream of events, one JSON object per line, with a tag naming the kind of event (`crates/events`). A handful of generic events carry the capability interface itself: `manifest_request` asks for every manifest, `action_request` names a model, an action, params and base64 blobs, and the answer is zero or more `action_progress` lines and one `action_result`, or an `error`. A client may stamp a request with a `req_id`, and every event the server emits while handling it carries the same id, so several requests in flight on one stream can be told apart; a line without one behaves as it always did. The lines are exact:

<!-- worked: capabilities_and_apis.event_lines -->
Each event is one line of JSON; keys are written in byte order, because the maps that build the lines are ordered maps. The first table prints each line with a space after each comma and colon so that it can break; the wire has none. `aGloaWhp` is the base64 of the 6 bytes `hihihi`. The second table decodes lines: a missing field takes its default (the empty text, 0, false), and an unknown or missing tag is an error.

| what | line, spaced |
|---|---|
| a request to run an action, correlated | {"action": "echo", "blobs": [], "event": "action_request", "model": "brain/demo", "params": {"text": "hi", "times": 3}, "req_id": "r1"} |
| its progress | {"event": "action_progress", "message": "echoing", "req_id": "r1", "step": 1, "total": 1} |
| its result | {"blobs": [{"b64": "aGloaWhp", "media": "text", "meta": null, "name": "result"}], "event": "action_result", "outputs": {"chars": 6}, "req_id": "r1"} |
| a text chunk | {"done": false, "event": "brain_text_chunk", "seq": 0, "text": "Hel"} |
| the last text chunk, with its seed | {"done": true, "event": "brain_text_chunk", "seed": 42, "seq": 1, "text": "lo"} |
| cancel the operation in flight | {"event": "cancel"} |
| a line over the limit | {"code": "line_too_long", "event": "error", "message": "line exceeds 67108864 bytes", "retryable": false} |
| a connection over the cap | {"code": "server_busy", "event": "error", "message": "connection limit reached", "retryable": true} |

| line received | decoded as |
|---|---|
| `{"event":"brain_text_chunk","text":"a"}` | text chunk: text `a`, seq 0, done false, no seed |
| `{"event":"user_text"}` | user text: the empty text |
| `{"event":"cancel","req_id":7}` | cancel, with no correlation id (a non-text `req_id` reads as absent) |
| `{"text":"x"}` | error: missing "event" tag |
| `{"event":"nope"}` | error: unknown event tag: "nope" |
<!-- /worked -->

Decoding is lenient where it is safe and strict where it is not: a missing field takes its default, but a line with no tag or an unknown tag is an error, never a guess. The `brain/demo` model of the first rows is the catalog's weight-free example (`crates/catalog/src/demo.rs`), whose `echo` repeats a text, and the runtime's own tests send an `action_request` of this form to a demo action and read back its result (`crates/runtime/src/lib.rs`).

The server for these lines is the `crates/server` crate, and its design fits in a sentence: one protocol, two transports. A `Session` turns one inbound line into response lines; `pump_connection` greets, then reads lines and writes responses until the stream ends, flushing after every line so that a streamed answer reaches the peer as it is produced; `serve_stdio` runs one session over standard input and output, and `serve_unix` runs one per connection on a Unix domain socket. Because both carry identical frames, a client's demultiplexing does not depend on the transport, and `crates/server/tests/it/transport_parity.rs` proves it by producing the same frames for one forecast both ways. Every bound is explicit. A line is read through a counter that stops at 67,108,864 bytes, so a peer that never sends a newline is refused with `line_too_long` and disconnected instead of growing the server's memory; a panic inside a session is caught, answered with an `internal` error and ends that connection only, since the session's state can no longer be trusted; at most 64 connections are served at once, and the 65th is told `server_busy` (retryable) and closed. Commit `73314545c` (6 November 2026) set the properties that make it a local interface: the socket file is made readable and writable by its owner only (mode 0600) before the first accept (pinned by `the_socket_file_admits_only_its_owner` in `crates/server/tests/it/socket_hardening.rs`), the line limit above was added, and the transport over TCP was removed altogether, because the protocol carries no credential and a camera frame may name a file that the server will read. A protocol with no authentication is safe only where the operating system authenticates the caller, and on a Unix socket of mode 0600 it does.

The second is D-Bus, the message bus of Linux desktops (`crates/dbus`). The server takes a well-known name on the bus, `com.swedishembedded.Brain1`, and serves a `Manager` object with methods that mirror the generic routes: `Manifests` returns the served manifests as JSON, `Run` runs one action and returns its result, `Subscribe` starts a streaming run and returns a stream to read its events from, `Cancel` stops a job by id. A bus caller is not the serving process, so `Run` and `Subscribe` check a call with `validate_served` (through `served_invocation`), and a training action runs only on a server started with `--training`; `Plan`, which may be asked about a partial call and so is not validated in full, refuses only a param that the host alone may set. Like the HTTP handlers, a method only validates, builds an `Invocation` and submits a job to the same executor.

What the bus does differently is how it moves bulk data. A bus message is meant for small values, and a picture is not small. So a payload travels as a file descriptor: the sender writes the bytes into an anonymous in-memory file (a memfd), seals it against shrinking, growing and writing so that the receiver can trust its size and contents, and passes the descriptor in the message; the receiver reads it with `pread`, at explicit offsets and never through the descriptor's file position, which the sender shares (only a DMA buffer, which is not a regular file, is mapped instead; Chapter 7 says why). No base64 is involved, and the bus message carries the descriptor's index instead of the payload. How the transports compare is a count of bytes:

<!-- worked: capabilities_and_apis.transport_bytes -->
The raw bytes of a payload and its base64 text, 4 characters for every 3 bytes, the last group padded: 4 x ceil(n / 3). A text transport carries the base64; the bus carries a file descriptor and the receiver reads the raw bytes from it. The limits are 8,388,608 bytes for a dialect's request body, 67,108,864 bytes for one event line and for a `/v1/run` body, and 268,435,456 bytes for one descriptor. 'Fits' compares the payload alone; the JSON around it needs a few bytes more.

| payload | raw bytes | base64 bytes | fits a dialect body | fits an event line or /v1/run | fits one bus descriptor |
|---|---:|---:|---|---|---|
| frame 1920 x 1080, RGB 8-bit | 6,220,800 | 8,294,400 | yes | yes | yes |
| frame 3840 x 2160, RGB 8-bit | 24,883,200 | 33,177,600 | no | yes | yes |
| image 3840 x 2160, RGB 32-bit float | 99,532,800 | 132,710,400 | no | no | yes |
| audio, 160,000 samples (ten seconds at 16 kHz), mono 32-bit float | 640,000 | 853,336 | yes | yes | yes |
| image 8192 x 8192, RGB 32-bit float | 805,306,368 | 1,073,741,824 | no | no | no |
<!-- /worked -->

The base64 text is a third larger than the payload, 132,710,400 bytes for a 3840 x 2160 image of 32-bit floats against its 99,532,800 raw bytes, and that alone decides which transport can carry what. An 8-bit frame of that size fits one event line, which is the case the line limit was sized for; the same image as 32-bit floats, which is the format brain's image blobs use, fits neither an event line nor a `/v1/run` body and does fit one bus descriptor. The bus has bounds of its own, checked before a byte is read: a descriptor larger than 268,435,456 bytes is refused (its size is the caller's choice, and a sparse memfd of many gigabytes costs the caller nothing to make), a call carries at most 16 descriptors, and at most 1,073,741,824 bytes in all. The largest image a caller may ask to have rendered, 8,192 pixels a side, is 805,306,368 bytes as floats and fits none of the three.

## Command lines generated from the manifest

The command line is the surface people meet first, and it is the one most often written by hand. In brain it is generated. `crates/invoke` turns an action's spec into its command line: every param becomes `--<name>`, with a dashed alias for an underscored name; an `Enum` accepts only its texts; a number is parsed as its type and checked against its range while the line is read, a seed never below 0; a `Bool` flag alone means true and also takes `false`, so that a switch whose default is on can be turned off; every input blob becomes `--<name> PATH` or `--in name=PATH`, and every output `--out name=PATH`. A flag that is left out sets nothing, so the action's own default applies when the call is validated, and the manifest stays the only place a default is written.

<!-- worked: capabilities_and_apis.flags_from_spec -->
The action `sketch` declares six params, one required input blob `image` and one output blob `result`. Every param becomes a flag of its own; an input blob becomes `--image PATH` (or `--in image=PATH`) and an output `--out result=PATH`. A flag left out sets nothing, so the action's own default applies when it validates. A refusal is printed as `--flag: reason`, a simplified framing of the longer message the flag-parsing library prints; the reason is the program's own wording for a number out of bounds and for an unknown output, and the refusal of a missing required flag is paraphrased.

| param | flags | value | required on the command line | help line |
|---|---|---|---|---|
| `text` | `--text` | `STR` | yes | the text |
| `times` | `--times` | `INT`: a whole number, at least 0 and at most 1000 | no | repeat count [default: 1] |
| `mode` | `--mode` | `ENUM`: one of as-is, upper, lower | no | casing [default: "as-is"] |
| `held_out` | `--held_out`, `--held-out` | `INT`: a whole number | no | rows held out [default: 4] |
| `preview` | `--preview` | `BOOL`, optional: absent value means true; one of true, false, 1, 0 | no | show a preview [default: false] |
| `seed` | `--seed` | `INT`: a whole number, at least 0 | no | seed |

| flags after `<program> sketch` | what is read | exit code |
|---|---|---:|
| `--text hi --image a.png` | params `{"text":"hi"}`, inputs image from a.png; validated as `{"held_out":4,"mode":"as-is","preview":false,"text":"hi","times":1}` | 0 |
| `--text hi --held-out 2 --in image=b.png --preview` | params `{"held_out":2,"preview":true,"text":"hi"}`, inputs image from b.png; validated as `{"held_out":2,"mode":"as-is","preview":true,"text":"hi","times":1}` | 0 |
| `--text hi --image a.png --preview false` | params `{"preview":false,"text":"hi"}`, inputs image from a.png; validated as `{"held_out":4,"mode":"as-is","preview":false,"text":"hi","times":1}` | 0 |
| `--text hi` | parsed; refused by the action's validation: missing required input 'image' for action 'sketch' | 2 |
| `--image a.png` | refused while parsing: the flag --text is required | 2 |
| `--text hi --image a.png --times 1001` | refused while parsing: --times: 1001 is above the most allowed, 1000 | 2 |
| `--text hi --image a.png --seed -1` | refused while parsing: --seed: -1 is below the least allowed, 0 | 2 |
| `--text hi --image a.png --out picture=o.txt` | refused while parsing: --out "picture=o.txt": the action has no output 'picture' (it has: result) | 2 |
<!-- /worked -->

The fourth run is the instructive one. The input blob is required, but the flag reader does not make `--image` a required flag; the line parses, and the action's validation refuses it for the missing input, with the same exit code. Requiring things in one place, the spec, and enforcing them in one place, the validation, is what keeps the command line and the HTTP route from disagreeing.

The order in which `run_model` (`crates/invoke/src/model.rs`) does its work is chosen so that mistakes are cheap. Help and each action's flags come from the static manifest, so `--help` and a malformed value cost no weight lookup. The flags are checked against the action before the weights are resolved, so a bad value is named as a bad value rather than surfacing as a checkpoint the store could not pick. The weights are resolved next, a `--<role> PATH` flag naming one checkpoint role outright; when two checkpoints of one model tie in the store (a small and a large release side by side), the action's `variant` param decides, the one given or its default, and only a checkpoint of that variant assembles. The provider is built last, and `run_with` validates the call against the action's spec (the step that fills the defaults), loads input files into blobs, runs the action with a progress line, writes each output, and, when the outcome carries the seed the run drew, writes a record beside each output file (`clip.mp4.json` beside `clip.mp4`) with the command, the seed and every param as validated, so that the output can be reproduced from its record alone.

One implementation serves two kinds of program. `brain <architecture> <action>` looks the model up in the catalog and calls `run_model` with the catalog's entry (`run_do` in `crates/cli/src/caps_cli.rs`), and a per-model sample program is a `main` of one line, `invoke::main` over that model's manifest and provider (`samples/models/sam2/src/main.rs` is the whole program). The path there is in the history. The first capability commit, `da5bdf257` (26 July 2026), introduced `brain caps` and `brain do <model> <action>` together with the interface, and its message gives the design in one line: adding a capability is implementing an action and listing it, "no new CLI command, no new event variant". Commit `4c1c9c26e` (13 August 2026) replaced the hand-written dispatch of 26 subcommands with one resolver in which `brain <verb> <architecture>` and `brain <architecture> <verb>` are the same command, routing an architecture without a module of its own through the same `run_do`; `brain do` itself was removed then. Commits `8350c4a47` and `f312abef8` (13 November 2026) moved that machinery out of the binary into `crates/invoke`, so a sample drives a model's actions with no second copy, and the second of them deleted the command line's own table from architecture to store spec, which had drifted from the catalog's (each resolved two models the other did not): the catalog's table is now the only one.

What remains of a command line once the action's own flags are generated is the options every program shares: which device to run on, which backend, which model. `crates/appopts` holds them as groups that a program composes, over a consuming grammar. `Args` keeps the tokens and a mark per token; `take_str(name)` finds the first unused token equal to `name`, marks it and the token after it, and returns that next token as the value; `take_flag` marks a flag alone; `positional` takes the next unused token that does not start with `--`; and `finish` refuses to run while any token is unused.

<!-- worked: capabilities_and_apis.consuming_grammar -->
Each run gives the tokens of a command line and the calls an application makes on them, in order. A call marks what it takes as used; `finish` refuses to run while any token is left unused.

| tokens | calls in order and what each returns | left over | finish |
|---|---|---|---|
| `--steps 7 --device cpu --backend vulkan --window` | take `--device` -> `cpu`; take `--backend` -> `vulkan`; take `--steps` -> `7`; flag `--window` -> present | none | runs |
| `--device cpu --warmup-episodes 16` | take `--device` -> `cpu`; take `--warmup` -> nothing | `--warmup-episodes`, `16` | exit 2: unrecognised: --warmup-episodes 16 |
| `--seed 5 in.png` | take `--seed` -> `5`; positional -> `in.png` | none | runs |
| `--seed 5 in.png` | positional -> `5`; take `--seed` -> `5` | `in.png` | exit 2: unrecognised: in.png |
| `--model --device cpu` | take `--model` -> `--device`; take `--device` -> nothing | `cpu` | exit 2: unrecognised: cpu |
<!-- /worked -->

The second run is the failure the grammar exists for. A program that ignored the unknown `--warmup-episodes` would run, finish and print numbers for a configuration nobody asked for, and commit `8a4d937e2` (18 September 2026) records that this cost a training run before `finish` was made to stop with the flag's name. The last three runs show the grammar's two rules a program must respect. A value is the token after the flag, whatever it is, so `--model --device cpu` makes `--device` the model and leaves `cpu` over, which `finish` then refuses. And because `positional` takes any token that does not start with `--`, a program must take its flags before its positionals, or a flag's value is taken as a positional: the fourth run reads `5` as the input file. A group's `take` and its `help` are paired in one trait (`Options`), so a flag cannot be added without appearing in `--help`, and nothing in the core groups reads the environment: a run is reproducible from its own command line (commit `cdd7f9c43`, 18 September 2026, which moved the command line's parser into the crate rather than copying it).

## A library with no protocol in between

Every surface so far puts a protocol between the caller and the model: text to parse, a request to validate, a job to queue. A program that wants one model inside its own process pays for all of that and gains nothing from it, and it loses the one thing a protocol cannot give: types. A **public SDK** (software development kit) is the library an engine offers for that case: a small set of typed entry points, such as a pipeline that is loaded from a model name and generates an image from a prompt, whose names and signatures the engine promises to keep stable across versions, so that a program can be built against it and keep building.

brain's is `crates/sdk`, the package deliberately named `brain` (the one exception to the workspace's `brain-<short>` naming), begun in commit `c44076117` (15 September 2026) with one entry point:

```text
let model = "black-forest-labs/FLUX.2-klein-9B";
let mut pipe = brain::ImagePipeline::from_pretrained(model)?;
pipe.generate("a whale submarine")?.save("out.png")?;
```

`from_pretrained` takes a model reference of Chapter 1 and resolves it through `crates/loader`, the same resolver the command line uses, never a parallel one; `generate` runs the model crate's own pipeline in the caller's process. There is no capability dispatch in the loop, which is the reason the crate exists: the typed call is the interface. The crate now has a pipeline per surface (text and chat, embeddings, detection and segmentation, speech and transcription, forecasting, video and more), and its Cargo features name surfaces, not infrastructure, so a program that asks for `image` gets the image pipelines' dependencies and nothing else. It still shares two contracts with the capability layer: its long calls take the same `CancelToken` and a progress callback of the same step, total and message form, and it maps the literal `cancelled` back onto its typed `Error::Cancelled`.

A library is also honest about what it is not. `crates/sdk/src/lib.rs` says that the SDK has no admission control, no queue and no concurrency limit of its own, and that each pipeline holds gigabytes of device and host memory for as long as it lives; a service built on it that accepts requests from an untrusted network needs its own admission layer in front of it. Its errors carry the underlying message verbatim, paths included (`crates/sdk/src/error.rs`), which is right for a caller in the same process, who has the filesystem anyway and needs the real reason; a caller that turns those errors into responses for a network must translate them, as `crates/apiserve` and `crates/dbus` do. The structured errors of the resolver, an ambiguous choice with its candidates and a missing role with its near misses, are kept whole instead of being flattened to text, because they are what a caller acts on.

## Where the interfaces are kept in brain

The descriptor, the validation, the provider and action traits, the registry, cancellation, failure classes and the shared ceilings are `crates/capability` (`lib.rs`, `failure.rs`, `bounds.rs`, `hot.rs`), a crate whose only brain dependency is the shared seed rules (`crates/seed`), so that every model crate can describe itself without a cycle. The weight-free stand-ins are `crates/capability-mock`, the catalog is `crates/catalog`, the HTTP surfaces are `crates/apiserve`, the bus surface is `crates/dbus`, and the event protocol and its server are `crates/events` and `crates/server`. The command-line front end of a model's actions is `crates/invoke`, the shared options are `crates/appopts`, the `brain` binary is `crates/cli`, and the typed library is `crates/sdk`. Every one of the serving surfaces submits to the executor of `crates/residency`, which is where the scheduling of Chapter 4 happens.

## Summary

A **capability** is an action with a contract a program can read: typed params with defaults and inclusive ranges, named binary inputs and outputs, and a streaming flag. Validation holds every call to it in a fixed order of checks and fills every default, so an action never interprets a missing value. A param that only the serving machine can answer is marked as such, removed from the served view, and checked by validating twice, first against the served view so that a remote caller cannot name it and then against the full spec so that the host fills it in. A **provider** hands out a model's manifest and its actions; a registry dispatches by model and action, serializes nothing, recovers a model whose call panicked by discarding it, and bounds a call by its work, with cancellation honoured at the next step. A **model catalog** is one entry per model from which listing, construction and scheduling are all derived, so a model cannot be half-registered. An **HTTP inference API** offers each model on the dialect routes its manifest's shape allows, resolves a request's name through aliases, submits every request to the one executor so that continuous batching sees all of them, and maps a tagged failure class to its status while keeping internals in the log. An **IPC interface** carries the same actions between processes of one machine, as bounded lines of JSON on an owner-only socket or as bus calls whose payloads travel as sealed file descriptors. A command line is generated from the manifest, and a **public SDK** gives an embedding program typed calls with no protocol in between.

The limits of these interfaces are part of their contract. A manifest does not say which actions poll cancellation; a first call is not interruptible while it loads; an untagged failure reaches a caller as a generic refusal; and the ceilings bound the work a request can name, which is only the first of the defences a server exposed to strangers needs. Those defences, against hostile requests and hostile model files alike, are the subject of the next chapter.

## Check your understanding

1. A colleague validates a remote call against the full spec only, arguing that "the variable fills `weights` anyway". Using calls 11 and 12 of the validation table, say what goes wrong and what the served check does instead.
2. A spec declares `steps` as `Int` with range `[1, 1000]` and no default. Predict the outcome of a call that gives `steps` as `1000`, as `1000.0`, and not at all.
3. Someone proposes making `MockProvider` stop polling cancellation, since "it finishes in microseconds anyway". What would a client's tests no longer be able to show?
4. Find the error: "Because every route submits to the same executor, a request over the bus and a request over HTTP for the same model are run in the order they arrived, one after the other."
5. A model's `embed` action takes a required `image` blob and a `text` param. Is it listed by `/v1/embeddings`? Would a call to that route succeed?
6. An event client sends one line of 70,000,000 bytes followed by a valid request. What does it receive, and why is the second request not answered?
7. A model returns the error `load x/y: [failed-to-load] weights file /srv/m/x.gguf is truncated`. What does an HTTP caller receive, and what does the operator need to change so the caller is not shown the path?

## References

- Bray, T. (Ed.). The JavaScript Object Notation (JSON) Data Interchange Format. RFC 8259, 2017.
- Fielding, R., Nottingham, M. and Reschke, J. (Eds.). HTTP Semantics. RFC 9110, 2022.
- Nottingham, M. and Fielding, R. Additional HTTP Status Codes. RFC 6585, 2012.

## Exercises

1. **[Calculate]** An engine has 40 models and offers them on a command line, two HTTP dialects, the generic routes and a bus. How many pieces of glue does wiring each pair by hand need, and how many descriptions and readers does one description per model need? How many places must one new param be written in each case?
2. **[Calculate]** A decode step of a model of $3 \times 10^9$ parameters costs 2 FLOPs per parameter. A generation of 1,000 tokens is cancelled after $1.3 \times 10^{11}$ FLOPs of steps. How many steps complete, and how many FLOPs are done after the request?
3. **[Calculate]** Find the base64 length of a 2048 x 2048 RGB image of 32-bit floats and say which of a dialect body, an event line and a bus descriptor can carry it.
4. **[Implement]** Reproduce `capabilities_and_apis.validate_table`: write `validate`, `for_serving` and `validate_served` from the rule and the pseudocode, run the fourteen calls against `paint` and print the same results, refusal texts included.
5. **[Implement]** Reproduce `capabilities_and_apis.failure_replies`: `classify`, the sorting of an untagged error, the client message and the status, for the nine errors printed.
6. **[Implement]** Reproduce `capabilities_and_apis.mock_seeds` in a language with fixed-width integers, and extend it to the pixel at (3, 2) of `mock/vision`'s `go` with seed 1.
7. **[Implement]** Reproduce `capabilities_and_apis.api_capabilities` from the shape rules, then add a manifest whose `generate` streams text from `messages` and declares a `suffix` param, and say which routes list it.
8. **[Investigate]** Read `ActionSpec::validate` in `crates/capability/src/lib.rs` and `check_type`. Which JSON values does an `Int` param accept, and which does a `Float` param accept? Give a value that a `Float` accepts and an `Int` refuses, and one that neither accepts.
9. **[Investigate]** Read `crates/apiserve/src/catalog.rs` and list, for each dialect, the routes a model is offered on and the rule that admits it. Why does the chat rule require the literal name `generate` and not only the shape?
10. **[Investigate]** Read `pump_connection` in `crates/server/src/lib.rs`. Explain why a line exactly 67,108,864 bytes long is served and one byte longer is refused, naming the statement that reads at most one byte more than the limit, and find the two tests that pin the boundary.

## Key terms

- **capability**: one action a model can perform, together with a description a program can read: its name, typed params with defaults and ranges, named binary inputs and outputs, and whether it streams progress. <!-- concept: capability -->
- **provider**: the object that stands for one model behind the capability interface: it hands out the model's manifest and, given an action's name, something that runs that action. <!-- concept: provider -->
- **model catalog**: the single list of the models an engine offers, one entry per model, from which listing, construction by name and scheduling are all derived so that they cannot disagree. <!-- concept: model-catalog -->
- **HTTP inference API**: the routes a server answers over HTTP to run models, in a vendor's request and response shapes or in the engine's own, with models offered on each route according to what their manifests declare. <!-- concept: http-inference-api -->
- **IPC interface**: the protocol by which processes on one machine exchange requests and results with an engine: the messages, their framing, how replies are matched to requests and how large payloads travel. <!-- concept: ipc-interface -->
- **public SDK**: the library an engine offers for use inside another program's process: a small set of typed entry points whose names and signatures are kept stable across versions. <!-- concept: public-sdk -->

## Answers

**Recall.**

1. The grammar splits the adapter suffix off first, `+maren`, which is the local adapter `local:maren:latest`; then it finds the quantization `Q4_K_M` at the end of what is left, leaving vendor `Qwen` and repo `Qwen3-8B`. No reference that parses names a place outside the store because a vendor or repo that is `.` or `..` is refused and a `/` cannot appear inside either, and the store joins the vendor and the repo onto its root verbatim.
2. `brain`, `local` and `test` are reserved, as is a name with no `/`; none of them is ever fetched. Whether a reference is fetched is then a pure function of its text, which a test can call with no network.
3. A waiting request joins at the next step of the running batch, as soon as there is room for it, without waiting for the batch to finish; a request that finishes leaves the batch at the step it finishes, and its slot and its cache are free for a waiting request at once.
4. The batch can only be formed from the requests the scheduler sees. A scheduler that saw only a share of the requests for a device would run smaller batches than the device could hold and could not budget the device's memory for the requests it did not see.

**Check your understanding.**

1. Against the full spec alone, call 12 is accepted with `weights` set to `/etc/passwd`, as call 10 is locally, so a remote caller points the model at a file of the server. The served check validates against the served view first, where `weights` is not declared, and refuses it as an unknown param; only a call that passes is validated against the full spec, where the variable fills `weights`, as in call 11.
2. `1000` is accepted (the range is inclusive). `1000.0` is refused as a type error, since a JSON number with a fraction is not an integer, before the range is checked. Absent, with no default and no variable, it is accepted only if the param is not required; a required one is refused as missing.
3. That a client handles progress and cancellation: the stand-in ticks three steps and polls before each and after the last, so a test can cancel mid-run and expect the exact text `cancelled`. A stand-in that never polled would let a client that ignores cancellation pass its tests.
4. The shared executor batches continuously: requests for the same model join the running batch at its next step, whatever surface they came from; they are not run one after the other in arrival order. (A model that cannot batch serializes its calls behind its lock, but that is the model's limit, not the surface's.)
5. Yes: the rule needs only the name `embed` and a `text` param. A call to the route sends `text` and no blob, so validation refuses it for the missing required input `image`. The rule is a necessary condition for a route to work, not a guarantee.
6. One line `{"code":"line_too_long","event":"error","message":"line exceeds 67108864 bytes","retryable":false}`, and then the connection is closed. The rest of the stream is the middle of the oversized line, so there is no point from which the server could read the next request with certainty.
7. The error is tagged `failed-to-load` after the prefix `load x/y: `, so the caller receives status 500 with the message `weights file /srv/m/x.gguf is truncated`, path included: a tag is the model's promise that its message names no path, and this model broke it. The fix is in the model: it must tag a message without the path and log the path itself.

**Exercises.**

1. Five surfaces: $40 \times 5 = 200$ pieces of glue against $40 + 5 = 45$ descriptions and readers; one new param is written in 5 places against 1.
2. $c = 6 \times 10^9$ FLOPs per step. $\lceil 1.3 \times 10^{11} / 6 \times 10^9 \rceil = \lceil 21.67 \rceil = 22$ steps complete, and $22 \times 6 \times 10^9 - 1.3 \times 10^{11} = 2 \times 10^9$ FLOPs are done after the request.
3. $2048 \times 2048 \times 3 \times 4 = 50{,}331{,}648$ bytes; base64 $4 \times 16{,}777{,}216 = 67{,}108{,}864$ bytes. It does not fit a dialect body (8,388,608). The base64 alone equals the event line's limit exactly, so with the JSON around it the line is too long; it fits neither an event line nor a `/v1/run` body. It fits one bus descriptor (268,435,456 bytes).
4. Implementation: the fourteen printed rows, refusal texts included, are the check.
5. Implementation: the nine printed rows are the check.
6. Implementation: the printed rows are the check. For (3, 2) with seed 1, h = 1,293,170,019 = 0x4d143563: red $(3 + h) \bmod 256 = (3 + 99) = 102$, green $(2 + (h \gg 8)) \bmod 256 = 2 + 53 = 55$, blue $(3 + 2 + h) \bmod 256 = 104$, so (102, 55, 104)/255.
7. Implementation: the printed rows are the check. The added model is chat (a streaming `generate` taking `messages` and emitting text), without tools or vision, so the chat-only dialect and the others both list it; the text-completion route reads `suffix` as fill-in-the-middle support (`resolve_text`).
8. `check_type` accepts for an `Int` a JSON number that is an integer (`is_i64` or `is_u64`), and for a `Float` any JSON number. `2.5` and `2.0` are accepted by a `Float` and refused by an `Int` (the second because JSON keeps its fraction); the text `"2"` is refused by both.
9. The chat-only dialect offers a model on its messages route when the chat rule admits it. The OpenAI dialect offers chat completions (chat rule), completions (chat, with `suffix` for fill-in-the-middle), embeddings (`embed` with `text`), image generations (an image output from a `prompt` with no required input), transcriptions (`transcribe` with an `audio` input and text output), speech (`speak` with `text` and an `audio` output) and video (a sellable video contract with no input clip); the OpenRouter dialect offers the same routes except completions. The chat handlers always dispatch the literal action `generate` with chat params, so an action of another name, however similar its shape, would be listed and then refused on every call.
10. The reader is wrapped in `take(MAX_LINE_BYTES + 1)` before `read_until(b'\n', ...)`, so it reads at most one byte more than the limit. After the newline is removed, a line of exactly the limit has length 67,108,864 and is served; a line one byte longer reaches 67,108,865 bytes, which is more than the limit, and is refused. The tests are `a_line_at_the_limit_is_served` and `an_oversized_line_is_refused_and_closes_the_connection` in `crates/server/tests/it/socket_hardening.rs`.
