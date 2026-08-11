# Benchmarking your setup (`brain perf`)

`brain perf` measures how brain performs on the hardware in front of it:
latency, saturated throughput, and behavior under a realistic arrival of
requests — plus a way to compare a result against a saved baseline over
time. It is distinct from `brain bench`, which asks whether a model
architecture learns a task; `perf` asks how much correct work the engine
delivers per unit of hardware, memory, and time.

## What it measures

Every `perf` run reports against the same few axes, regardless of which
model or scenario you're running:

- **Latency** — time to first output artifact (a token, a detected frame, an
  audio chunk, a denoise step — whatever the model's natural unit is) and the
  time between subsequent artifacts, as full percentile distributions
  (P50/P95/P99), not just an average.
- **Throughput** — output artifacts per second under saturation, and
  "goodput": how much of that throughput still lands inside a declared
  latency budget.
- **Serving behavior under load** — what happens at a given request
  concurrency over the real transport, not just in a tight in-process loop.
- **Correctness** — every run carries a validity gate against a reference;
  a run whose output drifted outside tolerance is marked invalid and
  excluded from comparisons, so a "faster" result can never quietly also be
  a wrong one.

Every result records the hardware it ran on (device, backend, adapter
string, core count, RAM) so a result from one machine is never silently
compared against a result from another.

## Basic usage

```bash
brain perf list                              # registered scenarios + targets
brain perf run <scenario> --target <spec> [--workload W] [--seed S] [--out F] [--smoke]
brain perf compare results/perf-*.json       # leaderboard across saved runs
brain perf gate <candidate.json> --baseline <baseline.json>   # regression check
```

`<scenario>` is one of `latency`, `throughput`, `serve`, or `sweep` (a
concurrency ladder). `--target` names what to measure, e.g.
`qwen-synth:28x1024x16` (the real serving engine on randomly-initialized
weights of that shape — no checkpoint needed, good for hardware comparison)
or `qwen:out/qwen.safetensors` (a real checkpoint). Run `brain perf --help`
for the full list of target specs (one per served model family).

`--smoke` shrinks a run to a few seconds, useful for a quick sanity check
rather than a real measurement.

## Comparing over time

Each run writes a JSON artifact under `results/`. `brain perf compare`
reads a set of these and prints a leaderboard, refusing to rank runs whose
artifact unit differs (comparing tokens/s against frames/s is meaningless)
or whose correctness gate failed. `brain perf gate` compares one candidate
artifact against a saved baseline and fails if throughput drops, or latency
rises, past a floor — useful for catching a regression before it ships.

Because results are just JSON files, you can diff them, graph them, or keep
a history of them in your own tooling — `perf` doesn't lock you into a
particular report format.

## Makefile shortcuts

```bash
make perf                    # core scenarios (latency, throughput, serve, sweep) on the current device
make perf/<scenario>         # one scenario, e.g. make perf/sweep
make perf/compare            # leaderboard over everything in results/
make perf/smoke              # every scenario shrunk to CI-sized runs
```

See `brain perf --help` for the complete set of flags and target specs.
