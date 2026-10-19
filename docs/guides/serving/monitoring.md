# Monitoring — `braintop`

`braintop` is a live, btop-like TUI for a running `brain serve` process. It
shows what's resident, where, and how busy it is: resident models,
accelerator memory, executor counters, and in-flight requests, refreshed
several times a second.

Under the hood, brain exposes this state as a JSON snapshot over D-Bus — a
hierarchical object with `accelerators`, `models`, `executor`, `requests`,
and `connections` sections, each a list keyed by id (so N GPUs render as N
rows, with no fixed schema). `braintop` subscribes to a live stream of this
snapshot and renders it; it does not require the `brain serve` process to be
running elsewhere on the same machine — point it at the right bus and it
reconnects automatically, showing a "waiting for brain…" state until the
server comes up.

## Invocation modes

```bash
braintop                       # live dashboard (session bus)
braintop --system              # live dashboard (system bus)
braintop --cli                 # flat, shell-parseable snapshot, then exit
```

## A real snapshot

`--cli` prints the same state the dashboard renders, one `path=value` per
line, so it composes with `grep` and needs no terminal. Below is an actual
snapshot taken after four concurrent chat requests were sent to one server,
trimmed to the interesting lines:

```console
$ braintop --cli | grep -E '^(accelerator|executor)\.|resident=true'
accelerator.cpu.kind=cpu
accelerator.cpu.mem_total=47981772800
accelerator.gpu0.kind=gpu
accelerator.gpu0.mem_total=20090035470
accelerator.gpu0.mem_used=12149169422
accelerator.gpu0.mem_reserved=2147483648
accelerator.gpu1.kind=gpu
accelerator.gpu1.mem_total=11647582208
accelerator.gpu1.mem_used=0
accelerator.gpu1.mem_reserved=2147483648
model.Qwen/Qwen3-0.6B.resident=true
executor.builds=1
executor.evictions=0
executor.batches=2
executor.jobs=5
executor.resident=1
executor.queue_peak=4
executor.max_batch=4
executor.max_parallel=1
```

Read the executor counters together and they describe what the scheduler
actually did: **five jobs arrived, the queue reached four, and they were
served in two batches of up to four** - continuous batching, not five
sequential passes. `builds=1` says the model was constructed once and reused;
`evictions=0` that nothing had to be unloaded to make room.

The accelerator rows are the residency budget, not the card's raw capacity:
`mem_total` is what brain may use after `mem_reserved` headroom is subtracted,
which is why the two GPUs report different totals here - one was already
carrying other work.

## The dashboard

Progressive-reveal, responsive layout:

- Accelerator memory/utilization gauges, one per device.
- A per-model **residency bar split by device** — CPU red / NPU yellow / GPU
  green — showing where each model is currently loaded.
- Executor counters (builds, evictions, batches, jobs, queue depth, max
  batch/parallelism seen).
- Requests currently in progress.
- Active connections.

### Keyboard

| Key | Action |
| --- | --- |
| `q` / `Ctrl-C` | quit |
| `j`/`k` or `↑`/`↓` | move selection |
| `Tab` | cycle panels |
| `Enter` / `→` | drill into a subview (a model's per-device instances; an accelerator's, request's, or connection's detail) |
| `Esc` / `h` / `←` | back |

Every detail view ends with a generic key/value tree for any metric that
doesn't have a dedicated widget yet, so new metrics show up without waiting
on a `braintop` update.

## `--cli`, for scripts

`braintop --cli` prints one snapshot as stable `path.to.metric=value` lines
— collections keyed by id — then exits. Useful for scripting and health
checks:

```bash
braintop --cli | grep '^accelerator\.' | awk -F= '{print $1, $2}'
```

Example lines: `accelerator.gpu0.mem_used=…`, `model.qwen.instances.gpu0.tier=hot`.
