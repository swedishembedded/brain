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
