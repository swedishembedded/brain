# Observability — stats subsystem + `braintop`

brain exposes a live, **self-describing** view of its serving state (what models are
resident and where, accelerator memory, executor counters, in-flight requests) as a
JSON snapshot over D-Bus, and ships **`braintop`** — a btop-like TUI — to render it.

## The stats model (`crates/stats`)

`brain_stats::StatsSnapshot` is a hierarchical, serde-JSON tree. Every section is a
**data-driven collection keyed by id** (nothing hardcoded — 8 GPUs render as 8 rows),
and every level carries an open `extra: BTreeMap<String, Value>` so new leaf metrics
need no schema change:

- `accelerators[]` — {id, kind cpu/gpu/npu, name, index, mem_total, mem_used,
  mem_reserved, util?} (nvidia-smi-like, from the executor's device budgets).
- `models[]` — {id, family, capabilities, resident, instances:[{device, tier, mem}]}
  (the catalog joined with residency placement — where each model is loaded).
- `executor` — {builds, evictions, batches, jobs, resident, queue_peak, max_batch,
  max_parallel}.
- `requests[]`, `connections[]` — in-flight work (data-driven; populated as the request
  registry is wired).

Components contribute via the `StatsSource` trait + an `Assembler` — no central
switchboard. `brain_stats::snapshot_from_executor(&Executor)` builds the live snapshot.

**Adding a metric (the AGENTS.md contract):** add a field to the relevant typed section
in `crates/stats` (or emit into `extra`); it flows through the JSON snapshot and
`braintop` renders it automatically — typed views for the known sections, the generic
tree view for `extra`. Never hardcode counts.

## Over D-Bus

`brain serve --dbus` exposes on `com.swedishembedded.Brain1`:
- `stats_snapshot() -> String` — the JSON snapshot on demand.
- a `StatsStream` signal emitting the snapshot every 500 ms (≥2 Hz).

## `braintop` (`crates/braintop`)

A standalone binary (ratatui + zbus + the shared `brain-stats` types — it does **not**
link the engine). It subscribes to `StatsStream` (falling back to polling
`stats_snapshot()`), and reconnects with a "waiting for brain…" state if brain isn't up.

```bash
braintop                       # live dashboard (session bus)
braintop --system              # system bus
braintop --cli                 # flat, shell-parseable snapshot, then exit
```

- **Dashboard** (progressive-reveal, responsive): accelerator mem/util gauges; a
  per-model **residency bar split by device — CPU red / NPU yellow / GPU green** (where
  each model is loaded); executor counters; requests-in-progress; connections.
- **Keyboard:** `q`/`Ctrl-C` quit; `j`/`k` or `↑`/`↓` select; `Tab` cycles panels;
  `Enter`/`→` drills into a subview (a model → its per-device instances; an accelerator/
  request/connection → its detail); `Esc`/`h`/`←` back. Every detail view ends with a
  generic `extra` key→value tree, so new metrics appear with no braintop change.
- **`--cli`** prints stable `path.to.metric=value` lines (collections keyed by id, e.g.
  `accelerator.gpu0.mem_used=…`, `model.qwen.instances.gpu0.tier=hot`) — for scripts:
  ```bash
  braintop --cli | grep '^accelerator\.' | awk -F= '...'
  ```
