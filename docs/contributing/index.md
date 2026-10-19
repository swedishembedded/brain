# Contributing

## Building and checking

Everything goes through the Makefile, which wraps cargo with the flags the
project expects. `make help` lists the targets as a tree.

```bash
make build/release      # optimized ./target/release/brain
make test/gradcheck     # finite-difference backprop gate
make check/scope        # no tracked file names a project brain does not depend on
make check/generated    # generated files match their sources
```

Run the tests for what you changed rather than the whole suite. Do not run
`cargo fmt` workspace-wide: there is no shared rustfmt configuration, and a
blanket format rewrites unrelated files.

Every source file carries an SPDX identifier and a copyright line; `make check/spdx`
verifies them.

## Documentation

The documentation lives in `docs/` and is the source of truth for the website.
`docs/manifest.yaml` is the one authoritative navigation order: it drives the
published sidebar, the PDF and the coverage check (every Markdown file under
`docs/` is either listed or explicitly excluded).

| Target | What it does |
|---|---|
| `make docs/build` | build the publication into `build/docs/` (offline, no Rust build) |
| `make docs/check` | build twice and compare, check links and coverage, check generated references are current |
| `make docs/pdf` | build the single-file manual and PDF into `build/pdf/` (needs pandoc and xelatex) |
| `make docs/clean` | remove generated documentation output |

Where each kind of page goes:

- `getting-started/`: install and first run.
- `guides/`: task-oriented how-tos (inference, training, hardware, serving).
- `concepts/`: explanations of how the system works.
- `reference/`: precise interfaces (CLI, SDK, API, configuration, one page per model, kernels).
- `examples/`: published from the `samples/` READMEs, listed by the samples generator.

Some references are generated and must not be edited by hand. `make generate`
lists the generators; `make check/generated` fails when one is stale.
`docs/reference/kernels/index.md` and `docs/reference/kernels/cuda.md` come from
the kernel sources, and the Examples block of `docs/manifest.yaml` comes from
`samples/**/README.md`.

Links between pages are ordinary relative Markdown links so they work on
GitHub. Links to other projects use the canonical
`https://swedishembedded.com/docs/<project>` URLs. Performance figures do not
belong in prose; `scripts/gates/check-no-perf-numbers.sh` explains how to mark
a reviewed one.
