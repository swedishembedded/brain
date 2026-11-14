# Release gates

A release is published from a tag only when four tiers of gates have passed on
the tagged commit. Each tier is a Makefile target, so the gate a workflow runs
is the one a developer runs locally; `.github/workflows/ci.yml` runs tiers 1
and 2 on every push and pull request, and `.github/workflows/release.yml` calls
it again for the tag and adds tiers 3 and 4.

| tier | what it proves | target | where it runs |
|---|---|---|---|
| 1. Build and static | the binary builds on every shipped system; the gate scripts and their self-tests hold; clippy is clean | `cargo check` (build matrix), `make ci/static`, `make check/clippy` | hosted runners |
| 2. Deterministic CPU correctness | unit and integration tests of the CPU-only crates, the end-to-end suites against the mock model | `make ci/cpu` (`ci/cpu/unit`, `ci/cpu/e2e`) | hosted runners |
| 3. Accelerated correctness | CPU == Vulkan == CUDA parity on the devices the machine has | `make ci/gpu` (`scripts/gates/parity-gate.sh`) | self-hosted runner labelled `gpu`, required by `release.yml` |
| 4. Release qualification | the whole suite with its fixtures (`make test/full`), a dependency audit, a CycloneDX SBOM, a provenance attestation for every binary | `make test/full` on the maintainer's machine; `audit`, `sbom` and `publish` jobs of `release.yml` | maintainer's machine and `release.yml` |

`make test/full` stays the full qualification: it needs the model fixtures
(`make testdata/fetch`) and GPUs, so no hosted runner can run it. The release
workflow does not replace it; it makes the parts a runner can verify
unskippable.

## No silent skips

`cargo test` reports a skipped comparison as a pass and `bats` exits 0 when
every test skips, so an exit code cannot tell "every check matched" from "no
check ran". Each suite target (`ci/static/gate-specs`, `ci/cpu/*`, `ci/gpu/*`)
captures its output and passes it to `scripts/ci/suite-report.py`, which fails
the suite when it reports no tests, a failure, a `# skip` line (bats) or a
`SKIP:` line (`brain_testutil::skip`). Tests marked `#[ignore]` are counted and
shown apart: the attribute names its reason in the source.

`make ci/matrix` turns the verdicts in `target/ci/results` into a
pass/fail/skip/not-run table, appended to the job summary in a workflow, and
fails when a required suite (`CI_REQUIRED`) has no pass. A job that died before
reporting is a `not-run` row.

## What is, and is not, in the tiers today

- **Static tier.** `check/generated/book` is not clean on the tree: the book figure check reports existing
  findings. `ci/static` runs every gate that is clean, listed in `ci/static/scripts`, and a gate rejoins the
  tier when its findings are fixed. A required gate that is red from its first run gates nothing.
- **CPU unit tests.** They run over `CI_CPU_CRATES`, the crates whose tests are
  deterministic on the CPU backend with no model files and skip nothing; the
  whole workspace lane (`make test`) needs more disk and time than a hosted
  runner has. Add a crate to the list when its tests meet that contract.
  The gradcheck suite is not in this tier: it creates a GPU device in several
  of its tests even under `BRAIN_DEVICE=cpu`, so it belongs to tier 3. Doc tests of those crates are empty, so no doc-test suite is required there;
  `make test/doc` belongs to tier 4.
- **Release.** `release.yml` waits for a runner labelled `gpu`; with none
  online the release is not published. The dependency audit is
  `cargo deny check advisories sources` (`deny.toml`); there is no crate-license
  allowlist, because licensing here is per model and per ported table
  (`.agents/rules/licensing.md`).
- **Provenance.** The attestation action is pinned by version tag, the one pin
  in the workflows that is not a commit.
