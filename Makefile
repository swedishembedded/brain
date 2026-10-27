# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# brain - top-level workflow.
#
# Pure Rust + WGSL training/eval engine. These targets generate datasets, train
# the GPT baseline, evaluate it (perplexity + task exact-match), run the
# gradient-check correctness gate, and drive the federated-MoE artifact pipeline.
# The browser/WebGPU demo lives in crates/web (delegated below).
#
# Quick start (`make help` prints every target as a tree):
#   make build/release
#   make gpt/calculator/data
#   make gpt/calculator/train
#   make gpt/calculator/eval
#   make bench           # run the architecture-evaluation benchmark suite
#   make test/gradcheck  # backprop correctness gate
#   make test            # fast test lane
#
# MAKE TARGETS (rule: AGENTS.md, "Make targets"). A target is
# `make <concept>/<subconcept>/.../<action>`: concept first, verb last, one
# segment per directory level of a test or sample hierarchy, and every prefix
# is itself a target so `make <TAB>` descends one level at a time. Where a set
# follows from the tree (samples, crate examples, perf scenarios, benchmarks,
# e2e suites) it is GENERATED as explicit targets by static pattern rules over
# a computed list - never `%` pattern rules, which bash completion cannot see -
# so adding a directory adds a target and nobody maintains a list. `make help`
# prints the tree from the ##@ section and ## doc lines;
# scripts/gates/check-make-targets.sh (check/make-targets) enforces the naming.
# An area target is either the default action for its area (test, test/e2e,
# test/parity, bench, perf, docs, package/deb, check/generated) or, where no
# default is safe or sensible (build, package, release, install, flash, anything
# that writes outside the working tree), only lists its children and does
# nothing else.

# CARGO_HOME selects where registry caches, compiled crates, and binaries are
# stored. Pinned to the ordinary default so every invocation (make, CI,
# agent) resolves the same absolute source paths in Cargo's fingerprint
# files rather than rebuilding from scratch if $HOME ever differs between
# them.
CARGO_HOME = $(HOME)/.cargo
export CARGO_HOME

# sccache caches rustc invocations by (source, flags, deps) hash across
# separate `cargo` processes - auto-enabled here so every developer/CI box
# with it installed benefits by default, not just a machine that happens to set it in a
# local, gitignored `.cargo/config.toml`. A machine that also wants the
# faster `mold` linker or a capped job count (see that file's own comment
# for why - 22 concurrent thin-LTO test-binary links exceeded this box's
# RAM) still sets those there; only the wrapper itself is universal enough
# to default on for everyone.
ifneq ($(shell command -v sccache 2>/dev/null),)
export RUSTC_WRAPPER = sccache
endif

.DEFAULT_GOAL := help

BRAIN  ?= ./target/release/brain
# The debug-build binary the e2e bats suites drive by default (they build via the
# `build/debug` target, not `build/release`, so the fast lane stays fast). Override
# to point an e2e run at a release build instead: `BRAIN_BIN=./target/release/brain
# make test/e2e/api-conformance`.
BRAIN_BIN ?= ./target/debug/brain
# Rewritten in place by `make release/{patch,minor,major}` (bump2version) -
# the single source the release targets below read AFTER the bump, since a
# sub-make re-reads the Makefile from disk. Cargo.toml's [workspace.package]
# version is the actual build-time source of truth (--version, D-Bus, the
# .deb); this mirrors it for the release machinery only.
BRAIN_VERSION := 3.0.0
# The git remote whose GitHub repository receives a release (release/publish,
# package/deb/publish).
RELEASE_REMOTE ?= origin
PIP    ?= python3 -m pip
DATA   ?= data
OUT    ?= out
# SEED has no default: every `make` target that takes a seed draws a random
# one per invocation and prints it to stderr (pass `SEED=<n>` to reproduce).
SEED   ?=
SEED_FLAG = $(if $(SEED),--seed $(SEED))
STEPS  ?= 1000
N      ?= 100000
ARCH   ?= gpt2

# model size (GPT)
LAYERS ?= 4
DMODEL ?= 128
HEADS  ?= 4
BLOCK  ?= 64
BATCH  ?= 32
LR     ?= 3e-3

# YOLO detector (CPU-friendly tiny config; geometry matches the synthetic
# `detect` dataset, so the dataset's 128px images upload without letterboxing).
YOLO_N     ?= 64
YOLO_STEPS ?= 150
YOLO_BATCH ?= 4
YOLO_LR    ?= 3e-3
YOLO_CONF  ?= 0.1
YOLO_IOU   ?= 0.45

SHAKE_URL := https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# The recipe of every list-only umbrella: print the children of the target
# (read from the Makefile's own ## doc lines by scripts/make-help.sh) and do
# nothing else.
LIST_CHILDREN = @bash scripts/make-help.sh $@

# ---------------------------------------------------------------------------
##@ Build and package
## Builds, packages, the container image, CUDA kernels, GH200 probes and the
## Python/NPU environment. Only package/deb acts by default; the other areas
## list their children.

## build - lists the build targets; builds nothing (a build costs tens of GB of target/)
build:
	$(LIST_CHILDREN)

# The DEV profile on purpose, and NOT `cargo check`. The four fast e2e bats
# suites `test/e2e` aggregates - api-conformance, shutdown, samples, ready -
# each name this target as their prerequisite and drive `$(BRAIN_BIN)`, which
# defaults to ./target/debug/brain; `package/deb/debug` packages that same file.
# Those need a LINKED binary, which `cargo check` never produces, and each suite
# SKIPS itself when the binary is absent rather than failing - so swapping in
# `cargo check`, or pointing this at the release profile, would leave the whole
# e2e lane reporting green while running nothing. The dev tree is the price of
# that (tens of GB on the shared build volume, alongside release's own); it
# buys a link step fast enough to keep the e2e lane a fast lane, which is why
# `build/release` below stays the separate, slower target everything else runs.
## build/debug - dev-profile 'brain' binary (what the fast e2e suites drive)
build/debug:
	cargo build

## build/release - optimised 'brain' binary
build/release:
	cargo build --release

## package - lists the package targets; builds nothing
package:
	$(LIST_CHILDREN)

## package/deb - default: package/deb/release (output in target/debian/)
##   Build self-contained Debian packages for package-only integrations.
package/deb: package/deb/release

## package/deb/debug - Debian package of the debug build (fast, unoptimised)
package/deb/debug: build/debug
	bash scripts/build/build-deb.sh --binary target/debug/brain --flavor debug

## package/deb/release - Debian package of the release build
package/deb/release: build/release
	bash scripts/build/build-deb.sh

# Checks every precondition (gh, its login, the tag checked out and on GitHub,
# a signing key) before the release build, so a missing one costs seconds.
## package/deb/publish - sign the release package and attach it to the GitHub release of v<version> (created if missing; an earlier upload is replaced); needs gh, a GnuPG key and the tag on GitHub (RELEASE_REMOTE, default origin)
package/deb/publish: release/notes
	bash scripts/build/publish-deb.sh --check --version $(BRAIN_VERSION) --remote $(RELEASE_REMOTE)
	$(MAKE) package/deb/release
	bash scripts/build/publish-deb.sh --version $(BRAIN_VERSION) --remote $(RELEASE_REMOTE) \
		--notes-file target/release-notes-v$(BRAIN_VERSION).md \
		target/debian/brain_$(BRAIN_VERSION)_$$(dpkg --print-architecture).deb

# The aarch64 CUDA image and its CI-equivalent gates; see
# tools/container/aarch64-cuda.Containerfile and scripts/build/container.sh.
## container - lists the container image targets; builds nothing
container:
	$(LIST_CHILDREN)

## container/runtime - build the aarch64 CUDA runtime image (driver-only, AOT kernels)
container/runtime:
	scripts/build/container.sh runtime

## container/ci - build the image's CI stage: the gates that need no GPU
container/ci:
	scripts/build/container.sh ci

## container/smoke - run the image on this host's GPU (brain devices)
container/smoke:
	scripts/build/container.sh smoke

## cuda - lists the CUDA toolchain and kernel targets; does nothing
cuda:
	$(LIST_CHILDREN)

# Compile the generated tier and the native kernels into cubins + PTX with a
# manifest, in the AOT directory the CUDA backend reads (BRAIN_CUDA_AOT_DIR, or
# `cuda-aot` under the pipeline cache). A machine with a driver and no toolkit
# then runs them without NVRTC. `make cuda/aot LANE=12` and `LANE=13` fill one
# directory in turn; ARGS passes flags through (`ARGS="--targets sm_90"`).
## cuda/aot - build the CUDA kernels ahead of time (cubins + PTX + manifest) for
##   driver-only machines; LANE=12|13 selects the toolkit lane
cuda/aot:
	scripts/build/cuda-aot.sh $(ARGS)

# Install a CUDA toolkit lane (12 or 13), or the Nsight profilers (`nsight`),
# into a user-owned prefix (`$BRAIN_CUDA_PREFIX`, default ~/.local/cuda) with no
# root: `make cuda/install LANE=13`, then
# `eval "$(scripts/build/install-cuda-userspace.py env 13)"`. Downloads are
# SHA-256 verified. See the script's own header for what a lane contains.
## cuda/install - install a CUDA toolkit lane (LANE=12|13|nsight) into a user-owned prefix, no root
cuda/install:
	scripts/build/install-cuda-userspace.py install $(LANE)

# Which of the WGSL catalogue the CUDA backend can run on THIS device: every
# kernel is generated and compiled with NVRTC for the device's architecture, and
# the runnable and refused sets are written to cuda-coverage.json (override with
# OUT=path). Needs a CUDA device and toolkit lane; without them it checks
# generation only.
## cuda/coverage - which WGSL kernels the CUDA backend runs on this device (-> cuda-coverage.json)
cuda/coverage:
	BRAIN_CUDA_COVERAGE_OUT=$(or $(OUT),cuda-coverage.json) cargo test --release --offline -p brain-backend-cuda --test catalogue_coverage -- --nocapture

## gh200 - lists the GH200 memory probes; does nothing
gh200:
	$(LIST_CHILDREN)

## gh200/probe - build and run the GH200 hardware probe
gh200/probe:
	$(MAKE) -C tools/gh200-probe
	tools/gh200-probe/probe

# Device vs managed vs system memory: first touch, steady-state read bandwidth
# and the prefetch effect (BUFFER_MIB sizes the buffer; OVERSUBSCRIBE=1 adds the
# oversubscription run, which takes all free device memory).
## gh200/placement - device vs managed vs system memory bandwidth and prefetch effect
##   BUFFER_MIB sizes the buffer; OVERSUBSCRIBE=1 adds the run that takes all free device memory
gh200/placement:
	$(MAKE) -C tools/gh200-probe
	tools/gh200-probe/probe --placement

## environment - lists the Python/NPU environment targets; installs nothing
environment:
	$(LIST_CHILDREN)

# Install the Python tooling (OpenVINO/NPU runtime, torch + transformers for the
# benchmark reference rows, etc.) into the current environment. The Rust engine
# needs none of these - this is for tools/ and the `--device npu` runtime.
## environment/requirements - pip-install the Python tooling (OpenVINO/NPU, torch, ...)
environment/requirements:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# `environment/requirements` + hardware-specific setup that a package list alone
# can't cover: detects a real Intel NPU (if any -- a no-op, exit 0 otherwise),
# checks the kernel/driver prerequisites, installs/upgrades OpenVINO, and
# PROVES it can see the NPU (not just that the package imported) by asking
# OpenVINO's own Core() for its device list. See scripts/build/
# setup-npu-runtime.sh for why there is no single pinned driver->OpenVINO
# version table to check against.
#
# This pulls in the FULL requirements.txt first, including torch -- whose
# default PyPI wheel depends unconditionally on nvidia-*-cuXX / cuda-bindings
# / cuda-toolkit on Linux (that's pip resolving torch's own declared deps,
# not anything this repo lists -- `pip show torch` names them; there is no
# `cuda` line in requirements.txt). Those packages install as dead weight on
# an NVIDIA-less box like this one -- torch never touches them here, brain's
# own iGPU/NPU paths don't use torch at all -- but this target still pays for
# the download/disk because `environment/requirements` installs torch for the
# benchmark reference rows (tools/bench/*, tools/goldens/*), which do need it.
## environment/full - environment/requirements, then detect and verify a real Intel NPU (no-op if absent)
environment/full: environment/requirements
	scripts/build/setup-npu-runtime.sh

# OpenVINO/NPU setup only, skipping the rest of requirements.txt (torch +
# its CUDA deps, transformers, ultralytics, ...) -- use this to iterate on
# NPU detection/driver setup without re-installing the whole Python stack.
## environment/openvino - OpenVINO/NPU setup only, skips torch/CUDA (fast iteration)
environment/openvino:
	scripts/build/setup-npu-runtime.sh

## npu - lists the NPU diagnostics; does nothing (the YOLO NPU pipeline is yolo/npu)
npu:
	$(LIST_CHILDREN)

# Read-only diagnosis, NOT install: runs every NPU check this repo has hit
# (device node/driver binding/kernel version, firmware, kernel debugfs
# reset/fault counters, the userspace libze_intel_vpu.so.1 compat symlink,
# OpenVINO's own device enumeration run twice to catch flaky/wedged
# behavior, and brain's own crates/npu test) and prints one clear verdict.
# Use this to answer "is the NPU actually accessible right now",
# `environment/full`/`environment/openvino` above to install/fix. See
# scripts/build/npu-diagnose.sh's header for exit codes.
## npu/diagnose - read-only: is the Intel NPU actually accessible right now (VERBOSE=1 for detail)
npu/diagnose:
	scripts/build/npu-diagnose.sh $(if $(VERBOSE),--verbose)

# ---------------------------------------------------------------------------
##@ Test and check
## `test` is the fast lane. `test/full` is the release gate and runs everything
## below it. Checks (`check/*`) are static gates; `check/generated` is the only
## check area that runs by default.

# `make test` is the FAST LANE and must stay fast: unit + integration tests
# only, reusing the release build.
#
# Three things were making the full suite take ~an hour, all measured:
#
#  1. `cargo test` defaults to the DEBUG profile, so it recompiled the whole
#     workspace even right after `make build/release`. Tests now run `--release` and
#     reuse that build.
#  2. Doc-tests link one binary per crate against the full graph - ~18s per
#     crate for ~30 examples, most of them `no_run`. They are real coverage but
#     they are not fast feedback, so they get their own lane.
#  3. Many concurrent GPU DEVICES in one process deadlock the NVIDIA driver
#     (~50% of parallel runs), and a device leaked into process exit crashes
#     it. FIXED at the root: models share one device (Gpu::share/new_like) and
#     test binaries use the weak-pool fixture gpu_core::testgpu, whose device
#     dies with its last in-process handle - kronos is proven clean at
#     --test-threads=48, and after the fixture migration every GPU-test crate
#     either shares the pooled device (qwen3/gpt2/qwen3tts/ecapatdnn/glmdsa/
#     toymoe/toypid/toyseq2seq/toyautoencoder/chronos2/fincast/splat/model/
#     wm-core/yolov8), pins the CPU backend (never touches the card), or
#     gates multi-device tests on the hardware being present. 8 is the
#     suite-wide proven point (qwen3 + the six heaviest migrated crates,
#     0 failures).
#
# The timeout turns a deadlock into a fast, loud failure instead of an hour of
# silence. Override any of these: TEST_THREADS=1 make test restores serial.
TEST_THREADS ?= 8
# The guard is a DEADLOCK detector, not a performance target: it must sit above
# the measured run time so a completing suite never reads as a hang. The VLM
# crates (fastvlm/qwen3vl/moondream3) grew the suite past the old 1500s budget -
# 266 suites / 1492 tests now complete at TEST_THREADS=8 inside ~2000s wall on
# the 2xP40 box, so the guard sits at 2400. Making the suite faster is real
# work (slow-lane moves), not a tighter timeout.
TEST_TIMEOUT ?= 2400
CARGO_TEST   ?= cargo test --release --offline

# Build first WITHOUT the timeout, then run WITH it. The deadlock guard is a
# statement about *running tests* - a cold rebuild after an engine change takes
# minutes on its own, and letting it eat the budget turns "compiling" into a
# false "TIMED OUT" that reads like a hang.
## test - default: FAST lane, unit + integration tests, no doc-tests, GPU serialised
##   (test/doc, test/slow, test/e2e, test/parity ... are separate; test/full runs them all)
test:
	@echo "test: fast lane (unit + integration, no doc-tests, GPU serialised)"
	@$(CARGO_TEST) --lib --bins --tests --no-run
	@timeout $(TEST_TIMEOUT) $(CARGO_TEST) --lib --bins --tests -- --test-threads=$(TEST_THREADS); \
	rc=$$?; \
	if [ $$rc -eq 124 ]; then \
		echo; echo "TIMED OUT after $(TEST_TIMEOUT)s of RUNNING - almost certainly a deadlock."; \
		echo "Find it with:  scripts/gates/test-times.sh --top 10"; \
	fi; \
	exit $$rc

# cargo-nextest: same tests as `test`, but each one runs as its own OS
# process with a per-test timeout (.config/nextest.toml), not one shared
# binary under one whole-suite timeout. `test`'s TEST_TIMEOUT is a real
# deadlock guard, but it only fires once, for the WHOLE run, and does not say
# which test hung - a single wedged GPU test (a fence wait or lock that never
# returns) blocks every other test behind it for up to TEST_TIMEOUT seconds
# with no attribution. Observed for real: several unrelated GPU tests each
# independently hung for 10-30+ minutes under heavy concurrent GPU load
# (multiple processes competing for the same physical device), with nothing
# short of manually inspecting `/proc` to find which process and kill it.
# nextest turns that into: this ONE named test is killed and reported after
# ~90-190s, everything else keeps running. Requires `cargo install
# cargo-nextest --locked` once; not yet the default `test` target since it
# has not been run against the full workspace end to end, only smoke-tested.
## test/nextest - the fast lane under cargo-nextest: one process and one timeout per test
test/nextest:
	cargo nextest run --release --offline --lib --bins --tests

## test/package - the spec of the release publishing script against a scratch repository and a fake gh (bats; no build, seconds)
test/package:
	bats tests/package/publish_deb.bats

# Doc-tests: real coverage, but each crate links its own binary, so this is
# minutes of linking for a handful of examples. Separate lane, not the default.
## test/doc - doc-tests (slow: one link per crate)
test/doc:
	$(CARGO_TEST) --doc

# Tests marked `#[ignore = "slow: ..."]` - long training/parity runs that do not
# belong in fast feedback.
## test/slow - the #[ignore]d long-running tests
test/slow:
	$(CARGO_TEST) --lib --bins --tests -- --ignored --test-threads=$(TEST_THREADS)

# Everything, for a release gate.
#
# `test/parity/strict` is in this list on purpose, and it is the only member
# that can fail for want of DATA rather than for want of correct code. That is
# the point of it. Every other target here is green on a box with no fixtures
# at all, because cargo reports a skipped test as a PASS - so `test`, on its
# own, cannot distinguish "every reference comparison matched" from "no
# reference comparison ran". It has failed to distinguish exactly that,
# repeatedly, and each time the suite stayed green while the model was wrong.
#
# The default suite list is the one `make testdata/fetch` provisions, so the
# prerequisite for a release gate is one documented command. Where a suite
# genuinely cannot be provisioned, narrow the list explicitly and visibly
# (`make test/full PARITY_STRICT_SUITES="..."`) rather than dropping the
# target - a narrowed list still says which comparisons were certified.
#
# `test/parity` (bare, not `/strict`) is in this list for a different reason:
# it is the ONLY target here that runs the gradcheck suite under
# BRAIN_DEVICE=cpu. `test`/`test/gradcheck` run on the default (Vulkan) backend
# only, so a model whose CPU path silently diverges from its GPU path - a
# kernel pairing that assumes a shared Params contract neither kernel actually
# shares, two kernel table entries the CPU JIT's single function namespace
# collides on - stayed green on every routine run and was only ever caught by
# someone remembering to invoke `make test/parity` by hand. It found exactly
# that failure mode twice in one sitting once someone finally ran it
# (timesfm3's RMSNorm epsilon, controlnet's duplicate `scale_chan`
# registration - see `.agents/rules/.agents/knowledge/`). Needs no external
# fixtures, so unlike `test/parity/strict` it carries no narrowing knob and no
# "green because skipped" risk.
## test/full - release gate: test + test/doc + test/slow + test/e2e + test/conformance + test/rl + test/gradcheck + test/package + every check/* gate
##   + test/parity + test/parity/strict (needs make testdata/fetch)
test/full: test test/doc test/slow test/e2e test/conformance test/rl test/gradcheck test/package check/scripts check/env-reads check/env-writes check/test-skips check/seed-literals check/serve-conformance test/gates check/spdx check/paths check/scope check/saved-policy check/agents-md check/files check/samples check/sdk-features check/make-targets check/generated test/parity test/parity/strict

# Rank every test binary by wall time; --budget fails if any exceeds it. This is
# what keeps the fast lane fast.
## test/times - rank test binaries by wall time
test/times: build/release
	scripts/gates/test-times.sh --top 15

# Render the capability-skip ledger `brain_testutil::skip_unvalidated_capability`
# appends to: which hardware-gated code paths ran UNVALIDATED on this box, why,
# how many times, and which call sites. Run the suite first - the ledger only
# grows when a gated test actually executes its fallback path.
## test/capability-report - render the hardware-capability skip ledger
##   (brain_testutil::skip_unvalidated_capability)
test/capability-report:
	scripts/gates/capability-report.sh

# The documented backprop gate must BE the real gate: `brain gradcheck` runs
# exactly one model (GPT), while the ~20 model checks + kernel FD suites live
# in crates/gradcheck's tests. Point the target at those (audit F17).
## test/gradcheck - the FULL numerical backprop gate (every model check + kernel FD suite in crates/gradcheck)
test/gradcheck:
	$(CARGO_TEST) -p brain-gradcheck

# crates/rl's DPO/GRPO/distill gradchecks and its fast qwen3-gated
# integration tests never run in `make test`: `qwen3` is off by default on
# brain-rl (deliberately - crates/rl/Cargo.toml: a generic consumer of the
# crate must not be forced to link brain-qwen3), and every one of these
# targets carries `required-features = ["qwen3"]`, so `cargo test -p
# brain-rl` with no flag silently builds and runs NONE of them - not a
# reported skip, an absence. Without this target, "the objectives are
# gradient-checked" is true only of whatever tree a human last ran manually.
#
# Runs `continual_study`'s fast tests; its multi-seed experiment is
# `#[ignore]`d (minutes on one GPU, run it alone with `--ignored --nocapture`).
# Excludes `document_study`: documented in crates/rl/Cargo.toml as multi-minute,
# one-GPU, "run it alone" - batching it into a parallel lane would either
# serialise the whole target behind it or risk exactly the GPU-contention hangs
# `test/nextest`'s own doc comment above describes. Run it individually when
# touching that code, not as part of routine verification.
#
# A test that skips through `brain_testutil::gpu_tests_skipped` prints its
# `SKIP:` line, but libtest captures stdout/stderr from a PASSING test and
# discards it unless `--nocapture` is given - so the loud warning belongs here,
# at the one place a human actually looks.
## test/rl - crates/rl's qwen3-gated DPO/GRPO/distill gradchecks, GRPO correctness and continual-study integration tests
test/rl:
	@if [ -n "$$MOE_SKIP_GPU_TESTS" ]; then \
		echo "WARNING: MOE_SKIP_GPU_TESTS is set - every gradcheck below will report PASS without actually running its FD check. Unset it for a real result."; \
	fi
	$(CARGO_TEST) -p brain-rl --features qwen3 \
		--test grpo_gradcheck --test grpo_objective \
		--test dpo_gradcheck --test dpo_objective --test distill_gradcheck \
		--test distill_full_kl --test mixture_anchor_regression \
		--test qwen3_fit_weighted --test improve_cycle \
		--test grpo_fixes --test grpo_qwen35 --test improve_cycle_qwen35 \
		--test continual_study

# The training-capability conformance suite (model::conformance for the
# capability contracts, rl::conformance for the objectives) over every model
# registered in crates/rl/tests/conformance.rs. The binary asserts its registry
# is not empty, and naming a deleted test target fails the build, so the target
# cannot pass over zero models. A new model joins by one line in that file.
## test/conformance - the training-capability conformance suite over every registered model
test/conformance:
	$(CARGO_TEST) -p brain-rl --features qwen3 --test conformance

# Every workspace crate's tests on the CUDA backend (BRAIN_BACKEND=cuda), one
# crate at a time, as a per-crate JSON report. Slow: a build and a run of the
# whole workspace. `make test/cuda-matrix CRATES="qwen3 gpt2"` runs a few.
## test/cuda-matrix - every workspace crate's tests on the CUDA backend, one at a time (CRATES="a b" for a few)
test/cuda-matrix:
	scripts/test/cuda-matrix.py -o cuda-matrix.json $(CRATES)

# Cross-backend parity gate: CPU == Vulkan == NPU (gradcheck on both backends +
# direct CPU-vs-GPU forward parity + TTS NPU codec vs CPU reference).
## test/parity - default: cross-backend parity gate, CPU == Vulkan == NPU
##   (test/parity/strict, /forecast and /wan are separate)
test/parity:
	scripts/gates/parity-gate.sh

# Reference parity with skipping DISABLED.
# `test/parity/wan` generalized to every fixture-gated parity suite. Cargo
# reports a skipped test as a PASS, so a green `cargo test -p <crate>` over any
# of these is evidence of nothing on its own: the goldens are gitignored and the
# reference weights are named by environment variable, so on a box that has
# neither, the comparisons that matter quietly do not happen and the suite
# still says ok.
#
# BRAIN_REQUIRE_FIXTURES=1 turns every absent-fixture skip into a hard failure.
# Skips for absent HARDWARE (no discrete GPU, no NPU, no second card, no
# session bus, no SDL) are a different helper and stay skips - a fixture flag
# has nothing to say about missing silicon.
#
# RUN THIS WHERE THE CHECKPOINTS LIVE. The default list is the suites whose
# goldens `make testdata/fetch` provides, or that need no released checkpoint
# at all; several of them still want a reference-weight variable that only a
# machine holding the checkpoint can set, and each suite names its own in the
# message it used to skip with. Anywhere else it fails immediately, which is
# the honest answer rather than a green run that proved nothing.
#
# Override the list to gate one suite, or to add a suite whose fixtures are not
# mirrored (sdxl, controlnet, codeformer, vqgan, flux1, flux2, clip, sam2, ...):
#
#   make test/parity/strict PARITY_STRICT_SUITES="brain-clip:parity"
#
# brain-ltxv's audio_parity needs a real torchaudio (its golden dumper's own
# self-check imports it); on a host where torchaudio's compiled extension
# will not load, run its dumper with PYTHONPATH=tools/goldens/torchaudio_shim
# first (see that dumper's own module doc).
PARITY_STRICT_SUITES ?= \
        brain-deepseek2ocr:tiny_ref \
        brain-deepseekocr2:tiny_ref \
        brain-t5encoder:tiny_ref \
        brain-diffusion:discrete_parity,klein_schedule,wan_schedule_parity \
        brain-data:clip_tokenizer_parity \
        brain-lfm2:parity \
        brain-qwen3:encoder_parity \
        brain-vae:decode_parity,encode_parity \
        brain-s3dit:block_parity,dev_parity,model_parity,real_parity \
        brain-rrdbnet:parity \
        brain-wan:dit_parity,vae_parity \
        brain-ltxv:vae_parity,audio_parity,upsampler_parity,duration_head_parity,na_decoder_parity \
        brain-gemma4:parity \
        brain-mimi:decode,encode \
        brain-ecapatdnn:encoder \
        brain-timesfm3:parity

## test/parity/strict - reference-parity suites with skips DISABLED (needs the fixtures + checkpoints;
##   PARITY_STRICT_SUITES=...)
test/parity/strict:
	@fail=0; \
	for suite in $(PARITY_STRICT_SUITES); do \
	    crate=$${suite%%:*}; tests=$${suite#*:}; args=""; \
	    for t in $$(echo $$tests | tr ',' ' '); do args="$$args --test $$t"; done; \
	    echo "=== $$crate$$args (BRAIN_REQUIRE_FIXTURES=1) ==="; \
	    if BRAIN_REQUIRE_FIXTURES=1 $(CARGO_TEST) -p $$crate $$args; \
	    then echo "  PASS: $$crate"; else echo "  FAIL: $$crate"; fail=1; fi; \
	done; \
	if [ $$fail -eq 0 ]; then echo "PARITY (STRICT): PASS - every comparison actually ran"; \
	else echo "PARITY (STRICT): FAIL - a suite either regressed or could not run its comparisons"; fi; \
	exit $$fail

# Forecasting correctness gate: every time-series optimization stays fp32-exact
# (kronos KV-cache/shared-prefill/cross-section + batched-training parity).
## test/parity/forecast - forecasting fp32-exactness gate
test/parity/forecast:
	scripts/gates/forecast-parity-gate.sh

# The Wan parity suite with skipping DISABLED. `cargo test -p brain-wan`
# reports a skipped comparison as a pass, so a green run there proves nothing
# about parity on its own - the weights come from the environment, and without
# them most of the real-checkpoint comparisons quietly do not happen. This is
# the form to run when the claim matters.
## test/parity/wan - Wan parity vs the reference, skips DISABLED (BRAIN_WAN_* env)
test/parity/wan:
	@test -n "$(BRAIN_WAN_DIT)" || (echo "set BRAIN_WAN_DIT/_VAE (and BRAIN_TESTDATA if the goldens are not in ./testdata)"; exit 2)
	BRAIN_REQUIRE_FIXTURES=1 $(CARGO_TEST) -p brain-wan --test dit_parity --test vae_parity

## test/e2e - default: every fast end-to-end bats suite (api-conformance, shutdown, samples, ready)
##   Opt-in and heavier: claude-code, scheduler, quickstart, deb, results-hygiene, s3dit-int8.
test/e2e: test/e2e/api-conformance test/e2e/shutdown test/e2e/samples test/e2e/ready

# End-to-end: HTTP API conformance over a real socket against a single `brain serve`
# backed by the built-in deterministic mock model (BRAIN_MOCK=1) - no weights, no GPU,
# no `claude`. Validates every provider dialect (OpenAI/Anthropic/OpenRouter) against
# the vendored OpenAPI specs. Fast + deterministic. Needs only a debug/release brain
# binary + jq (+ optional Python jsonschema for full schema validation).
#   make test/e2e/api-conformance   (or: BRAIN_BIN=./target/debug/brain bats tests/e2e/api_conformance.bats)
## test/e2e/api-conformance - HTTP API conformance of every provider dialect against the mock model
test/e2e/api-conformance: build/debug
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/api_conformance.bats

# End-to-end: `brain serve` actually stops on SIGINT/SIGTERM, for every combination
# of surfaces it can be told to serve (D-Bus alone, D-Bus+HTTP together, HTTP
# alone). Each test starts and kills its own server; the D-Bus cases use a private
# per-test dbus-daemon, never the real session/system bus. Needs a debug/release
# binary + a working dbus-daemon.
## test/e2e/shutdown - `brain serve` stops on SIGINT/SIGTERM for every surface combination
test/e2e/shutdown: build/debug
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/shutdown.bats

# End-to-end: every sample script under samples/{shell,python}/ is actually
# exercised - the harness that did not exist when they all silently rotted
# after the P19 brain-py rewrite. ONE shared BRAIN_MOCK=1 server (D-Bus +
# Anthropic HTTP); each sample that CAN run against the weight-free mock does
# so for real, the rest skip honestly with a printed reason. A completeness
# check (tests/e2e/samples_manifest.bats, not gated behind this harness's own
# skip conditions) fails the suite if a tracked sample script is missing from
# tests/e2e/samples/manifest.tsv (or vice versa), so a new, unwired sample
# cannot silently rot the way these did. Needs a debug/release binary,
# dbus-daemon, curl, and `pip install -e brain-py` (jeepney) on EXAMPLES_PY
# (default python3).
#   make test/e2e/samples   (or: EXAMPLES_PY=/path/to/python3 bats tests/e2e/samples.bats tests/e2e/samples_manifest.bats)
## test/e2e/samples - every shell/python sample script against one shared mock server
test/e2e/samples: build/debug
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/samples.bats tests/e2e/samples_manifest.bats

# End-to-end: `brain serve --ready-file PATH` fires only once EVERY requested
# surface (HTTP dialects + D-Bus) has actually bound, never on a failed or
# partial bind, and strictly AFTER --api-keys-out is written - so a script can
# wait on PATH alone and then read the keys with no retry. BRAIN_MOCK=1,
# CPU-only, no real weights. Needs a debug/release binary + jq + curl (+
# dbus-daemon/busctl for the D-Bus cases, which skip cleanly without them).
## test/e2e/ready - `brain serve --ready-file` fires only after every surface has bound
test/e2e/ready: build/debug
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/ready.bats

# End-to-end: drive the real `claude` CLI against a local `brain serve --anthropic`
# on the BRAIN_MOCK model, proving brain works as a Claude Code backend. Skips
# cleanly unless `claude` is installed.
## test/e2e/claude-code - OPT-IN: drive the real `claude` CLI against a local `brain serve --anthropic`
test/e2e/claude-code: build/release
	bats tests/e2e/claude_code.bats

# Heavy, opt-in: brain's residency scheduler (batching/eviction) + the generate ->
# detect -> annotate demo against REAL model weights and a GPU. NOT part of
# test/e2e (that's test/e2e/samples' job, against the mock) - see
# tests/e2e/scheduler.bats for the required env vars.
## test/e2e/scheduler - OPT-IN, heavy: residency scheduler + generate/detect/annotate demo on REAL weights
test/e2e/scheduler:
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/scheduler.bats

# Heavy, opt-in, same shape as test/e2e/scheduler: re-runs README.md's Quick
# start against REAL auto-fetched weights (tens of GB) and asserts the
# cross-model agreement it claims (yolo/qwen3vl on the same generated image,
# the TTS->ASR->LLM round trip). NOT part of test/e2e. Run `make
# docs/quickstart` first (or BRAIN_QUICKSTART_E2E=1 scripts/demo/quickstart.sh)
# to populate docs/getting-started/img/, which this asserts against rather than
# re-fetching itself.
## test/e2e/quickstart - OPT-IN, heavy: README Quick start against REAL auto-fetched weights
test/e2e/quickstart: build/release
	BRAIN_QUICKSTART_E2E=1 BRAIN_BIN=$(BRAIN) bats tests/e2e/quickstart.bats

# End-to-end: the release .deb is a real, correctly-formed Debian package -
# Depends actually derived from the binary, a real Installed-Size, md5sums for
# every shipped file, and the debug flavor never collides with it. Needs
# dpkg-deb + both flavors built.
## test/e2e/deb - OPT-IN: the release .deb is a correctly-formed Debian package (needs dpkg-deb)
test/e2e/deb: package/deb/release package/deb/debug
	bats tests/e2e/deb.bats

# Regression guard for `make perf/compare` / `make bench/compare`'s file
# enumeration. It moves results/ aside while it runs and restores it after.
## test/e2e/results-hygiene - OPT-IN: perf/bench compare picks up result files with awkward names
test/e2e/results-hygiene: build/release
	bats tests/e2e/results_hygiene.bats

# Real end-to-end s3dit (Z-Image) int8 generation (256x256) against the fetched
# checkpoint under out/models/ - the no-OOM regression run from fa7b576.
# Heavy: real weights + a GPU; writes results/zimage-int8-e2e.log.
## test/e2e/s3dit-int8 - OPT-IN, heavy: real z-image (s3dit) int8 generation, the no-OOM regression run
test/e2e/s3dit-int8:
	bash scripts/run_s3dit_int8_e2e.sh

## check - lists the check targets; runs nothing (test/full runs them all)
check:
	$(LIST_CHILDREN)

# Environment variables are being phased out in favour of CLI flags: every env
# read under crates/ is on scripts/gates/env-reads.allow and the list only
# shrinks. See scripts/gates/check-env-reads.sh.
## check/env-reads - no new environment reads under crates/ (add a flag); allowlist entries must still exist
check/env-reads:
	bash scripts/gates/check-env-reads.sh

# The process environment is shared by every thread: library code that sets a
# variable around a build lets a concurrent build read it. Writes outside test
# code are listed per file, with a reason, in scripts/gates/env-writes.allow;
# the list only shrinks. See scripts/gates/check-env-writes.sh.
## check/env-writes - no environment writes (set_var/remove_var) in library code outside the reasoned allowlist; entries must still exist
check/env-writes:
	bash scripts/gates/check-env-writes.sh

# A test that returns green without running its check certifies nothing:
# every skip goes through brain_testutil (or #[ignore]), and the legacy ones on
# scripts/gates/test-skips.allow only shrink. See scripts/gates/check-test-skips.sh.
## check/test-skips - no new silent test skips under crates/ (MOE_SKIP_GPU_TESTS early return, bare "skipping" + return); allowlist entries must still exist
check/test-skips:
	bash scripts/gates/check-test-skips.sh

# Every `impl PagedDecoder for <Engine>` (an engine `brain serve` schedules) runs the
# generic conformance suite model::serve::conformance (continuous batching, no
# head-of-line blocking, prefix reuse, adapter KV isolation) or states why not.
# See scripts/gates/check-serve-conformance.sh.
## check/serve-conformance - every serving engine runs the generic conformance suite (or is exempt with a reason)
check/serve-conformance:
	bash scripts/gates/check-serve-conformance.sh

# Every default seed is random. A literal seed in production code (Rust, Python
# `--seed` defaults, Makefile `SEED ?=`) must be on
# scripts/gates/seed-literals.allow.d/<crate-or-dir>.allow with a reason (salt,
# world constant, frozen artifact, greedy path, test double, bench input) or a
# `todo:` that a later change removes. See scripts/gates/check-seed-literals.sh.
## check/seed-literals - no new literal seeds in production code (draw a random one); allowlist entries must still exist
check/seed-literals:
	bash scripts/gates/check-seed-literals.sh

# Test code may fix a seed only where determinism is necessary, with a
# `// seed: <reason>` on the line or the line above. Not part of `check` yet:
# it lists the test literals that still lack their reason.
## check/seed-literals/tests - OPT-IN: every literal seed in a test carries a `// seed:` reason
check/seed-literals/tests:
	bash scripts/gates/check-seed-literals.sh --tests

# Spec for the gate above, run against throwaway trees (needs only bats).
## test/gates - default: every gate self-test (env-reads, env-writes, test-skips, seed-literals, group-kernels, crate-layers, whole-tensor-loads, format-logic, adapter-naming, placement-accounting, training-capabilities, model-flags, tier-gemm)
test/gates: test/gates/env-reads test/gates/env-writes test/gates/test-skips test/gates/seed-literals test/gates/group-kernels test/gates/crate-layers test/gates/whole-tensor-loads test/gates/format-logic test/gates/adapter-naming test/gates/placement-accounting test/gates/training-capabilities test/gates/model-flags test/gates/tier-gemm

## test/gates/test-skips - the silent-skip ratchet fails on a new bare skip and on a stale entry, passes on a match
test/gates/test-skips:
	bash scripts/gates/test-skips-spec.sh

## test/gates/env-reads - the env-read ratchet fails on a new read and on a stale entry, passes on a match
test/gates/env-reads:
	bats tests/gates/env_reads.bats

## test/gates/env-writes - the env-write ratchet fails on a new write, a missing reason and a stale entry, passes on a match, ignores test code
test/gates/env-writes:
	bats tests/gates/env_writes.bats

## test/gates/seed-literals - the seed-literal ratchet fails on a new literal, a stale entry and (in test mode) a bare test seed
test/gates/seed-literals:
	bats tests/gates/seed_literals.bats

## test/gates/crate-layers - the layer map gate fails on an unassigned package, an upward dependency, a stale allowlist entry and a drifted crate table
test/gates/crate-layers:
	bats tests/gates/crate_layers.bats

## test/gates/whole-tensor-loads - the whole-tensor load ratchet fails on a new whole-tensor call and on a stale entry, ignores bounded readers, tests and checkpoint internals
test/gates/whole-tensor-loads:
	bats tests/gates/whole_tensor_loads.bats

## test/gates/format-logic - the format-logic ratchet fails on Weight-variant, kernel-lookup and codec calls in a layer-4 crate and on a stale entry, ignores layer 3, tests and comments
test/gates/format-logic:
	bats tests/gates/format_logic.bats

## test/gates/tier-gemm - the tier-coverage gate fails on a format without a row, an unrecorded or stale gap and a missing kernel file
test/gates/tier-gemm:
	bats tests/gates/tier_gemm.bats

## test/gates/adapter-naming - the adapter-naming ratchet fails on a .lora_a/.lora_b string outside model::adapter and on a stale entry, ignores comments and tests
test/gates/adapter-naming:
	bats tests/gates/adapter_naming.bats

## test/gates/placement-accounting - the placement-accounting ratchet fails on a new caller of an fp32-element sizing helper and on a stale entry, ignores the helper's own file, comments and tests
test/gates/placement-accounting:
	bats tests/gates/placement_accounting.bats

## test/gates/training-capabilities - the capability page is generated from its declaration; a missing or ignored test, an unregistered tier and a drifted page fail
test/gates/training-capabilities:
	bats tests/gates/training_capabilities.bats

## test/gates/model-flags - the model-flag gate fails on an architecture-prefixed flag (incl. a version-less stem) and a stale alias entry, ignores generic flags, tool crates and comments
test/gates/model-flags:
	bats tests/gates/model_flags.bats

## test/gates/group-kernels - the kernel-table grouping sums to the table total and bills unknown kernels to misc
test/gates/group-kernels:
	bats tests/gates/group_kernels.bats

# Runs BEFORE the clippy gate: if a manifest is broken, cargo fails at
# manifest-parse time with an error naming an innocent crate, and every other
# gate inherits that confusion. Cheap enough to front every slow gate with.
## check/workspace - every crates/* manifest parses (names the broken file, not an innocent crate)
check/workspace:
	@scripts/gates/check-workspace-members.sh

# The clippy gate - a ratchet. Checks that clippy EXITS 0 (it stops at the first
# deny-by-default lint and then silently reports nothing about everything after
# it) and that the warning count has not grown. See scripts/gates/clippy-gate.sh.
## check/clippy - the clippy ratchet gate (exit 0 + no new warnings)
check/clippy: check/workspace
	@scripts/gates/clippy-gate.sh

# Self-validation for scripts/ and tools/: every one parses, every one is named
# somewhere else in the repo (Makefile target / bats test / crate doc comment /
# doc - an orphan gate), and no non-overridable absolute machine path. See
# scripts/gates/check-scripts.sh for the full rationale. check-env-docs.sh
# additionally requires every BRAIN_* env var read anywhere in crates/ to be
# documented in docs/reference/configuration.md, a docs/reference/models/<model>.md page, or
# .agents/rules/testing.md (env-only config MUST have a reference).
# check-no-doc-citations.sh additionally requires that crates/, scripts/,
# tools/, and samples/ never cite a docs/ or .agents/ file path - see that
# script for why (also wired as a pre-commit hook, so this is a slow-path
# backstop for anything pre-commit was bypassed for).
# check-no-perf-numbers.sh additionally denies a bare number next to a
# performance unit/claim (ms, s, fps, tok/s, % of peak, Nx speedup, ...)
# anywhere in docs/**/*.md unless reviewed via a `<!-- perf-number: ... -->`
# comment - see that script for the full rationale and escape-hatch syntax.
# check-measured-citations.sh additionally requires every backticked path,
# test/bench/example name or make target that a doc cites after "measured by/
# with/via/using" (or "measured in/on") to exist in the tree - see
# scripts/gates/measured-citations.py for the exact definition.
# check-device-env-single-source.sh additionally requires BRAIN_DEVICE be read
# via std::env::var in exactly one file (crates/gpu-core/src/devices.rs's
# ambient_compute_set) rather than a second, independently-drifting parser -
# see A4 in .agents/roadmap/dtype.md for the bug this keeps fixed.
# check-arch-names.sh additionally requires every crates/arch architecture
# row to name a real, correspondingly-named crate (bar an explicit, dated
# exception list), the CLI's top-level dispatch to never hard-code an
# architecture name as its own match arm, and every non-toy architecture to
# have (and every docs/reference/models/*.md page to BE) a real registry id in both
# directions - see that script for the full rationale.
# check-linear-history.sh fails if HEAD's history contains a merge commit -
# the working branch must always be rebased, never merged into.
# check-kernel-selection.sh cross-references docs/reference/kernels/index.md for
# every kernel that has a structurally faster sibling (a `_rows`/`_wg`/
# `_reg*`/`_tiled`/`_part`/`_dyn` suffix, or a higher `@opt` sibling of the
# same stem) and fails if it is dispatched by name outside a
# `backend_api::select` seam and outside its own reasoned allow-list - see
# that script for the full rationale and the M1.3 entry in
# .agents/roadmap/kernel-performance.md for the inventory it produced.
# check-crate-layers.sh additionally requires every crate that exists so a
# MODEL crate can depend on it to keep a dependency closure within layers 1-3
# - the layering in .agents/rules/architecture.md is otherwise unenforced, and
# one violation there is not a style problem but a hard Cargo cycle that only
# surfaces at the next call site.
# check-cancellable-actions.sh is a RATCHET, not a rule: every action is handed
# a cancel token whether or not it honours one, so a caller that builds request
# lifetime on cancellation cannot tell from the manifest which models will
# actually stop. The script carries the list of models that knowingly ignore it
# and fails when a new streaming action joins them (or when a listed one starts
# polling and the list goes stale).
## check/scripts - scripts/tools self-validation + env-var doc gate + no-doc-citations
##   + no-perf-numbers + measured-citations + architecture, layering and link gates
check/scripts:
	bash scripts/gates/check-scripts.sh
	bash scripts/gates/check-env-docs.sh
	bash scripts/gates/check-no-doc-citations.sh
	bash scripts/gates/check-no-perf-numbers.sh
	bash scripts/gates/check-measured-citations.sh
	bash scripts/gates/check-device-env-single-source.sh
	bash scripts/gates/check-arch-names.sh
	bash scripts/gates/check-linear-history.sh
	bash scripts/gates/check-golden-source.sh
	bash scripts/gates/check-doc-links.sh
	bash scripts/gates/check-kernel-selection.sh
	bash scripts/gates/check-multi-gpu-sharding.sh
	bash scripts/gates/check-crate-layers.sh
	bash scripts/gates/check-placement-accounting.sh
	bash scripts/gates/check-training-capabilities.sh
	bash scripts/gates/check-model-flags.sh
	bash scripts/gates/check-adapter-naming.sh
	bash scripts/gates/check-format-logic.sh
	bash scripts/gates/check-tier-gemm.sh
	bash scripts/gates/check-whole-tensor-loads.sh
	bash scripts/gates/check-cancellable-actions.sh

# SPDX/copyright header gate: every Rust/C/Python/shell/Makefile/WGSL/...
# source file must carry exactly one "SPDX-License-Identifier: Apache-2.0"
# line, immediately followed by the copyright line. scripts/spdx/rules.py has
# the file-selection rules (shared with scripts/spdx/check.py); `make
# hooks/install` wires the same check into a git pre-commit hook so a
# non-compliant commit is refused locally, not just caught here.
## check/spdx - SPDX-License-Identifier + copyright header gate
check/spdx:
	python3 scripts/spdx/check.py $$(git ls-files)

## check/paths - no baked-in absolute machine paths under crates/
check/paths:
	bash scripts/gates/check-no-machine-paths.sh

# Repository scope (AGENTS.md "Repository scope"): a tracked file names another
# project only if brain depends on it; CHANGELOG.md is history and exempt.
## check/scope - no tracked file names a project brain does not depend on
check/scope:
	bash scripts/gates/check-repo-scope.sh

# Saved-activation coverage: every crate that defines a backward lets its training run choose what the backward
# keeps (model::saved), or is listed in scripts/gates/saved-policy.allow with the reason it needs none.
## check/saved-policy - every backward takes the saved-activation policy (or states why it needs none)
check/saved-policy:
	bash scripts/gates/check-saved-policy.sh

# AGENTS.md is read by every agent session: a line budget keeps detail in the
# file that owns it, and every path it names must exist.
## check/agents-md - AGENTS.md is within its line budget and every path it names exists
check/agents-md:
	bash scripts/gates/check-agents-md.sh

## check/hf-fixtures - every architecture declaring HuggingFace names has a tiny HF-layout fixture test (or a shrinking allowlist entry)
check/hf-fixtures:
	bash scripts/gates/check-hf-fixtures.sh

## check/files - no video, model weights, or oversized files in git
check/files:
	bash scripts/gates/check-large-files.sh

# Enforces samples/README.md: SDK-only brain dependency, declared surfaces, a
# path-derived package name, SPDX headers - and the two measured properties,
# that a sample's dependency CLOSURE contains nothing from a surface it did not
# enable, and that rebuilding it after an edit compiles exactly one crate.
## check/samples - the samples contract: SDK-only dependency, declared surfaces, naming, SPDX
check/samples:
	bash scripts/gates/check-samples.sh

# The SDK is the workspace's feature vocabulary; this keeps that vocabulary
# honest (surface names, `default` == `full`, no value-shaped feature names,
# and every surface compiling on its own).
## check/sdk-features - the SDK feature vocabulary stays honest (every surface compiles on its own)
check/sdk-features:
	bash scripts/gates/check-sdk-features.sh

# The Makefile's public targets form a tree that `make <TAB>` reveals one level
# at a time; this gate enforces the naming, the doc lines and the sections.
## check/make-targets - make targets form a documented concept/subconcept/action tree
check/make-targets:
	bash scripts/gates/check-make-targets.sh

# Checked-in files that are generated from the tree. Each `check/generated/*`
# target fails when the file has drifted from what its `generate/*` twin would
# write; a stale table is wrong silently, which no accuracy metric shows.
## check/generated - default: every generated-file drift check below (read-only)
check/generated: check/generated/kernels-wgsl-table check/generated/kernels-cuda-table check/generated/samples-manifest check/generated/knowledge-index check/generated/roadmap-index

# Every column of docs/reference/kernels/index.md is derived from the sources, so the
# table cannot be edited by hand - and this check is what stops it drifting
# silently, which is the failure mode .agents/knowledge/ #29 records for the
# registry generator.
## check/generated/kernels-wgsl-table - docs/reference/kernels/index.md matches crates/kernels/wgsl/
check/generated/kernels-wgsl-table:
	scripts/build/gen-kernel-table.py --check

## check/generated/kernels-cuda-table - the CUDA catalogue matches crates/kernels-cuda, and no .cu is unregistered
check/generated/kernels-cuda-table:
	scripts/build/gen-cuda-kernel-table.py --check

## check/generated/samples-manifest - docs/manifest.yaml's Examples part matches the READMEs under samples/
check/generated/samples-manifest:
	scripts/build/gen-samples-manifest.py --check

# /check also fails when two entries claim one citation id.
## check/generated/knowledge-index - .agents/knowledge/index.md matches the findings beside it
check/generated/knowledge-index:
	scripts/build/gen-knowledge-index.py --check

## check/generated/roadmap-index - .agents/roadmap/index.md matches the ledgers beside it
check/generated/roadmap-index:
	scripts/build/gen-roadmap-index.py --check

# ---------------------------------------------------------------------------
##@ Models and datasets
## One area per model family or pipeline: dataset generation, training,
## evaluation, inference demos. Heavy ones need weights and say so.

# Synthetic and downloaded char datasets for the GPT baseline. A dataset has
# three actions: `data` generates it, `train` trains a GPT on it and `eval`
# scores the result (perplexity + exact-match). Target names use hyphens; the
# dataset directory and generator name keep the underscore (shakespeare_char).
# LHS=RHS datasets train with answer-masking; shakespeare-char does not.
GPT_SYNTHETIC := calculator reverser wordcalc
GPT_DATASETS  := $(GPT_SYNTHETIC) timeseries shakespeare-char gpt
GPT_DATASET_TARGETS := $(addprefix gpt/,$(GPT_DATASETS))
GPT_SYNTHETIC_DATA_TARGETS := $(addsuffix /data,$(addprefix gpt/,$(GPT_SYNTHETIC)))
GPT_TRAIN_TARGETS := $(addsuffix /train,$(GPT_DATASET_TARGETS))
GPT_EVAL_TARGETS  := $(addsuffix /eval,$(GPT_DATASET_TARGETS))
# The dataset's on-disk name for a target stem.
gpt_dataset = $(subst -,_,$*)
MASK_calculator := --mask =
MASK_reverser   := --mask =
MASK_wordcalc   := --mask =

## gpt - lists the GPT baseline datasets; does nothing
gpt:
	$(LIST_CHILDREN)

## gpt/<dataset> - lists a dataset's actions (data, train, eval); does nothing
##   Datasets: calculator, reverser, wordcalc, timeseries, shakespeare-char, gpt.
$(GPT_DATASET_TARGETS):
	$(LIST_CHILDREN)

## gpt/<dataset>/data - generate the dataset into data/<dataset>
##   (shakespeare-char and gpt download the tiny-shakespeare text first)
$(GPT_SYNTHETIC_DATA_TARGETS): gpt/%/data: build/release
	$(BRAIN) data gen $* --out $(DATA)/$* --n $(N) $(SEED_FLAG)

gpt/timeseries/data: build/release
	$(BRAIN) data gen timeseries --out $(DATA)/timeseries --n 200000 $(SEED_FLAG)

$(DATA)/shakespeare_char/input.txt:
	mkdir -p $(DATA)/shakespeare_char && curl -sSL -o $@ $(SHAKE_URL)
gpt/shakespeare-char/data: build/release $(DATA)/shakespeare_char/input.txt
	$(BRAIN) data gen shakespeare_char --out $(DATA)/shakespeare_char

$(DATA)/gpt/input.txt:
	mkdir -p $(DATA)/gpt && curl -sSL -o $@ $(SHAKE_URL)
gpt/gpt/data: build/release $(DATA)/gpt/input.txt
	$(BRAIN) data gen gpt --out $(DATA)/gpt

## gpt/<dataset>/train - train a GPT on the dataset -> out/gpt-<dataset>.safetensors
$(GPT_TRAIN_TARGETS): gpt/%/train: build/release
	@mkdir -p $(OUT)
	$(BRAIN) gpt2 train $(DATA)/$(gpt_dataset) --out $(OUT)/gpt-$(gpt_dataset).safetensors \
		--steps $(STEPS) --batch $(BATCH) --block $(BLOCK) \
		--layers $(LAYERS) --d-model $(DMODEL) --heads $(HEADS) --lr $(LR) \
		$(SEED_FLAG) $(MASK_$(gpt_dataset))

## gpt/<dataset>/eval - perplexity + exact-match of a trained GPT
$(GPT_EVAL_TARGETS): gpt/%/eval: build/release
	$(BRAIN) gpt2 eval --weights $(OUT)/gpt-$(gpt_dataset).safetensors --data $(DATA)/$(gpt_dataset)

# ---- YOLO detector (synthetic detection dataset) --------------------------
# `make yolo/data` generates a synthetic object-detection dataset (RGB shapes
# + exact GT boxes) at the tiny YOLO's native 128px geometry. `yolo/train`
# trains the tiny detector on it; `yolo/eval` reports mAP@0.5/precision/recall;
# `yolo/detect` runs inference on a sample image (dataset image 0) and prints the
# boxes as JSON lines. All CPU-friendly (the YOLO model runs on the CPU backend).
## yolo - lists the YOLO detector targets; does nothing
yolo:
	$(LIST_CHILDREN)

## yolo/data - synthetic object-detection dataset (shapes + exact boxes) -> data/detect
yolo/data: build/release
	$(BRAIN) data gen detect --out $(DATA)/detect --n $(YOLO_N) $(SEED_FLAG)

## yolo/train - train the tiny YOLO on it -> out/yolo.safetensors
yolo/train: build/release
	@mkdir -p $(OUT)
	$(BRAIN) yolov8 train $(DATA)/detect --out $(OUT)/yolo.safetensors \
		--steps $(YOLO_STEPS) --batch $(YOLO_BATCH) --lr $(YOLO_LR) $(SEED_FLAG)

## yolo/eval - mAP@0.5 + precision/recall of the trained YOLO
yolo/eval: build/release
	$(BRAIN) yolov8 eval --weights $(OUT)/yolo.safetensors --data $(DATA)/detect \
		--conf $(YOLO_CONF) --iou $(YOLO_IOU)

## yolo/detect - run detection on a sample image (JSON boxes)
yolo/detect: build/release
	$(BRAIN) yolov8 detect --weights $(OUT)/yolo.safetensors --image $(DATA)/detect \
		--conf $(YOLO_CONF) --iou $(YOLO_IOU)

# ---- Intel NPU deployment (OpenVINO) --------------------------------------
# Quantize the trained YOLO to INT8 and compile it to a real NPU graph.
# `yolo/npu/export` and `yolo/npu/quantize` are PURE RUST (run on any machine);
# `yolo/npu/sim-int8` measures fp32-vs-INT8 mAP with NO NPU. `yolo/npu/run` and
# `yolo/npu/bench` REQUIRE OpenVINO 2024.x + an Intel NPU (3720 / Meteor Lake) at
# run time - they are NOT part of `make build/debug`/`make test`. The NPU is a
# whole-graph compiler, separate from --device cpu|gpu; see docs/yolo/NPU.md.
ONNX        ?= $(OUT)/yolo.onnx
ONNX_INT8   ?= $(OUT)/yolo.int8.onnx
NPU_DEVICE  ?= NPU
NPU_CACHE   ?= $(OUT)/npu-cache
NPU_CALIB   ?= $(DATA)/detect
NPU_NCALIB  ?= 256

## yolo/npu - lists the YOLO INT8 / Intel NPU pipeline; does nothing
yolo/npu:
	$(LIST_CHILDREN)

## yolo/npu/export - export the trained YOLO to ONNX (pure Rust, any machine)
yolo/npu/export: build/release
	@mkdir -p $(OUT)
	$(BRAIN) npu export --weights $(OUT)/yolo.safetensors --out $(ONNX)

## yolo/npu/quantize - quantize the trained YOLO to INT8 ONNX (pure Rust, any machine)
yolo/npu/quantize: build/release
	@mkdir -p $(OUT)
	$(BRAIN) npu quantize --weights $(OUT)/yolo.safetensors --calib $(NPU_CALIB) \
		--out $(ONNX_INT8) --num-calib $(NPU_NCALIB) --scales-out $(OUT)/yolo.scales.json

## yolo/npu/sim-int8 - measure fp32-vs-INT8 mAP with no NPU
yolo/npu/sim-int8: build/release
	$(BRAIN) npu sim --weights $(OUT)/yolo.safetensors --data $(DATA)/detect \
		--calib $(NPU_CALIB) --num-calib $(NPU_NCALIB) --conf $(YOLO_CONF) --iou $(YOLO_IOU)

## yolo/npu/run - run the INT8 graph on the NPU (needs OpenVINO + an Intel NPU)
yolo/npu/run: build/release
	$(BRAIN) npu run --onnx $(ONNX_INT8) --image $(DATA)/detect --device $(NPU_DEVICE) \
		--cache-dir $(NPU_CACHE) --conf $(YOLO_CONF) --iou $(YOLO_IOU)

## yolo/npu/bench - benchmark the INT8 graph on the NPU (needs OpenVINO + an Intel NPU)
yolo/npu/bench: build/release
	$(BRAIN) npu bench --onnx $(ONNX_INT8) --device $(NPU_DEVICE) \
		--cache-dir $(NPU_CACHE) --hint throughput --iters 200 --warmup 20

# ---- Qwen LoRA fine-tuning: one command to fully retrain and overwrite the
# LoRA checkpoint. DATASET is a bench `datasets build` output dir (train.jsonl [+ validation.jsonl]);
# ADAPTER is OWNER/NAME[:TAG] (TAG defaults to "latest" and is OVERWRITTEN on
# every rerun -- that's the "retrain and overwrite" part). QWEN_BASE is a model
# store ref (default the published Qwen3-0.6B) or a direct .safetensors path.
QWEN_BASE  ?= Qwen/Qwen3-0.6B
LORA_RANK  ?= 8
LORA_ALPHA ?= 16
QWEN_STEPS ?= 500
QWEN_LR    ?= 5e-5
QWEN_BATCH ?= 4
QWEN_BLOCK ?= 1024

## qwen - lists the Qwen targets; does nothing
qwen:
	$(LIST_CHILDREN)

## qwen/finetune - LoRA-finetune a Qwen checkpoint (DATASET=<dir> ADAPTER=<owner/name[:tag]>)
qwen/finetune: build/release
	@test -n "$(DATASET)" || (echo "usage: make qwen/finetune DATASET=<dir> ADAPTER=<owner/name[:tag]>" && exit 1)
	@test -n "$(ADAPTER)" || (echo "usage: make qwen/finetune DATASET=<dir> ADAPTER=<owner/name[:tag]>" && exit 1)
	$(BRAIN) qwen3 finetune --lora $(LORA_RANK) --alpha $(LORA_ALPHA) \
		--weights $(QWEN_BASE) --adapter $(ADAPTER) --dataset $(DATASET) \
		--steps $(QWEN_STEPS) --lr $(QWEN_LR) --batch $(QWEN_BATCH) --block $(QWEN_BLOCK) \
		$(SEED_FLAG) $(if $(MODELS_DIR),--models-dir $(MODELS_DIR)) $(if $(DATASET_ID),--dataset-id $(DATASET_ID))

# ---- depth (ZipDepth) ------------------------------------------------------
# Set ZIPDEPTH_PTH to a released checkpoint (see resources/depth-models).
ZIPDEPTH_PTH ?=
DEPTH_IMG    ?=

## depth - lists the ZipDepth monocular-depth targets; does nothing
depth:
	$(LIST_CHILDREN)

## depth/demo - depth map of one image in a window (ZIPDEPTH_PTH=<ckpt> DEPTH_IMG=<image.ppm>)
depth/demo: build/release
	@test -n "$(ZIPDEPTH_PTH)" || (echo "set ZIPDEPTH_PTH=<zipdepth_base.pth>"; exit 2)
	@test -n "$(DEPTH_IMG)"    || (echo "set DEPTH_IMG=<image.ppm>"; exit 2)
	$(BRAIN) zipdepth --image $(DEPTH_IMG) --weights $(ZIPDEPTH_PTH)

## depth/smoke - headless depth map of one image -> out/depth.ppm (ZIPDEPTH_PTH, DEPTH_IMG)
depth/smoke: build/release
	@test -n "$(ZIPDEPTH_PTH)" || (echo "set ZIPDEPTH_PTH=<zipdepth_base.pth>"; exit 2)
	@test -n "$(DEPTH_IMG)"    || (echo "set DEPTH_IMG=<image.ppm>"; exit 2)
	DISPLAY= $(BRAIN) zipdepth --image $(DEPTH_IMG) --weights $(ZIPDEPTH_PTH) \
		--headless --out $(OUT)/depth.ppm

## depth/camera - live depth map from a camera (ZIPDEPTH_PTH=<ckpt>, DEPTH_ARGS=...)
depth/camera: build/release
	@test -n "$(ZIPDEPTH_PTH)" || (echo "set ZIPDEPTH_PTH=<zipdepth_base.pth>"; exit 2)
	$(BRAIN) zipdepth --camera --weights $(ZIPDEPTH_PTH) $(DEPTH_ARGS)

# Train ZipDepth end to end on the synthetic RGB->depth pairs (placeholder data,
# real loop: forward -> masked L1 -> backward -> AdamW; loss printed per step).
# Fine-tune a released checkpoint instead with ZIPDEPTH_PTH set.
## depth/train - train ZipDepth on synthetic RGB->depth pairs (fine-tune a release with ZIPDEPTH_PTH)
depth/train: build/release
	$(BRAIN) zipdepth train --out $(OUT)/zipdepth.safetensors --steps 50 --batch 2 \
		$(if $(ZIPDEPTH_PTH),--weights $(ZIPDEPTH_PTH),)

# WorldMirror-2 (multi-view 3D reconstruction). MIRROR_CKPT = the reference
# model.safetensors (or its HF dir); the converted .safetensors is what infer uses.
MIRROR_CKPT    ?=
MIRROR_WEIGHTS ?= $(OUT)/mirror.safetensors
# images -> 3DGS scene (+ view). MIRROR_IMAGES = dir of .ppm or comma list.
MIRROR_IMAGES ?=

## mirror - lists the WorldMirror-2 multi-view reconstruction targets; does nothing
mirror:
	$(LIST_CHILDREN)

## mirror/import - convert the reference checkpoint (MIRROR_CKPT=<model.safetensors|hf_dir>)
mirror/import: build/release
	@test -n "$(MIRROR_CKPT)" || (echo "set MIRROR_CKPT=<model.safetensors|hf_dir>"; exit 2)
	$(BRAIN) worldmirror2 import $(MIRROR_CKPT) --out $(MIRROR_WEIGHTS)

## mirror/infer - images -> 3DGS scene (MIRROR_IMAGES=<dir|a.ppm,b.ppm>, MIRROR_ARGS=...)
mirror/infer: build/release
	@test -n "$(MIRROR_IMAGES)" || (echo "set MIRROR_IMAGES=<dir|a.ppm,b.ppm>"; exit 2)
	$(BRAIN) worldmirror2 infer --weights $(MIRROR_WEIGHTS) --images $(MIRROR_IMAGES) $(MIRROR_ARGS)

## mirror/demo - images -> 3DGS scene, then view it (MIRROR_IMAGES=<dir|a.ppm,b.ppm>)
mirror/demo: build/release
	@test -n "$(MIRROR_IMAGES)" || (echo "set MIRROR_IMAGES=<dir|a.ppm,b.ppm>"; exit 2)
	$(BRAIN) worldmirror2 demo --weights $(MIRROR_WEIGHTS) --images $(MIRROR_IMAGES) $(MIRROR_ARGS)

# 3DGS scene viewer (interactive fly-through; WASD + mouse, see --help).
SPLAT_SCENE ?=

## splat - lists the 3D Gaussian splat targets; does nothing
splat:
	$(LIST_CHILDREN)

## splat/view - interactive 3DGS fly-through (SPLAT_SCENE=<scene.ply>, SPLAT_ARGS=...)
splat/view: build/release
	@test -n "$(SPLAT_SCENE)" || (echo "set SPLAT_SCENE=<scene.ply>"; exit 2)
	$(BRAIN) splat view $(SPLAT_SCENE) $(SPLAT_ARGS)

## tts - lists the text-to-speech targets; does nothing
tts:
	$(LIST_CHILDREN)

# Synthetic Qwen3-TTS `text -> codebook-0 codes` stream (for Talker SFT smokes).
## tts/data - synthetic TTS text->codes dataset (Talker SFT smokes) -> data/tts
tts/data: build/release
	$(BRAIN) data gen tts --out $(DATA)/tts --n $(N) $(SEED_FLAG)

## flux2 - lists the FLUX.2 Klein image targets (weights via BRAIN_FLUX2_* env); does nothing
flux2:
	$(LIST_CHILDREN)

## flux2/generate - generate one image with FLUX.2 Klein (PROMPT=..., BRAIN_FLUX2_* env)
flux2/generate: build/release
	@test -n "$(BRAIN_FLUX2_DIT)" || (echo "set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER"; exit 2)
	$(BRAIN) flux2 generate --prompt "$(PROMPT)" --out out/flux2.ppm $(FLUX2_FLAGS)

## flux2/edit - reference-image edit with FLUX.2 Klein (FLUX2_REF=<ref.ppm> PROMPT=...)
flux2/edit: build/release
	@test -n "$(FLUX2_REF)" || (echo "set FLUX2_REF=<ref.ppm> PROMPT=..."; exit 2)
	$(BRAIN) flux2 generate --prompt "$(PROMPT)" --ref $(FLUX2_REF) --out out/flux2-edit.ppm $(FLUX2_FLAGS)

# WAN_FLAGS carries no seed: wan t2v draws one and prints it (add `--seed N` to
# reproduce). It is deliberately NOT upstream's sampling defaults: 81 frames at
# 832x480 over 50 steps is about an hour on a Tesla P40. These are the settings
# a run can be checked with, and the ones the quickstart demo uses; drop
# WAN_FLAGS entirely for the real thing.
WAN_PROMPT ?= a golden retriever running along a sandy beach at sunset, waves in the background, cinematic
WAN_FLAGS ?= --frames 9 --width 416 --height 240 --steps 20

## wan - lists the Wan2.1 text-to-video targets (weights via BRAIN_WAN_* env); does nothing
wan:
	$(LIST_CHILDREN)

## wan/t2v - generate one video clip with Wan2.1 (BRAIN_WAN_* env, WAN_PROMPT, WAN_FLAGS)
wan/t2v: build/release
	@test -n "$(BRAIN_WAN_DIT)" || (echo "set BRAIN_WAN_DIT/_VAE/_T5/_TOKENIZER"; exit 2)
	$(BRAIN) wan t2v --prompt "$(WAN_PROMPT)" --output-path out/wan.mp4 $(WAN_FLAGS)

# ---- forecasting ----------------------------------------------------------
## forecast - lists the forecasting targets; does nothing
forecast:
	$(LIST_CHILDREN)

# `make forecast/compare` runs the scenario battery against the statistical
# baselines and renders the model x scenario x metric report (markdown to
# stdout). The random-walk negative control is a HARD gate: the command exits
# non-zero if any model falsely beats naive on it. Add HTML=path to also write a
# self-contained HTML report. Foundation models join the same battery as they
# are imported.
## forecast/compare - scenario battery vs the statistical baselines, model x scenario x metric report
##   (HTML=path also writes a self-contained report)
forecast/compare: build/release
	$(BRAIN) forecast compare $(SEED_FLAG) $(if $(HTML),--html $(HTML),)

# `make forecast/serve` starts the unified JSONL server with the baselines
# registered. Defaults to a Unix socket; override with LISTEN=host:port for TCP
# or SOCKET=path for a different socket path.
## forecast/serve - start the unified JSONL forecasting server (LISTEN=host:port or SOCKET=path)
forecast/serve: build/release
	$(BRAIN) forecast serve $(if $(LISTEN),--listen $(LISTEN),--socket $(or $(SOCKET),/tmp/brain-forecast.sock))

## wm - lists the world-model targets; does nothing
wm:
	$(LIST_CHILDREN)

# Play the deterministic fake world model in an SDL window (WASD; Esc quits).
# The SDL window is always compiled into the standard build (needs system
# libSDL2 at link); it only OPENS when a run needs it.
## wm/play - play the fake world model in an SDL window (WASD)
wm/play: build/release
	./target/release/brain diamond play --model fake

# Regenerate the DIAMOND parity fixtures (gitignored - never committed) from
# the reference implementation (resources/world-models/repos/diamond, or set
# BRAIN_DIAMOND_REPO). Needs python3 + torch; see docs/world-models/FIXTURES.md.
## wm/fixtures - regenerate the DIAMOND parity fixtures (needs torch)
wm/fixtures:
	python3 scripts/parity-dump/diamond.py --out crates/diamond/tests/fixtures/diamond

## federated - lists the federated-MoE targets; does nothing
federated:
	$(LIST_CHILDREN)

## federated/demo - MoE train -> split -> verify -> merge round-trip
federated/demo: build/release
	@mkdir -p $(OUT)
	$(BRAIN) train --steps 50 --out $(OUT)/moe.safetensors
	$(BRAIN) federated split $(OUT)/moe.safetensors $(OUT)/shards
	$(BRAIN) federated verify $(OUT)/shards
	$(BRAIN) federated merge $(OUT)/shards --out $(OUT)/moe-reassembled.safetensors
	@echo "federated round-trip complete: $(OUT)/moe-reassembled.safetensors"

# ---- web (delegate to the web crate's Makefile) ---------------------------
## web - lists the WebGPU browser demo targets (crates/web); does nothing
web:
	$(LIST_CHILDREN)

## web/dev - build the wasm and start the WebGPU browser demo dev server (crates/web)
web/dev:
	$(MAKE) -C crates/web web/dev

## web/build - production build of the WebGPU browser demo (crates/web)
web/build:
	$(MAKE) -C crates/web web/build

# ---------------------------------------------------------------------------
##@ Benchmarks and performance
## `bench` asks whether an architecture LEARNS a task; `perf` asks how much
## correct work the engine delivers per unit of hardware, memory, energy and time.

# ---- architecture-evaluation benchmark suite ------------------------------
# `make bench` runs every registered benchmark (crates/bench) and prints one
# comparison table (benchmark | score | threshold | pass/fail). `make
# bench/run/<name>` runs a single benchmark, e.g. `make bench/run/mqar`
# (multi-query associative recall). Add new benchmarks by registering them in
# crates/bench/src/lib.rs::registry; a benchmark whose `fn name` returns a
# string literal gets its bench/run/<name> target from the sources with no
# Makefile change (forecast_* benchmarks are named at run time and run only as
# part of `make bench`).
BENCH_NAMES := $(sort $(subst _,-,$(shell sed -sn '/fn name(&self)/{n;s/^ *"\(.*\)"$$/\1/p}' crates/bench/src/*.rs)))
BENCH_RUN_TARGETS := $(addprefix bench/run/,$(BENCH_NAMES))

## bench - default: run the whole architecture-evaluation benchmark suite
bench: build/release
	$(BRAIN) bench $(SEED_FLAG)

## bench/run - lists the single-benchmark targets; runs nothing (bench runs them all)
bench/run:
	$(LIST_CHILDREN)

## bench/run/<name> - run one registered benchmark, e.g. bench/run/mqar
##   Underscores in a benchmark's name are hyphens in the target (mad-recall).
$(BENCH_RUN_TARGETS): bench/run/%: build/release
	$(BRAIN) bench $(subst -,_,$*) $(SEED_FLAG)

# `make bench/scaling` runs the multi-scale scaling-law sweep (a separate entry
# point, not a registry benchmark): it trains the MQAR task at several model
# sizes and fits L(N) = E + A*N^-alpha, printing the size|params|flops|loss table
# plus the fitted exponent alpha and fit R^2. Foundation for the later
# per-capability predictive-scaling / eval-harness work. ~5 min on the CPU backend.
## bench/scaling - scaling-law sweep: fit L(N)=E+A*N^-alpha across model sizes
bench/scaling: build/release
	$(BRAIN) bench scaling $(SEED_FLAG)

# `make bench/distributed` runs the single-machine distributed-training
# profile: a tiny GPT trained as world 1/2/4 over the host staging collective,
# loopback TCP and a shaped link (propagation delay + bandwidth cap), each
# wrapped by the measured collective, and prints one JSON document with step
# times, throughput, communication cost/share, speedup and replica identity.
# Pass flags through ARGS, e.g. `ARGS="--steps 5 --worlds 1,2"`.
## bench/distributed - distributed-training profile on one machine: world 1/2/4 x host/loopback/shaped, one JSON document (ARGS="--steps 5")
bench/distributed:
	cargo run --release --offline -p brain-bench --bin distributed_profile -- $(ARGS)

# `make bench/eval ARCH=<name>` runs the turn-key architecture-eval harness: the
# WHOLE registered battery against one architecture, aggregated per capability
# axis, writing a structured artifact to results/<arch>-<seed>.json. Add a new
# architecture in crates/bench/src/arch.rs::arch_registry, then ARCH=<name> here.
## bench/eval - the WHOLE battery vs one architecture (ARCH=gpt2|gpt2-small|gpt2-wide), aggregated per axis
##   -> results/<arch>-<seed>.json
bench/eval: build/release
	$(BRAIN) bench eval --arch $(ARCH) $(SEED_FLAG)

# `make bench/scale ARCH=<name>` runs the PREDICTIVE per-capability scaling sweep:
# train+score one representative benchmark per capability axis across a small SIZE
# grid, fit how each axis's score scales with params N, extrapolate the predicted
# score at 2x/4x the largest N, and write results/scale-<arch>-<seed>.json. This
# answers "how will each capability improve as we grow the model?" before paying
# for the bigger run. ~few min on the CPU backend (3 sizes x 6 axes, smoke budget).
## bench/scale - predictive per-capability scaling: score at 2x/4x per axis (ARCH=<name>)
bench/scale: build/release
	$(BRAIN) bench scale --arch $(ARCH) $(SEED_FLAG)

# `make bench/advise` prints RANKED tuning recommendations (what to tune to improve
# in the best capability direction): headroom x size-slope per axis, with a concrete
# action (increase size | change mechanism | more data/reg | deprioritize). Consumes
# the eval artifact and, if present, the scaling artifact for the same ARCH/SEED.
## bench/advise - ranked tuning recommendations from the eval (+scale) artifacts (ARCH=<name> SEED=<n>)
bench/advise: build/release
	@test -n "$(SEED)" || (echo "set SEED=<n> to the seed of the eval run (bench/eval prints it, and names results/$(ARCH)-<seed>.json)"; exit 2)
	@set -e; ev="results/$(ARCH)-$(SEED).json"; sc="results/scale-$(ARCH)-$(SEED).json"; \
	if [ ! -f "$$ev" ]; then \
		echo "no $$ev - run 'make bench/eval ARCH=$(ARCH)' first"; exit 2; \
	fi; \
	if [ -f "$$sc" ]; then $(BRAIN) bench advise "$$ev" "$$sc"; \
	else $(BRAIN) bench advise "$$ev"; fi

# `make bench/compare` prints a side-by-side leaderboard (overall pass-rate +
# per-axis + per-benchmark scores, columns = architectures) over every artifact
# under results/, so a new architecture is diffed against priors at a glance.
## bench/compare - side-by-side leaderboard of every results/<arch>-<seed>.json
bench/compare: build/release
	@set -e; \
	n=$$(find results -maxdepth 1 -name '*.json' ! -name 'scale-*' -print0 2>/dev/null | tr -cd '\0' | wc -c); \
	if [ "$$n" -eq 0 ]; then \
		echo "no eval artifacts yet - run 'make bench/eval ARCH=<name>' first"; exit 2; \
	fi; \
	find results -maxdepth 1 -name '*.json' ! -name 'scale-*' -print0 | xargs -0 $(BRAIN) bench compare

# ---- shared GPT char-dataset benchmark (legacy) ---------------------------
# Train + eval the GPT baseline on the same char datasets (pass SEED=<n> to pin the data),
# so results are comparable. (MoE-on-char-data + federated rows are a documented
# follow-up - the MoE engine currently trains on its own 64-symbol rule task.)
## bench/char - train + eval GPT on the shared char datasets (legacy)
bench/char: build/release
	@for d in calculator reverser; do \
		echo "=== dataset: $$d ==="; \
		$(MAKE) gpt/$$d/data N=$(N) SEED=$(SEED); \
		$(MAKE) gpt/$$d/train STEPS=$(STEPS); \
		$(MAKE) gpt/$$d/eval; \
	done

# ---- performance benchmarking (brain perf) --------------------------------
# Distinct from `make bench`: bench asks whether an architecture LEARNS a task;
# perf asks how much correct work the engine delivers per unit of hardware,
# memory, energy and time. Design: docs/guides/hardware/benchmarking.md.
#
# TARGET selects what is measured. There is no synthetic-harness stand-in --
# every target exercises a real engine. Default is `qwen-synth:<L>x<D>x<H>x<V>x
# <HeadDim>x<NKvHeads>` at Qwen3-0.6B's REAL KV geometry (the REAL paged
# serving engine on random weights - same kernels, KV traffic and batching, so
# hardware comparison works with no checkpoint on the machine); override with
# `qwen:<weights>` to measure a real checkpoint.
PERF_TARGET ?= qwen-synth:28x1024x16x151936x128x8
PERF_WORKLOAD ?= chat
PERF_LADDER ?= 1,2,4,8,16,32
# The scenarios `brain perf run` accepts: the names in the SCENARIOS table of
# crates/perf/src/scenarios/mod.rs.
PERF_SCENARIOS := $(shell sed -n 's/^    ."\([a-z0-9-]*\)", ".*/\1/p' crates/perf/src/scenarios/mod.rs)
PERF_SCENARIO_TARGETS := $(addprefix perf/scenario/,$(PERF_SCENARIOS))

# The whole core battery on the current device: latency floor, saturated
# ceiling, realistic serving, and the concurrency curve.
## perf - default: the core PERFORMANCE battery (latency, throughput, serve, sweep) on qwen-synth
##   (how fast + at what cost; docs/guides/hardware/benchmarking.md)
perf: build/release
	@set -e; for s in latency throughput serve; do \
		$(BRAIN) perf run $$s --target $(PERF_TARGET) --workload $(PERF_WORKLOAD) $(SEED_FLAG); \
	done; \
	$(BRAIN) perf run sweep --target $(PERF_TARGET) --workload $(PERF_WORKLOAD) --ladder $(PERF_LADDER) $(SEED_FLAG)

## perf/scenario - lists the perf scenarios; runs nothing (perf runs the core battery)
perf/scenario:
	$(LIST_CHILDREN)

# One scenario: `make perf/scenario/sweep`, `make perf/scenario/serve`, and so on.
## perf/scenario/<scenario> - run one perf scenario, e.g. perf/scenario/sweep
##   Scenarios are the names in crates/perf/src/scenarios/mod.rs.
$(PERF_SCENARIO_TARGETS): perf/scenario/%: build/release
	$(BRAIN) perf run $* --target $(PERF_TARGET) --workload $(PERF_WORKLOAD) --ladder $(PERF_LADDER) $(SEED_FLAG)

# Leaderboard over every perf artifact. Refuses to rank across artifact units,
# excludes runs whose correctness gate failed, and warns on differing axes.
## perf/compare - leaderboard over results/perf-*.json
perf/compare: build/release
	@set -e; \
	n=$$(find results -maxdepth 1 -name 'perf-*.json' -print0 2>/dev/null | tr -cd '\0' | wc -c); \
	if [ "$$n" -eq 0 ]; then echo "no perf artifacts yet - run 'make perf' first"; exit 2; fi; \
	find results -maxdepth 1 -name 'perf-*.json' -print0 | xargs -0 $(BRAIN) perf compare

# CI-sized: every scenario shrunk to seconds, on the CPU backend.
## perf/smoke - CI-sized run of every perf scenario
perf/smoke: build/release
	@set -e; for s in latency throughput serve sweep mixed overload soak frontend; do \
		$(BRAIN) perf run $$s --target $(PERF_TARGET) --workload interactive --smoke $(SEED_FLAG); \
	done

## perf/model - lists the per-model perf benchmarks; runs nothing
perf/model:
	$(LIST_CHILDREN)

# LFM2.5-Encoder concurrency benchmark, standalone: the residency-executor
# target (real scheduler + budgets + lanes + equal-length batching) at 8k
# context. LFM_WEIGHTS/LFM_TOKENIZER select the model (230m/350m).
# LFM_TOKENIZER has NO default: the tokenizer lives outside the repo, and a
# baked-in absolute path is one dev box's layout (same @test -n guard as the
# depth/mirror/splat targets).
LFM_WEIGHTS ?= out/lfm-230m.safetensors
LFM_INPUT ?= 8192
## perf/model/lfm - LFM2.5 8k-context concurrency benchmark (LFM_WEIGHTS / LFM_TOKENIZER select the model)
perf/model/lfm: build/release
	@test -n "$(LFM_TOKENIZER)" || (echo "set LFM_TOKENIZER=<path to LFM2.5-Encoder tokenizer.json>"; exit 2)
	@set -e; for s in latency sweep; do \
		$(BRAIN) perf run $$s --target lfm:$(LFM_WEIGHTS):$(LFM_TOKENIZER) \
			--input $(LFM_INPUT) --output 1 --ladder $(PERF_LADDER) $(SEED_FLAG); \
	done

# FLUX.2 Klein denoise-step benchmark, standalone: the residency-executor
# target (real scheduler + budgets + lanes) on klein-4b; weights from the
# BRAIN_FLUX2_* env (same as flux2/generate). One denoise step is MINUTES on a
# CPU backend, so the request count is a knob and defaults tiny - size the run,
# don't let it size you. FLUX2_SIZE is <W>x<H>x<steps>; --output mirrors the
# step count so the workload's requested artifacts match what a request emits.
FLUX2_SIZE ?= 512x512x4
FLUX2_REQUESTS ?= 2
FLUX2_WARMUP ?= 1
## perf/model/flux2 - FLUX.2 Klein denoise-step benchmark (BRAIN_FLUX2_* env, FLUX2_SIZE=<W>x<H>x<steps>)
perf/model/flux2: build/release
	@test -n "$(BRAIN_FLUX2_DIT)" || (echo "set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER"; exit 2)
	$(BRAIN) perf run latency --target flux2:$(FLUX2_SIZE) \
		--concurrency 1 --requests $(FLUX2_REQUESTS) --warmup $(FLUX2_WARMUP) \
		--input 1 --output $(word 3,$(subst x, ,$(FLUX2_SIZE))) $(SEED_FLAG)

# Wan2.1 text-to-video denoise-step benchmark, standalone: the residency-executor
# target (real scheduler + budgets + lanes) on t2v-1.3B; weights from the
# BRAIN_WAN_* env (same as wan/t2v). WAN_SIZE is <frames>x<W>x<H>x<steps> and
# defaults to a SMOKE size, not upstream's 81x832x480x50 (about an hour a
# request on a Tesla P40); --output mirrors the step count so the workload's
# requested artifacts match what a request emits. Every request also pays a
# fixed umT5-XXL text encode on the CPU before its first denoise step, so keep
# the request count small.
WAN_SIZE ?= 9x256x256x4
WAN_REQUESTS ?= 2
WAN_WARMUP ?= 1
## perf/model/wan - Wan2.1 text-to-video denoise-step benchmark (BRAIN_WAN_* env, WAN_SIZE=<frames>x<W>x<H>x<steps>)
perf/model/wan: build/release
	@test -n "$(BRAIN_WAN_DIT)" || (echo "set BRAIN_WAN_DIT/_VAE/_T5/_TOKENIZER"; exit 2)
	$(BRAIN) perf run latency --target wan:$(WAN_SIZE) \
		--concurrency 1 --requests $(WAN_REQUESTS) --warmup $(WAN_WARMUP) \
		--input 1 --output $(word 4,$(subst x, ,$(WAN_SIZE))) $(SEED_FLAG)

# Qwen3.8-27B batch-size sweep: aggregate decode tokens/s and per-sequence
# latency of the released GGUF at 1..64 sequences decoding together, through
# the same engine the continuous-batching scheduler drives. QWEN35_GGUF has NO
# default (the checkpoint lives outside the repo). Run it on an idle card:
# `lease/gpu --lock <file> --exclusive -- make perf/model/qwen35-batch ...`.
QWEN35_CTX ?= 4096
QWEN35_KV ?= bf16
QWEN35_PROMPT_TOKENS ?= 512
QWEN35_BATCHES ?= 1,4,8,16,32,64
## perf/model/qwen35-batch - Qwen3.8-27B batch sweep (QWEN35_GGUF=<Q8_0.gguf>, QWEN35_CTX, QWEN35_KV=f32|bf16|int8, QWEN35_BATCHES=1,4,..)
perf/model/qwen35-batch:
	@test -n "$(QWEN35_GGUF)" || (echo "set QWEN35_GGUF=<path to Qwen3.8-27B Q8_0 .gguf>"; exit 2)
	cargo run --release -p brain-qwen35 --bin qwen35_batch_sweep -- $(QWEN35_GGUF) \
		--ctx $(QWEN35_CTX) --kv $(QWEN35_KV) --prompt-tokens $(QWEN35_PROMPT_TOKENS) --batches $(QWEN35_BATCHES)

## perf/gate - lists the perf regression gates; runs nothing
perf/gate:
	$(LIST_CHILDREN)

# Forecasting latency regression gate: each forecaster through `brain perf run`
# vs a local baseline (scripts/gates/forecast-perf-baselines/, gitignored -
# `--update` to capture/refresh it on this machine). Weights via env
# (BRAIN_KRONOS_*/BRAIN_CHRONOS2/BRAIN_FINCAST).
## perf/gate/forecast - forecasting latency regression gate (vs a local baseline)
perf/gate/forecast: build/release
	scripts/gates/forecast-perf-gate.sh

# World-model fps regression gate (best-of-3 vs a local
# scripts/gates/wm-perf-baselines.json, gitignored; hard floors only).
# Dev-box gate, not CI: needs out/diamond-breakout.weights (brain diamond
# import ...) and a real display/GPU. `--update` captures/rewrites it.
## perf/gate/wm - world-model fps regression gate (vs a local baseline; needs a display/GPU)
perf/gate/wm: build/release
	scripts/gates/wm-perf-gate.sh

# Concurrent-serving-performance regression gate: the real HTTP-served path
# (http:qwen-synth: target, random weights, no checkpoint needed) through
# `brain perf run sweep` at concurrency 1,2, vs a local baseline
# (scripts/gates/qwen-serving-perf-baselines/, gitignored - `--update` to
# capture/refresh it on this machine).
# Needs a real tokenizer via QWEN_TOKENIZER; SKIPS (not fails) when unset.
## perf/gate/qwen-serving - qwen concurrent-serving regression gate (QWEN_TOKENIZER; skips when unset)
perf/gate/qwen-serving: build/release
	scripts/gates/qwen-serving-perf-gate.sh

# ---- qwen35 training-step instrument (device time by component) ------------
# The probe (random weights, real 27B shape) or the finetune CLI, run under an
# exclusive-GPU wrapper that refuses to start when another compute context is on
# the card. The wrapper is the caller's: GPU_WRAPPER names its executable and is
# invoked as `$(GPU_WRAPPER) --exclusive <command>`; nothing here knows what it
# is. BRAIN_PROFILE=1 makes the trainer print the per-kernel device table, which
# group-kernels.py folds into the components a step is budgeted by. The raw
# output and the component table are kept under TRAIN_RUN_DIR.
GPU_WRAPPER ?=
TRAIN_BACKEND ?= cuda
TRAIN_RUN_DIR ?= results/qwen35-train
ARGS ?=

# Fails before any build starts when no wrapper was given.
_train/need-wrapper:
	@test -n "$(GPU_WRAPPER)" || { echo "set GPU_WRAPPER=<exclusive-GPU wrapper executable>"; exit 2; }

# Usage: $(call train_run,<command>,<log name>)
define train_run
@mkdir -p $(TRAIN_RUN_DIR)
@set -o pipefail; \
	BRAIN_BACKEND=$(TRAIN_BACKEND) BRAIN_PROFILE=1 $(GPU_WRAPPER) --exclusive $(1) 2>&1 | tee $(TRAIN_RUN_DIR)/$(2).log
@python3 scripts/perf/group-kernels.py $(TRAIN_RUN_DIR)/$(2).log | tee $(TRAIN_RUN_DIR)/$(2)-components.txt
endef

## qwen35 - lists the qwen35 targets; runs nothing
qwen35:
	$(LIST_CHILDREN)

## qwen35/train - lists the qwen35 training-step instruments; runs nothing
qwen35/train:
	$(LIST_CHILDREN)

## qwen35/train/probe - PROBE lines + device time by component from the sizing probe (GPU_WRAPPER=..., ARGS='--tokens 8192 --base nf4 ...')
##   Use --steps 3 or more: the last step of a profiled run is the one tabulated.
qwen35/train/probe: _train/need-wrapper
	cargo build --release -p brain-qwen35 --bin qwen35_train_probe
	$(call train_run,target/release/qwen35_train_probe --profile $(ARGS),probe)

## qwen35/train/step - the same through `brain qwen35 finetune` (GPU_WRAPPER=..., ARGS='<data_dir> --base DIR --out F ...')
qwen35/train/step: _train/need-wrapper build/release
	$(call train_run,$(BRAIN) qwen35 finetune $(ARGS),step)

# ---------------------------------------------------------------------------
##@ Samples and crate examples
## Standalone applications and cargo examples, generated from the tree: a new
## directory adds its build and run targets.

# A sample is a standalone APPLICATION built on the public `brain` SDK, living
# in samples/<category>/<name>/ - Zephyr's samples/ idea. Each one NAMES the SDK
# surfaces it uses and links those and nothing else. Samples are workspace
# members (so they share ./target, one lockfile and one registry) but never
# `default-members`, so `make build/release` and `make test` never build one.
# The full contract is samples/README.md.
#
#   make samples
#   make samples/imagegen/generate/build
#   make samples/imagegen/generate/run ARGS="--prompt 'a whale submarine'"
#
# `build` and `run` are separate on purpose, and `run` does NOT invoke cargo.
# A sample is an ordinary binary once it is built: copying
# `target/release/sample-<a>-<b>` to a machine with no Rust toolchain and
# running it there is the whole distribution story, and a `run` target that
# shelled out to `cargo run` quietly made the toolchain a runtime dependency
# of every demonstration.
#
# The cost of the split is that `run` cannot notice a stale binary, and a
# stale binary has already cost this repo an hour of CPU and a result that no
# replay could reproduce. So `run` prints what it is about to execute and when
# that file was built, which is the fact that would have made it obvious.
SAMPLE_PROFILE ?= release
SAMPLE_CARGO_FLAGS = $(if $(filter release,$(SAMPLE_PROFILE)),--release,)
ARGS ?=

# samples/<a>/<b> -> package `sample-<a>-<b>`. check/samples enforces it.
sample_pkg = sample-$(subst /,-,$(1))

# A sample is built with `-p <sample>` and nothing else. It CANNOT be built
# alongside the engine's package selection: `crates/sdk` is a default member,
# so a selection containing both it and a sample unions the sample's declared
# features with the SDK's own defaults, and the sample's `features = ["image"]`
# becomes a silent no-op that links every model crate anyway. Cargo has no
# per-package feature flag and features cannot be subtracted, so a narrow
# selection is the only mechanism that actually honours the declaration.
SAMPLE_PATHS := $(patsubst samples/%/Cargo.toml,%,$(wildcard samples/*/*/Cargo.toml))
SAMPLE_CATEGORIES := $(sort $(patsubst %/,%,$(dir $(SAMPLE_PATHS))))
SAMPLE_CATEGORY_TARGETS := $(addprefix samples/,$(SAMPLE_CATEGORIES))
SAMPLE_NAME_TARGETS := $(addprefix samples/,$(SAMPLE_PATHS))
SAMPLE_BUILD_TARGETS := $(addsuffix /build,$(SAMPLE_NAME_TARGETS))
SAMPLE_RUN_TARGETS := $(addsuffix /run,$(SAMPLE_NAME_TARGETS))

## samples - lists the standalone sample applications and how to build and run them; builds nothing
samples:
	@echo "Rust samples (make samples/<category>/<name>/{build,run}):"
	@find samples -mindepth 3 -maxdepth 3 -name Cargo.toml -printf '%h\n' 2>/dev/null \
		| sed 's|^samples/||' | sort | while read -r s; do \
			printf '  %-34s %s\n' "$$s" "$$(sed -n 's/^description = "\(.*\)"/\1/p' "samples/$$s/Cargo.toml" | head -1)"; \
		done
	@echo "Shell/Python samples (run directly - see each README.md):"
	@find samples/shell samples/python -mindepth 3 -maxdepth 3 -name README.md 2>/dev/null \
		| sed -E 's|/README\.md$$||' | sort | while read -r s; do printf '  %s\n' "$$s"; done

## samples/<category> - lists the samples of a category; builds nothing
$(SAMPLE_CATEGORY_TARGETS):
	$(LIST_CHILDREN)

## samples/<category>/<name> - lists a sample's actions (build, run); does nothing
$(SAMPLE_NAME_TARGETS):
	$(LIST_CHILDREN)

## samples/<category>/<name>/build - build one sample (shares brain's ./target cache)
$(SAMPLE_BUILD_TARGETS): samples/%/build:
	cargo build $(SAMPLE_CARGO_FLAGS) -p $(call sample_pkg,$*)

## samples/<category>/<name>/run - run a built sample, printing when it was built (ARGS=...)
$(SAMPLE_RUN_TARGETS): samples/%/run:
	@bin="target/$(SAMPLE_PROFILE)/$(call sample_pkg,$*)"; \
	test -x "$$bin" || { echo "not built: $$bin"; echo "run: make samples/$*/build"; exit 2; }; \
	echo "$$bin (built $$(date -r "$$bin" '+%Y-%m-%d %H:%M:%S'))"; \
	exec env $(ENV) "$$bin" $(ARGS)

# A crate's cargo EXAMPLES, built by one target and run by another.
#
# The target path mirrors the source path, exactly as `samples/<a>/<b>` does:
# `crates/fly/examples/cpg.rs` is `make crates/fly/examples/cpg/{build,run}`
# (underscores in the file name are hyphens in the target:
# `walk_search.rs` is `walk-search`). There is no separate category for these.
# They are examples of a crate, they live in that crate's directory, and the
# target says so. The set is generated from crates/*/examples/*.rs.
#
#   make crates/fly/examples/walk-search/build
#   make crates/fly/examples/walk-search/run ENV="OUT=/tmp/walk.txt"
#   make crates/fly/examples/replay/run ARGS=/tmp/walk.txt
#
# ENV carries the run's own `NAME=value` settings, ARGS its positional
# arguments. Everything the examples read from the wider environment
# (BRAIN_CONNECTOME_DIR, BRAIN_FLYBODY_XML, ...) is expected to be set already,
# as `tools/buzzfly/collect.sh`'s env.sh sets it.
#
# Split build from run for the reason samples are split: an example is an
# ordinary binary once built, and a `run` that shelled out to cargo would make
# the toolchain a runtime dependency of every measurement.
#
# The hazard that split reintroduces is worth naming, because it has already
# been paid for. `cargo build -p brain-fly` does not build examples, so a
# change to an objective can be committed, believed, and simply not be in the
# binary a search executes. That happened: a walk search ran for forty
# generations against an objective without its attitude termination, found the
# slide that termination exists to refuse, and reported a score no replay could
# reproduce. An hour of CPU, and the only symptom was a number that did not
# survive re-measurement. `run` therefore prints the binary it is about to
# execute and when that file was built, which is exactly the fact that would
# have made it obvious at the time.
EXAMPLE_PATHS := $(subst _,-,$(patsubst crates/%.rs,%,$(wildcard crates/*/examples/*.rs)))
EXAMPLE_CRATES := $(sort $(foreach p,$(EXAMPLE_PATHS),$(word 1,$(subst /, ,$(p)))))
EXAMPLE_CRATE_TARGETS := $(addprefix crates/,$(EXAMPLE_CRATES))
EXAMPLE_GROUP_TARGETS := $(addsuffix /examples,$(EXAMPLE_CRATE_TARGETS))
EXAMPLE_NAME_TARGETS := $(addprefix crates/,$(EXAMPLE_PATHS))
EXAMPLE_BUILD_TARGETS := $(addsuffix /build,$(EXAMPLE_NAME_TARGETS))
EXAMPLE_RUN_TARGETS := $(addsuffix /run,$(EXAMPLE_NAME_TARGETS))
# From the stem `<crate>/examples/<name-with-hyphens>`: the crate directory, the
# example's file name (hyphens back to underscores) and the cargo package.
example_crate = $(word 1,$(subst /, ,$*))
example_name = $(subst -,_,$(word 3,$(subst /, ,$*)))
example_pkg = $(shell sed -n 's/^name = "\(.*\)"/\1/p' crates/$(example_crate)/Cargo.toml 2>/dev/null | head -1)

## crates - lists the crates that carry cargo examples; builds nothing
crates:
	$(LIST_CHILDREN)

## crates/<crate> - lists a crate's example groups; does nothing
$(EXAMPLE_CRATE_TARGETS):
	$(LIST_CHILDREN)

## crates/<crate>/examples - lists a crate's cargo examples; builds nothing
$(EXAMPLE_GROUP_TARGETS):
	$(LIST_CHILDREN)

## crates/<crate>/examples/<name> - lists an example's actions (build, run); does nothing
$(EXAMPLE_NAME_TARGETS):
	$(LIST_CHILDREN)

## crates/<crate>/examples/<name>/build - build one crate example in release mode
$(EXAMPLE_BUILD_TARGETS): crates/%/build:
	cargo build --release --offline -p $(example_pkg) --example $(example_name)

## crates/<crate>/examples/<name>/run - run a built crate example, printing when it was built (ARGS=..., ENV=...)
$(EXAMPLE_RUN_TARGETS): crates/%/run:
	@bin="target/release/examples/$(example_name)"; \
	test -x "$$bin" || { echo "not built: $$bin"; echo "run: make crates/$*/build"; exit 2; }; \
	echo "$$bin (built $$(date -r "$$bin" '+%Y-%m-%d %H:%M:%S'))"; \
	exec env $(ENV) "$$bin" $(ARGS)

# ---------------------------------------------------------------------------
##@ Documentation, generated files and release
## The docs bundle, files generated from the tree, test data, git hooks and
## the release flow. generate, testdata, hooks and release only list.

# The documentation publication: docs/manifest.yaml is the one navigation order,
# and the vendored publisher (tools/docs-publisher) turns docs/ into a validated
# set of pages in build/docs/. Offline: it never compiles Rust or fetches
# anything. The pin is verified unless DOCS_PUBLISHER is overridden on the
# command line (developing against a newer publisher).
NODE ?= node
DOCS_PUBLISHER ?= tools/docs-publisher/docs-publisher.mjs

## docs - default: build the manual (pandoc + xelatex) -> build/pdf/brain-docs.{md,pdf}
docs: docs/pdf

_docs/publisher-pin:
	@$(if $(filter command line,$(origin DOCS_PUBLISHER)),true,\
	  cd tools/docs-publisher && sha256sum -c docs-publisher.mjs.sha256 >/dev/null)

## docs/build - build the documentation publication -> build/docs/ (offline)
docs/build: _docs/publisher-pin
	$(NODE) $(DOCS_PUBLISHER) build --root . --manifest docs/manifest.yaml --out build/docs

# Publisher check (builds twice, compares, validates links and coverage), the
# generated references (all three generators are pure Python over the sources
# and run offline) and the quick documentation gates. The slower prose scan for
# unreviewed performance numbers stays in `check/scripts`.
## docs/check - the publication builds reproducibly and the docs gates pass
docs/check: _docs/publisher-pin check/generated/kernels-wgsl-table check/generated/kernels-cuda-table check/generated/samples-manifest
	$(NODE) $(DOCS_PUBLISHER) check --root . --manifest docs/manifest.yaml
	bash scripts/gates/check-doc-links.sh
	bash scripts/gates/check-arch-names.sh
	bash scripts/gates/check-env-docs.sh

## docs/clean - remove generated documentation output (build/docs and build/pdf)
docs/clean:
	rm -rf build/docs build/pdf

# The compiled manual, in the order of docs/manifest.yaml. Requires pandoc +
# xelatex. No HTML is produced.
## docs/pdf - build the manual (pandoc + xelatex) -> build/pdf/brain-docs.{md,pdf}
docs/pdf:
	python3 docs/pandoc/build-docs.py

# Research papers: one directory per paper under papers/, each with its own
# Makefile over papers/paper.mk. Adding a directory adds its targets.
PAPER_DIRS := $(patsubst %/Makefile,%,$(wildcard papers/*/Makefile))
PAPER_TARGETS := $(addsuffix /pdf,$(PAPER_DIRS))

## papers - default: papers/pdf (every paper's PDF -> build/papers/)
papers: papers/pdf

## papers/pdf - build every paper's PDF -> build/papers/<name>.pdf
papers/pdf: $(PAPER_TARGETS)

## papers/<name> - build that paper's PDF (same as papers/<name>/pdf)
$(PAPER_DIRS): %: %/pdf

## papers/<name>/pdf - build one paper's PDF (LaTeX via latexmk; tables from its data/)
$(PAPER_TARGETS): papers/%/pdf:
	$(MAKE) -C papers/$*

# Regenerates README.md's Quick start assets: every one-liner in it, run for
# real against auto-fetched weights (tens of GB on a cold cache), writing
# docs/getting-started/img/*.png and the *.txt/*.json outputs the README quotes.
# Heavy and network-bound, not compute-bound - see scripts/demo/quickstart.sh's
# own docs. `test/e2e/quickstart` asserts on this target's output afterward.
## docs/quickstart - regenerate README's Quick start assets against REAL auto-fetched weights (heavy, network)
docs/quickstart: build/release
	scripts/demo/quickstart.sh

## generate - lists the generators of checked-in files; writes nothing
##   (check/generated verifies each one has not drifted)
generate:
	$(LIST_CHILDREN)

## generate/kernels - lists the kernel catalogue generators; writes nothing
generate/kernels:
	$(LIST_CHILDREN)

# Regenerate the kernel const block + ALL registry in crates/kernels/src/lib.rs
# from the contents of crates/kernels/wgsl/. Run after adding/removing a .wgsl
# file; merge conflicts in lib.rs are resolved by union-ing wgsl/ + this target.
## generate/kernels/registry - regenerate the kernel const block + ALL registry in crates/kernels/src/lib.rs
generate/kernels/registry:
	scripts/build/kernels-regen.sh

# Regenerate docs/reference/kernels/index.md's catalogue from crates/kernels/wgsl/.
# Every column is derived from the sources, so the table cannot be edited by
# hand - and `check/generated/kernels-wgsl-table` is what stops it drifting.
## generate/kernels/wgsl-table - regenerate docs/reference/kernels/index.md from the .wgsl sources
generate/kernels/wgsl-table:
	scripts/build/gen-kernel-table.py

# Regenerate the NATIVE CUDA catalogue from crates/kernels-cuda's own registry.
# A sibling of the WGSL pair above, not a mode of it: that generator derives
# const names from .wgsl stems, compiles every entry on the test device and
# cross-checks five metadata fields against shader text - none of which can
# read CUDA C++. This one also fails when a .cu file exists that no registry
# entry embeds, which `include_str!` cannot catch (it proves registry -> file,
# never file -> registry). The other half of the CUDA tier contract - that an
# operator the policy declares tuned never runs as generated - is a Rust test
# (`brain-gpu-core`'s cuda_impl_policy), because it needs the real dispatch
# record, not the table.
## generate/kernels/cuda-table - regenerate docs/reference/kernels/cuda.md from crates/kernels-cuda
generate/kernels/cuda-table:
	scripts/build/gen-cuda-kernel-table.py

## generate/samples - lists the sample generators; writes nothing
generate/samples:
	$(LIST_CHILDREN)

# Regenerate docs/manifest.yaml's Examples part from the READMEs that actually
# exist under samples/, so a new sample cannot ship without reaching the
# published docs and the manual, and a deleted sample cannot leave a dangling one.
## generate/samples/manifest - regenerate docs/manifest.yaml's Examples part from the READMEs under samples/
generate/samples/manifest:
	scripts/build/gen-samples-manifest.py

## generate/knowledge - lists the knowledge-base generators; writes nothing
generate/knowledge:
	$(LIST_CHILDREN)

# Regenerate `.agents/knowledge/index.md` from the findings beside it. The
# knowledge base is one file per finding so that writing, citing and
# superseding one is an edit to that finding rather than to a file every
# other finding shares; the index is generated so it cannot drift from the
# directory.
## generate/knowledge/index - regenerate .agents/knowledge/index.md from the findings beside it
generate/knowledge/index:
	scripts/build/gen-knowledge-index.py

## generate/roadmap - lists the roadmap generators; writes nothing
generate/roadmap:
	$(LIST_CHILDREN)

# Regenerate `.agents/roadmap/index.md`, the listing of every status ledger.
## generate/roadmap/index - regenerate .agents/roadmap/index.md from the ledgers beside it
generate/roadmap/index:
	scripts/build/gen-roadmap-index.py

## testdata - lists the test-data targets; does nothing
testdata:
	$(LIST_CHILDREN)

# Populate the gitignored testdata/ tree (checkpoints/goldens/audio) that parity
# and integration tests read from $BRAIN_TESTDATA. Idempotent - fetches only what
# is missing, from a local mirror (hard-linked) or a URL. See scripts/data/fetch-testdata.sh.
## testdata/fetch - populate the gitignored testdata/ tree (checkpoints, goldens, audio); fetches only what is missing
testdata/fetch:
	bash scripts/data/fetch-testdata.sh

## hooks - lists the git hook targets; installs nothing
hooks:
	$(LIST_CHILDREN)

# Install the local git hooks into .git/hooks - a one-time-per-clone step,
# not run automatically, since it writes outside version control:
#   pre-commit  - the check/spdx gate above, plus check-no-doc-citations.sh
#                 (crates/scripts/tools/samples must never cite a docs/ or
#                 .agents/ file path) and check-repo-scope.sh (the check/scope
#                 gate, on the staged files)
#   commit-msg  - silently strips Co-Authored-By:/Claude-Session: trailer
#                 lines from every new commit message (never fails)
#   pre-push    - fails the push if a trailer line survived anyway, OR if a
#                 docs/.agents citation survived anyway (a full-tree re-run
#                 of check-no-doc-citations.sh, since the pre-commit hook's
#                 own check is diff-scoped and not guaranteed to fire on
#                 every commit a rebase replays); see scripts/hooks/pre-push,
#                 the trailer half shares its stripping logic with
#                 scripts/hooks/trailers.py
## hooks/install - install the local git hooks (SPDX check, commit trailer cleanup/gate); one-time per clone
hooks/install:
	install -m 755 scripts/hooks/pre-commit .git/hooks/pre-commit
	install -m 755 scripts/hooks/commit-msg .git/hooks/commit-msg
	install -m 755 scripts/hooks/pre-push .git/hooks/pre-push
	@echo "installed: .git/hooks/{pre-commit,commit-msg,pre-push}"

# ---- release -----------------------------------------------------------------
# Version bump + changelog + tag, mirroring vemu's release flow
# (applications/emulators/vemu/Makefile) with one deliberate difference: these
# targets stop at the local tag. Pushing and creating the GitHub release are a
# separate, explicit release/publish step below - not folded into
# release/{patch,minor,major} - because both need credentials (a working git
# remote, `gh auth login`) that a routine version bump should not silently
# assume are present.
## release - lists the release steps; does nothing (a release tags and publishes)
release:
	$(LIST_CHILDREN)

# Every tool `release/{patch,minor,major}` -> `_release/finalize` needs,
# checked BEFORE the first commit - bump2version commits immediately and
# unconditionally, so a missing tool discovered only once `_release/finalize`
# is already running (as `release/changelog`'s own `git-cliff` check used to do)
# strands the repo mid-release: version bumped and committed, but no
# changelog and no tag, with no clean way to resume (re-running the bump
# would skip straight past the stranded version). Fail here instead, while
# nothing has been touched yet.
_release/preflight:
	@command -v bump2version >/dev/null 2>&1 || \
		{ echo "error: bump2version not found - install with 'pip install bump2version'"; exit 1; }
	@command -v git-cliff >/dev/null 2>&1 || \
		{ echo "error: git-cliff not found - install with 'cargo install git-cliff'"; exit 1; }
	@[ -z "$$(git status --porcelain)" ] || \
		{ echo "error: dirty working tree"; git status --short; exit 1; }

## release/patch - bump the patch version, commit, tag (local only - see release/publish)
release/patch: _release/preflight
	bump2version --no-tag patch
	$(MAKE) _release/finalize

## release/minor - bump the minor version, commit, tag (local only - see release/publish)
release/minor: _release/preflight
	bump2version --no-tag minor
	$(MAKE) _release/finalize

## release/major - bump the major version, commit, tag (local only - see release/publish)
release/major: _release/preflight
	bump2version --no-tag major
	$(MAKE) _release/finalize

# Runs as a sub-make so BRAIN_VERSION is re-read AFTER bump2version rewrote
# the Makefile (a sub-make always re-parses the Makefile from disk; the
# parent make's variable value is stale the instant bump2version returns).
# Refreshes Cargo.lock for the new workspace version (`--locked` at the tag
# would otherwise fail), prepends the new CHANGELOG.md section, folds both
# into the bump commit, and creates the release tag on the final commit -
# same trick as vemu's _release/finalize, and the same reason its tag is
# created last: every generated file must already be IN the commit the tag
# points at, not added after.
_release/finalize:
	cargo update --workspace
	$(MAKE) release/changelog
	git add Cargo.lock CHANGELOG.md
	git commit --amend --no-edit
	git tag "v$(BRAIN_VERSION)"

# Prepend the section for the version just bumped to CHANGELOG.md, generated
# from conventional-commit history since the last tag (git-cliff, cliff.toml
# at the repo root). `--prepend` only inserts at the top - the curated prose
# above older sections (1.0.0's subsystem intro) is left alone.
## release/changelog - prepend the current version's section to CHANGELOG.md (git-cliff)
release/changelog:
	@command -v git-cliff >/dev/null 2>&1 || \
		{ echo "error: git-cliff not found - install with 'cargo install git-cliff'"; exit 1; }
	@# --prepend requires the target to already exist (git-cliff has no
	@# create-if-missing mode); a fresh repo's first release starts it empty.
	touch CHANGELOG.md
	git-cliff --config cliff.toml --tag "v$(BRAIN_VERSION)" --unreleased --prepend CHANGELOG.md

# Render just the tag's own section, no running header - this is the GitHub
# release body, not the cumulative changelog. Sliced out of CHANGELOG.md
# itself (not re-derived with `git-cliff --latest`), because CHANGELOG.md may
# carry hand-curated prose above the auto-generated commit list (1.0.0's
# subsystem intro) that a fresh git-cliff run has no way to know about.
## release/notes - write the tag's own CHANGELOG section to target/release-notes-v<version>.md
release/notes:
	@[ -f CHANGELOG.md ] || { echo "error: no CHANGELOG.md - run 'make release/changelog' first"; exit 1; }
	mkdir -p target
	awk -v ver="$(BRAIN_VERSION)" '\
		/^## \[/ { if (in_section) exit; if (index($$0, "## [" ver "]") == 1) { in_section = 1; next }; next } \
		in_section { print }' CHANGELOG.md > target/release-notes-v$(BRAIN_VERSION).md
	@[ -s target/release-notes-v$(BRAIN_VERSION).md ] || \
		{ echo "error: v$(BRAIN_VERSION) section not found in CHANGELOG.md"; exit 1; }
	@echo "wrote target/release-notes-v$(BRAIN_VERSION).md"

# The one target in this section that touches the remote: push the tagged
# bump commit and its tag (only that tag), then publish the signed package to
# the tag's GitHub release. Deliberately not part of release/{patch,minor,major}
# - see that section's note above.
## release/publish - push HEAD and the tag v<version> to RELEASE_REMOTE, then package/deb/publish
release/publish:
	git push $(RELEASE_REMOTE) HEAD "v$(BRAIN_VERSION)"
	$(MAKE) package/deb/publish

# ---------------------------------------------------------------------------
##@ Housekeeping

## clean - cargo clean, and remove out/
clean:
	cargo clean
	rm -rf $(OUT)

## help - show every target as a tree, grouped by section
help:
	@bash scripts/make-help.sh

# Declared last so every generated list above is already defined.
.PHONY: build build/debug build/release \
        package package/deb package/deb/debug package/deb/release package/deb/publish \
        container container/runtime container/ci container/smoke \
        cuda cuda/aot cuda/install cuda/coverage gh200 gh200/probe gh200/placement \
        environment environment/requirements environment/full environment/openvino \
        npu npu/diagnose \
        test test/nextest test/package test/doc test/slow test/full test/times test/capability-report \
        test/gradcheck test/rl test/conformance test/cuda-matrix \
        test/parity test/parity/strict test/parity/forecast test/parity/wan \
        test/e2e test/e2e/api-conformance test/e2e/shutdown test/e2e/samples test/e2e/ready \
        test/e2e/claude-code test/e2e/scheduler test/e2e/quickstart test/e2e/deb \
        test/e2e/results-hygiene test/e2e/s3dit-int8 \
        check check/workspace check/clippy check/scripts check/env-reads check/test-skips check/seed-literals check/seed-literals/tests check/serve-conformance test/gates test/gates/env-reads test/gates/test-skips test/gates/seed-literals test/gates/group-kernels _train/need-wrapper qwen35 qwen35/train qwen35/train/probe qwen35/train/step check/spdx check/paths check/scope check/saved-policy check/hf-fixtures \
        check/files check/samples check/sdk-features check/make-targets \
        check/generated check/generated/kernels-wgsl-table check/generated/kernels-cuda-table \
        check/generated/samples-manifest \
        check/generated/knowledge-index \
        gpt $(GPT_DATASET_TARGETS) $(GPT_SYNTHETIC_DATA_TARGETS) gpt/timeseries/data \
        gpt/shakespeare-char/data gpt/gpt/data $(GPT_TRAIN_TARGETS) $(GPT_EVAL_TARGETS) \
        yolo yolo/data yolo/train yolo/eval yolo/detect \
        yolo/npu yolo/npu/export yolo/npu/quantize yolo/npu/sim-int8 yolo/npu/run yolo/npu/bench \
        qwen qwen/finetune depth depth/demo depth/smoke depth/camera depth/train \
        mirror mirror/import mirror/infer mirror/demo splat splat/view tts tts/data \
        flux2 flux2/generate flux2/edit wan wan/t2v forecast forecast/compare forecast/serve \
        wm wm/play wm/fixtures federated federated/demo web web/dev web/build \
        bench bench/run $(BENCH_RUN_TARGETS) bench/scaling bench/distributed bench/eval bench/scale bench/advise \
        bench/compare bench/char \
        perf perf/scenario $(PERF_SCENARIO_TARGETS) perf/compare perf/smoke \
        perf/model perf/model/lfm perf/model/flux2 perf/model/wan \
        perf/gate perf/gate/forecast perf/gate/wm perf/gate/qwen-serving \
        samples $(SAMPLE_CATEGORY_TARGETS) $(SAMPLE_NAME_TARGETS) \
        $(SAMPLE_BUILD_TARGETS) $(SAMPLE_RUN_TARGETS) \
        crates $(EXAMPLE_CRATE_TARGETS) $(EXAMPLE_GROUP_TARGETS) $(EXAMPLE_NAME_TARGETS) \
        $(EXAMPLE_BUILD_TARGETS) $(EXAMPLE_RUN_TARGETS) \
        docs docs/build docs/check docs/clean docs/pdf docs/quickstart \
        papers papers/pdf $(PAPER_DIRS) $(PAPER_TARGETS) \
        generate generate/kernels generate/kernels/registry generate/kernels/wgsl-table \
        generate/kernels/cuda-table generate/samples generate/samples/manifest \
        generate/knowledge generate/knowledge/index \
        testdata testdata/fetch hooks hooks/install \
        release release/patch release/minor release/major release/changelog release/notes \
        release/publish _release/preflight _release/finalize \
        clean help
