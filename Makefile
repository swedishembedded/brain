# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# brain — top-level workflow.
#
# Pure Rust + WGSL training/eval engine. These targets generate datasets, train
# the GPT baseline, evaluate it (perplexity + task exact-match), run the
# gradient-check correctness gate, and drive the federated-MoE artifact pipeline.
# The browser/WebGPU demo lives in crates/web (delegated below).
#
# Quick start:
#   make release
#   make data/calculator
#   make train/gpt/calculator
#   make eval/gpt/calculator
#   make bench           # train+eval GPT on the shared char datasets
#   make gradcheck       # backprop correctness gate
#   make test            # full cargo test suite

BRAIN  ?= ./target/release/brain
# The debug-build binary the e2e bats suites drive by default (they build via the
# `build` target, not `release`, so the fast lane stays fast). Override to point
# an e2e run at a release build instead: `BRAIN_BIN=./target/release/brain make
# test/e2e/api-conformance`.
BRAIN_BIN ?= ./target/debug/brain
PIP    ?= python3 -m pip
DATA   ?= data
OUT    ?= out
SEED   ?= 1337
STEPS  ?= 1000
N      ?= 100000
ARCH   ?= gpt

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

.PHONY: help build release deb deb/debug deb/release test/doc test/slow test/full test/times wm/play wm-fixtures test gradcheck kernels-regen kernels-table kernels-table/check parity requirements environment environment/openvino npu-diagnose bench bench/char bench/eval bench/scale bench/advise bench/compare perf perf/compare perf/smoke clean federated-demo depth/demo depth/smoke depth/camera train/zipdepth mirror/import mirror/infer mirror/demo splat/view \
        data/calculator data/reverser data/wordcalc data/timeseries \
        data/shakespeare_char data/gpt data/detect data/tts \
        train/yolo eval/yolo detect/yolo train/qwen/lora \
        export/yolo-onnx quantize/yolo sim/yolo-int8 run/yolo-npu bench/yolo-npu \
        web/dev web/build forecast/compare forecast/serve forecast/parity forecast/perf-gate wm/perf-gate fetch/testdata \
        clippy check/scripts check/spdx hooks/install qwen/serving-perf-gate \
        test/e2e test/e2e/claude-code test/e2e/api-conformance test/e2e/shutdown test/e2e/examples test/e2e/scheduler test/e2e/ready \
        perf/lfm perf/flux2 flux2/generate flux2/edit zimage/int8-e2e

help:
	@echo "brain targets:"
	@echo "  make release                 build the optimized 'brain' binary"
	@echo "  make requirements            pip-install the Python tooling (OpenVINO/NPU, torch, ...)"
	@echo "  make environment             requirements + detect/verify a real Intel NPU (no-op if absent)"
	@echo "  make environment/openvino    OpenVINO/NPU setup only, skips torch/CUDA (fast iteration)"
	@echo "  make test                    FAST lane: unit+integration, no doc-tests"
	@echo "  make test/doc                doc-tests (slow: one link per crate)"
	@echo "  make test/slow               #[ignore]d long-running tests"
	@echo "  make test/full               test + test/doc + test/slow"
	@echo "  make test/times              rank test binaries by wall time"
	@echo "  make test/e2e                every fast end-to-end bats suite (api-conformance|"
	@echo "                               shutdown|examples|ready; test/e2e/<name> runs one;"
	@echo "                               claude-code + scheduler are heavier, opt-in)"
	@echo "  make clippy                  the clippy ratchet gate (exit 0 + no new warnings)"
	@echo "  make check/scripts           scripts/tools self-validation + env-var doc gate"
	@echo "                               + no-doc-citations + no-perf-numbers gates"
	@echo "  make check/spdx              SPDX-License-Identifier + copyright header gate"
	@echo "  make hooks/install           install the local git hooks (SPDX check, commit"
	@echo "                               trailer cleanup/gate)"
	@echo "  make gradcheck               the FULL numerical backprop gate (every model check"
	@echo "                               + kernel FD suite in crates/gradcheck)"
	@echo "  make parity                  cross-backend parity gate: CPU == Vulkan == NPU"
	@echo "  make forecast/parity         forecasting fp32-exactness gate"
	@echo "  make forecast/perf-gate      forecasting perf regression gate (vs baselines)"
	@echo "  make wm/perf-gate            world-model perf regression gate (vs baselines)"
	@echo "  make qwen/serving-perf-gate  qwen serving perf regression gate (vs baselines)"
	@echo "  make kernels-table           regenerate docs/reference/kernels.md from the .wgsl sources"
	@echo "  make kernels-table/check     fail if that catalogue has drifted (part of test/full)"
	@echo "  make data/<name>             generate a dataset (calculator|reverser|wordcalc|"
	@echo "                               timeseries|shakespeare_char|gpt) into $(DATA)/<name>"
	@echo "  make train/gpt/<name>        train GPT on a dataset -> $(OUT)/gpt-<name>.safetensors"
	@echo "  make eval/gpt/<name>         perplexity + exact-match for a trained GPT"
	@echo "  make data/detect             synthetic object-detection dataset -> $(DATA)/detect"
	@echo "  make train/yolo              train tiny YOLO on it -> $(OUT)/yolo.safetensors"
	@echo "  make eval/yolo               mAP@0.5 + precision/recall for the trained YOLO"
	@echo "  make detect/yolo             run detection on a sample image (JSON boxes)"
	@echo "  make bench                   run the architecture-evaluation benchmark suite (all)"
	@echo "  make bench/<name>            run one benchmark (e.g. bench/mqar)"
	@echo "  make bench/scaling           scaling-law sweep: fit L(N)=E+A*N^-alpha across sizes"
	@echo "  make bench/eval ARCH=<name>  run the WHOLE battery vs one architecture (gpt|gpt-small|"
	@echo "                               gpt-wide), aggregate per axis -> results/<arch>-<seed>.json"
	@echo "  make bench/scale ARCH=<name> predictive per-capability scaling: score@2x/@4x per axis"
	@echo "  make bench/advise ARCH=<name> ranked tuning recommendations from eval(+scale) artifacts"
	@echo "  make bench/compare           side-by-side leaderboard of every results/<arch>-<seed>.json"
	@echo "  make bench/char              train+eval GPT on the shared char datasets (legacy)"
	@echo "  make perf                    PERFORMANCE suite: latency/throughput/serve/sweep"
	@echo "                               (how fast + at what cost; docs/performance/benchmarking.md)"
	@echo "  make perf/<scenario>         one scenario (perf/sweep, perf/serve, ...)"
	@echo "  make perf/compare            leaderboard over results/perf-*.json"
	@echo "  make perf/smoke              CI-sized run of every perf scenario"
	@echo "  make perf/lfm                LFM2.5 8k-context concurrency benchmark (LFM_WEIGHTS/"
	@echo "                               LFM_TOKENIZER select the model)"
	@echo "  make perf/flux2              FLUX.2 Klein denoise-step benchmark (BRAIN_FLUX2_* env)"
	@echo "  make train/qwen/lora         LoRA-finetune a Qwen checkpoint (DATASET=<dir> ADAPTER=<ref>)"
	@echo "  make flux2/generate          generate one image with FLUX.2 Klein (BRAIN_FLUX2_* env)"
	@echo "  make flux2/edit              reference-image edit with FLUX.2 Klein (FLUX2_REF=...)"
	@echo "  make zimage/int8-e2e         real z-image int8 e2e generation (no-OOM regression)"
	@echo "  make data/tts                synthetic TTS text->codes dataset (Talker SFT smokes)"
	@echo "  make docs                    build the full docs bundle (pandoc + xelatex)"
	@echo "  make wm/play                  play the fake world model in an SDL window (WASD)"
	@echo "  make wm-fixtures             regenerate DIAMOND parity fixtures (needs torch)"
	@echo "  make federated-demo          MoE train -> split -> verify -> merge round-trip"
	@echo "  make web/dev | web/build     WebGPU browser demo (crates/web)"

build:
	cargo build

release:
	cargo build --release

# Build self-contained Debian packages for package-only integrations.
deb: deb/release

deb/debug: build
	bash scripts/build/build-deb.sh --binary target/debug/brain

deb/release: release
	bash scripts/build/build-deb.sh

# ---- tests -----------------------------------------------------------------
# `make test` is the FAST LANE and must stay fast: unit + integration tests
# only, reusing the release build.
#
# Three things were making the full suite take ~an hour, all measured:
#
#  1. `cargo test` defaults to the DEBUG profile, so it recompiled the whole
#     workspace even right after `make release`. Tests now run `--release` and
#     reuse that build.
#  2. Doc-tests link one binary per crate against the full graph — ~18s per
#     crate for ~30 examples, most of them `no_run`. They are real coverage but
#     they are not fast feedback, so they get their own lane.
#  3. Many concurrent GPU DEVICES in one process deadlock the NVIDIA driver
#     (~50% of parallel runs), and a device leaked into process exit crashes
#     it. FIXED at the root: models share one device (Gpu::share/new_like) and
#     test binaries use the weak-pool fixture gpu_core::testgpu, whose device
#     dies with its last in-process handle — kronos is proven clean at
#     --test-threads=48, and after the fixture migration every GPU-test crate
#     either shares the pooled device (qwen/gpt/tts/speaker/glm/moe/pid/
#     seq2seq/autoencoder/chronos2/fincast/splat/model/wm-core/yolo), pins the
#     CPU backend (never touches the card), or gates multi-device tests on the
#     hardware being present. 8 is the suite-wide proven point (qwen + the six
#     heaviest migrated crates, 0 failures).
#
# The timeout turns a deadlock into a fast, loud failure instead of an hour of
# silence. Override any of these: TEST_THREADS=1 make test restores serial.
TEST_THREADS ?= 8
# The guard is a DEADLOCK detector, not a performance target: it must sit above
# the measured run time so a completing suite never reads as a hang. The VLM
# crates (fastvlm/qwenvl/moondream) grew the suite past the old 1500s budget —
# 266 suites / 1492 tests now complete at TEST_THREADS=8 inside ~2000s wall on
# the 2xP40 box, so the guard sits at 2400. Making the suite faster is real
# work (slow-lane moves), not a tighter timeout.
TEST_TIMEOUT ?= 2400
CARGO_TEST   ?= cargo test --release --offline

# Build first WITHOUT the timeout, then run WITH it. The deadlock guard is a
# statement about *running tests* — a cold rebuild after an engine change takes
# minutes on its own, and letting it eat the budget turns "compiling" into a
# false "TIMED OUT" that reads like a hang.
# The clippy gate — a ratchet. Checks that clippy EXITS 0 (it stops at the first
# deny-by-default lint and then silently reports nothing about everything after
# it) and that the warning count has not grown. See scripts/gates/clippy-gate.sh.
clippy:
	@scripts/gates/clippy-gate.sh

test:
	@echo "test: fast lane (unit + integration, no doc-tests, GPU serialised)"
	@$(CARGO_TEST) --lib --bins --tests --no-run
	@timeout $(TEST_TIMEOUT) $(CARGO_TEST) --lib --bins --tests -- --test-threads=$(TEST_THREADS); \
	rc=$$?; \
	if [ $$rc -eq 124 ]; then \
		echo; echo "TIMED OUT after $(TEST_TIMEOUT)s of RUNNING — almost certainly a deadlock."; \
		echo "Find it with:  scripts/gates/test-times.sh --top 10"; \
	fi; \
	exit $$rc

# Doc-tests: real coverage, but each crate links its own binary, so this is
# minutes of linking for a handful of examples. Separate lane, not the default.
test/doc:
	$(CARGO_TEST) --doc

# Tests marked `#[ignore = "slow: ..."]` — long training/parity runs that do not
# belong in fast feedback.
test/slow:
	$(CARGO_TEST) --lib --bins --tests -- --ignored --test-threads=$(TEST_THREADS)

# Self-validation for scripts/ and tools/: every one parses, every one is named
# somewhere else in the repo (Makefile target / bats test / crate doc comment /
# doc — an orphan gate), and no non-overridable absolute machine path. See
# scripts/gates/check-scripts.sh for the full rationale. check-env-docs.sh
# additionally requires every BRAIN_* env var read anywhere in crates/ to be
# documented in docs/using/configuration.md, a docs/models/<model>.md page, or
# .agents/rules/testing.md (env-only config MUST have a reference).
# check-no-doc-citations.sh additionally requires that crates/, scripts/,
# tools/, and examples/ never cite a docs/ or .agents/ file path — see that
# script for why (also wired as a pre-commit hook, so this is a slow-path
# backstop for anything pre-commit was bypassed for).
# check-no-perf-numbers.sh additionally denies a bare number next to a
# performance unit/claim (ms, s, fps, tok/s, % of peak, Nx speedup, ...)
# anywhere in docs/**/*.md unless reviewed via a `<!-- perf-number: ... -->`
# comment - see that script for the full rationale and escape-hatch syntax.
check/scripts:
	bash scripts/gates/check-scripts.sh
	bash scripts/gates/check-env-docs.sh
	bash scripts/gates/check-no-doc-citations.sh
	bash scripts/gates/check-no-perf-numbers.sh

# SPDX/copyright header gate: every Rust/C/Python/shell/Makefile/WGSL/...
# source file must carry exactly one "SPDX-License-Identifier: Apache-2.0"
# line, immediately followed by the copyright line. scripts/spdx/rules.py has
# the file-selection rules (shared with scripts/spdx/check.py); `make
# hooks/install` wires the same check into a git pre-commit hook so a
# non-compliant commit is refused locally, not just caught here.
check/spdx:
	python3 scripts/spdx/check.py $$(git ls-files)

# Install the local git hooks into .git/hooks — a one-time-per-clone step,
# not run automatically, since it writes outside version control:
#   pre-commit  - the check/spdx gate above, plus check-no-doc-citations.sh
#                 (crates/scripts/tools/examples must never cite a docs/ or
#                 .agents/ file path)
#   commit-msg  - silently strips Co-Authored-By:/Claude-Session: trailer
#                 lines from every new commit message (never fails)
#   pre-push    - fails the push if a trailer line survived anyway, OR if a
#                 docs/.agents citation survived anyway (a full-tree re-run
#                 of check-no-doc-citations.sh, since the pre-commit hook's
#                 own check is diff-scoped and not guaranteed to fire on
#                 every commit a rebase replays); see scripts/hooks/pre-push,
#                 the trailer half shares its stripping logic with
#                 scripts/hooks/trailers.py
hooks/install:
	install -m 755 scripts/hooks/pre-commit .git/hooks/pre-commit
	install -m 755 scripts/hooks/commit-msg .git/hooks/commit-msg
	install -m 755 scripts/hooks/pre-push .git/hooks/pre-push
	@echo "installed: .git/hooks/{pre-commit,commit-msg,pre-push}"

# Everything, for a release gate.
test/full: test test/doc test/slow test/e2e check/scripts check/spdx kernels-table/check

# Rank every test binary by wall time; --budget fails if any exceeds it. This is
# what keeps the fast lane fast.
test/times: release
	scripts/gates/test-times.sh --top 15

# End-to-end: drive the real `claude` CLI against a local `brain serve --anthropic`,
# proving brain works as a Claude Code backend. Skips cleanly unless `claude` is
# installed AND a served qwen model is configured:
#   BRAIN_QWEN_WEIGHTS=... BRAIN_QWEN_TOKENIZER=... make test/e2e/claude-code
# (import one first: brain qwen import --hf <hf_qwen_dir> --out qwen.safetensors)
test/e2e/claude-code: release
	bats tests/e2e/claude_code.bats

# End-to-end: HTTP API conformance over a real socket against a single `brain serve`
# backed by the built-in deterministic mock model (BRAIN_MOCK=1) — no weights, no GPU,
# no `claude`. Validates every provider dialect (OpenAI/Anthropic/OpenRouter) against
# the vendored OpenAPI specs. Fast + deterministic. Needs only a debug/release brain
# binary + jq (+ optional Python jsonschema for full schema validation).
#   make test/e2e/api-conformance   (or: BRAIN_BIN=./target/debug/brain bats tests/e2e/api_conformance.bats)
test/e2e/api-conformance: build
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/api_conformance.bats

# End-to-end: `brain serve` actually stops on SIGINT/SIGTERM, for every combination
# of surfaces it can be told to serve (D-Bus alone, D-Bus+HTTP together, HTTP
# alone). Each test starts and kills its own server; the D-Bus cases use a private
# per-test dbus-daemon, never the real session/system bus. Needs a debug/release
# binary + a working dbus-daemon.
test/e2e/shutdown: build
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/shutdown.bats

# End-to-end: every example under examples/ is actually exercised — the harness
# that did not exist when they all silently rotted after the P19 brain-py rewrite.
# ONE shared BRAIN_MOCK=1 server (D-Bus + Anthropic HTTP); each example that CAN
# run against the weight-free mock does so for real, the rest skip honestly with
# a printed reason. A completeness check fails the suite if a tracked example is
# missing from tests/e2e/examples/manifest.tsv (or vice versa), so a new, unwired
# example cannot silently rot the way these did. Needs a debug/release binary,
# dbus-daemon, curl, and `pip install -e brain-py` (jeepney) on EXAMPLES_PY
# (default python3).
#   make test/e2e/examples   (or: EXAMPLES_PY=/path/to/python3 bats tests/e2e/examples.bats)
test/e2e/examples: build
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/examples.bats

# Heavy, opt-in: brain's residency scheduler (batching/eviction) + the generate ->
# detect -> annotate demo against REAL model weights and a GPU. NOT part of
# test/e2e (that's test/e2e/examples' job, against the mock) — see
# tests/e2e/scheduler.bats for the required env vars.
test/e2e/scheduler:
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/scheduler.bats

# End-to-end: `brain serve --ready-file PATH` fires only once EVERY requested
# surface (HTTP dialects + D-Bus) has actually bound, never on a failed or
# partial bind, and strictly AFTER --api-keys-out is written — so a script can
# wait on PATH alone and then read the keys with no retry. BRAIN_MOCK=1,
# CPU-only, no real weights. Needs a debug/release binary + jq + curl (+
# dbus-daemon/busctl for the D-Bus cases, which skip cleanly without them).
test/e2e/ready: build
	BRAIN_BIN=$(BRAIN_BIN) bats tests/e2e/ready.bats

# Every fast (no real weights, no GPU) end-to-end bats suite, in one target.
test/e2e: test/e2e/api-conformance test/e2e/shutdown test/e2e/examples test/e2e/ready

# Install the Python tooling (OpenVINO/NPU runtime, torch + transformers for the
# benchmark reference rows, etc.) into the current environment. The Rust engine
# needs none of these — this is for tools/ and the `--device npu` runtime.
requirements:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# `requirements` + hardware-specific setup that a package list alone can't
# cover: detects a real Intel NPU (if any -- a no-op, exit 0 otherwise),
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
# own iGPU/NPU paths don't use torch at all -- but `make environment` still
# pays for the download/disk because `requirements` installs torch for the
# benchmark reference rows (tools/bench/*, tools/goldens/*), which do need it.
environment: requirements
	scripts/build/setup-npu-runtime.sh

# OpenVINO/NPU setup only, skipping the rest of requirements.txt (torch +
# its CUDA deps, transformers, ultralytics, ...) -- use this to iterate on
# NPU detection/driver setup without re-installing the whole Python stack.
environment/openvino:
	scripts/build/setup-npu-runtime.sh

# Read-only diagnosis, NOT install: runs every NPU check this repo has hit
# (device node/driver binding/kernel version, firmware, kernel debugfs
# reset/fault counters, the userspace libze_intel_vpu.so.1 compat symlink,
# OpenVINO's own device enumeration run twice to catch flaky/wedged
# behavior, and brain's own crates/npu test) and prints one clear verdict.
# Use this to answer "is the NPU actually accessible right now", `environment`/
# `environment/openvino` above to install/fix. See scripts/build/
# npu-diagnose.sh's header for exit codes.
npu-diagnose:
	scripts/build/npu-diagnose.sh $(if $(VERBOSE),--verbose)

# The documented backprop gate must BE the real gate: `brain gradcheck` runs
# exactly one model (GPT), while the ~20 model checks + kernel FD suites live
# in crates/gradcheck's tests. Point the target at those (audit F17).
gradcheck:
	$(CARGO_TEST) -p brain-gradcheck

# Regenerate the kernel const block + ALL registry in crates/kernels/src/lib.rs
# from the contents of crates/kernels/wgsl/. Run after adding/removing a .wgsl
# file; merge conflicts in lib.rs are resolved by union-ing wgsl/ + this target.
kernels-regen:
	scripts/build/kernels-regen.sh

# Regenerate docs/reference/kernels.md's catalogue from crates/kernels/wgsl/.
# Every column is derived from the sources, so the table cannot be edited by
# hand — and `kernels-table/check` is what stops it drifting silently, which
# is the failure mode .agents/rules/lessons.md #29 records for the generator above.
kernels-table:
	scripts/build/gen-kernel-table.py

kernels-table/check:
	scripts/build/gen-kernel-table.py --check

# Regenerate the DIAMOND parity fixtures (gitignored — never committed) from
# the reference implementation in resources/world-models/repos/diamond.
# Needs python3 + torch; see docs/world-models/FIXTURES.md for provenance.
wm-fixtures:
	python3 scripts/parity-dump/diamond.py --out crates/wm-diamond/tests/fixtures/diamond

# Play the deterministic fake world model in an SDL window (WASD; Esc quits).
# The SDL window is always compiled into the standard build (needs system
# libSDL2 at link); it only OPENS when a run needs it.
wm/play: release
	./target/release/brain wm play --model fake

# Cross-backend parity gate: CPU == Vulkan == NPU (gradcheck on both backends +
# direct CPU-vs-GPU forward parity + TTS NPU codec vs CPU reference).
parity:
	scripts/gates/parity-gate.sh

# Forecasting correctness gate: every time-series optimization stays fp32-exact
# (kronos KV-cache/shared-prefill/cross-section + batched-training parity).
forecast/parity:
	scripts/gates/forecast-parity-gate.sh

# Forecasting latency regression gate: each forecaster through `brain perf run`
# vs the committed baseline (scripts/gates/forecast-perf-baselines/, `--update` to
# refresh). Weights via env (BRAIN_KRONOS_*/BRAIN_CHRONOS2/BRAIN_FINCAST).
forecast/perf-gate: release
	scripts/gates/forecast-perf-gate.sh

# World-model fps regression gate (best-of-3 vs scripts/gates/wm-perf-baselines.json,
# hard floors only). Dev-box gate, not CI: needs out/diamond-breakout.weights
# (brain wm import ...) and a real display/GPU. `--update` rewrites baselines.
wm/perf-gate: release
	scripts/gates/wm-perf-gate.sh

# Concurrent-serving-performance regression gate: the real HTTP-served path
# (http:qwen-synth: target, random weights, no checkpoint needed) through
# `brain perf run sweep` at concurrency 1,2, vs the committed baseline
# (scripts/gates/qwen-serving-perf-baselines/, `--update` to refresh).
# Needs a real tokenizer via QWEN_TOKENIZER; SKIPS (not fails) when unset.
qwen/serving-perf-gate: release
	scripts/gates/qwen-serving-perf-gate.sh

# ---- data generation ------------------------------------------------------
data/calculator data/reverser data/wordcalc: release
	$(BRAIN) data gen $(@F) --out $(DATA)/$(@F) --n $(N) --seed $(SEED)

data/timeseries: release
	$(BRAIN) data gen timeseries --out $(DATA)/timeseries --n 200000 --seed $(SEED)

# Synthetic Qwen3-TTS `text -> codebook-0 codes` stream (for Talker SFT smokes).
data/tts: release
	$(BRAIN) data gen tts --out $(DATA)/tts --n $(N) --seed $(SEED)

# Populate the gitignored testdata/ tree (checkpoints/goldens/audio) that parity
# and integration tests read from $BRAIN_TESTDATA. Idempotent — fetches only what
# is missing, from a local mirror (hard-linked) or a URL. See scripts/data/fetch-testdata.sh.
fetch/testdata:
	bash scripts/data/fetch-testdata.sh

$(DATA)/shakespeare_char/input.txt:
	mkdir -p $(DATA)/shakespeare_char && curl -sSL -o $@ $(SHAKE_URL)
data/shakespeare_char: release $(DATA)/shakespeare_char/input.txt
	$(BRAIN) data gen shakespeare_char --out $(DATA)/shakespeare_char

$(DATA)/gpt/input.txt:
	mkdir -p $(DATA)/gpt && curl -sSL -o $@ $(SHAKE_URL)
data/gpt: release $(DATA)/gpt/input.txt
	$(BRAIN) data gen gpt --out $(DATA)/gpt

# ---- train (pattern: train/gpt/<dataset>) ---------------------------------
# LHS=RHS datasets train with answer-masking; shakespeare_char does not.
MASK_calculator := --mask =
MASK_reverser   := --mask =
MASK_wordcalc   := --mask =

train/gpt/%: release
	@mkdir -p $(OUT)
	$(BRAIN) gpt train $(DATA)/$* --out $(OUT)/gpt-$*.safetensors \
		--steps $(STEPS) --batch $(BATCH) --block $(BLOCK) \
		--layers $(LAYERS) --d-model $(DMODEL) --heads $(HEADS) --lr $(LR) \
		--seed $(SEED) $(MASK_$*)

# ---- eval (pattern: eval/gpt/<dataset>) -----------------------------------
eval/gpt/%: release
	$(BRAIN) gpt eval --weights $(OUT)/gpt-$*.safetensors --data $(DATA)/$*

# ---- Qwen LoRA fine-tuning: the "one command to fully retrain and overwrite
# the lora checkpoint" from applications/edgeai/brain/.todo/bench-training.md.
# DATASET is a bench `datasets build` output dir (train.jsonl [+ validation.jsonl]);
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

train/qwen/lora: release
	@test -n "$(DATASET)" || (echo "usage: make train/qwen/lora DATASET=<dir> ADAPTER=<owner/name[:tag]>" && exit 1)
	@test -n "$(ADAPTER)" || (echo "usage: make train/qwen/lora DATASET=<dir> ADAPTER=<owner/name[:tag]>" && exit 1)
	$(BRAIN) qwen finetune --lora $(LORA_RANK) --alpha $(LORA_ALPHA) \
		--weights $(QWEN_BASE) --adapter $(ADAPTER) --dataset $(DATASET) \
		--steps $(QWEN_STEPS) --lr $(QWEN_LR) --batch $(QWEN_BATCH) --block $(QWEN_BLOCK) \
		--seed $(SEED) $(if $(MODELS_DIR),--models-dir $(MODELS_DIR)) $(if $(DATASET_ID),--dataset-id $(DATASET_ID))

# ---- YOLO detector (synthetic detection dataset) --------------------------
# `make data/detect` generates a synthetic object-detection dataset (RGB shapes
# + exact GT boxes) at the tiny YOLO's native 128px geometry. `train/yolo`
# trains the tiny detector on it; `eval/yolo` reports mAP@0.5/precision/recall;
# `detect/yolo` runs inference on a sample image (dataset image 0) and prints the
# boxes as JSON lines. All CPU-friendly (the YOLO model runs on the CPU backend).
data/detect: release
	$(BRAIN) data gen detect --out $(DATA)/detect --n $(YOLO_N) --seed $(SEED)

train/yolo: release
	@mkdir -p $(OUT)
	$(BRAIN) yolo train $(DATA)/detect --out $(OUT)/yolo.safetensors \
		--steps $(YOLO_STEPS) --batch $(YOLO_BATCH) --lr $(YOLO_LR) --seed $(SEED)

eval/yolo: release
	$(BRAIN) yolo eval --weights $(OUT)/yolo.safetensors --data $(DATA)/detect \
		--conf $(YOLO_CONF) --iou $(YOLO_IOU)

detect/yolo: release
	$(BRAIN) yolo detect --weights $(OUT)/yolo.safetensors --image $(DATA)/detect \
		--conf $(YOLO_CONF) --iou $(YOLO_IOU)

# ---- depth (ZipDepth) ------------------------------------------------------
# Set ZIPDEPTH_PTH to a released checkpoint (see resources/depth-models).
ZIPDEPTH_PTH ?= 
DEPTH_IMG    ?= 

depth/demo: release
	@test -n "$(ZIPDEPTH_PTH)" || (echo "set ZIPDEPTH_PTH=<zipdepth_base.pth>"; exit 2)
	@test -n "$(DEPTH_IMG)"    || (echo "set DEPTH_IMG=<image.ppm>"; exit 2)
	$(BRAIN) depth --image $(DEPTH_IMG) --weights $(ZIPDEPTH_PTH)

depth/smoke: release
	@test -n "$(ZIPDEPTH_PTH)" || (echo "set ZIPDEPTH_PTH=<zipdepth_base.pth>"; exit 2)
	@test -n "$(DEPTH_IMG)"    || (echo "set DEPTH_IMG=<image.ppm>"; exit 2)
	DISPLAY= $(BRAIN) depth --image $(DEPTH_IMG) --weights $(ZIPDEPTH_PTH) \
		--headless --out $(OUT)/depth.ppm

depth/camera: release
	@test -n "$(ZIPDEPTH_PTH)" || (echo "set ZIPDEPTH_PTH=<zipdepth_base.pth>"; exit 2)
	$(BRAIN) depth --camera --weights $(ZIPDEPTH_PTH) $(DEPTH_ARGS)

# WorldMirror-2 (multi-view 3D reconstruction). MIRROR_CKPT = the reference
# model.safetensors (or its HF dir); the converted .safetensors is what infer uses.
MIRROR_CKPT    ?=
MIRROR_WEIGHTS ?= $(OUT)/mirror.safetensors

mirror/import: release
	@test -n "$(MIRROR_CKPT)" || (echo "set MIRROR_CKPT=<model.safetensors|hf_dir>"; exit 2)
	$(BRAIN) mirror import $(MIRROR_CKPT) --out $(MIRROR_WEIGHTS)

# 3DGS scene viewer (interactive fly-through; WASD + mouse, see --help).
SPLAT_SCENE ?=

splat/view: release
	@test -n "$(SPLAT_SCENE)" || (echo "set SPLAT_SCENE=<scene.ply>"; exit 2)
	$(BRAIN) splat view $(SPLAT_SCENE) $(SPLAT_ARGS)

# images -> 3DGS scene (+ view). MIRROR_IMAGES = dir of .ppm or comma list.
MIRROR_IMAGES ?=

mirror/infer: release
	@test -n "$(MIRROR_IMAGES)" || (echo "set MIRROR_IMAGES=<dir|a.ppm,b.ppm>"; exit 2)
	$(BRAIN) mirror infer --weights $(MIRROR_WEIGHTS) --images $(MIRROR_IMAGES) $(MIRROR_ARGS)

mirror/demo: release
	@test -n "$(MIRROR_IMAGES)" || (echo "set MIRROR_IMAGES=<dir|a.ppm,b.ppm>"; exit 2)
	$(BRAIN) mirror demo --weights $(MIRROR_WEIGHTS) --images $(MIRROR_IMAGES) $(MIRROR_ARGS)

# Train ZipDepth end to end on the synthetic RGB->depth pairs (placeholder data,
# real loop: forward -> masked L1 -> backward -> AdamW; loss printed per step).
# Fine-tune a released checkpoint instead with ZIPDEPTH_PTH set.
train/zipdepth: release
	$(BRAIN) depth train --out $(OUT)/zipdepth.safetensors --steps 50 --batch 2 \
		$(if $(ZIPDEPTH_PTH),--weights $(ZIPDEPTH_PTH),)

# ---- Intel NPU deployment (OpenVINO) --------------------------------------
# Quantize the trained YOLO to INT8 and compile it to a real NPU graph.
# `export/yolo-onnx` and `quantize/yolo` are PURE RUST (run on any machine);
# `sim/yolo-int8` measures fp32-vs-INT8 mAP with NO NPU. `run/yolo-npu` and
# `bench/yolo-npu` REQUIRE OpenVINO 2024.x + an Intel NPU (3720 / Meteor Lake) at
# run time — they are NOT part of `make build`/`make test`. The NPU is a
# whole-graph compiler, separate from --device cpu|gpu; see docs/yolo/NPU.md.
ONNX        ?= $(OUT)/yolo.onnx
ONNX_INT8   ?= $(OUT)/yolo.int8.onnx
NPU_DEVICE  ?= NPU
NPU_CACHE   ?= $(OUT)/npu-cache
NPU_CALIB   ?= $(DATA)/detect
NPU_NCALIB  ?= 256

export/yolo-onnx: release
	@mkdir -p $(OUT)
	$(BRAIN) npu export --weights $(OUT)/yolo.safetensors --out $(ONNX)

quantize/yolo: release
	@mkdir -p $(OUT)
	$(BRAIN) npu quantize --weights $(OUT)/yolo.safetensors --calib $(NPU_CALIB) \
		--out $(ONNX_INT8) --num-calib $(NPU_NCALIB) --scales-out $(OUT)/yolo.scales.json

sim/yolo-int8: release
	$(BRAIN) npu sim --weights $(OUT)/yolo.safetensors --data $(DATA)/detect \
		--calib $(NPU_CALIB) --num-calib $(NPU_NCALIB) --conf $(YOLO_CONF) --iou $(YOLO_IOU)

run/yolo-npu: release
	$(BRAIN) npu run --onnx $(ONNX_INT8) --image $(DATA)/detect --device $(NPU_DEVICE) \
		--cache-dir $(NPU_CACHE) --conf $(YOLO_CONF) --iou $(YOLO_IOU)

bench/yolo-npu: release
	$(BRAIN) npu bench --onnx $(ONNX_INT8) --device $(NPU_DEVICE) \
		--cache-dir $(NPU_CACHE) --hint throughput --iters 200 --warmup 20

# ---- architecture-evaluation benchmark suite ------------------------------
# `make bench` runs every registered benchmark (crates/bench) and prints one
# comparison table (benchmark | score | threshold | pass/fail). `make bench/<name>`
# runs a single benchmark, e.g. `make bench/mqar` (multi-query associative recall).
# Add new benchmarks by registering them in crates/bench/src/lib.rs::registry —
# the generic `bench/%` rule runs any registered name with no Makefile change.
bench: release
	$(BRAIN) bench --seed $(SEED)

# ---- forecasting ----------------------------------------------------------
# `make forecast/compare` runs the scenario battery against the statistical
# baselines and renders the model x scenario x metric report (markdown to
# stdout). The random-walk negative control is a HARD gate: the command exits
# non-zero if any model falsely beats naive on it. Add HTML=path to also write a
# self-contained HTML report. Foundation models join the same battery as they
# are imported.
forecast/compare: release
	$(BRAIN) forecast compare --seed $(SEED) $(if $(HTML),--html $(HTML),)

# `make forecast/serve` starts the unified JSONL server with the baselines
# registered. Defaults to a Unix socket; override with LISTEN=host:port for TCP
# or SOCKET=path for a different socket path.
forecast/serve: release
	$(BRAIN) forecast serve $(if $(LISTEN),--listen $(LISTEN),--socket $(or $(SOCKET),/tmp/brain-forecast.sock))

# `make bench/scaling` runs the multi-scale scaling-law sweep (a separate entry
# point, not a registry benchmark): it trains the MQAR task at several model
# sizes and fits L(N) = E + A*N^-alpha, printing the size|params|flops|loss table
# plus the fitted exponent alpha and fit R^2. Foundation for the later
# per-capability predictive-scaling / eval-harness work. ~5 min on the CPU backend.
bench/scaling: release
	$(BRAIN) bench scaling --seed $(SEED)

# `make bench/eval ARCH=<name>` runs the turn-key architecture-eval harness: the
# WHOLE registered battery against one architecture, aggregated per capability
# axis, writing a structured artifact to results/<arch>-<seed>.json. Add a new
# architecture in crates/bench/src/arch.rs::arch_registry, then ARCH=<name> here.
bench/eval: release
	$(BRAIN) bench eval --arch $(ARCH) --seed $(SEED)

# `make bench/scale ARCH=<name>` runs the PREDICTIVE per-capability scaling sweep:
# train+score one representative benchmark per capability axis across a small SIZE
# grid, fit how each axis's score scales with params N, extrapolate the predicted
# score at 2x/4x the largest N, and write results/scale-<arch>-<seed>.json. This
# answers "how will each capability improve as we grow the model?" before paying
# for the bigger run. ~few min on the CPU backend (3 sizes x 6 axes, smoke budget).
bench/scale: release
	$(BRAIN) bench scale --arch $(ARCH) --seed $(SEED)

# `make bench/advise` prints RANKED tuning recommendations (what to tune to improve
# in the best capability direction): headroom x size-slope per axis, with a concrete
# action (increase size | change mechanism | more data/reg | deprioritize). Consumes
# the eval artifact and, if present, the scaling artifact for the same ARCH/SEED.
bench/advise: release
	@set -e; ev="results/$(ARCH)-$(SEED).json"; sc="results/scale-$(ARCH)-$(SEED).json"; \
	if [ ! -f "$$ev" ]; then \
		echo "no $$ev — run 'make bench/eval ARCH=$(ARCH)' first"; exit 2; \
	fi; \
	if [ -f "$$sc" ]; then $(BRAIN) bench advise "$$ev" "$$sc"; \
	else $(BRAIN) bench advise "$$ev"; fi

# `make bench/compare` prints a side-by-side leaderboard (overall pass-rate +
# per-axis + per-benchmark scores, columns = architectures) over every artifact
# under results/, so a new architecture is diffed against priors at a glance.
bench/compare: release
	@set -e; files="$$(ls results/*.json 2>/dev/null | grep -v '/scale-' || true)"; \
	if [ -z "$$files" ]; then \
		echo "no eval artifacts yet — run 'make bench/eval ARCH=<name>' first"; exit 2; \
	fi; \
	$(BRAIN) bench compare $$files

# Generic single-benchmark rule (`make bench/mqar`, …). The explicit bench/eval,
# bench/compare, bench/scaling, bench/char targets above take precedence.
bench/%: release
	$(BRAIN) bench $* --seed $(SEED)

# ---- shared GPT char-dataset benchmark (legacy) ---------------------------
# Train + eval the GPT baseline on the same char datasets, fixed seed/splits,
# so results are comparable. (MoE-on-char-data + federated rows are a documented
# follow-up — the MoE engine currently trains on its own 64-symbol rule task.)
bench/char: release
	@for d in calculator reverser; do \
		echo "=== dataset: $$d ==="; \
		$(MAKE) data/$$d N=$(N) SEED=$(SEED); \
		$(MAKE) train/gpt/$$d STEPS=$(STEPS); \
		$(MAKE) eval/gpt/$$d; \
	done

# ---- federated MoE artifact round-trip ------------------------------------
federated-demo: release
	@mkdir -p $(OUT)
	$(BRAIN) train --steps 50 --out $(OUT)/moe.safetensors
	$(BRAIN) federated split $(OUT)/moe.safetensors $(OUT)/shards
	$(BRAIN) federated verify $(OUT)/shards
	$(BRAIN) federated merge $(OUT)/shards --out $(OUT)/moe-reassembled.safetensors
	@echo "federated round-trip complete: $(OUT)/moe-reassembled.safetensors"

# ---- web (delegate to the web crate's Makefile) ---------------------------
web/dev:
	$(MAKE) -C crates/web web/dev
web/build:
	$(MAKE) -C crates/web web/build

clean:
	cargo clean
	rm -rf $(OUT)


# ---- performance benchmarking (brain perf) --------------------------------
# Distinct from `make bench`: bench asks whether an architecture LEARNS a task;
# perf asks how much correct work the engine delivers per unit of hardware,
# memory, energy and time. Design: docs/performance/benchmarking.md.
#
# TARGET selects what is measured. There is no synthetic-harness stand-in --
# every target exercises a real engine. Default is `qwen-synth:<L>x<D>x<H>x<V>x
# <HeadDim>x<NKvHeads>` at Qwen3-0.6B's REAL KV geometry (the REAL paged
# serving engine on random weights — same kernels, KV traffic and batching, so
# hardware comparison works with no checkpoint on the machine); override with
# `qwen:<weights>` to measure a real checkpoint.
PERF_TARGET ?= qwen-synth:28x1024x16x151936x128x8
PERF_WORKLOAD ?= chat
PERF_LADDER ?= 1,2,4,8,16,32

# The whole core battery on the current device: latency floor, saturated
# ceiling, realistic serving, and the concurrency curve.
perf: release
	@set -e; for s in latency throughput serve; do \
		$(BRAIN) perf run $$s --target $(PERF_TARGET) --workload $(PERF_WORKLOAD) --seed $(SEED); \
	done; \
	$(BRAIN) perf run sweep --target $(PERF_TARGET) --workload $(PERF_WORKLOAD) --ladder $(PERF_LADDER) --seed $(SEED)

# One scenario: `make perf/sweep`, `make perf/serve`, …
perf/%: release
	$(BRAIN) perf run $* --target $(PERF_TARGET) --workload $(PERF_WORKLOAD) --ladder $(PERF_LADDER) --seed $(SEED)

# LFM2.5-Encoder concurrency benchmark, standalone: the residency-executor
# target (real scheduler + budgets + lanes + equal-length batching) at 8k
# context. LFM_WEIGHTS/LFM_TOKENIZER select the model (230m/350m).
# LFM_TOKENIZER has NO default: the tokenizer lives outside the repo, and a
# baked-in absolute path is one dev box's layout (same @test -n guard as the
# depth/mirror/splat targets).
LFM_WEIGHTS ?= out/lfm-230m.safetensors
LFM_INPUT ?= 8192
perf/lfm: release
	@test -n "$(LFM_TOKENIZER)" || (echo "set LFM_TOKENIZER=<path to LFM2.5-Encoder tokenizer.json>"; exit 2)
	@set -e; for s in latency sweep; do \
		$(BRAIN) perf run $$s --target lfm:$(LFM_WEIGHTS):$(LFM_TOKENIZER) \
			--input $(LFM_INPUT) --output 1 --ladder $(PERF_LADDER) --seed $(SEED); \
	done

# FLUX.2 Klein denoise-step benchmark, standalone: the residency-executor
# target (real scheduler + budgets + lanes) on klein-4b; weights from the
# BRAIN_FLUX2_* env (same as flux2/generate). One denoise step is MINUTES on a
# CPU backend, so the request count is a knob and defaults tiny — size the run,
# don't let it size you. FLUX2_SIZE is <W>x<H>x<steps>; --output mirrors the
# step count so the workload's requested artifacts match what a request emits.
FLUX2_SIZE ?= 512x512x4
FLUX2_REQUESTS ?= 2
FLUX2_WARMUP ?= 1
perf/flux2: release
	@test -n "$(BRAIN_FLUX2_DIT)" || (echo "set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER"; exit 2)
	$(BRAIN) perf run latency --target flux2:$(FLUX2_SIZE) \
		--concurrency 1 --requests $(FLUX2_REQUESTS) --warmup $(FLUX2_WARMUP) \
		--input 1 --output $(word 3,$(subst x, ,$(FLUX2_SIZE))) --seed $(SEED)

# Leaderboard over every perf artifact. Refuses to rank across artifact units,
# excludes runs whose correctness gate failed, and warns on differing axes.
perf/compare: release
	@set -e; files="$$(ls results/perf-*.json 2>/dev/null || true)"; \
	if [ -z "$$files" ]; then echo "no perf artifacts yet — run 'make perf' first"; exit 2; fi; \
	$(BRAIN) perf compare $$files

# CI-sized: every scenario shrunk to seconds, on the CPU backend.
perf/smoke: release
	@set -e; for s in latency throughput serve sweep mixed overload soak frontend; do \
		$(BRAIN) perf run $$s --target $(PERF_TARGET) --workload interactive --smoke --seed $(SEED); \
	done

# ---- documentation ----------------------------------------------------------
# Build the full docs bundle: build/docs/brain-docs.{md,pdf}. Requires pandoc +
# xelatex. No HTML is produced.
.PHONY: docs
docs:
	python3 docs/pandoc/build-docs.py

# Real end-to-end z-image int8 generation (256x256) against the fetched
# checkpoint under out/models/ — the no-OOM regression run from fa7b576.
# Heavy: real weights + a GPU; writes results/zimage-int8-e2e.log.
zimage/int8-e2e:
	bash scripts/run_zimage_int8_e2e.sh

# ---- FLUX.2 Klein (crates/flux2; weights via BRAIN_FLUX2_* env) ----
flux2/generate: release
	@test -n "$(BRAIN_FLUX2_DIT)" || (echo "set BRAIN_FLUX2_DIT/_VAE/_TE/_TOKENIZER"; exit 2)
	$(BRAIN) flux2 generate --prompt "$(PROMPT)" --out out/flux2.ppm $(FLUX2_FLAGS)

flux2/edit: release
	@test -n "$(FLUX2_REF)" || (echo "set FLUX2_REF=<ref.ppm> PROMPT=..."; exit 2)
	$(BRAIN) flux2 generate --prompt "$(PROMPT)" --ref $(FLUX2_REF) --out out/flux2-edit.ppm $(FLUX2_FLAGS)
