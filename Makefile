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

.PHONY: help build release test/doc test/slow test/full test/times wm/play wm-fixtures test gradcheck kernels-regen parity requirements bench bench/char bench/eval bench/scale bench/advise bench/compare perf perf/compare perf/smoke clean federated-demo depth/demo depth/smoke depth/camera train/zipdepth mirror/import mirror/infer mirror/demo splat/view \
        data/calculator data/reverser data/wordcalc data/timeseries \
        data/shakespeare_char data/gpt data/detect \
        train/yolo eval/yolo detect/yolo \
        export/yolo-onnx quantize/yolo sim/yolo-int8 run/yolo-npu bench/yolo-npu \
        web/dev web/build forecast/compare forecast/serve

help:
	@echo "brain targets:"
	@echo "  make release                 build the optimized 'brain' binary"
	@echo "  make requirements            pip-install the Python tooling (OpenVINO/NPU, torch, ...)"
	@echo "  make test                    FAST lane: unit+integration, no doc-tests"
	@echo "  make test/doc                doc-tests (slow: one link per crate)"
	@echo "  make test/slow               #[ignore]d long-running tests"
	@echo "  make test/full               test + test/doc + test/slow"
	@echo "  make test/times              rank test binaries by wall time"
	@echo "  make gradcheck               numerical backprop correctness gate (GPT)"
	@echo "  make data/<name>             generate a dataset (calculator|reverser|wordcalc|"
	@echo "                               timeseries|shakespeare_char|gpt) into $(DATA)/<name>"
	@echo "  make train/gpt/<name>        train GPT on a dataset -> $(OUT)/gpt-<name>.weights"
	@echo "  make eval/gpt/<name>         perplexity + exact-match for a trained GPT"
	@echo "  make data/detect             synthetic object-detection dataset -> $(DATA)/detect"
	@echo "  make train/yolo              train tiny YOLO on it -> $(OUT)/yolo.weights"
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
	@echo "  make wm/play                  play the fake world model in an SDL window (WASD)"
	@echo "  make wm-fixtures             regenerate DIAMOND parity fixtures (needs torch)"
	@echo "  make federated-demo          MoE train -> split -> verify -> merge round-trip"
	@echo "  make web/dev | web/build     WebGPU browser demo (crates/web)"

build:
	cargo build

release:
	cargo build --release

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
test:
	@echo "test: fast lane (unit + integration, no doc-tests, GPU serialised)"
	@$(CARGO_TEST) --lib --bins --tests --no-run
	@timeout $(TEST_TIMEOUT) $(CARGO_TEST) --lib --bins --tests -- --test-threads=$(TEST_THREADS); \
	rc=$$?; \
	if [ $$rc -eq 124 ]; then \
		echo; echo "TIMED OUT after $(TEST_TIMEOUT)s of RUNNING — almost certainly a deadlock."; \
		echo "Find it with:  scripts/test-times.sh --top 10"; \
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

# Everything, for a release gate.
test/full: test test/doc test/slow

# Rank every test binary by wall time; --budget fails if any exceeds it. This is
# what keeps the fast lane fast.
test/times: release
	scripts/test-times.sh --top 15

# Install the Python tooling (OpenVINO/NPU runtime, torch + transformers for the
# benchmark reference rows, etc.) into the current environment. The Rust engine
# needs none of these — this is for tools/ and the `--device npu` runtime.
requirements:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

gradcheck: release
	$(BRAIN) gradcheck

# Regenerate the kernel const block + ALL registry in crates/kernels/src/lib.rs
# from the contents of crates/kernels/wgsl/. Run after adding/removing a .wgsl
# file; merge conflicts in lib.rs are resolved by union-ing wgsl/ + this target.
kernels-regen:
	scripts/kernels-regen.sh

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
	scripts/parity-gate.sh

# ---- data generation ------------------------------------------------------
data/calculator data/reverser data/wordcalc: release
	$(BRAIN) data gen $(@F) --out $(DATA)/$(@F) --n $(N) --seed $(SEED)

data/timeseries: release
	$(BRAIN) data gen timeseries --out $(DATA)/timeseries --n 200000 --seed $(SEED)

# Synthetic Qwen3-TTS `text -> codebook-0 codes` stream (for Talker SFT smokes).
data/tts: release
	$(BRAIN) data gen tts --out $(DATA)/tts --n $(N) --seed $(SEED)

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
	$(BRAIN) gpt train $(DATA)/$* --out $(OUT)/gpt-$*.weights \
		--steps $(STEPS) --batch $(BATCH) --block $(BLOCK) \
		--layers $(LAYERS) --d-model $(DMODEL) --heads $(HEADS) --lr $(LR) \
		--seed $(SEED) $(MASK_$*)

# ---- eval (pattern: eval/gpt/<dataset>) -----------------------------------
eval/gpt/%: release
	$(BRAIN) gpt eval --weights $(OUT)/gpt-$*.weights --data $(DATA)/$*

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
	$(BRAIN) yolo train $(DATA)/detect --out $(OUT)/yolo.weights \
		--steps $(YOLO_STEPS) --batch $(YOLO_BATCH) --lr $(YOLO_LR) --seed $(SEED)

eval/yolo: release
	$(BRAIN) yolo eval --weights $(OUT)/yolo.weights --data $(DATA)/detect \
		--conf $(YOLO_CONF) --iou $(YOLO_IOU)

detect/yolo: release
	$(BRAIN) yolo detect --weights $(OUT)/yolo.weights --image $(DATA)/detect \
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
# model.safetensors (or its HF dir); the converted .weights is what infer uses.
MIRROR_CKPT    ?=
MIRROR_WEIGHTS ?= $(OUT)/mirror.weights

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
	$(BRAIN) depth train --out $(OUT)/zipdepth.weights --steps 50 --batch 2 \
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
	$(BRAIN) npu export --weights $(OUT)/yolo.weights --out $(ONNX)

quantize/yolo: release
	@mkdir -p $(OUT)
	$(BRAIN) npu quantize --weights $(OUT)/yolo.weights --calib $(NPU_CALIB) \
		--out $(ONNX_INT8) --num-calib $(NPU_NCALIB) --scales-out $(OUT)/yolo.scales.json

sim/yolo-int8: release
	$(BRAIN) npu sim --weights $(OUT)/yolo.weights --data $(DATA)/detect \
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
	$(BRAIN) train --steps 50 --out $(OUT)/moe.weights
	$(BRAIN) federated split $(OUT)/moe.weights $(OUT)/shards
	$(BRAIN) federated verify $(OUT)/shards
	$(BRAIN) federated merge $(OUT)/shards --out $(OUT)/moe-reassembled.weights
	@echo "federated round-trip complete: $(OUT)/moe-reassembled.weights"

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
# TARGET selects what is measured: `fake` (built-in synthetic engine — validates
# the harness anywhere, absolute numbers meaningless), `qwen-synth:<L>x<D>x<H>`
# (the REAL paged serving engine on random weights — same kernels, KV traffic and
# batching, so hardware comparison works with no checkpoint on the machine), or
# `qwen:<weights>` (the serving engine on a real checkpoint).
PERF_TARGET ?= fake
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
LFM_WEIGHTS ?= out/lfm-230m.weights
LFM_TOKENIZER ?= /data/workspace/resources/lfm/LFM2.5-Encoder-230M/tokenizer.json
LFM_INPUT ?= 8192
perf/lfm: release
	@set -e; for s in latency sweep; do \
		$(BRAIN) perf run $$s --target lfm:$(LFM_WEIGHTS):$(LFM_TOKENIZER) \
			--input $(LFM_INPUT) --output 1 --ladder $(PERF_LADDER) --seed $(SEED); \
	done

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
