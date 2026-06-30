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

.PHONY: help build release test gradcheck parity requirements bench bench/char bench/eval bench/scale bench/advise bench/compare clean federated-demo \
        data/calculator data/reverser data/wordcalc data/timeseries \
        data/shakespeare_char data/gpt data/detect \
        train/yolo eval/yolo detect/yolo \
        export/yolo-onnx quantize/yolo sim/yolo-int8 run/yolo-npu bench/yolo-npu \
        web/dev web/build

help:
	@echo "brain targets:"
	@echo "  make release                 build the optimized 'brain' binary"
	@echo "  make requirements            pip-install the Python tooling (OpenVINO/NPU, torch, ...)"
	@echo "  make test                    full cargo test suite"
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
	@echo "  make federated-demo          MoE train -> split -> verify -> merge round-trip"
	@echo "  make web/dev | web/build     WebGPU browser demo (crates/web)"

build:
	cargo build

release:
	cargo build --release

test:
	cargo test

# Install the Python tooling (OpenVINO/NPU runtime, torch + transformers for the
# benchmark reference rows, etc.) into the current environment. The Rust engine
# needs none of these — this is for tools/ and the `--device npu` runtime.
requirements:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

gradcheck: release
	$(BRAIN) gradcheck

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
