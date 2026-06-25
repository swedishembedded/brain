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

SHAKE_URL := https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

.PHONY: help build release test gradcheck bench bench/char bench/eval bench/compare clean federated-demo \
        data/calculator data/reverser data/wordcalc data/timeseries \
        data/shakespeare_char data/gpt web/dev web/build

help:
	@echo "brain targets:"
	@echo "  make release                 build the optimized 'brain' binary"
	@echo "  make test                    full cargo test suite"
	@echo "  make gradcheck               numerical backprop correctness gate (GPT)"
	@echo "  make data/<name>             generate a dataset (calculator|reverser|wordcalc|"
	@echo "                               timeseries|shakespeare_char|gpt) into $(DATA)/<name>"
	@echo "  make train/gpt/<name>        train GPT on a dataset -> $(OUT)/gpt-<name>.weights"
	@echo "  make eval/gpt/<name>         perplexity + exact-match for a trained GPT"
	@echo "  make bench                   run the architecture-evaluation benchmark suite (all)"
	@echo "  make bench/<name>            run one benchmark (e.g. bench/mqar)"
	@echo "  make bench/scaling           scaling-law sweep: fit L(N)=E+A*N^-alpha across sizes"
	@echo "  make bench/eval ARCH=<name>  run the WHOLE battery vs one architecture (gpt|gpt-small|"
	@echo "                               gpt-wide), aggregate per axis -> results/<arch>-<seed>.json"
	@echo "  make bench/compare           side-by-side leaderboard of every results/*.json"
	@echo "  make bench/char              train+eval GPT on the shared char datasets (legacy)"
	@echo "  make federated-demo          MoE train -> split -> verify -> merge round-trip"
	@echo "  make web/dev | web/build     WebGPU browser demo (crates/web)"

build:
	cargo build

release:
	cargo build --release

test:
	cargo test

gradcheck: release
	$(BRAIN) gradcheck

# ---- data generation ------------------------------------------------------
data/calculator data/reverser data/wordcalc: release
	$(BRAIN) data gen $(@F) --out $(DATA)/$(@F) --n $(N) --seed $(SEED)

data/timeseries: release
	$(BRAIN) data gen timeseries --out $(DATA)/timeseries --n 200000 --seed $(SEED)

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

# `make bench/compare` prints a side-by-side leaderboard (overall pass-rate +
# per-axis + per-benchmark scores, columns = architectures) over every artifact
# under results/, so a new architecture is diffed against priors at a glance.
bench/compare: release
	@set -e; files="$$(ls results/*.json 2>/dev/null || true)"; \
	if [ -z "$$files" ]; then \
		echo "no results/*.json yet — run 'make bench/eval ARCH=<name>' first"; exit 2; \
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
