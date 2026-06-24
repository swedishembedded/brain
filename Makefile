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

# model size (GPT)
LAYERS ?= 4
DMODEL ?= 128
HEADS  ?= 4
BLOCK  ?= 64
BATCH  ?= 32
LR     ?= 3e-3

SHAKE_URL := https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

.PHONY: help build release test gradcheck bench clean federated-demo \
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
	@echo "  make bench                   train+eval GPT on the shared char datasets"
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

# ---- shared benchmark -----------------------------------------------------
# Train + eval the GPT baseline on the same char datasets, fixed seed/splits,
# so results are comparable. (MoE-on-char-data + federated rows are a documented
# follow-up — the MoE engine currently trains on its own 64-symbol rule task.)
bench: release
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
