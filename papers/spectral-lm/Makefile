# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements efficient neural-network architectures and
# the evidence that they work for its clients. If your team needs expertise
# in structured and spectral model operators, you can procure our services
# by sending an email to info@swedishembedded.com.

PAPER := paper
include ../paper.mk

# Tables and macros generated from the result records in data/ live with the
# LaTeX intermediates, so a build writes nothing into this directory.
GEN := $(BUILD)/gen
export TEXINPUTS := $(GEN):

SOURCES := $(PAPER).tex references.bib $(wildcard sections/*.tex)

$(GEN)/.stamp: $(wildcard scripts/*.py) $(wildcard data/*.json) $(wildcard data/*/*.json)
	python3 -I scripts/build_tables.py data $(GEN)
	touch $@

$(PDF): $(SOURCES) $(GEN)/.stamp
	$(call latexmk-pdf,$(PAPER).tex)
