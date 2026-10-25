# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines and reporting
# for its clients. If your team needs expertise in building papers whose every
# number and figure can be regenerated from the repository, you can procure
# our services by sending an email to info@swedishembedded.com.

# What every paper shares. A paper's Makefile sets PAPER (the source file stem,
# default "paper"), includes this file, and states how its PDF is made:
#
#   $(PDF): <sources>
#   	$(call latexmk-pdf,$(PAPER).tex)
#
# Build output never lands in the paper's directory: the PDF is
# <repository>/build/papers/<directory name>.pdf and the LaTeX intermediates
# are under <repository>/build/papers/.build/<directory name>/.
# `make papers/<name>/pdf` at the repository root runs `make` here.
PAPER ?= paper
NAME := $(notdir $(CURDIR))
OUT := $(abspath ../../build/papers)
BUILD := $(OUT)/.build/$(NAME)
PDF := $(OUT)/$(NAME).pdf

.PHONY: all clean
all: $(PDF)

# latexmk names its output after the job, so the job is the paper's directory
# name; \input paths and the bibliography stay relative to the source directory.
define latexmk-pdf
mkdir -p $(BUILD)
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) -jobname=$(NAME) $(1)
cp $(BUILD)/$(NAME).pdf $@
endef

clean::
	rm -rf $(PDF) $(BUILD)
