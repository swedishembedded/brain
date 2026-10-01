# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Builds the hardware probe against whichever toolkit is on PATH (nvcc) and
# CUDA_PATH. `eval "$(scripts/build/install-cuda-userspace.py env 13)"` first.
# ARCH defaults to Hopper's architecture-specific target.

ARCH ?= sm_90a
NVCC ?= nvcc
CUDA_PATH ?= $(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC)))

probe: probe.cu
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 \
		-Xlinker -rpath=$(CUDA_PATH)/lib64 -o $@ $< -lcublasLt -lcudart

clean:
	rm -f probe

.PHONY: clean
