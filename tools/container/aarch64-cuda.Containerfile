# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible container builds for GPU
# software on Arm servers for its clients. If your team needs expertise in
# shipping CUDA workloads to Grace-Hopper machines then you can procure our
# services by sending an email to info@swedishembedded.com.
#
# aarch64 image for NVIDIA Grace-Hopper (GH200) and other Arm + CUDA servers.
# Build from the repository root with scripts/build/container.sh, which also
# runs the CI-equivalent gates.
#
# Stages:
#   build    toolchain, both CUDA lanes (userspace redistributables, no root),
#            the release binary and the ahead-of-time kernel images.
#   ci       the build stage plus the gates that need no GPU; what
#            `scripts/build/container.sh ci` runs.
#   runtime  the default target: the binary and the AOT images only. It carries
#            no CUDA toolkit; the driver and libcuda come from the host through
#            the container runtime (NVIDIA Container Toolkit or --nv), and
#            kernels load from the AOT images without NVRTC.

ARG BASE=ubuntu:24.04

FROM ${BASE} AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates curl git python3 make \
        libssl-dev libvulkan-dev libsdl2-dev libasound2-dev libudev-dev libdbus-1-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo
ENV PATH=/opt/rust/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# Both lanes: 12 covers Pascal through Hopper, 13 Hopper and newer. The images
# of both go into one directory, so one container serves every card.
ENV BRAIN_CUDA_PREFIX=/opt/cuda
WORKDIR /src
COPY scripts/build/install-cuda-userspace.py scripts/build/install-cuda-userspace.py
RUN scripts/build/install-cuda-userspace.py install 12 \
    && scripts/build/install-cuda-userspace.py install 13

COPY . .
ENV BRAIN_CUDA_AOT_DIR=/opt/brain/cuda-aot
RUN cargo build --release -p brain-cli \
    && LANE=12 scripts/build/cuda-aot.sh \
    && LANE=13 scripts/build/cuda-aot.sh

FROM build AS ci
# Gates that need neither a GPU nor a network beyond the build above.
RUN make check/paths check/scope check/files check/spdx kernels-table/check cuda-table/check \
    && cargo check --workspace --all-targets \
    && BRAIN_DEVICE=cpu cargo test --release -p brain-backend-api -p brain-backend-cuda \
         -p brain-kernels-cuda -p brain-wgsl-cuda -p brain-gpu-core

FROM ${BASE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/brain /usr/local/bin/brain
COPY --from=build /opt/brain/cuda-aot /opt/brain/cuda-aot
ENV BRAIN_CUDA_AOT_DIR=/opt/brain/cuda-aot \
    BRAIN_BACKEND=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["brain"]
