# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# moe-rs — web (WebGPU/wasm) developer workflow.
#
# The browser demo is a Next.js (App Router) app in web/app that runs the PID
# control transformer (compiled to wasm/WebGPU) and compares it against the
# per-plant PID oracle. `make web/dev` builds the wasm, refreshes the app's
# bindings, and starts the dev server. Requires a WebGPU-capable browser
# (Chrome/Edge 113+).
#
# Prerequisites are installed automatically by `make requirements` (run as a
# prerequisite of the build targets below). To install them by hand:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.125   # match the locked crate
#   Node 18+ (npm).
#
# cargo/wasm-bindgen are taken from your environment/PATH. Override if needed:
#   make web/dev PORT=8080 WASM_BINDGEN=/path/to/wasm-bindgen

PORT                 ?= 5173
APP                  := app
WASM_TARGET          := wasm32-unknown-unknown
WASM_ARTIFACT        := ../../target/wasm32-unknown-unknown/release/brain_web.wasm
# wasm-bindgen-cli version — must match the locked `wasm-bindgen` crate in Cargo.lock.
WASM_BINDGEN_VERSION ?= 0.2.125
# Prefer wasm-bindgen on PATH; otherwise fall back to the standard cargo-install
# location (~/.cargo/bin), which is often not on PATH. Override with WASM_BINDGEN=.
WASM_BINDGEN         ?= $(shell command -v wasm-bindgen 2>/dev/null || echo $(HOME)/.cargo/bin/wasm-bindgen)
WEIGHTS              ?= moe_pid.weights

.PHONY: requirements web/wasm web/deps web/dev web/build web/clean help

help:
	@echo "targets:"
	@echo "  make web/dev      build wasm + serve the Next.js demo (http://localhost:$(PORT))"
	@echo "  make web/build    static production export into $(APP)/out"
	@echo "  make web/wasm     compile Rust -> wasm + JS bindings into $(APP)/public/pkg"
	@echo "  make requirements install the wasm32 target + wasm-bindgen-cli if missing"
	@echo "  make web/clean    remove node_modules, build output, and generated pkg"

## Install the toolchain prerequisites if (and only if) they are missing:
##   - the rustup wasm32-unknown-unknown target
##   - wasm-bindgen-cli pinned to $(WASM_BINDGEN_VERSION)
## Node/npm cannot be installed from here; we just check it is present.
requirements:
	@if rustup target list --installed 2>/dev/null | grep -qx '$(WASM_TARGET)'; then \
		echo "ok: rust target $(WASM_TARGET) installed" ; \
	else \
		echo "installing rust target $(WASM_TARGET)..." ; \
		rustup target add $(WASM_TARGET) \
		|| sudo env "PATH=$$PATH" RUSTUP_HOME="$$RUSTUP_HOME" CARGO_HOME="$${CARGO_HOME:-$$RUSTUP_HOME}" rustup target add $(WASM_TARGET) ; \
	fi
	@if [ "$$('$(WASM_BINDGEN)' --version 2>/dev/null | awk '{print $$2}')" = "$(WASM_BINDGEN_VERSION)" ]; then \
		echo "ok: wasm-bindgen-cli $(WASM_BINDGEN_VERSION) installed" ; \
	else \
		echo "installing wasm-bindgen-cli $(WASM_BINDGEN_VERSION)..." ; \
		cargo install wasm-bindgen-cli --version $(WASM_BINDGEN_VERSION) --force ; \
	fi
	@command -v npm >/dev/null 2>&1 \
		&& echo "ok: npm $$(npm --version) installed" \
		|| { echo "ERROR: npm (Node 18+) not found — install Node: https://nodejs.org" ; exit 1 ; }
	@echo "requirements satisfied"

## Compile the Rust inference path to wasm + wasm-bindgen JS into the Next.js
## app (served as static assets from public/pkg), and stage the trained weights
## where the page fetches them.
web/wasm: requirements
	cargo build --release --target wasm32-unknown-unknown -p brain-web --features webgpu
	@command -v "$(WASM_BINDGEN)" >/dev/null 2>&1 || { \
		echo "ERROR: wasm-bindgen not found (looked for '$(WASM_BINDGEN)')." ; \
		echo "  install:  cargo install wasm-bindgen-cli --version 0.2.125" ; \
		echo "  then ensure it is on PATH (e.g. export PATH=\"\$$HOME/.cargo/bin:\$$PATH\")" ; \
		echo "  or pass it explicitly:  make web/dev WASM_BINDGEN=/path/to/wasm-bindgen" ; \
		exit 1 ; }
	"$(WASM_BINDGEN)" $(WASM_ARTIFACT) --target web --out-dir $(APP)/public/pkg
	@if [ -f $(WEIGHTS) ]; then \
		mkdir -p $(APP)/public && cp -f $(WEIGHTS) $(APP)/public/$(notdir $(WEIGHTS)) ; \
		echo "staged $(WEIGHTS) -> $(APP)/public/$(notdir $(WEIGHTS))" ; \
	else \
		echo "note: $(WEIGHTS) not found — train one first:  ./target/release/moe pid train ... --out $(WEIGHTS)" ; \
	fi

## Install npm deps only when missing.
web/deps:
	cd $(APP) && [ -d node_modules ] || npm install

## Build the wasm bindings, then serve the Next.js dev server.
## `-H 0.0.0.0` binds all interfaces so the server is reachable from outside the
## container (e.g. via a forwarded port or the container's public address).
## NOTE on WebGPU: navigator.gpu is gated to secure contexts. http://localhost
## qualifies, but a plain-http LAN/public IP does NOT — so if you open the
## Network URL over bare http, WebGPU will be hidden. Reach it via a forwarded
## localhost port, or put a TLS proxy (https) in front of the exposed IP.
web/dev: web/wasm web/deps
	@echo "open http://localhost:$(PORT) in a WebGPU browser (Chrome/Edge 113+)."
	@echo "use localhost (or a forwarded localhost port) — a plain-http LAN/public IP is NOT a secure context, so WebGPU is hidden."
	cd $(APP) && npm run dev -- -p $(PORT) -H 0.0.0.0

## Static production export into web/app/out.
web/build: web/wasm web/deps
	cd $(APP) && npm run build

web/clean:
	rm -rf $(APP)/node_modules $(APP)/.next $(APP)/out $(APP)/public/pkg
