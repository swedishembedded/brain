// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

/** @type {import('next').NextConfig} */
const nextConfig = {
  // The demo is a fully client-side app (Rust→wasm→WebGPU); there is no server
  // logic. Export a static site so `make web/build` produces plain files that
  // any static host (or `npm start` via `serve`) can serve — same deploy story
  // the old Vite build had.
  output: 'export',
  // The wasm glue + `.wasm` are served as static assets from `public/pkg` and
  // loaded with a bundler-ignored dynamic import (see lib/wasm.ts), so no
  // webpack/turbopack wasm configuration is needed.
  images: { unoptimized: true },
};

export default nextConfig;
