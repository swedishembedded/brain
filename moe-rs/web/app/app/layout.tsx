// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';

import './globals.css';

// Self-hosted via next/font — same families the original demo loaded from the
// Google Fonts CDN (Inter for UI, JetBrains Mono for numerics). Exposed as the
// CSS variables the design system reads (`--font-sans`, `--font-mono`).
const inter = Inter({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-sans',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'PID Transformer — Model vs. Oracle (WebGPU)',
  description:
    'A tiny transformer imitating a per-plant-tuned PID controller, racing an analytically-tuned oracle on the same first-order plant — all in Rust → WebAssembly → WebGPU.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
