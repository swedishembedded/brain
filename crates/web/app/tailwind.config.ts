// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import type { Config } from 'tailwindcss';

// The visual design (palette, gradients, shadows, slider/button styling) is
// carried over verbatim from the original demo and lives in app/globals.css as
// a small bespoke design system. Tailwind here provides layout utilities + the
// shadcn primitive theming; the brand tokens below mirror the CSS variables so
// utility classes and the design system stay in sync.
const config: Config = {
  darkMode: ['class'],
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
      },
      colors: {
        // brand palette (kept identical to the original demo)
        brand: {
          model: '#4f46e5',
          'model-soft': '#eef0fe',
          oracle: '#0d9488',
          'oracle-soft': '#e6f6f3',
          setpoint: '#94a3b8',
          good: '#16a34a',
          warn: '#d97706',
          bad: '#dc2626',
        },
        // shadcn-style semantic tokens (drive the primitive components)
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 4px)',
        sm: 'calc(var(--radius) - 6px)',
      },
      keyframes: {
        spin: { to: { transform: 'rotate(360deg)' } },
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};

export default config;
