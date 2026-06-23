// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Plant-parameter grid the transformer was trained / validated against.
export const TRAIN_TAU = [0.45, 0.65, 0.85];
export const TRAIN_GAIN = [1.0, 1.25, 1.5];
export const VAL_TAU = [0.55, 0.75];
export const VAL_GAIN = [1.125, 1.375];

export type GridKind = 'training' | 'validation' | 'offgrid';

const EPS = 1e-3;
const near = (x: number, set: number[]) => set.some((v) => Math.abs(v - x) < EPS);

/**
 * Classify a (tau, gain) point relative to the training/validation grid.
 *  - training:   both coords are training-grid nodes (seen during training)
 *  - validation: both coords are validation interpolation nodes (held-out)
 *  - offgrid:    anything else the slider lands on
 */
export function classifyGrid(tau: number, gain: number): GridKind {
  if (near(tau, TRAIN_TAU) && near(gain, TRAIN_GAIN)) return 'training';
  if (near(tau, VAL_TAU) && near(gain, VAL_GAIN)) return 'validation';
  return 'offgrid';
}

export const GRID_COPY: Record<GridKind, { label: string; hint: string }> = {
  training: {
    label: 'training node',
    hint: 'this plant was in the training grid',
  },
  validation: {
    label: 'interpolated (unseen)',
    hint: 'held-out validation plant — never trained on',
  },
  offgrid: {
    label: 'off-grid',
    hint: 'between grid nodes — pure generalization',
  },
};
