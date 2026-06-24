// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

// Maps the grid-classification kinds to the bespoke `.badge` color variants.
const badgeVariants = cva('badge', {
  variants: {
    kind: {
      training: 'training',
      validation: 'validation',
      offgrid: 'offgrid',
    },
  },
  defaultVariants: {
    kind: 'offgrid',
  },
});

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, kind, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ kind }), className)} {...props} />;
}

export { Badge, badgeVariants };
