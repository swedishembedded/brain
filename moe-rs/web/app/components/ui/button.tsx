// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

// `run` is the primary call-to-action styled by the bespoke `.run` design-system
// class (indigo gradient, lift on hover, spinner slot). Base utilities are kept
// minimal so they never fight the ported CSS.
const buttonVariants = cva('inline-flex items-center justify-center', {
  variants: {
    variant: {
      run: 'run',
    },
  },
  defaultVariants: {
    variant: 'run',
  },
});

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, type = 'button', ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(buttonVariants({ variant }), className)}
      {...props}
    />
  ),
);
Button.displayName = 'Button';

export { Button, buttonVariants };
