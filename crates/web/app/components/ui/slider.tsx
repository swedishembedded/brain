// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

'use client';

import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';

import { cn } from '@/lib/utils';

/** Radix slider themed to match the original gradient range input
 *  (`.range-*` classes live in app/globals.css). */
const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root ref={ref} className={cn('range-root', className)} {...props}>
    <SliderPrimitive.Track className="range-track">
      <SliderPrimitive.Range className="range-range" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="range-thumb" aria-label="value" />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
